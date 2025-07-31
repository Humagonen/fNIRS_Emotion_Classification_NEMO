import streamlit as st
import zipfile
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import re
import mne
from mne.preprocessing.nirs import scalp_coupling_index, tddr, beer_lambert_law
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="fNIRS Trial-wise Emotion Classifier", layout="wide")
st.title("fNIRS Trial-wise Emotion Classifier")
st.write("""
### Welcome to the fNIRS Emotion Classification App

This tool allows you to upload an individual subject's **BIDS-formatted fNIRS data (ZIP file)** and receive **trial-wise emotion predictions** based on brain activity during an **emotional perception task**.

#### Here's how it works:
Once you upload the data, the app will:

1. **Extract the `.snirf` file** from your zipped BIDS folder
2. Apply full **preprocessing**, including:
   - Bad channel interpolation (using scalp coupling index)
   - Motion correction (TDDR)
   - Filtering (0.01–0.1 Hz)
   - HbO/HbR conversion (Beer–Lambert Law)
3. **Extract events/trials** from annotations (no event labels needed)
4. **Slice each trial** from 0 to 12 seconds post-stimulus, and split into **three time windows** (0–4s, 4–8s, 8–12s)
5. Compute the **mean signal for each window and channel** → this becomes your feature vector
6. Apply **z-score normalization** using a pre-fitted `StandardScaler`
7. Run predictions using a **pre-trained LightGBM model** trained with **LOSO-CV**

#### Classification Output

Each trial will be classified into one of the following emotion categories:

- **HANV** – High Arousal, Negative Valence  
- **HAPV** – High Arousal, Positive Valence  
- **LANV** – Low Arousal, Negative Valence  
- **LAPV** – Low Arousal, Positive Valence  

The app will also report **model confidence** for each prediction.

#### Upload Instructions

- Upload a **ZIP file** containing a single subject folder in **BIDS format**
- It must contain the `*_empe_nirs.snirf` file inside the `nirs/` directory
- You do **not** need `events.tsv` or `events.json` – the app handles e# thing automatically

#### fNIRS Channel Format

Channel names like `S3_D3 756` represent:
- `S3_D3`: Source 3 – Detector 3 (optode pair / channel)
- `756` or `853`: Wavelengths in nanometers (nm)  
  These represent **near-infrared light** used to measure oxy- (HbO) and deHbR) concentrations.


Upload your subject's ZIP file using the sidebar on the left and view the predictions below!

---
""")


# --- Upload section ---
with st.sidebar:
    uploaded_file = st.file_uploader("📂 Upload a ZIP file of a subject's BIDS folder", type="zip")

# --- Helper functions ---
def extract_zip_to_temp(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return Path(temp_dir)

def preprocess_and_extract_features(snirf_file):
    st.subheader("Read and Preprocess SNIRF")
    raw = mne.io.read_raw_snirf(snirf_file, preload=True, verbose=False)
    st.write("📁 File loaded:", snirf_file.name)
    st.write(f"Number of channels: {len(raw.ch_names)}")
    st.write(", ".join(raw.ch_names))


    # Step 2: SCI & bad channel interpolation
    sci = scalp_coupling_index(raw)
    bad_channels = list(np.array(raw.ch_names)[sci < 0.8])
    raw.info["bads"] = bad_channels
    raw.interpolate_bads()
    st.write("Bad channels interpolated (SCI < 0.8):", bad_channels)

    # Step 3: TDDR
    raw_od = tddr(raw.copy())

    # Step 4: Beer-Lambert
    raw_hb = beer_lambert_law(raw_od, ppf=6.0)
    st.success("TDDR motion correction and Beer-Lambert law applied")

    # Step 5: Filtering
    raw_hb = raw_hb.filter(l_freq=0.01, h_freq=0.1,
                           l_trans_bandwidth=0.004,
                           h_trans_bandwidth=0.01,
                           verbose=False)
    st.success("Bandpass filtering (0.01–0.1 Hz) done")

    # Step 6: Events
    events, _ = mne.events_from_annotations(raw_hb)
    if len(events) == 0:
        st.error("No events found in the SNIRF file.")
        return None
    st.write("Number of event markers found:", len(events))

    dummy_event_dict = {'unknown': -1}
    dummy_events = np.copy(events)
    dummy_events[:, 2] = -1
    metadata = pd.DataFrame({
        'condition': ['unknown'] * len(dummy_events),
        'subject': ['inference'] * len(dummy_events)
    })

    # Step 7: Epoching
    epochs = mne.Epochs(
        raw_hb, dummy_events, event_id=dummy_event_dict,
        metadata=metadata,
        tmin=-5.0, tmax=12.0,
        baseline=(None, 0),
        preload=True,
        reject=dict(hbo=80e-6),
        verbose=False
    )
    st.write("Total valid trials after epoching:", len(epochs))

    # Step 8: Extract only HbO
    hbo_epochs = epochs.copy().pick(picks="hbo")
    X = hbo_epochs.get_data()[:, :, 250:-1]
    ch_names = hbo_epochs.ch_names
    st.write("Selecting HbO channels only...")

    # Step 9: Flatten time-series
    n_trials, n_channels, n_times = X.shape
    st.write(f"Data shape: {n_trials} trials × {n_channels} channels × {n_times} timepoints")

    X_flat = X.reshape(n_trials, n_channels * n_times)
    columns = [f'{ch}_t{t+1}' for ch in ch_names for t in range(n_times)]
    df = pd.DataFrame(X_flat, columns=columns)

    # --- Step 10: Windowing Features ---
    st.subheader("Feature Engineering (Windowing)")
    signal_cols = df.columns.tolist()

    def sort_key(ch):
        nums = re.findall(r'\d+', ch)
        return list(map(int, nums))

    channel_names = sorted(
        {re.match(r'(.*) hbo_t\d+', col).group(1) for col in signal_cols},
        key=sort_key
    )

    channel_to_times = {
        ch: sorted([col for col in signal_cols if col.startswith(ch)]) for ch in channel_names
    }

    features = []
    for _, row in df.iterrows():
        trial_data = np.array([row[channel_to_times[ch]].values for ch in channel_names])
        means = [trial_data[:, i*200:(i+1)*200].mean(axis=1) for i in range(3)]
        trial_features = np.concatenate(means)
        features.append(trial_features)

    feature_names = [f"{ch}_w{w+1}" for ch in channel_names for w in range(3)]
    df_feat = pd.DataFrame(features, columns=feature_names)
    st.write("Feature shape after windowing:", df_feat.shape)
    return df_feat

# --- Run prediction ---
if uploaded_file:
    with st.spinner("Running pipeline..."):
        try:
            extracted_dir = extract_zip_to_temp(uploaded_file)
            snirf_file = list(extracted_dir.rglob("*-empe_nirs.snirf"))[0]
        except IndexError:
            st.error("SNIRF file (*-empe_nirs.snirf) not found in ZIP.")
            st.stop()

        df_features = preprocess_and_extract_features(snirf_file)
        if df_features is None:
            st.stop()

        st.subheader("Load Model & Predict")
        # Load model + scaler
        model = lgb.Booster(model_file='4Class_best_lgbm_loso_029.txt')
        scaler = StandardScaler()
        scaler.mean_ = np.load('scaler2_mean.npy')
        scaler.scale_ = np.load('scaler2_scale.npy')
        scaler.n_features_in_ = df_features.shape[1]
        st.success("Model and scaler loaded")

        df_scaled = scaler.transform(df_features)
        preds = model.predict(df_scaled)
        pred_labels = np.argmax(preds, axis=1)
        pred_conf = np.max(preds, axis=1) * 100
        class_names = ['HANV', 'HAPV', 'LANV', 'LAPV']
        pred_classes = [class_names[i] for i in pred_labels]

        st.subheader("Trial-wise Predictions")
        result_df = pd.DataFrame({
            "Trial": [f"Trial {i+1}" for i in range(len(pred_labels))],
            "Predicted Class": pred_classes,
            "Confidence (%)": [f"{c:.2f}%" for c in pred_conf]
        })
        st.dataframe(result_df, use_container_width=True)

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
