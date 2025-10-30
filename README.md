# Digit Recognition AI — Self-Learning (Drive-integrated)

**Author:** Abu Sufyan — Student (Organization: Abu Zar)  

This repository contains a Streamlit-based digit recognition chatbot with self-learning capability. Corrections and manual data are saved locally and (optionally) synced to Google Drive using service-account credentials via `st.secrets`.

## Quick start
1. Add your pretrained model to `model/digit_recognition_model.keras` (or train using the Colab notebook in `notebook/`).
2. Add Google Drive service account credentials in Streamlit secrets under `gdrive` (TOML format). Optionally set `gdrive_folder_id` for a drive folder.
3. Install requirements: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Features
- Draw or upload digits, get predictions.
- Save corrections (user-provided labels) and manual dataset images.
- Automatic retraining (fine-tune) after each 5 corrections or manual entries.
- History page with color-coded rows (green=correct, red=incorrect).
- Performance page with accuracy metrics.
- Google Drive sync for persistence (optional; requires `st.secrets`).

## Drive secrets example (in Streamlit Cloud secrets settings)
```toml
[gdrive]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "service-account@your-project.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"

# optional: folder id to use as root
gdrive_folder_id = "your_drive_folder_id"
```

## Notes
- For Streamlit Cloud, place the TOML secrets in the app's Secrets settings (do not commit credentials to GitHub).
- Local mode uses the same `st.secrets` if you create `.streamlit/secrets.toml` locally.
- Model file is intentionally excluded from Git to keep repo small.
