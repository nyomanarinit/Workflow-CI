import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# =====================================================
# 1. Load Credential Service Account dari GitHub Secrets
# =====================================================
creds_json = os.environ.get("GDRIVE_CREDENTIALS", "")

if creds_json.strip() == "":
    raise ValueError("ERROR: GDRIVE_CREDENTIALS is EMPTY. Pastikan Secret sudah diset!")

try:
    creds_dict = json.loads(creds_json)
except json.JSONDecodeError:
    raise ValueError("ERROR: GDRIVE_CREDENTIALS bukan JSON valid. Periksa formatting Secret.")

credentials = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/drive"]
)

# =====================================================
# 2. Build Drive API
# =====================================================
service = build("drive", "v3", credentials=credentials)

# =====================================================
# 3. Folder ID Google Drive tujuan upload
# =====================================================
PARENT_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")

if PARENT_FOLDER_ID.strip() == "":
    raise ValueError("ERROR: GDRIVE_FOLDER_ID belum diset di GitHub Secrets!")

# =====================================================
# 4. Helper Upload Folder/Files
# =====================================================
def upload_directory(local_dir_path, parent_drive_id):
    for item_name in os.listdir(local_dir_path):
        item_path = os.path.join(local_dir_path, item_name)

        # ---------- If Folder ----------
        if os.path.isdir(item_path):
            metadata = {
                "name": item_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_drive_id]
            }
            created = service.files().create(
                body=metadata,
                fields="id",
                supportsAllDrives=True
            ).execute()

            new_folder_id = created["id"]
            print(f"[FOLDER] Created: {item_name} (ID: {new_folder_id})")

            upload_directory(item_path, new_folder_id)

        # ---------- If File ----------
        else:
            print(f"[FILE] Uploading: {item_name}")
            metadata = {"name": item_name, "parents": [parent_drive_id]}
            media = MediaFileUpload(item_path, resumable=True)

            service.files().create(
                body=metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True
            ).execute()

# =====================================================
# 5. Upload seluruh run_id dari folder mlruns/0
# =====================================================
LOCAL_MLRUNS = "./mlruns/0"

print("\n=== STARTING UPLOAD TO GOOGLE DRIVE ===")

if not os.path.exists(LOCAL_MLRUNS):
    raise FileNotFoundError("Folder './mlruns/0' tidak ditemukan. Pastikan MLflow sudah jalan!")

for run_id in os.listdir(LOCAL_MLRUNS):
    run_path = os.path.join(LOCAL_MLRUNS, run_id)

    if os.path.isdir(run_path):
        folder_metadata = {
            "name": run_id,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [PARENT_FOLDER_ID]
        }

        created_run_folder = service.files().create(
            body=folder_metadata,
            fields="id",
            supportsAllDrives=True
        ).execute()

        run_drive_id = created_run_folder["id"]

        print(f"\n=== Created run_id folder: {run_id} (ID: {run_drive_id}) ===")

        upload_directory(run_path, run_drive_id)

print("\n=== ALL RUNS SUCCESSFULLY UPLOADED ===")
