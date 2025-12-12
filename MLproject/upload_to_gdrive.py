import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# =====================================================
# 1. Load Credential Service Account dari GitHub Secrets
# =====================================================
creds_json = os.environ["GDRIVE_CREDENTIALS"]
creds_dict = json.loads(creds_json)

credentials = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/drive"]
)

# =====================================================
# 2. Build Drive API
# =====================================================
service = build("drive", "v3", credentials=credentials)

# =====================================================
# 3. ID Folder Google Drive tujuan upload
#    (bisa Shared Drive atau MyDrive)
# =====================================================
PARENT_FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]


# =====================================================
# 4. Helper: Upload folder secara rekursif
# =====================================================
def upload_directory(local_dir_path, parent_drive_id):

    for item_name in os.listdir(local_dir_path):

        item_path = os.path.join(local_dir_path, item_name)

        # ---------- Jika folder ----------
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

            # Rekursif
            upload_directory(item_path, new_folder_id)

        # ---------- Jika file ----------
        else:
            print(f"[FILE] Uploading: {item_name}")

            metadata = {
                "name": item_name,
                "parents": [parent_drive_id]
            }

            media = MediaFileUpload(item_path, resumable=True)

            service.files().create(
                body=metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True
            ).execute()


# =====================================================
# 5. Upload semua run_id dari folder ./mlruns/0
# =====================================================
LOCAL_MLRUNS = "./mlruns/0"

print("\n=== STARTING UPLOAD TO GOOGLE DRIVE ===")

for run_id in os.listdir(LOCAL_MLRUNS):

    run_path = os.path.join(LOCAL_MLRUNS, run_id)

    if os.path.isdir(run_path):

        # Buat folder run_id di Drive
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

        # Upload isi run_id
        upload_directory(run_path, run_drive_id)

print("\n=== ALL RUNS SUCCESSFULLY UPLOADED ===")
