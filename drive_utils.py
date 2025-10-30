import streamlit as st
import os, io, json
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
except Exception:
    # If google libs not installed, Drive won't be available
    pass

def drive_available():
    return 'gdrive' in st.secrets and isinstance(st.secrets['gdrive'], dict)

class DriveClient:
    def __init__(self):
        creds_dict = dict(st.secrets['gdrive'])
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/drive'])
        self.service = build('drive','v3', credentials=creds)
        self.root_folder = st.secrets.get('gdrive_folder_id', None)
    def upload_file(self, local_path, remote_folder=None):
        name = os.path.basename(local_path)
        folder_id = self.root_folder if remote_folder is None else None
        file_metadata = {'name': name}
        if remote_folder:
            folder_id = self._ensure_folder(remote_folder)
            if folder_id:
                file_metadata['parents'] = [folder_id]
        media = MediaFileUpload(local_path, resumable=True)
        uploaded = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return uploaded.get('id')
    def _ensure_folder(self, folder_name):
        q = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        res = self.service.files().list(q=q, fields='files(id,name)').execute()
        files = res.get('files',[])
        if files: return files[0]['id']
        meta = {'name': folder_name, 'mimeType':'application/vnd.google-apps.folder'}
        if self.root_folder:
            meta['parents']=[self.root_folder]
        created = self.service.files().create(body=meta, fields='id').execute()
        return created.get('id')
    def download_file(self, file_id, dest_path):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(dest_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return dest_path
