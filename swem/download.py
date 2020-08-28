import os
import zipfile
from typing import Dict

import requests

# File ids of Google Drive.
W2V_IDS: Dict[str, str] = {
    'ja': '1aoo0t_hhey-7J8wl7Vh-l6dz0n2BoYIr'
}


def download_w2v(lang: str = 'ja') -> None:
    """Download pretrained w2v model of specified language.
    Args:
        lang (str): What language to download.
    """
    file_id: str = W2V_IDS[lang]
    home_dir: str = os.path.expanduser('~')
    swem_dir: str = os.path.join(home_dir, '.swem')
    os.makedirs(swem_dir, exist_ok=True)
    zip_path: str = os.path.join(swem_dir, f'{lang}.zip')
    if os.path.exists(zip_path):
        print(f'{zip_path} is already exists.')
        return
    print(f'Downloading w2v file to {zip_path}')
    _download_file_from_google_drive(file_id, zip_path)

    open_dir: str = os.path.join(swem_dir, f'{lang}')
    print(f'Extract zipfile into {open_dir}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(open_dir)
        print('Success to extract files.')
        os.remove(zip_path)


def _download_file_from_google_drive(file_id: str, destination: str):
    """Download file from Google Drive directly.
    Args:
        file_id (str): File id of the file to download.
        destination (str): A path to save file.
    """
    url: str = 'https://docs.google.com/uc?export=download'

    session = requests.Session()

    resp = session.get(url, params={'id': file_id}, stream=True)
    token = _get_confirm_token(resp)

    if token:
        params: Dict[str, str] = {'id': file_id, 'confirm': token}
        resp = session.get(url, params=params, stream=True)
        _save_response_content(resp, destination)


def _get_confirm_token(response) -> str:
    """Get confirm token from cookies to download large files directory."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return ''


def _save_response_content(response, destination):
    chunk_size: int = 32768

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
