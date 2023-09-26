from huggingface_hub import HfApi
import os
from huggingface_hub import whoami, create_repo

api = HfApi()
root_dir = os.path.dirname(os.path.dirname(__file__))
user = whoami()["name"]
repo_id = f"{user}/vae"
# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path=root_dir,
    repo_id=repo_id,
    repo_type="model",
    create_pr=True,
)
