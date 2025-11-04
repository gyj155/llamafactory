from huggingface_hub import snapshot_download
import os

# Define dataset name and target path
repo_id = "zd11024/Video-3D-LLM_data"
local_dir = "llamafactory/data-3dllm"  # Create a subfolder in the target path to store the dataset

# Ensure the target directory exists, create if it does not
os.makedirs(local_dir, exist_ok=True)

print(f"Starting download of dataset '{repo_id}' to '{local_dir}'...")

# Use snapshot_download to download the dataset
# local_dir: Specify the download directory
# repo_type: Explicitly specify that the download is a dataset
# resume_download: Allow resuming download if interrupted
snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset", resume_download=True)

print(f"Dataset '{repo_id}' has been successfully downloaded to '{local_dir}'.")

# The VSI-Bench dataset may contain compressed files (e.g., .zip), you may need to manually extract them
# Below is an extraction example, you need to adjust according to the actual downloaded file type
# Check if there are .zip files in the download directory, if so, extract them
for root, dirs, files in os.walk(local_dir):
    for file in files:
        if file.endswith(".zip"):
            zip_file_path = os.path.join(root, file)
            extract_dir = os.path.splitext(zip_file_path)[0] # Extract to a folder with the same name
            print(f"Detected compressed file: {zip_file_path}, extracting to {extract_dir}...")
            try:
                import zipfile
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Successfully extracted: {zip_file_path}")
                # Optional: Delete the original zip file after extraction
                # os.remove(zip_file_path)
                # print(f"Deleted original compressed file: {zip_file_path}")
            except Exception as e:
                print(f"Failed to extract file {zip_file_path}: {e}")

print("Download and processing complete.")