from huggingface_hub import snapshot_download, hf_hub_download
import os
import pickle



def download_ftw2v():
    repo_id = "AchOk78/ftw2v_fr_healthcare_v1"
    target_dir = "./data/m_models"
    os.makedirs(target_dir, exist_ok=True)
    
    files = ["fr_w2v.pkl", "fr_w2v_fasttext.pkl"]
    
    for filename in files:
        print(f"Attempting to download {filename}...")
        hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_dir=target_dir,
            etag_timeout=30,  # Increase timeout to 30 seconds
            resume_download=True # Resume if a partial download exists
        )
        print(f"Successfully processed {filename}")


def download_bert_base(local_dir:str='./data/m_models/bert-base-uncased'):

    # Download the model
    snapshot_download(
        repo_id="bert-base-uncased",
        local_dir=local_dir,
        local_dir_use_symlinks=False # Set to True if you want to save disk space
    )

    print(f"Model downloaded to: {os.path.abspath(local_dir)}")

if __name__=="__main__":
    download_bert_base()
    download_ftw2v