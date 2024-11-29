from transformers import AutoModel, AutoTokenizer
if __name__ == "__main__":
    print('hello')
    import os
    from huggingface_hub import snapshot_download
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print('downloading entire files...')
    snapshot_download(repo_id="mistralai/Pixtral-12B-2409", repo_type="model",
                    local_dir="/data/models/Pixtral-12B",
                    local_dir_use_symlinks=False, resume_download=True,
                    token='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')
