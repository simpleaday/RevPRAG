# RevPRAG
We are excited to share the code and datasets from our study on the RevPRAG, making them publicly available for further research and development.

![流程图](images/workflow.svg)

# Models Used
Our experiments leverage several Large Language Models (LLMs) and retrievers from Huggingface, a reputable platform hosting a diverse array of LLMs. The specific models utilized in our study include:
## LLMs
- GPT2-XL：[View on Huggingface](https://huggingface.co/openai-community/gpt2-xl)
- Llama-2-7B：[View on Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Mistral-7B：[View on Huggingface](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- Llama-3-8B：[View on Huggingface](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Llama-2-13B：[View on Huggingface](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
## retrievers
- contriever：[View on Huggingface](https://huggingface.co/facebook/contriever)
- contriever-msmarco：[View on Huggingface](https://huggingface.co/facebook/contriever-msmarco)
- dpr-multi：[View on Huggingface](https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base)
- ance：[View on Huggingface](https://huggingface.co/sentence-transformers/msmarco-roberta-base-ance-firstp)


# Setup Environment
- Python: ``3.9.18``
- Ecco: ``0.1.2``
- H5py: ``3.6.0``
- Captum: ``0.6.0``
- Huggingface_hub: ``0.24.2``
- Beir:``2.0.0``
- PyTorch:``2.2.1+cu121``

# Dataset and Codebase
When running our code, the datasets will be automatically downloaded and saved in ``datasets``.

## Code Files
- Poisoned Data Collection (``get_poisoned_data.py``): Using the retrieved texts and the prompts to query the LLMs and collect activations for correct reponses and poisoned responses.
- Prepare Data (``manage_save_data_h5.py``): Preprocesses and transforms theactivations for effective learning.
- Train the Model (``train_model_test``): Trains the RevPRAG.

## Replicating Our Experiments
1. Fill the Target Path: Specify the directory where you wish the datasets and models to be stored.
2. Run Poisoned Data Collection: Execute python ``run.py`` in the terminal.
3. Execute Prepare Data: Run python ``manage_save_data_h5.py`` in the terminal.
4. Train the Model: Execute python ``train_model_test`` in the terminal.

Following these steps will guide you through the process of poisoned data collection, preparing the data, and training the RevPRAG on your dataset.




