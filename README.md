# Embed2KDE: Omission Detection in LLM Summaries

Embed2KDE is a modular framework designed to detect omissions in summaries by leveraging word embeddings. 

## Environment Setup
This repository supports multiple embedding backends. Please choose the environment based on the module you intend to use:

### Option 1: BERT Embedding Backend
Use this environment for transformer-based embeddings.
```bash
conda env create -f environment-bert.yml
conda activate embed2kde-bert
```

### Option 2: FTW2V (FastText/Word2Vec) Backend
Use this environment for static embedding models.
```bash
conda env create -f environment-ftw2v.yml
conda activate embed2kde-ftw2v
```

## Extending the Framework
The system is designed as a "black box" that accepts any embedding model. To integrate a new embedding type (e.g., custom models), you must implement a wrapper class in `src/modules/` following the interface pattern:

1. Create a new file in `src/modules/`.
2. Inherit from the base embedding class.
3. Ensure your class implements the necessary methods to vectorize input text.

Example pattern:
```python
class CustomEmbeddingModel:
    def __init__(self, model_name):
        # Implementation details...
        pass
    
    def get_tokens_and_embeddings(self, text):
        # Return vector representation...
        pass
```
