# -*- coding: utf-8 -*-

import torch
from transformers import BertModel, BertTokenizer

class BertEmbeddingModel:
    """
    A class to represent a BERT model for generating token embeddings.

    Attributes
    ----------
    model_name : str
        The name of the pre-trained BERT model to use.

    Methods
    -------
    get_tokens_and_embeddings(text):
        Tokenizes the input text and returns the tokens and their corresponding embeddings.
    """
    def __init__(self, model_name='bert-base-uncased'):
        """
        Constructs all the necessary attributes for the BertEmbeddingModel object.

        Parameters
        ----------
        model_name : str, optional
            The name of the pre-trained BERT model to use (default is 'bert-base-uncased').
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_tokens_and_embeddings(self, text):
        """
        Tokenizes the input text and returns the tokens and their corresponding embeddings.

        Parameters
        ----------
        text : str
            The input text to tokenize and get embeddings for.

        Returns
        -------
        tokens : list of str
            The list of tokens from the input text.
        token_embeddings : torch.Tensor
            The tensor of token embeddings corresponding to the tokens.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # Get the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state

        # Convert token IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return tokens, token_embeddings

# Example usage
if __name__ == "__main__":
    model = BertEmbeddingModel()
    text = "Your input text here"
    tokens, embeddings = model.get_tokens_and_embeddings(text)
    
    for token, embedding in zip(tokens, embeddings[0]):
        print(f"Token: {token}, Embedding: {embedding}")