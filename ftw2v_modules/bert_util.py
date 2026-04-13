# -*- coding: utf-8 -*-


try:
    import torch
    from ftw2v_modules.util import split_long_strings
    import numpy as np
    import ftw2v_modules.util as utl_nlp
    from ftw2v_modules.utils.character_cnn import CharacterIndexer

    def get_embedding_with_camembert(text, tokenizer, model,
                                     limit_nb_car=512,
                                     embedding_size=(768,)):
        """Get embeddings for short sentences.
    
        Args:
            text (_type_): text to embedd
            tokenizer (_type_): camembert tokenizer
            model (_type_): camembert model
    
        Returns:
            (np.array): the embedding of the text
        """
        sentences = split_long_strings(text, nb_char=limit_nb_car)
        embedding_sentences = []
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            sentence = sentence[:limit_nb_car]
            tokenized_sentence = tokenizer.tokenize(sentence)
            encoded_sentence = tokenizer.encode(tokenized_sentence)
            encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
            embedding_sentences.append(model(encoded_sentence)["pooler_output"]
                                       .squeeze().detach().numpy().copy())
        try:
            embedding_sentences = sum(embedding_sentences) / \
                len(embedding_sentences)
        except:
            embedding_sentences = np.zeros(embedding_size)
    
        return embedding_sentences


    def get_embedding_with_charbert(text, tokenizer, model,
                                    limit_nb_car=512,
                                    embedding_size=(768,)):
        """Get embeddings for short sentences.
    
        Args:
            text (_type_): text to embedd
            tokenizer (_type_): camembert tokenizer
            model (_type_): camembert model
    
        Returns:
            (np.array): the embedding of the text
        """
        sentences = split_long_strings(text, nb_char=limit_nb_car)
        embedding_sentences = []
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            sentence = sentence[:limit_nb_car]
            x = tokenizer.basic_tokenizer.tokenize(sentence)
            x = ['[CLS]', *x, '[SEP]']
            indexer = CharacterIndexer()
            batch = [x]
            batch_ids = indexer.as_padded_tensor(batch)
            embeddings_for_batch, _ = model(batch_ids)
            embedding_sentences.append(torch.tensor(
                embeddings_for_batch).squeeze().detach().numpy().mean(axis=0).copy())
        try:
            embedding_sentences = sum(embedding_sentences) / \
                len(embedding_sentences)
        except:
            embedding_sentences = np.zeros(embedding_size)
    
        return embedding_sentences

except:
    print("TORCH NOT INSTALLED")
