# -*- coding: utf-8 -*-


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_max_min_kde_based_on_x(xi, kde):
    xi_scores = np.exp(kde.score_samples(xi))
    return xi_scores.max(), xi_scores.min()

def hallucination_omissions_detector_with_kde_with_pca(xi_raw, xo_raw, bandwidth=None, n_components=5):
    # Standardize the data

    if not(type(xi_raw)==pd.core.frame.DataFrame) or not(type(xo_raw)==pd.core.frame.DataFrame):
        xi_raw = pd.DataFrame(xi_raw)
        xo_raw = pd.DataFrame(xo_raw)

    scaler = StandardScaler()
    scaler = scaler.fit(pd.concat([xi_raw, xo_raw], axis=0))
    xi = scaler.transform(xi_raw)
    xo = scaler.transform(xo_raw)
    xi = pd.DataFrame(xi)
    xo = pd.DataFrame(xo)
    # Apply PCA
    
    pca = PCA(n_components=n_components)  # Retain 95% of variance
    pca = pca.fit(pd.concat([xi, xo], axis=0))
    xi = pca.transform(xi)
    xo = pca.transform(xo)
    xi = pd.DataFrame(xi)
    xo = pd.DataFrame(xo)

    # Select optimal bandwidth
    if bandwidth is None:
        bandwidthi = select_bandwidth(xi)
        bandwidtho = select_bandwidth(xo)
        bandwidth = min(bandwidthi, bandwidtho)
    else:
        bandwidth = bandwidth

    # Fit KDE model
    kde_i = KernelDensity(bandwidth=bandwidth)
    kde_i.fit(xi)
    max_kde_i, min_kde_i = get_max_min_kde_based_on_x(xi, kde_i)                     
                        
    kde_o = KernelDensity(bandwidth=bandwidth)
    kde_o.fit(xo)
    max_kde_o, min_kde_o = get_max_min_kde_based_on_x(xo, kde_o)
    
    xo_score_with_kde_o = np.exp(kde_o.score_samples(xo))
    xo_score_with_kde_i = np.exp(kde_i.score_samples(xo))
    xi_score_with_kde_o = np.exp(kde_o.score_samples(xi))
    xi_score_with_kde_i = np.exp(kde_i.score_samples(xi))
           
    return {"xo_kde_o/i":xo_score_with_kde_o/xo_score_with_kde_i*max_kde_i/max_kde_o,
            "xo_kde_i/o":xo_score_with_kde_i/xo_score_with_kde_o*max_kde_o/max_kde_i,
            "xi_kde_o/i":xi_score_with_kde_o/xi_score_with_kde_i*max_kde_i/max_kde_o,
            "xi_kde_i/o":xi_score_with_kde_i/xi_score_with_kde_o*max_kde_o/max_kde_i,
            "xo_kde_i/max_i":xo_score_with_kde_i/max_kde_i,
            "xi_kde_o/max_o":xi_score_with_kde_o/max_kde_o,
            "xo_kde_i/min_i":xo_score_with_kde_i/min_kde_i,
            "xi_kde_o/min_o":xi_score_with_kde_o/min_kde_o,
            "bandwidth":bandwidth}

def select_bandwidth(xi):
    # Define a range of bandwidth values to try
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    
    # Determine the number of samples
    n_samples = xi.shape[0]

    # Ensure the number of splits in cross-validation is less than or equal to the number of samples
    n_splits = min(5, n_samples)  # Limiting to a maximum of 5 splits
    
    # Perform grid search to find the best bandwidth
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=n_splits)
    grid.fit(xi)
    
    # Retrieve the best bandwidth
    best_bandwidth = grid.best_params_['bandwidth']
    
    return best_bandwidth

def get_scores_from_input_output_embeddings(embed_i, embed_o, bandwidth=None, n_components_pca=5):
    
    xi = embed_i
    xo = embed_o
    
    l_pred = hallucination_omissions_detector_with_kde_with_pca(xi, xo,
                                                                bandwidth=bandwidth,  
                                                                n_components=n_components_pca)
    
    hallucination_scores = [sigmoid(1./i) for i in l_pred['xo_kde_i/min_i']]
    id_max_hallucination_score = np.argmax(hallucination_scores)
    max_hallucination_score = np.max(hallucination_scores)

    omission_scores = [sigmoid(1./i) for i in l_pred['xi_kde_o/min_o']]
    id_max_omission_score = np.argmax(omission_scores)
    max_omission_score = np.max(omission_scores)

    res_scores = {
        "hallucination_scores" : hallucination_scores,
        "id_max_hallucination_score" : id_max_hallucination_score,
        "max_hallucination_score" : max_hallucination_score,
        "omission_scores" : omission_scores,
        "id_max_omission_score" : id_max_omission_score,
        "max_omission_score" : max_omission_score}
    
    return res_scores

def get_scores_from_input_output_texts(text_i, text_o, embed_model, bandwidth=None, n_components_pca=5):
    
    embed_i = embed_model.get_tokens_and_embeddings(text_i)[1][0,:,:]
    embed_o = embed_model.get_tokens_and_embeddings(text_o)[1][0,:,:]

    res_scores = get_scores_from_input_output_embeddings(embed_i, embed_o, 
                                            bandwidth=bandwidth, n_components_pca=n_components_pca)

    return res_scores