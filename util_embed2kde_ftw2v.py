# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ftw2v_modules.nlp_preprocessing import get_sentence_spe_embedding_and_pos_tag
from ftw2v_modules.load_model_ftw2v import load_trained_w2v_models
from sklearn.ensemble import IsolationForest

res_w2v, res_fasttext, fast_text_embeddings = load_trained_w2v_models()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def clean_text(text):
    # Remove HTML symbols
    text = re.sub(r'<.*?>', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove punctuations and non-alphabetic characters
    words = [word.lower() for word in words if word.isalpha()]
    
    # Remove French stopwords
    french_stopwords = set(stopwords.words('french'))
    words = [word for word in words if word not in french_stopwords]
    
    # Keep only words with more than 5 characters
    words = [word for word in words if len(word) > 5]
    
    return ' '.join(words)



def get_scores_from_input_output(text_i, text_o, bandwidth=None, n_components_pca=5):

    text_i = clean_text(text_i)
    text_o = clean_text(text_o)
    
    if not(len(str(text_i).split(" "))>5 and len(str(text_o).split(" "))>5):
        return {"hallucination_score":None, 
                "omission_score":None}
    
    xi = get_sentence_spe_embedding_and_pos_tag(text_i, 
                                           res_w2v, 
                                           res_fasttext, 
                                           fast_text_embeddings)
    xi = xi.iloc[:,:300]
    xo = get_sentence_spe_embedding_and_pos_tag(text_o, 
                                           res_w2v, 
                                           res_fasttext, 
                                           fast_text_embeddings)
    xo = xo.iloc[:,:300]
    
    l_pred = hallucination_omissions_detector_with_kde_with_pca(xi, xo, bandwidth=bandwidth,  n_components=n_components_pca)
    
    res_scores = {
        "hallucination_score" :l_pred['xo_kde_i/min_i'].max(),
        "omission_score" : l_pred['xi_kde_o/min_o'].max()}
    return res_scores

def detect_anomalies(xi, xo):
    # Concatenate xi and xo dataframes
    combined_df = pd.concat([xi, xo], axis=0)

    # Train Isolation Forest
    isolation_forest = IsolationForest(n_estimators=10)
    isolation_forest.fit(combined_df)

    # Predict anomalies for xi and xo
    xi_scores = isolation_forest.decision_function(xi)
    xo_scores = isolation_forest.decision_function(xo)

    # Identify anomalies
    xi_anomalies = sum(xi_scores < 0)
    xo_anomalies = sum(xo_scores < 0)

    # Return results as dictionary
    results = {
        'xi_anomalies': xi_anomalies,
        'xo_anomalies': xo_anomalies,
        'xi_scores': xi_scores.tolist(),
        'xo_scores': xo_scores.tolist()
    }

    return results


def get_scores_from_input_output_with_IF(text_i, text_o, bandwidth=None, n_components_pca=5):
    
    if not(len(str(text_i).split(" "))>5 and len(str(text_o).split(" "))>5):
        return None
    
    xi = get_sentence_spe_embedding_and_pos_tag(text_i, 
                                           res_w2v, 
                                           res_fasttext, 
                                           fast_text_embeddings)
    xi = xi.iloc[:,:300]
    xo = get_sentence_spe_embedding_and_pos_tag(text_o, 
                                           res_w2v, 
                                           res_fasttext, 
                                           fast_text_embeddings)
    xo = xo.iloc[:,:300]
    
    pca = PCA(n_components=n_components_pca)  # Retain 95% of variance
    pca = pca.fit(pd.concat([xi, xo], axis=0))
    xi = pca.transform(xi)
    xo = pca.transform(xo)
    xi = pd.DataFrame(xi)
    xo = pd.DataFrame(xo)
    
    res_pred = detect_anomalies(xi, xo)
    
    res_scores = {
        "hallucination_score" : res_pred["xo_anomalies"],
        "omission_score" : res_pred["xi_anomalies"]}
    return res_scores


def anomaly_detection(xi, xo, bandwidth=None):
    # Standardize the data
    scaler = StandardScaler()
    xi_scaled = scaler.fit_transform(xi)
    xo_scaled = scaler.transform(xo)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    xi_pca = pca.fit_transform(xi_scaled)
    xo_pca = pca.transform(xo_scaled)

    # Select optimal bandwidth
    if bandwidth is None:
        bandwidth = select_bandwidth(xi_pca)
    else:
        bandwidth = bandwidth

    # Fit KDE model
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(xi_pca)
    
    max_kde_on_xi, min_kde_on_xi = get_max_min_kde_based_on_x(xi_pca, kde)
    # Compute anomaly scores for observations in xo
    anomaly_scores = np.exp(kde.score_samples(xo_pca))

    return {"anomaly_proba":anomaly_scores,
            "max_kde":max_kde_on_xi,
            "min_kde":min_kde_on_xi,
            "rate_proba_min": anomaly_scores / min_kde_on_xi,
            "rate_proba_max": anomaly_scores / max_kde_on_xi,
            "scaler":scaler,
            "pca":pca,
            "kde":kde,
            "bandwidth":bandwidth}


def hallucination_omissions_detector_with_kde_with_pca(xi_raw, xo_raw, bandwidth=None, n_components=5):
    # Standardize the data
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


def hallucination_omissions_detector_with_kde(xi_raw, xo_raw, bandwidth=None):
    # Standardize the data

    xi = xi_raw.copy()
    xo = xo_raw.copy()

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
    print(kde_o.score_samples(xo))
    print(kde_i.score_samples(xi))
    
    print(max_kde_i)
    print(max_kde_o)
                        
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


def get_max_min_kde_based_on_x(xi, kde):
    xi_scores = np.exp(kde.score_samples(xi))
    return xi_scores.max(), xi_scores.min()


def estimate_max_kde_value(data, bandwidth=0.1, grid_resolution=100):
    # Fit KDE model
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(data)
    
    # Create a grid of points covering the range of the data
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    grid = [np.linspace(min_val, max_val, grid_resolution) for min_val, max_val in zip(data_min, data_max)]
    grid_points = np.meshgrid(*grid)
    grid_points_flat = np.vstack([grid_point.flatten() for grid_point in grid_points]).T
    
    # Evaluate KDE model on the grid points
    kde_values = np.exp(kde.score_samples(grid_points_flat))
    
    # Find the maximum value
    max_kde_value = np.max(kde_values)
    
    return max_kde_value


def get_data_range(xi):
    # Compute the minimum and maximum values for each feature (column)
    min_values = xi.min(axis=0)
    max_values = xi.max(axis=0)
    
    # Combine the minimum and maximum values into a range for each feature
    data_range = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
    
    return data_range


def get_prediction_with_embed2kde_ftw2v(text_i, text_o,bandwidth=3, n_components_pca=5, threshold=7):    
    res = get_scores_from_input_output(text_i, text_o, 
                                                       bandwidth=bandwidth,
                                                       n_components_pca=n_components_pca)
    if pd.isnull(res):
        return None

    try:
        predicted_omission = 1 if res["omission_score"] < threshold else 0
        predicted_hallucination = 1 if res["hallucination_score"] < threshold else 0
    except:
        predicted_omission = None
        predicted_hallucination = None

    return {"is omission": predicted_omission, "omission score": res["omission_score"]}