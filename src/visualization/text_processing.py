from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def compute_term_matrix(pd_series):
    # Initialize CountVectorizer
    count_vectorizer = CountVectorizer(stop_words="english")
    
    # Fit and transform the text data
    term_matrix = count_vectorizer.fit_transform(pd_series).toarray()
    
    return term_matrix, count_vectorizer

def compute_cumulative_variance(term_matrix):
    # Initialize TruncatedSVD
    svd = TruncatedSVD(n_components=min(term_matrix.shape), random_state=42)
    
    # Fit TruncatedSVD to the data
    svd.fit(term_matrix)
    
    # Compute cumulative explained variance ratio
    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
    
    return cumulative_variance