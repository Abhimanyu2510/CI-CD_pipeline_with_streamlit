import pytest
import numpy as np
from src.train import generate_sample_data, train_clustering_model

def test_generate_sample_data():
    data = generate_sample_data(n_samples=100)
    assert data.shape == (100, 2)

def test_train_clustering_model():
    data = generate_sample_data(n_samples=100)
    kmeans, scaler = train_clustering_model(data, n_clusters=3)
    
    # Test model properties
    assert kmeans.n_clusters == 3
    assert kmeans.cluster_centers_.shape == (3, 2)
    
    # Test prediction
    test_point = np.array([[0, 0]])
    scaled_point = scaler.transform(test_point)
    prediction = kmeans.predict(scaled_point)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1, 2]
