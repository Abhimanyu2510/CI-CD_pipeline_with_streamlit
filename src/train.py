import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    return np.random.randn(n_samples, 2)

def train_clustering_model(data, n_clusters=3):
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)
    
    # Save the models
    joblib.dump(kmeans, 'kmeans_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return kmeans, scaler

if __name__ == "__main__":
    # Generate and train on sample data
    data = generate_sample_data()
    train_clustering_model(data)
