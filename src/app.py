import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained models
kmeans = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Clustering Demo')

# Input for new data points
st.subheader('Enter Data Points')
x = st.number_input('X coordinate', value=0.0)
y = st.number_input('Y coordinate', value=0.0)

if st.button('Predict Cluster'):
    # Transform the input
    input_data = np.array([[x, y]])
    scaled_input = scaler.transform(input_data)
    
    # Predict cluster
    cluster = kmeans.predict(scaled_input)[0]
    
    st.success(f'This point belongs to cluster {cluster}')

    # Optional: Plot the point and its cluster
    if st.checkbox('Show visualization'):
        # Generate sample points for context
        x_range = np.linspace(-3, 3, 100)
        y_range = np.linspace(-3, 3, 100)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Get cluster assignments for all points
        scaled_grid = scaler.transform(grid_points)
        clusters = kmeans.predict(scaled_grid)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': grid_points[:, 0],
            'y': grid_points[:, 1],
            'cluster': clusters
        })
        
        # Plot using Streamlit
        st.scatter_chart(
            df,
            x='x',
            y='y',
            color='cluster',
            size=1
        )
