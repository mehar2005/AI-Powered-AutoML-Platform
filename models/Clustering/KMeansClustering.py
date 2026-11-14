from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

def calculate_silhouette(X, labels):
    unique_labels = set(labels)
    if len(X) == 0:
        return None
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return None
    return silhouette_score(X, labels)

def KMeansClusteringShow():
    try:
        st.session_state.k_means_clustering_exception = None
        st.session_state.k_means_clustering_training_status = None

        test_ratio_str = st.session_state.get('train_test_ratio')
        test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
        custom_test_ratio = st.session_state.get('custom_test_ratio')
        df = st.session_state.get('model_df').copy()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Upload a Dataset First!")

        X = st.session_state.get('final_model_df').copy()

        if len(X) < 3:
            raise ValueError("Atleast 3 samples are required!")

        X_train, X_test = train_test_split(
            X,
            test_size = test_ratio_num or custom_test_ratio,
            shuffle = True,
            random_state = 42
        )

        best_params = st.session_state.get('best_kmc_params') or {}

        n_clusters = best_params.get('n_clusters', 8)
        if n_clusters < 2 or n_clusters >= len(X_train):
            raise ValueError("Invalid number of clusters!")
        if n_clusters >= len(X_test):
            raise ValueError("Test dataset contains insufficient samples!")
        
        model = KMeans(
            random_state = 42,
            **best_params
        )

        model.fit(X_train)

        y_train_pred = model.labels_
        y_pred = model.predict(X_test)

        st.session_state.kmc_model = model

        train_silhouette = calculate_silhouette(X_train, y_train_pred)

        test_silhouette = calculate_silhouette(X_test, y_pred)


        train_metrics = {
            'silhouette' : train_silhouette,
        }

        test_metrics = {
            'silhouette' : test_silhouette,
        }

        st.session_state.k_means_clustering_training_status = 'Trained ✅'
        return train_metrics, test_metrics
    except Exception as e:
        st.session_state.k_means_clustering_exception = str(e)
        return None, None