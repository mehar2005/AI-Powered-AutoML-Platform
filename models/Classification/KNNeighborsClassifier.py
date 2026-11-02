from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

def KNNClassificationShow():
    try:
        test_ratio_str = st.session_state.get('train_test_ratio')
        test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
        custom_test_ratio = st.session_state.get('custom_test_ratio')
        df = st.session_state.get('model_df').copy()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Upload a Dataset First!")

        target_var = st.session_state.get('target_variable')
        
        y = st.session_state.get('file_uploaded')[target_var].copy()
        X = st.session_state.get('final_model_df').copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size = test_ratio_num or custom_test_ratio,
            shuffle = True,
            stratify = y if st.session_state.get('if_stratify_needed') else None,
            random_state = 42
        )
        
        best_params = st.session_state.get('best_knnc_params') or {}

        model = KNeighborsClassifier(
            **best_params
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average = 'weighted', zero_division = 0)
        train_recall = recall_score(y_train, y_train_pred, average = 'weighted', zero_division = 0)
        train_f1 = f1_score(y_train, y_train_pred, average = 'weighted', zero_division = 0)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average = 'weighted', zero_division = 0)
        test_recall = recall_score(y_test, y_pred, average = 'weighted', zero_division = 0)
        test_f1 = f1_score(y_test, y_pred, average = 'weighted', zero_division = 0)

        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict = True)

        train_metrics = {
            'accuracy' : train_accuracy,
            'precision' : train_precision,
            'recall' : train_recall,
            'f1' : train_f1
        }
        test_metrics = {
            'accuracy' : test_accuracy,
            'precision' : test_precision,
            'recall' : test_recall,
            'f1' : test_f1
        }
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict = True)

        st.session_state.knn_classification_training_status = 'Trained ✅'

        return train_metrics, test_metrics, conf_matrix, class_report 

    except Exception as e:
        st.session_state.knn_classification_training_status = 'Not Trained ✅'
        st.session_state.knn_classification_exception = str(e)
        return None, None, None, None