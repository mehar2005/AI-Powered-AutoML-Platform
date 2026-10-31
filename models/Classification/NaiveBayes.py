from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

def NaiveBayesShow():
    try:
        test_ratio_str = st.session_state.get('train_test_ratio')
        test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
        custom_test_ratio = st.session_state.get('custom_test_ratio')
        df = st.session_state.get('model_df').copy()
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Upload a Dataset First!")

        target_var = st.session_state.get('target_variable')
        
        y = st.session_state.get('file_uploaded')[target_var].copy()
        scaled_data = st.session_state.get('scaled_data') 
        if scaled_data is not None:
            X = scaled_data
        else:
            st.session_state.get('model_df')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size = test_ratio_num or custom_test_ratio,
            shuffle = True,
            stratify = y if st.session_state.get('if_stratify_needed') else None,
            random_state = 42
        )
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict = True)

        st.session_state.naive_bayes_training_status = 'Trained ✅'

        return accuracy, precision, recall, f1, conf_matrix, class_report 

    except Exception as e:
        st.session_state.naive_bayes_training_status = 'Not Trained ✅'
        st.session_state.naive_bayes_exception = str(e)
        return None, None, None, None, None, None
