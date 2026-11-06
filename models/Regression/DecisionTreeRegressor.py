from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

def DecisionTreeRegressorShow():
    try:
        test_ratio_str = st.session_state.get('train_test_ratio')
        test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
        custom_test_ratio = st.session_state.get('custom_test_ratio')
        df = st.session_state.get('model_df').copy()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Upload a Dataset First!")

        target_var = st.session_state.get('target_variable')

        X = st.session_state.get('final_model_df').copy()
        y = st.session_state.get('file_uploaded').loc[X.index, target_var].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size = test_ratio_num or custom_test_ratio,
            shuffle = True,
            random_state = 42
        )

        best_params = st.session_state.get('best_dtr_params') or {}
        model = DecisionTreeRegressor(
            random_state = 42,
            **best_params
        )

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mae = mean_absolute_error(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = root_mean_squared_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        train_metrics = {
            'mae' : train_mae,
            'mse' : train_mse,
            'rmse' : train_rmse,
            'r2' : train_r2
        }

        test_metrics = {
            'mae' : test_mae,
            'mse' : test_mse,
            'rmse' : test_rmse,
            'r2' : test_r2
        }

        st.session_state.decision_tree_regressor_training_status = 'Trained ✅'
        return train_metrics, test_metrics
    except Exception as e:
        st.session_state.decision_tree_regressor_exception = str(e)
        return None, None