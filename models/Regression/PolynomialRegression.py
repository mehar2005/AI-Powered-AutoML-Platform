from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

def PolynomialRegressionShow():
    try:
        test_ratio_str = st.session_state.get('train_test_ratio')
        test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
        custom_test_ratio = st.session_state.get('custom_test_ratio')

        df = st.session_state.get('model_df')

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Upload a Dataset First!")

        target_var = st.session_state.get('target_variable')
        y = st.session_state.get('file_uploaded')[target_var].copy()
        X = st.session_state.get('final_model_df').copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_ratio_num or custom_test_ratio,
            shuffle=True,
            random_state=42
        )

        model = Pipeline([
            ('polynomialfeatures', PolynomialFeatures(include_bias=False)),
            ('linearregression', LinearRegression())
        ])

        best_params = st.session_state.get('best_pr_params')

        if best_params is not None:
            model.set_params(**best_params)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': root_mean_squared_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }

        test_metrics = {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': root_mean_squared_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }

        st.session_state.polynomial_regression_training_status = 'Trained ✅'

        return train_metrics, test_metrics

    except Exception as e:
        st.session_state.polynomial_regression_exception = str(e)
        return None, None