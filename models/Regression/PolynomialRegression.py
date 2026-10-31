import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

def PolynomialRegressionShow():
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
            random_state = 42
        )

        poly = PolynomialFeatures(degree = 2, include_bias=False)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()

        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.session_state.polynomial_regression_training_status = 'Trained ✅'
        return mae, mse, rmse, r2

    except Exception as e:
        st.session_state.polynomial_regression_exception = str(e)
        return None, None, None, None
