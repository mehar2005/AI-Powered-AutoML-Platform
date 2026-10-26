import streamlit as st
import pandas as pd
from models.Regression.LinearRegression import LinearRegressionShow


ml_models = {
    'Regression' : [
        'Linear Regression', 
        'Polynomial Regression',
        'Decision Tree Regressor',
        'Random Forest Regressor'
    ],
    'Classification' : [
        'Logistic Regression',
        'Decision Tree Classifier',
        'Random Forest Classifier',
        'KNN Classifier'
    ],
    'Clustering' : [
        'K-Means',
        'Agglomerative Clustering'
    ]
}

def ml_show():
    try:
        st.title('Predictions', text_alignment = 'center')

        col1, col2 = st.columns(2, border = True)

        df = st.session_state.get('file_uploaded').copy()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Upload a Dataset first!")

        Regression_models = ['Linear Regression', 'Polynomial Regression', 'Decision Tree Regressor', 'Random Forest Regressor']
        Classification_models = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'K Neighbors Classifier']
        Clustering_models = ['K Means Clustering', 'Agglomerative Clustering']

        with col1:
            st.subheader('Target Variable Selection', text_alignment = 'center') 
            target_var = st.selectbox(
                label = 'Select Target Variable',
                options = sorted(df.columns),
                index = 0,
                key = 'target_variable'
            )

        with col2:
            st.subheader('Problem Type', text_alignment = 'center')

            st.markdown(f'#### Target Variable: {st.session_state.get('target_variable')}')

            if pd.api.types.is_numeric_dtype(df[target_var]):
                if df[target_var].nunique() > 2:
                    st.session_state.detected_problem_type = 'Regression'
                    st.markdown(f"#### Detected Problem Type: {st.session_state.get('detected_problem_type')}")
                else:
                    st.markdown('#### Please verify whether the target represents a regression or classification problem.')
            elif df[target_var].nunique() == 2:
                st.session_state.detected_problem_type = 'Binary Classification'
                st.markdown(f"#### Detected Problem Type: {st.session_state.get('detected_problem_type')}")
            elif df[target_var].nunique() > 2:
                st.session_state.detected_problem_type = 'Multiclass Classification'
                st.markdown(f"#### Detected Problem Type: {st.session_state.get('detected_problem_type')}")
            else:
                st.markdown('#### Invalid Target Variable!')

            problem_type = st.selectbox(
                label = 'Confirm/Change Problem Type',
                options = sorted(list(set([st.session_state.detected_problem_type] + ['Regression', 'Binary Classification', 'Multiclass Classification', 'Clustering']))),
                index = 0,
                key = 'problem_type'
                )

        with st.container(border = True):
            problem_type_heading = 'Regression' if st.session_state.get('problem_type') == 'Regression' else 'Classification' if st.session_state.get('problem_type') in ['Binary Classification', 'Multiclass Classification'] else 'Clustering'
            if st.session_state.problem_type == 'Regression':
                st.subheader(f"List of Available {problem_type_heading} Models : ")
                for models in sorted(Regression_models):
                    st.write(f'• {models}')
            elif st.session_state.problem_type in ['Binary Classification', 'Multiclass Classification']:
                st.subheader(f"List of Available {problem_type_heading} Models : ")
                for models in sorted(Classification_models):
                    st.write(f'• {models}')
            elif st.session_state.problem_type == 'Clustering':
                st.subheader(f"List of Available {problem_type_heading} Models : ")
                for models in sorted(Clustering_models):
                    st.write(f'• {models}')

            st.multiselect(
                label = 'Select Models',
                options = Regression_models if problem_type_heading == 'Regression' else Classification_models if problem_type_heading == 'Classification' else Clustering_models,
                default = ["Linear Regression"] if problem_type_heading == 'Regression' else ['Logistic Regression'] if problem_type_heading == 'Classification' else ["K Means Clustering"],
                key = 'models_tb_train'
            )
            st.selectbox(
                label = 'Select Training:Testing Data Ratio',
                options = sorted(['80:20', '70:30', '75:25', 'Custom']),
                index = 0,
                key = 'train_test_ratio'
            )
            if st.session_state.get('train_test_ratio') == 'Custom':
                st.number_input(
                    label = 'Enter Test Ratio',
                    max_value = 0.50,
                    min_value = 0.20,
                    key = 'custom_test_ratio'
                )
            if st.session_state.get('models_tb_train'):
                st.button(
                    label = 'Train Listed Models',
                    key = 'train_models_bt'
                    )

        with st.container(border = True):

            models_metrics_information = {}
            st.subheader("Model Training Information")
            if st.session_state.get('train_models_bt') == True:
                models = st.session_state.get('models_tb_train')

                test_ratio_str = st.session_state.get('train_test_ratio')
                test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
                custom_test_ratio = st.session_state.get('custom_test_ratio')

                with st.container(border = True):
                    st.markdown("## Train-Test Split Percentage", text_alignment = 'center')
                    col = st.columns(8)
                    col[3].metric(label = 'Training Data', value = 1 - (test_ratio_num or custom_test_ratio), format = 'percent')
                    col[4].metric(label = 'Testing Data', value = (test_ratio_num or custom_test_ratio), format = 'percent')
                
                for model in models:
                    if model == 'Linear Regression':
                        with st.container(border = True):
                            st.markdown("# Linear Regression", text_alignment = 'center')
                            st.markdown(f'### Status -> {st.session_state.get('linear_regression_training_status', 'Not Trained ❎')}', text_alignment = 'center')

                            mae, mse, rmse, r2 = LinearRegressionShow()
                            
                            col1, col2, col3, col4 = st.columns(4, border = True)
                            with col1:
                                st.metric(
                                    label = 'MAE',
                                    value = f'{mae : .3f}'
                                )
                            with col2:
                                st.metric(
                                    label = 'MSE',
                                    value = f'{mse : .3f}'
                                )
                            with col3:
                                st.metric(
                                    label = 'RMSE',
                                    value = f'{rmse : .3f}'
                                )
                            with col4:
                                st.metric(
                                    label = 'R2',
                                    value = f'{r2 * 100 : .3f} %'
                                )

                            models_metrics_information["Linear Regression"] = r2 * 100
                    
    except ValueError as err:
        st.error(err)

            