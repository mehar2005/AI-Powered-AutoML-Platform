import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from models.Regression.LinearRegression import LinearRegressionShow
from models.Regression.PolynomialRegression import PolynomialRegressionShow
from models.Regression.KNNeighborsRegression import KNNRegressionShow
from models.Regression.DecisionTreeRegressor import DecisionTreeRegressorShow
from models.Regression.RandomForestRegressor import RandomForestRegressorShow
from models.Classification.KNNeighborsClassifier import KNNClassificationShow
from models.Classification.LogisticRegression import LogisticRegressionShow
from models.Classification.DecisionTreeClassifier import DecisionTreeClassifierShow
from models.Classification.RandomForestClassifier import RandomForestClassifierShow
from models.Classification.NaiveBayes import NaiveBayesShow
from automatic_model_tuning import tuneDTC, tuneKNNC, tuneLGR, tuneNB, tuneRFC, tuneDTR, tuneKNNR, tuneLNR, tunePR, tuneRFR


ml_models = {
    'Regression' : [
        'Linear Regression', 
        'Polynomial Regression',
        'Decision Tree Regressor',
        'Random Forest Regressor',
        'KNN Regressor'
    ],
    'Classification' : [
        'Logistic Regression',
        'Decision Tree Classifier',
        'Random Forest Classifier',
        'KNN Classifier',
        'Naive Bayes Classifier'
    ],
    'Clustering' : [
        'K-Means',
        'Agglomerative Clustering'
    ]
}

def get_test_ratio():
    test_ratio_str = st.session_state.get('train_test_ratio')
    test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
    return test_ratio_num or st.session_state.get('custom_test_ratio')


def get_scaler():
    scaling_tech = st.session_state.get('scaling_technique')

    if scaling_tech == 'Standardization':
        return StandardScaler()
    elif scaling_tech == 'Normalization':
        return MinMaxScaler()

    return None


def get_training_split(X, y):
    return train_test_split(
        X,
        y,
        test_size = get_test_ratio(),
        shuffle = True,
        stratify = y if st.session_state.get('problem_type') in ['Binary Classification', 'Multiclass Classification'] and st.session_state.get('if_stratify_needed') else None,
        random_state = 42
    )


def store_fitted_pipeline(model_name):
    X = st.session_state.get('model_df')
    target_var = st.session_state.get('target_variable')
    uploaded_data = st.session_state.get('file_uploaded')

    if not isinstance(X, pd.DataFrame) or target_var is None or not isinstance(uploaded_data, pd.DataFrame):
        return

    y = uploaded_data.loc[X.index, target_var].copy()
    X_train, _, y_train, _ = get_training_split(X, y)
    scaler = get_scaler()
    steps = []

    if scaler is not None:
        steps.append(('preprocessor', scaler))

    if model_name == 'Linear Regression':
        model = LinearRegression(**(st.session_state.get('best_lnr_params') or {}))
        model_key = 'lnr_model'
    elif model_name == 'Polynomial Regression':
        model = Pipeline([
            ('polynomialfeatures', PolynomialFeatures(include_bias = False)),
            ('linearregression', LinearRegression())
        ])
        best_params = st.session_state.get('best_pr_params')
        if best_params is not None:
            model.set_params(**best_params)
        model_key = 'pr_model'
    elif model_name == 'KNN Regressor':
        model = KNeighborsRegressor(**(st.session_state.get('best_knnr_params') or {}))
        model_key = 'knnr_model'
    elif model_name == 'Decision Tree Regressor':
        model = DecisionTreeRegressor(random_state = 42, **(st.session_state.get('best_dtr_params') or {}))
        model_key = 'dtr_model'
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(**(st.session_state.get('best_lgr_params') or {}))
        model_key = 'lgr_model'
    elif model_name == 'KNN Classifier':
        model = KNeighborsClassifier(**(st.session_state.get('best_knnc_params') or {}))
        model_key = 'knnc_model'
    elif model_name == 'Naive Bayes Classifier':
        model = GaussianNB(**(st.session_state.get('best_nb_params') or {}))
        model_key = 'nb_model'
    elif model_name == 'Decision Tree Classifier':
        model = DecisionTreeClassifier(random_state = 42, **(st.session_state.get('best_dtc_params') or {}))
        model_key = 'dtc_model'
    else:
        return

    steps.append(('model', model))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    if 'trained_model_pipelines' not in st.session_state:
        st.session_state.trained_model_pipelines = {}

    st.session_state.trained_model_pipelines[model_name] = pipeline
    st.session_state[model_key] = pipeline


def apply_scaling():
    try:
        data = st.session_state.get('model_df')

        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Please upload a dataset first!")

        if_cat_col_in_model_df = []

        for col in data.columns:
            if pd.api.types.is_object_dtype(data[col]):
                if_cat_col_in_model_df.append(col)

        if len(if_cat_col_in_model_df) > 0:
            raise ValueError("Dataset contains Categorical column(s). Encode them first!")

        target = st.session_state.get('target_variable')

        if target is None:
            raise ValueError("Select a target variable first!")

        y = st.session_state.get('file_uploaded').loc[data.index, target].copy()
        X_train, X_test, _, _ = get_training_split(data, y)
        scaler = get_scaler()

        if scaler is None:
            raise ValueError("Select a scaling technique first!")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scaled_data = pd.DataFrame(index = data.index, columns = data.columns, dtype = float)
        scaled_data.loc[X_train.index] = X_train_scaled
        scaled_data.loc[X_test.index] = X_test_scaled

        st.session_state.scaled_data = scaled_data.copy()
        st.session_state.fitted_preprocessor = scaler
        st.session_state.scaling_success_message = 'Scaling Applied Successfully!'

    except (ValueError, TypeError, KeyError) as e:
        st.session_state.scaling_exception = f'An Error Occurred during Scaling: {e}'

def use_unscaled_data():
    st.session_state.scaled_data = None
    st.session_state.final_model_df = st.session_state.get('model_df')

def reset_hyperparameters():
    st.session_state.best_lnr_params = None
    st.session_state.best_lgr_params = None
    st.session_state.best_dtr_params = None
    st.session_state.best_dtc_params = None
    st.session_state.best_pr_params = None
    st.session_state.best_knnr_params = None
    st.session_state.best_knnc_params = None
    st.session_state.best_nb_params = None
    st.session_state.best_rfr_params = None
    st.session_state.best_rfc_params = None

def ml_show():
    try:
        df = st.session_state.get('file_uploaded')

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            st.warning("Upload a dataset to start making predictions!")
            return
        
        df = df.copy()
        
        col1, col2 = st.columns(2, border = True)

        Regression_models = sorted(['Linear Regression', 'Polynomial Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'KNN Regressor'])
        Classification_models = sorted(['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'KNN Classifier', 'Naive Bayes Classifier'])
        Clustering_models = sorted(['K Means Clustering', 'Agglomerative Clustering'])

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

            st.markdown(f"#### Target Variable: {st.session_state.get('target_variable')}")

            if df[target_var].nunique() == 2:
                st.session_state.detected_problem_type = 'Binary Classification'
                st.markdown(f"#### Detected Problem Type: {st.session_state.get('detected_problem_type')}")
            elif pd.api.types.is_numeric_dtype(df[target_var]):
                if df[target_var].nunique() > 2:
                    st.session_state.detected_problem_type = 'Regression'
                    st.markdown(f"#### Detected Problem Type: {st.session_state.get('detected_problem_type')}")
                else:
                    st.markdown('#### Please verify whether the target represents a regression or classification problem.')
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

            basic_ml, hyper_tuning = st.tabs(['Basic Model Training', 'Hyper-Parameter Tuning'])

            with basic_ml:
                st.multiselect(
                    label = 'Select Models',
                    options = Regression_models if problem_type_heading == 'Regression' else Classification_models if problem_type_heading == 'Classification' else Clustering_models,
                    default = ["Linear Regression"] if problem_type_heading == 'Regression' else ['Logistic Regression'] if problem_type_heading == 'Classification' else ["K Means Clustering"],
                    key = 'models_tb_train'
                )
                st.multiselect(
                    label = 'Select Column(s) for model',
                    options = ['All'] + sorted([col for col in df.columns if col != st.session_state.get('target_variable')]),
                    default = ['All'],
                    key = 'cols_for_model'
                )
                required_cols = st.session_state.get('cols_for_model')

                if not required_cols or 'All' in required_cols:
                    st.session_state.model_df = df.drop(columns = [st.session_state.get('target_variable')]).copy()
                else:
                    st.session_state.model_df = df[required_cols].copy()

                col1, col2 = st.columns(2, border = True)

                with col1:
                    st.markdown("### Select Scaling Technique", text_alignment = 'center')
                    st.selectbox(
                        label = '',
                        options = sorted(['Normalization', 'Standardization']),
                        index = 0,
                        key = 'scaling_technique'
                    )
                    scaling_bt = st.button(
                        label = "Apply Scaling",
                        key = 'scaling_apply_bt',
                        disabled = st.session_state.get('scaled_data') is not None,
                        on_click = apply_scaling,
                        type = 'primary'
                    )

                    if st.session_state.get('scaled_data') is not None:
                        st.button(
                            label = 'Use Unscaled Data?',
                            key = 'get_back_to_unscaled_data',
                            on_click = use_unscaled_data
                        )

                    scaled_data = st.session_state.get('scaled_data')
                    model_df = st.session_state.get('model_df')

                    if isinstance(scaled_data, pd.DataFrame):
                        if scaled_data.index.equals(model_df.index) and scaled_data.columns.equals(model_df.columns):
                            st.session_state.final_model_df = scaled_data.copy()
                        else:
                            st.session_state.scaled_data = None
                            st.session_state.final_model_df = model_df.copy()
                            st.warning("Dataset or selected columns changed. Please apply scaling again.")
                    else:
                        st.session_state.final_model_df = model_df.copy()

                    if 'scaling_exception' in st.session_state:
                        st.error(st.session_state.pop('scaling_exception'))
                with col2:
                    if st.session_state.get('scaled_data') is None:
                        st.warning("Apply Scaling before training the model(s).")
                    if 'unscaling_sucesss_message' in st.session_state:
                        st.success(st.session_state.pop('unscaling_sucesss_message'))
                    else:
                        if 'scaling_success_message' in st.session_state:
                            st.write(st.session_state.get('scaling_success_message'))

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
                if problem_type_heading == 'Classification':
                    st.checkbox(
                        label = 'Stratify Needed?',
                        key = 'if_stratify_needed'
                    )
                    st.checkbox(
                        label = 'Class Weights Imbalanced?',
                        key = 'if_class_weights_imbalanced'
                    )

                if st.session_state.get('models_tb_train'):
                    st.button(
                        label = 'Train Listed Models',
                        key = 'train_models_bt',
                        type = 'primary'
                        )
                    st.button(
                        label = 'Reset Hyper Parameters',
                        key = 'reset_hyper_params',
                        disabled = True if st.session_state.get('auto_tune_applied') is None else False,
                        on_click = reset_hyperparameters
                    )

            with hyper_tuning:
                st.multiselect(
                    label = 'Select models for Hyperparameter Tuning',
                    options = list(st.session_state.get('models_tb_train')),
                    key = 'models_for_tuning',
                    disabled = True if not st.session_state.get('models_tb_train') else False
                )
                apply_auto_tune = st.button(
                    label = 'Apply Automatic Model Tuning',
                    key = 'apply_auto_tuning',
                    type = 'primary'
                )
                if apply_auto_tune:
                    with st.spinner(
                        text = 'Tuning Model'
                    ):
                        if st.session_state.get('models_tb_train') is None:
                            st.warning("Train models first!")   
                        else:                             
                            st.session_state.auto_tune_applied = True
                            models_for_tuning = st.session_state.get('models_for_tuning')
                            target_var = st.session_state.get('target_variable')

                            original_df = st.session_state.get('file_uploaded')
                            X = st.session_state.get('model_df')

                            if X is None:
                                X = original_df.drop(columns=[target_var]).copy()
                            else:
                                X = X.copy()

                            if X is not None and target_var is not None:
                                y = original_df.loc[X.index, target_var].copy()

                                if len(X) != len(y):
                                    raise ValueError("Features and target have different lengths!")
                                
                                if 'Decision Tree Classifier' in models_for_tuning:
                                    tuneDTC(X, y)
                                if 'KNN Classifier' in models_for_tuning:
                                    tuneKNNC(X, y)
                                if 'Logistic Regression' in models_for_tuning:
                                    tuneLGR(X, y)
                                if 'Naive Bayes Classifier' in models_for_tuning:
                                    tuneNB(X, y)
                                if 'Random Forest Classifier' in models_for_tuning:
                                    tuneRFC(X, y)

                                if 'Decision Tree Regressor' in models_for_tuning:
                                    tuneDTR(X, y)
                                if 'KNN Regressor' in models_for_tuning:
                                    tuneKNNR(X, y)
                                if 'Linear Regression' in models_for_tuning:
                                    tuneLNR(X, y)
                                if 'Polynomial Regression' in models_for_tuning:
                                    tunePR(X, y)
                                if 'Random Forest Regressor' in models_for_tuning:
                                    tuneRFR(X, y)

        if st.session_state.get('train_models_bt'):
            with st.spinner("Training Models"):
                with st.container(border = True):

                    st.session_state.regression_models_metrics_information = {}
                    st.session_state.classification_models_metrics_information = {}
                    if st.session_state.get('train_models_bt') == True:
                        models = st.session_state.get('models_tb_train')

                        st.subheader("Model Training Information")
                        test_ratio_str = st.session_state.get('train_test_ratio')
                        test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
                        custom_test_ratio = st.session_state.get('custom_test_ratio')

                        if st.session_state.get('problem_type') in ['Binary Classification', 'Multiclass Classification']:
                            col1, col2 = st.columns(2, border = True)
                            with col1:
                                st.markdown("## Train-Test Split Percentage", text_alignment = 'center')
                                col = st.columns(8)
                                col[3].metric(label = 'Training Data', value = 1 - (test_ratio_num or custom_test_ratio), format = 'percent')
                                col[4].metric(label = 'Testing Data', value = (test_ratio_num or custom_test_ratio), format = 'percent')
                            with col2:
                                st.markdown("## Target Variable Value Counts")
                                vc = df[st.session_state.get('target_variable')].value_counts()
                                vc_df = vc.reset_index()
                                vc_df.columns = ['Class', 'Count']

                                st.dataframe(vc_df, hide_index = True)

                        elif st.session_state.get('problem_type') == 'Regression':
                            with st.container(border = True):
                                st.markdown("## Train-Test Split Percentage", text_alignment = 'center')
                                col = st.columns(8)
                                col[3].metric(label = 'Training Data', value = 1 - (test_ratio_num or custom_test_ratio), format = 'percent')
                                col[4].metric(label = 'Testing Data', value = (test_ratio_num or custom_test_ratio), format = 'percent')
                    
                        for model in models:
                            if model == 'Linear Regression':
                                with st.container(border = True):
                                    train_metrics, test_metrics = LinearRegressionShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Linear Regression')
                                    status = st.session_state.get('linear_regression_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Linear Regression", text_alignment = 'center')
                                        st.markdown(f"### Status -> {st.session_state.get('linear_regression_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                        with st.expander("View Training Results"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{train_metrics['mae'] : .3f}' if train_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{train_metrics['mse'] : .3f}' if train_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{train_metrics['rmse'] : .3f}' if train_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{train_metrics['r2'] : .3f}' if train_metrics['r2'] is not None else "N/A"
                                                    )
                                            else:
                                                st.error(f'Error: {st.session_state.get('linear_regression_exception')}')
                                                st.warning("No Metrics found!")

                                        with st.expander("View Testing Results"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{test_metrics['mae'] : .3f}' if test_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{test_metrics['mse'] : .3f}' if test_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{test_metrics['rmse'] : .3f}' if test_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{test_metrics['r2'] : .3f}' if test_metrics['r2'] is not None else "N/A"
                                                    )
                                                if 'regression_models_metrics_information' in st.session_state:
                                                    st.session_state.regression_models_metrics_information["Linear Regression"] = test_metrics['r2'] if test_metrics['r2'] is not None else "N/A"
                                            else:
                                                st.error(f'Error: {st.session_state.get('linear_regression_exception')}')
                                                st.warning("No Metrics found!")

                                    else:
                                        if 'linear_regression_exception' in st.session_state:
                                            st.error(st.session_state.pop('linear_regression_exception'))
                                            
                            elif model == 'Polynomial Regression':
                                with st.container(border = True):
                                    train_metrics, test_metrics = PolynomialRegressionShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Polynomial Regression')
                                    status = st.session_state.get('polynomial_regression_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Polynomial Regression", text_alignment = 'center')
                                        st.markdown(f"### Status -> {st.session_state.get('polynomial_regression_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                        with st.expander("View Training Results"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{train_metrics['mae'] : .3f}' if train_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{train_metrics['mse'] : .3f}' if train_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{train_metrics['rmse'] : .3f}' if train_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{train_metrics['r2'] : .3f}' if train_metrics['r2'] is not None else "N/A"
                                                    )
                                            else:
                                                st.error(f'Error: {st.session_state.get('polynomial_regression_exception')}')
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Results"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{test_metrics['mae'] : .3f}' if test_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{test_metrics['mse'] : .3f}' if test_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{test_metrics['rmse'] : .3f}' if test_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{test_metrics['r2'] : .3f}' if test_metrics['r2'] is not None else "N/A"
                                                    )
                                                if 'regression_models_metrics_information' in st.session_state:
                                                    st.session_state.regression_models_metrics_information["Polynomial Regression"] = test_metrics['r2'] if test_metrics['r2'] is not None else "N/A"
                                            else:
                                                st.error(f'Error: {st.session_state.get('polynomial_regression_exception')}')
                                                st.warning("No Metrics found!")

                                    else:
                                        if 'polynomial_regression_exception' in st.session_state:
                                            st.error(st.session_state.pop('polynomial_regression_exception'))
                            
                            elif model == 'KNN Regressor':
                                with st.container(border = True):
                                    train_metrics, test_metrics = KNNRegressionShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('KNN Regressor')
                                    status = st.session_state.get('knn_regression_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# KNN Regressor", text_alignment = 'center')
                                        st.markdown(f"### Status -> {st.session_state.get('knn_regression_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                        with st.expander("View Training Metrics"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{train_metrics['mae'] : .3f}' if train_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{train_metrics['mse'] : .3f}' if train_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{train_metrics['rmse'] : .3f}' if train_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{train_metrics['r2'] : .3f}' if train_metrics['r2'] is not None else "N/A"
                                                    )
                                            else:
                                                st.error(f'Error: {st.session_state.get('knn_regression_exception')}')
                                                st.warning("No Metrics found!")
                                        with st.expander("View Training Metrics"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{test_metrics['mae'] : .3f}' if test_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{test_metrics['mse'] : .3f}' if test_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{test_metrics['rmse'] : .3f}' if test_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{test_metrics['r2'] : .3f}' if test_metrics['r2'] is not None else "N/A"
                                                    )
                                                if 'regression_models_metrics_information' in st.session_state:
                                                    st.session_state.regression_models_metrics_information["KNN Regression"] = test_metrics['r2'] if test_metrics['r2'] is not None else "N/A"
                                            else:
                                                st.error(f'Error: {st.session_state.get('knn_regression_exception')}')
                                                st.warning("No Metrics found!")

                                    else:
                                        if 'knn_regression_exception' in st.session_state:
                                            st.error(st.session_state.pop('knn_regression_exception'))
                            
                            elif model == 'Decision Tree Regressor':
                                with st.container(border = True):
                                    train_metrics, test_metrics = DecisionTreeRegressorShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Decision Tree Regressor')
                                    status = st.session_state.get('decision_tree_regressor_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Decision Tree Regressor", text_alignment = 'center')
                                        st.markdown(f"### Status -> {st.session_state.get('decision_tree_regressor_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                        with st.expander("View Training Metrics"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{train_metrics['mae'] : .3f}' if train_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{train_metrics['mse'] : .3f}' if train_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{train_metrics['rmse'] : .3f}' if train_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{train_metrics['r2'] : .3f}' if train_metrics['r2'] is not None else "N/A"
                                                    )
                                            else:
                                                st.error(f'Error: {st.session_state.get('decision_tree_regressor_exception')}')
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Metrics"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{test_metrics['mae'] : .3f}' if test_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{test_metrics['mse'] : .3f}' if test_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{test_metrics['rmse'] : .3f}' if test_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{test_metrics['r2'] : .3f}' if test_metrics['r2'] is not None else "N/A"
                                                    )
                                                if 'regression_models_metrics_information' in st.session_state:
                                                    st.session_state.regression_models_metrics_information["Decision Tree Regression"] = test_metrics['r2'] if test_metrics['r2'] is not None else "N/A"
                                            else:
                                                st.error(f'Error: {st.session_state.get('decision_tree_regressor_exception')}')
                                                st.warning("No Metrics found!")

                                    else:
                                        if 'decision_tree_regressor_exception' in st.session_state:
                                            st.error(st.session_state.pop('decision_tree_regressor_exception'))
                            elif model == 'Random Forest Regressor':
                                with st.container(border = True):
                                    train_metrics, test_metrics = RandomForestRegressorShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Random Forest Regressor')
                                    status = st.session_state.get('random_forest_regressor_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Random Forest Regressor", text_alignment = 'center')
                                        st.markdown(f"### Status -> {st.session_state.get('random_forest_regressor_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                        with st.expander("View Training Metrics"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{train_metrics['mae'] : .3f}' if train_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{train_metrics['mse'] : .3f}' if train_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{train_metrics['rmse'] : .3f}' if train_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{train_metrics['r2'] : .3f}' if train_metrics['r2'] is not None else "N/A"
                                                    )
                                            else:
                                                st.error(f'Error: {st.session_state.get('random_forest_regressor_exception')}')
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Metrics"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'MAE',
                                                        value = f'{test_metrics['mae'] : .3f}' if test_metrics['mae'] is not None else "N/A"
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'MSE',
                                                        value = f'{test_metrics['mse'] : .3f}' if test_metrics['mse'] is not None else "N/A"
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'RMSE',
                                                        value = f'{test_metrics['rmse'] : .3f}' if test_metrics['rmse'] is not None else "N/A"
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'R2',
                                                        value = f'{test_metrics['r2'] : .3f}' if test_metrics['r2'] is not None else "N/A"
                                                    )
                                                if 'regression_models_metrics_information' in st.session_state:
                                                    st.session_state.regression_models_metrics_information["Random Forest Regressor"] = test_metrics['r2'] if test_metrics['r2'] is not None else "N/A"
                                            else:
                                                st.error(f'Error: {st.session_state.get('random_forest_regressor_exception')}')
                                                st.warning("No Metrics found!")

                                    else:
                                        if 'decision_tree_regressor_exception' in st.session_state:
                                            st.error(st.session_state.pop('decision_tree_regressor_exception'))

                            elif model == 'Logistic Regression':
                                with st.container(border = True):
                                    train_metrics, test_metrics, confusion_matrix, class_report = LogisticRegressionShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Logistic Regression')
                                    status = st.session_state.get('logistic_regression_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Logistic Regression", text_alignment = 'center')
                                        st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                        with st.expander("View Training Metrics"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{train_metrics['accuracy'] * 100 : .3f} %' if train_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{train_metrics['precision'] * 100 : .3f} %' if train_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{train_metrics['recall'] * 100 : .3f} %' if train_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{train_metrics['f1'] : .3f} %' if train_metrics['f1'] is not None else 'N/A'
                                                    )
                                            else:
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Metrics"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{test_metrics['accuracy'] * 100 : .3f} %' if test_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{test_metrics['precision'] * 100 : .3f} %' if test_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{test_metrics['recall'] * 100 : .3f} %' if test_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{test_metrics['f1'] : .3f} %' if test_metrics['f1'] is not None else 'N/A'
                                                    )
                                                    if 'classification_models_metrics_information' in st.session_state:
                                                        st.session_state.classification_models_metrics_information["Logistic Regression"] = test_metrics['f1'] if test_metrics['f1'] is not None else np.nan
                                            else:
                                                st.warning("No Metrics found!")

                                        if confusion_matrix is not None:
                                            conf_matrix_df = pd.DataFrame(
                                                confusion_matrix,
                                                index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                                columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                            )

                                        if class_report is not None:
                                            class_report_df = pd.DataFrame(class_report).transpose()

                                        if confusion_matrix is not None and class_report is not None:
                                            with st.expander("View Other Metrics"):
                                                st.markdown("## Confusion Matrix")
                                                st.dataframe(conf_matrix_df)
                                                st.markdown("## Classification Report")
                                                st.dataframe(class_report_df)
                                        else:
                                            st.warning("No Metrics found!")

                                    else:
                                        if 'logistic_regression_exception' in st.session_state:
                                            st.error(st.session_state.pop('logistic_regression_exception'))
                            
                            elif model == 'KNN Classifier':
                                with st.container(border = True):
                                    train_metrics, test_metrics, confusion_matrix, class_report = KNNClassificationShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('KNN Classifier')
                                    status = st.session_state.get('knn_classification_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# KNN Classifier", text_alignment = 'center')
                                        st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                        with st.expander("View Training Metrics"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{train_metrics['accuracy'] * 100 : .3f} %' if train_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{train_metrics['precision'] * 100 : .3f} %' if train_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{train_metrics['recall'] * 100 : .3f} %' if train_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{train_metrics['f1'] : .3f} %' if train_metrics['f1'] is not None else 'N/A'
                                                    )
                                            else:
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Metrics"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{test_metrics['accuracy'] * 100 : .3f} %' if test_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{test_metrics['precision'] * 100 : .3f} %' if test_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{test_metrics['recall'] * 100 : .3f} %' if test_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{test_metrics['f1'] : .3f} %' if test_metrics['f1'] is not None else 'N/A'
                                                    )
                                                if 'classification_models_metrics_information' in st.session_state:
                                                    st.session_state.classification_models_metrics_information["KNN Classifier"] = test_metrics['f1'] if test_metrics['f1'] is not None else np.nan
                                            else:
                                                st.warning("No Metrics found!")

                                        if confusion_matrix is not None:
                                            conf_matrix_df = pd.DataFrame(
                                            confusion_matrix,
                                            index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                            columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                        )

                                        if class_report is not None:
                                            class_report_df = pd.DataFrame(class_report).transpose()

                                        if confusion_matrix is not None and class_report is not None:
                                            with st.expander("View Other Metrics"):
                                                st.markdown("## Confusion Matrix")
                                                st.dataframe(conf_matrix_df)
                                                st.markdown("## Classification Report")
                                                st.dataframe(class_report_df)
                                        else:
                                            st.warning("No Metrics found!")

                                    else:
                                        if 'knn_classification_exception' in st.session_state:
                                            st.error(st.session_state.pop('knn_classification_exception'))
                            
                            elif model == 'Naive Bayes Classifier':
                                with st.container(border = True):
                                    train_metrics, test_metrics, confusion_matrix, class_report = NaiveBayesShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Naive Bayes Classifier')
                                    status = st.session_state.get('naive_bayes_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Naive Bayes Classifier", text_alignment = 'center')
                                        st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                        with st.expander("View Training Status"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{train_metrics['accuracy'] * 100 : .3f} %' if train_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{train_metrics['precision'] * 100 : .3f} %' if train_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{train_metrics['recall'] * 100 : .3f} %' if train_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{train_metrics['f1'] : .3f} %' if train_metrics['f1'] is not None else 'N/A'
                                                    )
                                            else:
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Status"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{test_metrics['accuracy'] * 100 : .3f} %' if test_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{test_metrics['precision'] * 100 : .3f} %' if test_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{test_metrics['recall'] * 100 : .3f} %' if test_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{test_metrics['f1'] : .3f} %' if test_metrics['f1'] is not None else 'N/A'
                                                    )
                                                if 'classification_models_metrics_information' in st.session_state:
                                                    st.session_state.classification_models_metrics_information["Naive Bayes Classifier"] = test_metrics['f1'] if test_metrics['f1'] is not None else np.nan
                                            else:
                                                st.warning("No Metrics found!")

                                        if confusion_matrix is not None:
                                            conf_matrix_df = pd.DataFrame(
                                            confusion_matrix,
                                            index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                            columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                        )

                                        if class_report is not None:
                                            class_report_df = pd.DataFrame(class_report).transpose()

                                        if confusion_matrix is not None and class_report is not None:
                                            with st.expander("View Other Metrics"):
                                                st.markdown("## Confusion Matrix")
                                                st.dataframe(conf_matrix_df)
                                                st.markdown("## Classification Report")
                                                st.dataframe(class_report_df)
                                        else:
                                            st.warning("No Metrics found!")

                                    else:
                                        if 'naive_bayes_exception' in st.session_state:
                                            st.error(st.session_state.pop('naive_bayes_exception'))
                            
                            elif model == 'Decision Tree Classifier':
                                with st.container(border = True):
                                    train_metrics, test_metrics, confusion_matrix, class_report = DecisionTreeClassifierShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Decision Tree Classifier')
                                    status = st.session_state.get('decision_tree_classifier_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Decision Tree Classifier", text_alignment = 'center')
                                        st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                        with st.expander("View Training Metrics"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{train_metrics['accuracy'] * 100 : .3f} %' if train_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{train_metrics['precision'] * 100 : .3f} %' if train_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{train_metrics['recall'] * 100 : .3f} %' if train_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{train_metrics['f1'] : .3f} %' if train_metrics['f1'] is not None else 'N/A'
                                                    )
                                            else:
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Metrics"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{test_metrics['accuracy'] * 100 : .3f} %' if test_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{test_metrics['precision'] * 100 : .3f} %' if test_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{test_metrics['recall'] * 100 : .3f} %' if test_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{test_metrics['f1'] : .3f} %' if test_metrics['f1'] is not None else 'N/A'
                                                    )
                                                if 'classification_models_metrics_information' in st.session_state:
                                                    st.session_state.classification_models_metrics_information["Decision Tree Classifier"] = test_metrics['f1'] if test_metrics['f1'] is not None else np.nan
                                            else:
                                                st.warning("No Metrics found!")

                                        if confusion_matrix is not None:
                                            conf_matrix_df = pd.DataFrame(
                                            confusion_matrix,
                                            index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                            columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                        )

                                        if class_report is not None:
                                            class_report_df = pd.DataFrame(class_report).transpose()

                                        if confusion_matrix is not None and class_report is not None:
                                            with st.expander("View Other Metrics"):
                                                st.markdown("## Confusion Matrix")
                                                st.dataframe(conf_matrix_df)
                                                st.markdown("## Classification Report")
                                                st.dataframe(class_report_df)
                                        else:
                                            st.warning("No Metrics found!")

                                    else:
                                        if 'decision_tree_classifier_exception' in st.session_state:
                                            st.error(st.session_state.pop('decision_tree_classifier_exception'))
                            elif model == 'Random Forest Classifier':
                                with st.container(border = True):
                                    train_metrics, test_metrics, confusion_matrix, class_report = RandomForestClassifierShow()
                                    if train_metrics is not None:
                                        store_fitted_pipeline('Random Forest Classifier')
                                    status = st.session_state.get('random_forest_classifier_training_status', 'Not Trained ❎')
                                    if status == 'Trained ✅':
                                        st.markdown("# Random Forest Classifier", text_alignment = 'center')
                                        st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                        with st.expander("View Training Metrics"):
                                            if train_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{train_metrics['accuracy'] * 100 : .3f} %' if train_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{train_metrics['precision'] * 100 : .3f} %' if train_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{train_metrics['recall'] * 100 : .3f} %' if train_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{train_metrics['f1'] : .3f} %' if train_metrics['f1'] is not None else 'N/A'
                                                    )
                                            else:
                                                st.warning("No Metrics found!")
                                        with st.expander("View Testing Metrics"):
                                            if test_metrics is not None:
                                                col1, col2, col3, col4 = st.columns(4, border = True)
                                                with col1:
                                                    st.metric(
                                                        label = 'Accuracy',
                                                        value = f'{test_metrics['accuracy'] * 100 : .3f} %' if test_metrics['accuracy'] is not None else 'N/A'
                                                    )
                                                with col2:
                                                    st.metric(
                                                        label = 'Precision',
                                                        value = f'{test_metrics['precision'] * 100 : .3f} %' if test_metrics['precision'] is not None else 'N/A'
                                                    )
                                                with col3:
                                                    st.metric(
                                                        label = 'Recall',
                                                        value = f'{test_metrics['recall'] * 100 : .3f} %' if test_metrics['recall'] is not None else 'N/A'
                                                    )
                                                with col4:
                                                    st.metric(
                                                        label = 'F1-Score',
                                                        value = f'{test_metrics['f1'] : .3f} %' if test_metrics['f1'] is not None else 'N/A'
                                                    )
                                                if 'classification_models_metrics_information' in st.session_state:
                                                    st.session_state.classification_models_metrics_information["Random Forest Classifier"] = test_metrics['f1'] if test_metrics['f1'] is not None else np.nan
                                            else:
                                                st.warning("No Metrics found!")

                                        if confusion_matrix is not None:
                                            conf_matrix_df = pd.DataFrame(
                                            confusion_matrix,
                                            index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                            columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                        )

                                        if class_report is not None:
                                            class_report_df = pd.DataFrame(class_report).transpose()

                                        if confusion_matrix is not None and class_report is not None:
                                            with st.expander("View Other Metrics"):
                                                st.markdown("## Confusion Matrix")
                                                st.dataframe(conf_matrix_df)
                                                st.markdown("## Classification Report")
                                                st.dataframe(class_report_df)
                                        else:
                                            st.warning("No Metrics found!")

                                    else:
                                        if 'random_forest_classifier_exception' in st.session_state:
                                            st.error(st.session_state.pop('random_forest_classifier_exception'))

        regression_model_trained = sorted(st.session_state.get('regression_models_metrics_information'), key = lambda x : x[1])      
        classification_model_trained = sorted(st.session_state.get('classification_models_metrics_information'), key = lambda x : x[1])

        if problem_type_heading == 'Regression':
            with st.container(border = True):

                st.markdown("## Best Model Selection")
                if regression_model_trained is not None:
                    regression_model_trained_df = pd.DataFrame(
                        regression_model_trained.items(),
                        columns = ['Model', 'R2 Value']
                    )
                    st.dataframe(regression_model_trained_df, hide_index = True) 

                    models_dict = {
                        'lnr_model' : st.session_state.get('lnr_model'),
                        'pr_model' : st.session_state.get('pr_model'),
                        'dtr_model' : st.session_state.get('dtr_model'),
                        'knnr_model' : st.session_state.get('knnr_model'),
                        'rfr_model' : st.session_state.get('rfr_model')
                    }

                    if all(x is not None for x in models_dict if x in models):

                        st.selectbox(
                            label = 'Select Model',
                            options = regression_model_trained.keys(),
                            index = 0,
                            key = 'selected_best_reg_model'
                        )

                        if st.button(
                            label = 'Confirm Selection',
                            type = 'primary'
                        ):
                            selected_model = st.session_state.get('selected_best_reg_model')

                            st.session_state.final_selected_model = ''

                            if selected_model == 'Decision Tree Regression':
                                st.session_state.final_selected_model = st.session_state.get('dtr_model')
                            if selected_model == 'Linear Regression':
                                st.session_state.final_selected_model = st.session_state.get('lnr_model')
                            if selected_model == 'Polynomial Regression':
                                st.session_state.final_selected_model = st.session_state.get('pr_model')
                            if selected_model == 'Random Forest Regressor':
                                st.session_state.final_selected_model = st.session_state.get('rfr_model')
                            if selected_model == 'KNN Regression':
                                st.session_state.final_selected_model = st.session_state.get('knnr_model')

                            if st.session_state.get('final_selected_model') is not None:
                                st.session_state.final_model_selected_confirmation = 'Model Selected Successfully!'
                                st.success(st.session_state.get('final_model_selected_confirmation'))

                            # if selected_model == 'Decision Tree Regression':
                            #     st.session_state.best_selected_model = 

        elif problem_type_heading == 'Classification':
            with st.container(border = True):

                st.markdown("## Best Model Selection")
                if classification_model_trained is not None:
                    classification_model_trained_df = pd.DataFrame(
                        classification_model_trained.items(),
                        columns = ['Model', 'F1']
                    )
                    st.dataframe(classification_model_trained_df, hide_index = True) 

                    models_dict = {
                        'lgr_model' : st.session_state.get('lgr_model'),
                        'dtc_model' : st.session_state.get('dtc_model'),
                        'rfc_model' : st.session_state.get('rfc_model'),
                        'knnc_model' : st.session_state.get('knnc_model'),
                        'nb_model' : st.session_state.get('nbc_model')
                    }

                    if all(x is not None for x in models_dict if x in models):

                        st.selectbox(
                            label = 'Select Model',
                            options = classification_model_trained.keys(),
                            index = 0,
                            key = 'selected_best_reg_model'
                        )

                        if st.button(
                            label = 'Confirm Selection',
                            type = 'primary'
                        ):
                            selected_model = st.session_state.get('selected_best_reg_model')

                            st.session_state.final_selected_model = ''

                            if selected_model == 'Decision Tree Classifier':
                                st.session_state.final_selected_model = st.session_state.get('dtc_model')
                            if selected_model == 'Logistic Regression':
                                st.session_state.final_selected_model = st.session_state.get('lgr_model')
                            if selected_model == 'Naive Bayes Classifier':
                                st.session_state.final_selected_model = st.session_state.get('nbc_model')
                            if selected_model == 'Random Forest Classifier':
                                st.session_state.final_selected_model = st.session_state.get('rfc_model')
                            if selected_model == 'KNN Classifier':
                                st.session_state.final_selected_model = st.session_state.get('knnc_model')

                            if st.session_state.get('final_selected_model') is not None:
                                st.session_state.final_model_selected_confirmation = 'Model Selected Successfully!'
                                st.success(st.session_state.get('final_model_selected_confirmation'))
                                st.write(st.session_state.get('final_selected_model'))

                    
    except Exception as err:
        st.exception(err)
        # st.session_state.ml_exception = str(err)
        # if 'ml_exception' in st.session_state:
        #     st.error(st.session_state.pop('ml_exception'))

            
