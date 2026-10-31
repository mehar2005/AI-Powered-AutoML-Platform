import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.Regression.LinearRegression import LinearRegressionShow
from models.Regression.PolynomialRegression import PolynomialRegressionShow
from models.Regression.KNNeighborsRegression import KNNRegressionShow
from models.Regression.DecisionTreeRegressor import DecisionTreeRegressorShow
from models.Classification.KNNeighborsClassifier import KNNClassificationShow
from models.Classification.LogisticRegression import LogisticRegressionShow
from models.Classification.DecisionTreeClassifier import DecisionTreeClassifierShow
from models.Classification.NaiveBayes import NaiveBayesShow


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

        scaling_tech = st.session_state.get('scaling_technique')
        target = st.session_state.get('target_variable')

        X = data

        if scaling_tech == 'Standardization':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        elif scaling_tech == 'Normalization':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        st.session_state.scaled_data = X
        st.session_state.scaling_success_message = 'Scaling Applied Successfully!'

    except (ValueError, TypeError, KeyError) as e:
        st.session_state.scaling_exception = f'An Error Occurred during Scaling: {e}'

def ml_show():
    try:
        st.title('Predictions', text_alignment = 'center')

        df = st.session_state.get('file_uploaded').copy()

        if not isinstance(df, pd.DataFrame) or df.empty:
            st.warning("Upload a dataset to start making predictions!")
            return
        
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
                    st.button(
                        label = "Apply Scaling",
                        key = 'scaling_apply_bt',
                        disabled = True if st.session_state.get('scaled_data') is not None else False,
                        on_click = apply_scaling,
                        type = 'primary'
                        )

                    if 'scaling_exception' in st.session_state:
                        st.error(st.session_state.pop('scaling_exception'))
                with col2:
                    if st.session_state.get('scaled_data') is None:
                        st.warning("Apply Scaling before training the model(s).")
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

        if st.session_state.get('train_models_bt'):
            with st.container(border = True):

                st.session_state.regression_models_metrics_information = {}
                st.session_state.classification_models_metrics_information = {}
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
                                mae, mse, rmse, r2 = LinearRegressionShow()
                                status = st.session_state.get('linear_regression_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# Linear Regression", text_alignment = 'center')
                                    st.markdown(f"### Status -> {st.session_state.get('linear_regression_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'MAE',
                                            value = f'{mae : .3f}' if mae is not None else "N/A"
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'MSE',
                                            value = f'{mse : .3f}' if mse is not None else "N/A"
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'RMSE',
                                            value = f'{rmse : .3f}' if rmse is not None else "N/A"
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'R2',
                                            value = f'{r2 * 100 : .3f} %' if r2 is not None else "N/A"
                                        )

                                    if 'regression_models_metrics_information' in st.session_state:
                                        st.session_state.regression_models_metrics_information["Linear Regression"] = r2 * 100 if r2 is not None else "N/A"

                                else:
                                    if 'linear_regression_exception' in st.session_state:
                                        st.error(st.session_state.pop('linear_regression_exception'))
                                        
                        elif model == 'Polynomial Regression':
                            with st.container(border = True):
                                mae, mse, rmse, r2 = PolynomialRegressionShow()
                                status = st.session_state.get('polynomial_regression_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# Polynomial Regression", text_alignment = 'center')
                                    st.markdown(f"### Status -> {st.session_state.get('polynomial_regression_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'MAE',
                                            value = f'{mae : .3f}' if mae is not None else "N/A"
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'MSE',
                                            value = f'{mse : .3f}' if mse is not None else "N/A"
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'RMSE',
                                            value = f'{rmse : .3f}' if rmse is not None else "N/A"
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'R2',
                                            value = f'{r2 * 100 : .3f} %' if r2 is not None else "N/A"
                                        )

                                    if 'regression_models_metrics_information' in st.session_state:
                                        st.session_state.regression_models_metrics_information["Polynomial Regression"] = r2 * 100 if r2 is not None else "N/A"

                                else:
                                    if 'polynomial_regression_exception' in st.session_state:
                                        st.error(st.session_state.pop('polynomial_regression_exception'))
                        
                        elif model == 'KNN Regressor':
                            with st.container(border = True):
                                mae, mse, rmse, r2 = KNNRegressionShow()
                                status = st.session_state.get('knn_regression_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# KNN Regressor", text_alignment = 'center')
                                    st.markdown(f"### Status -> {st.session_state.get('knn_regression_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'MAE',
                                            value = f'{mae : .3f}' if mae is not None else "N/A"
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'MSE',
                                            value = f'{mse : .3f}' if mse is not None else "N/A"
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'RMSE',
                                            value = f'{rmse : .3f}' if rmse is not None else "N/A"
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'R2',
                                            value = f'{r2 * 100 : .3f} %' if r2 is not None else "N/A"
                                        )

                                    if 'regression_models_metrics_information' in st.session_state:
                                        st.session_state.regression_models_metrics_information["KNN Regression"] = r2 * 100 if r2 is not None else "N/A"

                                else:
                                    if 'knn_regression_exception' in st.session_state:
                                        st.error(st.session_state.pop('knn_regression_exception'))
                        
                        elif model == 'Decision Tree Regressor':
                            with st.container(border = True):
                                mae, mse, rmse, r2 = DecisionTreeRegressorShow()
                                status = st.session_state.get('decision_tree_regressor_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# Decision Tree Regressor", text_alignment = 'center')
                                    st.markdown(f"### Status -> {st.session_state.get('decision_tree_regressor_training_status', 'Not Trained ❎')}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'MAE',
                                            value = f'{mae : .3f}' if mae is not None else "N/A"
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'MSE',
                                            value = f'{mse : .3f}' if mse is not None else "N/A"
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'RMSE',
                                            value = f'{rmse : .3f}' if rmse is not None else "N/A"
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'R2',
                                            value = f'{r2 * 100 : .3f} %' if r2 is not None else "N/A"
                                        )

                                    if 'regression_models_metrics_information' in st.session_state:
                                        st.session_state.regression_models_metrics_information["Decision Tree Regression"] = r2 * 100 if r2 is not None else "N/A"

                                else:
                                    if 'decision_tree_regressor_exception' in st.session_state:
                                        st.error(st.session_state.pop('decision_tree_regressor_exception'))

                        elif model == 'Logistic Regression':
                            with st.container(border = True):
                                accuracy, precision, recall, f1, confusion_matrix, class_report = LogisticRegressionShow()
                                status = st.session_state.get('logistic_regression_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# Logistic Regression", text_alignment = 'center')
                                    st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'Accuracy',
                                            value = f'{accuracy * 100 : .3f} %' if accuracy is not None else 'N/A'
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'Precision',
                                            value = f'{precision * 100 : .3f} %' if precision is not None else 'N/A'
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'Recall',
                                            value = f'{recall * 100 : .3f} %' if recall is not None else 'N/A'
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'F1-Score',
                                            value = f'{f1 * 100 : .3f} %' if f1 is not None else 'N/A'
                                        )

                                    conf_matrix_df = pd.DataFrame(
                                        confusion_matrix,
                                        index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                        columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                    )

                                    class_report_df = pd.DataFrame(class_report).transpose()

                                    with st.expander("View Other Metrics"):
                                        st.markdown("## Confusion Matrix")
                                        st.dataframe(conf_matrix_df)
                                        st.markdown("## Classification Report")
                                        st.dataframe(class_report_df)

                                    if 'classification_models_metrics_information' in st.session_state:
                                        st.session_state.classification_models_metrics_information["Logistic Regression"] = f1 * 100 if f1 is not None else np.nan

                                else:
                                    if 'logistic_regression_exception' in st.session_state:
                                        st.error(st.session_state.pop('logistic_regression_exception'))
                        
                        elif model == 'KNN Classifier':
                            with st.container(border = True):
                                accuracy, precision, recall, f1, confusion_matrix, class_report = KNNClassificationShow()
                                status = st.session_state.get('knn_classification_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# KNN Classifier", text_alignment = 'center')
                                    st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'Accuracy',
                                            value = f'{accuracy * 100 : .3f} %' if accuracy is not None else 'N/A'
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'Precision',
                                            value = f'{precision * 100 : .3f} %' if precision is not None else 'N/A'
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'Recall',
                                            value = f'{recall * 100 : .3f} %' if recall is not None else 'N/A'
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'F1-Score',
                                            value = f'{f1 * 100 : .3f} %' if f1 is not None else 'N/A'
                                        )

                                    if confusion_matrix is not None:
                                        conf_matrix_df = pd.DataFrame(
                                        confusion_matrix,
                                        index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                        columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                    )

                                    if class_report is not None:
                                        class_report_df = pd.DataFrame(class_report).transpose()

                                    with st.expander("View Other Metrics"):
                                        st.markdown("## Confusion Matrix")
                                        st.dataframe(conf_matrix_df)
                                        st.markdown("## Classification Report")
                                        st.dataframe(class_report_df)

                                    if 'classification_models_metrics_information' in st.session_state:
                                        st.session_state.classification_models_metrics_information["KNN Classifier"] = f1 * 100 if f1 is not None else np.nan

                                else:
                                    if 'knn_classification_exception' in st.session_state:
                                        st.error(st.session_state.pop('knn_classification_exception'))
                        
                        elif model == 'Naive Bayes Classifier':
                            with st.container(border = True):
                                accuracy, precision, recall, f1, confusion_matrix, class_report = NaiveBayesShow()
                                status = st.session_state.get('naive_bayes_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# Naive Bayes Classifier", text_alignment = 'center')
                                    st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'Accuracy',
                                            value = f'{accuracy * 100 : .3f} %' if accuracy is not None else 'N/A'
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'Precision',
                                            value = f'{precision * 100 : .3f} %' if precision is not None else 'N/A'
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'Recall',
                                            value = f'{recall * 100 : .3f} %' if recall is not None else 'N/A'
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'F1-Score',
                                            value = f'{f1 * 100 : .3f} %' if f1 is not None else 'N/A'
                                        )

                                    if confusion_matrix is not None:
                                        conf_matrix_df = pd.DataFrame(
                                        confusion_matrix,
                                        index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                        columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                    )

                                    if class_report is not None:
                                        class_report_df = pd.DataFrame(class_report).transpose()

                                    with st.expander("View Other Metrics"):
                                        st.markdown("## Confusion Matrix")
                                        st.dataframe(conf_matrix_df)
                                        st.markdown("## Classification Report")
                                        st.dataframe(class_report_df)

                                    if 'classification_models_metrics_information' in st.session_state:
                                        st.session_state.classification_models_metrics_information["Naive Bayes Classifier"] = f1 * 100 if f1 is not None else np.nan

                                else:
                                    if 'naive_bayes_exception' in st.session_state:
                                        st.error(st.session_state.pop('naive_bayes_exception'))
                        
                        elif model == 'Decision Tree Classifier':
                            with st.container(border = True):
                                accuracy, precision, recall, f1, confusion_matrix, class_report = DecisionTreeClassifierShow()
                                status = st.session_state.get('decision_tree_classifier_training_status', 'Not Trained ❎')
                                if status == 'Trained ✅':
                                    st.markdown("# Decision Tree Classifier", text_alignment = 'center')
                                    st.markdown(f"### Status -> {status}", text_alignment = 'center')

                                    col1, col2, col3, col4 = st.columns(4, border = True)
                                    with col1:
                                        st.metric(
                                            label = 'Accuracy',
                                            value = f'{accuracy * 100 : .3f} %' if accuracy is not None else 'N/A'
                                        )
                                    with col2:
                                        st.metric(
                                            label = 'Precision',
                                            value = f'{precision * 100 : .3f} %' if precision is not None else 'N/A'
                                        )
                                    with col3:
                                        st.metric(
                                            label = 'Recall',
                                            value = f'{recall * 100 : .3f} %' if recall is not None else 'N/A'
                                        )
                                    with col4:
                                        st.metric(
                                            label = 'F1-Score',
                                            value = f'{f1 * 100 : .3f} %' if f1 is not None else 'N/A'
                                        )

                                    if confusion_matrix is not None:
                                        conf_matrix_df = pd.DataFrame(
                                        confusion_matrix,
                                        index = [f'Actual {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))],
                                        columns = [f'Predicted {i}' for i in sorted(set(df[st.session_state.get('target_variable')].unique()))]
                                    )

                                    if class_report is not None:
                                        class_report_df = pd.DataFrame(class_report).transpose()

                                    with st.expander("View Other Metrics"):
                                        st.markdown("## Confusion Matrix")
                                        st.dataframe(conf_matrix_df)
                                        st.markdown("## Classification Report")
                                        st.dataframe(class_report_df)

                                    if 'classification_models_metrics_information' in st.session_state:
                                        st.session_state.classification_models_metrics_information["Decision Tree Classifier"] = f1 * 100 if f1 is not None else np.nan

                                else:
                                    if 'decision_tree_classifier_exception' in st.session_state:
                                        st.error(st.session_state.pop('decision_tree_classifier_exception'))

                    st.json(st.session_state.get('regression_models_metrics_information'))
                    st.json(st.session_state.get('classification_models_metrics_information'))
                    
    except Exception as err:
        st.session_state.ml_exception = str(err)
        if 'ml_exception' in st.session_state:
            st.error(st.session_state.pop('ml_exception'))

            