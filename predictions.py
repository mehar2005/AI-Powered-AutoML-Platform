import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from io import BytesIO
import joblib
from feature_engineering import DT_OPERATIONS

def _get_datetime_feature_mapping(data, model_df):
    mapping = st.session_state.get('datetime_feature_mapping', {})
    mapping_id = st.session_state.get('datetime_feature_mapping_id')

    if mapping and data.attrs.get('datetime_feature_mapping_id') != mapping_id:
        mapping = {}
        st.session_state['datetime_feature_mapping'] = mapping

    return {
        output_column: details
        for output_column, details in mapping.items()
        if (
            output_column in model_df.columns
            and isinstance(details, dict)
            and details.get('source')
            and details.get('component') in DT_OPERATIONS
        )
    }

def _get_datetime_value(value, input_type):
    if value is None or (input_type == 'datetime' and not str(value).strip()):
        return pd.NaT
    if input_type == 'datetime':
        return pd.to_datetime(value, errors='raise')
    if input_type == 'time':
        return pd.Timestamp.combine(pd.Timestamp('1970-01-01').date(), value)
    return pd.Timestamp(value)

def _get_input_options(data, column):
    if column not in data.columns:
        return []
    return sorted(data[column].dropna().unique().tolist())

def _get_numeric_categorical_columns(model_df, categorical_mapping, one_hot_mapping, datetime_columns): #To identify actual numerical columns.
    excluded_columns = set(categorical_mapping)
    excluded_columns.update(
        dummy_column
        for category_mapping in one_hot_mapping.values()
        for dummy_column in category_mapping.values()
    )
    excluded_columns.update(datetime_columns)

    return sorted([
        col for col in model_df.columns
        if (
            col not in excluded_columns
            and pd.api.types.is_numeric_dtype(model_df[col])
            and not pd.api.types.is_bool_dtype(model_df[col])
        )
    ])

def _get_prediction_input_data(input_data, model_df, datetime_inputs, datetime_sources):
    for source, extracted_features in datetime_sources.items():
        input_type, raw_value = datetime_inputs[source]
        datetime_value = _get_datetime_value(raw_value, input_type)
        source_series = pd.Series([datetime_value], dtype='datetime64[ns]')

        if source in model_df.columns:
            input_data[source] = datetime_value

        for output_column, component in extracted_features:
            input_data[output_column] = DT_OPERATIONS[component](source_series).iloc[0]

    input_df = pd.DataFrame([input_data], columns=model_df.columns)

    missing_columns = input_df.columns[input_df.isna().any()].tolist()
    if missing_columns:
        raise ValueError(
            f"Prediction inputs are missing values for: {', '.join(missing_columns)}"
        )

    return input_df

def _get_model_classes(model):
    classes = getattr(model, 'classes_', None)

    if classes is None and hasattr(model, 'named_steps'):
        final_estimator = list(model.named_steps.values())[-1]
        classes = getattr(final_estimator, 'classes_', None)

    return classes

def predictions_show():
    try:
        df = st.session_state.get('file_uploaded')
        original_df = st.session_state.get('original_file')
        model_df = st.session_state.get('model_df')
        model = st.session_state.get('final_selected_model')

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Upload a Dataset First!")

        if not isinstance(model_df, pd.DataFrame) or model_df.empty:
            raise ValueError("Train a model first!")

        if model is None or not hasattr(model, 'predict'):
            raise ValueError("Select a trained model first!")

        model_buffer = BytesIO()
        joblib.dump(model, model_buffer)
        st.download_button(
            label = 'Download Selected Model',
            data = model_buffer.getvalue(),
            file_name = 'selected_model.joblib',
            mime = 'application/octet-stream'
        )

        if not isinstance(original_df, pd.DataFrame):
            original_df = df

        input_data = {}
        one_hot_mapping = st.session_state.get('one_hot_mapping', {})
        categorical_mapping = st.session_state.get('categorical_mapping', {})
        datetime_feature_mapping = _get_datetime_feature_mapping(df, model_df)
        encoded_columns = set()
        datetime_columns = set(datetime_feature_mapping)
        datetime_sources = {}

        for output_column, details in datetime_feature_mapping.items():
            datetime_sources.setdefault(details['source'], []).append(
                (output_column, details['component'])
            )

        datetime_columns.update(datetime_sources)

        numeric_categorical_candidates = _get_numeric_categorical_columns(
            model_df,
            categorical_mapping,
            one_hot_mapping,
            datetime_columns
        )

        st.subheader('Prediction Input')
        st.caption('Enter values in their original form. Encoded values and extracted datetime features are prepared automatically.')

        numeric_categorical_columns = st.multiselect(
            label = 'Select numerical features that should be treated as categories',
            options = numeric_categorical_candidates,
            key = 'prediction_numeric_categorical_columns'
        )

        datetime_inputs = {}

        with st.form('prediction_form'):

            for col, category_mapping in categorical_mapping.items():

                if col in model_df.columns and category_mapping:
                    categories = list(category_mapping.keys())

                    selected_category = st.selectbox(
                        label = col,
                        options = categories,
                        key = f'prediction_encoded_{col}'
                    )

                    input_data[col] = category_mapping[selected_category]
                    encoded_columns.add(col)

            for original_col, category_mapping in one_hot_mapping.items():
                dummy_cols = [
                    dummy_col for dummy_col in category_mapping.values()
                    if dummy_col in model_df.columns
                ]

                if dummy_cols:
                    categories = list(category_mapping.keys())

                    selected_category = st.selectbox(
                        label = original_col,
                        options = categories,
                        key = f'prediction_onehot_{original_col}'
                    )

                    selected_dummy = category_mapping[selected_category]

                    for dummy_col in dummy_cols:
                        input_data[dummy_col] = int(dummy_col == selected_dummy)

                    encoded_columns.update(dummy_cols)

            for source, extracted_features in datetime_sources.items():
                components = {component for _, component in extracted_features}
                has_date_component = bool(
                    components.difference({'Hour', 'Minute', 'Second', 'Time Only'})
                )
                has_time_component = bool(
                    components.intersection({'Hour', 'Minute', 'Second', 'Time Only'})
                )

                if has_date_component and has_time_component:
                    datetime_inputs[source] = (
                        'datetime',
                        st.text_input(
                            label = source,
                            placeholder = 'YYYY-MM-DD HH:MM:SS',
                            key = f'prediction_datetime_{source}'
                        )
                    )
                elif has_time_component:
                    datetime_inputs[source] = (
                        'time',
                        st.time_input(
                            label = source,
                            value = None,
                            key = f'prediction_time_{source}'
                        )
                    )
                else:
                    datetime_inputs[source] = (
                        'date',
                        st.date_input(
                            label = source,
                            value = None,
                            key = f'prediction_date_{source}'
                        )
                    )

            for col in model_df.columns:

                if col in encoded_columns or col in datetime_columns:
                    continue

                if col in numeric_categorical_columns:
                    available_categories = _get_input_options(df, col)

                    if not available_categories:
                        available_categories = _get_input_options(original_df, col)

                    if not available_categories:
                        raise ValueError(
                            f"Column '{col}' has no available category values for prediction input."
                        )

                    selected_value = st.selectbox(
                        label = col,
                        options = available_categories,
                        key = f'prediction_category_{col}'
                    )

                    input_data[col] = selected_value

                elif pd.api.types.is_bool_dtype(model_df[col]):
                    input_data[col] = st.checkbox(
                        label = col,
                        key = f'prediction_bool_{col}'
                    )

                elif pd.api.types.is_numeric_dtype(model_df[col]):
                    median_value = pd.to_numeric(model_df[col], errors='coerce').median()
                    if pd.isna(median_value):
                        median_value = 0.0

                    input_data[col] = st.number_input(
                        label = col,
                        value = float(median_value),
                        key = f'prediction_numeric_{col}'
                    )

                elif pd.api.types.is_datetime64_any_dtype(model_df[col]):
                    input_data[col] = pd.Timestamp(
                        st.date_input(
                            label = col,
                            key = f'prediction_date_column_{col}'
                        )
                    )

                else:
                    available_categories = _get_input_options(df, col)

                    if not available_categories:
                        available_categories = _get_input_options(original_df, col)

                    if not available_categories:
                        raise ValueError(
                            f"Column '{col}' has no available values for prediction input."
                        )

                    input_data[col] = st.selectbox(
                        label = col,
                        options = available_categories,
                        key = f'prediction_object_{col}'
                    )

            predict_button = st.form_submit_button('Predict')

        if predict_button:
            input_df = _get_prediction_input_data(
                input_data,
                model_df,
                datetime_inputs,
                datetime_sources
            )

            prediction = model.predict(input_df)

            problem_type = st.session_state.get('problem_type')

            if problem_type == 'Clustering':
                display_prediction = prediction[0]
                result_label = 'Assigned Cluster'
            else:
                target_variable = st.session_state.get('target_variable')
                target_mapping = st.session_state.get('categorical_mapping', {}).get(target_variable, {})
                inverse_mapping = {value: key for key, value in target_mapping.items()}

                display_prediction = inverse_mapping.get(prediction[0], prediction[0])
                result_label = f'Predicted {target_variable}'

            with st.container(border=True):
                st.markdown('## Prediction Result', text_alignment='center')
                st.markdown(f"### {result_label}: {display_prediction}", text_alignment='center')

            if problem_type != 'Clustering' and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_df)[0]

                    print('Actual probabilities:', probabilities)
                    print('Probability sum:', probabilities.sum())
                    print('Model classes:', model.classes_)

                    for cls, prob in zip(model.classes_, probabilities):
                        print(f"Class {cls}: {prob:.10f}")

                    classes = _get_model_classes(model)

                    if classes is not None:
                        target_encoder = st.session_state.get('target_encoder')

                        if target_encoder is not None:
                            classes = target_encoder.inverse_transform(
                                classes.astype(int)
                            )

                        probability_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': probabilities
                        })

                        st.write('Prediction Probabilities')

                        st.dataframe(
                            probability_df,
                            column_config = {
                                'Probability': st.column_config.NumberColumn(
                                    format = '%.10f'
                                )
                            },
                            use_container_width = True
                        )

    except Exception as err:
        st.error(err)
