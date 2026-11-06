import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore

def _get_cat_col(data):
    return sorted(data.select_dtypes(include = ['object', 'category', 'string']).columns.tolist(), key = str)
def _get_num_col(data):
    return sorted(data.select_dtypes(include = ['number']).columns.tolist(), key = str)
def _get_bool_col(data):
    return sorted(data.select_dtypes(include = ['bool', 'boolean']).columns.tolist(), key = str)
def _get_datetime_col(data):
    return sorted([col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])], key=str)

def update_active_dataset(updated_df):
    try:
        if not isinstance(updated_df, pd.DataFrame):
            raise ValueError('Updated dataset must be a DataFrame.')

        st.session_state.file_uploaded = updated_df.copy()

        st.session_state.dataset_version = (
            st.session_state.get('dataset_version', 0) + 1
        )

    except Exception:
        st.error("An error occured while saving the data!")

def remove_null_values():
    try:
        column = st.session_state.get('remove_null_column')
        df = st.session_state.get('file_uploaded').copy()

        if column in df.columns:
            updated_df = df.dropna(subset=[column])

        update_active_dataset(updated_df)

        st.session_state.null_removal_message = "Null values removed successfully!"
    except Exception:
        st.session_state.null_removal_exception = "An error occured while removing null values!"

def fill_null_values():
    try:
        uploaded_df = st.session_state.get('file_uploaded')

        if uploaded_df is None:
            raise ValueError("Please upload a dataset first!")

        df = uploaded_df.copy()

        column_type = st.session_state.get('fill_null_column_type')

        if column_type == 'Categorical':
            column = st.session_state.get('fill_null_column_cat')
            mode = st.session_state.get('fill_null_mode_cat')
            if column == 'All':
                columns = _get_cat_col(df)
            else:
                columns = [column]

            for col in columns:
                if col in df.columns:
                    if df[col].isnull().sum() > 0:
                        if mode == 'Mode':
                            mode_values = df[col].mode()
                            if not mode_values.empty:
                                df[col] = df[col].fillna(mode_values[0])
                        elif mode == 'Forward Fill':
                            df[col] = df[col].ffill()
                        elif mode == 'Backward Fill':
                            df[col] = df[col].bfill()
                    else:
                        continue
                else:
                    err_in_col = col
                    raise ValueError(f'{err_in_col} column not found!')

        elif column_type == 'Numerical':
            column = st.session_state.get('fill_null_column_num')
            mode = st.session_state.get('fill_null_mode_num')
            if column == 'All':
                columns = _get_num_col(df)
            else:
                columns = [column]
            for col in columns:
                if col in df.columns:
                    if df[col].isnull().sum() > 0:
                        if mode == 'Mean':
                            mean_value = df[col].mean()
                            if pd.notna(mean_value):
                                df[col] = df[col].fillna(mean_value)
                        elif mode == 'Median':
                            median_value = df[col].median()
                            if pd.notna(median_value):
                                df[col] = df[col].fillna(median_value)
                    else:
                        continue
                else:
                    err_in_col = col
                    raise ValueError(f'{err_in_col} column not found!')

        elif column_type == 'Datetime':
            column = st.session_state.get('fill_null_column_dt')
            mode = st.session_state.get('fill_null_mode_dt')
            if column == 'All':
                cols = _get_datetime_col(df)
            else:
                cols = [column]

            for col in cols:
                if col in df.columns:
                    if df[col].isnull().sum() > 0:
                        if mode == 'Forward Fill':
                            df[col] = df[col].ffill()
                        elif mode == 'Backward Fill':
                            df[col] = df[col].bfill()
                        elif mode == 'Mode':
                            most_frequent_date = df[col].mode()
                            if not most_frequent_date.empty:
                                df[col] = df[col].fillna(most_frequent_date.iloc[0])
                    else:
                        continue
                else:
                    err_in_col = col
                    raise ValueError(f'{err_in_col} column not found!')
            
        elif column_type == 'Boolean':
            column = st.session_state.get('fill_null_column_bool')
            mode = st.session_state.get('fill_null_mode_bool')
            if column == 'All':
                cols = _get_bool_col(df)
            else:
                cols = [column]

            for col in cols:
                if col in df.columns:
                    if df[col].isnull().sum() > 0:
                        if mode == 'Forward Fill':
                            df[col] = df[col].ffill()
                        elif mode == 'Backward Fill':
                            df[col] = df[col].bfill()
                        elif mode == 'Mode':
                            mode_values = df[col].mode()
                            if not mode_values.empty:
                                df[col] = df[col].fillna(mode_values.iloc[0])
                    else:
                        continue
                else:
                    err_in_col = col
                    raise ValueError(f'{err_in_col} column not found!')

        update_active_dataset(df)
        st.session_state.null_fill_message = 'Null values filled successfully!'
    except Exception:
        st.session_state.null_filling_exception = "An error occured while filling null values!"

def duplicates_removal():
    try:
        dup_rem_type = st.session_state.get('dup_rem_type')
        df = st.session_state.get('file_uploaded').copy()

        subset = None
        
        if dup_rem_type == 'Keep First':
            df = df.drop_duplicates(subset = subset, keep = 'first')
        if dup_rem_type == 'Keep Last':
            df = df.drop_duplicates(subset = subset, keep = 'last')
        if dup_rem_type == 'Remove All':
            df = df.drop_duplicates(subset = subset, keep = False)

        update_active_dataset(df)

        st.session_state.dup_rem_message = 'Duplicates Removed Successfully!'
    except Exception:
        st.session_state.duplicates_removal_exception = "An error occured while removing duplicate values!"


def apply_column_name_transformation():
    try:
        df_copy = st.session_state.file_uploaded.copy()
        c_type = st.session_state.get("col_name_correctness_type")
        df_copy.columns = df_copy.columns.astype(str)
                    
        if c_type in ["Strip", "Left Strip", "Right Strip"]:
            char = st.session_state.get("col_name_strip_char", " ")
            if c_type == "Strip":
                df_copy.columns = df_copy.columns.str.strip(char)
            elif c_type == "Left Strip":
                df_copy.columns = df_copy.columns.str.lstrip(char)
            elif c_type == "Right Strip":
                df_copy.columns = df_copy.columns.str.rstrip(char)
            st.session_state.column_name_transformation_msg = "Column names stripped successfully!"
                            
        elif c_type in ["Capitalize", "Capitalize + Strip"]:
            if c_type == "Capitalize":
                df_copy.columns = df_copy.columns.str.capitalize()
                st.session_state.column_name_transformation_msg = "Column names capitalized successfully!"
            elif c_type == "Capitalize + Strip":
                df_copy.columns = df_copy.columns.str.strip().str.capitalize()
                st.session_state.column_name_transformation_msg = "Column names capitalized and stripped successfully!"
                            
        update_active_dataset(df_copy)

        st.session_state.column_name_op_applied_message = "Column Name Updated Successfully!"

        st.rerun()
    except Exception:
        st.session_state.apply_column_name_transformation_exception = "An error occured while updating column names."

def apply_operation(x, op, chars=" "):
    try:
        if pd.isna(x):
            return x
        if not isinstance(x, (str, np.str_)):
            return x
        if op in ["Strip", "Capitalize + Strip"]:
            x = x.strip(chars) if chars else x.strip()
        elif op == "Left Strip":
            x = x.lstrip(chars) if chars else x.lstrip()
        elif op == "Right Strip":
            x = x.rstrip(chars) if chars else x.rstrip()
        if op in ["Capitalize", "Capitalize + Strip"]:
            x = x.capitalize()
        return x
    except Exception as e:
        st.session_state.apply_operation_exception = str(e)

def apply_column_values_transformation():
    try:
        df_copy = st.session_state.get('file_uploaded').copy()

        c_type = st.session_state.get("col_values_correctness_type")
        column = st.session_state.get("column_values_correctness")
        chars = st.session_state.get("col_values_strip_char", " ")

        if column is None or column not in df_copy.columns:
            st.session_state.column_values_transformation_msg = (
                "Please select a valid column."
            )
            return

        if c_type is None:
            st.session_state.column_values_transformation_msg = (
                "Please select a transformation."
            )
            return

        valid_operations = [
            "Strip",
            "Left Strip",
            "Right Strip",
            "Capitalize",
            "Capitalize + Strip",
            "Make all values equal",
            "Change specific value(s)"
        ]

        if c_type not in valid_operations:
            st.session_state.column_values_transformation_msg = (
                "Invalid transformation selected."
            )
            return

        if c_type == 'Make all values equal':
            df_copy[column] = st.session_state.new_value
        elif c_type == 'Change specific value(s)':
            old_value = st.session_state.old_value
            new_value = st.session_state.new_value

            if pd.api.types.is_float_dtype(df_copy[column]):
                old_value = float(old_value)
                new_value = float(new_value)

                df_copy.loc[df_copy[column].round(2) == round(old_value, 2), column] = new_value

            elif pd.api.types.is_integer_dtype(df_copy[column]):
                old_value = int(old_value)
                new_value = int(new_value)

                df_copy[column] = df_copy[column].replace(old_value, new_value)

            else:
                df_copy[column] = df_copy[column].replace(old_value, new_value)

        df_copy[column] = df_copy[column].apply(lambda x: apply_operation(x, c_type, chars))

        update_active_dataset(df_copy)

        messages = {
            "Strip": "Column values stripped successfully!",
            "Left Strip": "Leading characters removed successfully!",
            "Right Strip": "Trailing characters removed successfully!",
            "Capitalize": "Column values capitalized successfully!",
            "Capitalize + Strip": (
                "Column values capitalized and stripped successfully!"
            ),
            "Make all values equal" : "All values changed successfully!",
            "Change specific value(s)" : "Value(s) changed successfully!"
        }

        st.session_state.column_values_transformation_msg = messages[c_type]
        st.rerun()
    except Exception as e:
        st.session_state.apply_column_values_transformation_exception = "An error occured while updating column values!"


def apply_data_format_transformation(): #AI, need to understand this!
    try:
        df_copy = st.session_state.get('file_uploaded').copy()

        column = st.session_state.get("format_column")
        format_type = st.session_state.get("format_type")
        operation = st.session_state.get("format_operation")

        if column is None or column not in df_copy.columns:
            st.session_state.data_format_transformation_msg = (
                "Please select a valid column."
            )
            return

        if format_type is None or operation is None:
            st.session_state.data_format_transformation_msg = (
                "Please select a valid format and operation."
            )
            return

        series = df_copy[column]
        invalid_values = 0

        if format_type == "Text":

            if operation == "Strip Whitespace":
                df_copy[column] = series.apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

            elif operation == "Collapse Multiple Spaces":
                df_copy[column] = series.apply(
                    lambda x: " ".join(x.split())
                    if isinstance(x, str) else x
                )

            elif operation == "Lowercase":
                df_copy[column] = series.apply(
                    lambda x: x.lower() if isinstance(x, str) else x
                )

            elif operation == "Uppercase":
                df_copy[column] = series.apply(
                    lambda x: x.upper() if isinstance(x, str) else x
                )

            elif operation == "Title Case":
                df_copy[column] = series.apply(
                    lambda x: x.title() if isinstance(x, str) else x
                )

        # Numerical Format Correction
        elif format_type == "Numerical":

            values = series.astype("string").str.strip()

            if operation == "Remove Symbols":
                values = values.str.replace(
                    r"^[^\w.]+|[^\w.]+$", "", regex=True
                )

                converted = pd.to_numeric(
                    values, errors="coerce"
                )

            elif operation == "Convert Percentage to Decimal":
                percentage_mask = values.str.endswith(
                    "%", na=False
                )

                values = values.str.replace(
                    "%", "", regex=False
                )

                converted = pd.to_numeric(
                    values, errors="coerce"
                )

                converted.loc[percentage_mask] = (
                    converted.loc[percentage_mask] / 100
                )

                invalid_values = int(
                    (series.notna() & converted.isna()).sum()
                )

            df_copy[column] = converted

        # Datetime Format Correction
        elif format_type == "Datetime":

            input_format = st.session_state.get(
                "format_input_date"
            )
            output_format = st.session_state.get(
                "format_output_date"
            )

            date_formats = {
                "DD/MM/YYYY": "%d/%m/%Y",
                "MM/DD/YYYY": "%m/%d/%Y",
                "YYYY-MM-DD": "%Y-%m-%d",
                "DD-MM-YYYY": "%d-%m-%Y",
                "MM-DD-YYYY": "%m-%d-%Y",
                "YYYY/MM/DD": "%Y/%m/%d",
                "DD Mon YYYY": "%d %b %Y",
            }

            output_formats = {
                "YYYY-MM-DD": "%Y-%m-%d",
                "DD/MM/YYYY": "%d/%m/%Y",
                "MM/DD/YYYY": "%m/%d/%Y",
                "DD-MM-YYYY": "%d-%m-%Y",
                "YYYY/MM/DD": "%Y/%m/%d",
            }


            if input_format == "Auto Detect":
                date_order = st.session_state.get(
                    "format_date_order", "Day First"
                )

                converted = pd.to_datetime(
                    series,
                    errors="coerce",
                    dayfirst=(date_order == "Day First")
                )
            else:
                date_format = (
                    st.session_state.get("format_custom_date")
                    if input_format == "Custom"
                    else date_formats[input_format]
                )

                converted = pd.to_datetime(
                    series,
                    format=date_format,
                    errors="coerce"
                )

            invalid_values = int(
                (series.notna() & converted.isna()).sum()
            )

            df_copy[column] = converted.dt.strftime(
                output_formats[output_format]
            )

        # Boolean Format Correction
        elif format_type == "Boolean":

            boolean_values = {
                "yes": True,
                "y": True,
                "true": True,
                "1": True,
                "no": False,
                "n": False,
                "false": False,
                "0": False,
            }

            def convert_boolean(x):
                if pd.isna(x):
                    return pd.NA

                if isinstance(x, (bool, np.bool_)):
                    return bool(x)

                value = str(x).strip().lower()

                if value in boolean_values:
                    return boolean_values[value]

                return pd.NA

            converted = series.apply(convert_boolean).astype(
                "boolean"
            )

            invalid_values = int(
                (series.notna() & converted.isna()).sum()
            )

            df_copy[column] = converted

        update_active_dataset(df_copy)

        st.session_state.data_format_transformation_msg = (
            f"Data format corrected successfully! "
            f"{invalid_values} invalid values could not be converted."
        )
        st.rerun()

    except (ValueError, TypeError, KeyError) as e:
        st.session_state.apply_data_format_transformation_exception = "An error occured while changing format of data!"

def refresh_data():
    st.rerun()

@st.fragment
def data_preview():
    try:
        data = st.session_state.get('file_uploaded').copy()
        st.subheader("Dataset Preview", text_alignment = 'center')
        col1, col2 = st.columns([0.20, 0.80], border = True)
        with col1:
            preview_type = st.selectbox(
                label = "Select Preview Type",
                options = ["First", "Last", "Random"],
                index = 0,
                key = 'preview_type',
                help = 'Enter preview type you want in the dataset preview'
            )
            preview_number = st.number_input(
                label = 'Number of rows to display',
                value = 1,
                min_value = 1,
                max_value = len(data),
                step = 1,
                key = 'preview_number',
                help = 'Enter number of rows to show in the dataset preview'
            )
            if st.session_state.get('preview_type') == "First":
                if st.session_state.get('preview_number'):
                    with col2:
                        st.dataframe(data.head(preview_number), hide_index = True)
            elif st.session_state.get('preview_type') == "Last":
                if st.session_state.get('preview_number'):
                    with col2:
                        st.dataframe(data.tail(preview_number).iloc[::-1], hide_index = True)
            elif st.session_state.get('preview_type') == "Random":
                if st.session_state.get('preview_number'):
                    with col2:
                        st.dataframe(data.sample(n = preview_number), hide_index = True)

            st.button(
                label = 'Refresh Preview',
                type = 'primary',
                on_click = refresh_data
            )
    except Exception:
        st.session_state.data_preview_exception = 'An error occured while previewing the data!'

@st.cache_data
def detect_data_types(df):
    try:
        num_col = []
        cat_col = []
        bool_col = []
        dt_col = []
        text_col = []
        id_col = []

        for col in df.columns:
            series = df[col]
            if pd.api.types.is_bool_dtype(series):
                bool_col.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                unique_ratio = series.nunique() / len(series) if len(series) else 0
                if unique_ratio >= 0.95:
                    id_col.append(col)
                else:
                    num_col.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                dt_col.append(col)
            else:
                numeric_ratio = pd.to_numeric(series, errors = 'coerce').notna().mean()
                datetime_ratio = pd.to_datetime(series, errors = 'coerce').notna().mean()
                unique_ratio = series.nunique() / series.notna().sum() if series.notna().sum() else 0

                if numeric_ratio >= 0.95:
                    num_col.append(col)
                elif datetime_ratio >= 0.95:
                    dt_col.append(col)
                elif unique_ratio >= 0.95:
                    id_col.append(col)
                elif series.nunique() <= 20:
                    cat_col.append(col)
                else:
                    text_col.append(col)

        return {
            'Categorical': cat_col,
            'Numerical': num_col,
            'Datetime': dt_col,
            'Boolean': bool_col,
            'Text': text_col,
            'Identifier': id_col
        }
    except Exception:
        st.session_state.dtype_detection_exception = 'An error occured while detecting the data types of columns!'

def dtype_detection_changing(col1_name, col2_name):
    try:
        st.subheader('Datatype Detection & Handling', text_alignment = 'center')

        df = st.session_state.file_uploaded.copy()
        types = detect_data_types(df)

        with col1_name:
            st.subheader("Automatically Detected Data Types", text_alignment = 'center')

            col = st.columns(3, border = True)

            for i, (datatype, columns) in enumerate(types.items()):
                with col[i % 3]:
                    st.metric(
                        label = f'{datatype} columns',
                        value = len(columns)
                    )
                    with st.expander("View columns"):
                        for column in columns:
                            st.markdown(f'• `{column}`')

        with col2_name:
            st.subheader("Manual Datatype Conversion", text_alignment = 'center')

            if len(df.columns) == 0:
                st.warning("No Columns found!")
                return

            col_nm = st.selectbox(
                label = 'Select column',
                options = df.columns,
                key = 'dtype_column'
            )

            detected_type = next(datatype for datatype, columns in types.items() if col_nm in columns)

            st.write(f'Detected Datatype: {detected_type}')
            st.write(f'Pandas Datatype: {df[col_nm].dtype}')

            chng_dtype = st.selectbox(
                label = 'Change Datatype to:',
                options = sorted(['Integer', 'Float', 'Boolean', 'Datetime', 'Category', 'String']),
                key = 'change_datatype'
            )

            if st.session_state.get('change_datatype') == 'Datetime':
                st.checkbox(
                    label = 'Date First?',
                    key = 'dt_chng_datatype_checkbox'
                )

            st.caption("*Note - Performing wrong operation on wrong datatype will not cause any error. Infact, it will show as operation completed.")

            apply_bt = st.button(
                label = 'Apply',
                key = 'apply_dtype'
            )

            if apply_bt:
                if chng_dtype == 'Integer':
                    converted = pd.to_numeric(df[col_nm], errors = 'coerce').astype('Int64')
                elif chng_dtype == 'Float':
                    converted = pd.to_numeric(df[col_nm], errors = 'coerce')
                elif chng_dtype == 'Category':
                    converted = df[col_nm].astype('category')
                elif chng_dtype == 'String':
                    converted = df[col_nm].astype('string')
                elif chng_dtype == 'Datetime':
                    converted = pd.to_datetime(
                        df[col_nm],
                        format='mixed',
                        dayfirst=st.session_state.get('dt_chng_datatype_checkbox', False),
                        errors='coerce'
                    )
                elif chng_dtype == 'Boolean':
                    converted = df[col_nm].astype('boolean')

                updated_df = st.session_state.file_uploaded.copy()
                updated_df[col_nm] = converted
                update_active_dataset(updated_df)
                st.session_state.change_dtype_success_message = 'Datatype Changed Successfully!'

            if 'change_dtype_success_message' in st.session_state:
                st.success(st.session_state.pop('change_dtype_success_message'))
                
    except Exception:
        st.session_state.dtype_changing_exception = 'An error occured while changing the data type of column!'
        if 'dtype_changing_exception' in st.session_state:
            st.error(st.session_state.pop('dtype_changing_exception'))
@st.fragment
def data_format_handling():
    try:
        df = st.session_state.get('file_uploaded').copy()
        st.subheader("Data Format Handling", text_alignment="center")
        with st.container(border=True):
                                        format_column = st.selectbox(
                                            label="Select Column",
                                            options=sorted(df.columns.tolist()),
                                            key="format_column"
                                        )
                                        format_type = st.selectbox(
                                            label="Select Format Type",
                                            options=[
                                                "Text",
                                                "Numerical",
                                                "Datetime",
                                                "Boolean"
                                            ],
                                            key="format_type"
                                        )
                                        if format_type == "Text":
                                            format_operations = [
                                                "Strip Whitespace",
                                                "Collapse Multiple Spaces",
                                                "Lowercase",
                                                "Uppercase",
                                                "Title Case"
                                            ]
                                        elif format_type == "Numerical":
                                            format_operations = [
                                                "Remove Symbols",
                                                "Convert Percentage to Decimal"
                                            ]    
                                        elif format_type == "Datetime":
                                            format_operations = [
                                                "Standardize Date Format"
                                            ]
                                            input_date = st.selectbox(
                                                label="Current Date Format",
                                                options=[
                                                    "Auto Detect",
                                                    "DD/MM/YYYY",
                                                    "MM/DD/YYYY",
                                                    "YYYY-MM-DD",
                                                    "DD-MM-YYYY",
                                                    "MM-DD-YYYY",
                                                    "YYYY/MM/DD",
                                                    "DD Mon YYYY",
                                                    "Custom"
                                                ],
                                                key="format_input_date"
                                            )
                                            if input_date == "Auto Detect":
                                                date_order = st.selectbox(
                                                    label="Interpret Date As",
                                                    options=["Day First", "Month First"],
                                                    key="format_date_order"
                                                )
                                            if input_date == "Custom":
                                                st.text_input(
                                                    label="Enter Python Date Format",
                                                    placeholder="%d/%m/%Y",
                                                    key="format_custom_date"
                                                )
                                            output_date = st.selectbox(
                                                label="Standardized Date Format",
                                                options=[
                                                    "YYYY-MM-DD",
                                                    "DD/MM/YYYY",
                                                    "MM/DD/YYYY",
                                                    "DD-MM-YYYY",
                                                    "YYYY/MM/DD"
                                                ],
                                                key="format_output_date"
                                            )
                                        else:
                                            format_operations = [
                                                "Standardize Boolean Values"
                                            ]
                                        operation = st.selectbox(
                                            label="Select Operation",
                                            options=format_operations,
                                            key="format_operation"
                                        )
                                        st.caption("*Note - Performing wrong operation on wrong datatype will not cause any error. Infact, it will show as operation completed.")
                                        st.button(
                                            label="Apply Format Correction",
                                            key="apply_data_format",
                                            on_click=apply_data_format_transformation
                                        )
                                        if st.session_state.get(
                                            "data_format_transformation_msg"
                                        ):
                                            st.success(
                                                st.session_state.data_format_transformation_msg
                                            )
                                            del st.session_state.data_format_transformation_msg
                                        if 'apply_data_format_transformation_exception' in st.session_state:
                                            st.error(st.session_state.pop('apply_data_format_transformation_exception'))
    except Exception:
        st.session_state.data_format_exception = 'An error occured while changing format of the data!'

@st.fragment
def columns_handling():
    try:
        st.subheader('Columns Handling', text_alignment = 'center')
        data = st.session_state.get('file_uploaded').copy()
        if not isinstance(data, pd.DataFrame):
            st.warning('Upload a dataset first!')
            return
        
        clm_correct_nm_col1, clm_correct_nm_col2 = st.columns(2, border=True)
                        
        with clm_correct_nm_col1:
                            st.subheader('Column Name', text_alignment = 'center')
                            correct_type = st.selectbox(
                                label='Select Correctness Type',
                                options=sorted(["Strip", "Capitalize", "Capitalize + Strip", "Left Strip", "Right Strip"]),
                                index=0,
                                key = 'col_name_correctness_type'
                            )
                            if correct_type in [
                                "Strip",
                                "Left Strip",
                                "Right Strip",
                                "Capitalize + Strip",
                            ]:
                                st.text_input(
                                    label='Enter character to strip',
                                    value=' ',
                                    key='col_name_strip_char' 
                                )      
                            st.button(
                                label = "Apply Transformation", 
                                on_click=apply_column_name_transformation
                                )
        
                            if "column_name_transformation_msg" in st.session_state:
                                st.success(st.session_state.pop('column_name_transformation_msg'))
                            if 'apply_column_name_transformation_exception' in st.session_state:
                                st.error(st.session_state.pop('apply_column_name_transformation_exception'))
        
        with clm_correct_nm_col2:
                            st.subheader('Column Values', text_alignment = 'center')
                            columns = data.columns.tolist()
                            if len(columns) == 0:
                                st.warning('No Columns found!')
                            else:
                                clmn = st.selectbox(
                                    label = 'Select Column',
                                    options = sorted(columns),
                                    key = 'column_values_correctness'
                                )
                                correct_type = st.selectbox(
                                    label='Select Correctness Type',
                                    options=sorted(["Strip", "Capitalize", "Capitalize + Strip", "Left Strip", "Right Strip", 'Make all values equal', 'Change specific value(s)']),
                                    index=0,
                                    key = 'col_values_correctness_type'
                                )
                                if correct_type in [
                                    "Strip",
                                    "Left Strip",
                                    "Right Strip",
                                    "Capitalize + Strip",
                                ]:
                                    st.text_input(
                                        label='Enter character to strip',
                                        value=' ',
                                        key='col_values_strip_char' 
                                    ) 
                                elif correct_type == 'Change specific value(s)':
                                    old_val = st.text_input(
                                        label = 'Enter old value',
                                        key = 'old_value'
                                    )
                                    new_val = st.text_input(
                                        label = 'Enter new value',
                                        key = 'new_value'
                                    )
                                elif correct_type == 'Make all values equal':
                                    new_val = st.text_input(
                                        label = 'Enter new value',
                                        key = 'new_value'
                                    )
        
                                st.caption("*Note - Performing wrong operation on wrong datatype will not cause any error. Infact, it will show as operation completed.")
        
                                st.button(
                                    label = 'Apply Changes',
                                    on_click = apply_column_values_transformation
                                )
        
                                if "column_values_transformation_msg" in st.session_state:
                                    st.success(st.session_state.pop('column_values_transformation_msg'))
                                if "apply_column_values_transformation_exception" in st.session_state:
                                    st.success(st.session_state.pop('apply_column_values_transformation_exception'))
    except Exception:
        st.session_state.columns_handling_exception = 'An unknown error occured!'

def dataset_description():
    try:
        st.subheader("Dataset Description", text_alignment = 'center')

        df = st.session_state.get('file_uploaded').copy()

        num_tab, cat_tab = st.tabs(["Numerical Columns", "Categorical Columns"])

        with num_tab:
            num_df = df.select_dtypes(include = 'number')

            if num_df.empty:
                st.warning('No Columns found!')
            else:
                st.dataframe(num_df.describe())

        with cat_tab:
            cat_df = df.select_dtypes(include = ['object', 'category', 'string'])

            if cat_df.empty:
                st.warning('No Columns found!')
            else:
                st.dataframe(cat_df.describe())
    except Exception:
        st.error("An error occured while displaying description of the data!")

def data_values_handling(cat_col_with_null_values, num_col_with_null_values, dt_col_with_null_values, bool_col_with_null_values):
    try:
        st.subheader("Data Values Handling", text_alignment = 'center')

        missing, duplicate, outliers = st.columns(3, border = True)

        df = st.session_state.get('file_uploaded').copy()

        if not isinstance(df, pd.DataFrame):
            st.warning("Upload a dataset first!")
            return

        with missing:
                        remove, fill = st.tabs(['Remove Null Values', "Fill Null Values"])

                        with remove:
                            if df.isnull().sum().sum() > 0:
                                    col_with_null_values = [col for col in df if df[col].isnull().sum() > 0]
                                    if len(col_with_null_values) > 0:
                                        column = st.selectbox(
                                            label = 'Select Column(s):',
                                            options = sorted(col_with_null_values),
                                            index = 0,
                                            key = 'remove_null_column'
                                        )
                                        st.button(
                                            label="Remove Null",
                                            key="remove_null_bt",
                                            on_click=remove_null_values
                                        )

                                        if st.session_state.get("null_removal_message"):
                                            st.success(st.session_state.pop('null_removal_message'))
                                        if st.session_state.get("null_removal_exception"):
                                            st.success(st.session_state.pop('null_removal_exception'))
                                    else:
                                        st.warning('No columns with null values found!')
                            else:
                                st.warning('No null values found')

                        with fill:
                            if st.session_state.get('file_uploaded').isnull().sum().sum() > 0:
                                column_type = st.selectbox(
                                    label = 'Select Column Type',
                                    options = ['Categorical', 'Numerical', 'Datetime', 'Boolean'],
                                    index = 0,
                                    key = 'fill_null_column_type'
                                )
                                if column_type == 'Categorical':
                                        if len(cat_col_with_null_values) > 0:
                                            column = st.selectbox(
                                                label = ' Select column',
                                                options = sorted(cat_col_with_null_values),
                                                index = 0,
                                                key = 'fill_null_column_cat'
                                            )

                                            if column != 'All':
                                                st.write(f"Null Values Count: {df[st.session_state.get('fill_null_column_cat')].isnull().sum()}")

                                            null_fill_mode = st.selectbox(
                                                label = 'Select mode',
                                                options = sorted(['Mode', 'Forward Fill', 'Backward Fill']),
                                                index = 0,
                                                key = 'fill_null_mode_cat'
                                            )
                                            st.button(
                                                label = 'Fill Null Values',
                                                key = 'fill_null_bt',
                                                on_click = fill_null_values
                                            )
                                        else:
                                            st.warning('No Categorical Column containing Null Values found!')
                                        if st.session_state.get('null_fill_message'):
                                                st.success(st.session_state.pop('null_fill_message'))
                                        if 'null_filling_exception' in st.session_state:
                                            st.error(st.session_state.pop('null_filling_exception'))

                                elif column_type == 'Numerical':
                                        if len(num_col_with_null_values) > 0:
                                            column = st.selectbox(
                                                label = ' Select column',
                                                options = sorted(num_col_with_null_values),
                                                index = 0,
                                                key = 'fill_null_column_num'
                                            )
                                            if column != 'All':
                                                st.write(f"Null Values Count: {df[st.session_state.get('fill_null_column_num')].isnull().sum()}")
                                            null_fill_mode = st.selectbox(
                                                label = 'Select mode',
                                                options = sorted(['Mean', 'Median']),
                                                index = 0,
                                                key = 'fill_null_mode_num'
                                            )
                                            st.button(
                                                label = 'Fill Null Values',
                                                key = 'fill_null_bt',
                                                on_click = fill_null_values
                                            )
                                        else:
                                            st.warning('No Numerical Column containing Null Values found!')
                                        if st.session_state.get('null_fill_message'):
                                            st.success(st.session_state.pop('null_fill_message'))
                                        if 'null_filling_exception' in st.session_state:
                                            st.error(st.session_state.pop('null_filling_exception'))

                                elif column_type == 'Datetime':
                                        if len(dt_col_with_null_values) > 0:
                                            column = st.selectbox(
                                                label = 'Select Column',
                                                options = sorted(dt_col_with_null_values),
                                                index = 0,
                                                key = 'fill_null_column_dt'
                                            )
                                            if column != 'All':
                                                st.write(f"Null Values Count: {df[st.session_state.get('fill_null_column_dt')].isnull().sum()}")
                                            mode = st.selectbox(
                                                label = 'Select mode',
                                                options = sorted(['Forward Fill', 'Backward Fill', 'Mode']),
                                                index = 0,
                                                key = 'fill_null_mode_dt'
                                            )
                                            st.button(
                                                label = 'Fill Null Values',
                                                key = 'fill_null_bt',
                                                on_click = fill_null_values
                                            )
                                        else:
                                            st.warning('No Datetime Column containing Null Values found!')
                                        if st.session_state.get('null_fill_message'):
                                            st.success(st.session_state.pop('null_fill_message'))
                                        if 'null_filling_exception' in st.session_state:
                                            st.error(st.session_state.pop('null_filling_exception'))
                                elif column_type == 'Boolean':
                                        if len(bool_col_with_null_values) > 0:
                                            column = st.selectbox(
                                                label = 'Select Column',
                                                options = sorted(bool_col_with_null_values),
                                                index = 0,
                                                key = 'fill_null_column_bool'
                                            )
                                            if column != 'All':
                                                st.write(f"Null Values Count: {df[st.session_state.get('fill_null_column_bool')].isnull().sum()}")
                                            null_fill_mode = st.selectbox(
                                                label = 'Select mode',
                                                options = sorted(['Forward Fill', 'Backward Fill', 'Mode']),
                                                index = 0,
                                                key = 'fill_null_mode_bool'
                                            )
                                            st.button(
                                                label = 'Fill Null Values',
                                                key = 'fill_null_bt',
                                                on_click = fill_null_values
                                            )
                                        else:
                                            st.warning('No Boolean Column containing Null Values found!')
                                        if st.session_state.get('null_fill_message'):
                                                st.success(st.session_state.pop('null_fill_message'))
                                        if 'null_filling_exception' in st.session_state:
                                            st.error(st.session_state.pop('null_filling_exception'))
                            else:
                                st.warning('No null values found')

        with duplicate:
                        st.subheader('Duplicates Handling', text_alignment = 'center')

                        if (df.duplicated().sum() > 0):
                            if len(df.columns) != 0:
                                dup_rem_mode = st.selectbox(
                                    label = 'Select duplicates removal type: ',
                                    options = sorted(['Keep First', 'Keep Last', 'Remove All']),
                                    index = 0,
                                    key = 'dup_rem_type'
                                    )

                                st.button(
                                    label = 'Apply Changes',
                                    key = 'rem_dup_bt',
                                    on_click = duplicates_removal
                                    )

                                if st.session_state.get('dup_rem_message'):
                                    st.success(st.session_state.dup_rem_message)
                                    st.session_state.dup_rem_message = None
                                if 'duplicates_removal_exception' in st.session_state:
                                    st.error(st.session_state.pop('duplicates_removal_exception'))
                            else:
                                st.warning('No Columns found!')
                        else:
                            st.warning('No null values found')

                
        with outliers:
                        try:
                            st.subheader('Outliers Handling', text_alignment='center')
                            df = st.session_state.file_uploaded
                            num_col = df.select_dtypes(include='number').columns.tolist()
                            if len(num_col) == 0:
                                st.warning('No Numerical Columns found!')
                            else:
                                column_not_null_removed = [col for col in num_col if df[col].isnull().any()]
                                if column_not_null_removed:
                                    st.warning(
                                        'Please handle missing values in all numerical columns '
                                        'before proceeding with outlier detection.'
                                    )
                                    with st.expander('View Columns with Missing Values'):
                                        for col in column_not_null_removed:
                                            st.write(f'{col}: {df[col].isnull().sum()} missing values')
                                    st.session_state.outlier_detection_done = False
                                    st.session_state.outlier_masks = {}
                                    st.session_state.outlier_limits = {}
                                    st.session_state.outlier_data_snapshot = None
                                else:
                                    st.selectbox(
                                        label='Select Outlier Detection Method',
                                        options=['IQR', 'Z-Score'],
                                        key='outlier_detection_method'
                                    )
                                    if st.button('Detect Outliers', key='outliers_detection_bt'):
                                        outlier_masks = {}
                                        outlier_limits = {}
                                        for col in num_col:
                                            if st.session_state.outlier_detection_method == 'IQR':
                                                q1 = df[col].quantile(0.25)
                                                q3 = df[col].quantile(0.75)
                                                iqr = q3 - q1
                                                lower_limit = q1 - 1.5 * iqr
                                                upper_limit = q3 + 1.5 * iqr
                                            else:
                                                mean_value = df[col].mean()
                                                std_value = df[col].std(ddof=0)
                                                if std_value == 0:
                                                    continue
                                                lower_limit = mean_value - 3 * std_value
                                                upper_limit = mean_value + 3 * std_value
                                            outlier_mask = (
                                                (df[col] < lower_limit) |
                                                (df[col] > upper_limit)
                                            )
                                            if outlier_mask.any():
                                                outlier_masks[col] = outlier_mask
                                                outlier_limits[col] = lower_limit, upper_limit
                                        st.session_state.outlier_masks = outlier_masks
                                        st.session_state.outlier_limits = outlier_limits
                                        st.session_state.outlier_detection_done = True
                                        st.session_state.outlier_data_snapshot = df.copy()
                                        st.session_state.outlier_detection_method_used = st.session_state.outlier_detection_method
                                    if st.session_state.get('outlier_detection_done', False):
                                        snapshot = st.session_state.get('outlier_data_snapshot')
                                        if (
                                            snapshot is None
                                            or not snapshot.equals(df)
                                            or st.session_state.get(
                                                'outlier_detection_method_used'
                                            ) != st.session_state.outlier_detection_method
                                        ):
                                            st.session_state.outlier_detection_done = False
                                            st.session_state.outlier_masks = {}
                                            st.session_state.outlier_limits = {}
                                            st.session_state.outlier_data_snapshot = None
                                            st.info(
                                                'Dataset or detection method changed. '
                                                'Please detect outliers again.'
                                            )
                                    if st.session_state.get('outlier_detection_done', False):
                                        outlier_masks = st.session_state.outlier_masks
                                        if len(outlier_masks) == 0:
                                            st.info(
                                                'No Outliers detected in the Numerical Columns!'
                                            )
                                        else:
                                            st.write('**Detected Outliers**')
                                            for col, mask in outlier_masks.items():
                                                st.write(f'{col}: {int(mask.sum())} outliers')
                                            selected_col = st.selectbox(
                                                label='Select Column',
                                                options=list(outlier_masks.keys()),
                                                key='outlier_handling_column'
                                            )
                                            technique = st.selectbox(
                                                label='Select Method to handle Outliers',
                                                options=[
                                                    'Remove',
                                                    'Replace with Upper and Lower Limits',
                                                    'Replace with a Specific Value'
                                                ],
                                                key='out_rem_method'
                                            )
                                            if technique == 'Replace with a Specific Value':
                                                value_type = st.selectbox(
                                                    label='Select Value Type',
                                                    options=['Mean', 'Median', 'Constant'],
                                                    key='out_filling_val'
                                                )
                                                if value_type == 'Constant':
                                                    constant_value = st.number_input(
                                                        label='Enter Number',
                                                        value=0.0,
                                                        key='out_filling_constant_value'
                                                    )
                                            if st.button(
                                                'Apply Outlier Handling',
                                                key='out_rem_bt'
                                            ):
                                                updated_df = df.copy()
                                                mask = outlier_masks[selected_col]
                                                if technique == 'Remove':
                                                    updated_df = updated_df.loc[~mask].copy()
                                                elif technique == 'Replace with Upper and Lower Limits':
                                                    lower_limit, upper_limit = (
                                                        st.session_state.outlier_limits[selected_col]
                                                    )
                                                    updated_df[selected_col] = (
                                                        updated_df[selected_col].clip(
                                                            lower=lower_limit,
                                                            upper=upper_limit
                                                        )
                                                    )
                                                else:
                                                    values = updated_df[selected_col]
                                                    inliers = values.loc[~mask]
                                                    if value_type == 'Mean':
                                                        replacement_value = inliers.mean()
                                                    elif value_type == 'Median':
                                                        replacement_value = inliers.median()
                                                    else:
                                                        replacement_value = constant_value
                                                    if pd.isna(replacement_value):
                                                        st.error(
                                                            'Cannot calculate the replacement value.'
                                                        )
                                                        st.stop()
                                                    updated_df.loc[
                                                        mask, selected_col
                                                    ] = replacement_value
                                                update_active_dataset(updated_df)
                                                st.session_state.outlier_detection_done = False
                                                st.session_state.outlier_masks = {}
                                                st.session_state.outlier_limits = {}
                                                st.session_state.outlier_data_snapshot = None

                                                st.session_state.outliers_handling_message = 'Outlier handling applied successfully!'
                                                st.success(st.session_state.pop('outliers_handling_message'))
                        except Exception as e:
                            st.error(f"An Unknown error occured: {e}")
    except Exception:
        st.error("An unknown error occured!")

def file_uploader_show():
    try:
        upload_file = st.file_uploader(
            label = "📂 Upload Dataset",
            key = 'uploaded_dataset',
            max_upload_size = 10,
            type = ['csv', 'xlsx'],
            help = 'Uploaed your original dataset file containing the data'
        )
        if upload_file:
            if upload_file.size == 0: #Validation
                st.error("Upload a valid file.")
                return
            
            df = pd.DataFrame({})

            file_ext = upload_file.name.split('.')[-1].lower() ##Determining type of file
            if file_ext == 'csv':
                df = pd.read_csv(upload_file)

            elif file_ext == 'xlsx':
                df = pd.read_excel(upload_file)

            else:
                st.error("Unsupported file format.")
                return
            
            if df.empty:
                st.error("An error occured while reading the file!")
            else:
                if "loaded_file_name" not in st.session_state or (
                    st.session_state.loaded_file_name != upload_file.name
                ):
                    st.session_state.original_file = df.copy()
                    st.session_state.file_uploaded = df.copy()
                    st.session_state.loaded_file_name = upload_file.name

                if st.session_state.file_uploaded.empty:
                    st.warning("Your dataset is empty. Please upload or restore the dataset")
                    return

                df = st.session_state.get('file_uploaded')

                num_col = sorted(df.select_dtypes(
                        include = 'number'
                    ).columns.tolist())
                cat_col = sorted(df.select_dtypes(
                        include = ['object', 'string', 'category']
                    ).columns.tolist())
                bool_col = sorted(df.select_dtypes(
                        include = ['boolean', 'bool']
                    ).columns.tolist())
                dt_col = sorted([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])])

                col_with_null_values = [col for col in df.columns if df[col].isnull().sum() > 0]
                col_with_null_values_greater_than_50_percentage = [col for col in df.columns if (df[col].isnull().sum() / len(df[col])) > 0.5]

                cat_col_with_null_values = [col for col in cat_col if df[col].isnull().sum() > 0]
                num_col_with_null_values = [col for col in num_col if df[col].isnull().sum() > 0]
                dt_col_with_null_values = [col for col in dt_col if df[col].isnull().sum() > 0]
                bool_col_with_null_values = [col for col in bool_col if df[col].isnull().sum() > 0]

                st.success("Dataset uploaded successfully!")

                st.title("Data Quality", text_alignment = 'center')
                st.subheader("Your dataset is ready for analysis", text_alignment = 'center')

                col = st.columns(5, border = True)

                col[0].metric(
                    label = "Rows",
                    value = st.session_state.file_uploaded.shape[0]
                )
                col[1].metric(
                    label = "Columns",
                    value = st.session_state.file_uploaded.shape[1]
                )
                col[2].metric(
                    label = "Empty cells",
                    value = st.session_state.file_uploaded.isnull().sum().sum()
                )
                col[3].metric(
                    label = "Duplicate Rows",
                    value = st.session_state.file_uploaded.duplicated().sum()
                )
                col[4].metric(
                    label = "Approx. memory usage",
                    value = f'{st.session_state.file_uploaded.memory_usage(deep = True).sum() / (1024**2) : .3f} MB'
                )

                with st.expander("View Data Quality Recommendations"):
                    st.write(f"• Dataset has {st.session_state.file_uploaded.shape[0]} rows and {st.session_state.file_uploaded.shape[1]} columns. A total of {st.session_state.file_uploaded.shape[0] * st.session_state.file_uploaded.shape[1]} values.")
                    st.write(f"• {len(st.session_state.file_uploaded.isnull().sum().index) if st.session_state.file_uploaded.isnull().sum().sum() > 0 else 0} columns contain missing values.")
                    st.write(f"• A total of {st.session_state.file_uploaded.isnull().sum().sum()} values.")
                    st.write(f"• {st.session_state.file_uploaded.duplicated().sum()} duplicate rows found")
                    with st.expander("View Per Column Null Values"):
                        for i in st.session_state.file_uploaded.columns:
                            st.markdown(f"• {i} -> {st.session_state.file_uploaded[i].isnull().sum()} ")

                st.subheader("Data Quality Report", text_alignment = 'center')
                dqr_num, dqr_cat, dqr_dt, dqr_bool = st.columns(4, border = True)

                df = st.session_state.file_uploaded

                with dqr_num:
                    st.subheader('Numerical Columns', text_alignment = 'center')
                    columns = num_col
                    if not(columns):
                        st.warning('No Columns found!')
                    else:
                        for col in columns:
                            with st.expander(f'{col}'):
                                st.markdown(f'Column Name: {col}')
                                st.markdown(f'Total Values: {len(df[col])}')
                                st.markdown(f'Minimum Value: {df[col].min()}')
                                st.markdown(f'Maximum Value: {df[col].max()}')
                                st.markdown(f'Missing Values: {df[col].isnull().sum()}')
                                st.markdown(f'Missing Percentage: {(df[col].isnull().sum()) / (len(df[col])) * 100}')
                                st.markdown(f"Duplicate Contribution: {'No need to investigate' if (df[col].nunique(dropna = False) == len(df[col])) else 'Investigate'}")
                                st.markdown(f'Constant Column: {'Constant' if len(df[col].unique()) == 1 else 'Not Constant'}')
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    q1 = df[col].quantile(0.25)
                                    q3 = df[col].quantile(0.75)
                                    IQR = q3 - q1
                                    lower_limit = q1 - 1.5 * IQR
                                    upper_limit = q3 + 1.5 * IQR
                                    outlier_vls_col = df[col][~(df[col] >= lower_limit) | ~(df[col] <= upper_limit)]
                                    st.markdown('Outliers : ' + 'No Outliers' if len(outlier_vls_col) == 0 else 'Outliers : exist')
                with dqr_cat:
                    st.subheader('Categorical Columns', text_alignment = 'center')
                    columns = cat_col
                    if not(columns):
                        st.warning('No Columns found!')
                    else:
                        for col in columns:
                            with st.expander(f'{col}'):
                                st.markdown(f'Column Name: {col}')
                                st.markdown(f'Total Values: {len(df[col])}')
                                st.markdown(f'Missing Values: {df[col].isnull().sum()}')
                                st.markdown(f'Missing Percentage: {(df[col].isnull().sum()) / (len(df[col])) * 100}')
                                st.markdown(f"Duplicate Contribution: {'No need to investigate' if (df[col].nunique(dropna = False) == len(df[col])) else 'Investigate'}")
                                st.markdown(f'Constant Column: {'Constant' if len(df[col].unique()) == 1 else 'Not Constant'}')

                with dqr_dt:
                    st.subheader('Datetime Columns', text_alignment = 'center')
                    columns = dt_col
                    if not(columns):
                        st.warning('No Columns found!')
                    else:
                        for col in columns:
                            with st.expander(f'{col}'):
                                st.markdown(f'Column Name: {col}')
                                st.markdown(f'Total Values: {len(df[col])}')
                                st.markdown(f'Missing Values: {df[col].isnull().sum()}')
                                st.markdown(f'Missing Percentage: {(df[col].isnull().sum()) / (len(df[col])) * 100}')
                                st.markdown(f"Duplicate Contribution: {'No need to investigate' if (df[col].nunique(dropna = False) == len(df[col])) else 'Investigate'}")
                                st.markdown(f'Constant Column: {'Constant' if len(df[col].unique()) == 1 else 'Not Constant'}')

                with dqr_bool:
                    st.subheader('Boolean Columns', text_alignment = 'center')
                    columns = bool_col
                    if not(columns):
                        st.warning('No Columns found!')
                    else:
                        for col in columns:
                            with st.expander(f'{col}'):
                                st.markdown(f'Column Name: {col}')
                                st.markdown(f'Total Values: {len(df[col])}')
                                st.markdown(f'Missing Values: {df[col].isnull().sum()}')
                                st.markdown(f'Missing Percentage: {(df[col].isnull().sum()) / (len(df[col])) * 100}')
                                st.markdown(f"Duplicate Contribution: {'No need to investigate' if (df[col].nunique(dropna = False) == len(df[col])) else 'Investigate'}")
                                st.markdown(f'Constant Column: {'Constant' if len(df[col].unique()) == 1 else 'Not Constant'}')

                col1, col2 = st.columns(2, border = True)
                df = st.session_state.file_uploaded
                with col1:
                    st.subheader('Columns containing Null values', text_alignment = 'center')
                    if len(col_with_null_values) > 0:
                        for col in col_with_null_values:
                            st.markdown(f'• {col}')
                    else:
                        st.warning('No Columns with Null Values found!')
                with col2:
                    st.subheader("Columns to be removed", text_alignment = 'center')

                    if len(col_with_null_values_greater_than_50_percentage) == 0:
                        st.warning('No suggestions found!')
                    else:
                        st.markdown("#### These columns contain null values greater than 50% and can be removed:")
                        for col in col_with_null_values_greater_than_50_percentage:
                            st.markdown(f'• {col}')

                data_preview()

                dtype_col, clm_correct_dtype_col = st.columns(2, border = True)

                dtype_detection_changing(dtype_col, clm_correct_dtype_col)

                data_format_handling()

                columns_handling()
                        
                dataset_description()

                data_values_handling(cat_col_with_null_values, num_col_with_null_values, dt_col_with_null_values, bool_col_with_null_values)

                dw_bt = st.download_button(
                    label = 'Download data as csv',
                    data = st.session_state.file_uploaded.to_csv(index = False).encode('utf-8'),
                    file_name = 'cleaned_data.csv',
                    mime = 'text/csv'
                )
    except Exception as err:
        st.error(f"An Unknown error occured : {err}")