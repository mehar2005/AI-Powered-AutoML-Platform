import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = "Not Uploaded"
if 'null_not_removed' not in st.session_state:
    st.session_state.null_not_removed = False

def remove_null_values():
    column = st.session_state.remove_null_column
    df = st.session_state.file_uploaded

    if column == 'All':
        st.session_state.file_uploaded = df.dropna()
    else:
        st.session_state.file_uploaded = df.dropna(subset=[column])

    st.session_state.null_removal_message = "Null values removed successfully!"

def fill_null_values():
    column_type = st.session_state.fill_null_column_type
    column = st.session_state.fill_null_column
    mode = st.session_state.fill_null_mode

    df = st.session_state.file_uploaded

    if column_type == 'Categorical':
        if column == 'All':
            columns = df.select_dtypes(
                include = ['category', 'object', 'string']
            ).columns.tolist()
        else:
            columns = [column]

        for col in columns:
            if mode == 'Mode':
                mode_values = df[col].mode()
                if not mode_values.empty:
                    df[col] = df[col].fillna(mode_values.iloc[0])
            elif mode == 'Forward Fill':
                df[col] = df[col].ffill()
            elif mode == 'Backward Fill':
                df[col] = df[col].bfill()

    elif column_type == 'Numerical':
        if column == 'All':
            columns = df.select_dtypes(
                include = 'number'
            ).columns.tolist()
        else:
            columns = [column]
        for col in columns:
            if mode == 'Mean':
                mean_value = df[col].mean()
                if pd.notna(mean_value):
                    df[col] = df[col].fillna(mean_value)
            elif mode == 'Median':
                median_value = df[col].median()
                if pd.notna(median_value):
                    df[col] = df[col].fillna(median_value)

    elif column_type == 'Datetime':
        if column == 'All':
            cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        else:
            cols = [column]

        for col in cols:
            if mode == 'Forward Fill':
                df[col] = df[col].ffill()
            elif mode == 'Backward Fill':
                df[col] = df[col].bfill()
            elif mode == 'Current Date':
                df[col] = df[col].fillna(pd.Timestamp('today'))
            elif mode == 'Mode':
                most_frequent_date = df[col].mode()
                if not most_frequent_date.empty:
                    df[col] = df[col].fillna(most_frequent_date.iloc[0])
            elif mode == 'Specific Date':
                dt = st.date_input(
                    label = 'Select date',
                    value = 'today',
                    key = 'fill_null_dt_date'
                )


    st.session_state.file_uploaded = df
    st.session_state.null_fill_message = 'Null values filled successfully!'

def duplicates_removal():
    dup_rem_type = st.session_state.dup_rem_type
    df = st.session_state.file_uploaded
    dup_rem_column = st.session_state.dup_rem_column

    subset = None if dup_rem_column == 'All' else [dup_rem_column]

    if dup_rem_type == 'Keep First':
        df = df.drop_duplicates(subset = subset, keep = 'first')
    if dup_rem_type == 'Keep Last':
        df = df.drop_duplicates(subset = subset, keep = 'last')
    if dup_rem_type == 'Remove All':
        df = df.drop_duplicates(subset = subset, keep = False)

    st.session_state.file_uploaded = df

    st.session_state.dup_rem_message = 'Duplicates Removed Successfully!'

def outlier_detecting():
    df = st.session_state.file_uploaded
    mode = st.session_state.outlier_detecting_mode
    column = st.session_state.outlier_handling_column

    if mode == 'Z-Score':
        z_score = zscore(df[column])

        outliers = df[column][np.abs(z_score) > 3]

        st.session_state[f'{column}_outliers'] = len(outliers)

    elif mode == 'IQR':
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        IQR = q3 - q1
        lower_limit = q1 - 1.5 * IQR
        upper_limit = q3 + 1.5 * IQR

        outliers = df[column][~(df[column] >= lower_limit) | ~(df[column] <= upper_limit)]

        st.session_state[f'{column}_outliers'] = len(outliers)

    st.session_state.outlier_detection_done = True

def apply_column_name_transformation():
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
                        
    st.session_state.file_uploaded = df_copy



def apply_operation(x, op, chars=" "):
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

def apply_column_values_transformation():
    df_copy = st.session_state.file_uploaded.copy()

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
    ]

    if c_type not in valid_operations:
        st.session_state.column_values_transformation_msg = (
            "Invalid transformation selected."
        )
        return

    df_copy[column] = df_copy[column].apply(
        lambda x: apply_operation(x, c_type, chars)
    )

    st.session_state.file_uploaded = df_copy

    messages = {
        "Strip": "Column values stripped successfully!",
        "Left Strip": "Leading characters removed successfully!",
        "Right Strip": "Trailing characters removed successfully!",
        "Capitalize": "Column values capitalized successfully!",
        "Capitalize + Strip": (
            "Column values capitalized and stripped successfully!"
        ),
    }

    st.session_state.column_values_transformation_msg = messages[c_type]


def apply_data_format_transformation(): #AI, need to understand this!
    df_copy = st.session_state.file_uploaded.copy()

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

    try:
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

            if operation == "Remove Commas/Currency Symbols":
                values = values.str.replace(
                    r"[₹$€£,\s]", "", regex=True
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

        st.session_state.file_uploaded = df_copy

        st.session_state.data_format_transformation_msg = (
            f"Data format corrected successfully! "
            f"{invalid_values} invalid values could not be converted."
        )

    except (ValueError, TypeError, KeyError) as e:
        st.session_state.data_format_transformation_msg = (
            f"Unable to apply transformation: {e}"
        )



def file_uploader_show():
    st.title("Dataset Upload", text_alignment = 'center')
    upload_file = st.file_uploader(
        label = "📂 Upload Dataset",
        key = 'uploaded_dataset',
        max_upload_size = 10,
        type = ['csv', 'xlsx'], #Have to fix this later.
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
                value = f'{st.session_state.file_uploaded.memory_usage(deep = True).sum() / (1024**2) : .2f} MB'
            )

            with st.expander("View Data Quality Recommendations"):
                st.write(f"• Dataset has {st.session_state.file_uploaded.shape[0]} rows and {st.session_state.file_uploaded.shape[1]} columns. A total of {st.session_state.file_uploaded.shape[0] * st.session_state.file_uploaded.shape[1]} values.")
                st.write(f"• {len(st.session_state.file_uploaded.isnull().sum().index) if st.session_state.file_uploaded.isnull().sum().sum() > 0 else 0} columns contain missing values.")
                st.write(f"• A total of {st.session_state.file_uploaded.isnull().sum().sum()} values.")
                st.write(f"• {st.session_state.file_uploaded.duplicated().sum()} duplicate rows found")

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
                    value = 5,
                    min_value = 1,
                    max_value = len(st.session_state.file_uploaded),
                    step = 1,
                    help = 'Enter number of rows to show in the dataset preview'
                )

                if preview_type == "First":
                    if preview_number:
                        with col2:
                            st.dataframe(st.session_state.file_uploaded.head(preview_number), hide_index = True)
                if preview_type == "Last":
                    if preview_number:
                        with col2:
                            st.dataframe(st.session_state.file_uploaded.tail(preview_number), hide_index = True)
                if preview_type == "Random":
                    if preview_number:
                        with col2:
                            st.dataframe(st.session_state.file_uploaded.sample(n = preview_number), hide_index = True)

            dtype_col, clm_correct_dtype_col = st.columns(2, border = True)

            with dtype_col:
                st.subheader("Automatically Detected Data Types", text_alignment = 'center')

                num_col = []
                cat_col = []
                bool_col = []
                dt_col = []
                text_col = []
                id_col = []

                df = st.session_state.file_uploaded.copy()

                for col in df.columns:
                    series = df[col]
                    if pd.api.types.is_bool_dtype(series):
                        bool_col.append(col)
                    elif pd.api.types.is_numeric_dtype(series):
                        unique_ratio = series.nunique() / len(series)
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

                types = [
                    ('Categorical', cat_col),
                    ('Numerical', num_col),
                    ('Datetime', dt_col),
                    ('Boolean', bool_col),
                    ('Text', text_col),
                    ('Identifier', id_col)
                ]

                col = st.columns(3, border = True)

                for i, (datatype, columns) in enumerate(types):
                    with col[i % 3]:
                        st.metric(
                            label = f'{datatype} columns',
                            value = len(columns)
                        )
                        with st.expander("View columns"):
                            for column in columns:
                                st.markdown(f'• `{column}`')

            with clm_correct_dtype_col:
                st.subheader("Manual Datatype Conversion", text_alignment = 'center')

                col_nm = st.selectbox(
                    label = 'Select column',
                    options = df.columns,
                    key = 'dtype_column'
                )
                st.write(f'Detected Datatype: {next(datatype for datatype, columns in types if col_nm in columns)}')

                st.write(f'Pandas Datatype: {df[col_nm].dtype}')

                chng_dtype = st.selectbox(
                    label = 'Change Datatype to:',
                    options = sorted(['Integer', 'Float', 'Boolean', 'Datetime', 'Category', 'String']),
                    key = 'change_datatype'
                )

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
                        converted = pd.to_datetime(df[col_nm], errors = 'coerce')
                    elif chng_dtype == 'Boolean':
                        converted = df[col_nm].astype('boolean')

                    st.session_state.file_uploaded[col_nm] = converted
                    st.rerun()

            st.subheader(
                            "Data Format Handling",
                            text_alignment="center"
                        )
            with st.container(border=True):
                            df = st.session_state.file_uploaded
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
                                    "Remove Commas/Currency Symbols",
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

            st.subheader('Columns Handling', text_alignment = 'center')

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
                st.success(st.session_state.column_name_transformation_msg)
                del st.session_state.column_name_transformation_msg 

            with clm_correct_nm_col2:
                st.subheader('Column Values', text_alignment = 'center')
                columns = st.session_state.file_uploaded.columns.tolist()

                clmn = st.selectbox(
                    label = 'Select Column',
                    options = sorted(columns),
                    key = 'column_values_correctness'
                )
                correct_type = st.selectbox(
                    label='Select Correctness Type',
                    options=sorted(["Strip", "Capitalize", "Capitalize + Strip", "Left Strip", "Right Strip"]),
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

                st.button(
                    label = 'Apply Chages',
                    on_click = apply_column_values_transformation
                )

                if "column_values_transformation_msg" in st.session_state:
                    st.success(st.session_state.column_values_transformation_msg)
                    del st.session_state.column_values_transformation_msg 
                      
            st.subheader("Dataset Description", text_alignment = 'center')

            num_tab, cat_tab = st.tabs(["Num_tab", "Cat_tab"])

            with num_tab:
                df = st.session_state.file_uploaded

                num_df = df.select_dtypes(include = 'number')

                st.dataframe(num_df.describe())

            with cat_tab:
                df = st.session_state.file_uploaded

                cat_df = df.select_dtypes(include = 'object')

                st.dataframe(cat_df.describe())


            st.subheader("Data Values Handling", text_alignment = 'center')

            missing, duplicate, outliers = st.columns(3, border = True)

            with missing:
                remove, fill = st.tabs(['Remove Null Values', "Fill Null Values"])

                with remove:
                    if st.session_state.file_uploaded.isnull().sum().sum() > 0:
                        column = st.selectbox(
                            label = 'Select Column(s):',
                            options = ['All'] + list(st.session_state.file_uploaded.columns),
                            index = 0,
                            key = 'remove_null_column'
                        )
                        st.button(
                            label="Remove Null",
                            key="remove_null_bt",
                            on_click=remove_null_values
                        )

                        if st.session_state.get("null_removal_message"):
                            st.success(st.session_state.null_removal_message)
                            st.session_state.null_removal_message = None
                    else:
                        st.info('No null values found')

                with fill:
                    if st.session_state.file_uploaded.isnull().sum().sum() > 0:
                        column_type = st.selectbox(
                            label = 'Select Column Type',
                            options = ['Categorical', 'Numerical', 'Datetime', 'Boolean'],
                            index = 0,
                            key = 'fill_null_column_type'
                        )
                        if column_type == 'Categorical':
                            column = st.selectbox(
                                label = ' Select column',
                                options = ['All'] + sorted(list(
                                    st.session_state.file_uploaded.select_dtypes(
                                        include = ['category', 'object', 'string']
                                        ).columns
                                    )),
                                index = 0,
                                key = 'fill_null_column'
                            )

                            if column != 'All':
                                st.write(f'Null Values Count: {st.session_state.file_uploaded[column].isnull().sum()}')

                            null_fill_mode = st.selectbox(
                                label = 'Select mode',
                                options = sorted(['Mode', 'Forward Fill', 'Backward Fill']),
                                index = 0,
                                key = 'fill_null_mode'
                            )
                            st.button(
                                label = 'Fill Null Values',
                                key = 'fill_null_bt',
                                on_click = fill_null_values
                            )

                            if st.session_state.get('null_fill_message'):
                                st.success(st.session_state.null_fill_message)
                                st.session_state.null_fill_message = None

                        elif column_type == 'Numerical':
                            column = st.selectbox(
                                label = ' Select column',
                                options = ['All'] + sorted(list(
                                    st.session_state.file_uploaded.select_dtypes(
                                        include = ['number']
                                        ).columns
                                    )),
                                index = 0,
                                key = 'fill_null_column'
                            )
                            null_fill_mode = st.selectbox(
                                label = 'Select mode',
                                options = sorted(['Mean', 'Median']),
                                index = 0,
                                key = 'fill_null_mode'
                            )
                            st.button(
                                label = 'Fill Null Values',
                                key = 'fill_null_bt',
                                on_click = fill_null_values
                            )

                            if st.session_state.get('null_fill_message'):
                                st.success(st.session_state.null_fill_message)
                                st.session_state.null_fill_message = None

                        elif column_type == 'Datetime':
                            datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                            column = st.selectbox(
                                label = 'Select Column',
                                options = ['All'] + sorted(datetime_cols),
                                key = 'fill_null_column'
                            )
                            null_fill_mode = st.selectbox(
                                label = 'Select mode',
                                options = sorted(['Forward Fill', 'Backward Fill', 'Mode', 'Current Date', 'Specific Date']),
                                key = 'fill_null_mode'
                            )
                            st.button(
                                label = 'Fill Null Values',
                                key = 'fill_null_bt',
                                on_click = fill_null_values
                            )

                            if st.session_state.get('null_fill_message'):
                                st.success(st.session_state.null_fill_message)
                                st.session_state.null_fill_message = None
                    else:
                        st.info('No null values found')

            with duplicate:
                st.subheader('Duplicates Handling', text_alignment = 'center')

                column = st.selectbox(
                    label = 'Select Column',
                    options = ['All'] + sorted(st.session_state.file_uploaded.columns.tolist()),
                    key = 'dup_rem_column'
                    )

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

            with outliers:
                st.subheader('Outliers Handling', text_alignment = 'center')
                columns = st.session_state.file_uploaded.select_dtypes(
                    include = ['number']
                ).columns.tolist()
                df = st.session_state.file_uploaded
                with_outlier_columns = []
                with_outlier_column_not_null_removed = []

                for col in columns:
                    if df[col].isnull().sum() > 0:
                        with_outlier_column_not_null_removed.append(col)
                    else:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        IQR = q3 - q1
                        lower_limit = q1 - 1.5 * IQR
                        upper_limit = q3 + 1.5 * IQR
                        with_outlier_vls_iqr = df[col][(df[col] < lower_limit) | (df[col] > upper_limit)]

                        z_score = zscore(df[col])

                        with_outlier_vls_zscore = df[col][np.abs(z_score) > 3]

                        if len(with_outlier_vls_iqr) > 0:
                            with_outlier_columns.append(col)
                        elif len(with_outlier_vls_zscore) > 0:
                            with_outlier_columns.append(col)

                if len(with_outlier_columns) == 0:
                    st.warning('No Non-Null Columns found!')
                else:
                    clmn = st.selectbox(
                        label = 'Select column: ',
                        options = with_outlier_columns,
                        key = 'outlier_handling_column'
                    )
                    outlier_detecting_mode = st.selectbox(
                        label = 'Outlier detecting mode: ',
                        options = sorted(['Z-Score', 'IQR']),
                        key = 'outlier_detecting_mode'
                    )
                    bt = st.button(
                        label = 'Detect Outliers',
                        key = 'outlier_detecting_bt',
                        on_click = outlier_detecting
                    )

                    if st.session_state.get('outlier_detection_done', False):
                        count = st.session_state.get(f'{clmn}_outliers')

                        if count is not None:
                            st.write(
                                f'Outliers detected through {outlier_detecting_mode}: {count}'
                            )

                        out_rem_technique = st.selectbox(
                            label = 'Select Outlier Removal Technique: ',
                            options = sorted(['Remove', 'Replace with Lower and Upper Limits', 'Replace with a specific value']),
                            key = 'outlier_removal_technique'
                        )

                        if st.session_state.outlier_removal_technique == 'Replace with a specific value':
                            vl_mode = st.selectbox(
                                label = 'Select Value Type',
                                options = sorted(['Mean', 'Mode', 'Constant']),
                                key = 'out_filling_vl_mode'
                            )
                            if st.session_state.out_filling_vl_mode == 'Constant':
                                value = st.number_input(
                                    label = 'Enter value: ',
                                    step = 1,
                                    key = 'out_filling_value'
                                )

                        # st.button(
                        #     label = ''
                        # )

                if len(with_outlier_column_not_null_removed) > 0:
                    st.warning('These columns require handling missing values first')
                    with st.expander('View Columns'):
                        for col in with_outlier_column_not_null_removed:
                            st.write(f'• {col}')

            st.subheader("Data Quality Report", text_alignment = 'center')
            dqr_num, dqr_cat, dqr_dt, dqr_bool = st.columns(4, border = True)

            df = st.session_state.file_uploaded

            with dqr_num:
                st.subheader('Numerical Columns', text_alignment = 'center')
                columns = st.session_state.file_uploaded.select_dtypes(
                    include = ['number']
                )
                if columns.empty:
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
                columns = st.session_state.file_uploaded.select_dtypes(
                    include = ['object', 'category', 'string']
                )
                if columns.empty:
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
                            if pd.api.types.is_numeric_dtype(df[col]):
                                q1 = df[col].quantile(0.25)
                                q3 = df[col].quantile(0.75)
                                IQR = q3 - q1
                                lower_limit = q1 - 1.5 * IQR
                                upper_limit = q3 + 1.5 * IQR
                                outlier_vls_col = df[col][~(df[col] >= lower_limit) | ~(df[col] <= upper_limit)]
                                st.markdown('Outliers : ' + 'No Outliers' if len(outlier_vls_col) == 0 else 'Outliers : exist')
            with dqr_dt:
                st.subheader('Datetime Columns', text_alignment = 'center')
                columns = st.session_state.file_uploaded.select_dtypes(
                    include = ['datetime']
                )
                if columns.empty:
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
                            if pd.api.types.is_numeric_dtype(df[col]):
                                q1 = df[col].quantile(0.25)
                                q3 = df[col].quantile(0.75)
                                IQR = q3 - q1
                                lower_limit = q1 - 1.5 * IQR
                                upper_limit = q3 + 1.5 * IQR
                                outlier_vls_col = df[col][~(df[col] >= lower_limit) | ~(df[col] <= upper_limit)]
                                st.markdown('Outliers : ' + 'No Outliers' if len(outlier_vls_col) == 0 else 'Outliers : exist')
            with dqr_bool:
                st.subheader('Boolean Columns', text_alignment = 'center')
                columns = st.session_state.file_uploaded.select_dtypes(
                    include = ['bool']
                )
                if columns.empty:
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
                            if pd.api.types.is_numeric_dtype(df[col]):
                                q1 = df[col].quantile(0.25)
                                q3 = df[col].quantile(0.75)
                                IQR = q3 - q1
                                lower_limit = q1 - 1.5 * IQR
                                upper_limit = q3 + 1.5 * IQR
                                outlier_vls_col = df[col][~(df[col] >= lower_limit) | ~(df[col] <= upper_limit)]
                                st.markdown('Outliers : ' + 'No Outliers' if len(outlier_vls_col) == 0 else 'Outliers : exist')

            dw_bt = st.download_button(
                label = 'Download data as csv',
                data = st.session_state.file_uploaded.to_csv(index = False).encode('utf-8'),
                file_name = 'cleaned_data.csv',
                mime = 'text/csv'
            )

