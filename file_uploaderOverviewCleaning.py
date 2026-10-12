import streamlit as st
import pandas as pd

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


            def apply_column_transformation():
                df_copy = st.session_state.file_uploaded.copy()
                c_type = st.session_state.get("col_correctness_type")

                df_copy.columns = df_copy.columns.astype(str)
                
                if c_type in ["Strip", "Left Strip", "Right Strip"]:
                    char = st.session_state.get("col_strip_char", " ")
                    if c_type == "Strip":
                        df_copy.columns = df_copy.columns.str.strip(char)
                    elif c_type == "Left Strip":
                        df_copy.columns = df_copy.columns.str.lstrip(char)
                    elif c_type == "Right Strip":
                        df_copy.columns = df_copy.columns.str.rstrip(char)
                    st.session_state.transformation_msg = "Column names stripped successfully!"
                        
                elif c_type in ["Capitalize", "Capitalize + Strip"]:
                    if c_type == "Capitalize":
                        df_copy.columns = df_copy.columns.str.capitalize()
                        st.session_state.transformation_msg = "Column names capitalized successfully!"
                    elif c_type == "Capitalize + Strip":
                        df_copy.columns = df_copy.columns.str.strip().str.capitalize()
                        st.session_state.transformation_msg = "Column names capitalized and stripped successfully!"
                        
                st.session_state.file_uploaded = df_copy

            clm_correct_nm_col1, clm_correct_nm_col2 = st.columns([0.80, 0.20], border=True)

            with clm_correct_nm_col1:

                correct_type = st.selectbox(
                    label='Select Correctness Type',
                    options=sorted(["Strip", "Capitalize", "Capitalize + Strip", "Left Strip", "Right Strip"]),
                    index=0,
                    key = 'col_correctness_type'
                )
                
            with clm_correct_nm_col2:
                if correct_type in ["Strip", "Left Strip", "Right Strip"]:
                    st.text_input(
                        label='Enter character to strip',
                        value=' ',
                        key='col_strip_char' 
                    )
                
                st.button("Apply Transformation", on_click=apply_column_transformation)
            if "transformation_msg" in st.session_state:
                st.success(st.session_state.transformation_msg)
                del st.session_state.transformation_msg 

            st.subheader("Dataset Preview", text_alignment = 'center')
            st.dataframe(st.session_state.file_uploaded.head())

            st.subheader("Missing Values Handling", text_alignment = 'center')

            missing, duplicate = st.columns(2, border = True)

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
                            options = ['Categorical', 'Numerical'],
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
