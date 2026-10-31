import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

DT_OPERATIONS = {
    "Year": lambda s: s.dt.year,
    "Month": lambda s: s.dt.month,
    "Day": lambda s: s.dt.day,
    "Hour": lambda s: s.dt.hour,
    "Minute": lambda s: s.dt.minute,
    "Second": lambda s: s.dt.second,
    "Day Name": lambda s: s.dt.day_name(),
    "Month Name": lambda s: s.dt.month_name(),
    "Quarter": lambda s: s.dt.quarter,
    "Date Only": lambda s: s.dt.date,
    "Time Only": lambda s: s.dt.time,
}

def _get_numeric_columns(data):
    try:
        return sorted(
            data.select_dtypes(include=["number"]).columns.tolist(),
            key=str,
        )
    except Exception as e:
        st.session_state.get_num_col_exception = str(e)

def _get_categoric_columns(data):
    try:
        return sorted(
            data.select_dtypes(include=["object", "string", "category"]).columns.tolist(),
            key=str,
        )
    except Exception as e:
        st.session_state.get_cat_col_exception = str(e)

def _get_datetime_columns(data):
    try:
        return sorted(
            [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])],
            key=str,
        )
    except Exception as e:
        st.session_state.get_dt_col_exception = str(e)

def _get_data_or_raise():
        data = st.session_state.get("file_uploaded")
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Upload a valid, non-empty dataset first.")
        return data

def _require_column(data, column, label="selected"):
        if column is None or column not in data.columns:
            raise ValueError(f"The {label} column no longer exists. Please select it again.")

def _clean_new_name(value):
    try:
        return str(value or "").strip()
    except Exception as e:
        st.session_state.clean_new_name_exception = str(e)

def _validate_new_name(data, name, *, allow_existing=False):
        if not name:
            raise ValueError("Enter a valid output column name.")
        if not allow_existing and name in data.columns:
            raise ValueError(
                f"A column named '{name}' already exists. Choose a unique name."
            )

def _parse_categories(raw_value, bins_num):
        raw_value = str(raw_value or "")
        categories = [item.strip() for item in raw_value.split("|")]
        if len(categories) != bins_num or any(not item for item in categories):
            raise ValueError(
                "Enter exactly one non-empty category label for each bin, separated by |."
            )
        if len(set(categories)) != len(categories):
            raise ValueError("Category labels must be unique.")
        return categories

def _parse_boundaries(raw_value, bins_num, series):
    try:
        boundaries = [float(item.strip()) for item in str(raw_value or "").split("|")]
    except (TypeError, ValueError) as exc:
        raise ValueError("Boundaries must be valid numbers separated by |.") from exc
    if len(boundaries) != bins_num + 1:
        raise ValueError("Enter exactly one more boundary than the number of bins.")
    if not np.isfinite(boundaries).all():
        raise ValueError("Boundaries must be finite numbers.")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Boundaries must be strictly increasing.")
    values = series.dropna()
    if not values.empty:
        if boundaries[0] > values.min() or boundaries[-1] < values.max():
            raise ValueError(
                "Custom boundaries must cover the full range of the selected column."
            )
    return boundaries

def _validate_numeric_series(series, column_name):
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column_name}' must be numerical.")
    if pd.api.types.is_bool_dtype(series):
        raise ValueError(f"Column '{column_name}' must be numerical, not boolean.")

def _validate_result(result, valid_inputs):
    result = pd.Series(result, index=valid_inputs.index)
    generated_missing = result.isna() & valid_inputs
    if generated_missing.any():
        raise ValueError(
            "The transformation produced invalid or missing results for valid inputs. "
            "Check the values and operation."
        )
    try:
        numeric_result = result.dropna().astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("The transformation did not produce valid numerical results.") from exc
    if not np.isfinite(numeric_result).all():
        raise ValueError("The transformation produced non-finite values (infinity or NaN).")
    return result

def apply_transformation():
    try:
        data = _get_data_or_raise()
        transformation = st.session_state.get("num_column_mathematical_transformation")
        if transformation == "Combine Two Features":
            col1_name = st.session_state.get("combine_first_col")
            col2_name = st.session_state.get("combine_second_col")
            operator = st.session_state.get("combine_col_arithmetic_operation")
            output_name = _clean_new_name(
                st.session_state.get("combine_col_new_name")
            )
            _require_column(data, col1_name, "first")
            _require_column(data, col2_name, "second")
            _validate_new_name(data, output_name)
            left = data[col1_name]
            right = data[col2_name]
            _validate_numeric_series(left, col1_name)
            _validate_numeric_series(right, col2_name)
            if operator in ("/", "//", "%") and right.eq(0).fillna(False).any():
                raise ValueError(
                    "The second column contains zero values; division, floor division, "
                    "and modulo by zero are not allowed."
                )
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                if operator == "+":
                    result = left + right
                elif operator == "-":
                    result = left - right
                elif operator == "*":
                    result = left * right
                elif operator == "/":
                    result = left / right
                elif operator == "**":
                    result = left ** right
                elif operator == "//":
                    result = left // right
                elif operator == "%":
                    result = left % right
                else:
                    raise ValueError("Select a valid arithmetic operator.")
            valid_inputs = left.notna() & right.notna()
            result = _validate_result(result, valid_inputs)
        else:
            column = st.session_state.get("num_column_maths_op")
            _require_column(data, column, "numerical")
            values = data[column]
            _validate_numeric_series(values, column)
            valid_inputs = values.notna()
            if transformation == "Log":
                if (values.dropna() <= 0).any():
                    raise ValueError(
                        "Log transformation requires all non-missing values to be greater than zero."
                    )
                result = np.log(values)
            elif transformation == "Square Root":
                if (values.dropna() < 0).any():
                    raise ValueError(
                        "Square root transformation requires non-negative values."
                    )
                result = np.sqrt(values)
            elif transformation == "Square":
                result = values ** 2
            elif transformation == "Power":
                try:
                    power = float(st.session_state.get("num_power", 2.0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Power must be a valid number.") from exc
                if not np.isfinite(power):
                    raise ValueError("Power must be a finite number.")
                result = values ** power
            elif transformation == "Add/Subtract/Multiply/Divide a Value":
                operation = st.session_state.get("change_value_operation")
                try:
                    number = float(st.session_state.get("change_value_number", 0.0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Enter a valid numerical constant.") from exc
                if not np.isfinite(number):
                    raise ValueError("The entered number must be finite.")
                if operation == "Addition":
                    result = values + number
                elif operation == "Subtraction":
                    result = values - number
                elif operation == "Multiplication":
                    result = values * number
                elif operation == "Division":
                    if number == 0:
                        raise ValueError("Cannot divide by zero.")
                    result = values / number
                else:
                    raise ValueError("Select a valid arithmetic operation.")
            else:
                raise ValueError("Select a valid mathematical transformation.")
            result = _validate_result(result, valid_inputs)
            create_new = bool(st.session_state.get("if_new_num_col", False))
            if create_new:
                output_name = _clean_new_name(
                    st.session_state.get("if_new_num_col_name")
                )
                _validate_new_name(data, output_name)
            else:
                output_name = column
        updated_data = data.copy()
        updated_data[output_name] = result
        st.session_state["file_uploaded"] = updated_data
        st.session_state["transformation_message"] = "Transformation applied successfully."
        st.session_state.pop("transformation_error", None)
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        ZeroDivisionError,
        FloatingPointError,
    ) as exc:
        st.session_state["transformation_error"] = str(exc)
        st.session_state.pop("transformation_message", None)

def apply_binning():
    try:
        data = _get_data_or_raise()
        column = st.session_state.get("num_column_binning")
        _require_column(data, column, "numerical")
        values = data[column]
        _validate_numeric_series(values, column)
        method = st.session_state.get("bin_method")
        try:
            bins_num = int(st.session_state.get("number_bins", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("The number of bins must be a whole number.") from exc
        if bins_num < 2:
            raise ValueError("The number of bins must be at least 2.")
        output_name = _clean_new_name(st.session_state.get("bin_new_feature_name"))
        _validate_new_name(data, output_name)
        categories = _parse_categories(
            st.session_state.get("bins_categories", ""), bins_num
        )
        if values.dropna().empty:
            raise ValueError("The selected column has no non-missing values to bin.")
        if method == "Equal Width":
            result = pd.cut(
                values, bins=bins_num, labels=categories, include_lowest=True
            )
        elif method == "Equal Frequency":
            result = pd.qcut(values, q=bins_num, duplicates="drop")
            actual_bins = result.cat.categories.size
            if actual_bins != bins_num:
                raise ValueError(
                    "Equal-frequency binning could not create the requested number "
                    "of distinct bins. Reduce the bin count or choose another method."
                )
            result = result.cat.rename_categories(categories)
        elif method == "Custom":
            boundaries = _parse_boundaries(
                st.session_state.get("bins_boundaries", ""), bins_num, values
            )
            result = pd.cut(
                values, bins=boundaries, labels=categories, include_lowest=True
            )
        else:
            raise ValueError("Select a valid binning method.")
        updated_data = data.copy()
        updated_data[output_name] = result
        st.session_state["file_uploaded"] = updated_data
        st.session_state["binning_message"] = "Binning applied successfully."
        st.session_state.pop("binning_error", None)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        st.session_state["binning_error"] = str(exc)
        st.session_state.pop("binning_message", None)

def apply_encoding():
    try:
        data = _get_data_or_raise()
        column = st.session_state.get("cat_column")
        _require_column(data, column, "categorical")
        if column not in _get_categoric_columns(data):
            raise ValueError("The selected column is no longer categorical.")
        technique = st.session_state.get("feature_engineering_cat_encoding_technique")
        updated_data = data.copy()
        series = updated_data[column]
        if technique == "One-Hot Encoding":
            updated_data = pd.get_dummies(
                updated_data, columns=[column], dtype=int, dummy_na=False
            )
        elif technique == "Label Encoding":
            non_missing = series.notna()
            encoder = LabelEncoder()
            encoded = pd.Series(pd.NA, index=series.index, dtype="Int64")
            if non_missing.any():
                encoded.loc[non_missing] = encoder.fit_transform(
                    series.loc[non_missing].astype(str)
                )
            updated_data[column] = encoded
        elif technique == "Ordinal Encoding":
            raw_categories = str(
                st.session_state.get("cat_col_ordinal_encoding_categories") or ""
            ).strip()
            categories = [item.strip() for item in raw_categories.split("|")]
            if (
                not raw_categories
                or any(not item for item in categories)
                or len(set(categories)) != len(categories)
            ):
                raise ValueError(
                    "Enter non-empty, unique ordinal categories separated by |."
                )
            observed = set(series.dropna().astype(str).unique())
            unknown = observed.difference(categories)
            if unknown:
                preview = ", ".join(sorted(unknown)[:5])
                raise ValueError(
                    f"These observed categories are missing from your ordinal order: {preview}"
                )
            encoder = OrdinalEncoder(
                categories=[categories],
                handle_unknown="error",
                encoded_missing_value=np.nan,
            )
            ordinal_values = series.map(
                lambda value: str(value) if pd.notna(value) else np.nan
            )
            non_missing = series.notna()
            encoded = pd.Series(np.nan, index=series.index, dtype=float)
            if non_missing.any():
                encoded.loc[non_missing] = encoder.fit_transform(
                    ordinal_values.loc[non_missing].to_frame()
                ).ravel()
            updated_data[column] = encoded
        else:
            raise ValueError("Select a valid encoding technique.")
        st.session_state["file_uploaded"] = updated_data
        st.session_state["encoding_message"] = "Encoding applied successfully."
        st.session_state.pop("encoding_exception", None)
        st.session_state.pop("oe_categories_warning", None)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        st.session_state["encoding_exception"] = str(exc)
        st.session_state.pop("encoding_message", None)

def del_cat_col():
    try:
        data = _get_data_or_raise()
        column = st.session_state.get("cat_column")
        _require_column(data, column, "categorical")
        updated_data = data.drop(columns=[column]).copy()
        st.session_state["file_uploaded"] = updated_data
        st.session_state["del_message"] = "Column deleted successfully."
        st.session_state.pop("del_error", None)
    except (ValueError, KeyError) as exc:
        st.session_state["del_error"] = str(exc)

def combine_cat_col():
    try:
        data = _get_data_or_raise()
        col1 = st.session_state.get("cat_column1")
        col2 = st.session_state.get("cat_column2")
        _require_column(data, col1, "first categorical")
        _require_column(data, col2, "second categorical")
        output_name = _clean_new_name(
            st.session_state.get("cat_col_combine_new_name")
        )
        _validate_new_name(data, output_name)
        separator = str(st.session_state.get("cat_col_combine_middle_char") or "")
        left = data[col1].astype("string")
        right = data[col2].astype("string")
        combined = (left + separator + right).astype("string")
        updated_data = data.copy()
        updated_data[output_name] = combined
        st.session_state["file_uploaded"] = updated_data
        st.session_state["combine_message"] = "Columns combined successfully."
        st.session_state.pop("combine_error", None)
    except (ValueError, TypeError, KeyError) as exc:
        st.session_state["combine_error"] = str(exc)

def dt_col_op():
    try:
        data = _get_data_or_raise()
        column = st.session_state.get("dt_column")
        operation = st.session_state.get("feature_engineering_dt_encoding_technique")
        _require_column(data, column, "datetime")
        if operation == "Delete":
            updated_data = data.drop(columns=[column]).copy()
            st.session_state["file_uploaded"] = updated_data
            st.session_state["dt_delete_message"] = "Datetime column deleted successfully."
            st.session_state.pop("dt_exception", None)
            st.session_state.pop("col_not_found_error", None)
        elif operation == "Extracting Individual Date Components":
            if not pd.api.types.is_datetime64_any_dtype(data[column]):
                raise ValueError("The selected column is no longer datetime typed.")
            components = st.session_state.get("dt_components_op", [])
            if not components:
                raise ValueError("Select at least one datetime component.")
            updated_data = data.copy()
            output_columns = {}
            for component in components:
                if component not in DT_OPERATIONS:
                    raise ValueError(f"Unsupported datetime component: {component}")
                output_name = f"extracted_{component.lower().replace(' ', '_')}"
                if output_name in updated_data.columns:
                    raise ValueError(
                        f"Column '{output_name}' already exists. Rename or remove it before extracting again."
                    )
                output_columns[output_name] = DT_OPERATIONS[component](data[column])
            for name, result in output_columns.items():
                updated_data[name] = result
            st.session_state["file_uploaded"] = updated_data
            st.session_state["extracted_col_message"] = "Datetime components extracted successfully."
            st.session_state.pop("dt_exception", None)
        else:
            raise ValueError("Select a valid datetime operation.")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        st.session_state["dt_exception"] = str(exc)
        st.session_state.pop("extracted_col_message", None)

def del_num_col():
    try:
        data = _get_data_or_raise()
        column = st.session_state.get("num_column_delete")
        _require_column(data, column, "numerical")
        if column not in _get_numeric_columns(data):
            raise ValueError("The selected column is no longer numerical.")
        updated_data = data.drop(columns=[column]).copy()
        st.session_state.file_uploaded = updated_data
        st.session_state.del_num_col_message = "Numerical column deleted successfully."
    except (ValueError, KeyError) as exc:
        st.session_state["num_col_not_found_error"] = str(exc)

def fe_show():
    data = st.session_state.get("file_uploaded")
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.warning("Upload a file to start feature engineering.")
        return
    numeric_columns = _get_numeric_columns(data)
    categorical_columns = _get_categoric_columns(data)
    datetime_columns = _get_datetime_columns(data)
    st.selectbox(
        label="Select Column Type",
        options=["Categorical", "Datetime", "Numerical"],
        index=0,
        key="column_type",
    )
    if st.session_state.column_type == "Numerical":
        if not numeric_columns:
            st.info("The uploaded file has no numerical columns to transform or bin.")
            return
        operation = st.selectbox(
            label="Select Operation",
            options=["Binning", "Delete", "Mathematical Operation"],
            index = 0,
            key = "feature_engineering_operation_num",
        )
        if operation == "Binning":
            st.selectbox(
                label="Select Column",
                options=numeric_columns,
                index = 0,
                key="num_column_binning",
            )
            st.selectbox(
                label="Select Binning Method",
                options=["Custom", "Equal Frequency", "Equal Width"],
                index = 0,
                key="bin_method",
            )
            st.number_input(
                label="Enter number of bins",
                min_value=2,
                value=2,
                step=1,
                key="number_bins",
            )
            st.text_input(
                label="Enter New Feature Name",
                key="bin_new_feature_name",
            )
            st.text_input(
                label="Enter one bin label per bin (separated by |)",
                key="bins_categories",
            )
            if st.session_state.bin_method == "Custom":
                st.text_input(
                    label=(
                        "Enter strictly increasing numerical boundaries "
                        "(one more than the number of bins, separated by |)"
                    ),
                    key="bins_boundaries",
                )
            st.button(
                label = "Apply Changes", 
                key="binning_bt", 
                on_click=apply_binning
                )
            if "binning_message" in st.session_state:
                st.success(st.session_state.pop("binning_message"))
            if "binning_error" in st.session_state:
                st.error(st.session_state.pop("binning_error"))
        elif operation == "Delete":
            st.selectbox(
                label="Select Column",
                options=numeric_columns,
                index = 0,
                key="num_column_delete",
            )
            st.checkbox(
                label="Are you sure you want to remove this column permanently?",
                key="num_col_del_confirmation",
            )
            if st.session_state.get("num_col_del_confirmation", False):
                st.button(
                    label = "Delete Column", 
                    key="del_num_col_bt", 
                    on_click=del_num_col
                    )
            if "num_col_not_found_error" in st.session_state:
                st.warning(st.session_state.pop("num_col_not_found_error"))
            if "del_num_col_message" in st.session_state:
                st.success(st.session_state.pop("del_num_col_message"))
        elif operation == "Mathematical Operation":
            transformation = st.selectbox(
                label="Select Transformation",
                options=[
                    "Log",
                    "Square Root",
                    "Square",
                    "Power",
                    "Add/Subtract/Multiply/Divide a Value",
                    "Combine Two Features",
                ],
                index = 0,
                key="num_column_mathematical_transformation",
            )
            if transformation != "Combine Two Features":
                st.selectbox(
                    label="Select Column",
                    options=numeric_columns,
                    index = 0,
                    key="num_column_maths_op",
                )
                st.checkbox(
                    label="Create New Column?",
                    key="if_new_num_col",
                )
                if st.session_state.get("if_new_num_col", False):
                    st.text_input(
                        label="Enter New Column Name",
                        key="if_new_num_col_name",
                    )
            if transformation == "Power":
                st.number_input(
                    label="Enter Power",
                    value=2.0,
                    key="num_power",
                )
            elif transformation == "Add/Subtract/Multiply/Divide a Value":
                st.selectbox(
                    label="Select Operation",
                    options=["Addition", "Division", "Multiplication", "Subtraction"],
                    index = 0,
                    key="change_value_operation",
                )
                st.number_input(
                    label="Enter Number",
                    value=0.0,
                    key="change_value_number",
                )
            elif transformation == "Combine Two Features":
                st.selectbox(
                    label="Select First Numerical Feature",
                    options=numeric_columns,
                    index = 0,
                    key="combine_first_col",
                )
                st.selectbox(
                    label="Select Second Numerical Feature",
                    options=numeric_columns,
                    index = 0,
                    key="combine_second_col",
                )
                st.selectbox(
                    label="Select Arithmetic Operator",
                    options=["%", "*", "+", "-", "/", "//", "**"],
                    index = 0,
                    key="combine_col_arithmetic_operation",
                )
                st.text_input(
                    label="Enter New Column Name",
                    key="combine_col_new_name",
                )
            st.button(
                "Apply Transformation",
                key="transformation_bt",
                on_click=apply_transformation,
            )
            if "transformation_message" in st.session_state:
                st.success(st.session_state.pop("transformation_message"))
            if "transformation_error" in st.session_state:
                st.error(st.session_state.pop("transformation_error"))

    elif st.session_state.column_type == "Categorical":
        if not categorical_columns:
            st.info("The uploaded file has no categorical columns to transform.")
            return
        cat_operation = st.selectbox(
            label="Select Operation",
            options=sorted(["Combine", "Delete", "Encoding"]),
            index = 0,
            key="feature_engineering_operation_cat",
        )
        if cat_operation != "Combine":
            st.selectbox(
                label="Select Column",
                options=categorical_columns,
                index = 0,
                key="cat_column",
            )
            if cat_operation == "Encoding":
                technique = st.selectbox(
                    label="Select Encoding Technique",
                    options=sorted(["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"]),
                    index = 0,
                    key="feature_engineering_cat_encoding_technique",
                )
                if technique == "Ordinal Encoding":
                    st.text_input(
                        label="Write categories in increasing order, separated by |",
                        key="cat_col_ordinal_encoding_categories",
                    )
                st.button(
                    f"Apply Technique",
                    key="cat_col_encoding_bt",
                    on_click=apply_encoding,
                )
                if "encoding_message" in st.session_state:
                    st.success(st.session_state.pop("encoding_message"))
                if "oe_categories_warning" in st.session_state:
                    st.warning(st.session_state.pop("oe_categories_warning"))
                if "encoding_exception" in st.session_state:
                    st.error(st.session_state.pop("encoding_exception"))
            elif cat_operation == "Delete":
                st.button("Delete Column", key="cat_col_del", on_click=del_cat_col)
                if "del_message" in st.session_state:
                    st.success(st.session_state.pop("del_message"))
                if "del_error" in st.session_state:
                    st.error(st.session_state.pop("del_error"))
        else:
            st.selectbox(
                label="Select First Column",
                options=categorical_columns,
                index = 0,
                key="cat_column1",
            )
            st.selectbox(
                label="Select Second Column",
                options=categorical_columns,
                index = 0,
                key="cat_column2",
            )
            st.text_input(
                label="Enter New Column Name",
                key="cat_col_combine_new_name",
            )
            st.text_input(
                label="Any character to insert between the values? (Default: nothing)",
                value="",
                key="cat_col_combine_middle_char",
            )
            st.button(
                "Combine Columns",
                key="cat_col_combine_bt",
                on_click=combine_cat_col,
            )
            if "combine_message" in st.session_state:
                st.success(st.session_state.pop("combine_message"))
            if "combine_error" in st.session_state:
                st.error(st.session_state.pop("combine_error"))

    elif st.session_state.column_type == "Datetime":
        if not datetime_columns:
            st.warning("The uploaded file has no datetime-typed columns to transform.")
            return
        st.selectbox(
            label="Select Column",
            options=datetime_columns,
            index = 0,
            key="dt_column",
        )
        operation = st.selectbox(
            label="Select Operation",
            options=["Delete", "Extracting Individual Date Components"],
            index = 0,
            key="feature_engineering_dt_encoding_technique",
        )
        if operation == "Extracting Individual Date Components":
            st.multiselect(
                label="Select Components",
                options=sorted(DT_OPERATIONS.keys()),
                key="dt_components_op",
            )
        st.button("Apply Changes", key="dt_col_op_bt", on_click=dt_col_op)
        if "dt_delete_message" in st.session_state:
            st.success(st.session_state.pop("dt_delete_message"))
        if "col_not_found_error" in st.session_state:
            st.warning(st.session_state.pop("col_not_found_error"))
        if "extracted_col_message" in st.session_state:
            st.success(st.session_state.pop("extracted_col_message"))
        if "dt_exception" in st.session_state:
            st.error(st.session_state.pop("dt_exception"))
