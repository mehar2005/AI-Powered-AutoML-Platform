import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def get_value_counts(df, column):
    return df[column].value_counts(dropna = False)


@st.cache_data
def get_crosstab(df, first_col, second_col, normalize = False):
    return pd.crosstab(
        df[first_col],
        df[second_col],
        normalize = normalize
    )


@st.cache_data
def get_grouped_values(df, category_col, numeric_col, aggregation):
    aggregation_map = {
        'Mean': 'mean',
        'Median': 'median',
        'Sum': 'sum',
        'Count': 'count'
    }

    return (
        df.groupby(category_col, observed = True)[numeric_col]
        .agg(aggregation_map[aggregation])
        .reset_index()
    )


@st.cache_data
def get_correlation(df, selected_cols, method):
    return df[list(selected_cols)].corr(method = method)


@st.cache_data
def get_pivot_table(df, row_col, column_col, value_col, aggregation):
    aggregation_map = {
        'Sum': 'sum',
        'Mean': 'mean',
        'Median': 'median',
        'Count': 'count'
    }

    return df.pivot_table(
        index = row_col,
        columns = column_col,
        values = value_col,
        aggfunc = aggregation_map[aggregation],
        fill_value = 0
    )

@st.fragment
def univariate_analysis(cat_col, num_col, bool_col):
            try:
                df = st.session_state.get('file_uploaded')
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("Upload a Dataset First!")
                with st.container():
                    col1, col2 = st.columns(2, border = True)
    
                    with col1:
                        column_type = st.selectbox(
                            label = 'Select Column Type',
                            options = sorted(['Numerical', 'Categorical', 'Boolean']),
                            index = 0,
                            key = 'univariate_chart_column_type'
                        )
    
                        if st.session_state.get('univariate_chart_column_type') == 'Numerical':
                            if len(num_col) == 0:
                                st.warning('No Columns found!')
                            else:
                                chart_type = st.selectbox(
                                    label = 'Select Chart Type',
                                    options = sorted(['Histogram', 'Box Plot', 'Violin Plot']),
                                    index = 0,
                                    key = 'univariate_chart_type_numerical'
                                )
    
                                if st.session_state.get('univariate_chart_type_numerical') in ['Histogram']:
                                    column = st.selectbox(
                                        label = 'Select Column',
                                        options = num_col,
                                        index = 0,
                                        key = 'univariate_chart_column_numerical'
                                    )
                                    nbins = st.number_input(
                                        label = 'Enter number of bins',
                                        min_value = 1,
                                        max_value = max(1, len(df[column])),
                                        value = max(1, min(int(len(df[column]) ** 0.5), len(df[column]))),
                                        step = 1,
                                        key = 'histogram_number_bins'
                                    )
                                    cumulative = st.checkbox(
                                        label = 'Cumulative?',
                                        key = 'cumulative_need_histogram'
                                    )
                                elif st.session_state.get('univariate_chart_type_numerical') in ['Box Plot']:
                                    column = st.selectbox(
                                        label = 'Select Column',
                                        options = num_col,
                                        index = 0,
                                        key = 'univariate_chart_column_numerical'
                                    )
                                    mean_line = st.checkbox(
                                        label = 'Mean Line Needed?',
                                        key = 'mean_line_boxplot_need'
                                    )
                                    notched = st.checkbox(
                                        label = 'Notched?',
                                        key = 'notched_boxplot_need'
                                    )
                                elif st.session_state.get('univariate_chart_type_numerical') in ['Violin Plot']:
                                    column = st.selectbox(
                                        label = 'Select Column',
                                        options = num_col,
                                        index = 0,
                                        key = 'univariate_chart_column_numerical'
                                    )
                                    mean_line = st.checkbox(
                                        label = 'Mean Line Needed?',
                                        key = 'mean_line_violin_need'
                                    )
                                    box_needed = st.checkbox(
                                        label = 'Box Needed?',
                                        key = 'box_violin_needed'
                                    )
    
                                with col2:
                                        col_for_chart_numerical = st.session_state.get('univariate_chart_column_numerical')
                                        chart_type = st.session_state.get('univariate_chart_type_numerical')

                                        if col_for_chart_numerical not in num_col:
                                            raise ValueError('Select a valid numerical column first!')

                                        elif chart_type == 'Histogram':
                                            fig = go.Figure()

                                            fig.add_trace(
                                                go.Histogram(
                                                    x = df[col_for_chart_numerical],
                                                    name = col_for_chart_numerical,
                                                    opacity = 0.8,
                                                    nbinsx = st.session_state.get('histogram_number_bins', 10),
                                                    cumulative = dict(enabled = st.session_state.get('cumulative_need_histogram', False))
                                                )
                                            )

                                            fig.update_layout(
                                                title = f'{col_for_chart_numerical} Distribution',
                                                showlegend = True,
                                                template = 'seaborn'
                                            )

                                            st.plotly_chart(fig, width = 'stretch')

                                        elif chart_type == 'Box Plot':
                                            fig = go.Figure()

                                            fig.add_trace(
                                                go.Box(
                                                    y = df[col_for_chart_numerical],
                                                    name = col_for_chart_numerical,
                                                    boxmean = st.session_state.get('mean_line_boxplot_need', False),
                                                    opacity = 0.8,
                                                    quartilemethod = 'linear',
                                                    notched = st.session_state.get('notched_boxplot_need', False)
                                                )
                                            )

                                            fig.update_layout(
                                                title = f'{col_for_chart_numerical} Box Plot',
                                                yaxis_title = col_for_chart_numerical
                                            )

                                            st.plotly_chart(fig, width = 'stretch')

                                        elif chart_type == 'Violin Plot':
                                            fig = go.Figure()

                                            fig.add_trace(
                                                go.Violin(
                                                    y = df[col_for_chart_numerical],
                                                    name = col_for_chart_numerical,
                                                    opacity = 0.8,
                                                    box = dict(
                                                        visible = st.session_state.get('box_violin_needed', False)
                                                    ),
                                                    meanline = dict(visible = st.session_state.get('mean_line_violin_need', False))
                                                )
                                            )

                                            fig.update_layout(
                                                title = f'{col_for_chart_numerical} Violin Plot',
                                                yaxis_title = col_for_chart_numerical
                                            )

                                            st.plotly_chart(fig, width = 'stretch')

                        elif st.session_state.get('univariate_chart_column_type') == 'Categorical':
                            if len(cat_col) == 0:
                                st.warning('No Columns found!')
                            else:
                                chart_type = st.selectbox(
                                    label = 'Select Chart Type',
                                    options = sorted(['Bar Chart', 'Pie Chart']),
                                    index = 0,
                                    key = 'univariate_chart_type_categorical'
                                )
    
                                if st.session_state.get('univariate_chart_type_categorical') in ['Bar Chart']:
                                    column = st.selectbox(
                                        label = 'Select Column',
                                        options = cat_col,
                                        index = 0,
                                        key = 'univariate_chart_column_categorical'
                                    )
                                    if st.session_state.univariate_chart_column_categorical:
                                        selected_col = st.session_state.get('univariate_chart_column_categorical')
                                elif st.session_state.get('univariate_chart_type_categorical') in ['Pie Chart']:
                                    column = st.selectbox(
                                        label = 'Select Column',
                                        options = cat_col,
                                        index = 0,
                                        key = 'univariate_chart_column_categorical'
                                    )
                                    if st.session_state.univariate_chart_column_categorical:
                                        selected_col = st.session_state.get('univariate_chart_column_categorical')

                                        donut_need = st.checkbox(
                                            label = 'Donut?',
                                            key = 'categorical_donut_need_pie'
                                        )

    
                                with col2:
                                    if selected_col not in cat_col:
                                        st.warning('Select a valid categorical column first!')

                                    else:
                                        value_counts = get_value_counts(df, selected_col)

                                        if len(value_counts) > 20:
                                            value_counts = value_counts.head(20)
                                            st.info('Showing the top 20 categories.')

                                        fig = go.Figure()

                                        if st.session_state.get('univariate_chart_type_categorical') == 'Bar Chart':
                                            fig.add_trace(
                                                go.Bar(
                                                    x = value_counts.index.astype(str),
                                                    y = value_counts.values,
                                                    name = selected_col,
                                                    opacity = 0.8
                                                )
                                            )

                                            fig.update_layout(
                                                title = f'{selected_col} Categories',
                                                xaxis_title = selected_col,
                                                yaxis_title = 'Count'
                                            )

                                        elif st.session_state.get('univariate_chart_type_categorical') == 'Pie Chart':
                                            fig.add_trace(
                                                go.Pie(
                                                    labels = value_counts.index.astype(str),
                                                    values = value_counts.values,
                                                    name = selected_col,
                                                    opacity = 0.8,
                                                    hole = 0.5 if donut_need else 0,
                                                    textinfo = 'label+percent'
                                                )
                                            )

                                            fig.update_layout(
                                                title = f'{selected_col} Categories'
                                            )

                                        st.plotly_chart(fig, width = 'stretch')
    
                        elif st.session_state.get('univariate_chart_column_type') == 'Boolean':
                            if len(bool_col) == 0:
                                st.warning('No Columns found!')
                            else:
                                chart_type = st.selectbox(
                                    label = 'Select Chart Type',
                                    options = sorted(['Bar Chart', 'Pie Chart']),
                                    index = 0,
                                    key = 'univariate_chart_type_boolean'
                                )
    
                                if st.session_state.get('univariate_chart_type_boolean') in ['Bar Chart']:
                                    column = st.selectbox(
                                        label = 'Select Column',
                                        options = bool_col,
                                        index = 0,
                                        key = 'univariate_chart_column_boolean'
                                    )
                                    if st.session_state.univariate_chart_column_boolean:
                                        st.session_state.univariate_bar_col_selected_bool = True
                                        selected_col = st.session_state.get('univariate_chart_column_boolean')
                                elif st.session_state.get('univariate_chart_type_boolean') in ['Pie Chart']:
                                    column = st.selectbox(
                                        label = 'Select Column',
                                        options = bool_col,
                                        index = 0,
                                        key = 'univariate_chart_column_boolean'
                                    )
                                    if st.session_state.univariate_chart_column_boolean:
                                        st.session_state.univariate_pie_col_selected_bool = True
                                        selected_col = st.session_state.get('univariate_chart_column_boolean')
                                    donut_need = st.checkbox(
                                        label = 'Donut?',
                                        key = 'donut_need_pie'
                                    )
                                    col_for_chart_boolean = st.session_state.univariate_chart_column_boolean
                                    value_counts = df[col_for_chart_boolean].value_counts()
    
                                with col2:
                                    if selected_col not in bool_col:
                                        st.warning('Select a valid Boolean column first!')

                                    else:
                                        value_counts = get_value_counts(df, selected_col)

                                        fig = go.Figure()

                                        if st.session_state.get('univariate_chart_type_boolean') == 'Bar Chart':
                                            fig.add_trace(
                                                go.Bar(
                                                    x = value_counts.index.astype(str),
                                                    y = value_counts.values,
                                                    name = selected_col,
                                                    opacity = 0.8
                                                )
                                            )

                                            fig.update_layout(
                                                title = f'{selected_col} Categories',
                                                xaxis_title = selected_col,
                                                yaxis_title = 'Count'
                                            )

                                        elif st.session_state.get('univariate_chart_type_boolean') == 'Pie Chart':
                                            fig.add_trace(
                                                go.Pie(
                                                    labels = value_counts.index.astype(str),
                                                    values = value_counts.values,
                                                    name = selected_col,
                                                    opacity = 0.8,
                                                    hole = 0.5 if donut_need else 0,
                                                    textinfo = 'label+percent'
                                                )
                                            )

                                            fig.update_layout(
                                                title = f'{selected_col} Categories'
                                            )

                                        st.plotly_chart(fig, width = 'stretch')
            except Exception as err:
                st.exception(err)

@st.fragment
def bivariate_analysis(cat_col, num_col, dt_col, bool_col):
        try:
            df = st.session_state.get('file_uploaded')
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Upload a dataset first!")
            col1, col2 = st.columns(2, border = True)

            with col1:
                columns_type = st.selectbox(
                    label = 'Select Column Type',
                    options = sorted(['Numerical v/s Numerical', 'Numerical v/s Categorical', 'Categorical v/s Categorical', 'Numerical v/s Datetime']),
                    index = 0,
                    key = 'bivariate_columns_type'
                )

                if st.session_state.get('bivariate_columns_type') == 'Numerical v/s Numerical':
                    if len(num_col) == 0:
                        st.warning('No Numerical Columns found!')
                    else:
                        chart = st.selectbox(
                            label = 'Select Chart Type',
                            options = sorted(['Scatter Plot']),
                            index = 0,
                            key = 'bivariate_chart_type_nvn'
                        )
                        if st.session_state.get('bivariate_chart_type_nvn') == 'Scatter Plot':
                            column1 = st.selectbox(
                                label = 'Select First Column',
                                options = num_col,
                                index = 0,
                                key = 'bivariate_nvn_first_col'
                            )
                            column2 = st.selectbox(
                                label = 'Select Second Column',
                                options = num_col,
                                index = 0,
                                key = 'bivariate_nvn_second_col'
                            )

                        with col2:
                            first_col = st.session_state.get('bivariate_nvn_first_col')
                            second_col = st.session_state.get('bivariate_nvn_second_col')

                            if first_col not in num_col or second_col not in num_col:
                                st.warning('Select valid numerical columns first!')

                            elif first_col == second_col:
                                st.warning('Select two different numerical columns!')

                            else:
                                fig = go.Figure()

                                fig.add_trace(
                                    go.Scatter(
                                        x = df[first_col],
                                        y = df[second_col],
                                        mode = 'markers',
                                        opacity = 0.7,
                                        name = f'{first_col} vs {second_col}'
                                    )
                                )

                                fig.update_layout(
                                    title = f'{first_col} vs {second_col}',
                                    xaxis_title = first_col,
                                    yaxis_title = second_col
                                )

                                st.plotly_chart(fig, width = 'stretch')

                elif st.session_state.get('bivariate_columns_type') == 'Numerical v/s Categorical':
                    if len(num_col) == 0 and len(cat_col) == 0:
                        st.warning('No Numerical or Categorical Columns found!')
                    elif len(num_col) == 0:
                        st.warning('No Numerical Columns Found!')
                    elif len(cat_col) == 0:
                        st.warning('No Categorical Columns Found!')
                    else:
                        chart_type = st.selectbox(
                            label = 'Select Chart Type',
                            options = sorted(['Bar Chart', 'Line Chart', 'Box Plot', 'Violin Plot']),
                            key = 'bivariate_chart_type_nvc'
                        )

                        if st.session_state.get('bivariate_chart_type_nvc') == 'Bar Chart':
                            orientation = st.selectbox(
                                label = 'Select Orientation',
                                options = sorted(['Horizontal', 'Vertical']),
                                key = 'bivariate_chart_orientation',
                                index = 0
                            )
                            aggregation = st.selectbox(
                                label = 'Select Aggregation',
                                options = ['Mean', 'Median', 'Sum', 'Count'],
                                key = 'bivariate_bar_aggregation_nvc'
                            )

                            column1 = st.selectbox(
                                    label = 'Select Numerical Column',
                                    options = num_col,
                                    index = 0,
                                    key = 'bivariate_bar_chart_num_column_nvc'
                                )
                            column2 = st.selectbox(
                                    label = 'Select Categorical Column',
                                    options = cat_col,
                                    index = 0,
                                    key = 'bivariate_bar_chart_cat_column_nvc'
                                )

                        elif st.session_state.get('bivariate_chart_type_nvc') == 'Box Plot':
                            cat_column = st.selectbox(
                                label = 'Select Categorical Column',
                                options = cat_col,
                                key = 'bivariate_boxplot_cat_col_nvc'
                            )
                            num_column = st.selectbox(
                                label = 'Select Numerical Column',
                                options = num_col,
                                key = 'bivariate_boxplot_num_col_nvc'
                            )
                            mean_line_needed = st.checkbox(
                                label = 'Mean Line Needed?',
                                key = 'bivariate_boxplot_mean_line_needed'
                            )
                            notched_needed = st.checkbox(
                                label = 'Notched?',
                                key = 'bivariate_boxplot_notched_needed'
                            )

                        elif st.session_state.get('bivariate_chart_type_nvc') == 'Line Chart':
                            cat_column = st.selectbox(
                                label = 'Select Categorical Column',
                                options = cat_col,
                                key = 'bivariate_line_cat_col_nvc'
                                )
                            num_column = st.selectbox(
                                label = 'Select Numerical Column',
                                options = num_col,
                                key = 'bivariate_line_num_col_nvc'
                            )
                            agg_func = st.selectbox(
                                label = 'Select Aggregation Function',
                                options = sorted(['Mean', 'Sum', 'Median', 'Count']),
                                key = 'bivariate_nvc_line_agg_func'
                            )
                            if st.session_state.bivariate_line_cat_col_nvc and st.session_state.bivariate_line_num_col_nvc:
                                st.session_state.bivariate_line_col_selected_nvc = True
                        
                        elif st.session_state.get('bivariate_chart_type_nvc') == 'Violin Plot':
                            cat_column = st.selectbox(
                                label = 'Select Categorical Column',
                                options = cat_col,
                                key = 'bivariate_violin_cat_col_nvc'
                                )
                            num_column = st.selectbox(
                                label = 'Select Numerical Column',
                                options = num_col,
                                key = 'bivariate_violin_num_col_nvc'
                            )
                            mean_line = st.checkbox(
                                label = 'Mean Line Needed?',
                                key = 'bivariate_violin_mean_line_need'
                            )
                            box_needed = st.checkbox(
                                label = 'Box Needed?',
                                key = 'bivariate_violin_box_need'
                            )
                            if st.session_state.bivariate_violin_cat_col_nvc and st.session_state.bivariate_violin_num_col_nvc:
                                st.session_state.bivariate_violin_col_selected_nvc = True

                        with col2:
                            fig = go.Figure()

                            if st.session_state.get('bivariate_chart_type_nvc') == 'Bar Chart':
                                num_column = st.session_state.get('bivariate_bar_chart_num_column_nvc')
                                cat_column = st.session_state.get('bivariate_bar_chart_cat_column_nvc')
                                aggregation = st.session_state.get('bivariate_bar_aggregation_nvc')
                                orientation = st.session_state.get('bivariate_chart_orientation')

                                if num_column not in num_col or cat_column not in cat_col:
                                    st.warning('Select valid columns first!')

                                else:
                                    bar_df = get_grouped_values(
                                        df,
                                        cat_column,
                                        num_column,
                                        aggregation
                                    )

                                    is_horizontal = orientation == 'Horizontal'

                                    fig.add_trace(
                                        go.Bar(
                                            x = bar_df[num_column] if is_horizontal else bar_df[cat_column],
                                            y = bar_df[cat_column] if is_horizontal else bar_df[num_column],
                                            orientation = 'h' if is_horizontal else 'v',
                                            opacity = 0.7,
                                            name = f'{aggregation} {num_column} by {cat_column}'
                                        )
                                    )

                                    fig.update_layout(
                                        title = f'{aggregation} {num_column} by {cat_column}',
                                        xaxis_title = num_column if is_horizontal else cat_column,
                                        yaxis_title = cat_column if is_horizontal else num_column
                                    )

                                    st.plotly_chart(fig, width = 'stretch')
                                

                            elif st.session_state.get('bivariate_chart_type_nvc') == 'Box Plot':
                                    fig.add_trace(
                                        go.Box(
                                            x = df[st.session_state.bivariate_boxplot_cat_col_nvc],
                                            y = df[st.session_state.bivariate_boxplot_num_col_nvc],
                                            boxmean = st.session_state.get('bivariate_boxplot_mean_line_needed', False),
                                            notched = st.session_state.get('bivariate_boxplot_notched_needed', False),
                                            opacity = 0.8,
                                            name = f'{cat_column} by {num_column}'
                                        )
                                    )
                                    fig.update_layout(
                                        title = f'{st.session_state.bivariate_boxplot_cat_col_nvc} by {st.session_state.bivariate_boxplot_num_col_nvc}',
                                        xaxis_title = st.session_state.bivariate_boxplot_cat_col_nvc,
                                        yaxis_title = st.session_state.bivariate_boxplot_num_col_nvc
                                    )
                                    st.plotly_chart(fig, width = 'stretch')

                            
                            elif st.session_state.get('bivariate_chart_type_nvc') == 'Line Chart':
                                    agg = st.session_state.get('bivariate_nvc_line_agg_func', 'Mean')
                                    if agg == 'Mean':
                                        line_df = df.groupby(st.session_state.get('bivariate_line_cat_col_nvc'), observed = True)[st.session_state.get('bivariate_line_num_col_nvc')].mean()
                                    elif agg == 'Median':
                                        line_df = df.groupby(st.session_state.get('bivariate_line_cat_col_nvc'), observed = True)[st.session_state.get('bivariate_line_num_col_nvc')].median()
                                    elif agg == 'Sum':
                                        line_df = df.groupby(st.session_state.get('bivariate_line_cat_col_nvc'), observed = True)[st.session_state.get('bivariate_line_num_col_nvc')].sum()
                                    elif agg == 'Count':
                                        line_df = df.groupby(st.session_state.get('bivariate_line_cat_col_nvc'), observed = True)[st.session_state.get('bivariate_line_num_col_nvc')].count()
                                    fig.add_trace(
                                        go.Scatter(
                                            x = line_df.index.astype(str),
                                            y = line_df.values,
                                            mode = 'lines+markers',
                                            name = f'{agg} {st.session_state.get('bivariate_line_num_col_nvc')}',
                                            line = dict(width = 3),
                                            marker = dict(size = 10),
                                            text = st.session_state.get('bivariate_line_num_col_nvc'),
                                            opacity = 0.8
                                        )
                                    )
                                    fig.update_layout(
                                        title = f'{agg} of {st.session_state.get('bivariate_line_num_col_nvc')} by {st.session_state.get('bivariate_line_cat_col_nvc')}',
                                        xaxis_title = st.session_state.get('bivariate_line_cat_col_nvc'),
                                        yaxis_title = st.session_state.get('bivariate_line_num_col_nvc')
                                    )
                                    st.plotly_chart(fig, width = 'stretch')


                            elif st.session_state.get('bivariate_chart_type_nvc') == 'Violin Plot':
                                if st.session_state.get('bivariate_violin_col_selected_nvc') == True:
                                    fig.add_trace(
                                        go.Violin(
                                            x = df[st.session_state.get('bivariate_violin_cat_col_nvc')],
                                            y = df[st.session_state.get('bivariate_violin_num_col_nvc')],
                                            box = dict(visible = st.session_state.get('bivariate_violin_box_need', False)),
                                            meanline = dict(visible = st.session_state.get('bivariate_violin_mean_line_need', False)),
                                            opacity = 0.8
                                        )
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')

                elif st.session_state.get('bivariate_columns_type') == 'Categorical v/s Categorical':
                    if len(cat_col) == 0:
                        st.warning('No Categorical Columns found!')
                    else:
                        chart_type = st.selectbox(
                            label = 'Select Chart Type',
                            options = sorted(['Heatmap', 'Stacked Bar Chart', 'Grouped Bar Chart', '100% Stacked Bar Chart']),
                            index = 0,
                            key = 'bivariate_chart_type_cvc'
                        )

                        if st.session_state.get('bivariate_chart_type_cvc') == 'Heatmap':
                            cat_columns = sorted(cat_col + bool_col)

                            if len(cat_columns) < 2:
                                st.warning('Atleast two categorical columns are required!')
                            else:
                                cat_col1 = st.selectbox(
                                    label = 'Select first categorical column',
                                    options = cat_columns,
                                    index = 0,
                                    key = 'bivariate_heatmap_first_cat_col_cvc'
                                )
                                cat_col2 = st.selectbox(
                                    label = 'Select second categorical column',
                                    options = [col for col in cat_columns if col != cat_col1],
                                    index = 0,
                                    key = 'bivariate_heatmap_second_cat_col_cvc'
                                )
                            if st.session_state.bivariate_heatmap_first_cat_col_cvc and st.session_state.bivariate_heatmap_second_cat_col_cvc:
                                st.session_state.bivariate_heatmap_col_selected_cvc = True
                        elif st.session_state.get('bivariate_chart_type_cvc') == 'Grouped Bar Chart':
                            cat_columns = sorted(cat_col + bool_col)

                            if len(cat_columns) < 2:
                                st.warning('Atleast two categorical columns are required!')
                            else:
                                cat_col1 = st.selectbox(
                                    label = 'Select first categorical column',
                                    options = cat_columns,
                                    index = 0,
                                    key = 'bivariate_grouped_bar_chart_first_cat_col_cvc'
                                )
                                cat_col2 = st.selectbox(
                                    label = 'Select second categorical column',
                                    options = cat_columns,
                                    index = 0,
                                    key = 'bivariate_grouped_bar_chart_second_cat_col_cvc'
                                )
                            if st.session_state.bivariate_grouped_bar_chart_first_cat_col_cvc and st.session_state.bivariate_grouped_bar_chart_second_cat_col_cvc:
                                st.session_state.bivariate_grouped_bar_chart_col_selected_cvc = True
                        elif st.session_state.get('bivariate_chart_type_cvc') == 'Stacked Bar Chart':
                            cat_columns = sorted(cat_col + bool_col)

                            if len(cat_columns) < 2:
                                st.warning('Atleast two categorical columns are required!')
                            else:
                                cat_col1 = st.selectbox(
                                    label = 'Select first categorical column',
                                    options = cat_columns,
                                    index = 0,
                                    key = 'bivariate_stacked_bar_chart_first_cat_col_cvc'
                                )
                                cat_col2 = st.selectbox(
                                    label = 'Select second categorical column',
                                    options = cat_columns,
                                    index = 0,
                                    key = 'bivariate_stacked_bar_chart_second_cat_col_cvc'
                                )
                            if st.session_state.bivariate_stacked_bar_chart_first_cat_col_cvc and st.session_state.bivariate_stacked_bar_chart_second_cat_col_cvc:
                                st.session_state.bivariate_stacked_bar_chart_col_selected_cvc = True
                        elif st.session_state.get('bivariate_chart_type_cvc') == '100% Stacked Bar Chart':
                            cat_columns = sorted(cat_col + bool_col)

                            if len(cat_columns) < 2:
                                st.warning('Atleast two categorical columns are required!')
                            else:
                                cat_col1 = st.selectbox(
                                    label = 'Select first categorical column',
                                    options = cat_columns,
                                    index = 0,
                                    key = 'bivariate_100_stacked_first_cat_col_cvc'
                                )
                                cat_col2 = st.selectbox(
                                    label = 'Select second categorical column',
                                    options = cat_columns,
                                    index = 0,
                                    key = 'bivariate_100_stacked_second_cat_col_cvc'
                                )
                            if st.session_state.get('bivariate_100_stacked_first_cat_col_cvc') and st.session_state.get('bivariate_100_stacked_second_cat_col_cvc'):
                                st.session_state.bivariate_100_stacked_bar_chart_col_selected_cvc = True

                        with col2:
                            fig = go.Figure()

                            if st.session_state.get('bivariate_chart_type_cvc') == 'Heatmap':
                                if st.session_state.get('bivariate_heatmap_col_selected_cvc') == True:
                                    cross_tab = get_crosstab(
                                        df,
                                        st.session_state.get('bivariate_heatmap_first_cat_col_cvc'),
                                        st.session_state.get('bivariate_heatmap_second_cat_col_cvc')
                                    )
                                    fig.add_trace(
                                        go.Heatmap(
                                            z = cross_tab.values,
                                            x = cross_tab.columns.astype(str),
                                            y = cross_tab.index.astype(str),
                                            colorscale = 'reds',
                                            text = cross_tab.values,
                                            opacity = 0.8,
                                            texttemplate = '%{text}'
                                        )
                                    )
                                    fig.update_layout(
                                        title = f'{st.session_state.get('bivariate_heatmap_first_cat_col_cvc')} vs {st.session_state.get('bivariate_heatmap_second_cat_col_cvc')} Heatmap',
                                        xaxis_title = st.session_state.get('bivariate_heatmap_second_cat_col_cvc'),
                                        yaxis_title = st.session_state.get('bivariate_heatmap_first_cat_col_cvc')
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')

                            elif st.session_state.get('bivariate_chart_type_cvc') == 'Grouped Bar Chart':
                                if st.session_state.get('bivariate_grouped_bar_chart_col_selected_cvc') == True:
                                    cross_tab = get_crosstab(
                                        df,
                                        st.session_state.get('bivariate_grouped_bar_chart_first_cat_col_cvc'),
                                        st.session_state.get('bivariate_grouped_bar_chart_second_cat_col_cvc')
                                    )
                                    for category in cross_tab.columns:
                                        fig.add_trace(
                                            go.Bar(
                                            x = cross_tab.index.astype(str),
                                            y = cross_tab[category],
                                            name = str(category),
                                            opacity = 0.8,
                                            text = cross_tab[category],
                                            textposition ='outside'
                                            )
                                        )
                                    fig.update_layout(
                                        title = f'{st.session_state.get('bivariate_grouped_bar_chart_first_cat_col_cvc')} vs {st.session_state.get('bivariate_grouped_bar_chart_second_cat_col_cvc')} Grouped Bar Chart',
                                        xaxis_title = st.session_state.get('bivariate_grouped_bar_chart_first_cat_col_cvc'),
                                        yaxis_title = 'Count',
                                        barmode = 'group'
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')
                            elif st.session_state.get('bivariate_chart_type_cvc') == 'Stacked Bar Chart':
                                if st.session_state.get('bivariate_stacked_bar_chart_col_selected_cvc') == True:
                                    cross_tab = get_crosstab(
                                        df,
                                        st.session_state.get('bivariate_stacked_bar_chart_first_cat_col_cvc'),
                                        st.session_state.get('bivariate_stacked_bar_chart_second_cat_col_cvc')
                                    )
                                    for category in cross_tab.columns:
                                        fig.add_trace(
                                            go.Bar(
                                            x = cross_tab.index.astype(str),
                                            y = cross_tab[category],
                                            name = str(category),
                                            opacity = 0.8,
                                            text = cross_tab[category],
                                            textposition ='outside'
                                            )
                                        )
                                    fig.update_layout(
                                        title = f'{st.session_state.get('bivariate_stacked_bar_chart_first_cat_col_cvc')} vs {st.session_state.get('bivariate_stacked_bar_chart_second_cat_col_cvc')} Stacked Bar Chart',
                                        xaxis_title = st.session_state.get('bivariate_stacked_bar_chart_first_cat_col_cvc'),
                                        yaxis_title = 'Count',
                                        barmode = 'stack'
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')
                            elif st.session_state.get('bivariate_chart_type_cvc') == '100% Stacked Bar Chart':
                                if st.session_state.get('bivariate_100_stacked_bar_chart_col_selected_cvc', False) == True:
                                    cross_tab = get_crosstab(
                                        df, 
                                        st.session_state.get('bivariate_100_stacked_first_cat_col_cvc'),
                                        st.session_state.get('bivariate_100_stacked_second_cat_col_cvc'),
                                        normalize = 'index'
                                    ) * 100
                                    for category in cross_tab.columns:
                                        fig.add_trace(
                                            go.Bar(
                                            x = cross_tab.index.astype(str),
                                            y = cross_tab[category],
                                            name = str(category),
                                            opacity = 0.8,
                                            text = [f'{value:.2f}%'
                                                for value in cross_tab[category]],
                                            textposition ='inside',
                                            hovertemplate=(
                                                f'{st.session_state.get('bivariate_100_stacked_first_cat_col_cvc')}: %{{x}}<br>'
                                                f'{st.session_state.get('bivariate_100_stacked_second_cat_col_cvc')}: {category}<br>'
                                                'Percentage: %{y:.1f}%'
                                                '<extra></extra>'
                                            )
                                            )
                                        )
                                    fig.update_layout(
                                        title = f'{st.session_state.get('bivariate_100_stacked_first_cat_col_cvc')} vs {st.session_state.get('bivariate_100_stacked_second_cat_col_cvc')} (100% Stacked)',
                                        xaxis_title = st.session_state.get('bivariate_100_stacked_first_cat_col_cvc'),
                                        yaxis_title = 'Percentage (%)',
                                        barmode = 'stack',
                                        yaxis = dict( range = [0, 100])
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')

                elif st.session_state.get('bivariate_columns_type') == 'Numerical v/s Datetime':
                    if len(num_col) == 0 and len(dt_col) == 0:
                        st.warning('No Numerical or Datetime Columns found!')
                    elif len(num_col) == 0:
                        st.warning('No Numerical Columns found!')
                    elif len(dt_col) == 0:
                        st.warning('No Datetime Columns found!')
                    else:
                        chart_type = st.selectbox(
                            label = 'Select Chart Type',
                            options = sorted(['Line Chart']),
                            key = 'bivariate_chart_type_nvd'
                        )
                        date_column = st.selectbox(
                            label = 'Select Datetime Column',
                            options = dt_col,
                            index = 0,
                            key = 'bivariate_dt_col_nvd'
                        )
                        num_column = st.selectbox(
                            label = 'Select Numerical Column',
                            options = num_col,
                            index = 0,
                            key = 'bivariate_num_col_nvd'
                        )
                        frequency = st.selectbox(
                            label = 'Select Time Frequency',
                            options = sorted(['Raw Data', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']),
                            index = 0,
                            key = 'bivariate_line_datetime_frequency'
                        )
                        if frequency != 'Raw Data':
                            aggregation = st.selectbox(
                                label = 'Select Aggregation',
                                options = sorted(['Mean', 'Median', 'Sum', 'Count']),
                                index = 0,
                                key = 'bivariate_datetime_agg_func'
                            )

                        with col2:
                            date_column = st.session_state.get('bivariate_dt_col_nvd')
                            num_column = st.session_state.get('bivariate_num_col_nvd')
                            freq = st.session_state.get('bivariate_line_datetime_frequency')

                            if date_column not in dt_col or num_column not in num_col:
                                st.warning('Select valid columns first!')

                            else:
                                plot_df = df[[num_column, date_column]].dropna().copy()

                                plot_df[date_column] = pd.to_datetime(
                                    plot_df[date_column],
                                    errors = 'coerce'
                                )

                                plot_df = plot_df.dropna(subset = [date_column])

                                if freq == 'Raw Data':
                                    line_df = plot_df.sort_values(date_column)
                                    x_values = line_df[date_column]
                                    y_values = line_df[num_column]
                                    yaxis_title = num_column

                                else:
                                    aggregation_map = {
                                        'Sum': 'sum',
                                        'Mean': 'mean',
                                        'Median': 'median',
                                        'Count': 'count'
                                    }

                                    frequency_map = {
                                        'Daily': 'D',
                                        'Weekly': 'W',
                                        'Monthly': 'MS',
                                        'Quarterly': 'QS',
                                        'Yearly': 'YS'
                                    }

                                    agg = st.session_state.get('bivariate_datetime_agg_func')

                                    line_df = (
                                        plot_df.groupby(
                                            pd.Grouper(
                                                key = date_column,
                                                freq = frequency_map[freq]
                                            )
                                        )[num_column]
                                        .agg(aggregation_map[agg])
                                        .dropna()
                                        .sort_index()
                                    )

                                    x_values = line_df.index
                                    y_values = line_df.values
                                    yaxis_title = f'{agg} of {num_column}'

                                if len(x_values) == 0:
                                    st.warning('No data available to plot!')

                                else:
                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Scatter(
                                            x = x_values,
                                            y = y_values,
                                            mode = 'lines+markers',
                                            name = num_column,
                                            line = dict(width = 2),
                                            marker = dict(size = 8)
                                        )
                                    )

                                    fig.update_layout(
                                        title = f'{num_column} Over Time',
                                        xaxis_title = date_column,
                                        yaxis_title = yaxis_title
                                    )

                                    st.plotly_chart(fig, width = 'stretch')
        except Exception as err:
            st.exception(err)

@st.fragment
def multivariate_analysis(cat_col, num_col, dt_col, bool_col):
    try:
                df = st.session_state.get('file_uploaded')
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("Upload a Dataset First!")
                col1, col2 = st.columns(2, border = True)
                num_columns = num_col
                cat_columns = cat_col + bool_col
                with col1:
                    chart_type = st.selectbox(
                        label = 'Select Chart Type',
                        options = sorted(['Correlation Heatmap', 'Pair Plot', 'Grouped Bar Chart', 'Stacked Bar Chart']),
                        index = 0,
                        key = 'multivariate_chart_type'
                    )
                    if st.session_state.get('multivariate_chart_type') == 'Correlation Heatmap':
                        if len(num_col) == 0:
                            st.warning('No Numerical Columns found!')
                        else:
                            selected_cols = st.multiselect(
                                label = 'Select Numerical Columns',
                                options = num_columns,
                                default = [],
                                key = 'multivariate_heatmap_num_col'
                            )
                            corr_method = st.selectbox(
                                label = 'Select Correlation Method',
                                options = sorted(['Pearson', 'Spearman', 'Kendall']),
                                index = 0,
                                key = 'multivariate_heatmap_corr_method'
                            )
                            corr_method_map = {
                                'Pearson' : 'pearson',
                                'Spearman' : 'spearman',
                                'Kendall' : 'kendall'
                            }
                            if st.session_state.multivariate_heatmap_num_col and st.session_state.multivariate_heatmap_corr_method:
                                st.session_state.multivariate_heatmap_col_selected = True
                    elif st.session_state.get('multivariate_chart_type') == 'Pair Plot':
                        if len(num_col) == 0:
                            st.warning('No Numerical Columns found!')
                        else:
                            selected_cols = st.multiselect(
                                label = 'Select Numerical Columns',
                                options = num_columns,
                                default = [],
                                key = 'multivariate_selected_num_cols'
                            )

                    elif st.session_state.get('multivariate_chart_type') in ['Grouped Bar Chart', 'Stacked Bar Chart']:
                        if len(num_col) == 0 and len(cat_columns) < 2:
                            st.warning('Numerical data and at least two categorical or Boolean columns are required!')
                        if len(num_col) == 0:
                            st.warning('No Numerical Columns found!')
                        elif len(cat_columns) < 2:
                            st.warning('At least two categorical or Boolean columns are required!')
                        else:
                            if len(cat_columns) < 2:
                                st.warning("Atleast two categorical columns are required!")
                            else:
                                column1 = st.selectbox(
                                label='Select Numerical Column',
                                options=num_columns,
                                key='multivariate_bar_num_col' 
                                )
                                column2 = st.selectbox(
                                    label='Select First Categorical Column',
                                    options=cat_columns,
                                    key='multivariate_bar_cat_col1'
                                )
                                column3 = st.selectbox(
                                    label='Select Second Categorical Column',
                                    options=[
                                        col for col in cat_columns
                                        if col != column2
                                    ],
                                    key='multivariate_bar_cat_col2'
                                )
                                aggregation = st.selectbox(
                                label='Select Aggregation',
                                options=['Sum', 'Mean', 'Median', 'Count'],
                                key='multivariate_bar_aggregation'
                                )
                                orientation = st.selectbox(
                                    label='Select Orientation',
                                    options=['Horizontal', 'Vertical'],
                                    key='multivariate_bar_orientation'
                                )
    
                    with col2:
                        if st.session_state.get('multivariate_chart_type') == 'Correlation Heatmap':

                            selected_cols = st.session_state.get('multivariate_heatmap_num_col', [])

                            corr_method = st.session_state.get('multivariate_heatmap_corr_method')

                            corr_method_map = {
                                'Pearson': 'pearson',
                                'Spearman': 'spearman',
                                'Kendall': 'kendall'
                            }

                            if (len(selected_cols) < 2 or not all(col in num_col for col in selected_cols)):
                                st.warning('Select at least two valid numerical columns!')

                            else:
                                corr_df = get_correlation(
                                    df,
                                    selected_cols,
                                    corr_method_map[corr_method]
                                )

                                fig = go.Figure(
                                    data = go.Heatmap(
                                        z = corr_df.values,
                                        x = corr_df.columns,
                                        y = corr_df.index,
                                        colorscale = 'RdBu',
                                        zmin = -1,
                                        zmax = 1,
                                        text = corr_df.round(3).values,
                                        texttemplate = '%{text}'
                                    )
                                )

                                fig.update_layout(
                                    title = f'{corr_method} Correlation Heatmap',
                                    xaxis_title = 'Numerical Variables',
                                    yaxis_title = 'Numerical Variables'
                                )

                                st.plotly_chart(fig, width = 'stretch')
                        elif st.session_state.get('multivariate_chart_type') == 'Pair Plot':

                            selected_cols = st.session_state.get('multivariate_selected_num_cols', [])

                            if (
                                len(selected_cols) < 2
                                or not all(col in num_col for col in selected_cols)
                            ):
                                st.warning('Select at least two valid numerical columns!')

                            elif len(selected_cols) > 6:
                                st.warning('Select no more than six columns for a readable pair plot!')

                            else:
                                plot_df = df[selected_cols].dropna()

                                if len(plot_df) > 2000:
                                    plot_df = plot_df.sample(2000, random_state = 42)
                                    st.info('Pair plot uses a random sample of 2,000 rows.')

                                fig = px.scatter_matrix(
                                    plot_df,
                                    dimensions = selected_cols,
                                    title = 'Pair Plot of Numerical Variables'
                                )

                                fig.update_layout(height = 800)

                                st.plotly_chart(fig, width = 'stretch')
                        elif st.session_state.get('multivariate_chart_type') in ['Grouped Bar Chart', 'Stacked Bar Chart']:

                            selected_cat_col1 = st.session_state.get('multivariate_bar_cat_col1')
                            selected_cat_col2 = st.session_state.get('multivariate_bar_cat_col2')
                            selected_num_col = st.session_state.get('multivariate_bar_num_col')
                            aggregation = st.session_state.get('multivariate_bar_aggregation')
                            orientation = st.session_state.get('multivariate_bar_orientation')
                            chart_type = st.session_state.get('multivariate_chart_type')

                            if selected_num_col not in num_col:
                                st.warning('Select a valid numerical column!')

                            elif (
                                selected_cat_col1 not in cat_columns
                                or selected_cat_col2 not in cat_columns
                            ):
                                st.warning('Select valid categorical or Boolean columns!')

                            elif selected_cat_col1 == selected_cat_col2:
                                st.warning('Select two different categorical columns!')

                            else:
                                bar_df = get_pivot_table(
                                    df,
                                    selected_cat_col1,
                                    selected_cat_col2,
                                    selected_num_col,
                                    aggregation
                                )

                                if bar_df.empty:
                                    st.warning('No data available to plot!')

                                else:
                                    fig = go.Figure()

                                    is_horizontal = orientation == 'Horizontal'

                                    for category in bar_df.columns:
                                        fig.add_trace(
                                            go.Bar(
                                                x = (
                                                    bar_df[category]
                                                    if is_horizontal
                                                    else bar_df.index.astype(str)
                                                ),
                                                y = (
                                                    bar_df.index.astype(str)
                                                    if is_horizontal
                                                    else bar_df[category]
                                                ),
                                                name = str(category),
                                                orientation = 'h' if is_horizontal else 'v',
                                                opacity = 0.8,
                                                text = bar_df[category],
                                                textposition = 'outside'
                                            )
                                        )

                                    fig.update_layout(
                                        title = (
                                            f'{aggregation} {selected_num_col} by '
                                            f'{selected_cat_col1} and {selected_cat_col2}'
                                        ),
                                        barmode = (
                                            'group'
                                            if chart_type == 'Grouped Bar Chart'
                                            else 'stack'
                                        ),
                                        xaxis_title = (
                                            selected_num_col
                                            if is_horizontal
                                            else selected_cat_col1
                                        ),
                                        yaxis_title = (
                                            selected_cat_col1
                                            if is_horizontal
                                            else selected_num_col
                                        ),
                                        legend_title = selected_cat_col2
                                    )

                                    st.plotly_chart(fig, width = 'stretch')
    except Exception as err:
        st.exception(err)

def eda_show():
    df = st.session_state.file_uploaded

    if not isinstance(df, pd.DataFrame):
        st.warning('Upload a Dataset First!')
        return
    num_col = sorted(df.select_dtypes(
        include = 'number'
    ).columns.tolist())
    cat_col = sorted(df.select_dtypes(
        include = ['object', 'string', 'category']
    ).columns.tolist())
    bool_col = sorted(df.select_dtypes(
        include = ['boolean', 'bool']
    ).columns.tolist())
    dt_col = sorted(list([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]))
    for key in list(st.session_state.keys()):
        value = st.session_state.get(key)
        if 'col' in key.lower() and 'type' not in key.lower() and 'selected' not in key.lower() and isinstance(value, str) and value not in df.columns:
            del st.session_state[key]
        elif 'selected' in key.lower() and isinstance(value, bool):
            st.session_state[key] = False

    univariate, bivariate, multivariate = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])
    
    with univariate:
        univariate_analysis(cat_col, num_col, bool_col)

    with bivariate:
        bivariate_analysis(cat_col, num_col, dt_col, bool_col)
        
    with multivariate:
        multivariate_analysis(cat_col, num_col, dt_col, bool_col)
        