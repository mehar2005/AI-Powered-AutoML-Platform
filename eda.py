import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def eda_show():
    st.title('Exploratory Data Analysis', text_alignment = 'center')
    univariate, bivariate, multivariate = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])

    df = st.session_state.file_uploaded
    num_col = sorted(st.session_state.file_uploaded.select_dtypes(
        include = 'number'
    ).columns.tolist())
    cat_col = sorted(st.session_state.file_uploaded.select_dtypes(
        include = ['object', 'string', 'category']
    ).columns.tolist())
    bool_col = sorted(st.session_state.file_uploaded.select_dtypes(
        include = ['boolean', 'bool']
    ).columns.tolist())
    dt_col = sorted(list([col for col in st.session_state.file_uploaded.columns if pd.api.types.is_datetime64_any_dtype(df[col])]))
    for key in list(st.session_state.keys()):
        value = st.session_state.get(key)
        if 'col' in key.lower() and 'type' not in key.lower() and 'selected' not in key.lower() and isinstance(value, str) and value not in df.columns:
            del st.session_state[key]
        elif 'selected' in key.lower() and isinstance(value, bool):
            st.session_state[key] = False

    with univariate:
        try:
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
                        if st.session_state.file_uploaded.select_dtypes(include = ['number']).columns.tolist() == []:
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
                                if st.session_state.univariate_chart_column_numerical:
                                    st.session_state.univariate_hist_col_selected = True
                                nbins = st.number_input(
                                    label = 'Enter number of bins',
                                    min_value = 1,
                                    max_value = max(1, len(df[st.session_state.univariate_chart_column_numerical])),
                                    value = max(1, min(
                                        int(len(df[st.session_state.univariate_chart_column_numerical]) ** 0.5),
                                        len(df[st.session_state.univariate_chart_column_numerical])
                                    )),
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
                                if st.session_state.univariate_chart_column_numerical:
                                    st.session_state.univariate_boxplot_col_selected = True
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
                                if st.session_state.univariate_chart_column_numerical:
                                    st.session_state.univariate_violinplot_col_selected = True
                                mean_line = st.checkbox(
                                    label = 'Mean Line Needed?',
                                    key = 'mean_line_violin_need'
                                )
                                box_needed = st.checkbox(
                                    label = 'Box Needed?',
                                    key = 'box_violin_needed'
                                )

                            with col2:
                                col_for_chart_numerical = st.session_state.univariate_chart_column_numerical
                                fig = go.Figure()

                                if st.session_state.get('univariate_chart_type_numerical') == 'Histogram':
                                    if st.session_state.get('univariate_hist_col_selected') == True:
                                        fig.add_traces(
                                            go.Histogram(
                                                x = df[col_for_chart_numerical],
                                                text = df[col_for_chart_numerical],
                                                name = col_for_chart_numerical,
                                                opacity = 0.8,
                                                nbinsx = st.session_state.histogram_number_bins,
                                                cumulative = dict(enabled = st.session_state.get('cumulative_need_histogram'))
                                                )
                                            )
                                        fig.update_layout(
                                            title = col_for_chart_numerical + ' Distribution',
                                            showlegend = True,
                                            template = 'seaborn'
                                        )
                                        st.plotly_chart(fig, width = 'stretch')
                                    else:
                                        st.warning('Select Columns first!')
                                elif st.session_state.get('univariate_chart_type_numerical') == 'Box Plot':
                                    if st.session_state.get('univariate_boxplot_col_selected') == True:
                                        fig.add_traces(
                                            go.Box(
                                                y = df[col_for_chart_numerical],
                                                name = col_for_chart_numerical,
                                                boxmean = st.session_state.get('mean_line_boxplot_need'),
                                                opacity = 0.8,
                                                quartilemethod = 'linear',
                                                notched = st.session_state.get('notched_boxplot_need'),
                                                text = df[col_for_chart_numerical]
                                            )
                                        )
                                        fig.update_layout(
                                            title = f'{col_for_chart_numerical} Box Plot',
                                            yaxis_title=col_for_chart_numerical
                                        )
                                        st.plotly_chart(fig, width = 'stretch')
                                    else:
                                        st.warning('Select Columns first!')
                                elif st.session_state.get('univariate_chart_type_numerical') == 'Violin Plot':
                                    if st.session_state.get('univariate_violinplot_col_selected') == True:
                                        fig.add_traces(
                                            go.Violin(
                                                y = df[col_for_chart_numerical],
                                                name = col_for_chart_numerical,
                                                opacity = 0.8,
                                                box = dict(visible = st.session_state.get('box_violin_needed')),
                                                meanline = dict(visible = st.session_state.get('mean_line_violin_need'))
                                            )
                                        )
                                        fig.update_layout(
                                            title = f'{col_for_chart_numerical} Violin Plot',
                                            yaxis_title=col_for_chart_numerical,
                                        )
                                        st.plotly_chart(fig, width = 'stretch')
                                    else:
                                        st.warning('Select Columns first!')

                    elif st.session_state.get('univariate_chart_column_type') == 'Categorical':
                        if st.session_state.file_uploaded.select_dtypes(include = ['object', 'category', 'string']).columns.tolist() == []:
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
                                    st.session_state.univariate_bar_col_selected_cat = True
                                    col_for_chart_categorical = st.session_state.univariate_chart_column_categorical
                                    value_counts = df[col_for_chart_categorical].value_counts()
                            elif st.session_state.get('univariate_chart_type_categorical') in ['Pie Chart']:
                                column = st.selectbox(
                                    label = 'Select Column',
                                    options = cat_col,
                                    index = 0,
                                    key = 'univariate_chart_column_categorical'
                                )
                                if st.session_state.univariate_chart_column_categorical:
                                    st.session_state.univariate_pie_col_selected_cat = True
                                donut_need = st.checkbox(
                                    label = 'Donut?',
                                    key = 'categorical_donut_need_pie'
                                )
                                col_for_chart_categorical = st.session_state.univariate_chart_column_categorical
                                value_counts = df[col_for_chart_categorical].value_counts()

                            with col2:
                                fig = go.Figure()

                                if st.session_state.get('univariate_chart_type_categorical') == 'Bar Chart':
                                    if st.session_state.get('univariate_bar_col_selected_cat') == True:
                                        fig.add_traces(
                                            go.Bar(
                                                x = value_counts.index,
                                                y = value_counts.values,
                                                name = col_for_chart_categorical,
                                                opacity = 0.8
                                            )
                                        )
                                        fig.update_layout(
                                            title = f'{col_for_chart_categorical} Categories',
                                            xaxis_title = col_for_chart_categorical,
                                            yaxis_title = 'Count'
                                        )
                                        st.plotly_chart(fig, width = 'stretch')
                                    else:
                                        st.warning('Select Columns first!')

                                elif st.session_state.get('univariate_chart_type_categorical') == 'Pie Chart':
                                    if st.session_state.get('univariate_pie_col_selected_cat') == True:
                                        if st.session_state.get('categorical_donut_need_pie', False):
                                            fig.add_traces(
                                                go.Pie(
                                                    labels = value_counts.index,
                                                    values = value_counts.values,
                                                    name = col_for_chart_categorical,
                                                    opacity = 0.8,
                                                    hole = 0.5,
                                                    textinfo = 'label+percent'
                                                )
                                            )
                                        else:
                                            fig.add_traces(
                                                go.Pie(
                                                    labels = value_counts.index,
                                                    values = value_counts.values,
                                                    name = col_for_chart_categorical,
                                                    opacity = 0.8,
                                                    textinfo = 'label+percent'
                                                )
                                            )
                                        fig.update_layout(
                                            title = f'{col_for_chart_categorical} Categories'
                                        )
                                        st.plotly_chart(fig, width = 'stretch')
                                    else:
                                        st.warning('Select Columns first!')

                    elif st.session_state.get('univariate_chart_column_type') == 'Boolean':
                        if st.session_state.file_uploaded.select_dtypes(include = ['boolean', 'bool']).columns.tolist() == []:
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
                                    col_for_chart_boolean = st.session_state.univariate_chart_column_boolean
                                    value_counts = df[col_for_chart_boolean].value_counts()
                            elif st.session_state.get('univariate_chart_type_boolean') in ['Pie Chart']:
                                column = st.selectbox(
                                    label = 'Select Column',
                                    options = bool_col,
                                    index = 0,
                                    key = 'univariate_chart_column_boolean'
                                )
                                if st.session_state.univariate_chart_column_boolean:
                                    st.session_state.univariate_pie_col_selected_bool = True
                                donut_need = st.checkbox(
                                    label = 'Donut?',
                                    key = 'donut_need_pie'
                                )
                                col_for_chart_boolean = st.session_state.univariate_chart_column_boolean
                                value_counts = df[col_for_chart_boolean].value_counts()

                            with col2:
                                fig = go.Figure()

                                if st.session_state.get('univariate_chart_type_boolean') == 'Bar Chart':
                                    if st.session_state.get('univariate_bar_col_selected_bool') == True:
                                        fig.add_traces(
                                            go.Bar(
                                                x = value_counts.index,
                                                y = value_counts.values,
                                                name = col_for_chart_boolean,
                                                opacity = 0.8
                                            )
                                        )
                                        fig.update_layout(
                                            title = f'{col_for_chart_boolean} Categories',
                                            xaxis_title = col_for_chart_boolean,
                                            yaxis_title = 'Count'
                                        )
                                        st.plotly_chart(fig, width = 'stretch')
                                    else:
                                        st.warning('Select Columns first!')

                                elif st.session_state.get('univariate_chart_type_boolean') == 'Pie Chart':
                                    if st.session_state.get('univariate_pie_col_selected_bool') == True:
                                        if st.session_state.get('donut_need_pie', False):
                                            fig.add_traces(
                                                go.Pie(
                                                    labels = value_counts.index,
                                                    values = value_counts.values,
                                                    name = col_for_chart_boolean,
                                                    opacity = 0.8,
                                                    hole = 0.5,
                                                    textinfo = 'label+percent'
                                                )
                                            )
                                        else:
                                            fig.add_traces(
                                                go.Pie(
                                                    labels = value_counts.index,
                                                    values = value_counts.values,
                                                    name = col_for_chart_boolean,
                                                    opacity = 0.8,
                                                    textinfo = 'label+percent'
                                                )
                                            )
                                        fig.update_layout(
                                            title = f'{col_for_chart_boolean} Categories'
                                        )
                                        st.plotly_chart(fig, width = 'stretch')
                                    else:
                                        st.warning('Select Columns first!')
        except Exception:
            st.error("An Unknown error occured!")

    with bivariate:
        try:
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
                            if (st.session_state.bivariate_nvn_first_col) and (st.session_state.bivariate_nvn_second_col):
                                st.session_state.bivariate_scatter_col_selected_nvn = True

                        with col2:
                            fig = go.Figure()
                            if st.session_state.get('bivariate_scatter_col_selected_nvn') == True:
                                fig.add_traces(
                                    go.Scatter(
                                        x = df[column1],
                                        y = df[column2],
                                        mode = 'markers',
                                        name = column1 + ' ' + column2
                                    )
                                )
                                fig.update_layout(
                                    title = f'{column1} vs {column2}',
                                    xaxis_title = column1,
                                    yaxis_title = column2
                                )
                                st.plotly_chart(fig, width = 'stretch')
                            else:
                                st.warning('Select Columns first!')

                elif st.session_state.get('bivariate_columns_type') == 'Numerical v/s Categorical':
                    if len(num_col) == 0:
                        st.warning('No Numerical Columns Found!')
                    elif len(cat_col) == 0:
                        st.warning('No Categorical Columns Found!')
                    elif len(num_col) == 0 and len(cat_col) == 0:
                        st.warning('No Columns found!')
                    elif len(num_col) > 0 and len(cat_col) > 0:
                        chart_type = st.selectbox(
                            label = 'Select Chart Type',
                            options = sorted(['Bar Chart', 'Line Chart', 'Box Plot', 'Violin Plot']),
                            key = 'bivariate_chart_type_nvc'
                        )

                        if st.session_state.get('bivariate_chart_type_nvc') in ('Bar Chart'):
                            orientation = st.selectbox(
                                label = 'Select Orientation',
                                options = sorted(['Horizontal', 'Vertical']),
                                key = 'bivariate_chart_orientation',
                                index = 0
                            )
                            if len(num_col) == 0:
                                st.warning('No Numerical Columns found!')
                                st.session_state.bivariate_bar_chart_num_column_nvc = ''
                            else:
                                column1 = st.selectbox(
                                    label = 'Select Numerical Column',
                                    options = num_col,
                                    index = 0,
                                    key = 'bivariate_bar_chart_num_column_nvc'
                                )
                            if len(cat_col) == 0:
                                st.warning('No Categorical Columns found!')
                                st.session_state.bivariate_bar_chart_cat_column_nvc = ''
                            else:
                                column2 = st.selectbox(
                                    label = 'Select Categorical Column',
                                    options = cat_col,
                                    index = 0,
                                    key = 'bivariate_bar_chart_cat_column_nvc'
                                )
                            if st.session_state.bivariate_bar_chart_num_column_nvc and st.session_state.bivariate_bar_chart_cat_column_nvc:
                                st.session_state.bivariate_bar_col_selected_nvc = True

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

                            if st.session_state.bivariate_boxplot_cat_col_nvc and st.session_state.bivariate_boxplot_num_col_nvc:
                                st.session_state.bivariate_boxplot_col_selected_nvc = True

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
                                key = 'bivariate_line_agg_func'
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
                                if st.session_state.get('bivariate_bar_col_selected_nvc') == True:
                                    if st.session_state.get('bivariate_chart_orientation') == 'Horizontal':
                                        xbar = df[st.session_state.bivariate_bar_chart_num_column_nvc]
                                        ybar = df[st.session_state.bivariate_bar_chart_cat_column_nvc]
                                        orientation = 'h'
                                    else:
                                        xbar = df[st.session_state.bivariate_bar_chart_cat_column_nvc]
                                        ybar = df[st.session_state.bivariate_bar_chart_num_column_nvc]
                                        orientation = 'v'
                                    fig.add_traces(
                                        go.Bar(
                                            x = xbar, 
                                            y = ybar,
                                            opacity = 0.7,
                                            name = f'{column1} vs {column2}',
                                            orientation = 'h' if st.session_state.get('bivariate_chart_orientation') == 'Horizontal' else 'v'
                                        )
                                    )
                                    fig.update_layout(
                                        title = f'{st.session_state.bivariate_bar_chart_cat_column_nvc} by {st.session_state.bivariate_bar_chart_num_column_nvc}',
                                        xaxis_title = f'{st.session_state.bivariate_bar_chart_cat_column_nvc}',
                                        yaxis_title = f'{st.session_state.bivariate_bar_chart_num_column_nvc}'
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')
                                

                            elif st.session_state.get('bivariate_chart_type_nvc') == 'Box Plot':
                                if st.session_state.get('bivariate_boxplot_col_selected_nvc') == True:
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
                                else:
                                    st.warning('Select Columns first!')
                            
                            elif st.session_state.get('bivariate_chart_type_nvc') == 'Line Chart':
                                agg =  st.session_state.get('bivariate_line_agg_func', 'Mean')
                                if st.session_state.get('bivariate_line_col_selected_nvc') == True:
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
                                else:
                                    st.warning('Select Columns first!')

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
                                    options = cat_columns,
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
                                    cross_tab = pd.crosstab(
                                        df[st.session_state.bivariate_heatmap_first_cat_col_cvc],
                                        df[st.session_state.bivariate_heatmap_second_cat_col_cvc]
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
                                    cross_tab = pd.crosstab(
                                        df[st.session_state.bivariate_grouped_bar_chart_first_cat_col_cvc],
                                        df[st.session_state.bivariate_grouped_bar_chart_second_cat_col_cvc]
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
                                        title = f'{st.session_state.get('bivariate_heatmap_first_cat_col_cvc')} vs {st.session_state.get('bivariate_heatmap_second_cat_col_cvc')} Grouped Bar Chart',
                                        xaxis_title = st.session_state.get('bivariate_heatmap_first_cat_col_cvc'),
                                        yaxis_title = 'Count',
                                        barmode = 'group'
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')
                            elif st.session_state.get('bivariate_chart_type_cvc') == 'Stacked Bar Chart':
                                if st.session_state.get('bivariate_stacked_bar_chart_col_selected_cvc') == True:
                                    cross_tab = pd.crosstab(
                                        df[st.session_state.bivariate_stacked_bar_chart_first_cat_col_cvc],
                                        df[st.session_state.bivariate_stacked_bar_chart_second_cat_col_cvc]
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
                                        title = f'{st.session_state.get('bivariate_heatmap_first_cat_col_cvc')} vs {st.session_state.get('bivariate_heatmap_second_cat_col_cvc')} Stacked Bar Chart',
                                        xaxis_title = st.session_state.get('bivariate_heatmap_first_cat_col_cvc'),
                                        yaxis_title = 'Count',
                                        barmode = 'stack'
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')
                            elif st.session_state.get('bivariate_chart_type_cvc') == '100% Stacked Bar Chart':
                                if st.session_state.get('bivariate_100_stacked_bar_chart_col_selected_cvc', False) == True:
                                    cross_tab = pd.crosstab(
                                        df[st.session_state.get('bivariate_100_stacked_first_cat_col_cvc')],
                                        df[st.session_state.get('bivariate_100_stacked_second_cat_col_cvc')],
                                        normalize = 'index'
                                    ) * 100
                                    for category in cross_tab.columns:
                                        fig.add_trace(
                                            go.Bar(
                                            x = cross_tab.index.astype(str),
                                            y = cross_tab[category],
                                            name = str(category),
                                            opacity = 0.8,
                                            text = [f'{value : .2f}%'
                                                for value in cross_tab[category]],
                                            textposition ='inside',
                                            hovertemplate=(
                                                f'{st.session_state.get('bivariate_heatmap_first_cat_col_cvc')}: %{{x}}<br>'
                                                f'{st.session_state.get('bivariate_heatmap_second_cat_col_cvc')}: {category}<br>'
                                                'Percentage: %{y:.1f}%'
                                                '<extra></extra>'
                                            )
                                            )
                                        )
                                    fig.update_layout(
                                        title = f'{st.session_state.get('bivariate_heatmap_first_cat_col_cvc')} vs {st.session_state.get('bivariate_heatmap_second_cat_col_cvc')} (100% Stacked)',
                                        xaxis_title = st.session_state.get('bivariate_heatmap_first_cat_col_cvc'),
                                        yaxis_title = 'Percentage (%)',
                                        barmode = 'stack',
                                        yaxis = dict( range = [0, 100])
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                                else:
                                    st.warning('Select Columns first!')

                elif st.session_state.get('bivariate_columns_type') == 'Numerical v/s Datetime':
                    if len(num_col) == 0:
                        st.warning('No Numerical Columns found!')
                    elif len(dt_col) == 0:
                        st.warning('No Datetime Columns found!')
                    elif len(num_col) == 0 and len(dt_col) == 0:
                        st.warning('No Columns found!')
                    elif len(num_col) > 0 and len(dt_col) > 0:
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
                        if st.session_state.bivariate_dt_col_nvd and st.session_state.bivariate_num_col_nvd:
                            st.session_state.bivariate_col_selected_nvd = True
                        frequency = st.selectbox(
                            label = 'Select Time Frequency',
                            options = sorted(['Raw Data', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']),
                            index = 0,
                            key = 'bivariate_line_datetime_frequency'
                        )
                        aggregation = st.selectbox(
                            label = 'Select Aggregation',
                            options = sorted(['Mean', 'Median', 'Sum', 'Count']),
                            index = 0,
                            key = 'bivariate_line_agg_func'
                        )

                        with col2:
                            if st.session_state.get('bivariate_col_selected_nvd') == True:
                                freq = st.session_state.get('bivariate_line_datetime_frequency')
                                agg = st.session_state.get('bivariate_line_agg_func')
                                plot_df = df[[num_column, date_column]].dropna().copy()
                                plot_df[date_column] = pd.to_datetime(
                                    plot_df[date_column],
                                    errors = 'coerce'
                                )
                                plot_df = plot_df.dropna(subset = [date_column])

                                if freq == 'Raw Data':
                                    line_df = plot_df.sort_values(date_column)
                                    if agg == 'Count':
                                        line_df = line_df.groupby(date_column)[num_column].count().reset_index()
                                        y_values = line_df[num_column]
                                        x_values = line_df[date_column]
                                    else:
                                        x_values = line_df[date_column]
                                        y_values = line_df[num_column]
                                else:
                                    frequency_map = {
                                        'Daily' : "D",
                                        'Weekly':'W',
                                        'Monthly':'MS',
                                        'Quarterly':'QS',
                                        'Yearly':'YS'
                                    }
                                    aggregation_map = {
                                        'Sum': 'sum',
                                        'Mean': 'mean',
                                        'Median': 'median',
                                        'Count': 'count'
                                    }
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

                                if len(x_values) == 0:
                                    st.warning("No data available to plot!")
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
                                        yaxis_title = num_column if freq == 'Raw Data' else f'{agg} of {num_column}'
                                    )
                                    st.plotly_chart(fig, width = 'stretch')
                            else:
                                st.warning('Select Columns first!')
        except Exception :
            st.error("An Unknown error occured!")

    with multivariate:
        try:
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
                            key = 'multivariate_heatmap_num_col'
                        )
                        if st.session_state.multivariate_heatmap_num_col and len(st.session_state.multivariate_heatmap_num_col) > 2:
                            st.session_state.multivariate_pair_plot_col_selected = True
                elif st.session_state.get('multivariate_chart_type') in ['Grouped Bar Chart', 'Stacked Bar Chart']:
                    if len(num_col) == 0:
                        st.warning('No Numerical Columns found!')
                    elif len(cat_col) == 0:
                        st.warning('No Categorical Columns found!')
                    elif len(cat_col) < 2:
                        st.warning('More than two categorical columns are required!')
                    elif len(num_col) == 0 and len(cat_col) == 0:
                        st.warning('No Columns found!')
                    elif len(num_col) > 0 and len(cat_col) >= 2:
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
                            if st.session_state.multivariate_bar_num_col and st.session_state.multivariate_bar_cat_col1 and st.session_state.multivariate_bar_cat_col2:
                                st.session_state.multivariate_bar_charts_col_selected = True
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
                        if st.session_state.get('multivariate_heatmap_col_selected') == True:
                            if len(selected_cols) < 2:
                                st.warning('Select atleast two numerical columns')
                            else:
                                corr_df = df[selected_cols].corr(
                                    method = corr_method_map[corr_method]
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
                                    title = f'{st.session_state.multivariate_heatmap_corr_method.title()} Correlation Heatmap',
                                    xaxis_title = 'Numerical Variables',
                                    yaxis_title = 'Numerical Variables'
                                )
                                st.plotly_chart(fig, width = 'stretch')
                        else:
                            st.warning('Select Columns first!')
                    elif st.session_state.get('multivariate_chart_type') == 'Pair Plot':
                        if st.session_state.get('multivariate_pair_plot_col_selected') == True:
                            if len(selected_cols) < 2:
                                st.warning('Select atleast two numerical columns!')
                            else:
                                fig = px.scatter_matrix(
                                    df,
                                    dimensions = selected_cols,
                                    title = 'Pair Plot of Numerical Variables'
                                )
                                fig.update_layout(
                                    height = 800
                                )
                                st.plotly_chart(fig, width = 'stretch')
                        else:
                            st.warning('Select Columns first!')
                    elif chart_type in [
                        'Grouped Bar Chart',
                        'Stacked Bar Chart'
                    ]:
                        if st.session_state.get('multivariate_bar_charts_col_selected') == True:
                            if num_col and len(cat_col) >= 2:
                                aggregation_map = {
                                    'Sum': 'sum',
                                    'Mean': 'mean',
                                    'Median': 'median',
                                    'Count': 'count',
                                }
                                if_missing_cat_col1 = st.session_state.get('multivariate_bar_cat_col1')
                                if_missing_cat_col2 = st.session_state.get('multivariate_bar_cat_col2')
                                if_missing_num_col = st.session_state.get('multivariate_bar_num_col')
                                agg = st.session_state.get('multivariate_bar_aggregation')

                                selected_cat_cols = [if_missing_cat_col1, if_missing_cat_col2]

                                if not all(selected_cat_cols):
                                    st.warning('Please Select all the required columns!')
                                elif not all(col in cat_col for col in selected_cat_cols):
                                    st.error('One or more selected categorical columns no longer exist in the dataset.')
                                elif if_missing_num_col not in num_col:
                                    st.error('Selected Numerical column no longer exist in the dataset.')
                                elif agg not in aggregation_map:
                                    st.error('Please Select a valid aggregation method.')
                                else:
                                    bar_df = df.pivot_table(
                                        index=st.session_state.get('multivariate_bar_cat_col1'),
                                        columns=st.session_state.get('multivariate_bar_cat_col2'),
                                        values=st.session_state.get('multivariate_bar_num_col'),
                                        aggfunc=aggregation_map[st.session_state.get('multivariate_bar_aggregation', '')],
                                        fill_value=0
                                    )
                                    if bar_df.empty:
                                        st.warning('No data available to plot!')
                                    else:                                
                                        fig = go.Figure()
                                        is_horizontal = orientation == 'Horizontal'
                                        orientation_chart = 'h' if is_horizontal else 'v'
                                        for category in bar_df.columns:
                                            if is_horizontal:
                                                x_values = bar_df[category]
                                                y_values = bar_df.index.astype(str)
                                            else:
                                                x_values = bar_df.index.astype(str)
                                                y_values = bar_df[category]
                                            fig.add_trace(
                                                go.Bar(
                                                    x=x_values,
                                                    y=y_values,
                                                    name=str(category),
                                                    orientation=orientation_chart,
                                                    opacity=0.8
                                                )
                                            )
                                        barmode = 'group' if chart_type == 'Grouped Bar Chart' else 'stack'
                                        fig.update_layout(
                                            title=f'{aggregation} {column1} by {column2} and {column3}',
                                            barmode=barmode,
                                            xaxis_title=column1 if is_horizontal else column2,
                                            yaxis_title=column2 if is_horizontal else column1,
                                            legend_title=column3
                                        )
                                        st.plotly_chart(fig, width='stretch')

                            else:
                                st.warning("No Columns found!")
                        else:
                            st.warning('Select Columns first!')
        except Exception:
            st.error("An Unknown error occured!")