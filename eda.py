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

    with univariate:
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
                            col_for_chart_numerical = st.session_state.univariate_chart_column_numerical
                            fig = go.Figure()

                            if st.session_state.get('univariate_chart_type_numerical') == 'Histogram':
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
                            elif st.session_state.get('univariate_chart_type_numerical') == 'Box Plot':
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
                            elif st.session_state.get('univariate_chart_type_numerical') == 'Violin Plot':
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
                            col_for_chart_categorical = st.session_state.univariate_chart_column_categorical
                            value_counts = df[col_for_chart_categorical].value_counts()
                        elif st.session_state.get('univariate_chart_type_categorical') in ['Pie Chart']:
                            column = st.selectbox(
                                label = 'Select Column',
                                options = cat_col,
                                index = 0,
                                key = 'univariate_chart_column_categorical'
                            )
                            donut_need = st.checkbox(
                                label = 'Donut?',
                                key = 'categorical_donut_need_pie'
                            )
                            col_for_chart_categorical = st.session_state.univariate_chart_column_categorical
                            value_counts = df[col_for_chart_categorical].value_counts()

                        with col2:
                            fig = go.Figure()

                            if st.session_state.get('univariate_chart_type_categorical') == 'Bar Chart':
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

                            elif st.session_state.get('univariate_chart_type_categorical') == 'Pie Chart':
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
                            col_for_chart_boolean = st.session_state.univariate_chart_column_boolean
                            value_counts = df[col_for_chart_boolean].value_counts()
                        elif st.session_state.get('univariate_chart_type_boolean') in ['Pie Chart']:
                            column = st.selectbox(
                                label = 'Select Column',
                                options = bool_col,
                                index = 0,
                                key = 'univariate_chart_column_boolean'
                            )
                            donut_need = st.checkbox(
                                label = 'Donut?',
                                key = 'donut_need_pie'
                            )
                            col_for_chart_boolean = st.session_state.univariate_chart_column_boolean
                            value_counts = df[col_for_chart_boolean].value_counts()

                        with col2:
                            fig = go.Figure()

                            if st.session_state.get('univariate_chart_type_boolean') == 'Bar Chart':
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

                            elif st.session_state.get('univariate_chart_type_boolean') == 'Pie Chart':
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

    with bivariate:
        col1, col2 = st.columns(2, border = True)

        with col1:
            columns_type = st.selectbox(
                label = 'Select Column Type',
                options = sorted(['Numerical v/s Numerical', 'Numerical v/s Categorical', 'Categorical v/s Categorical', 'Numerical v/s Datetime']),
                index = 0,
                key = 'bivariate_columns_type'
            )

            if st.session_state.get('bivariate_columns_type') == 'Numerical v/s Numerical':
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
                    fig = go.Figure()

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

            elif st.session_state.get('bivariate_columns_type') == 'Numerical v/s Categorical':
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

                    column1 = st.selectbox(
                        label = 'Select Numerical Column',
                        options = num_col,
                        index = 0,
                        key = 'bivariate_num_column_nvc'
                    )
                    column2 = st.selectbox(
                        label = 'Select Categorical Column',
                        options = cat_col,
                        index = 0,
                        key = 'bivariate_cat_column_nvc'
                    )
                elif st.session_state.get('bivariate_chart_type_nvc') == 'Box Plot':
                    cat_column = st.selectbox(
                        label = 'Select Categorical Column',
                        options = cat_col
                    )
                    num_column = st.selectbox(
                        label = 'Select Numerical Column',
                        options = num_col
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
                        key = 'bivariate_line_cat_col'
                        )
                    num_column = st.selectbox(
                        label = 'Select Numerical Column',
                        options = num_col,
                        key = 'bivariate_line_num_col'
                    )
                    agg_func = st.selectbox(
                        label = 'Select Aggregation Function',
                        options = sorted(['Mean', 'Sum', 'Median', 'Count']),
                        key = 'bivariate_line_agg_func'
                    )
                
                elif st.session_state.get('bivariate_chart_type_nvc') == 'Violin Plot':
                    cat_column = st.selectbox(
                        label = 'Select Categorical Column',
                        options = cat_col,
                        key = 'bivariate_line_cat_col'
                        )
                    num_column = st.selectbox(
                        label = 'Select Numerical Column',
                        options = num_col,
                        key = 'bivariate_line_num_col'
                    )
                    mean_line = st.checkbox(
                        label = 'Mean Line Needed?',
                        key = 'bivariate_violin_mean_line_need'
                    )
                    box_needed = st.checkbox(
                        label = 'Box Needed?',
                        key = 'bivariate_violin_box_need'
                    )

                with col2:
                    fig = go.Figure()

                    if st.session_state.get('bivariate_chart_type_nvc') == 'Bar Chart':
                        fig.add_traces(
                            go.Bar(
                                x = df[column2], 
                                y = df[column1],
                                opacity = 0.7,
                                name = f'{column1} vs {column2}',
                                orientation = 'h' if st.session_state.get('bivariate_chart_orientation') == 'Horizontal' else 'v'
                            )
                        )

                        st.plotly_chart(fig, width = 'stretch')

                    elif st.session_state.get('bivariate_chart_type_nvc') == 'Box Plot':
                        fig.add_trace(
                            go.Box(
                                x = df[cat_column],
                                y = df[num_column],
                                boxmean = st.session_state.get('bivariate_boxplot_mean_line_needed', False),
                                notched = st.session_state.get('bivariate_boxplot_notched_needed', False),
                                opacity = 0.8,
                                name = f'{cat_column} by {num_column}'
                            )
                        )

                        fig.update_layout(
                            title = f'{cat_col} by {num_col}',
                            xaxis_title = cat_column,
                            yaxis_title = num_column
                        )

                        st.plotly_chart(fig, width = 'stretch')
                    
                    elif st.session_state.get('bivariate_chart_type_nvc') == 'Line Chart':
                        agg =  st.session_state.get('bivariate_line_agg_func', 'Mean')

                        if agg == 'Mean':
                            line_df = df.groupby(st.session_state.get('bivariate_line_cat_col'), observed = True)[st.session_state.get('bivariate_line_num_col')].mean()
                        elif agg == 'Median':
                            line_df = df.groupby(st.session_state.get('bivariate_line_cat_col'), observed = True)[st.session_state.get('bivariate_line_num_col')].median()
                        elif agg == 'Sum':
                            line_df = df.groupby(st.session_state.get('bivariate_line_cat_col'), observed = True)[st.session_state.get('bivariate_line_num_col')].sum()
                        elif agg == 'Count':
                            line_df = df.groupby(st.session_state.get('bivariate_line_cat_col'), observed = True)[st.session_state.get('bivariate_line_num_col')].count()

                        fig.add_trace(
                            go.Scatter(
                                x = line_df.index.astype(str),
                                y = line_df.values,
                                mode = 'lines+markers',
                                name = f'{agg} {st.session_state.get('bivariate_line_num_col')}',
                                line = dict(width = 3),
                                marker = dict(size = 10),
                                text = st.session_state.get('bivariate_line_num_col'),
                                opacity = 0.8
                            )
                        )

                        fig.update_layout(
                            title = f'{agg} of {st.session_state.get('bivariate_line_num_col')} by {st.session_state.get('bivariate_line_cat_col')}',
                            xaxis_title = st.session_state.get('bivariate_line_cat_col'),
                            yaxis_title = st.session_state.get('bivariate_line_num_col')
                        )

                        st.plotly_chart(fig, width = 'stretch')

                    elif st.session_state.get('bivariate_chart_type_nvc') == 'Violin Plot':
                        fig.add_trace(
                            go.Violin(
                                x = df[cat_column],
                                y = df[num_column],
                                box = dict(visible = st.session_state.get('bivariate_violin_box_need', False)),
                                meanline = dict(visible = st.session_state.get('bivariate_violin_mean_line_need', False)),
                                opacity = 0.8
                            )
                        )
                        st.plotly_chart(fig, width = 'stretch')

            elif st.session_state.get('bivariate_columns_type') == 'Categorical v/s Categorical':
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
                            key = 'bivariate_heatmap_first_cat_col'
                        )
                        cat_col2 = st.selectbox(
                            label = 'Select second categorical column',
                            options = cat_columns,
                            index = 0,
                            key = 'bivariate_heatmap_second_cat_col'
                        )
                elif st.session_state.get('bivariate_chart_type_cvc') == 'Grouped Bar Chart':
                    cat_columns = sorted(cat_col + bool_col)

                    if len(cat_columns) < 2:
                        st.warning('Atleast two categorical columns are required!')
                    else:
                        cat_col1 = st.selectbox(
                            label = 'Select first categorical column',
                            options = cat_columns,
                            index = 0,
                            key = 'bivariate_heatmap_first_cat_col'
                        )
                        cat_col2 = st.selectbox(
                            label = 'Select second categorical column',
                            options = cat_columns,
                            index = 0,
                            key = 'bivariate_heatmap_second_cat_col'
                        )
                elif st.session_state.get('bivariate_chart_type_cvc') == 'Stacked Bar Chart':
                    cat_columns = sorted(cat_col + bool_col)

                    if len(cat_columns) < 2:
                        st.warning('Atleast two categorical columns are required!')
                    else:
                        cat_col1 = st.selectbox(
                            label = 'Select first categorical column',
                            options = cat_columns,
                            index = 0,
                            key = 'bivariate_heatmap_first_cat_col'
                        )
                        cat_col2 = st.selectbox(
                            label = 'Select second categorical column',
                            options = cat_columns,
                            index = 0,
                            key = 'bivariate_heatmap_second_cat_col'
                        )
                elif st.session_state.get('bivariate_chart_type_cvc') == '100% Stacked Bar Chart':
                    cat_columns = sorted(cat_col + bool_col)

                    if len(cat_columns) < 2:
                        st.warning('Atleast two categorical columns are required!')
                    else:
                        cat_col1 = st.selectbox(
                            label = 'Select first categorical column',
                            options = cat_columns,
                            index = 0,
                            key = 'bivariate_heatmap_first_cat_col'
                        )
                        cat_col2 = st.selectbox(
                            label = 'Select second categorical column',
                            options = cat_columns,
                            index = 0,
                            key = 'bivariate_heatmap_second_cat_col'
                        )

                with col2:
                    fig = go.Figure()

                    if st.session_state.get('bivariate_chart_type_cvc') == 'Heatmap':
                        cross_tab = pd.crosstab(
                            df[cat_col1],
                            df[cat_col2]
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
                            title = f'{cat_col1} vs {cat_col2} Heatmap',
                            xaxis_title = cat_col2,
                            yaxis_title = cat_col1
                        )
                        st.plotly_chart(fig, width = 'stretch')

                    elif st.session_state.get('bivariate_chart_type_cvc') == 'Grouped Bar Chart':
                        cross_tab = pd.crosstab(
                            df[cat_col1],
                            df[cat_col2]
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
                            title = f'{cat_col1} vs {cat_col2} Grouped Bar Chart',
                            xaxis_title = cat_col1,
                            yaxis_title = 'Count',
                            barmode = 'group'
                        )
                        st.plotly_chart(fig, width = 'stretch')
                    elif st.session_state.get('bivariate_chart_type_cvc') == 'Stacked Bar Chart':
                        cross_tab = pd.crosstab(
                            df[cat_col1],
                            df[cat_col2]
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
                            title = f'{cat_col1} vs {cat_col2} Stacked Bar Chart',
                            xaxis_title = cat_col1,
                            yaxis_title = 'Count',
                            barmode = 'stack'
                        )
                        st.plotly_chart(fig, width = 'stretch')
                    elif st.session_state.get('bivariate_chart_type_cvc') == '100% Stacked Bar Chart':
                        cross_tab = pd.crosstab(
                            df[cat_col1],
                            df[cat_col2],
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
                                    f'{cat_col1}: %{{x}}<br>'
                                    f'{cat_col2}: {category}<br>'
                                    'Percentage: %{y:.1f}%'
                                    '<extra></extra>'
                                )
                                )
                            )
                        fig.update_layout(
                            title = f'{cat_col1} vs {cat_col2} (100% Stacked)',
                            xaxis_title = cat_col1,
                            yaxis_title = 'Percentage (%)',
                            barmode = 'stack',
                            yaxis = dict( range = [0, 100])
                        )
                        st.plotly_chart(fig, width = 'stretch')
            elif st.session_state.get('bivariate_columns_type') == 'Numerical v/s Datetime':
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
                aggregation = st.selectbox(
                    label = 'Select Aggregation',
                    options = sorted(['Mean', 'Median', 'Sum', 'Count']),
                    index = 0,
                    key = 'bivariate_line_agg_func'
                )

                with col2:
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

    with multivariate:
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
            elif st.session_state.get('multivariate_chart_type') == 'Pair Plot':
                selected_cols = st.multiselect(
                    label = 'Select Numerical Columns',
                    options = num_columns,
                    default = [],
                    key = 'multivariate_heatmap_num_col'
                )
            elif st.session_state.get('multivariate_chart_type') in ['Grouped Bar Chart', 'Stacked Bar Chart']:
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
                elif st.session_state.get('multivariate_chart_type') == 'Pair Plot':
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
                elif chart_type in [
                    'Grouped Bar Chart',
                    'Stacked Bar Chart'
                ]:

                    if num_col and len(cat_col) >= 2:
                        aggregation_map = {
                            'Sum': 'sum',
                            'Mean': 'mean',
                            'Median': 'median',
                            'Count': 'count'
                        }
                        bar_df = df.pivot_table(
                            index=column2,
                            columns=column3,
                            values=column1,
                            aggfunc=aggregation_map[aggregation],
                            fill_value=0
                        )
                        if bar_df.empty:
                            st.warning('No data available to plot!')
                        else:
                            fig = go.Figure()
                            text_values = x_values if orientation == 'Horizontal' else y_values
                            bar_orientation = 'h' if orientation == 'Horizontal' else 'v'
                            for category in bar_df.columns:
                                if bar_orientation == 'h':
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
                                        orientation=bar_orientation,
                                        text = f'{y_values : .3f}',
                                        opacity=0.8
                                    )
                                )
                            barmode = 'group' if chart_type == 'Grouped Bar Chart' else 'stack'
                            fig.update_layout(
                                title=f'{aggregation} {column1} by {column2} and {column3}',
                                barmode=barmode,
                                xaxis_title=column1 if orientation == 'Horizontal' else column2,
                                yaxis_title=column2 if orientation == 'Horizontal' else column1,
                                legend_title=column3
                            )
                            st.plotly_chart(fig, width='stretch')