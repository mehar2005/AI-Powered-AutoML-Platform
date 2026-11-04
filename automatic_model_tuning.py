import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold, KFold

def get_cv(y, classification=True):
    if classification:
        min_count = y.value_counts().min()

        if min_count < 2:
            raise ValueError(
                "At least 2 observations per class are required for cross-validation."
            )

        return StratifiedKFold(
            n_splits=min(5, int(min_count)),
            shuffle=True,
            random_state=42
        )

    return KFold(
        n_splits=min(5, len(y)),
        shuffle=True,
        random_state=42
    )

target_var = st.session_state.get('target_variable')

if st.session_state.get('final_model_df') is not None:
    X = st.session_state.get('final_model_df').copy()
if st.session_state.get('file_uploaded') is not None:
    if target_var is not None:
        y = st.session_state.get('file_uploaded').loc[X.index, target_var].copy()
        min_class_count = y.value_counts().min()

param_grid_lnr = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}
param_grid_pr = {
    'polynomialfeatures__degree': [2, 3, 4, 5],
    'linearregression__fit_intercept': [True, False],
    'linearregression__positive': [True, False]
}
param_grid_dtr = {
    'criterion': [
        'squared_error',
        'friedman_mse',
        'absolute_error'
    ],
    'max_depth': [None, 3, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}
param_grid_rfr = {
    'n_estimators': [100, 200, 300],
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 'log2']
}
def get_knnr_param_grid(X, cv, max_train_size):
    min_train_size = min(min(len(train_index) for train_index, _ in cv.split(X)), max_train_size)

    return {
        'n_neighbors': [i for i in [3, 5, 7, 9, 11, 15] if i <= min_train_size],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 30, 40, 50],
        'p': [1, 2]
    }
param_grid_dtc = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 3, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}
def get_knnc_param_grid(X, y, cv, max_train_size):
    min_train_size = min(min(len(train_index) for train_index, _ in cv.split(X, y)), max_train_size)

    return {
        'n_neighbors': [i for i in [3, 5, 7, 9, 11, 15] if i <= min_train_size],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 30, 40, 50],
        'p': [1, 2]
    }
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}
param_grid_lgr = [
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': [None, 'balanced']
    },
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    },
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],
        'class_weight': [None, 'balanced']
    },
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.25, 0.5, 0.75],
        'class_weight': [None, 'balanced']
    }
]
param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': [None, 'balanced']
}


def tuneDTC(X, y):
    model = DecisionTreeClassifier()

    cv = get_cv(y, classification = True)

    rscv_dtc = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_dtc,
        n_iter = 10,
        scoring = 'accuracy',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_dtc.fit(
        X, y
    )

    best_model = rscv_dtc.best_params_
    st.session_state.best_dtc_params = rscv_dtc.best_params_

    st.dataframe(best_model)
def tuneKNNC(X, y):
    model = KNeighborsClassifier()

    cv = get_cv(y, classification=True)

    test_ratio_str = st.session_state.get('train_test_ratio')
    test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
    custom_test_ratio = st.session_state.get('custom_test_ratio')

    test_size = test_ratio_num or custom_test_ratio
    max_train_size = len(X) - int(np.ceil(len(X) * test_size))

    param_grid_knnc = get_knnc_param_grid(X, y, cv, max_train_size)

    rscv_knnc = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid_knnc,
        n_iter=10,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        cv=cv
    )

    rscv_knnc.fit(X, y)

    best_model = rscv_knnc.best_params_
    st.session_state.best_knnc_params = rscv_knnc.best_params_

    st.dataframe(best_model)
def tuneLGR(X, y):
    model = LogisticRegression()

    cv = get_cv(y, classification = True)

    rscv_lgr = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_lgr,
        n_iter = 20,
        scoring = 'accuracy',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_lgr.fit(
        X, y
    )

    best_model = rscv_lgr.best_params_
    st.session_state.best_lgr_params = rscv_lgr.best_params_

    st.dataframe(best_model)

def tuneNB(X, y):
    model = GaussianNB()

    cv = get_cv(y, classification = True)

    rscv_nb = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_nb,
        n_iter = 20,
        scoring = 'accuracy',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_nb.fit(
        X, y
    )

    best_model = rscv_nb.best_params_
    st.session_state.best_nb_params = rscv_nb.best_params_

    st.dataframe(best_model)
def tuneRFC(X, y):
    model = RandomForestClassifier()

    cv = get_cv(y, classification = True)

    rscv_rfc = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_rfc,
        n_iter = 20,
        scoring = 'accuracy',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_rfc.fit(
        X, y
    )

    best_model = rscv_rfc.best_params_
    st.session_state.best_rfc_params = rscv_rfc.best_params_

    st.dataframe(best_model)

def tuneDTR(X, y):
    model = DecisionTreeRegressor()

    cv = get_cv(y, classification = False)

    rscv_dtr = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_dtr,
        n_iter = 10,
        scoring = 'r2',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_dtr.fit(
        X, y
    )

    best_model = rscv_dtr.best_params_
    st.session_state.best_dtr_params = rscv_dtr.best_params_

    st.dataframe(best_model)
def tuneKNNR(X, y):
    model = KNeighborsRegressor()

    cv = get_cv(y, classification=False)

    test_ratio_str = st.session_state.get('train_test_ratio')
    test_ratio_num = 0.20 if test_ratio_str == '80:20' else 0.30 if test_ratio_str == '70:30' else 0.25 if test_ratio_str == '75:25' else None
    custom_test_ratio = st.session_state.get('custom_test_ratio')

    test_size = test_ratio_num or custom_test_ratio
    max_train_size = len(X) - int(np.ceil(len(X) * test_size))

    param_grid_knnr = get_knnr_param_grid(X, cv, max_train_size)

    rscv_knnr = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid_knnr,
        n_iter=10,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        cv=cv
    )

    rscv_knnr.fit(X, y)

    best_model = rscv_knnr.best_params_
    st.session_state.best_knnr_params = rscv_knnr.best_params_

    st.dataframe(best_model)
def tuneLNR(X, y):
    model = LinearRegression()

    cv = get_cv(y, classification = False)

    rscv_lnr = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_lnr,
        n_iter = 10,
        scoring = 'r2',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_lnr.fit(
        X, y
    )

    best_model = rscv_lnr.best_params_
    st.session_state.best_lnr_params = rscv_lnr.best_params_

    st.dataframe(best_model)
def tunePR(X, y):
    model = Pipeline([
        ('polynomialfeatures', PolynomialFeatures()),
        ('linearregression', LinearRegression())
    ])

    cv = get_cv(y, classification = False)

    rscv_pr = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_pr,
        n_iter = 10,
        scoring = 'r2',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_pr.fit(
        X, y
    )

    best_model = rscv_pr.best_params_
    st.session_state.best_pr_params = rscv_pr.best_params_

    st.dataframe(best_model)
def tuneRFR(X, y):
    model = RandomForestRegressor()

    cv = get_cv(y, classification = False)

    rscv_rfr = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid_rfr,
        n_iter = 10,
        scoring = 'r2',
        random_state=42,
        n_jobs = -1,
        cv = cv
    )
    rscv_rfr.fit(
        X, y
    )

    best_model = rscv_rfr.best_params_
    st.session_state.best_rfr_params = rscv_rfr.best_params_

    st.dataframe(best_model)

