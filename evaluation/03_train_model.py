__author__ = 'Mario Amrehn'

"""Predict AttrakDiff and SUS scores from interaction log data utilizing gradient boosted regression trees"""


import pickle
import json
from typing import Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from zenlog import log

from evaluation.questionnaires.questionnaire_eval import generate_dummy_questionnaire_data
from evaluation.questionnaires.questionnaire_eval import evaluate_sus_questionnaire_data_per_user
from evaluation.questionnaires.questionnaire_eval import evaluate_attrakdiff_questionnaire_data_per_user


def increase_nice_value_of_computation():
    try:
        from os import nice as os_nice
        if os_nice(0) < 2:
            nice_level = os_nice(10)
    except ImportError:  # We are on a windows machine
        import os
        import psutil
        p = psutil.Process(os.getpid())
        # p.nice(10)  # Unix
        p.nice(psutil.IDLE_PRIORITY_CLASS)  # Windows  BELOW_NORMAL_PRIORITY_CLASS


def _make_dict_hashable(di: dict):
    def _default(o):
        if hasattr(o, 'dtype') and not isinstance(o, np.ndarray):
            if 'int' in str(o.dtype):
                return int(o)
            elif 'float' in str(o.dtype):
                return float(o)
        raise TypeError(str(type(o)))

    return json.dumps(di, sort_keys=True, separators=(',', ':'), default=_default)


def hash_(obj: Any):
    try:
        hash_val = joblib.hash(obj=obj, hash_name='md5')
    except Exception:
        obj = _make_dict_hashable(obj)
        hash_val = joblib.hash(obj=obj, hash_name='md5')
    return hash_val


def get_questionnaire_dummy_data(num_users: int, num_prototypes: int):

    prototypes_sus_data = generate_dummy_questionnaire_data(mode='sus', num_users=num_users,
                                                            num_prototypes=num_prototypes)
    sus_scores = evaluate_sus_questionnaire_data_per_user(prototypes_sus_data)
    if num_prototypes == 1:
        sus_scores = [sus_scores]

    prototypes_attrakdiff_data = generate_dummy_questionnaire_data(mode='attrakdiff', num_users=num_users,
                                                                   num_prototypes=num_prototypes)
    _, category_names, *_ = prototypes_attrakdiff_data
    attrakdiff_scores = evaluate_attrakdiff_questionnaire_data_per_user(*prototypes_attrakdiff_data)
    if num_prototypes == 1:
        attrakdiff_scores = [attrakdiff_scores]

    columns = [*category_names, 'HQ']

    return sus_scores, attrakdiff_scores, columns


def get_data(input_file_path='./data/cache/correlation_raw_data.json'):
    """
    (1) AttrakDiff-2
     - "PQ": Pragmatic quality, intuition: How efficient is the system?
     - "HQ-S": Hedonic stimulus, intuition: How novel and interesting is the system?
     - "HQ-I": Hedonic identification: How socially accepted is the system?
     - "HQ": Hedonic quality computed by (HQ-S + HQ-I)/2
     - "ATT": Attractiveness

    (2) "SUS": Scalar value to measure overall usability

    (3) Log Data
     - "Σwtime": list of wall-clock time differences between interactions
     - "Σctime": list of computation times after each interaction
     - "RAVD": Relative absolute volume difference
     - "Mean(Σwtime/Σctime)": Mean value over all elemt-wise division of wall-clock time and computation time per interaction
     - "Median(Median_wtime)": the median value (over all 3 data sets) of the median value (over all interactions regarding one data set) of the wall-clock time

    :param input_file_path: path to the file providing the input data
    :return:
    """

    # Load users' interaction log data
    log.info(f'Load data from "{input_file_path}"')
    with Path(input_file_path).open('r') as fp:
        data_from_log = json.load(fp)

    # Example:
    # data_from_log == {"Mean(&#931;ctime)":{"feedback_interactive":{"TestUser1":1.8039599878247827}},
    #                   "Mean(&#931;ctime)/Mean(&#931;wtime)":{"feedback_interactive":{"TestUser1":3.8389887058549765e-08}},
    #                   ...}

    log_data_labels, log_data = zip(*data_from_log.items())
    log_data_labels, log_data = list(log_data_labels), list(log_data)
    user_ids = sorted({user_id for feature in log_data for user_data in feature.values()
                       for user_id in user_data.keys()})
    prototype_ids = sorted(log_data[0].keys())
    num_users = len(user_ids)
    num_prototypes = len(prototype_ids)

    log.info(f'Found the interaction logs from {num_prototypes} prototype(s) by {num_users} user(s)')

    # Load questionnaires' result data # TODO add your data here
    sus_scores, attrakdiff_scores, attributes_attrakdiff = get_questionnaire_dummy_data(num_users, num_prototypes)
    #
    attributes_sus = [u'SUS']
    attributes_log = [c.replace('&#931;', u'Σ') for c in log_data_labels]
    attributes = [*attributes_attrakdiff, *attributes_sus, *attributes_log]

    data_attrakdiff_all_prototypes = [[{k: v for k, v in zip(attributes_attrakdiff, vals)} for vals in prot_data]
                                      for prot_data in attrakdiff_scores]
    data_sus_all_prototypes = [[{u'SUS': v} for v in prot_data] for prot_data in sus_scores]
    #

    data_logs_all_prototypes = [[{feature_label: one_labels_data[prot][usr] for feature_label, one_labels_data in
                                  zip(log_data_labels, log_data)} for usr in user_ids]
                                for prot in prototype_ids]

    data = [[*a_.values(), *s_.values(), *l_.values()] for at, su, lo in
            zip(data_attrakdiff_all_prototypes, data_sus_all_prototypes, data_logs_all_prototypes)
            for a_, s_, l_ in zip(at, su, lo)]

    return attributes, data


def feature_select(x, feature_importances, n_top_features=20):
    if n_top_features < 1:
        n_top_features = int(n_top_features * feature_importances.size)
    indices = np.argsort(feature_importances)[-1:-(n_top_features + 1):-1]
    idx_array = np.zeros(feature_importances.size, dtype=np.bool)
    idx_array[indices] = True
    return x[:, idx_array]


def impute_and_scale(x):
    if np.count_nonzero(np.isnan(x)) > 0:
        log.info('Imputing NaN values')
        si = SimpleImputer(missing_values=np.nan, strategy='mean')
        x = si.fit_transform(x)

    if np.count_nonzero(np.isnan(x)) > 0:
        si = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        x = si.fit_transform(x)

    sc = StandardScaler().fit(x)
    x = sc.transform(x)
    return x, sc


def predict_log_data(
        input_file_path='./data/cache/correlation_raw_data.json',
        out_dir: Union[Path, str] = './data/results',
        remove_questionnaire_features: bool = True,
        keep_sus_as_feature: bool = False,  # May be used for AttrakDiff-only prediction
        add_features_from_pca: Union[int, float] = 0.1,  # Add N (int or frac) additional PCA features to X_{train|test}
        use_feature_selection: bool = True,
        use_num_features: int = 20,  # Note: only utilized if 'use_feature_selection'
        relative_test_size: float = 0.1,
        random_state: int = 42):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f'Load data from "./best_features.json"')
    with Path('best_features.json').open('r') as fp:
        # With 128 trees, best 1%
        feature_selection_columns = json.load(fp)
    feature_selection_columns = {k: [e.replace('\u03a3', u'Σ') for e in v] for
                                 k, v in feature_selection_columns.items()}

    if use_feature_selection:
        assert use_num_features <= len(feature_selection_columns['PQ'])

    y_labels = ('PQ', 'ATT', 'HQ-I', 'HQ-S', 'HQ', 'SUS')
    questionnaire_labels = [l.lower() for l in y_labels]

    if keep_sus_as_feature:
        questionnaire_labels = [l for l in questionnaire_labels if l.upper() != 'SUS']
        y_labels = [l for l in y_labels if l.upper() != 'SUS']

    attributes, data = get_data(input_file_path)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    for y_label in y_labels:

        log.info(f'# {y_label.upper()}')

        y_attribute = [y_label.lower() == a.lower() for a in attributes]

        if remove_questionnaire_features:
            x_attributes = [(a.lower() not in questionnaire_labels) for a in attributes]
        else:
            x_attributes = np.invert(y_attribute).tolist()

        X = np.array(data)[:, x_attributes]
        y = np.array(data)[:, y_attribute].ravel()

        if 0 < add_features_from_pca:
            if 1 > add_features_from_pca:
                n_components_pca = int(np.ceil(add_features_from_pca * X.shape[1]))
            else:
                n_components_pca = add_features_from_pca

            if n_components_pca > min(*X.shape):
                log.warn(f'Could not compute PCA with {n_components_pca} components, since input X {X.shape} has ' +
                         f'not enough data. Using {min(*X.shape)} components instead.')
                n_components_pca = min(*X.shape)

            X, _ = impute_and_scale(X)
            pca = PCA(n_components=n_components_pca, svd_solver='full')
            pca.fit(X)
            X_pca = pca.transform(X)
            log.debug(f'X_pca has shape {X_pca.shape}')
            X = np.concatenate((X, X_pca), axis=1)

            pca_attributes = [f'PCA_VAL_{i}' for i in range(X_pca.shape[1])]
            attributes_including_pca = [*attributes, *pca_attributes]
            log.info(f'Additional PCA features: {X_pca.shape[1]}')
            x_attributes.extend([True] * X_pca.shape[1])
            del X_pca

        df = pd.DataFrame(data=X, columns=np.array(attributes_including_pca)[x_attributes])
        # log.debug(df.head())
        # log.debug(df.describe(include='all'))

        if use_feature_selection:
            # Check if all features (PCA) are present
            feature_selection_columns = {k: [v_ for v_ in v if v_ in attributes_including_pca]
                                         for k, v in feature_selection_columns.items()}

            feature_names = feature_selection_columns[y_label.upper()][:use_num_features]

            df = df.loc[:, feature_names]
            # log.info(df.head())
            # log.info(df.describe(include='all'))

            log.info(f'Features before selection: {np.count_nonzero(x_attributes)}')
            x_attributes = [(x_attr and (atrr in feature_names))
                            for x_attr, atrr in zip(x_attributes, attributes_including_pca)]
            log.info(f'Features after selection: {np.count_nonzero(x_attributes)}')
            log.info(str([atrr for x_attr, atrr in zip(x_attributes, attributes_including_pca)
                          if (x_attr and (atrr in feature_names))]))
            X = df
            # X = X.as_matrix()

        X, scaler = impute_and_scale(X)

        log.debug(f'Number of train/test splits is {X.shape[0]}')
        kf = KFold(n_splits=X.shape[0], shuffle=False, random_state=random_state)

        log.info(f'Load data from "./best_parameters.json"')
        with Path('best_parameters.json').open('r') as fp:  # With 128 trees, best 1%
            parameters = json.load(fp)[y_label]
        parameters.pop('random_state', '<dummy/>')

        if X.shape[0] < 10 and parameters['min_samples_leaf'] > 1:
            log.warn(f'GBRF parameter "min_samples_leaf" is originally set to {parameters["min_samples_leaf"]}, ' +
                     f'however only {X.shape[0]} samples are in the overall input data. ' +
                     'Therefore, "min_samples_leaf" are set to 1 during training.')
            parameters['min_samples_leaf'] = 1

        log.info(f'Best parameters: {parameters}')

        for iteration_num, (train_indices, test_indices) in tqdm(enumerate(kf.split(X=X, y=y))):
            assert test_indices.size == 1
            X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]

            # Speed improvement
            X_train = np.asfortranarray(X_train, dtype=np.float64)
            y_train = np.ascontiguousarray(y_train, dtype=np.float32)
            X_test = np.asfortranarray(X_test, dtype=np.float64)
            y_test = np.ascontiguousarray(y_test, dtype=np.float32)

            x_feature_labels = feature_names if use_feature_selection else \
                [a for a, l in zip(attributes_including_pca, x_attributes) if l]
            assert len(x_feature_labels) == X_train.shape[1]

            log.info(f'y size: {y.shape} -> y_train size: {y_train.shape}')
            log.info(f'y label: {", ".join([a for a, l in zip(attributes_including_pca, y_attribute) if l])}')
            log.info(f'features: {len(x_feature_labels)}')

            # Set seed for reproducibility
            np.random.seed(random_state + iteration_num)
            parameters.update({'random_state': random_state + iteration_num})

            additional_hash = ''
            if use_feature_selection:
                additional_hash = f'-{hash_(feature_selection_columns)}'
            save_file = f'gbrf-label_{y_label.upper()}-PCA_{add_features_from_pca}-' + \
                        f'SUS_{keep_sus_as_feature}-FEATSEL_{use_feature_selection}-' + \
                        f'{test_indices[0]}-{iteration_num}-{hash_(parameters)}-{additional_hash}.pkl'
            save_file = Path(out_dir).joinpath(save_file)
            log.info(f'Current model\'s save file: "{save_file}"')

            if save_file.is_file():
                log.info('Load model from save file')
                try:
                    sf = joblib.load(filename=save_file)
                except AttributeError as ex:
                    log.error('You probably used another version of sklearn or Python to pickle this.')
                    raise ex
                model = sf['model']
            else:
                log.info('Fit new model')
                model = GradientBoostingRegressor(**parameters)
                model.fit(X_train, y_train)

                value = {'model': model, 'parameters': parameters,
                         'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
                         'X_feature_labels': x_feature_labels, 'y_label': y_label,
                         'additional_parameters': {
                             'relative_test_size': relative_test_size,
                             'remove_quest_features': remove_questionnaire_features,
                             'keep_sus_as_feature': keep_sus_as_feature,
                             'add_features_from_pca': add_features_from_pca,
                             'use_feature_selection': use_feature_selection,
                             'use_num_features': use_num_features,
                             'random_state': random_state + iteration_num,
                             'scaler': scaler,
                         }}
                joblib.dump(value=value, filename=save_file, compress=9, protocol=pickle.HIGHEST_PROTOCOL)

            y_pred = model.predict(X_test)

            for score_func in (explained_variance_score, mean_absolute_error, mean_squared_error,
                               median_absolute_error, r2_score):
                log.info(f'{score_func.__name__} {score_func(y_test, y_pred)}')


if __name__ == '__main__':
    predict_log_data()
