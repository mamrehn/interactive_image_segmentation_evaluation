import json
from copy import deepcopy
from pathlib import Path

import numpy as np


def mean_(interm_res, x):
    return {prot: {usr: np.nanmean(v_[x]) for usr, v_ in v.items()} for prot, v in interm_res.items()}


def median_(interm_res, x):
    return {prot: {usr: np.nanmedian(v_[x]) for usr, v_ in v.items()} for prot, v in interm_res.items()}


def mean_div_mean(interm_res, x, y):
    return {prot: {usr: np.nanmean(v_[x]) / np.nanmean(v_[y]) for usr, v_ in v.items()}
            for prot, v in interm_res.items()}


def mean_div(interm_res, x, y):
    return {prot: {usr: np.nanmean(np.array(v_[x]) / np.array(v_[y])) for usr, v_ in v.items()}
            for prot, v in interm_res.items()}


def median_div_median(interm_res, x, y):
    return {prot: {usr: np.nanmedian(v_[x]) / np.nanmedian(v_[y]) for usr, v_ in v.items()}
            for prot, v in interm_res.items()}


def median_div(interm_res, x, y):
    return {prot: {usr: np.nanmedian(np.array(v_[x]) / np.array(v_[y])) for usr, v_ in v.items()}
            for prot, v in interm_res.items()}


def capitalize_(x):
    return x.capitalize() if 'time' in x or 'interac' in x or 'dice' == x else x.upper()


def add_metric(interm_res, m):

    ml = m.lower()
    mc = capitalize_(ml)
    return {
        f'Mean({mc})': mean_(interm_res, ml),
        f'Median({mc})': median_(interm_res, ml),
    }


def add_metric_per_metric(interm_res, m1, m2='wtime'):
    m1l, m2l = m1.lower(), m2.lower()
    m1c, m2c = capitalize_(m1l), capitalize_(m2l)
    return {
        f'Mean({m1c}/{m2c})': mean_div(interm_res, m1l, m2l),
        f'Mean({m1c})/Mean({m2c})': mean_div_mean(interm_res, m1l, m2l),
        f'Median({m1c}/{m2c})': median_div(interm_res, m1l, m2l),
        f'Median({m1c})/Median({m2c})': median_div_median(interm_res, m1l, m2l),
    }


def get_data(base_dir=Path('./data/cache')):
    data = {}
    meta_data = {}

    for p in base_dir.iterdir():

        is_metrics_file = p.name.endswith('plot_metrics_per_test.json')
        is_meta_data_file = p.name.endswith('meta_data_per_test.json')

        if not (is_metrics_file or is_meta_data_file):
            continue

        prototype_id, _, user, _, test_id, *_ = p.name.split(' ')
        with p.open('r') as fp:
            fp_content = fp.read()

        d = data if is_metrics_file else meta_data
        if prototype_id not in d:
            d[prototype_id] = {user: {test_id: json.loads(fp_content)}}
        elif user not in d[prototype_id]:
            d[prototype_id][user] = {test_id: json.loads(fp_content)}
        d[prototype_id][user][test_id] = json.loads(fp_content)

    return data, meta_data


def extract_features_per_user(input_dir=Path('./data/cache'), outcome_file_name='correlation_raw_data.json'):

    d_struct = {'interactions': [],
                '&#931;wtime': [], 'mean_wtime': [], 'median_wtime': [],
                '&#931;ctime': [], 'mean_ctime': [], 'median_ctime': [],
                '&#931;otime': [], 'undos': []
                }
    struct_keys = ('ACC', 'KAP', 'F1', 'JAC', 'LOG', 'ASSD', 'HD', 'ARI', 'MI', 'HOM', 'COMPL', 'MSE', 'V_MEASURE',
                   'ROC_AUC', 'DICE', 'PRECISION', 'RECALL', 'RAVD', 'OBJ_TPR', 'OBJ_FPR', 'FPR', 'FNR')
    for sk in struct_keys:
        d_struct[sk.lower()] = []

    data, meta_data = get_data(input_dir)

    interm_res = {}
    for prot, prot_data in data.items():
        if prot not in interm_res:
            interm_res[prot] = {}
        for user_id, prot_user_data in prot_data.items():
            for data_set_time, prot_user_data_set_data in prot_user_data.items():
                if len(prot_user_data_set_data) is 0:
                    continue
                d = deepcopy(interm_res[prot].get(user_id, d_struct))
                del_keys = []

                md = meta_data[prot][user_id][data_set_time]

                for k in d:
                    if k == '&#931;otime':
                        val = md['overall_time']
                    elif k == 'interactions':
                        val = len(prot_user_data_set_data)
                    elif k == '&#931;wtime':
                        val = prot_user_data_set_data[-1]['timestamp'] - prot_user_data_set_data[0]['timestamp']
                        if val < 1:
                            continue
                    elif k == 'mean_wtime':
                        val = np.array([p['timestamp'] for p in prot_user_data_set_data])
                        val = np.mean(val[1:] - val[:-1])
                    elif k == 'median_wtime':
                        val = np.array([p['timestamp'] for p in prot_user_data_set_data])
                        val = np.median(val[1:] - val[:-1])
                    elif k == '&#931;ctime':
                        val = np.sum([p['computation_time'] for p in prot_user_data_set_data])
                    elif k == 'mean_ctime':
                        val = np.mean([p['computation_time'] for p in prot_user_data_set_data])
                    elif k == 'median_ctime':
                        val = np.median([p['computation_time'] for p in prot_user_data_set_data])
                    else:
                        try:
                            val = prot_user_data_set_data[-1]['metrics'][f'MetricEnums.{k.upper()}']
                            if val < 0.01 and k.lower() not in {'assd', 'hd', 'mse', 'ravd'}:
                                val = np.nan
                        except KeyError:
                            del_keys.append(k)
                            continue
                    d[k].append(val)
                    if len(d[k]) is 4:
                        d[k] = d[k][1:]

                for k in d:
                    if np.all(np.isnan(d[k])):
                        del_keys.append(k)

                for dk in set(del_keys):
                    del d[dk]
                interm_res[prot][user_id] = d

    res = {}
    for m in d_struct:
        try:
            res.update(add_metric(interm_res, m))
            if '&#931;wtime' != m:
                res.update(add_metric_per_metric(interm_res, m1=m, m2='&#931;wtime'))
            if '&#931;ctime' != m:
                res.update(add_metric_per_metric(interm_res, m1=m, m2='&#931;ctime'))
        except KeyError:
            continue

    with Path(input_dir).joinpath(outcome_file_name).open('w+') as fp:
        fp.write(json.dumps(res))


if __name__ == '__main__':
    extract_features_per_user()
