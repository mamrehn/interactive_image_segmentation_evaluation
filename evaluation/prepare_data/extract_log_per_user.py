import time
from typing import Union
from pathlib import Path

import numpy as np
from zenlog import log

from evaluation.prepare_data.tools import load_ground_truth_data, load_json_data, load_json_cache, save_json_cache
from evaluation.prepare_data.metrics import Metrics, MetricEnums
from evaluation.prepare_data.grow_cut import grow_cut


def traverse_data_structure(input_data, funcs, outcome_data=None):
    if funcs and not isinstance(funcs, (list, iter, tuple)):
        funcs = [[], [funcs]]
    assert 0 < len(funcs) and 0 < len(input_data)
    if outcome_data is None:
        from collections import defaultdict
        outcome_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for prototype_id, tests_per_prototype in input_data.items():  # prototype_id \in {Proto1, Proto2}
        for user, test_per_user in tests_per_prototype.items():  # user \in {User1, User2, ...}
            test_per_user_sorted_and_filtered = (d for d in sorted(test_per_user.items()) if d[0] != 'info')
            for test_id, one_tests_data in test_per_user_sorted_and_filtered:  # test_id \in {2019-05-31T09:33:52, ...}
                for func in funcs[0]:
                    func(outcome_data, prototype_id, user, test_id, one_tests_data)
                for timestamp, event in one_tests_data.items():
                    args = (outcome_data, prototype_id, user, test_id, int(timestamp), event)
                    for func in funcs[1]:
                        func(*args)
                    # print(timestamp, str(one_tests_data)[:5])
    return outcome_data


def traverse_extracted_data_structure(data, funcs, gt_data, func_data=None):
    if funcs and not isinstance(funcs, (list, iter, tuple)):
        funcs = [[], [], [], [], [funcs]]
    assert 0 != sum(len(f) for f in funcs) and 0 < len(data)
    if func_data is None:
        from collections import defaultdict
        func_data = defaultdict(dict)
    for func in funcs[0]:
        func(func_data, data, gt_data)
    for prototype_id, tests_per_prototype in data.items():  # prototype_id \in {Proto1, Proto2}
        for func in funcs[1]:
            func(func_data, prototype_id, tests_per_prototype, gt_data)
        for user, test_per_user in tests_per_prototype.items():  # user \in {User1, User2, ...}
            for func in funcs[2]:
                func(func_data, prototype_id, user, test_per_user, gt_data)
            # Note: int, because of 'count_undos'
            test_per_user_sorted_and_filtered = (d for d in sorted(test_per_user.items())
                                                 if 'info' != d[0] and not isinstance(d[1], int))
            for test_id, one_tests_data in test_per_user_sorted_and_filtered:  # test_id \in {2019-05-31T09:33:52, ...}
                for func in funcs[3]:
                    func(func_data, prototype_id, user, test_id, one_tests_data, gt_data)
                if isinstance(one_tests_data, (dict, defaultdict)):
                    for aggregation_id, extracted_value in one_tests_data.items():
                        args = (func_data, prototype_id, user, test_id, aggregation_id, extracted_value, gt_data)
                        for func in funcs[4]:
                            func(*args)
    return func_data


def traverse_plot_data_structure(data, funcs, func_data=None):
    if funcs and not isinstance(funcs, (list, iter, tuple)):
        funcs = [funcs]
    assert 0 < len(funcs)
    assert 0 < len(data)
    if func_data is None:
        func_data = {}
    for func in funcs:
        func(func_data, data)
    return func_data


def visitor_func_template(func_data, prototype_id, user, test_id, timestamp, event):
    """Template for event functions
    :param func_data: outcome dictionary. Processed data is added to this structure
    :type func_data: dict
    :param prototype_id: info about the origin of the current event["data_set"] data.
    :param user: info about the origin of the current event["data_set"] data.
    :param test_id: info about the origin of the current event["data_set"] data.
    :param timestamp: info about the origin of the current event["data_set"] data.
    :param event: single event recorded during testing. The key "type". event["type"]
                  selects function to execute on event["data_set"].
    :type event: dict
    :return: None
    """
    func_identifier = 'count_undos'
    event_type = event['type']
    if event_type == 'start':
        pass
    elif event_type == 'time_spent_segmenting':
        pass
    elif event_type == 'initial_seeds':
        pass
    elif event_type == 'decision':
        pass
    elif event_type == 'seed':
        pass
    elif event_type == 'inf_loop_prevention_for_next_two_points':
        pass
    elif event_type == 'undo_btn':
        func_data[prototype_id][user][func_identifier] = \
            func_data[prototype_id][user].get(func_identifier, 0) + 1
        func_data[prototype_id][user][test_id][func_identifier] = \
            func_data[prototype_id][user][func_identifier].get(func_identifier, 0) + 1
    elif event_type == 'confirm_btn':
        pass


def visitor_func_data_set_identifier(func_data, prototype_id, user, test_id, timestamp, event):
    func_identifier = 'data_set_identifier'
    if 'start' == event['type']:
        func_data[prototype_id][user][test_id][func_identifier] = event['data_set']


def visitor_func_count_undos(func_data, prototype_id, user, test_id, timestamp, event):
    func_identifier = 'count_undos'
    event_type = event['type']
    if 'undo_btn' == event_type:
        func_data[prototype_id][user][func_identifier] = \
            func_data[prototype_id][user].get(func_identifier, 0) + 1
        func_data[prototype_id][user][test_id][func_identifier] = \
            func_data[prototype_id][user][test_id].get(func_identifier, 0) + 1


def visitor_func_list_seed_locations(func_data, prototype_id, user, test_id, timestamp, event):
    func_identifier = 'seed_locations'
    event_type = event['type']
    if 'seed' == event_type:
        if func_identifier not in func_data[prototype_id][user]:
            func_data[prototype_id][user][func_identifier] = \
                {'all': {'w': [], 'h': []}, 'bg': {'w': [], 'h': []}, 'fg': {'w': [], 'h': []}, 'times': []}
        if func_identifier not in func_data[prototype_id][user][test_id]:
            func_data[prototype_id][user][test_id][func_identifier] = \
                {'all': {'w': [], 'h': []}, 'bg': {'w': [], 'h': []}, 'fg': {'w': [], 'h': []}, 'times': []}

        for f in (func_data[prototype_id][user][func_identifier],
                  func_data[prototype_id][user][test_id][func_identifier]):
            f['all']['w'].append(event['w'])
            f['all']['h'].append(event['h'])
            label = 'fg' if event['label'] > 0 else 'bg'
            f[label]['w'].append(event['w'])
            f[label]['h'].append(event['h'])
            f['times'].append(timestamp)


def visitor_func_overall_time(func_data, prototype_id, user, test_id, timestamp, event):
    func_identifier = 'overall_time'
    event_type = event['type']
    if 'start' == event_type:
        if func_identifier in func_data[prototype_id][user][test_id]:
            # Logged "confirm_btn" processed before "start" event
            func_data[prototype_id][user][test_id][func_identifier] -= timestamp
        else:
            func_data[prototype_id][user][test_id][func_identifier] = timestamp
    elif 'confirm_btn' == event_type:
        if func_identifier in func_data[prototype_id][user][test_id]:
            # Logged "start" processed before "confirm_btn" event
            func_data[prototype_id][user][test_id][func_identifier] = \
                timestamp - func_data[prototype_id][user][test_id][func_identifier]
        else:
            func_data[prototype_id][user][test_id][func_identifier] = timestamp


def visitor_func_collect_all_seeds_per_time_instance(func_data, prototype_id, user, test_id, one_tests_data):
    func_identifier = 'collect_all_seeds_per_time_instance'
    func_data[prototype_id][user][test_id][func_identifier] = {
        'initial_seeds': {
            'bg_seeds': [], 'fg_seeds': [],
        },
        'seeds': [],
    }
    timestamp_cache = 0
    seed_cache = (-1, -1, -1, -1)  # Last seed (w, h, label, timestamp)
    for timestamp, event in sorted(one_tests_data.items(), key=lambda tup: int(tup[0])):
        event_type = event['type']
        if 'initial_seeds' == event_type:
            if event['bg_seeds']:
                func_data[prototype_id][user][test_id][func_identifier]['initial_seeds']['bg_seeds'] = \
                    event['bg_seeds']
            if event['fg_seeds']:
                func_data[prototype_id][user][test_id][func_identifier]['initial_seeds']['fg_seeds'] = \
                    event['fg_seeds']
        elif 'decision' == event_type:
            timestamp_cache = 0
        elif 'seed' == event_type:
            timestamp_int = int(timestamp)
            new_elem = (event['w'], event['h'], event['label'], timestamp_int)

            # Note: new points may have been sampled in same bin (pixel coordinate). Ignore those.
            if all(s1 == s2 for s1, s2 in zip(seed_cache[:3], new_elem[:3])):
                continue
            else:
                seed_cache = list(new_elem)

            # 2e5 == 200ms
            if (timestamp_int - timestamp_cache) < 2e5 and \
                    len(func_data[prototype_id][user][test_id][func_identifier]['seeds']) > 0 and \
                    len(func_data[prototype_id][user][test_id][func_identifier]['seeds'][-1]) > 0 and \
                    ((func_data[prototype_id][user][test_id][func_identifier]['seeds'][-1])[-1])[:3] != new_elem[:3]:
                # Add to old elem as if added in one sweep
                func_data[prototype_id][user][test_id][func_identifier]['seeds'][-1].append(new_elem)
            else:
                func_data[prototype_id][user][test_id][func_identifier]['seeds'].append([new_elem])
            timestamp_cache = timestamp_int

        elif event_type == 'undo_btn':  # bg_seeds_deleted, fg_seeds_deleted
            timestamp_int = int(timestamp)
            if 'bg_seeds_deleted' in event and 'fg_seeds_deleted' in event:
                func_data[prototype_id][user][test_id][func_identifier]['seeds'].append(
                    [(non_seed[0], non_seed[1], 0, timestamp_int) for deleted_id in
                     ('bg_seeds_deleted', 'fg_seeds_deleted') for non_seed in event[deleted_id]]
                )
            else:
                if 'bg_seeds_deleted' in event:
                    func_data[prototype_id][user][test_id][func_identifier]['seeds'].append(
                        [(non_seed[0], non_seed[1], 0, timestamp_int) for non_seed in event['bg_seeds_deleted']]
                    )
                if 'fg_seeds_deleted' in event:
                    func_data[prototype_id][user][test_id][func_identifier]['seeds'].append(
                        [(non_seed[0], non_seed[1], 0, timestamp_int) for non_seed in event['fg_seeds_deleted']]
                    )


#################################################


def visitor_func_plot_metrics_and_undos_per_test(func_data, prototype_id, user, test_id, one_tests_data, gt_data):
    # Data structure
    # one_tests_data['collect_all_seeds_per_time_instance'] == {
    #     'initial_seeds': {
    #         'bg_seeds': [('w', 'h'), ('w', 'h')], 'fg_seeds': [('w', 'h'), ('w', 'h')]
    #     },
    #     'seeds': [[('w', 'h', 'label', 'timestamp_int'), ('w', 'h', 'label', 'timestamp_int')], [('w', 'h', 'label', 'timestamp_int')]]
    # }

    test_cache_name = f'{prototype_id} - {user} - {test_id} - plot_metrics_per_test.json'.replace(':', '-')
    meta_data_cache_name = f'{prototype_id} - {user} - {test_id} - meta_data_per_test.json'.replace(':', '-')

    if len(load_json_cache(meta_data_cache_name)) == 0:
        try:
            save_file_content = one_tests_data.copy()
            if 'count_undos' in save_file_content and isinstance(one_tests_data['count_undos'], dict) \
                    and len(one_tests_data['count_undos']) == 0:
                save_file_content['count_undos'] = 0
            # {
            #     'overall_time': one_tests_data['overall_time'],  # == 349188400
            #     'data_set_identifier': one_tests_data['data_set_identifier'],  # == 'llama'
            #     'count_undos': one_tests_data['count_undos']  # == 2
            # }
            save_json_cache(save_file_content, meta_data_cache_name)
        except KeyError as e:
            print(one_tests_data)
            raise e

    metrics_results = load_json_cache(test_cache_name)

    if len(metrics_results) == 0:
        data_set_name = one_tests_data['data_set_identifier']

        # Some test data sets may not have recorded data from all prototypes tested.
        # Therefore, do not throw an error if a data set's interaction record is not present.
        if data_set_name not in gt_data:
            return

        image, gt = gt_data[data_set_name]
        assert 'uint' in str(image.dtype)
        assert np.amax(image) - np.amin(image) > 100, f'{np.amin(image)}, {np.amax(image)}'
        image = image.astype(np.uint16, copy=False)
        seed_data = one_tests_data['collect_all_seeds_per_time_instance']

        current_seeds = np.zeros_like(image, dtype=np.int8)
        if len(seed_data['initial_seeds']['bg_seeds']) is 0:
            current_seeds[(0, -1), :] = -1
            current_seeds[:, (0, -1)] = -1
        else:
            for bg_point in seed_data['initial_seeds']['bg_seeds']:
                try:
                    current_seeds[bg_point[1], bg_point[0]] = -1
                except IndexError:
                    pass
        for fg_point in seed_data['initial_seeds']['fg_seeds']:
            try:
                current_seeds[fg_point[1], fg_point[0]] = 1
            except IndexError:
                pass

        perform_next_computation = True
        old_metrics_result = None
        metrics_results = []

        if 1 < len(seed_data['seeds']):
            for seed_point_list in seed_data['seeds']:

                if perform_next_computation:
                    seeds = np.copy(current_seeds)

                    start_time = time.perf_counter()
                    segmentation_volume, *_ = grow_cut(image, seeds, max_iter=1000)
                    end_time = time.perf_counter()

                    met = Metrics()
                    met.set_multiple_inputs((gt, segmentation_volume))
                    metrics_result = met.get_outcome()
                    log.debug(
                        f'[{prototype_id} - {user} - {test_id[-9:]}] performed 2-D GC with {np.count_nonzero(seeds)} ' +
                        f'seeds ({np.count_nonzero(seeds > 0)} fg, {np.count_nonzero(seeds < 0)} bg) in ' +
                        f'{end_time - start_time}s - {metrics_result}')

                else:
                    metrics_result = old_metrics_result.copy()
                perform_next_computation = True

                # add to overall results
                metrics_results.append(
                    {
                        'metrics': metrics_result,
                        'timestamp': (seed_point_list[0])[3],  # timestamp of first point in an consecutive action
                        'num_fg_seeds': np.count_nonzero(seeds > 0),
                        'num_bg_seeds': np.count_nonzero(seeds < 0),
                        'computation_time': (end_time - start_time)
                    }
                )

                for seed_point in seed_point_list:
                    try:
                        current_seeds[seed_point[1], seed_point[0]] = seed_point[2]
                    except IndexError:
                        perform_next_computation = False
                        old_metrics_result = metrics_result

        save_json_cache(metrics_results, test_cache_name)

    # ###############

    # evaluation_metrics = ('MetricEnums.ASSD', 'MetricEnums.RAVD', 'MetricEnums.DICE')
    metric_enums = (MetricEnums.ACC, MetricEnums.KAP, MetricEnums.F1, MetricEnums.JAC, MetricEnums.LOG,
                    MetricEnums.ASSD, MetricEnums.HD, MetricEnums.ARI, MetricEnums.MI, MetricEnums.HOM,
                    MetricEnums.COMPL, MetricEnums.MSE, MetricEnums.V_MEASURE, MetricEnums.ROC_AUC, MetricEnums.DICE,
                    MetricEnums.PRECISION, MetricEnums.RECALL, MetricEnums.RAVD, MetricEnums.OBJ_TPR,
                    MetricEnums.OBJ_FPR, MetricEnums.FPR, MetricEnums.FNR)
    evaluation_metrics = tuple(map(str, metric_enums))
    del metric_enums

    plot_data = {}
    for metric_identifier in evaluation_metrics:  # metrics_results[0]['metrics'].keys():
        d = [(m_r['timestamp'], m_r['metrics'][metric_identifier]) for
             m_r in metrics_results if metric_identifier in m_r['metrics']]
        if len(d) > 0:
            plot_data[metric_identifier] = d

    for metric_identifier in evaluation_metrics:  # This preserves the intended order of the keys during iteration
        try:
            data = plot_data[metric_identifier]
        except KeyError:
            continue  # Skip if not all metrics are present for all data sets

        # Save to data structure
        try:
            func_data['overall_metrics_eval'][prototype_id][user][test_id][metric_identifier] = data.copy()
        except KeyError:
            func_data['overall_metrics_eval'] = {prototype_id: {user: {test_id: {}}}}
            func_data['overall_metrics_eval'][prototype_id][user][test_id][metric_identifier] = data.copy()


def normalize_data(recorded_data: dict):
    try:
        int(next(iter(recorded_data.keys())))
    except ValueError:
        return recorded_data

    res = {}
    for fingerprint, dat in recorded_data.items():  # == "380752037"
        # Search for user name
        first_experiment = dat[next(iter(dat.keys()))]  # == "2019-05-10T14:24:44"
        first_event = first_experiment[next(iter(first_experiment.keys()))]  # == "2448185"
        user_name = first_event['all_user_names'][-1]  # "all_user_names": ["User1"],

        if user_name not in res:
            if user_name.lower() in map(lambda x: x.lower(), res.keys()):
                user_name = [u for u in res.keys() if u.lower() == user_name.lower()][0]
            else:
                res[user_name] = {}
        res[user_name].update(dat)
    return {'feedback_interactive': res}


########################################


def extract_log_per_user(input_file_name: str = 'user_study_feedback.json', outcome_dir: Union[None, Path, str] = None):

    recorded_data = load_json_data(input_file_name)
    recorded_data = normalize_data(recorded_data)

    gt_data = load_ground_truth_data()

    visitors_for_extraction = [
        [visitor_func_collect_all_seeds_per_time_instance],  # one test
        [visitor_func_count_undos, visitor_func_overall_time,
         visitor_func_data_set_identifier]  # one event  # visitor_func_list_seed_locations
    ]
    extracted_data = traverse_data_structure(recorded_data, visitors_for_extraction)

    visitors_for_combination_and_plotting = [
        [],  # all data
        [],  # one prototype
        [],  # one user
        [visitor_func_plot_metrics_and_undos_per_test],  # one test
        []  # one event
    ]
    plot_data = traverse_extracted_data_structure(extracted_data, visitors_for_combination_and_plotting,
                                                  gt_data)  # visitors_for_combination_and_plotting

    save_json_cache(plot_data['overall_metrics_eval'], 'overall_metrics_eval.json', base_path=outcome_dir)


if '__main__' == __name__:
    log.info('Note: download firebase database as JSON as file ' +
             '"evaluation/data/user_study_feedback.json" before running this script')
    extract_log_per_user('user_study_feedback.json')
