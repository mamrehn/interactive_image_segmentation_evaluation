__author__ = 'Mario Amrehn'

import json
from pathlib import Path
from typing import Dict

import numpy as np
from skimage.draw import line
from zenlog import log

from evaluation.prepare_data.metrics import Metrics
from evaluation.prepare_data.image_tools import ImageTools
from evaluation.prepare_data.grow_cut import grow_cut
from evaluation.prepare_data.tools import dict_generator, load_ground_truth_data, normalize_data
from evaluation.prepare_data.tools import load_json_data, load_json_cache, save_json_cache
from evaluation.prepare_data.extract_log_per_user import extract_log_per_user
from evaluation.prepare_data.extract_features_from_log import extract_features_per_user


def filter_null_data(dat):
    to_del = []
    for prot, v in dat.items():
        for usr, v_ in v.items():
            if (isinstance(v_, (list, tuple)) and len(v_) is 0) or v_ is None:
                to_del.append((prot, usr))
    for prot, usr in to_del:
        del dat[prot][usr]
    return dat


def seeds_to_metrics(seeds: Dict[str, Dict[str, Dict[str, list]]],
                     gt_data: Dict[str, tuple], image_ids: list,
                     load_from: str = 'seeds_to_metrics.json'):
    # image_ids == [['feedback_interactive', 'UserName1', '2019-04-10T14:24:44', '2448185', 'data_set', 'DataID1'], ...]

    load_ = Path(__file__).with_name(load_from)
    if load_.exists():
        with load_.open(mode='r') as fp:
            log.info(f'Reading file {load_}')
            metrics_results = json.load(fp)
    else:
        metrics_results = {}
    for prot, v in seeds.items():
        if prot not in metrics_results:
            metrics_results[prot] = {}
        for usr, v_ in v.items():
            if usr not in metrics_results[prot]:
                metrics_results[prot][usr] = {}
            for expe, seeds_list in v_.items():
                if expe not in metrics_results[prot][usr]:
                    image_id = [e[5] for e in image_ids if e[0] == prot and e[1] == usr and e[2] == expe]
                    if len(image_id) is 0:
                        print(f'[Warning] no ID found for {(prot, usr, expe)} in {image_ids}')
                        continue
                    image_id = image_id[0]
                    try:
                        img, gt = gt_data[image_id]
                    except KeyError:
                        print(f'[Warning] data ID not found, skipping "{image_id}"')
                        continue
                    metrics_ = []

                    current_seeds = np.zeros_like(img, dtype=np.int8)
                    current_seeds[(0, -1), :] = -1
                    current_seeds[:, (0, -1)] = -1

                    assert 'uint' in str(img.dtype), f'{image_id}, {img.dtype}, {img.shape}'
                    img = img.astype(np.uint16, copy=False)

                    met = Metrics()
                    met.set_multiple_inputs((gt, np.zeros_like(current_seeds, dtype=np.int8) - 1))
                    metrics_result = met.get_outcome()
                    metrics_.append(metrics_result)

                    for seeds_ in seeds_list:
                        if len(seeds_['bg']) > 0:
                            prev_w, prev_h = seeds_['bg'][0]
                            for i, (w_, h_) in enumerate(seeds_['bg']):
                                try:
                                    rr, cc = line(prev_h, prev_w, h_, w_)
                                    current_seeds[rr, cc] = -1
                                    prev_w, prev_h = w_, h_
                                except IndexError as ex:
                                    log.warn(ex)
                                    if len(seeds_['bg']) > (i + 1):
                                        prev_w, prev_h = seeds_['bg'][i + 1]
                        if len(seeds_['fg']) > 0:
                            prev_w, prev_h = seeds_['fg'][0]
                            for i, (w_, h_) in enumerate(seeds_['fg']):
                                try:
                                    rr, cc = line(prev_h, prev_w, h_, w_)
                                    current_seeds[rr, cc] = 1
                                    prev_w, prev_h = w_, h_
                                except IndexError as ex:
                                    log.warn(ex)
                                    if len(seeds_['bg']) > (i + 1):
                                        prev_w, prev_h = seeds_['bg'][i + 1]

                        segmentation_mask, *_ = grow_cut(img, current_seeds,
                                                         max_iter=2 ** 11, window_size=3)
                        met = Metrics()
                        met.set_multiple_inputs((gt, segmentation_mask))
                        metrics_result = met.get_outcome()
                        metrics_.append(metrics_result)

                    metrics_results[prot][usr][expe] = {image_id: metrics_}
    return metrics_results


def main(input_file_name: str = 'user_study_feedback.json'):

    rd = load_json_data(input_file_name)
    recorded_data = normalize_data(rd)
    del rd

    gt_data = load_ground_truth_data()

    # DATA:
    # {
    #   "1313880038" : {
    #     "2019-05-19T15:03:06" : {
    #       "1790000" : {
    #         "data_set" : "ceramic",
    #         "draw_mode" : "foreground",
    #         "grow_cut_accuracy" : 2.9802322387695312E-8,
    #         "segmentation_method" : "grow_cut",
    #         "type" : "start",
    #         "user_name" : "TestUser1",
    #       },
    #       "2223000" : {
    #         "type" : "slider_w",
    #         "value" : 255
    #       },
    #       "2224000" : {
    #         "type" : "transparency_btn",
    #         "value" : 1
    #       },
    #       "4997000" : {
    #         "type" : "time_spent_segmenting",
    #         "value" : 2702000
    #       },
    #       "8989000" : {
    #         "h" : 33,
    #         "label" : 1,
    #         "type" : "seed",
    #         "w" : 64
    #       },
    #       ...

    recorded_data_l = list(dict_generator(recorded_data))
    # Filter out some of the meta information per experiment
    recorded_data_l = [e for e in recorded_data_l if 'info' != e[2]]
    recorded_data_l.sort(key=lambda e: (*e[:3], int(e[3]) if len(e) > 3 else 0))  # Sort by timestamp

    segmentations_per_user = {name: len(dat) for prot, v in recorded_data.items() for name, dat in v.items()}

    log.info('Segmentations per user: ' + str(sorted(segmentations_per_user.items(), key=lambda x: -x[1])))

    prototypes = {e[0] for e in recorded_data_l}
    users = {e[1] for e in recorded_data_l}
    # experiments = {e[2] for e in recorded_data_l}
    # timestamps = {e[3] for e in recorded_data_l if not isinstance(e[3], list)}
    # events = {e[4] for e in recorded_data_l if len(e) > 4 and not isinstance(e[4], list)}
    # values = {tuple() if isinstance(e[5], list) else e[5] for e in recorded_data_l if len(e) > 5}

    # How much variance in seed point locations?
    seeds_placed = {k: {k_: {'fg': [], 'bg': [], 'all': []} for k_ in users} for k in prototypes}

    # Structure:
    # {k: {k_: {k__: [{'fg': [], 'bg': []}] for k__ in experiments} for k_ in users} for k in prototypes}
    seeds_per_interaction = {}

    # Structure:
    # ['feedback_interactive', 'UserID1', '2019-05-10T14:24:44', '2448185', 'data_set', 'b_584']
    image_ids = [e for e in recorded_data_l if 'data_set' == e[4]]

    interp_line = ImageTools.bresenham_line_interpolation_indices

    for e in recorded_data_l:
        if len(e) < 5:
            continue
        prot, usr, expe, timest, evnt, *val = e

        #
        if prot not in seeds_per_interaction:
            seeds_per_interaction[prot] = {usr: {expe: [{'fg': [], 'bg': []}]}}
        elif usr not in seeds_per_interaction[prot]:
            seeds_per_interaction[prot][usr] = {expe: [{'fg': [], 'bg': []}]}
        elif expe not in seeds_per_interaction[prot][usr]:
            seeds_per_interaction[prot][usr][expe] = [{'fg': [], 'bg': []}]
        #

        if 'type' == evnt:
            assert isinstance(val, list) and len(val) is 1
            if 'seed' == val[0]:
                r = recorded_data[prot][usr][expe][timest]
                label = 'fg' if r['label'] > 0 else 'bg'
                seeds_placed[prot][usr][label].append([r['w'], r['h']])
                seeds_placed[prot][usr]['all'].append([r['w'], r['h']])
                try:
                    old_w, old_h = seeds_per_interaction[prot][usr][expe][-1][label][-1]
                    old_timest = seeds_per_interaction[prot][usr][expe][-1].get(f'{label}_timest', 0)
                except IndexError:
                    # Init new interaction
                    if 0 == len(seeds_per_interaction[prot][usr][expe]) or \
                            len(seeds_per_interaction[prot][usr][expe][-1]['fg']) > 0 or \
                            len(seeds_per_interaction[prot][usr][expe][-1]['bg']) > 0:
                        seeds_per_interaction[prot][usr][expe].append({'fg': [], 'bg': []})
                else:  # "if no IndexError exception occurs"
                    if len(seeds_per_interaction[prot][usr][expe][-1][label][-1]) is 3:
                        seeds_per_interaction[prot][usr][expe][-1][label][-1] = [old_w, old_h]
                    if (int(timest) - old_timest) > 18000:  # 180ms (milliseconds)
                        # Eliminate duplicate seed entries in list
                        seen = set()
                        seeds_per_interaction[prot][usr][expe][-1]['fg'] = \
                            [x for x in seeds_per_interaction[prot][usr][expe][-1]['fg'] if
                             not (tuple(x) in seen or seen.add(tuple(x)))]
                        seen = set()
                        seeds_per_interaction[prot][usr][expe][-1]['bg'] = \
                            [x for x in seeds_per_interaction[prot][usr][expe][-1]['bg'] if
                             not (tuple(x) in seen or seen.add(tuple(x)))]

                        # Add points on line in-between seed points
                        interp_seeds_in_between = []
                        fg_seeds = seeds_per_interaction[prot][usr][expe][-1]['fg']
                        for start, stop in zip(fg_seeds, fg_seeds[1:]):
                            interp_seeds_in_between.extend(interp_line(start, stop)[1:-1])
                        fg_seeds.extend(interp_seeds_in_between)

                        interp_seeds_in_between = []
                        bg_seeds = seeds_per_interaction[prot][usr][expe][-1]['bg']
                        for start, stop in zip(bg_seeds, bg_seeds[1:]):
                            interp_seeds_in_between.extend(interp_line(start, stop)[1:-1])
                        bg_seeds.extend(interp_seeds_in_between)

                        # Init new interaction
                        if 0 == len(seeds_per_interaction[prot][usr][expe]) or \
                                len(seeds_per_interaction[prot][usr][expe][-1]['fg']) > 0 or \
                                len(seeds_per_interaction[prot][usr][expe][-1]['bg']) > 0:
                            seeds_per_interaction[prot][usr][expe].append({'fg': [], 'bg': []})

                seeds_per_interaction[prot][usr][expe][-1][label].append([r['w'], r['h']])
                seeds_per_interaction[prot][usr][expe][-1][f'{label}_timest'] = int(timest)

        elif 'labels' == evnt:
            if isinstance(val, list) and len(val) is 2:
                fg_seeds = [] if not val[1] else [int(k) for k in val[0].split(',')]
                bg_seeds = [] if val[1] else [int(k) for k in val[0].split(',')]
            else:
                # E.g.: fg_seeds == [[0, 1], [3, 2], ...]
                fg_seeds = [[int(k_) for k_ in k.split(',')] for k, v in val[0].items() if v]
                bg_seeds = [[int(k_) for k_ in k.split(',')] for k, v in val[0].items() if not v]

            if len(fg_seeds) > 0:
                if len(fg_seeds) > 1:
                    interp_seeds_in_between = []
                    for start, stop in zip(fg_seeds, fg_seeds[1:]):
                        interp_seeds_in_between.extend(interp_line(start, stop))
                    log.info([[fg_seeds], [interp_seeds_in_between]])
                    fg_seeds.extend(interp_seeds_in_between)
                seeds_placed[prot][usr]['fg'].append(fg_seeds)
            if len(bg_seeds) > 0:
                if len(bg_seeds) > 1:
                    interp_seeds_in_between = []
                    for start, stop in zip(bg_seeds, bg_seeds[1:]):
                        interp_seeds_in_between.extend(interp_line(start, stop))
                    log.info([[bg_seeds], [interp_seeds_in_between]])
                    bg_seeds.extend(interp_seeds_in_between)
                seeds_placed[prot][usr]['bg'].append(bg_seeds)
            seeds_placed[prot][usr]['all'].append([*fg_seeds, *bg_seeds])

    # How much variance in seed point locations?
    seeds_placed_std = {k: {k_: {'fg': [], 'bg': [], 'all': []} for k_ in users} for k in prototypes}
    seeds_placed_med = {k: {k_: {'fg': [], 'bg': [], 'all': []} for k_ in users} for k in prototypes}
    for e in list(dict_generator(seeds_placed)):
        prot, usr, evnt, val = e
        if len(val) is 0:
            continue
        w, h = map(list, zip(*val))
        seeds_placed_std[prot][usr][evnt] = np.std(w), np.std(h)
        seeds_placed_med[prot][usr][evnt] = np.median(w), np.median(h)

    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    log.info('All seed locations: ' + '\n'.join(map(str, seeds_placed.items())))
    log.info('Seeds\' stds: ' + '\n'.join(map(str, seeds_placed_std.items())))
    log.info('Seeds\' medians: ' + '\n'.join(map(str, seeds_placed_med.items())))
    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # how much variance in seed point locations?
    seeds_rel_std = {k: {k_: None for k_ in users} for k in prototypes}
    seeds_rel_med = {k: {k_: None for k_ in users} for k in prototypes}

    for s, a in ((seeds_rel_std, seeds_placed_std), (seeds_rel_med, seeds_placed_med)):
        for prot, v in s.items():
            for usr in v:
                v[usr] = tuple(np.array(a[prot][usr]['fg']) / np.array(a[prot][usr]['all']))  # v[usr]

    # Filter all users without actual user data
    seeds_rel_std = filter_null_data(seeds_rel_std)
    seeds_rel_med = filter_null_data(seeds_rel_med)

    log.info('Seeds\' relative stds: ' + '\n'.join(map(str, seeds_rel_std.items())))
    log.info('Seeds\' relative medians: ' + '\n'.join(map(str, seeds_rel_med.items())))

    save_json_cache(data=seeds_per_interaction, file_name='seeds_per_interaction.json')
    seeds_to_metrics_ = seeds_to_metrics(seeds=seeds_per_interaction, gt_data=gt_data,
                                         image_ids=image_ids, load_from='seeds_to_metrics.json')
    save_json_cache(data=seeds_to_metrics_, file_name='seeds_to_metrics.json')

    ########################

    # Writes files like "feedback_interactive - TestUser1 - 2019-05-19T15-04-31 - plot_metrics_per_test.json"
    # and               "feedback_interactive - TestUser1 - 2019-05-19T15-04-31 - meta_data_per_test.json"
    extract_log_per_user(input_file_name, outcome_dir=Path('./data/cache'))

    # Writes the file "./data/cache/correlation_raw_data.json" based on the contents of
    # the files "* - plot_metrics_per_test.json" and "* - meta_data_per_test.json"
    correlation_raw_data_file = 'correlation_raw_data.json'
    extract_features_per_user(input_dir=Path('./data/cache'), outcome_file_name=correlation_raw_data_file)

    ########################

    # Combine with previous data
    res = load_json_cache(correlation_raw_data_file)
    res['Relative_Std(Seed_Loc_W)'] = {prot: {usr: v_[0] for usr, v_ in v.items()}
                                       for prot, v in seeds_rel_std.items()}
    res['Relative_Std(Seed_Loc_H)'] = {prot: {usr: v_[1] for usr, v_ in v.items()}
                                       for prot, v in seeds_rel_std.items()}
    res['Relative_Median(Seed_Loc)_W'] = {prot: {usr: v_[0] for usr, v_ in v.items()}
                                          for prot, v in seeds_rel_med.items()}
    res['Relative_Median(Seed_Loc)_H'] = {prot: {usr: v_[1] for usr, v_ in v.items()}
                                          for prot, v in seeds_rel_med.items()}
    save_json_cache(data=res, file_name=correlation_raw_data_file)
    # /Combine with previous data


if '__main__' == __name__:
    log.info('Note: download firebase database as JSON as file ' +
             '"evaluation/data/user_study_feedback.json" before running this script')
    main('user_study_feedback.json')
