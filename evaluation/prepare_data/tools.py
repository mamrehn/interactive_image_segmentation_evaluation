__author__ = 'Mario Amrehn'

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import numpy as np
from zenlog import log
import matplotlib.pylab as plt

from evaluation.prepare_data.image_tools import ImageTools


def dict_generator(input_dict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(input_dict, dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, [key] + pre):
                    yield d
            else:
                yield pre[::-1] + [key, value]
    else:
        yield input_dict


def update_dict_recursively(base_dict, update_dict, copy=True):
    if 0 == len(base_dict):
        return deepcopy(update_dict) if copy else update_dict
    if copy:
        base_dict = deepcopy(base_dict)
    try:
        for k, v in update_dict.items():
            if isinstance(v, Mapping):
                base_dict[k] = update_dict_recursively(base_dict.get(k, {}), v, copy=False)
            else:
                base_dict[k] = update_dict[k]
    # AttributeError: 'list' object has no attribute 'items' (occurs if update dict is empty (list))
    except AttributeError as ex:
        log.warn(f'Empty dict update added to {base_dict}')
        # raise ex
    return base_dict


def load_json_data(file_name: Union[Path, str], base_path: Union[None, Path, str] = None):
    if base_path is None:
        base_path = Path(__file__).parent.joinpath('data')
    else:
        base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    try:
        path_to_read = base_path.joinpath(file_name)
        with path_to_read.open(mode='r') as fp:
            log.info(f'Reading file {path_to_read}')
            res = json.load(fp)
    except FileNotFoundError:
        log.warning(f'File to read not found, simulating empty one for {path_to_read}.')
        return {}
    return res


def load_json_cache(file_name: Union[Path, str], base_path: Union[None, Path, str] = None):
    if base_path is None:
        base_path = Path(__file__).parent.joinpath('data', 'cache')
    return load_json_data(file_name, base_path)


def save_json_cache(data: Any, file_name: Union[Path, str], base_path: Union[None, Path, str] = None):
    if base_path is None:
        base_path = Path(__file__).parent.joinpath('data', 'cache')
    else:
        base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    path_to_write = base_path.joinpath(file_name)
    with path_to_write.open(mode='w+') as fp:
        log.info(f'Writing file {path_to_write}')
        json.dump(data, fp, sort_keys=True, separators=(',', ':'))


def load_ground_truth_data(base_path: Union[None, Path, str] = None):
    if base_path is None:
        base_path = Path(__file__).parents[1].joinpath('webapp', 'data')
    else:
        base_path = Path(base_path)

    images_ids = sorted(p.stem for p in base_path.iterdir() if p.is_file() and not p.stem.endswith('_gt'))
    images = []
    ground_truths = []

    for img_id in images_ids:
        img_name = str(base_path.joinpath(f'{img_id}.png'))
        gt_name = str(base_path.joinpath(f'{img_id}_gt.png'))

        # Read image to segment
        try:
            img_data = plt.imread(img_name)
            if np.amax(img_data) <= 1:
                log.debug(f'Image {img_id}.png value range was converted from [0, 1] to [0, 255]')
                img_data *= 255
            img_data = img_data.astype(np.uint8, copy=False)
        except FileNotFoundError:
            log.warning(f'Skipping since no file found with name {img_name}')
            images.append(None)
            ground_truths.append(None)
            continue

        if 2 < img_data.ndim:
            img_data = np.rint(ImageTools.rgb_to_grayscale(img_data.astype(np.float64))).astype(np.uint8)
            assert np.amax(img_data) > 1
        images.append(img_data)

        # Read GT image
        gt_data = plt.imread(gt_name)
        if gt_data.ndim == 3:
            gt_data = gt_data[:, :, 0]
        ground_truths.append(gt_data > 0)

    return {img_id: (img, gt) for img_id, img, gt in zip(images_ids, images, ground_truths) if img is not None}


def normalize_data(recorded_data: dict):
    try:
        int(next(iter(recorded_data.keys())))
    except (StopIteration, ValueError):
        return recorded_data

    res = {}
    for fingerprint, dat in recorded_data.items():  # == "380752037"
        # Search for user name
        break_loops = False
        for first_experiment in dat.values():
            if break_loops:
                break
            # first_experiment = dat[next(iter(dat.keys()))]  # == "2019-05-10T14:24:44"
            for first_event in first_experiment.values():
                if break_loops:
                    break
                # first_event = first_experiment[next(iter(first_experiment.keys()))]  # == "2448185"
                if 'user_name' in first_event:
                    # first_event['all_user_names'][-1]  # "all_user_names": ["UserName1"],
                    user_name = first_event['user_name']
                    break_loops = True
                    break
        # /Search for user name

        if user_name not in res:
            if user_name.lower() in map(lambda x: x.lower(), res.keys()):
                user_name = [u for u in res.keys() if u.lower() == user_name.lower()][0]
            else:
                res[user_name] = {}
        res[user_name] = update_dict_recursively(base_dict=res[user_name], update_dict=dat, copy=False)
    return {'feedback_interactive': res}
