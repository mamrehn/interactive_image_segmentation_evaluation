__author__ = 'Mario Amrehn'

from typing import Tuple

import numpy as np
from zenlog import log

from evaluation.cython.growcut_cy import growcut_cython


def grow_cut(image: np.ndarray, seeds: np.ndarray, max_iter: int = 200, window_size: int = 3
             ) -> Tuple[np.ndarray, np.ndarray, int]:

    if 'uint16' != str(image.dtype):
        raise ValueError(f'"seed_sized_input_image" should be of dtype uint16, but is {image.dtype}')

    cell_changes = np.zeros_like(seeds, dtype=np.uint16)
    image_min_value = np.amin(image)
    image_max_value = np.amax(image)

    if 'int8' != str(seeds.dtype):
        if seeds.dtype in (np.bool, np.uint8):
            seeds = np.copy(seeds)  # Copies seed data
            seeds = seeds.view(np.int8)
            # Ensure that FG object is not on the edge of the image
            bg_label = -1
            seeds[:, (0, -1)] = seeds[(0, -1), :] = bg_label
            print('[SimpleGrowCut] added background labels at image edges')
        else:
            seeds = seeds.astype(np.int8)  # Copies seed data

    strength_map = np.empty_like(seeds, dtype=np.float64)
    strength_map[:] = (0 != seeds).view(np.int8)

    num_computed_iterations, segmentation_volume, *_ = growcut_cython(  # TODO return strength map as well?
        image=image,  # uint16
        labels_arr=seeds,  # int8
        strengths_arr=strength_map,  # float64
        cell_changes=cell_changes,  # uint16  # ref to same data as label_changes_per_cell
        max_iter=np.int32(max_iter),  # int32
        window_size=np.int32(window_size),  # int32
        image_max_distance_value=np.uint16(image_max_value - image_min_value),  # uint16
    )

    if num_computed_iterations == max_iter:
        neutral_label = 0
        log.warn('GrowCut used the given maximum of {} iterations. {} undefined labels left.'.format(
            num_computed_iterations, 'No' if np.any(segmentation_volume == neutral_label)
            else np.count_nonzero(0 == segmentation_volume)
        ))
    elif num_computed_iterations == 0:
        log.warn('GrowCut stopped after zero iterations')

    return segmentation_volume, seeds, num_computed_iterations
