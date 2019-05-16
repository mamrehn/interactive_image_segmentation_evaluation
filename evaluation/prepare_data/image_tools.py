
__author__ = 'Mario Amrehn'

from typing import List, Tuple, Union, Iterable

import numpy as np


class ImageTools:
    def __init__(self):
        pass

    @staticmethod
    def bresenham_line_interpolation_indices(start: Union[Tuple[int], List[int]],
                                             end: Union[Tuple[int], List[int]]):
        """Bresenham's line algorithm for 2-D coordinates/indices
        Produces a list of index tuples from start to end (including start and end points)

        >>> points1 = bresenham_line_interpolation_indices((0, 0), (3, 4))
        >>> points2 = bresenham_line_interpolation_indices((3, 4), (0, 0))
        >>> assert(set(points1) == set(points2))
        >>> print points1
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
        >>> print(points2)
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = dx // 2
        y_step = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points

    @staticmethod
    def rgb_to_grayscale(rgb_image, mode='BT.709-6'):
        """
        convert RGB color image to gray scale
        :param rgb_image: ndarray (*d, 3) with last dimension equal to 3 for the color information
        :param mode: str either 'BT.709-6' (modern default), 'BT.709-1', or 'BT.601'
        :return: ndarray (*d) with one dimension less than the input image, representing gray values.
        """

        if 3 != rgb_image.shape[-1] != 4:
            raise ValueError('rgb_to_grayscale detected an invalid image as input')

        if mode == 'BT.709-1':
            # ITU-R BT.709-1 recommendation from 1993: Y = 0.2125 R + 0.7154 G + 0.0721 B, outdated
            return np.dot(rgb_image[..., :3], [0.2125, 0.7154, 0.0721])  # (for HDTV color space)
        elif mode == 'BT.709-6':
            # ITU-R BT.709-6 recommendation from 2015: Y = 0.2126 R + 0.7152 G + 0.0722 B.
            return np.dot(rgb_image[..., :3], [0.2126, 0.7152, 0.0722])  # (for HDTV color space)
        elif mode == 'BT.601':
            # ITU-R BT.601 recommendation for PAL/NTSC color space
            return np.dot(rgb_image[..., :3], [0.2989, 0.587, 0.114])
        else:
            raise NotImplementedError(mode)
