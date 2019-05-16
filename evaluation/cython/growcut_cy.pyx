## cython: profile=True
## distutils: language=c++
# distutils: language_level=3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__author__ = 'Mario Amrehn'

import numpy as np
import cython
from cython.parallel cimport parallel, prange

from libc.stdlib cimport malloc, free

# @cython.profile(True)
# @cython.linetrace(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef growcut_cython(unsigned short[:, ::1] image, char[:, ::1] labels_arr, double[:, ::1] strengths_arr,
                unsigned short[:, ::1] cell_changes, int max_iter=2000, int window_size=3,
                unsigned short image_max_distance_value=255, char return_strength_map=1):
    cdef:
        char[:, ::1] labels_next_arr_
        char[:, ::1] labels_arr_
        double[:, ::1] strengths_next_arr_
        double[:, ::1] strengths_arr_
        int n

    labels_next_arr_out = np.zeros_like(labels_arr)
    labels_next_arr_ = labels_next_arr_out
    strengths_next_arr_out = strengths_arr.copy()
    strengths_next_arr_ = strengths_next_arr_out

    labels_arr_ = labels_arr.copy()
    strengths_arr_ = strengths_arr.copy()

    n = growcut_cython_helper(labels_next_arr_, strengths_next_arr_, image, labels_arr_, strengths_arr_,
                              cell_changes, max_iter, window_size, image_max_distance_value)
    if 0 != return_strength_map:
        return n, labels_next_arr_out, strengths_next_arr_out
    return n, labels_next_arr_out

# @cython.profile(True)
# @cython.linetrace(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef inline int growcut_cython_helper(char[:, ::1] labels_next_arr_, double[:, ::1] strengths_next_arr_,
                                      unsigned short[:, ::1] image, char[:, ::1] labels_arr_,
                                      double[:, ::1] strengths_arr_, unsigned short[:, ::1] cell_changes_,
                                      int max_iter, int window_size, unsigned short image_max_distance_value) nogil:
    cdef:
        char[:, ::1] labels_next_arr
        char[:, ::1] labels_arr
        double[:, ::1] strengths_next_arr
        double[:, ::1] strengths_arr
        unsigned short[:, ::1] cell_changes

        Py_ssize_t dim0_u = image.shape[0], dim1_u = image.shape[1]
        unsigned int ws_u = (window_size - 1) // 2
        Py_ssize_t dd0_u, dd1_u
        Py_ssize_t d0_u, d1_u
        Py_ssize_t dd0_min, dd1_min, dd0_max, dd1_max
        double one_div_by_max_distance_squared = (1.0 / (<double> image_max_distance_value))
        double *thread_local_double_data
        char * thread_local_char_data  # thread_local_char_data[0], changes
        int n = 0
        char * changes = <char *> malloc(sizeof(char))

    one_div_by_max_distance_squared *= one_div_by_max_distance_squared  # TODO TEST

    labels_next_arr = labels_next_arr_
    labels_arr = labels_arr_
    strengths_next_arr = strengths_next_arr_
    strengths_arr = strengths_arr_
    cell_changes = cell_changes_

    while changes[0] and n < max_iter:
        changes[0] = 0
        n += 1

        with nogil, parallel(num_threads=4):
            #
            # [attack_strength, defense_strength, defense_strength_orig, current_cell_value]
            # [0              , 1               , 2                    , 3                 ]
            thread_local_double_data = <double *> malloc(sizeof(double) * 4)
            thread_local_char_data = <char *> malloc(sizeof(char) * 2)  # winning_colony, changes
            thread_local_char_data[1] = 0

            for d0_u in prange(dim0_u, schedule='static'):
                #
                if ws_u >= d0_u:
                    dd0_min = 0
                    dd0_max = d0_u + ws_u + 1
                else:
                    dd0_min = d0_u - ws_u
                    if dim0_u > (d0_u + ws_u):  # if equal after +1, take either
                        dd0_max = d0_u + ws_u + 1
                    else:
                        dd0_max = dim0_u
                #
                for d1_u in range(dim1_u):
                    #
                    if ws_u >= d1_u:
                        dd1_min = 0
                        dd1_max = d1_u + ws_u + 1
                    else:
                        dd1_min = d1_u - ws_u
                        if dim1_u > (d1_u + ws_u):  # if equal after +1, take either
                            dd1_max = d1_u + ws_u + 1
                        else:
                            dd1_max = dim1_u
                    #
                    thread_local_double_data[1] = thread_local_double_data[2] = strengths_arr[d0_u, d1_u]  # Note: defense_strength
                    thread_local_double_data[2] += 2 ** -25  # Note: defense_strength_orig
                    thread_local_double_data[3] = image[d0_u, d1_u]  # Note: current_cell_value
                    thread_local_char_data[0] = labels_arr[d0_u, d1_u]  # Note: winning_colony


                    for dd0_u in range(dd0_min, dd0_max):
                        for dd1_u in range(dd1_min, dd1_max):
                            # Inexpensive test to reduce #computations: if cell cannot be conquered by the attacker,
                            # do not investigate further and continue with current cell's next neighbor.
                            if thread_local_double_data[2] >= strengths_arr[dd0_u, dd1_u]:
                                continue

                            # p -> current cell, (d0_u, d1_u, d2_u)
                            # q -> attacker, (dd0, dd1, dd2)
                            # attack_strength = g(distance(image, d0_u, d1_u, dd0_u, dd1_u),
                            #                     one_div_by_max_distance) * strengths_arr[dd0, dd1, dd2]
                            thread_local_double_data[0] = \
                                (1.0 - (one_div_by_max_distance_squared *
                                        (thread_local_double_data[3] - (<double> image[dd0_u, dd1_u]))**2)) * \
                                strengths_arr[dd0_u, dd1_u]

                            # differentiate here to increase cell changes independent of the neighbors' ordering
                            if thread_local_double_data[0] > thread_local_double_data[2]:
                                if thread_local_char_data[0] != labels_arr[dd0_u, dd1_u]:  # Note: winning_colony != label
                                    cell_changes[d0_u, d1_u] += 1
                                if thread_local_double_data[0] > (thread_local_double_data[1] + 2 ** -25):
                                    thread_local_double_data[1] = thread_local_double_data[0]
                                    thread_local_char_data[0] = labels_arr[dd0_u, dd1_u]  # Note: winning_colony = ...
                                    thread_local_char_data[1] = 1  # Note: changes (thread local)

                    labels_next_arr[d0_u, d1_u] = thread_local_char_data[0]  # Note: winning_colony
                    strengths_next_arr[d0_u, d1_u] = thread_local_double_data[1]  # Note: defense_strength

            # Propagate local data to outer scope
            if thread_local_char_data[1] and 0 == changes[0]:
                changes[0] = 1

            # Free thread local memory
            free(thread_local_double_data)
            free(thread_local_char_data)

        labels_next_arr, labels_arr = labels_arr, labels_next_arr
        strengths_next_arr, strengths_arr = strengths_arr, strengths_next_arr

    free(changes)
    return n
