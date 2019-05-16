from collections import namedtuple
from typing import List, Union

import numpy as np


def generate_dummy_questionnaire_data(mode: str, num_users: int, num_prototypes: int = 1):
    """Function to generate example data"""
    if mode.lower() == 'sus':
        if num_prototypes == 1:
            return np.random.randint(low=1, high=6, size=(num_users, 10), dtype=np.int8).tolist()
        return np.random.randint(low=1, high=6, size=(num_prototypes, num_users, 10), dtype=np.int8).tolist()
    elif mode.lower() == 'attrakdiff':
        category_names: List[str] = ['PQ', 'ATT', 'HQ-I', 'HQ-S']
        attrakdiff_data = np.random.randint(low=1, high=8, size=(num_prototypes, num_users, 28), dtype=np.int8).tolist()
        if len(attrakdiff_data) == 1:
            attrakdiff_data = attrakdiff_data[0]
        categories: List[int] = [[0] * 7 + [1] * 7 + [2] * 7 + [3] * 7] * num_prototypes
        invert_score: List[bool] = [[False] * 14 + [True] * 14] * num_prototypes
        return attrakdiff_data, category_names, categories, invert_score
    else:
        raise ValueError('mode needs to be either "SUS" or "AttrakDiff"')


def get_sus_score(sus_data):
    d = np.atleast_2d(sus_data) - 1  # dims: (num_subjects, 10_questions)
    d[:, 1::2] = 4 - d[:, 1::2]
    d = np.mean(d, axis=0)
    return 2.5 * np.sum(d)


def normalize_attrakdiff_ratings(attrakdiff_data: List[List[int]], category_names: List[str],
                                 categories: List[int], invert_score: List[bool]):
    attrakdiff_data = np.atleast_2d(attrakdiff_data)
    invert_score = np.squeeze(np.array(invert_score, dtype=np.bool))
    attrakdiff_data[:, invert_score] = 8 - attrakdiff_data[:, invert_score]
    del invert_score

    categories = np.squeeze(categories)
    categs = {na: attrakdiff_data[:, i == categories] for i, na in enumerate(category_names)}

    categs = dict(sorted(categs.items(), key=lambda t: category_names.index(t[0])))
    return attrakdiff_data, categs


def get_attrakdiff_score(attrakdiff_data: List[List[int]],
                         category_names: List[str] = ['PQ', 'ATT', 'HQ-I', 'HQ-S'],
                         categories: List[int] = [[0] * 7, [1] * 7, [2] * 7, [3] * 7],
                         invert_score: List[bool] = [[False] * 14, [True] * 14]):

    attrakdiff_data = np.atleast_2d(attrakdiff_data)
    attrakdiff_data, categs = normalize_attrakdiff_ratings(
        attrakdiff_data, category_names, categories, invert_score)

    res = [np.mean(categs[c]) for c in category_names]
    res.append(np.mean((categs['HQ-I'] + categs['HQ-S']) / 2))

    category_names_keyword_safe = [c.replace('-', '_') for c in category_names]
    category_names_keyword_safe.append('HQ')
    AttrakDiffScore = namedtuple('attrakdiff_score_for_user', category_names_keyword_safe)
    return AttrakDiffScore(*res)


def evaluate_sus_questionnaire_data_per_user(data: Union[List[List[int]], List[List[List[int]]]]):
    data = np.array(data)
    if data.ndim == 2:
        return [get_sus_score(d) for d in data]
    elif data.ndim == 3:
        return [evaluate_sus_questionnaire_data_per_user(d) for d in data]
    else:
        raise ValueError('data needs to be either 2-D or 3-D')


def evaluate_attrakdiff_questionnaire_data_per_user(data: Union[List[List[int]], List[List[List[int]]]],
                                                    category_names: List[str], categories: List[int],
                                                    invert_score: List[bool]):
    data = np.array(data)
    if data.ndim == 2:
        return [get_attrakdiff_score(d, category_names, categories, invert_score) for d in data]
    elif data.ndim == 3:
        return [evaluate_attrakdiff_questionnaire_data_per_user(d, category_names, c, i)
                for d, c, i in zip(data, categories, invert_score)]
    else:
        raise ValueError('data needs to be either 2-D or 3-D')


if __name__ == '__main__':

    prototypes_sus_data = generate_dummy_questionnaire_data(mode='sus', num_users=2, num_prototypes=3)
    print('# SUS')
    print(*evaluate_sus_questionnaire_data_per_user(prototypes_sus_data), sep='\n', end='\n\n')

    prototypes_attrakdiff_data = generate_dummy_questionnaire_data(mode='attrakdiff', num_users=2, num_prototypes=3)
    print('# Attrakdiff')
    print(*evaluate_attrakdiff_questionnaire_data_per_user(*prototypes_attrakdiff_data), sep='\n')
