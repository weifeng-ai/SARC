import collections
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data(dataset):
    train_data, eval_data, test_data, n_user, n_item = load_rating(dataset)
    n_entity, n_relation, kg = load_kg(dataset)

    return train_data, eval_data, test_data, n_user, n_item, n_entity, n_relation, kg

def load_rating(dataset):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + dataset + '/ratings_final'
    rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np)
    return train_data, eval_data, test_data, n_user, n_item


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    train_ratio = 0.6
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    train_test_indices = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(train_test_indices), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(train_test_indices - set(test_indices))

    print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with any positive rating
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data

def load_kg(dataset):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + dataset + '/kg_final'
    kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
    print(len(kg))
    n_entity = max(set(kg[:, 0]) | set(kg[:, 2]))+1
    n_relation = len(set(kg[:, 1]))
    return n_entity, n_relation, kg
