import argparse
import numpy as np
import random
from util import negative_sampling

RATING_FILE_NAME = dict({'movie': 'ratings.dat',
                         'book': 'BX-Book-Ratings.csv',
                         'music': 'user_artists.dat'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0})
MAP_FILE_NAME = 'item_index2entity_id.txt'

def read_item_index_to_entity_id_file(dataset):
    item_index_old2new = dict()
    entity_id2index = dict()

    file = '../data/' + dataset + '/'+ MAP_FILE_NAME
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        fb_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[fb_id] = i
        i += 1
    return item_index_old2new, entity_id2index


def convert_rating(dataset, item_index_old2new):
    file = '../data/' + dataset + '/' + RATING_FILE_NAME[dataset]

    print('reading rating file ...')
    n_items = len(item_index_old2new)
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[dataset])

        # remove prefix and suffix quotation marks for BX dataset
        if dataset == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = array[0]

        rating = float(array[2])
        if rating >= THRESHOLD[dataset]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('../data/' + dataset + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))

        if user_index_old in user_neg_ratings:
            pos_inds = np.array(sorted(pos_item_set | user_neg_ratings[user_index_old]))
        else:
            pos_inds = np.array(sorted(pos_item_set))
        
        for item in negative_sampling(pos_inds, n_items, n_samp=len(pos_item_set)):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % n_items)

def convert_kg(dataset, entity_id2index):
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0
    relation_id2index = dict()

    writer = open('../data/' + dataset + '/kg_final.txt', 'w', encoding='utf-8')

    file = open('../data/' + dataset + '/' + 'kg.txt', encoding='utf-8')

    for line in file:
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)
    return relation_id2index

if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    dataset = args.dataset

    item_index_old2new, entity_id2index = read_item_index_to_entity_id_file(dataset)
    convert_rating(dataset, item_index_old2new)
    relation_id2index = convert_kg(dataset, entity_id2index)

    print('done')
