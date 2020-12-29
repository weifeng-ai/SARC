import numpy as np
import scipy.sparse as sp
import pandas as pd
import tensorflow as tf
import collections
import copy

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def construct_network(n_user, n_item, n_entity, rating_data, kg):
    ratings = pd.DataFrame(rating_data, columns=['user', 'item', 'rating'])
    ratings = ratings[ratings['rating'] == 1]

    kg_df = pd.DataFrame(kg, columns=['h', 'r', 't'])
    user_entity_net = pd.merge(left=ratings, right=kg_df, left_on='item', right_on='h', how='left')[['user', 't']]
    user_entity_net = user_entity_net.groupby(['user', 't']).agg('size').reset_index()
    user_entity_net = user_entity_net.rename(columns={0:'count'})
    user_entity_net = user_entity_net[user_entity_net.t >= n_item]

    # constructing user-item graph
    # user_index: [0, n_user-1]        --> user_index_in_ui_graph: [0, n_user-1]
    # item_index:[0, n_item - 1]       --> item_index_in_ui_graph: [n_user, n_user + n_item - 1]
    # therefore: item_index + n_user = item_index_in_ui_graph 
    ui_adj = construct_adj(n_user, n_user+n_item, ratings.values, threshold=1)

    # constructing user-entity graph
    # user_index: [0, n_user-1]  --> user_index_in_ue_graph: [0, n_user-1]
    # entity_index: [n_item, n_entity - 1]         --> entity_index_in_ue_graph: [n_user, n_user + n_entity - n_item - 1]
    # therefore: entity_index - n_item + n_user = entity_index_in_ue_graph
    ue_adj = construct_adj(-n_item + n_user, n_user+n_entity-n_item, user_entity_net.values, threshold=1)
    return ui_adj, ue_adj

def construct_adj(shift, size, data, threshold=1):
    row = []; col = []; value = []

    for i in range(data.shape[0]):
        entity1_index = data[i][0]
        entity2_index = data[i][1]
        count = threshold if data[i][2] > threshold else data[i][2]

        entity2_index_shifted = entity2_index + shift
        row.append(entity1_index)
        col.append(entity2_index_shifted)
        value.append(count)
        row.append(entity2_index_shifted)
        col.append(entity1_index)
        value.append(count)
    adj = sp.coo_matrix((value, (row, col)), shape=(size, size))
    return adj

def preprocess_graph(adj):
    adj_ = adj + sp.diags(np.maximum(adj.max(axis=1).toarray()[:,0],1))
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized
    
def construct_kg(kg_np):
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((relation, tail))
    return kg


def get_n_hop_set(n_item, n_hop, kg):
    print('constructing n_hop set ...')
    kg = construct_kg(kg)
    n_hop_set = dict()

    for item in range(n_item):
        tails_of_last_hop = [item]
        kg_h_total = []
        kg_r_total = []
        kg_t_total = []

        for h in range(n_hop):
            kg_h = []
            kg_r = []
            kg_t = []

            for entity in tails_of_last_hop:
                for relation_and_tail in kg[entity]:
                    kg_h.append(entity)
                    kg_r.append(relation_and_tail[0])
                    kg_t.append(relation_and_tail[1])

            if len(kg_h) == 0:
                break
            else:
                kg_h_total.extend(kg_h)
                kg_r_total.extend(kg_r)
                kg_t_total.extend(kg_t)
                tails_of_last_hop = copy.deepcopy(kg_t)

        n_hop_set[item] = [kg_h_total, kg_r_total, kg_t_total]        
        
    return n_hop_set

def negative_sampling(pos_inds, n_items, n_samp=32):
    """fast negative sampling with binary search
    `pos_inds` is assumed to be ordered
    see https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html for more detail
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds