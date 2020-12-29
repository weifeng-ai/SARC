import tensorflow as tf
import collections
import numpy as np
from model import SARC
from util import *

MODEL_DIR = "../models/"

def train(args, data_info, show_loss):
    train_data, eval_data, test_data, n_user, n_item, n_entity, n_relation, kg = data_info
    train_data = pd.DataFrame(train_data, columns=['user', 'item', 'rating'])
    train_data = train_data[train_data['rating'] == 1].values
    history = collections.defaultdict(set)
    for i in range(train_data.shape[0]):
        user = train_data[i][0]
        item = train_data[i][1]
        history[user].add(item)

    sorted_history = dict()
    for user, item_set in history.items():
        sorted_item = np.array(sorted(item_set))
        sorted_history[user] = sorted_item

    n_hop_set = get_n_hop_set(n_item, args.n_hop, kg)
    ui_adj, ue_adj = construct_network(n_user, n_item, n_entity, train_data, kg)
    ui_adj_norm = preprocess_graph(ui_adj)
    ue_adj_norm = preprocess_graph(ue_adj)
    
    model = SARC(args, n_user, n_item, n_entity, n_relation, ui_adj_norm, ue_adj_norm)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        n_noprogress = 0
        max_noprogress = 15
        max_acc = 0

        for step in range(args.n_epoch):

            # training
            train_total = add_negative_sample(train_data, sorted_history, n_item)
            np.random.shuffle(train_total)
            start = 0
            while start < train_total.shape[0]:
                _, loss = model.train(
                    sess, get_train_feed_dict(args, model, train_total, n_hop_set, n_entity, start, start + args.batch_size, 0.2, n_item))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_total.shape[0] * 100, loss))

            # evaluation
            train_auc, train_acc= evaluation(sess, args, model, train_total, args.batch_size)
            eval_auc, eval_acc= evaluation(sess, args, model, eval_data, args.batch_size)
            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f'
              % (step, train_auc, train_acc, eval_auc, eval_acc))

            if eval_acc > max_acc:
                max_acc = eval_acc
                n_noprogress = 0
                saver.save(sess, MODEL_DIR + model.name + ".ckpt")
            else:
                n_noprogress += 1
           
            if n_noprogress > max_noprogress:
                break

        print('Training has done. Evaluating model ...')
        saver.restore(sess, MODEL_DIR + model.name + ".ckpt")
        test_auc, test_acc= evaluation(sess, args, model, test_data, args.batch_size)
        print('test auc: %.4f  acc: %.4f' % (test_auc, test_acc))

def add_negative_sample(data, history, n_item):
    neg_sample = []
    for user, pos_item in history.items():
        for item in negative_sampling(pos_item, n_item, n_samp=len(pos_item)):
            neg_sample.append(np.array([user, item, 0]))
    return np.vstack([data, np.array(neg_sample)])

def get_train_feed_dict(args, model, data, n_hop_set, n_entity, start, end, dropout, n_item):
    feed_dict = dict()
    feed_dict[model.users] = data[start:end, 0]
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    feed_dict[model.dropout] = dropout

    kg_h = []
    kg_r = []
    kg_t = []
    negative_t = []

    # for item in data[start:end, 1]:
    for item in np.random.choice(n_item, data[start:end, 1].shape[0]):
        kg_h.extend(n_hop_set[item][0])
        kg_r.extend(n_hop_set[item][1])
        kg_t.extend(n_hop_set[item][2])
    feed_dict[model.kg_h] = kg_h
    feed_dict[model.kg_r] = kg_r
    feed_dict[model.kg_t] = kg_t
    feed_dict[model.negative_t] = np.random.choice(n_entity, size=len(feed_dict[model.kg_h]))

    return feed_dict

def get_test_feed_dict(args, model, data, start, end, dropout):
    feed_dict = dict()
    feed_dict[model.users] = data[start:end, 0]
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    feed_dict[model.dropout] = dropout

    return feed_dict

def evaluation(sess, args, model, data, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    while start < data.shape[0]:
        auc, acc = model.eval(sess, get_test_feed_dict(args, model, data, start, start + batch_size, 0))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))
