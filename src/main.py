import argparse
import numpy as np
from data_loader import load_data
from train import train
import tensorflow as tf 

if __name__ == '__main__':

    show_loss = False
    np.random.seed(42)

    # default settings for movielens
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    # parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
    # parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
    # parser.add_argument('--kge_weight', type=float, default=1e-1, help='weight of the KGE term')
    # parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
    # parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--n_epoch', type=int, default=40, help='the number of epochs')
    # parser.add_argument('--max_margin', type=float, default=10, help='the maximum margin in KGE training')
    # parser.add_argument('--gcn_layer_dims', nargs='+', default=[64], help='the hidden dimensions of gcn layers')

    # default settings for Book-Crossing
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
    # parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
    # parser.add_argument('--n_hop', type=int, default=3, help='maximum hops')
    # parser.add_argument('--kge_weight', type=float, default=1e-1, help='weight of the KGE term')
    # parser.add_argument('--l2_weight', type=float, default=1e-3, help='weight of the l2 regularization term')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--n_epoch', type=int, default=60, help='the number of epochs')
    # parser.add_argument('--max_margin', type=float, default=10, help='the maximum margin in KGE training')
    # parser.add_argument('--gcn_layer_dims', nargs='+', default=[32], help='the hidden dimensions of gcn layers')

    # default settings for last.fm
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=3, help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=1e-1, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=80, help='the number of epochs') 
    parser.add_argument('--max_margin', type=float, default=10, help='the maximum margin in KGE training')
    parser.add_argument('--gcn_layer_dims', nargs='+', default=[32, 32], help='the hidden dimensions of gcn layers')

    args = parser.parse_args()
    dataset = args.dataset

    data_info = load_data(dataset)
    train(args, data_info, show_loss)
