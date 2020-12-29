import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from util import *

class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        with tf.variable_scope(self.name):
            self.weight = weight_variable_glorot(input_dim, output_dim, name='weight')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):        
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.weight)
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            x = tf.contrib.layers.bias_add(x)
            outputs = self.act(x)
        return outputs

class GCN():
    def __init__(self, name, feature, adj, feature_dim, hidden_dims, dropout=0.):
        self.name = name
        self.inputs = feature
        self.adj = adj
        self.layers = len(hidden_dims)
        self.layer_dim = [feature_dim]
        self.layer_dim.extend(hidden_dims)

        self.dropout = dropout
        with tf.variable_scope(self.name):
            self.build()
        
    def build(self):
        for i in range(self.layers):
            embeddings = self.inputs
            embeddings = GraphConvolution(
                name=self.name + '_layer_'+str(i),
                input_dim=self.layer_dim[i],
                output_dim=self.layer_dim[i+1],
                adj=self.adj,
                act=lambda x:x,
                dropout=self.dropout)(embeddings)

        self.embeddings = embeddings