import tensorflow as tf
import numpy as np
from gcn import GCN
from sklearn.metrics import roc_auc_score

class SARC(object):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, ui_adj, ue_adj):
        self._parse_args(args, n_user, n_item, n_entity, n_relation, ui_adj, ue_adj)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_user, n_item, n_entity, n_relation, ui_adj, ue_adj):
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.max_margin = args.max_margin
        self.gcn_layer_dims = args.gcn_layer_dims
        self.name =  "sarc."+args.dataset+".dim."+str(args.dim)+".lambda."+str(args.l2_weight)+".lr."+str(args.lr)

        self.ui_adj = self._get_sparse_tensor(ui_adj)
        self.ue_adj = self._get_sparse_tensor(ue_adj)

    def _get_sparse_tensor(self, adj):
        indices = np.mat([adj.row, adj.col]).transpose()
        return  tf.cast(tf.SparseTensor(indices, adj.data, adj.shape),tf.float32)

    def _build_inputs(self):
        self.users = tf.placeholder(dtype=tf.int32, shape=[None], name='users')
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")
        self.dropout = tf.placeholder_with_default(0., shape=())

        self.kg_h =  tf.placeholder(dtype=tf.int32, shape=[None], name="kg_h")
        self.kg_r =  tf.placeholder(dtype=tf.int32, shape=[None], name="kg_r")
        self.kg_t =  tf.placeholder(dtype=tf.int32, shape=[None], name="kg_t")
        self.negative_t = tf.placeholder(dtype=tf.int32, shape=[None], name="negative_t")

    def _build_embeddings(self):
        self.user_emb = tf.get_variable(name='user_emb_matrix', dtype=tf.float32,
                                    shape=[self.n_user, self.dim], initializer=tf.contrib.layers.xavier_initializer())

        self.entity_emb = tf.get_variable(name="entity_emb", dtype=tf.float32,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.relation_emb = tf.get_variable(name='relation_emb', dtype=tf.float32,
                                            shape=[self.n_relation, self.dim], initializer=tf.contrib.layers.xavier_initializer())

        self.relation_normal = tf.get_variable(name='relation_normal', dtype=tf.float32,
                                                shape=[self.n_relation, self.dim], initializer=tf.contrib.layers.xavier_initializer())

        self.attn_weight_ue = tf.get_variable(name='attn_weight_ue', dtype=tf.float32,
                                            shape=[self.dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.attn_weight_ui = tf.get_variable(name='attn_weight_ui', dtype=tf.float32,
                                            shape=[self.dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.attn_bias_u = tf.get_variable(name='attn_bias_u', dtype=tf.float32,
                                            shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())

        self.attn_weight_ie = tf.get_variable(name='attn_weight_ie', dtype=tf.float32,
                                            shape=[self.dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.attn_weight_iu = tf.get_variable(name='attn_weight_iu', dtype=tf.float32,
                                            shape=[self.dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.attn_bias_i = tf.get_variable(name='attn_bias_i', dtype=tf.float32,
                                            shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())

    def _build_model(self):

        # [kg_batch size, dim]
        self.h_emb_list = tf.nn.embedding_lookup(self.entity_emb, self.kg_h)
        self.r_emb_list = tf.nn.embedding_lookup(self.relation_emb, self.kg_r)
        self.r_normal_list = tf.nn.embedding_lookup(self.relation_normal, self.kg_r)
        self.t_emb_list = tf.nn.embedding_lookup(self.entity_emb, self.kg_t)
        self.neg_t_emb_list = tf.nn.embedding_lookup(self.entity_emb, self.negative_t)
        
        self.net_entity_emb = tf.gather(self.entity_emb, np.arange(self.n_entity-self.n_item)+self.n_item)
        ue_feature = tf.concat([self.user_emb, self.net_entity_emb], axis=0)
        self.gcn_ue = GCN(
            name = 'gcn_ue', 
            feature = ue_feature, 
            adj = self.ue_adj,  
            feature_dim = self.dim, 
            hidden_dims = self.gcn_layer_dims, 
            dropout=self.dropout)

        self.net_item_emb = tf.gather(self.entity_emb, np.arange(self.n_item))
        ui_feature = tf.concat([self.user_emb, self.net_item_emb], axis=0)        
        self.gcn_ui = GCN(
            name = 'gcn_ui', 
            feature = ui_feature, 
            adj = self.ui_adj,  
            feature_dim = self.dim,
            hidden_dims = self.gcn_layer_dims, 
            dropout=self.dropout)

        user_vec_ue = tf.nn.embedding_lookup(self.gcn_ue.embeddings, self.users)
        user_vec_ui = tf.nn.embedding_lookup(self.gcn_ui.embeddings, self.users)
        self.user_alpha = tf.sigmoid(tf.matmul(user_vec_ue, self.attn_weight_ue) + tf.matmul(user_vec_ui, self.attn_weight_ui) + self.attn_bias_u)
        self.user_vec = self.user_alpha * user_vec_ue + (1-self.user_alpha) * user_vec_ui

        item_vec_ie = tf.nn.embedding_lookup(self.entity_emb, self.items)
        item_vec_iu = tf.nn.embedding_lookup(self.gcn_ui.embeddings, self.items+self.n_user)
        self.item_alpha = tf.sigmoid(tf.matmul(item_vec_ie, self.attn_weight_ie) + tf.matmul(item_vec_iu, self.attn_weight_iu) + self.attn_bias_i)
        self.item_vec = self.item_alpha * item_vec_ie + (1-self.item_alpha) * item_vec_iu

        self.scores = tf.squeeze(self.predict(self.item_vec, self.user_vec))
        self.scores_normalized = tf.sigmoid(self.scores)

    def predict(self, item_vec, user_vec):
        scores = tf.reduce_sum(item_vec * user_vec, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        # kge loss

        norm = tf.nn.l2_normalize(self.r_normal_list, axis=1)
        h= tf.nn.l2_normalize(self.h_emb_list, axis=1)
        h_proj = h - tf.reduce_sum(h * norm, axis=1, keepdims=True) * norm

        t_pos = tf.nn.l2_normalize(self.t_emb_list, axis=1)
        t_pos_proj = t_pos - tf.reduce_sum(t_pos * norm, axis=1, keepdims=True) * norm
        
        t_neg = tf.nn.l2_normalize(self.neg_t_emb_list, axis=1)
        t_neg_proj = t_neg - tf.reduce_sum(t_neg * norm, axis=1, keepdims=True) * norm

        hrt_pos = tf.reduce_sum((h_proj + self.r_emb_list - t_pos_proj) ** 2, axis=1)
        hrt_neg = tf.reduce_sum((h_proj + self.r_emb_list - t_neg_proj) ** 2, axis=1)

        self.kge_loss = tf.reduce_mean(tf.maximum(hrt_pos - hrt_neg + self.max_margin, 0))

        self.kge_loss = self.kge_weight * self.kge_loss

        self.l2_loss = 0

        trainables = tf.trainable_variables()
        self.l2_loss += tf.add_n([tf.nn.l2_loss(v) for v in trainables if 'weight' in v.name])
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores= sess.run([self.labels, self.scores_normalized], feed_dict)
        # print(scores)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc