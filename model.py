import tensorflow as tf
import numpy as np
from rank_metrics import rank_eval
import argparse

def ill_cal(pred, sl):
    nll = 0
    cur_pos = 0
    for i in range(len(sl)):
        length = sl[i]
        cas_nll = pred[cur_pos : cur_pos+length]
        cur_pos += length
        nll += (np.sum(cas_nll)/float(length))
    return nll

# cas_emb:[b,n,d]  cas_mask:[b,n,1]
def hidan(cas_emb, cas_mask, time_weight, hidden_size, keep_prob):
    cas_encoding = user2user(cas_emb, cas_mask, hidden_size, keep_prob)     # [b,n,d]
    return user2cas(cas_encoding, cas_mask, time_weight, hidden_size, keep_prob)

def user2user(cas_emb, cas_mask, hidden_size, keep_prob):
    with tf.variable_scope('user2user'):
        bs, sl = tf.shape(cas_emb)[0], tf.shape(cas_emb)[1]
        col, row = tf.meshgrid(tf.range(sl), tf.range(sl))            # [n,n]
        direction_mask = tf.greater(row, col)                       # [n,n]
        direction_mask_tile = tf.tile(tf.expand_dims(direction_mask, 0), [bs, 1, 1])     # [b,n,n]
        length_mask_tile = tf.tile(tf.expand_dims(tf.squeeze(tf.cast(cas_mask,tf.bool),-1), 1), [1, sl, 1])             # [b,1,n] -> [b,n,n]
        attention_mask = tf.cast(tf.logical_and(direction_mask_tile, length_mask_tile), tf.float32)         # [b,n,n]
        cas_hidden = dense(cas_emb, hidden_size, tf.nn.elu, keep_prob, 'hidden') * cas_mask   # [b,n,d]

        head = dense(cas_hidden, hidden_size, tf.identity, keep_prob, 'head', False) # [b,n,d]
        tail = dense(cas_hidden, hidden_size, tf.identity, keep_prob, 'tail', False) # [b,n,d]

        matching_logit = tf.matmul(head, tf.transpose(tail,perm=[0,2,1])) + (1-attention_mask) * (-1e30)
        attention_score = tf.nn.softmax(matching_logit, -1) * attention_mask
        depend_emb = tf.matmul(attention_score, cas_hidden)         # [b,n,d]

        fusion_gate = dense(tf.concat([cas_hidden, depend_emb], 2), hidden_size, tf.sigmoid, keep_prob, 'fusion_gate')  # [b,n,d]
        return (fusion_gate*cas_hidden + (1-fusion_gate)*depend_emb) * cas_mask   # [b,n,d]

def user2cas(cas_encoding, cas_mask, time_weight, hidden_size, keep_prob):
    with tf.variable_scope('user2cas'):
        map1 = dense(cas_encoding, hidden_size, tf.nn.elu, keep_prob, 'map1')   # [b,n,d]
        time_influence = dense(time_weight, hidden_size, tf.nn.elu, keep_prob, 'time_influence')
        map2 = dense(map1 * time_influence, 1, tf.identity, keep_prob, 'map2')
        attention_score =  tf.nn.softmax(map2 + (-1e30) * (1 - cas_mask) , 1) * cas_mask
        return tf.reduce_sum(attention_score * cas_encoding, 1)


def dense(input, out_size, activation, keep_prob, scope, need_bias=True):
    with tf.variable_scope(scope):
        W = tf.get_variable('W', [input.get_shape()[-1], out_size], dtype=tf.float32)
        b = tf.get_variable('b', [out_size], tf.float32, tf.zeros_initializer(), trainable=need_bias)
        flatten = tf.matmul(tf.reshape(input, [-1, tf.shape(input)[-1]]), W) + b
        out_shape = [tf.shape(input)[i] for i in range(len(input.get_shape())-1)] + [out_size]
        return tf.nn.dropout(activation(tf.reshape(flatten, out_shape)), keep_prob)

class Model(object):
    def __init__(self, config):
        self.num_nodes = config.num_nodes
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.learning_rate = config.learning_rate
        self.l2_weight = config.l2_weight
        self.train_dropout = config.dropout
        self.n_time_interval = config.n_time_interval
        self.optimizer = config.optimizer


    def build_model(self):
        with tf.variable_scope("model",initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.cas = tf.placeholder(tf.int32, [None, None])                    # (b,n)

            self.cas_length= tf.reduce_sum(tf.sign(self.cas),1)
            self.cas_mask = tf.expand_dims(tf.sequence_mask(self.cas_length, tf.shape(self.cas)[1], tf.float32), -1)

            self.dropout = tf.placeholder(tf.float32)
            self.labels = tf.placeholder(tf.int32, [None])                          # (b,)

            self.time_interval_index = tf.placeholder(tf.int32, [None, None])       # (b,n)

            self.num_cas = tf.placeholder(tf.float32)

            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable(
                    "embedding", [self.num_nodes,
                        self.embedding_size], dtype=tf.float32)
                self.cas_emb = tf.nn.embedding_lookup(self.embedding, self.cas)      # (b,n,l)

                self.time_lambda = tf.get_variable('time_lambda', [self.n_time_interval+1, self.hidden_size], dtype=tf.float32) #,
                self.time_weight = tf.nn.embedding_lookup(self.time_lambda, self.time_interval_index)

            with tf.variable_scope("hidan") as scope:
                self.hidan = hidan(self.cas_emb, self.cas_mask, self.time_weight, self.hidden_size, self.dropout)

            with tf.variable_scope("loss"):
                l0 = self.hidan
                self.logits = dense(l0, self.num_nodes, tf.identity, 1.0, 'logits')
                self.nll = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, self.num_nodes, dtype=tf.float32), logits=self.logits)
                self.loss = tf.reduce_mean(self.nll,-1)
                for v in tf.trainable_variables():
                    self.loss += self.l2_weight * tf.nn.l2_loss(v)
                if self.optimizer == 'adaelta':
                    self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.999).minimize(self.loss)
                else:
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.99).minimize(self.loss)

    def train_batch(self, sess, batch_data):
        cas, next_user, time_interval_index, seq_len = batch_data
        feed = {self.cas: cas,
                self.labels: next_user,
                self.dropout: self.train_dropout,
                self.time_interval_index: time_interval_index,
                self.num_cas: len(seq_len)
               }
        _, _, nll = sess.run([self.train_op, self.loss, self.nll], feed_dict = feed)
        batch_nll = np.sum(nll)
        return batch_nll

    def test_batch(self, sess, batch_test):
        cas, next_user, time_interval_index, seq_len = batch_test
        feed = {self.cas: cas,
                self.labels: next_user,
                self.time_interval_index: time_interval_index,
                self.dropout: 1.0
               }
        logits, nll = sess.run([self.logits, self.nll], feed_dict = feed)
        # batch_rr = mrr_cal(logits, next_user, seq_len)
        mrr, macc1, macc5, macc10, macc50, macc100 = rank_eval(logits, next_user, seq_len)
        batch_cll = np.sum(nll)
        batch_ill = ill_cal(nll, seq_len)
        return batch_cll, batch_ill, mrr, macc1, macc5, macc10, macc50, macc100
