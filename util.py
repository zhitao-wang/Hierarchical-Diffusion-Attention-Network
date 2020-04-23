#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import os
import numpy as np


def read_nodes(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(':')[0]
            data.extend(line.replace('\n', '').split(','))
        return data

def read_graph(filename, node_to_id):
    N = len(node_to_id)
    A = np.zeros((N,N), dtype=np.float32)
    with open(filename, 'r') as f:
        for line in f:
            edge = line.strip().split()
            if edge[0] in node_to_id and edge[1] in node_to_id:
                source_id = node_to_id[edge[0]]
                target_id = node_to_id[edge[1]]
                if len(edge) >= 3:
                    A[source_id,target_id] = float(edge[2])
                else:
                    A[source_id,target_id] = 1.0
    return A

def build_vocab(filename):
    data = read_nodes(filename)

    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    nodes, _ = list(zip(*count_pairs))
    nodes = list(nodes)
    nodes.insert(0,'-1')
    node_to_id = dict(zip(nodes, range(len(nodes))))
    # print node_to_id
    # nodes = list(set(data))
    # nodes.insert(0,'-1')
    # node_to_id = {}
    # index = 0
    # for node in nodes:
    #     node_to_id[node] = index
    #     index += 1 # index begins from 1, 0 represents padding mark

    return nodes, node_to_id

def _file_to_node_ids(filename, node_to_id):
    data = []
    len_list = []
    with open(filename, 'r') as f:
        for line in f:
            seq = line.strip().split(',')
            ix_seq = [node_to_id[x] for x in seq if x in node_to_id]
            if len(ix_seq)>=2:
                data.append(ix_seq)
                len_list.append(len(ix_seq)-1)
    size = len(data)
    total_num = np.sum(len_list)
    return (data, len_list, size, total_num)

def _file_to_node_ids_withtime(filename, node_to_id):
    cas_data = []
    time_data = []
    len_list = []
    with open(filename, 'r') as f:
        for line in f:
            time = line.strip().split(':')[-1].split(',')
            time = [float(i) for i in time]
            seq = line.strip().split(':')[0].split(',')
            ix_seq = [node_to_id[x] for x in seq if x in node_to_id]
            if len(ix_seq)>=2:
                cas_data.append(ix_seq)
                time_data.append(time)
                len_list.append(len(ix_seq)-1)
    size = len(cas_data)
    total_num = np.sum(len_list)
    return (cas_data, time_data, len_list, size, total_num)

def to_nodes(seq, nodes):
    return list(map(lambda x: nodes[x], seq))

def read_raw_data(data_path=None):
    train_path = data_path + '-train'
    valid_path = data_path +  '-val'
    test_path = data_path +  '-test'

    nodes, node_to_id = build_vocab(train_path)
    train_data = _file_to_node_ids(train_path, node_to_id)
    valid_data = _file_to_node_ids(valid_path, node_to_id)
    test_data = _file_to_node_ids(test_path, node_to_id)

    return train_data, valid_data,  test_data,  nodes, node_to_id

def read_raw_data_withtime(data_path=None):
    train_path = data_path + '-time' + '-train'
    valid_path = data_path + '-time' + '-val'
    test_path = data_path + '-time' + '-test'

    nodes, node_to_id = build_vocab(train_path)
    train_data = _file_to_node_ids_withtime(train_path, node_to_id)
    valid_data = _file_to_node_ids_withtime(valid_path, node_to_id)
    test_data = _file_to_node_ids_withtime(test_path, node_to_id)

    return train_data, valid_data, test_data, nodes, node_to_id

def batch_generator(train_data, batch_size, max_length):
    x = []
    y = []
    xs = []
    ys = []
    ss = []
    train_seq = train_data[0]
    train_steps = train_data[1]

    batch_len = len(train_seq) // batch_size

    for i in range(batch_len):
        batch_steps = np.array(train_steps[i * batch_size : (i + 1) * batch_size])
        max_batch_steps = batch_steps.max()
        if max_batch_steps > max_length:
            max_batch_steps = max_length
        for j in range(batch_size):
            seq = train_seq[i * batch_size + j]
            for k in range(len(seq)-1):
                if k+1 > max_length:
                    start_id = k - (max_length-1)
                else:
                    start_id = 0
                padded_seq = np.pad(np.array(seq[start_id:k+1]),(0, max_batch_steps-len(seq[start_id:k+1])),'constant')
                x.append(padded_seq)
                y.append(seq[k+1])
        x = np.array(x)
        y = np.array(y)
        xs.append(x)
        ys.append(y)
        ss.append(batch_steps)
        x = []
        y = []
    rest_len = len(train_steps[batch_len * batch_size : ])
    if rest_len != 0:
        batch_steps = np.array(train_steps[batch_len * batch_size : ])
        max_batch_steps = batch_steps.max()
        if max_batch_steps > max_length:
            max_batch_steps = max_length
        for j in range(rest_len):
            seq = train_seq[batch_len * batch_size + j]
            for k in range(len(seq)-1):
                if k+1 > max_length:
                    start_id = k - (max_length-1)
                else:
                    start_id = 0
                padded_seq = np.pad(np.array(seq[start_id:k+1]),(0, max_batch_steps-len(seq[start_id:k+1])),'constant')
                x.append(padded_seq)
                y.append(seq[k+1])
        x = np.array(x)
        y = np.array(y)
        xs.append(x)
        ys.append(y)
        ss.append(batch_steps)
    # Enumerator over the batches.
    return xs, ys, ss


def batch_generator_withtime(train_data, batch_size, max_length, n_ti, max_time, time_unit):
    x = []
    y = []
    t = []
    xs = []
    ys = []
    ts = []
    ss = []
    train_seq = train_data[0]
    train_time = train_data[1]
    train_steps = train_data[2]

    ti = max_time/n_ti

    batch_len = len(train_seq) // batch_size

    for i in range(batch_len):
        batch_steps = np.array(train_steps[i * batch_size : (i + 1) * batch_size])
        max_batch_steps = batch_steps.max()
        if max_batch_steps > max_length:
            max_batch_steps = max_length
        for j in range(batch_size):
            seq = train_seq[i * batch_size + j]
            time = train_time[i * batch_size + j]
            for k in range(len(seq)-1):
                if k+1 > max_length:
                    start_id = k - (max_length-1)
                else:
                    start_id = 0
                padded_seq = np.pad(np.array(seq[start_id:k+1]),(0, max_batch_steps-len(seq[start_id:k+1])),'constant')
                trunc_time = np.array(time[start_id:k+1])
                trunc_time = np.ceil((trunc_time[-1] - trunc_time)/(ti*time_unit))
                for _ in range(len(trunc_time)):
                    if trunc_time[_] > n_ti:
                        trunc_time[_] = n_ti
                # trunc_time.astype(int)
                padded_time = np.pad(trunc_time,(0, max_batch_steps-len(trunc_time)),'constant')
                x.append(padded_seq)
                y.append(seq[k+1])
                t.append(padded_time)
        x = np.array(x)
        y = np.array(y)
        t = np.array(t)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        ss.append(batch_steps)
        x = []
        y = []
        t = []
    rest_len = len(train_steps[batch_len * batch_size : ])
    if rest_len != 0:
        batch_steps = np.array(train_steps[batch_len * batch_size : ])
        max_batch_steps = batch_steps.max()
        if max_batch_steps > max_length:
            max_batch_steps = max_length
        for j in range(rest_len):
            seq = train_seq[batch_len * batch_size + j]
            time = train_time[batch_len * batch_size + j]
            for k in range(len(seq)-1):
                if k+1 > max_length:
                    start_id = k - (max_length-1)
                else:
                    start_id = 0
                padded_seq = np.pad(np.array(seq[start_id:k+1]),(0, max_batch_steps-len(seq[start_id:k+1])),'constant')
                trunc_time = np.array(time[start_id:k+1])
                trunc_time = np.ceil((trunc_time[-1] - trunc_time)/(ti*time_unit))
                for _ in range(len(trunc_time)):
                    if trunc_time[_] > n_ti:
                        trunc_time[_] = n_ti
                # trunc_time.astype(int)
                padded_time = np.pad(trunc_time,(0, max_batch_steps-len(trunc_time)),'constant')
                x.append(padded_seq)
                y.append(seq[k+1])
                t.append(padded_time)
        x = np.array(x)
        y = np.array(y)
        t = np.array(t)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        ss.append(batch_steps)
    # Enumerator over the batches.
    return xs, ys, ts, ss

def main():
    train_data, valid_data,  test_data, nodes, node_to_id = \
        read_raw_data_withtime('data/meme-cascades')

    x_train, y_train, t_train, seq_length = batch_generator_withtime(test_data, 5, 5, 50, 100, 3600.)
    len1 = seq_length[0][0]
    len2 = seq_length[0][1]
    for i in range(len1):
        print (x_train[0][i], y_train[0][i], t_train[0][i])
    print (x_train[0][len1], y_train[0][len1], t_train[0][len1])


    # print(x_train.shape)

    # print(to_nodes(x_train[0][1], nodes))

    # print(to_nodes(y_train[0][1], nodes))
    # print(seq_length)

if __name__ == '__main__':
    main()