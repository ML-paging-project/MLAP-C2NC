#!/usr/bin/python
# !/usr/bin/env python


import os
from collections import OrderedDict

import xgboost as xgb

import other_algorithms_c2nc as oa
from lru import run_lru
from tool import run_box_nc


def get_feature_vector(seq, pointer_position, window_size):
    """
    pointer_position show which request in the seq we want to build feature.
    window_size: how many requests we look back to build the feature.
    Output is a list of length window_size+4.
    The i th number in the list (1<=i<=window_size) is the frequency of
    the i th most frequent id.
    Chop the window into 4 segments, the last 4 number show
    how many distinct page ids in each seg
    """
    vector = []
    if pointer_position == 0:
        for i in range(window_size + 4):
            vector.append(0.00)
    else:
        if pointer_position < window_size:
            window = seq[0:pointer_position]
        else:
            window = seq[pointer_position - window_size:pointer_position]
        frequency = {}
        distinct = {}
        segments_distinct = [0.00, 0.00, 0.00, 0.00]
        # chop the window into 4 segments
        # count how many distinct page ids in each seg

        i = 0
        seg_id = 0
        while i < len(window):
            if window[i] in frequency.keys():
                frequency[window[i]] = frequency[window[i]] + 1.00
            else:
                frequency[window[i]] = 1.00
            distinct[window[i]] = True
            if (i + 1) % (int(window_size / 4)) == 0:
                # segment ends. see how many distinct items in this seg.
                segments_distinct[seg_id] = len(distinct.keys())
                distinct = {}
                seg_id = seg_id + 1
            i = i + 1

        if len(distinct.keys()) > 0:
            segments_distinct[seg_id] = len(distinct.keys())

        # 1<=j<=window_size, the j-th variable of the vector is
        # the frequency of the j-th most frequent page id in
        # the window
        for v in frequency.values():
            vector.append(v)
        while len(vector) < window_size:
            vector.append(0)
        vector.sort(reverse=True)

        # We chop the window_size requests into four segments of length w/4.
        # Count how many distinct ids in each segment,
        # and put the counting in the last four variables.
        for v in segments_distinct:
            vector.append(v)
    return vector


def features_and_labels(sequence, k, number_of_box_kinds, miss_cost, window_size):
    print('calculating opt on training trace...')
    opt_mi, opt_boxes, decision_points = oa.opt(sequence, k, number_of_box_kinds,
                                                miss_cost)
    print('calculating features...')
    xs = []
    for i in range(len(opt_boxes)):
        xs.append(get_feature_vector(sequence, decision_points[i], window_size))
    return xs, opt_boxes


def train_xgboost(train_x, train_y, params, num_round):
    print('training xgboost...')
    print('creating DMatrix...')
    xgb_train = xgb.DMatrix(train_x, label=train_y)
    print('performing training...')
    model = xgb.train(params, xgb_train, num_round)
    return model


# Save the model to a file for use later
def save_model(model, filename):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model.save_model('saved_models/' + filename)


# Check if a model has already been saved. If so use it instead of training a new one.
def check_and_load_model(filename):
    if os.path.exists('saved_models/' + filename):
        # Model found
        model = xgb.Booster()
        model.load_model('saved_models/' + filename)
        return True, model
    # No model found
    return False, None


def test_xgboost(model, test_seq, k, miss_cost, window_size):
    req2mi = []
    print('testing xgboost...')
    pointer = 0
    total_impact = 0
    box_seq = []
    global_lru = OrderedDict()
    while pointer < len(test_seq):
        start_pointer = pointer
        feature = xgb.DMatrix([get_feature_vector(test_seq, pointer, window_size)])
        oracle = model.predict(feature)[0]
        box_seq.append(int(oracle))
        cache_size = k / (2 ** oracle)
        box_width = miss_cost * cache_size
        # Compartmentalization
        # Load top pages from LRU stack.
        mycache = OrderedDict()
        for pid in global_lru.keys():
            mycache[pid] = True
            mycache.move_to_end(pid, last=True)  # ##########
            if len(mycache) == cache_size:
                break

        pointer, _ = run_lru(mycache, cache_size, test_seq,
                             pointer, box_width, 1, miss_cost)
        endpointer = pointer

        # Update global stack
        for x in range(start_pointer, endpointer):
            if test_seq[x] in global_lru.keys():
                global_lru.move_to_end(test_seq[x], last=False)
            else:
                global_lru[test_seq[x]] = True
                global_lru.move_to_end(test_seq[x], last=False)

        mi = 3 * miss_cost * cache_size * cache_size
        total_impact = total_impact + mi
        while len(req2mi) < endpointer:
            req2mi.append(total_impact)
    return total_impact, box_seq, req2mi


def test_xgboost_nc(model, test_seq, k, miss_cost, window_size, adversarial=False):
    print('testing xgboost w/o compartmentalization...')
    pointer = 0
    total_impact = 0
    box_seq = []
    req2mi = []
    my_cache = OrderedDict()
    previous_size = 0
    if adversarial:
        oracle = 0
    else:
        feature = xgb.DMatrix([get_feature_vector(test_seq, pointer, window_size)])
        oracle = model.predict(feature)[0]
    cache_size = int(k / (2 ** oracle))
    while pointer < len(test_seq):
        box_seq.append(int(oracle))
        # for hybrid paging
        while len(req2mi) < pointer:
            req2mi.append(total_impact + miss_cost *
                          previous_size * max(0, len(my_cache) - cache_size))
        my_cache, total_impact, pointer = run_box_nc(my_cache, cache_size,
                                                     total_impact, miss_cost,
                                                     previous_size,
                                                     pointer, test_seq)

        # get new box
        previous_size = cache_size
        if adversarial:
            oracle = 0
        else:
            feature = xgb.DMatrix([get_feature_vector(test_seq, pointer, window_size)])
            oracle = model.predict(feature)[0]
        cache_size = int(k / (2 ** oracle))

    # flush out the last box
    total_impact += (len(my_cache) * previous_size * miss_cost)
    while len(req2mi) < pointer:
        req2mi.append(total_impact)
    return total_impact, box_seq, req2mi
