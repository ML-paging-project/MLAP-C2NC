#!/usr/bin/python
# !/usr/bin/env python
import math
import os
import time

import matplotlib.pyplot as plt

import tool
import xgboost_module_c2nc as xgbm
from other_algorithms_c2nc import deterministic_algorithm, random_pick, fixed_size
from hybrid_green_paging_c2nc import hybrid_green_paging


def main():
    # paging parameters
    k = 128
    number_of_box_kinds = 8
    window_size = 256
    miss_cost = 200
    alpha = [1, 1]
    check_length = 10000

    # model parameters
    max_depth = 256
    training_round = 300  # training rounds
    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': max_depth,
        'num_class': number_of_box_kinds
    }
    model_filename = 'xgb_model_k{0}_b{1}_w{2}_s{3}_d{4}_r{5}'.format(k, number_of_box_kinds,
                                                                      window_size,
                                                                      miss_cost, max_depth,
                                                                      training_round)
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = time_str + '-result-plots-C2NC-' + model_filename
    os.makedirs(result_dir)

    # files = os.listdir('datasets')
    # print(files)
    names = ['astar', 'bwaves', 'bzip', 'cactusadm',
             'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
             'milc', 'omnetpp', 'sphinx3', 'xalanc']

    print('\n\n')
    found, model = xgbm.check_and_load_model(model_filename)
    if not found:
        print('########### building train set ###########')
        train_x = []
        train_y = []
        for trace_name in names:
            begin = time.time()
            print('building features for ' + trace_name)
            train_seq = tool.read_crc_seq('datasets/' + trace_name + '_train.csv')
            x, y = xgbm.features_and_labels(train_seq, k,
                                            number_of_box_kinds,
                                            miss_cost, window_size)
            for i in range(len(x)):
                train_x.append(x[i])
                train_y.append(y[i])
            opt_latency = time.time() - begin
            print("Latency of OPT: ", opt_latency)

        print('########### training xgb ###########')
        begin = time.time()
        model = xgbm.train_xgboost(train_x, train_y, params, training_round)
        print("train latency = ", time.time() - begin)
        xgbm.save_model(model, model_filename)  # Save this model for later
    else:
        # We have a model to use, so no need to train
        print('Found a previous model to use!', 'saved_models/', model_filename)
        print('No training necessary :D')

    soda_array = []
    rand_array = []
    oracle_array = []
    tick = 1

    # enable next one line if want to see adversarial green paging
    # names.append('z_adversary')

    # names = ['z_adversary']
    for trace_name in names:
        print('++++++++++ running', tick, '/', len(names), '++++++++++')
        tick += 1
        if trace_name == 'z_adversary':
            test_seq = [ix for ix in range(60000 * 1000)]
        else:
            test_seq = tool.read_crc_seq('datasets/' + trace_name + '_test.csv')

        print('########### testing xgb ###########')
        xgb_impact, xgb_boxes, xgb_req2mi = xgbm.test_xgboost_nc(model, test_seq, k,
                                                                 miss_cost,
                                                                 window_size,
                                                                 trace_name == 'z_adversary')
        '''print('########### testing hybrid 1.1 ###########')
        hybrid1_1 = hybrid_green_paging(test_seq, k, miss_cost, number_of_box_kinds,
                                        model, window_size, '1.1', alpha[0],
                                        trace_name == 'z_adversary', check_length, xgb_req2mi)
                                        '''
        print('########### testing hybrid 1.2 ###########')
        hybrid1_2 = hybrid_green_paging(test_seq, k, miss_cost, number_of_box_kinds,
                                        model, window_size, '1.2', alpha[0],
                                        trace_name == 'z_adversary', check_length, xgb_req2mi)
        print('########### testing hybrid 3 ###########')
        hybrid3 = hybrid_green_paging(test_seq, k, miss_cost, number_of_box_kinds,
                                      model, window_size, '3', alpha[1],
                                      trace_name == 'z_adversary', check_length, xgb_req2mi)

        print('######### testing other methods ############')
        det_impact, det_boxes, _, _, _ = deterministic_algorithm(test_seq, k,
                                                                 number_of_box_kinds,
                                                                 miss_cost)
        random_impact = random_pick(test_seq, k, number_of_box_kinds, miss_cost)

        print('oracle:       ', int(xgb_impact))
        oracle_array.append(int(xgb_impact))
        print('random:       ', int(random_impact))
        rand_array.append(int(random_impact))
        print('deterministic:', int(det_impact))
        soda_array.append(int(det_impact))
        fixes = [fixed_size(test_seq,
                            2 ** iii,
                            miss_cost) for iii in range(int(math.log2(k)) + 1)]

        # draw plot
        methods = ['Random', 'SODA', 'Oracle', 'H1.2-' + str(alpha[0]),
                   'H3-' + str(alpha[1])]
        v = [random_impact, det_impact, xgb_impact, hybrid1_2, hybrid3]
        for iii in range(len(fixes)):
            methods.append('Fix' + str(2 ** iii))
            v.append(fixes[iii])

        bar = plt.barh(range(len(v)), v, tick_label=methods, color='magenta')
        # plt.bar(methods, v, width=0.3, color='magenta')
        # plt.ylabel('methods', fontdict={'weight': 'black'})
        plt.grid(axis='x', linestyle='-.', linewidth=1, color='black', alpha=0.5)
        plt.xlabel('Memory impact', fontdict={'weight': 'black'})
        plt.xlim(0, max(v) * 1.3)
        plt.title('Memory impact on the test trace of ' + trace_name,
                  fontdict={'weight': 'black'})
        plt.bar_label(bar, label_type='edge', fmt='%.3g')

        # plt.show()
        plt.savefig(
            r'' + result_dir + '/' + trace_name + '.jpg')
        plt.close()
        # break

    print("...........................................................")
    print(model_filename)
    print('random')
    print(rand_array)
    print('SODA')
    print(soda_array)
    print('oracle')
    print(oracle_array)


main()
