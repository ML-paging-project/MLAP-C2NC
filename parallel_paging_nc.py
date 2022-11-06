#!/usr/bin/python
# !/usr/bin/env python


import math
from collections import OrderedDict
from random import shuffle, randint
from copy import deepcopy

import xgboost as xgb

import other_algorithms_c2nc as oa
import xgboost_module_c2nc as xgbm
from lru import run_lru
from tool import read_crc_seq, format_box_nc

adv = 'z_adversary'
trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
               'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
               'milc', 'omnetpp', 'sphinx3', 'xalanc']
k = 64
s = 100
window = 256


class Trace:
    def __init__(self, tn):
        if tn == adv:
            self.seq = [ix for ix in range(30000)]
        else:
            self.seq = read_crc_seq('datasets/{0}_test.csv'.format(tn))
        self.name = tn
        self.box_start_pointer = 0
        self.box_end_pointer = 0
        self.box_start_stack = OrderedDict()
        self.box_end_stack = OrderedDict()
        self.box_height = int(k / 16)
        self.box_width = 2 * self.box_height * s
        self.memory_impact = 0
        self.completion_time = 0
        self.det_counter = [1, 0, 0, 0, 0]
        self.box_kind = 0
        self.run_time = 0
        # ###############for hybrid####################
        self.ml_req2mi = None
        self.det_req2box = None
        self.det_req2counter = None
        self.det_req2mi = None
        self.det_box_kind = None
        self.check_point = 0
        self.run_oracle = True
        # for marking algorithm
        self.start_marked = {}
        self.start_unmarked = {}
        self.end_marked = {}
        self.end_unmarked = {}

    def init_hybrid(self, model, number_of_box_kinds, hybrid_code):
        if self.name == adv:
            self.box_height = k
        else:
            feature = xgb.DMatrix([xgbm.get_feature_vector(self.seq, 0, window)])
            self.box_height = k / (2 ** model.predict(feature)[0])
        self.box_width = self.box_height * 2 * s
        if hybrid_code != '3':
            _, _, self.ml_req2mi = xgbm.test_xgboost_nc(model, self.seq, k, s, window,
                                                        self.name == adv)
        _, _, self.det_req2box, self.det_req2counter, self.det_req2mi = \
            oa.soda_lru(self.seq, k, number_of_box_kinds, s)
        self.det_box_kind = self.det_req2box[0]
        self.det_counter = []
        self.check_point = 0
        self.run_oracle = True

    def __eq__(self, other):
        return self.run_time == other.run_time

    def __le__(self, other):
        return self.run_time < other.run_time

    def __gt__(self, other):
        return self.run_time > other.run_time


def hybrid_parallel_paging(hybrid_code, alpha):
    if hybrid_code not in ['1.1', '1.2', '3']:
        return
    test_traces = {}
    number_of_box_kinds = 5
    phase = 0
    models = []
    for _ in range(4):
        model = xgb.Booster()
        models.append(model)
    '''
    models[0].load_model('pp_models/xgb_model_k64_b5_w256_s100_d20_r50')
    models[1].load_model('pp_models/xgb_model_k64_b4_w256_s100_d99_r99')
    models[2].load_model('pp_models/xgb_model_k64_b3_w256_s100_d199_r199')
    models[3].load_model('pp_models/xgb_model_k64_b2_w256_s100_d199_r199')
    '''
    models[0].load_model('pp_models/xgb_model_k128_b5_w256_s200_d230_r300')
    models[1].load_model('pp_models/xgb_model_k128_b4_w256_s200_d256_r300')
    models[2].load_model('pp_models/xgb_model_k128_b3_w256_s200_d256_r300')
    models[3].load_model('pp_models/xgb_model_k128_b2_w256_s200_d256_r300')

    packed_traces = {}
    finished = {}
    just_completed_box = []
    running_time = 0
    available_memory = k

    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name)
        test_traces[name].init_hybrid(models[0], number_of_box_kinds, hybrid_code)
        # test_traces[name] = format_box_nc(test_traces[name], s)

    while len(finished.keys()) != len(trace_names):
        test_traces, packed_traces, finished, available_memory, \
        running_time, just_completed_box = pack_and_run(just_completed_box, test_traces,
                                                        packed_traces,
                                                        finished,
                                                        available_memory,
                                                        running_time,
                                                        phase,
                                                        None,
                                                        'hybrid',
                                                        models,
                                                        hybrid_code,
                                                        alpha)

        # phasing
        phase_change = False
        if 8 >= len(trace_names) - len(finished.keys()) > 4 and phase != 1:
            phase = 1
            phase_change = True
        if 4 >= len(trace_names) - len(finished.keys()) > 2 and phase != 2:
            phase = 2
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 2 and phase != 3:
            phase = 3
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 1 and phase != 4:
            phase = 4
            phase_change = True
        if phase_change:
            number_of_box_kinds -= 1
            for name in trace_names:
                if name not in finished:
                    test_traces[name].box_end_pointer = test_traces[name].box_start_pointer
                    test_traces[name].box_end_stack = OrderedDict()
                    for pid in test_traces[name].box_start_stack:
                        test_traces[name].box_end_stack[pid] = True
                        test_traces[name].box_end_stack.move_to_end(pid, last=True)
                    if phase < 4:
                        if hybrid_code != '3':
                            _, _, test_traces[name].ml_req2mi = xgbm.test_xgboost_nc(models[phase],
                                                                                     test_traces[name].seq,
                                                                                     k, s, window,
                                                                                     name == adv)
                        _, _, test_traces[name].det_req2box, \
                        test_traces[name].det_req2counter, \
                        test_traces[name].det_req2mi = oa.soda_lru(test_traces[name].seq, k, number_of_box_kinds, s)
                    test_traces[name] = find_next_by_hybrid(test_traces[name],
                                                            phase,
                                                            models,
                                                            5 - phase,
                                                            hybrid_code,
                                                            alpha,
                                                            phase_change)

    print('hybrid', hybrid_code, 'alpha =', alpha)
    print_results(test_traces)


def xgb_parallel_paging():
    test_traces = {}
    phase = 0
    model = xgb.Booster()
    # model.load_model('pp_models/xgb_model_k64_b5_w256_s100_d20_r50')
    model.load_model('pp_models/xgb_model_k128_b5_w256_s200_d230_r300')
    packed_traces = {}
    finished = {}
    running_time = 0
    available_memory = k
    just_completed_box = []

    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name)
        if name == adv:
            test_traces[name].box_height = k
        else:
            fea = xgb.DMatrix([xgbm.get_feature_vector(test_traces[name].seq,
                                                       0,
                                                       window)])
            test_traces[name].box_height = k / (2 ** model.predict(fea)[0])
        test_traces[name].box_width = 2 * test_traces[name].box_height * s
        # test_traces[name] = format_box_nc(test_traces[name], s)

    while len(finished.keys()) != len(trace_names):
        test_traces, packed_traces, finished, available_memory, \
        running_time, just_completed_box = pack_and_run(just_completed_box, test_traces,
                                                        packed_traces,
                                                        finished,
                                                        available_memory,
                                                        running_time,
                                                        phase, model, 'xgb')
        # phasing
        phase_change = False
        if 8 >= len(trace_names) - len(finished.keys()) > 4 and phase != 1:
            phase = 1
            # model.load_model('pp_models/xgb_model_k64_b4_w256_s100_d99_r99')
            model.load_model('pp_models/xgb_model_k128_b4_w256_s200_d256_r300')
            phase_change = True
        if 4 >= len(trace_names) - len(finished.keys()) > 2 and phase != 2:
            phase = 2
            # model.load_model('pp_models/xgb_model_k64_b3_w256_s100_d199_r199')
            model.load_model('pp_models/xgb_model_k128_b3_w256_s200_d256_r300')
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 2 and phase != 3:
            phase = 3
            # model.load_model('pp_models/xgb_model_k64_b2_w256_s100_d199_r199')
            model.load_model('pp_models/xgb_model_k128_b2_w256_s200_d256_r300')
            phase_change = True
        if phase_change:
            for name in trace_names:
                if name not in finished.keys():
                    if name == adv:
                        test_traces[name].box_height = k
                    else:
                        fea = xgb.DMatrix([xgbm.get_feature_vector(test_traces[name].seq,
                                                                   test_traces[name].box_start_pointer,
                                                                   window)])
                        test_traces[name].box_height = k / (2 ** model.predict(fea)[0])
                    # test_traces[name] = format_box_nc(test_traces[name], miss_cost=s)
        if len(trace_names) - len(finished.keys()) == 1 and phase != 4:
            phase = 4
            for name in trace_names:
                if name not in finished.keys():
                    test_traces[name].box_height = k
                    # test_traces[name] = format_box_nc(test_traces[name], miss_cost=s)
    print('xgb PP result:')
    print_results(test_traces)


def soda_lru_parallel_paging():
    test_traces = {}
    phase = 0
    packed_traces = {}
    finished = {}
    running_time = 0
    available_memory = k
    just_completed_box = []

    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name)
        # test_traces[name] = format_box_nc(test_traces[name], s)
        print(test_traces[name].det_counter)

    while len(finished.keys()) != len(trace_names):
        test_traces, packed_traces, finished, \
        available_memory, running_time, just_completed_box = pack_and_run(just_completed_box,
                                                                          test_traces,
                                                                          packed_traces,
                                                                          finished,
                                                                          available_memory,
                                                                          running_time,
                                                                          phase, None, 'soda')

        # phasing
        phase_change = False
        if 8 >= len(trace_names) - len(finished.keys()) > 4 and phase != 1:
            phase = 1
            phase_change = True
        if 4 >= len(trace_names) - len(finished.keys()) > 2 and phase != 2:
            phase = 2
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 2 and phase != 3:
            phase = 3
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 1 and phase != 4:
            phase = 4
            phase_change = True
        if phase_change:
            for name in trace_names:
                if name not in finished.keys():
                    test_traces[name].det_counter = [0 for _ in range(5 - phase)]
                    test_traces[name].det_counter[0] = 1
                    test_traces[name].box_kind = 0
                    test_traces[name].box_height = k / (16 / (2 ** phase))
                    # test_traces[name] = format_box_nc(test_traces[name], miss_cost=s)

    print('SODA LRU PP result:')
    print_results(test_traces)


def soda_marking_parallel_paging():
    test_traces = {}
    phase = 0
    packed_traces = {}
    finished = {}
    running_time = 0
    available_memory = k
    just_completed_box = []

    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name)

    while len(finished.keys()) != len(trace_names):
        while len(packed_traces.keys()) + len(finished) < len(trace_names):
            # find the one to be packed
            next_trace = ''
            min_mi = math.inf
            for name in trace_names:
                if (name not in packed_traces) and \
                        (name not in finished) and \
                        test_traces[name].box_height <= available_memory and \
                        test_traces[name].memory_impact < min_mi:
                    min_mi = test_traces[name].memory_impact
                    next_trace = name

            if next_trace == '':
                break

            if next_trace not in just_completed_box:
                test_traces[next_trace].start_marked.clear()
                test_traces[next_trace].start_unmarked.clear()

            # run marking
            marked = deepcopy(test_traces[next_trace].start_marked)
            unmarked = deepcopy(test_traces[next_trace].start_unmarked)
            while len(marked) + len(unmarked) > test_traces[next_trace].box_height:
                if len(unmarked) > 0:
                    pop_key = list(unmarked.keys())[randint(0, len(unmarked) - 1)]
                    unmarked.pop(pop_key)
                else:
                    pop_key = list(marked.keys())[randint(0, len(marked) - 1)]
                    marked.pop(pop_key)

            width = 2 * test_traces[next_trace].box_height * s
            test_traces[next_trace].box_end_pointer = test_traces[next_trace].box_start_pointer
            while width > 0 and test_traces[next_trace].box_end_pointer < len(test_traces[next_trace].seq):
                if len(marked) == test_traces[next_trace].box_height:
                    for thing in marked:
                        unmarked[thing] = 1
                    marked.clear()
                req = test_traces[next_trace].seq[test_traces[next_trace].box_end_pointer]
                if req in marked or req in unmarked:
                    width -= 1
                else:
                    width -= s

                if req in unmarked:
                    unmarked.pop(req)
                marked[req] = 1

                while len(marked) + len(unmarked) > test_traces[next_trace].box_height:
                    popk = list(unmarked.keys())[randint(0, len(unmarked) - 1)]
                    unmarked.pop(popk)
                test_traces[next_trace].box_end_pointer += 1
            test_traces[next_trace].box_width = 2 * test_traces[next_trace].box_height * s - width
            test_traces[next_trace].end_marked = deepcopy(marked)
            test_traces[next_trace].end_unmarked = deepcopy(unmarked)
            # pack
            next_box_height = test_traces[next_trace].box_height
            next_box_width = test_traces[next_trace].box_width
            packed_traces[next_trace] = {'mem_size': next_box_height,
                                         'end_time': running_time + next_box_width,
                                         'last': test_traces[next_trace].box_end_pointer >= len(
                                             test_traces[next_trace].seq)}
            test_traces[next_trace].memory_impact += next_box_height * next_box_width
            available_memory -= next_box_height
            if test_traces[next_trace].box_end_pointer < len(test_traces[next_trace].seq):
                test_traces[next_trace] = find_next_by_soda(test_traces[next_trace])

        # find next decision time
        next_time_point = math.inf
        for trace in packed_traces.keys():
            if packed_traces[trace]['end_time'] < next_time_point:
                next_time_point = packed_traces[trace]['end_time']
        running_time = next_time_point

        just_completed_box = []
        for trace in list(packed_traces):
            if packed_traces[trace]['end_time'] == running_time:
                available_memory += packed_traces[trace]['mem_size']
                test_traces[trace].completion_time = running_time
                if packed_traces[trace]['last']:
                    finished[trace] = True
                    # test_traces[trace].completion_time += (packed_traces[trace]['mem_size'] * s)
                else:
                    just_completed_box.append(trace)
                packed_traces.pop(trace)
        # phasing
        phase_change = False
        if 8 >= len(trace_names) - len(finished.keys()) > 4 and phase != 1:
            phase = 1
            phase_change = True
        if 4 >= len(trace_names) - len(finished.keys()) > 2 and phase != 2:
            phase = 2
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 2 and phase != 3:
            phase = 3
            phase_change = True
        if len(trace_names) - len(finished.keys()) == 1 and phase != 4:
            phase = 4
            phase_change = True
        if phase_change:
            for name in trace_names:
                if name not in finished.keys():
                    test_traces[name].det_counter = [0 for _ in range(5 - phase)]
                    test_traces[name].det_counter[0] = 1
                    test_traces[name].box_kind = 0
                    test_traces[name].box_height = k / (16 / (2 ** phase))
                    # test_traces[name] = format_box_nc(test_traces[name], miss_cost=s)

    print('SODA MARKING PP result:')
    print_results(test_traces)


def evenly_split():
    test_traces = {}
    packed_traces = {}
    finished = {}
    running_time = 0
    available_memory = k
    just_completed_box = []

    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name)
        test_traces[name].box_height = int(k / len(trace_names))
        test_traces[name].box_width = int(k / len(trace_names)) * s * 2
        # test_traces[name] = format_box_nc(test_traces[name], s)

    while len(finished.keys()) != len(trace_names):
        old_len = len(finished)
        test_traces, packed_traces, finished, available_memory, \
        running_time, just_completed_box = pack_and_run(just_completed_box, test_traces,
                                                        packed_traces,
                                                        finished,
                                                        available_memory,
                                                        running_time,
                                                        -1, None, 'ES')

        # phasing
        if old_len != len(finished):
            for name in trace_names:
                if name not in finished.keys():
                    test_traces[name].box_height = int(k / (len(trace_names) - len(finished)))
                    # test_traces[name] = format_box_nc(test_traces[name], miss_cost=s)

    print('Evenly split PP result')
    print_results(test_traces)


def sequential_schedule():
    print('running sequential schedule')
    test_trace_list = []
    for nm in trace_names:
        tc = Trace(nm)
        test_trace_list.append(tc)
        cache = OrderedDict()
        pointer = 0
        while pointer < len(tc.seq):
            width = k * s * 10000
            end, rem = run_lru(cache, k, tc.seq,
                               pointer, width, 1, s)
            tc.run_time += width - rem
            for ii in range(pointer, end):
                if tc.seq[ii] not in cache.keys():
                    cache[tc.seq[ii]] = True
                cache.move_to_end(tc.seq[ii], last=False)
                while len(cache.keys()) > k:
                    cache.popitem(last=True)
            pointer = end
        print(nm, tc.run_time)

    test_trace_list = sorted(test_trace_list)
    for ii in range(len(test_trace_list)):
        if ii == 0:
            test_trace_list[ii].completion_time = test_trace_list[ii].run_time
        else:
            test_trace_list[ii].completion_time = test_trace_list[ii].run_time + test_trace_list[
                ii - 1].completion_time
        # print(test_traces[ii].name, test_traces[ii].completion_time, test_traces[ii].run_time)

    print('/////////////opt sequential//////////////////')
    max_ct = 0
    mean_ct = 0
    for nn in trace_names:
        for ttt in test_trace_list:
            if nn == ttt.name:
                print(nn, ttt.completion_time)
                max_ct = max(max_ct, ttt.completion_time)
                mean_ct += ttt.completion_time / len(trace_names)
    print('AVG', mean_ct)
    print('MAX', max_ct)

    max_ct = 0
    mean_ct = 0
    rounds = 100
    for _ in range(rounds):
        total = 0
        shuffle(test_trace_list)
        for ii in range(len(test_trace_list)):
            if ii == 0:
                test_trace_list[ii].completion_time = test_trace_list[ii].run_time
            else:
                test_trace_list[ii].completion_time = test_trace_list[ii].run_time + test_trace_list[
                    ii - 1].completion_time
            total += test_trace_list[ii].completion_time
        max_ct += test_trace_list[12].completion_time / rounds
        mean_ct += total / len(trace_names) / rounds
    print('/////////////avg result//////////////////')
    print(mean_ct, max_ct)


def test_round_robin():
    print('Round robin')
    repeat = 10
    avg_complete = 0
    max_complete = 0
    for t in range(repeat):
        print('RR', t)
        a, m = round_robin()
        avg_complete += a / repeat
        max_complete += m / repeat
    print('AVG', avg_complete)
    print('MAX', max_complete)


def round_robin():
    shuffle(trace_names)
    test_traces = {}
    packed_traces = {}
    finished = {}
    running_time = 0
    available_memory = k
    just_complete_box = []

    # initiate
    for name in trace_names:
        test_traces[name] = Trace(name)
        test_traces[name].box_height = k
        # test_traces[name] = format_box_nc(test_traces[name], s)

    while len(finished.keys()) != len(trace_names):
        test_traces, packed_traces, finished, available_memory, \
        running_time, just_complete_box = pack_and_run(just_complete_box, test_traces,
                                                       packed_traces,
                                                       finished,
                                                       available_memory,
                                                       running_time,
                                                       -1, None, 'RR')

    avg_complete = 0
    max_complete = 0
    for name in trace_names:
        avg_complete += test_traces[name].completion_time / len(trace_names)
        max_complete = max(max_complete, test_traces[name].completion_time)
    # print(avg_complete)
    # print(max_complete)
    return avg_complete, max_complete


def pack_and_run(just_completed_box, test_traces, packed_traces, finished,
                 available_memory, running_time,
                 phase, model, policy, model_array=None,
                 hybrid_code='', alpha=0):
    while len(packed_traces.keys()) + len(finished) < len(trace_names):
        # find the one to be packed
        next_trace = ''
        min_mi = math.inf
        for name in trace_names:
            if (name not in packed_traces) and \
                    (name not in finished) and \
                    test_traces[name].box_height <= available_memory and \
                    test_traces[name].memory_impact < min_mi:
                min_mi = test_traces[name].memory_impact
                next_trace = name

        if next_trace == '':
            break

        if next_trace not in just_completed_box:
            test_traces[next_trace].box_start_stack = OrderedDict()
        test_traces[next_trace] = format_box_nc(test_traces[next_trace], s)
        # pack
        next_box_height = test_traces[next_trace].box_height
        next_box_width = test_traces[next_trace].box_width
        packed_traces[next_trace] = {'mem_size': next_box_height,
                                     'end_time': running_time + next_box_width,
                                     'last': test_traces[next_trace].box_end_pointer >= len(
                                         test_traces[next_trace].seq)}
        test_traces[next_trace].memory_impact += next_box_height * next_box_width
        available_memory -= next_box_height
        if test_traces[next_trace].box_end_pointer < len(test_traces[next_trace].seq):
            if policy == 'xgb':
                test_traces[next_trace] = find_next_by_xgb(test_traces[next_trace], phase, model)
            if policy == 'soda':
                test_traces[next_trace] = find_next_by_soda(test_traces[next_trace])
            if policy == 'ES':
                test_traces[next_trace] = find_next_by_evenly_split(test_traces[next_trace],
                                                                    len(trace_names) - len(finished))
            if policy == 'RR':
                test_traces[next_trace] = find_next_rr(test_traces[next_trace])
            if policy == 'hybrid':
                test_traces[next_trace] = find_next_by_hybrid(test_traces[next_trace],
                                                              phase,
                                                              model_array,
                                                              5 - phase,
                                                              hybrid_code,
                                                              alpha)

    # find next decision time
    next_time_point = math.inf
    for trace in packed_traces.keys():
        if packed_traces[trace]['end_time'] < next_time_point:
            next_time_point = packed_traces[trace]['end_time']
    running_time = next_time_point

    just_completed_box = []
    for trace in list(packed_traces):
        if packed_traces[trace]['end_time'] == running_time:
            available_memory += packed_traces[trace]['mem_size']
            test_traces[trace].completion_time = running_time
            if packed_traces[trace]['last']:
                finished[trace] = True
                # test_traces[trace].completion_time += (packed_traces[trace]['mem_size'] * s)
            else:
                just_completed_box.append(trace)
            packed_traces.pop(trace)

    return test_traces, packed_traces, finished, available_memory, running_time, just_completed_box


def find_next_by_hybrid(t, phase, models, number_of_box_kinds,
                        hybrid_code,
                        alpha=1, phase_change=False, check_length=10000):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)

    if phase == 4:
        t.box_height = k
    else:
        if int(t.box_start_pointer / check_length) > t.check_point or phase_change:
            t.check_point = int(t.box_start_pointer / check_length)
            if hybrid_code in ['1.1', '1.2']:
                if (t.run_oracle or phase_change) and \
                        t.ml_req2mi[t.box_start_pointer - 1] >= alpha * t.det_req2mi[t.box_start_pointer - 1]:
                    t.det_box_kind = t.det_req2box[t.box_start_pointer - 1]
                    t.det_counter = t.det_req2counter[t.box_start_pointer - 1]

            if hybrid_code == '1.1':
                t.run_oracle = \
                    t.ml_req2mi[t.box_start_pointer - 1] < alpha * t.det_req2mi[t.box_start_pointer - 1]
            if hybrid_code == '1.2':
                if t.run_oracle:
                    t.run_oracle = \
                        t.ml_req2mi[t.box_start_pointer - 1] < alpha * t.det_req2mi[t.box_start_pointer - 1]
                else:
                    t.run_oracle = \
                        alpha * t.ml_req2mi[t.box_start_pointer - 1] < t.det_req2mi[t.box_start_pointer - 1]
            if hybrid_code == '3':
                if (t.run_oracle or phase_change) and \
                        t.memory_impact >= alpha * t.det_req2mi[t.box_start_pointer - 1]:
                    t.det_box_kind = t.det_req2box[t.box_start_pointer - 1]
                    t.det_counter = t.det_req2counter[t.box_start_pointer - 1]
                t.run_oracle = \
                    t.memory_impact < alpha * t.det_req2mi[t.box_start_pointer - 1]

        if t.run_oracle:
            if t.name == adv:
                oracle = 0
            else:
                feature = xgb.DMatrix([xgbm.get_feature_vector(t.seq, t.box_start_pointer, window)])
                oracle = models[phase].predict(feature)[0]
            t.box_height = k / (2 ** oracle)
        else:
            print('pick deterministic')
            if t.det_box_kind == number_of_box_kinds - 1:
                t.det_box_kind = 0
            elif t.det_counter[t.det_box_kind] % 4 == 0:
                t.det_box_kind = t.det_box_kind + 1
            else:
                t.det_box_kind = 0
            t.det_counter[t.det_box_kind] = t.det_counter[t.det_box_kind] + 1
            t.box_height = k / (2 ** (number_of_box_kinds - t.det_box_kind - 1))
        t.box_width = t.box_height
    # t = format_box_nc(t, s)
    return t


def find_next_by_xgb(t, phase, model):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    if phase == 4 or t.name == adv:
        t.box_height = k
    else:
        fea = xgb.DMatrix([xgbm.get_feature_vector(t.seq,
                                                   t.box_start_pointer,
                                                   window)])
        t.box_height = k / (2 ** model.predict(fea)[0])
    t.box_width = t.box_height * s * 2
    # t = format_box_nc(t, s)
    return t


def find_next_by_soda(t):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    t.start_marked.clear()
    for pid in t.end_marked:
        t.start_marked[pid] = 1
    t.start_unmarked.clear()
    for pid in t.end_unmarked:
        t.start_unmarked[pid] = 1
    if t.box_kind == (len(t.det_counter) - 1):
        t.box_kind = 0
    elif t.det_counter[t.box_kind] % 4 == 0:
        t.box_kind += 1
    else:
        t.box_kind = 0
    t.det_counter[t.box_kind] += 1
    t.box_height = k / (2 ** (len(t.det_counter) - t.box_kind - 1))
    t.box_width = t.box_height * s * 2
    # t = format_box_nc(t, s)
    return t


def find_next_by_evenly_split(t, trace_number):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    t.box_height = int(k / trace_number)
    t.box_width = t.box_height * 2 * s
    # t = format_box_nc(t, s)
    return t


def find_next_rr(t):
    t.box_start_pointer = t.box_end_pointer
    t.box_start_stack = OrderedDict()
    for pid in t.box_end_stack.keys():
        t.box_start_stack[pid] = True
        t.box_start_stack.move_to_end(pid, last=True)
    t.box_height = k
    t.box_width = t.box_height * 2 * s
    # t = format_box_nc(t, s)
    return t


def print_results(traces):
    max_c = 0
    mean = 0
    for name in trace_names:
        print(name, traces[name].completion_time)
        max_c = max(max_c, traces[name].completion_time)
        mean += traces[name].completion_time / len(trace_names)
    print('++++++++++++')
    print('AVG', mean)
    print('MAX', max_c)


# if you want to see what happens when one processor is adversarial
# enable next line
# trace_names.append(adv)
soda_marking_parallel_paging()
