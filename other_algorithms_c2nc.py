#!/usr/bin/python
# !/usr/bin/env python


import collections
import math
import random

import networkx as nx

from tool import run_box_nc


def random_pick(sequence, k, number_of_box_kinds, miss_cost):
    print('running random picking...')
    total_impact = 0
    rounds = 100  # number of rounds to evaluate over
    for _ in range(rounds):
        pointer = 0
        my_cache = collections.OrderedDict()
        box_id = random.randint(0, number_of_box_kinds - 1)
        cache_size = int(k / (2 ** box_id))
        previous_size = 0
        while pointer < len(sequence):
            my_cache, total_impact, pointer = run_box_nc(my_cache, cache_size,
                                                         total_impact, miss_cost,
                                                         previous_size,
                                                         pointer, sequence)

            # get a new box
            previous_size = cache_size
            box_id = random.randint(0, number_of_box_kinds - 1)
            cache_size = int(k / (2 ** box_id))

        # flush out the last box
        total_impact += (len(my_cache) * previous_size * miss_cost)
    return total_impact / rounds


def soda_lru(sequence, k, number_of_box_kinds, miss_cost):
    print('running soda w/ lru...')
    counting = [0 for _ in range(number_of_box_kinds)]
    counting[0] = 1
    current_box = 0
    pointer = 0
    total_impact = 0
    box_seq = []
    my_cache = collections.OrderedDict()

    req2counter = []
    req2box = []
    req2mi = []

    cache_size = int(k / (2 ** (number_of_box_kinds - current_box - 1)))
    previous_size = 0
    while pointer < len(sequence):
        box_seq.append(number_of_box_kinds - current_box - 1)
        my_cache, total_impact, pointer = run_box_nc(my_cache, cache_size,
                                                     total_impact, miss_cost,
                                                     previous_size,
                                                     pointer, sequence)

        # for hybrid paging
        while len(req2box) < pointer:
            req2box.append(current_box)
            req2counter.append(counting)
        # get new box
        previous_size = cache_size
        if current_box == number_of_box_kinds - 1:
            current_box = 0
        elif counting[current_box] % 4 == 0:
            current_box = current_box + 1
        else:
            current_box = 0
        counting[current_box] = counting[current_box] + 1
        cache_size = int(k / (2 ** (number_of_box_kinds - current_box - 1)))

        # for hybrid paging
        if pointer < len(sequence):
            while len(req2mi) < pointer:
                req2mi.append(total_impact +
                              miss_cost * previous_size *
                              max(0, len(my_cache) - cache_size))

    # flush out the last box
    total_impact += (len(my_cache) * previous_size * miss_cost)
    while len(req2mi) < pointer:
        req2mi.append(total_impact)
        req2box.append(current_box)
        req2counter.append(counting)
    return total_impact, box_seq, req2box, req2counter, req2mi


def soda_marking(sequence, k, number_of_box_kinds, miss_cost):
    print('running soda with marking')
    counting = [0 for _ in range(number_of_box_kinds)]
    counting[0] = 1
    current_box = 0
    pointer = 0
    total_impact = 0
    box_seq = []
    marked_cache = {}
    unmarked_cache = {}
    req2counter = []
    req2box = []
    req2mi = []
    cache_size = int(k / (2 ** (number_of_box_kinds - current_box - 1)))
    previous_size = 0
    while pointer < len(sequence):
        box_seq.append(number_of_box_kinds - current_box - 1)

        # flush out
        while len(marked_cache) + len(unmarked_cache) > cache_size:
            total_impact += (miss_cost * previous_size)
            if len(unmarked_cache) > 0:
                popk = list(unmarked_cache.keys())[random.randint(0,
                                                                  len(unmarked_cache) - 1)]
                unmarked_cache.pop(popk)
            else:
                popk = list(marked_cache.keys())[random.randint(0,
                                                                len(marked_cache) - 1)]
                marked_cache.pop(popk)

        # run a new box
        # start_pointer = pointer
        box_width = miss_cost * cache_size * 2
        remain_width = box_width
        total_impact += (box_width * cache_size)
        while remain_width > 0 and pointer < len(sequence):
            if len(marked_cache) == cache_size:
                for thing in marked_cache:
                    unmarked_cache[thing] = 1
                marked_cache.clear()

            if sequence[pointer] in marked_cache or sequence[pointer] in unmarked_cache:
                remain_width -= 1
            else:
                remain_width -= miss_cost

            if sequence[pointer] in unmarked_cache:
                unmarked_cache.pop(sequence[pointer])

            marked_cache[sequence[pointer]] = 1

            while len(marked_cache) + len(unmarked_cache) > cache_size:
                # print('mmmmm',len(unmarked_cache) - 1)
                popk = list(unmarked_cache.keys())[random.randint(0,
                                                                  len(unmarked_cache) - 1)]
                unmarked_cache.pop(popk)

            pointer += 1
        total_impact -= (remain_width * cache_size)

        # for hybrid paging
        while len(req2box) < pointer:
            req2box.append(current_box)
            req2counter.append(counting)

        # get new box
        previous_size = cache_size
        if current_box == number_of_box_kinds - 1:
            current_box = 0
        elif counting[current_box] % 4 == 0:
            current_box = current_box + 1
        else:
            current_box = 0
        counting[current_box] = counting[current_box] + 1
        cache_size = int(k / (2 ** (number_of_box_kinds - current_box - 1)))

        # for hybrid paging
        if pointer < len(sequence):
            while len(req2mi) < pointer:
                req2mi.append(total_impact +
                              miss_cost * previous_size *
                              max(0, len(marked_cache) + len(unmarked_cache) - cache_size))

    # flush out the last box
    total_impact += (len(marked_cache) + len(unmarked_cache) * previous_size * miss_cost)
    while len(req2mi) < pointer:
        req2mi.append(total_impact)
        req2box.append(current_box)
        req2counter.append(counting)
    return total_impact, box_seq, req2box, req2counter, req2mi


############################
# run lru, mark down hits and faults
def lru(sequence, size):
    stack = collections.OrderedDict()
    marks = []
    for r in sequence:
        if r in stack.keys():
            marks.append(True)  # a hit
        else:
            if len(stack.keys()) == size:  # memory is full
                stack.popitem(last=True)
            marks.append(False)  # a fault
            stack[r] = True
        stack.move_to_end(r, last=False)
    return marks


def opt(sequence, k, number_of_box_kinds, miss_cost):
    print('running offline opt')
    # parameters
    n = len(sequence)
    ############################
    print('building dag...')
    # build dag
    dag = nx.DiGraph()
    edge2box = {}  # to reconstruct the optimal box sequence
    for i in range(number_of_box_kinds):
        # loop for every row
        end_index = [-1 for _ in range(n)]
        memory_size = math.floor(k / (2 ** i))
        # run lru on the whole sequence
        # with memory_size = k / (2 ** i)
        is_a_hit = lru(sequence, memory_size)
        # find the end index for req[0]
        box_width = miss_cost * memory_size
        request_position = 0
        running_time = 0
        while running_time < box_width and request_position < n:  # So burst is allowed for the last req
            if is_a_hit[request_position]:
                running_time += 1
            else:
                running_time += miss_cost
            request_position += 1
        end_index[0] = request_position - 1
        dag.add_edge(0, end_index[0] + 1, weight=memory_size * miss_cost * 3 * memory_size)
        edge2box[(0, end_index[0] + 1)] = i

        # find end indexes for other nodes in row i
        for j in range(1, n):
            if is_a_hit[j - 1]:
                running_time = running_time - 1
            else:
                running_time = running_time - miss_cost
            if running_time >= box_width:
                end_index[j] = end_index[j - 1]
            else:
                request_position = end_index[j - 1] + 1
                while running_time < box_width and request_position < n:
                    if is_a_hit[request_position]:
                        running_time += 1
                    else:
                        running_time += miss_cost
                    request_position += 1
                end_index[j] = request_position - 1

            if (j, end_index[j] + 1) in edge2box:
                dag.remove_edge(j, end_index[j] + 1)  # smaller box is preferred
            dag.add_edge(j, end_index[j] + 1,
                         weight=memory_size * miss_cost * 3 * memory_size)
            edge2box[(j, end_index[j] + 1)] = i
    #########################################

    print('searching the shortest path...')
    nodes_sorted = list(nx.topological_sort(dag))
    build_path = {}
    for v in nodes_sorted:
        build_path[v] = (math.inf, -1)  # (distance to the beginning, predecessor)
    build_path[0] = (0, -1)
    for u in nodes_sorted:
        for v in list(dag.successors(u)):
            if build_path[v][0] > build_path[u][0] + dag.edges[u, v]['weight']:
                build_path[v] = (build_path[u][0] + dag.edges[u, v]['weight'], u)

    box_start_points = [n]
    opt_box_seq = []
    while True:
        box_start_points.append(build_path[box_start_points[-1]][1])
        # box_start_points.append()
        opt_box_seq.append(int(edge2box[(box_start_points[-1], box_start_points[-2])]))
        if box_start_points[-1] == 0:
            break
    opt_impact = build_path[n][0]
    opt_box_seq = list(reversed(opt_box_seq))
    box_start_points = list(reversed(box_start_points))
    # opt_path = list(reversed(box_start_points))
    # d = nx.dijkstra_path_length(dag, source='start', target='end')
    # print(d)
    return opt_impact, opt_box_seq, box_start_points


def fixed_size(seq, cache_size, miss_cost):
    print('testing fixed size', str(cache_size))
    impact = 0
    my_cache = collections.OrderedDict()
    for req in seq:
        if req not in my_cache.keys():
            my_cache[req] = True
            impact += cache_size * miss_cost
        else:
            impact += cache_size
        my_cache.move_to_end(req, last=False)
        while len(my_cache) > cache_size:
            my_cache.popitem(last=True)

    impact += cache_size * len(my_cache) * miss_cost
    return impact
