from collections import OrderedDict
from lru import run_lru


def read_crc_seq(file):
    sequence = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if len(line) == 0:
                continue
            data = line.split(',')
            sequence.append(data[0])
    f.close()
    return sequence


def format_box_nc(t, miss_cost):
    cache = OrderedDict()
    cache_size = t.box_height
    width = 2 * cache_size * miss_cost
    start = t.box_start_pointer
    # copy the start point
    for pid in t.box_start_stack.keys():
        cache[pid] = True
        cache.move_to_end(pid, last=True)
        if len(cache) == cache_size:
            break

    end, rem_width = run_lru(cache, cache_size, t.seq,
                             start, width,
                             1, miss_cost)
    t.box_end_pointer = end
    t.box_width = width - rem_width

    # Update ending stack
    t.box_end_stack = OrderedDict()
    for pid in t.box_start_stack.keys():
        t.box_end_stack[pid] = True
        t.box_end_stack.move_to_end(pid, last=True)
        if len(t.box_end_stack) == cache_size:
            break
    for x in range(start, end):
        if t.seq[x] not in t.box_end_stack.keys():
            t.box_end_stack[t.seq[x]] = True
        t.box_end_stack.move_to_end(t.seq[x], last=False)
        while len(t.box_end_stack) > cache_size:
            t.box_end_stack.popitem(last=True)
    return t


def run_box_nc(my_cache, cache_size, total_impact,
               miss_cost, previous_size, pointer, test_seq):
    # flush out
    while len(my_cache) > cache_size:
        total_impact += (miss_cost * previous_size)
        my_cache.popitem(last=True)

    # run a new box
    start_pointer = pointer
    box_width = miss_cost * cache_size * 2
    total_impact += (box_width * cache_size)
    pointer, remain_width = run_lru(my_cache, cache_size, test_seq,
                                    pointer, box_width, 1, miss_cost)
    endpointer = pointer
    total_impact -= (remain_width * cache_size)

    # Update the cache
    for x in range(start_pointer, endpointer):
        if test_seq[x] not in my_cache.keys():
            my_cache[test_seq[x]] = True
        my_cache.move_to_end(test_seq[x], last=False)
        while len(my_cache) > cache_size:
            my_cache.popitem(last=True)

    return my_cache, total_impact, pointer
