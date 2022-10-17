#!/usr/bin/python
# !/usr/bin/env python

def run_lru(cache_dict, cache_size, seq, start, box_width, hit_cost, miss_cost):
    i = start
    remain_width = box_width
    while remain_width > 0:
        if seq[i] in cache_dict.keys():
            remain_width = remain_width - hit_cost
            cache_dict.move_to_end(seq[i], last=False)
        else:
            remain_width = remain_width - miss_cost  # burst is allowed
            cache_dict[seq[i]] = True
            cache_dict.move_to_end(seq[i], last=False)
        while len(cache_dict.keys()) > cache_size:
            cache_dict.popitem(last=True)
        i = i + 1
        if i == len(seq):
            break
    return i, remain_width
    # Where the box ends/The position of the next request in the sequence
