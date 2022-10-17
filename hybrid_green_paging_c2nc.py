import other_algorithms_c2nc as oa
import xgboost_module_c2nc as xgbm
import collections
from tool import run_box_nc
import xgboost as xgb


def hybrid_green_paging(sequence, k, miss_cost, number_of_box_kinds,
                        model, window_size,
                        hybrid_code, alpha, adversarial,
                        check_length, xgb_req2mi=None):
    # simulate ML if needed
    if xgb_req2mi is None:
        xgb_req2mi = []
        _, _, ttt = xgbm.test_xgboost_nc(model, sequence, k, miss_cost, window_size,
                                         adversarial)
        for tt in ttt:
            xgb_req2mi.append(tt)
    # simulate soda
    _, _, soda_req2box, soda_req2counter, soda_req2mi = oa.deterministic_algorithm(sequence,
                                                                                   k,
                                                                                   number_of_box_kinds,
                                                                                   miss_cost)

    soda_box_counting = [0 for _ in range(number_of_box_kinds)]
    soda_box_counting[0] = 1
    soda_current_box = 0
    pointer = 0
    total_impact = 0
    run_oracle = True
    seg = 0

    my_cache = collections.OrderedDict()
    previous_size = 0
    if adversarial:
        cache_size = k
    else:
        feature = xgb.DMatrix([xgbm.get_feature_vector(sequence, 0, window_size)])
        oracle = model.predict(feature)[0]
        cache_size = k / (2 ** oracle)
    while pointer < len(sequence):
        my_cache, total_impact, pointer = run_box_nc(my_cache, cache_size, total_impact, miss_cost,
                                                     previous_size, pointer,
                                                     sequence)
        # more oracle?
        if int(pointer / check_length) > seg:
            # print(pointer)
            # print(len(xgb_req2mi))
            seg = int(pointer / check_length)
            mlmi = xgb_req2mi[pointer - 1]
            if hybrid_code == '1.1':
                if run_oracle and mlmi > alpha * soda_req2mi[pointer - 1]:
                    soda_current_box = soda_req2box[pointer]
                    soda_box_counting = soda_req2counter[pointer]
                run_oracle = mlmi <= alpha * soda_req2mi[pointer - 1]
            elif hybrid_code == '1.2':
                if run_oracle and mlmi > alpha * soda_req2mi[pointer - 1]:
                    soda_current_box = soda_req2box[pointer]
                    soda_box_counting = soda_req2counter[pointer]
                if run_oracle:
                    run_oracle = mlmi <= alpha * soda_req2mi[pointer - 1]
                else:
                    run_oracle = alpha * mlmi < soda_req2mi[pointer - 1]
            else:
                if run_oracle and total_impact / soda_req2mi[pointer - 1] > alpha:
                    soda_current_box = soda_req2box[pointer]
                    soda_box_counting = soda_req2counter[pointer]
                run_oracle = (total_impact / soda_req2mi[pointer - 1]) <= alpha

        # get new box
        previous_size = cache_size
        if run_oracle:
            if adversarial:
                cache_size = k
            else:
                feature = xgb.DMatrix([xgbm.get_feature_vector(sequence, pointer, window_size)])
                oracle = model.predict(feature)[0]
                cache_size = k / (2 ** oracle)
        else:
            cache_size = k / (2 ** (number_of_box_kinds - soda_current_box - 1))
            if soda_current_box == number_of_box_kinds - 1:
                soda_current_box = 0
            elif soda_box_counting[soda_current_box] % 4 == 0:
                soda_current_box = soda_current_box + 1
            else:
                soda_current_box = 0
            soda_box_counting[soda_current_box] += 1

    # flush out the last box
    total_impact += (len(my_cache) * previous_size * miss_cost)
    return total_impact
