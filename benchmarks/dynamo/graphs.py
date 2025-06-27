
from collections import defaultdict

def get_result(filename):
    result: defaultdict[tuple[int, int, int], list[tuple[int, int, int, int, int, int, float]]] = defaultdict(list)

    with open(filename, "r") as f:
        for idx, line in enumerate(f):
            vals = line.split(',')
            if len(vals) == 9:
                m, k, n, block_m, block_k, block_n, num_stages, num_warps, time = vals
                try:
                    m = int(m)
                    k = int(k)
                    n = int(n)
                    block_m = int(block_m)
                    block_k = int(block_k)
                    block_n = int(block_n)
                    num_stages = int(num_stages)
                    num_warps = int(num_warps)
                    time = float(time)
                    group_m = 8
                except ValueError:
                    print(f"Skipping line {idx} '{line}'")
                    continue
            elif len(vals) == 10:
                m, k, n, block_m, block_k, block_n, num_stages, num_warps, group_m, time = vals
                try:
                    m = int(m)
                    k = int(k)
                    n = int(n)
                    block_m = int(block_m)
                    block_k = int(block_k)
                    block_n = int(block_n)
                    num_stages = int(num_stages)
                    num_warps = int(num_warps)
                    time = float(time)
                    group_m = int(group_m)
                except ValueError:
                    print(f"Skipping line {idx} '{line}'")
                    continue
            else:
                print(f"Skipping line {idx} '{line}'")
                continue
            result[(m, k, n)].append((block_m, block_k, block_n, num_stages, num_warps, group_m, time))
    return result

exhaustive = get_result("aten.mm.default_torch.bfloat16_benchmark_results.csv")
predicted = get_result("/home/gabeferns/logs/model_inference.txt")

def sort_by_time(results):
    for k, v in results.items():
        results[k] = sorted(v, key=lambda x: x[-1])
    return results

excepted = sort_by_time(exhaustive)
predicted = sort_by_time(predicted)
    
# def compile_time_tradeoff(reference, predicted, foo):
#     returns the proportion of top foo reference configs found in top z predicted configs
#     num_shapes = len(reference)
#     for k, p in predicted.items():
#         if abs(len(p) - len(reference[k])) > 10:
#             breakpoint()
#     top_z = []
    
#     for z in range(1, 1000):
#         total_found = 0
        
#         for shape in reference.keys():
#             ref_configs = reference[shape][:foo]  # top foo reference configs
#             pred_configs = predicted[shape][:z]   # top z predicted configs
            
#             Count how many reference configs are found in predicted configs
#             found_count = 0
#             seen = False
#             for ref_config in ref_configs:
#                 for pred_config in pred_configs:
#                     if ref_config[:-1] == pred_config[:-1]:  # compare all except time
#                         found_count += 1
#                         seen = True
#                         break  # each reference config can only be matched once
#                 if seen:
#                     break
            
#             total_found += found_count
        
#         Calculate proportion of reference configs found in predicted configs
#         proportion = total_found / num_shapes
#         top_z.append(proportion)
    
#     return top_z


# ps1 = compile_time_tradeoff(excepted, predicted, foo=1)
# ps5 = compile_time_tradeoff(excepted, predicted, foo=5)
# ps10 = compile_time_tradeoff(excepted, predicted, foo=10)
# ps20 = compile_time_tradeoff(excepted, predicted, foo=20)
# ps50 = compile_time_tradeoff(excepted, predicted, foo=50)
# ps100 = compile_time_tradeoff(excepted, predicted, foo=100)
# ps200 = compile_time_tradeoff(excepted, predicted, foo=200)
# ps300 = compile_time_tradeoff(excepted, predicted, foo=300)
# ps400 = compile_time_tradeoff(excepted, predicted, foo=400)
# ps500 = compile_time_tradeoff(excepted, predicted, foo=500)
# ps700 = compile_time_tradeoff(excepted, predicted, foo=700)
# for i in range(len(ps1)):
#     print(f"{i + 1}, {ps1[i]}, {ps5[i]}, {ps10[i]}, {ps20[i]}, {ps50[i]}, {ps100[i]}, {ps200[i]}, {ps300[i]}, {ps400[i]}, {ps500[i]}, {ps700[i]}")

def load_default_configs():
    conf = set([])
    with open("/home/gabeferns/pt-envs/at/torch/_inductor/models/default_configs.txt", "r") as f:
        for line in f:
            foo = line.strip().split(",")
            tmp = foo[1]
            foo[1] = foo[2]
            foo[2] = tmp
            conf.add(tuple(int(x) for x in foo))
    return conf

def calculate_default_ranking():
    conf = load_default_configs()
    print("confs", conf)

    min_ranks = []
    for shape, configs in exhaustive.items():
        print("configs", configs[:10])
        ranks_for_shape = []
        for rank, config in enumerate(configs):
            # config format: (block_m, block_k, block_n, num_stages, num_warps, group_m, time)
            # conf format: (block_m, block_n, block_k, num_stages, num_warps) - 5 elements
            # So we need to exclude group_m and time, and compare first 5 elements
            if config[:-2] in conf:
                ranks_for_shape.append(rank)
        
        if ranks_for_shape:
            min_ranks.append(min(ranks_for_shape))
    
    return min_ranks

rankings = calculate_default_ranking()

def performance_improvement_analysis():
    """
    Analyze performance improvement of predicted configs over default configs.
    For each k in {1,5,10,20,50,100,200,300,400,500}, compare the minimum time
    of top k predicted configs against the minimum time of default configs.
    Also analyze ratios of exhaustive over default and exhaustive over top-k predicted.
    """
    conf = load_default_configs()
    k_values = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]
    
    results = {}
    
    for shape in exhaustive.keys():
        if shape not in predicted:
            continue
            
        # Get the best exhaustive time (first element since it's sorted)
        min_exhaustive_time = exhaustive[shape][0][6]  # time is at index 6
        
        # Find minimum time among default configs for this shape
        default_times = []
        for config in exhaustive[shape]:
            # config format: (block_m, block_k, block_n, num_stages, num_warps, group_m, time)
            # conf format: (block_m, block_n, block_k, num_stages, num_warps) - note the order difference
            config_tuple = (config[0], config[2], config[1], config[3], config[4])  # reorder to match conf format
            if config_tuple in conf:
                default_times.append(config[6])  # time is at index 6
        
        if not default_times:
            continue
            
        min_default_time = min(default_times)
        
        # For each k value, find minimum time among top k predicted configs
        shape_results = {
            'exhaustive_time': min_exhaustive_time,
            'default_time': min_default_time
        }
        
        for k in k_values:
            if k > len(predicted[shape]):
                k_actual = len(predicted[shape])
            else:
                k_actual = k
                
            # Get top k predicted configs
            top_k_predicted = predicted[shape][:k_actual]
            
            # Find actual times for these configs from exhaustive results
            predicted_times = []
            for pred_config in top_k_predicted:
                # Find matching config in exhaustive results
                for exh_config in exhaustive[shape]:
                    # Compare all fields except time (last element)
                    if pred_config[:-1] == exh_config[:-1]:
                        predicted_times.append(exh_config[6])  # time from exhaustive
                        break
            
            if predicted_times:
                min_predicted_time = min(predicted_times)
                shape_results[f'top_{k}_time'] = min_predicted_time
            else:
                shape_results[f'top_{k}_time'] = None
        
        results[shape] = shape_results
    
    # Print results
    print("Shape, Exhaustive_Time, Default_Time, " + ", ".join([f"Top_{k}_Time" for k in k_values]))
    
    for shape, shape_results in results.items():
        row = [f"{shape}", f"{shape_results['exhaustive_time']:.6f}", f"{shape_results['default_time']:.6f}"]
        for k in k_values:
            time_key = f'top_{k}_time'
            if shape_results[time_key] is not None:
                row.append(f"{shape_results[time_key]:.6f}")
            else:
                row.append("N/A")
        print(", ".join(row))
    
    # Calculate and print summary statistics for predicted over default
    print("\nSummary - Average speedup of predicted over default:")
    print("K_Value, Avg_Speedup, Num_Valid_Shapes")
    
    for k in k_values:
        speedups = []
        for shape_results in results.values():
            default_time = shape_results['default_time']
            predicted_time = shape_results[f'top_{k}_time']
            if predicted_time is not None and predicted_time > 0:
                speedup = default_time / predicted_time
                speedups.append(speedup)
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"{k}, {avg_speedup:.3f}, {len(speedups)}")
        else:
            print(f"{k}, N/A, 0")
    
    # Calculate and print exhaustive over default ratios
    print("\nSummary - Exhaustive over default ratios:")
    print("Avg_Ratio, Num_Valid_Shapes")
    
    exhaustive_over_default_ratios = []
    for shape_results in results.values():
        exhaustive_time = shape_results['exhaustive_time']
        default_time = shape_results['default_time']
        if exhaustive_time > 0 and default_time > 0:
            ratio = default_time / exhaustive_time
            exhaustive_over_default_ratios.append(ratio)
    
    if exhaustive_over_default_ratios:
        avg_ratio = sum(exhaustive_over_default_ratios) / len(exhaustive_over_default_ratios)
        print(f"{avg_ratio:.3f}, {len(exhaustive_over_default_ratios)}")
    else:
        print("N/A, 0")
    
    # Calculate and print exhaustive over top-k predicted ratios
    print("\nSummary - Exhaustive over top-k predicted ratios:")
    print("K_Value, Avg_Ratio, Num_Valid_Shapes")
    
    for k in k_values:
        exhaustive_over_predicted_ratios = []
        for shape_results in results.values():
            exhaustive_time = shape_results['exhaustive_time']
            predicted_time = shape_results[f'top_{k}_time']
            if exhaustive_time > 0 and predicted_time is not None and predicted_time > 0:
                ratio = predicted_time / exhaustive_time
                exhaustive_over_predicted_ratios.append(ratio)
        
        if exhaustive_over_predicted_ratios:
            avg_ratio = sum(exhaustive_over_predicted_ratios) / len(exhaustive_over_predicted_ratios)
            print(f"{k}, {avg_ratio:.3f}, {len(exhaustive_over_predicted_ratios)}")
        else:
            print(f"{k}, N/A, 0")
    
    return results

# Run the performance improvement analysis
perf_results = performance_improvement_analysis()

breakpoint()
    
            
