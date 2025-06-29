#!/bin/bash
echo "M, K, N, benchmarking, precompiling, num_choices" > /home/gabeferns/pt-envs/at/benchmarks/dynamo/autotune_results.csv && grep -A 1000 "AUTOTUNE mm(" "$1" | grep -B 1000 "SingleProcess AUTOTUNE benchmarking takes" | awk '
BEGIN { RS="AUTOTUNE mm\\(" }
NR > 1 {
    # Extract dimensions from the first line
    if (match($0, /^([0-9]+)x([0-9]+), ([0-9]+)x([0-9]+)\)/, dims)) {
        dim1 = dims[1]
        dim2 = dims[2]
        dim3 = dims[4]

        # Find the benchmarking line
        if (match($0, /SingleProcess AUTOTUNE benchmarking takes ([0-9.]+) seconds and ([0-9.]+) seconds precompiling for ([0-9]+) choices/, times)) {
            bench_time = times[1]
            precomp_time = times[2]
            choices = times[3]

            print dim1 ", " dim2 ", " dim3 ", " bench_time ", " precomp_time ", " choices
        }
    }
}'
