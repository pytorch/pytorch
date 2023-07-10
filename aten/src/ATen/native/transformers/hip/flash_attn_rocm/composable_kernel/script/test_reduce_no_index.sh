#!/bin/bash

## The following will be used for CI

set -x

## for float
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2,3  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,3  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,2,3  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1,2,3  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 2  0 2
bin/test_reduce_no_index -D 64,4,280,82  -R 3  0 2

## for float64
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2,3  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,3  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,2,3  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1,2,3  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 2  6 2
bin/test_reduce_no_index -D 64,4,280,82  -R 3  6 2

## for float16
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2,3  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,3  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,2,3  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1,2,3  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 2  1 2
bin/test_reduce_no_index -D 64,4,280,82  -R 3  1 2

## for int8_t
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2,3  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,3  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,2,3  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1,2,3  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 2  3 2
bin/test_reduce_no_index -D 64,4,280,82  -R 3  3 2

## for bfloat16
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2,3  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,2  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,1,3  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0,2,3  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1,2,3  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 0  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 1  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 2  5 2
bin/test_reduce_no_index -D 64,4,280,82  -R 3  5 2

set +x

