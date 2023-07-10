#!/bin/bash
DRIVER="../build/bin/ckProfiler"
VERIFY="-v $1"
INIT=$2
NREPEAT=$3
PRECISION=$4
##PRECISION=--half
##PRECISION=--double
##PRECISION=--int8
##PRECISION=--bf16

#### 2 - MIN,  3 - MAX,  4 - AMAX
Operations="2 4"

## for generic validation
for op in $Operations; do
    for use_idx in 0 1; do
        set -x
        #######        datatype   layout          reduce dims  op     use index    verify  init  repeats
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 0,1,2,3   -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 0         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 1         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 2         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 3         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 1,2,3     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 0,2,3     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 64,4,280,82  -R 0,1,3     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,22960    -R 0         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,22960    -R 1         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 4,1469440    -R 0         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 4,1469440    -R 1         -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        set +x
    done
done

Operations=2

## for performance evaluation (resnet50 NHWC => C)
for op in $Operations; do
    for use_idx in 0 1; do
        set -x
        #######        datatype   layout             reduce dims  op     use index    verify  init  repeats
        $DRIVER reduce $PRECISION -D 256,14,14,1024  -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,28,28,128   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,58,58,128   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,7,7,2048    -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,14,14,256   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,30,30,256   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,56,56,256   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,16,16,512   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,28,28,512   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,7,7,512     -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,56,56,64    -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 256,230,230,3   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,14,14,1024  -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,28,28,128   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,58,58,128   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,7,7,2048    -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,14,14,256   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,30,30,256   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,56,56,256   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,16,16,512   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,28,28,512   -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,7,7,512     -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        $DRIVER reduce $PRECISION -D 128,56,56,64    -R 0,1,2     -O $op -I $use_idx  $VERIFY $INIT $NREPEAT
        set +x
    done
done 

