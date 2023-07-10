#!/bin/bash

## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"

OP=$1
DATATYPE=$2
IN_LAYOUT=$3
WEI_LAYOUT=$4
OUT_LAYOUT=$5
VERIFY=$6
INIT=$7
LOG=$8
TIME=$9

 N=${10}

# Resnet50
######## op____________________  datatype  in_layout   wei_layout  out_layout  verify  init  log  time  N__ K___ C___ Y X Hi__ Wi__ Strides Dilations LeftPads RightPads
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N   64    3 7 7  224 224    2   2     1   1    3   3     3   3
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N   64   64 1 1   56  56    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N   64   64 3 3   56  56    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256   64 1 1   56  56    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N   64  256 1 1   56  56    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N   64   64 3 3   56  56    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256   64 1 1   56  56    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N   64  256 1 1   56  56    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N   64   64 3 3   56  56    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256   64 1 1   56  56    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  256 1 1   56  56    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  128 3 3   56  56    2   2     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  128 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  512 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  128 3 3   28  28    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  128 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  512 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  128 3 3   28  28    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  128 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  512 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  128  128 3 3   28  28    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  128 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256  512 1 1   28  28    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256  256 3 3   28  28    2   2     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 1024  256 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256 1024 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256  256 3 3   14  14    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 1024  256 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256 1024 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256  256 3 3   14  14    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 1024  256 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256 1024 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256  256 3 3   14  14    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 1024  256 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256 1024 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256  256 3 3   14  14    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 1024  256 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256 1024 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  256  256 3 3   14  14    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 1024  256 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512 1024 1 1   14  14    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  512 3 3   14  14    2   2     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 2048  512 1 1    7   7    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512 2048 1 1    7   7    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  512 3 3    7   7    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 2048  512 1 1    7   7    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512 2048 1 1    7   7    1   1     1   1    0   0     0   0
 $DRIVER conv_fwd_bias_relu     $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N  512  512 3 3    7   7    1   1     1   1    1   1     1   1
 $DRIVER conv_fwd_bias_relu_add $DATATYPE $IN_LAYOUT  $WEI_LAYOUT $OUT_LAYOUT $VERIFY $INIT $LOG $TIME   $N 2048  512 1 1    7   7    1   1     1   1    0   0     0   0
