#!/bin/bash
## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"
OP=$1
DATATYPE=$2
LAYOUT=$3
VERIFY=$4
INIT=$5
LOG=$6
TIME=$7
 
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideD StrideE Alpha Beta
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  960  1024 1024       -1      -1      -1      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1920  2048 2048       -1      -1      -1      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 3840  4096 4096       -1      -1      -1      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 7680  8192 8192       -1      -1      -1      -1     1    1
 
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideD StrideE Alpha Beta
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  960  1024 1024       -1      -1       0      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1920  2048 2048       -1      -1       0      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 3840  4096 4096       -1      -1       0      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 7680  8192 8192       -1      -1       0      -1     1    1
 
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideD StrideE Alpha Beta
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1000  1000 1000       -1      -1       0      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2000  2000 2000       -1      -1       0      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 4000  4000 4000       -1      -1       0      -1     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 8000  8000 8000       -1      -1       0      -1     1    1
 
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideD StrideE Alpha Beta
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024     1056    1056    1056    1056     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048     2080    2080    2080    2080     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 4096  4096 4096     4128    4128    4128    4128     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 8192  8192 8192     8224    8224    8224    8224     1    1
 
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideD StrideE Alpha Beta
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024     1088    1088    1088    1088     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048     2112    2112    2112    2112     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 4096  4096 4096     4160    4160    4160    4160     1    1
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 8192  8192 8192     8256    8256    8256    8256     1    1