#!/bin/bash

## GPU visibility
export HIP_VISIBLE_DEVICES=0
DRIVER="../build/bin/ckProfiler"
echo $DRIVER
OP=$1
DATATYPE=$2
LAYOUT=$3
VERIFY=$4
INIT=$5
LOG=$6
TIME=$7
KBatch=$8


# 120 CU
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC  KBatch_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  960  1024 1024       -1     -1      -1   $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  960  2048 2048       -1     -1      -1   $KBatch
 
# 104 CU
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC KBatch_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  832  1024 1024       -1     -1      -1  $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME  832  2048 2048       -1     -1      -1  $KBatch
 
# 110 CU
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC KBatch_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1280  1408 1024       -1     -1      -1  $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1280  2816 2048       -1     -1      -1  $KBatch

# testing different strides
########  op  datatype  layout  verify  init  log  time  M___ N___ K___  StrideA StrideB StrideC KBatch_
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024	    1024   1024    1024  $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048	    2048   2048    2048  $KBatch
 
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024	    1056   1056    1056  $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048	    2080   2080    2080  $KBatch
 
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 1024  1024 1024	    1088   1088    1088  $KBatch
 $DRIVER $OP $DATATYPE $LAYOUT $VERIFY $INIT $LOG $TIME 2048  2048 2048	    2112   2112    2112  $KBatch
