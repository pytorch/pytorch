#!/usr/bin/env bash

# set -e

DIM1=False
DIM2=True
DIM3=False
DATE=220317
GIT_HASH=4e6dfda
LOG_DIR=${DATE}_${GIT_HASH}
SUFFIX=${GIT_HASH}


#--------------------------------------------------------------------------
#   Commandline arguments parsing
#   like: cmd -key[--key] value
#--------------------------------------------------------------------------

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d1|--d1)
    DIM1=True
    echo DIM1: "${DIM1}"
    shift # past argument
    ;;
    -d2|--d2)
    DIM2=True
    echo DIM2: "${DIM2}"
    shift # past argument
    ;;
    -d3|--d3)
    DIM3=True
    echo DIM3: "${DIM3}"
    shift # past argument
    ;;
    -all|--all)
    DIM1=True
    DIM2=True
    DIM3=True
    echo DIM1: "${DIM1}"
    echo DIM2: "${DIM2}"
    echo DIM3: "${DIM3}"
    shift # past argument
    ;;
    -s|--suffix)
    SUFFIX=${SUFFIX}_"$2"
    echo SUFFIX: "${SUFFIX}"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

#--------------------------------------------------------------------------

# NUMACTL="numactl --cpunodebind=1 --membind=1"
NUMACTL=
# ENV_CONF=
GPU=mi100
PROF_ITER_COUNT=10000
LOG_DIR_PATH=../log/${LOG_DIR}
set -x

#-------------------------------------------------------------------------------
#               1D
#-------------------------------------------------------------------------------

if [[ "${DIM1}" == "True" ]]; then
    mkdir -p ${LOG_DIR_PATH}
    echo ">>>>>>>> RUN test conv1d nwc <<<<<<<<<<"
    CMD="./../build/bin/test_conv1d_fwd"
    ${NUMACTL} ${CMD} 2>&1 \
        | tee ${LOG_DIR_PATH}/test_conv1d_fwd_nwc_${SUFFIX}_${GPU}.log

fi

#-------------------------------------------------------------------------------
#               2D
#-------------------------------------------------------------------------------

if [[ "${DIM2}" == "True" ]]; then
    mkdir -p ${LOG_DIR_PATH}
    echo ">>>>>>>> RUN test conv2d nhwc <<<<<<<<<<"
    CMD="./../build/bin/test_conv2d_fwd"
    ${NUMACTL} ${CMD} 2>&1 \
        | tee ${LOG_DIR_PATH}/test_conv2d_fwd_nhwc_${SUFFIX}_${GPU}.log

fi

#-------------------------------------------------------------------------------
#               3D
#-------------------------------------------------------------------------------

if [[ "${DIM3}" == "True" ]]; then
    mkdir -p ${LOG_DIR_PATH}
    echo ">>>>>>>> RUN test conv3d ndhwc <<<<<<<<<<"
    CMD="./../build/bin/test_conv3d_fwd"
    ${NUMACTL} ${CMD} 2>&1 \
        | tee ${LOG_DIR_PATH}/test_conv3d_fwd_ndhwc_${SUFFIX}_${GPU}.log

fi
