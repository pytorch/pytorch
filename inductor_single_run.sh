export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

THREAD=${1:-multiple} # multiple / single / all
MODE=${2:-inference} # inference / training
SCENARIO=${3:-accuracy} # accuracy / performance
SUITE=${4:-huggingface} # torchbench / huggingface / timm_models
MODEL=${5:-GoogleFnet}
DT=${6:-float32} # float32 / amp
CHANNELS=${7:-first} # first / last
SHAPE=${8:-static} # static / dynamic
WRAPPER=${9:-default} # default / cpp
BS=${10:-0}
BACKEND=${11:-inductor}

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

# Wrapper_extra=""
if [[ $WRAPPER == "cpp" ]]; then
    echo "Testing with cpp wrapper."
    export TORCHINDUCTOR_CPP_WRAPPER=1
fi

Channels_extra=""
if [[ ${CHANNELS} == "last" ]]; then
    Channels_extra="--channels-last "
fi

BS_extra=""
if [[ ${BS} -gt 0 ]]; then
    BS_extra="--batch_size=${BS} "
fi

DT_extra=''
if [[ "$DT" == "amp_fp16" ]]; then
    DT=amp
    DT_extra="--amp-dtype float16 "
fi

cd ./benchmarks/dynamo
if [[ $BACKEND == "aot_inductor" ]]; then
    # Workaround for test with runner.py
    sed -i '/"inference": {/a \ \ \ \ \ \ \ \ "aot_inductor": "--inference -n50 --export-aot-inductor ",' runner.py    
fi

echo "Testing with ${BACKEND}."
Backend_extra="$(python -c "import runner; print(runner.TABLE['${MODE}']['${BACKEND}'])") "

Flag_extra=""
if [[ ${MODE} == "inference" ]]; then
    export TORCHINDUCTOR_FREEZING=1
    Flag_extra+="--freezing "
fi

cd ../..

cpu_allowed_list=$(cat /proc/self/status | grep Cpus_allowed_list | awk '{print $2}')
start_core=$(echo ${cpu_allowed_list} | awk -F- '{print $1}')
mem_allowed_list=$(cat /proc/self/status | grep Mems_allowed_list | awk '{print $2}')
CORES_PER_SOCKET=$(lscpu | grep Core | awk '{print $4}')
NUM_SOCKET=$(lscpu | grep "Socket(s)" | awk '{print $2}')
NUM_NUMA=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
CORES=$(expr $CORES_PER_SOCKET \* $NUM_SOCKET / $NUM_NUMA)
if [[ ${mem_allowed_list} =~ '-' ]];then
    end_core=$(expr ${start_core} + ${CORES} - 1)
    cpu_allowed_list="${start_core}-${end_core}"
    mem_allowed_list=$(echo ${mem_allowed_list} | awk -F- '{print $1}')
fi

multi_threads_test() {
    export OMP_NUM_THREADS=$CORES
    end_core=$(expr $CORES - 1)    
    numactl -C ${cpu_allowed_list} --membind=${mem_allowed_list} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} -dcpu --no-skip --dashboard --only "${MODEL}" ${Channels_extra} ${BS_extra} ${Shape_extra} ${Flag_extra} ${Backend_extra} ${DT_extra} --timeout 9000 --output=/tmp/inductor_single_test_mt.csv && \
    cat /tmp/inductor_single_test_mt.csv && rm /tmp/inductor_single_test_mt.csv
}

single_thread_test() {
    export OMP_NUM_THREADS=1
    numactl -C ${start_core}-${start_core} --membind=${mem_allowed_list} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} -dcpu --no-skip --dashboard --batch-size 1 --threads 1 --only "${MODEL}" ${Channels_extra} ${Shape_extra} ${Flag_extra} ${Backend_extra} ${DT_extra} --timeout 9000 --output=/tmp/inductor_single_test_st.csv && \
    cat /tmp/inductor_single_test_st.csv && rm /tmp/inductor_single_test_st.csv
}


if [[ $THREAD == "multiple" ]]; then
    echo "multi-threads testing...."
    multi_threads_test
elif [[ $THREAD == "single" ]]; then
    echo "single-thread testing...."
    single_thread_test
elif [[ $THREAD == "all" ]]; then
    echo "1. multi-threads testing...."
    multi_threads_test
    echo "2. single-thread testing...."
    single_thread_test
else
    echo "Please check thread mode with multiple / single / all"
fi
