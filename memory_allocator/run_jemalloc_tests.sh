names="alexnet
deeplabv3_mobilenet_v3_large
deeplabv3_resnet101
deeplabv3_resnet50
densenet121
densenet161
densenet169
densenet201
efficientnet_b0
efficientnet_b1
efficientnet_b2
efficientnet_b3
efficientnet_b4
efficientnet_b5
efficientnet_b6
efficientnet_b7
fcn_resnet101
fcn_resnet50
googlenet
inception_v3
lraspp_mobilenet_v3_large
mnasnet0_5
mnasnet0_75
mnasnet1_0
mnasnet1_3
mobilenet_v2
mobilenet_v3_large
mobilenet_v3_small
regnet_x_16gf
regnet_x_1_6gf
regnet_x_32gf
regnet_x_3_2gf
regnet_x_400mf
regnet_x_800mf
regnet_x_8gf
regnet_y_16gf
regnet_y_1_6gf
regnet_y_32gf
regnet_y_3_2gf
regnet_y_400mf
regnet_y_800mf
regnet_y_8gf
resnet101
resnet152
resnet18
resnet34
resnet50
resnext101_32x8d
resnext50_32x4d
shufflenet_v2_x0_5
shufflenet_v2_x1_0
shufflenet_v2_x1_5
shufflenet_v2_x2_0
squeezenet1_0
squeezenet1_1
vgg11
vgg11_bn
vgg13
vgg13_bn
vgg16
vgg16_bn
vgg19
vgg19_bn
wide_resnet101_2
wide_resnet50_2"
#export OPENBLAS_NUM_THREADS=$((64/NUM_WORKERS))
#export GOTO_NUM_THREADS=$((64/NUM_WORKERS))
#export OMP_NUM_THREADS=$((64/NUM_WORKERS))
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DEBUG_CPU_TYPE=5

for NAME in $names ; do
  mkdir -p "/home/mlevental/dev_projects/pytorch_memory_planning/je_malloc_runs/${NAME}/"
  mkdir -p "/home/mlevental/dev_projects/pytorch_memory_planning/perf_records/${NAME}/"
  for BATCH_SIZE in 1 ; do
    for HW in 64 ; do
      NUM_LOOPS=1
      for i in 0 1 2 3 4 5 6 ; do
        NUM_WORKERS=$((2**i))
        echo $NAME $NUM_WORKERS $NUM_LOOPS $BATCH_SIZE $HW

#        OVERSIZED_THRESHOLD=-1 ../cmake-build-debug/bin/pytorch_memory_allocator $NAME $NUM_WORKERS $NUM_LOOPS $BATCH_SIZE $HW
        OVERSIZED_THRESHOLD=-1 perf record -g ../build/bin/pytorch_memory_allocator $NAME $NUM_WORKERS $NUM_LOOPS $BATCH_SIZE $HW
        mv perf.data "/home/mlevental/dev_projects/pytorch_memory_planning/perf_records/${NAME}/normal_${NUM_WORKERS}_${NUM_LOOPS}_${BATCH_SIZE}_${HW}.perf.data"

#        OVERSIZED_THRESHOLD=0 JEMALLOC_CONF="narenas:1,tcache:false" ../cmake-build-debug/bin/pytorch_memory_allocator $NAME $NUM_WORKERS $NUM_LOOPS $BATCH_SIZE $HW
        OVERSIZED_THRESHOLD=0 JEMALLOC_CONF="narenas:1,tcache:false" perf record -g ../build/bin/pytorch_memory_allocator $NAME $NUM_WORKERS $NUM_LOOPS $BATCH_SIZE $HW
        mv perf.data "/home/mlevental/dev_projects/pytorch_memory_planning/perf_records/${NAME}/one_arena_${NUM_WORKERS}_${NUM_LOOPS}_${BATCH_SIZE}_${HW}.perf.data"
      done
    done
  done
done

