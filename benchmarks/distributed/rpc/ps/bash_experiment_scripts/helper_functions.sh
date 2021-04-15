run_benchmark_basic() {
    gpurun='srun -p q2 --cpus-per-task=16 -t 5:00:00 --gpus-per-node=4'
    $gpurun python benchmark.py --bconfig_id=$1 --dconfig_id=$2 --mconfig_id=$3
}