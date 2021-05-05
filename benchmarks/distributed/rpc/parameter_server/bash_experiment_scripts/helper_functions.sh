run_benchmark_basic() {
    # requires slurm
    gpurun='srun -p q2 --cpus-per-task=16 -t 5:00:00 --gpus-per-node=4'
    $gpurun python launcher.py --bconfig_id=$1 --dconfig_id=$2 --mconfig_id=$3 --tconfig_id=$4 --pconfig_id=$5
}
