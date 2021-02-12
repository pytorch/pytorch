alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx -v /data:/data -v /nfs_megatron:/nfs_megatron'
alias drun_nodevice='sudo docker run -it --rm --network=host --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx -v /data:/data -v /nfs_megatron:/nfs_megatron'


# WORK_DIR=/var/lib/jenkins/pytorch
WORK_DIR='/dockerx/pytorch'

# drun -w $WORK_DIR rocm/pytorch
# drun -w $WORK_DIR rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# drun -w $WORK_DIR rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch_1.7.0
# drun -w $WORK_DIR rocm/pytorch-private:rocm-3221-pytorch-rocblas-tuned
drun -w $WORK_DIR rocm/pytorch-private:rocm-3221-pytorch-rocblas-tuned-rnnfp16-miopen
# drun -w $WORK_DIR rnn_test
