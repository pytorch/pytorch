"""
This global flag controls whether to change the existing parameters
in-place instead of assigning new tensors to the parameters when
moving a `nn.Module` between CPU and GPU.
"""
change_nn_module_params_inplace_cpu_cuda = True