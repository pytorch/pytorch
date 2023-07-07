# counter for tracking how many kernels have been generated
generated_kernel_count = 0
generated_cpp_vec_kernel_count = 0
num_bytes_accessed = 0
nodes_num_elem = []

# counters for tracking fusions
ir_nodes_pre_fusion = 0

# counters for tracking to_dtype inserted
cpp_to_dtype_count = 0

# counters for tracking cpp_wrapper disabled
disable_cpp_wrapper = 0


# reset all counters
def reset():
    global generated_kernel_count
    global generated_cpp_vec_kernel_count
    global num_bytes_accessed, nodes_num_elem
    global ir_nodes_pre_fusion
    global cpp_to_dtype_count
    global disable_cpp_wrapper

    generated_kernel_count = 0
    generated_cpp_vec_kernel_count = 0
    num_bytes_accessed = 0
    nodes_num_elem.clear()
    ir_nodes_pre_fusion = 0
    cpp_to_dtype_count = 0
    disable_cpp_wrapper = 0
