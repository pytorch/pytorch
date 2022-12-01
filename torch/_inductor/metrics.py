# counter for tracking how many kernels have been generated
generated_kernel_count = 0
generated_cpp_vec_kernel_count = 0
num_bytes_accessed = 0
nodes_num_elem = []


# reset all counters
def reset():
    global generated_kernel_count
    global generated_cpp_vec_kernel_count
    global num_bytes_accessed, nodes_num_elem

    generated_kernel_count = 0
    generated_cpp_vec_kernel_count = 0
    num_bytes_accessed = 0
    nodes_num_elem.clear()
