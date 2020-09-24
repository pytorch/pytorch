from IPython import get_ipython



ipython = get_ipython()

print("------ BENCHMARK MATVEC -------")

for m in [1000, 5000, 10000, 25000]:
    for nnz_ratio in [0.01, 0.05, 0.1, 0.2, 0.3]:
        nnz_num = int(m * m * nnz_ratio)

        gcs = gen_sparse_gcs((m, m), nnz_num)
        vec = torch.randn(m, dtype=torch.double)

        coo = gen_sparse_coo((m, m), nnz_num)

        print("=====================================")
        print(f"SIZE: {m}x{m}, NNZ ratio: {nnz_ratio}")
        print(f"\tGCS: ")
        ipython.magic("timeit gcs.matmul(vec)")

        print(f"\tCOO: ")
        ipython.magic("timeit coo.matmul(vec)")
        print("=====================================")

print("------------------------------")
