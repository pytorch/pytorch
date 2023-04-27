# compare different dtypes
def compare_dtype(m, k, n, batch_size, dtype):
    print(m, k, n, batch_size, dtype)
    model = Model(m, k).cuda().half().eval()
    # create input tensor
    input_tensor = torch.randint(
        2,
        (batch_size, n, k),
        device=model.linear.weight.device,
        dtype=dtype,
    )

    print("input_tensor")
    print(input_tensor)

    weight, _ = gen_two_four_sparse_mask(m, k, dtype)
    bias = torch.zeros(model.linear.bias.data.shape, dtype=dtype).cuda()

    print("weight: ")
    print(weight)

    # model.linear.weight.data = weight.float()
    # model.linear.bias.data = bias.float()

    num_bytes = weight.nelement() * weight.element_size()
    compressed_size = num_bytes * 10 // 16
    # compressed_size = 1536
    print(f"weight_compressed: {num_bytes} bytes, mask size: {compressed_size} bytes")
    weight_compressed = torch.empty(
        (compressed_size // weight.element_size(),), dtype=dtype, device=device
    )

    cslt = torch.classes.cusparselt.CusparseLtLinear(weight_compressed, bias)
    cslt.set_compressed(weight)

    print("weight compressed")
    print(weight_compressed[: m * k // 2].view(m, -1))
    print("mask")
    print(weight_compressed[m * k // 2 : m * k // 2 + compressed_size].view(m, -1))
    s_res = cslt.masked_mm(input_tensor.mT).mT

    res = (torch.matmul(weight.half(), input_tensor.mT.half()) + bias.half()).mT
    res = res.to(dtype)
    # model_res = model(input_tensor.half())
    # assert torch.allclose(model_res, res)

    print("dense result:")
    print(res)

    print("sparse result:")
    print(s_res)

    sparse_same_dense = torch.allclose(res, s_res)
    print(f"dense result - sparse result: {sparse_same_dense}")
    print(res - s_res)

    # assert torch.allclose(
    #     s_res.float(), res, rtol=1e-3, atol=1e-3
    # )
    # devnull = open("/dev/null", "w")
    # oldstdout_fno = os.dup(sys.stdout.fileno())
    # os.dup2(devnull.fileno(), 1)

    # sparse_latency = benchmark.Timer(
    #     stmt="cslt.masked_mm(input_tensor.mT).mT",
    #     globals={"input_tensor": input_tensor, "cslt": cslt}
    # ).blocked_autorange()

    # float_input_tensor = torch.clone(input_tensor.half())
    # dense_latency = benchmark.Timer(
    #     stmt="model(float_input_tensor)",
    #     globals={"model": model, "float_input_tensor": float_input_tensor}
    # ).blocked_autorange()
    # os.dup2(oldstdout_fno, 1)

    return {
        "m": m,
        "k": k,
        "n": n,
        "eval_batch_size": batch_size,
        "dtype": str(dtype),
        # "sparse_latency (ms)": sparse_latency.median * 1000,
        # "dense_latency (ms)": dense_latency.median * 1000,
        # "speedup (d/s)": dense_latency.median / sparse_latency.median,
    }


def compare_memory(m, k, n, batch_size):
    print("+" * 100)
    print(f"start: {sizeof_fmt(torch.cuda.memory_allocated())}")
    print(torch.cuda.memory_summary())
    # create dense model
    model = Model(m, k).half().cuda().eval()
    print("+" * 100)
    print(f"model: {sizeof_fmt(torch.cuda.memory_allocated())}")
    print(torch.cuda.memory_summary())

    # create input tensor
    input_tensor = torch.randn(
        batch_size,
        n,
        k,
        device=model.linear.weight.device,
        dtype=model.linear.weight.dtype,
    )
    print("+" * 100)
    print(f"input: {sizeof_fmt(torch.cuda.memory_allocated())}")
    print(torch.cuda.memory_summary())

    # get sparse model
    print(f"sparse start: {sizeof_fmt(torch.cuda.memory_allocated())}")
    pruner = WeightNormPruner(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
    pruner.step()
    print("+" * 100)
    print(f"step: {sizeof_fmt(torch.cuda.memory_allocated())}")
    print(torch.cuda.memory_summary())

    sparse_model = pruner.convert(
        model, mapping={nn.Linear: cuSPARSELtLinear}, inplace=False
    )

    sparse_model.load_state_dict(torch.load("sparse_model.pt"))

    print(model)
    print("+" * 100)
    print(f"convert: {sizeof_fmt(torch.cuda.memory_allocated())}")
    print(torch.cuda.memory_summary())

    # zero out dense tensor weights for correctness check
    pruner.squash_mask()
    model.load_state_dict(torch.load("dense_model.pt"))
    print("+" * 100)
    print(f"squash: {sizeof_fmt(torch.cuda.memory_allocated())}")
    print(torch.cuda.memory_summary())

    del pruner
    torch.cuda.empty_cache()
    print("+" * 100)
    print(f"del pruner: {sizeof_fmt(torch.cuda.memory_allocated())}")
    print(torch.cuda.memory_summary())

    assert torch.allclose(
        model(input_tensor), sparse_model(input_tensor), rtol=1e-3, atol=1e-3
    )

    alg_id = sparse_model.linear.cslt.get_alg_id()
    print(model)

    # del model
    # torch.cuda.empty_cache()
    # print("+"*100)
    # print(f"del model: {sizeof_fmt(torch.cuda.memory_allocated())}")
    # print(torch.cuda.memory_summary())

    # sparse_model(input_tensor)

    # del input_tensor
    # torch.cuda.empty_cache()
    # print("+"*100)
    # print(f"del input: {sizeof_fmt(torch.cuda.memory_allocated())}")
    # print(torch.cuda.memory_summary())

    # del sparse_model
    # torch.cuda.empty_cache()
    # print("+"*100)
    # print(f"del sparse: {sizeof_fmt(torch.cuda.memory_allocated())}")
    # print(torch.cuda.memory_summary())
    torch.save(sparse_model.state_dict(), "sparse_model.pt")
    torch.save(model.state_dict(), "dense_model.pt")
    from pprint import pprint

    pprint(torch.load("sparse_model.pt"))
    # sparse_model_2(input_tensor)

    return {
        "m": m,
        "k": k,
        "n": n,
        "eval_batch_size": batch_size,
        "init_batch_size": batch_size,
        "alg_id": alg_id,
        "sparse_model_size": sizeof_fmt(os.stat("sparse_model.pt").st_size),
        "dense_model_size": sizeof_fmt(os.stat("dense_model.pt").st_size),
    }


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


    # run a sweep for the n, batch_size combination
    # then try running batch size different from the initialized batch size to see effect of caching alg plan.
    elif args.mode == "alg-id-sweep":
        dim_range = list(range(96, 3072 + 1, 96))
        batch_sizes = list(range(4, 128 + 1, 4))
        results = [
            compare_linear(768, 3072, n, batch_size)
            for n, batch_size in tqdm(
                product(dim_range, batch_sizes), total=len(dim_range) * len(batch_sizes)
            )
        ]

        results += [
            compare_linear(768, 3072, 96, batch_size, init_batch_size=init_batch_size)
            for batch_size, init_batch_size in tqdm(
                product(batch_sizes, batch_sizes), total=len(batch_sizes) ** 2
            )
        ]


