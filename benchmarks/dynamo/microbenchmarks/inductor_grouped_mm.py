# FIXME: move this to tritonbench project.

import triton
import triton.testing

import torch


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[
        0
    ] == 10 and torch.cuda.get_device_capability()[1] in [0, 3]


def benchmark_grouped_mm(problem_sizes=None):
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.bfloat16
    align = 16 // dtype.itemsize

    if problem_sizes is None:
        problem_sizes = [
            [5, 2, 16, 16],
            [13, 3, 16, 32],
            [128, 8, 16, 16],
            [253, 7, 24, 24],
            [512, 8, 32, 64],
            [1024, 16, 256, 1024],
            [2048, 32, 512, 256],
            [2048, 32, 512, 2048],
            [4834, 24, 5120, 1536],
            [8257, 32, 5120, 1536],
            [32768, 24, 6144, 2048],
            [32768, 48, 6144, 2048],
            [32768, 64, 6144, 2048],
            [65536, 24, 6144, 2048],
            [65536, 32, 6144, 2048],
            [65536, 48, 6144, 2048],
            [65536, 64, 6144, 2048],
            [131072, 24, 6144, 2048],
            [131072, 32, 6144, 2048],
            [131072, 48, 6144, 2048],
            [131072, 64, 6144, 2048],
        ]

    results = []

    for M, G, N, K in problem_sizes:
        K_align = (K + align - 1) // align * align

        A = torch.randn(M, K_align, device=device, dtype=dtype)[:, :K]
        B = torch.randn(G, N, K_align, device=device, dtype=dtype)[:, :, :K]
        offs = torch.arange(M // G, M + 1, M // G, device=device, dtype=torch.int32)
        if offs[-1] != M:
            offs[-1] = M

        print(f"M={M}, G={G}, N={N}, K={K}")

        flops = 2 * M * N * K
        result = {"M": M, "G": G, "N": N, "K": K}

        C_ref = torch._grouped_mm(A, B.transpose(-2, -1), offs)

        fn_aten = lambda: torch._grouped_mm(A, B.transpose(-2, -1), offs)  # noqa: E731
        us_aten = triton.testing.do_bench(fn_aten, warmup=2, rep=20) * 1e3
        tflops_aten = flops * 1e-12 / (us_aten * 1e-6)
        print(f"  ATen: {us_aten:.2f} us ({tflops_aten:.2f} TFLOPS)")
        result["ATen (us)"] = us_aten

        try:
            torch._dynamo.reset()
            compiled_triton = torch.compile(
                torch._grouped_mm,
                options={"max_autotune": True, "max_autotune_gemm_backends": "TRITON"},
            )
            fn_triton = lambda: compiled_triton(A, B.transpose(-2, -1), offs)  # noqa: E731
            us_triton = triton.testing.do_bench(fn_triton, warmup=2, rep=20) * 1e3
            tflops_triton = flops * 1e-12 / (us_triton * 1e-6)
            print(f"  Triton: {us_triton:.2f} us ({tflops_triton:.2f} TFLOPS)")
            result["Triton (us)"] = us_triton
            result["Triton speedup"] = us_aten / us_triton

            try:
                C_triton = compiled_triton(A, B.transpose(-2, -1), offs)
                torch.testing.assert_close(C_triton, C_ref, rtol=1e-2, atol=1e-2)
                print("  ✓ Triton correctness check passed")
            except AssertionError:
                print("  ✗ Triton correctness check FAILED")
        except Exception as e:
            print(f"  Triton: Failed ({e})")

        if is_blackwell():
            try:
                torch._dynamo.reset()
                compiled_cutedsl = torch.compile(
                    torch._grouped_mm,
                    options={
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "CUTEDSL",
                    },
                    dynamic=False,
                )
                fn_cutedsl = lambda: compiled_cutedsl(A, B.transpose(-2, -1), offs)  # noqa: E731
                us_cutedsl = triton.testing.do_bench(fn_cutedsl, warmup=2, rep=20) * 1e3
                tflops_cutedsl = flops * 1e-12 / (us_cutedsl * 1e-6)
                print(f"  CuTeDSL: {us_cutedsl:.2f} us ({tflops_cutedsl:.2f} TFLOPS)")
                result["CuTeDSL (us)"] = us_cutedsl
                result["CuTeDSL speedup"] = us_aten / us_cutedsl

                try:
                    C_cutedsl = compiled_cutedsl(A, B.transpose(-2, -1), offs)
                    torch.testing.assert_close(C_cutedsl, C_ref, rtol=1e-2, atol=1e-2)
                    print("  ✓ CuTeDSL correctness check passed")
                except AssertionError:
                    print("  ✗ CuTeDSL correctness check FAILED")
            except Exception as e:
                print(f"  CuTeDSL: Failed ({e})")

            try:
                torch._dynamo.reset()
                compiled_gluon = torch.compile(
                    torch._grouped_mm,
                    options={
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "GLUON",
                    },
                )
                fn_gluon = lambda: compiled_gluon(A, B.transpose(-2, -1), offs)  # noqa: E731
                us_gluon = triton.testing.do_bench(fn_gluon, warmup=2, rep=20) * 1e3
                tflops_gluon = flops * 1e-12 / (us_gluon * 1e-6)
                print(f"  Gluon: {us_gluon:.2f} us ({tflops_gluon:.2f} TFLOPS)")
                result["Gluon (us)"] = us_gluon
                result["Gluon speedup"] = us_aten / us_gluon

                try:
                    C_gluon = compiled_gluon(A, B.transpose(-2, -1), offs)
                    torch.testing.assert_close(C_gluon, C_ref, rtol=1e-2, atol=1e-2)
                    print("  ✓ Gluon correctness check passed")
                except AssertionError:
                    print("  ✗ Gluon correctness check FAILED")
            except Exception as e:
                print(f"  Gluon: Failed ({e})")

        results.append(result)
        print()

    import pandas as pd

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    return results


if __name__ == "__main__":
    benchmark_grouped_mm()
