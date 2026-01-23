# FIXME: move this to tritonbench project.

import argparse

import triton
import triton.testing

import torch


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[
        0
    ] == 10 and torch.cuda.get_device_capability()[1] in [0, 3]


def _major_label(is_k_major, other_major):
    return "k-major" if is_k_major else f"{other_major}-major"


def _normalize_major(value):
    return value.replace("-", "").replace("_", "")


def _parse_tensor_spec(value, allowed_majors, expected_str, example):
    value = value.lower()
    try:
        dim_str, layout_str = value.split(":")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_str} (e.g., {example})."
        ) from exc
    if dim_str not in {"2d", "3d"}:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_str} (e.g., {example})."
        )
    major_norm = _normalize_major(layout_str)
    if major_norm not in allowed_majors:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_str} (e.g., {example})."
        )
    return (int(dim_str[0]), major_norm == "kmajor")


def _parse_a_spec(value):
    return _parse_tensor_spec(
        value,
        {"kmajor", "mmajor"},
        "'<2d|3d>:<k-major|m-major>'",
        "2d:k-major",
    )


def _parse_b_spec(value):
    return _parse_tensor_spec(
        value,
        {"kmajor", "nmajor"},
        "'<2d|3d>:<k-major|n-major>'",
        "3d:k-major",
    )


def _parse_input_dtype(value):
    value = value.lower()
    if value != "bf16":
        raise argparse.ArgumentTypeError(
            "Only bf16 is supported for --input-dtype for now."
        )
    return value


def _parse_gmnk(value):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Invalid gmnk '{value}'. Expected G,M,N,K.")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid gmnk '{value}'. Expected integers."
        ) from exc
    if any(value <= 1 for value in values):
        raise argparse.ArgumentTypeError(
            f"Invalid gmnk '{value}'. Expected G,M,N,K > 1."
        )
    return values


def _generate_offsets(total, groups, device, align=1):
    if total <= 0:
        return torch.zeros(groups, device=device, dtype=torch.int32)
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")

    if align == 1:
        probs = torch.full((groups,), 1.0 / groups, device=device)
        counts = torch.distributions.Multinomial(
            total_count=total, probs=probs
        ).sample()
        counts = counts.to(dtype=torch.int64)
    else:
        units = total // align
        remainder = total - units * align
        probs = torch.full((groups,), 1.0 / groups, device=device)
        if units == 0:
            counts = torch.zeros(groups, device=device, dtype=torch.int64)
        else:
            counts = torch.distributions.Multinomial(
                total_count=units, probs=probs
            ).sample()
            counts = counts.to(dtype=torch.int64) * align
        counts[-1] += remainder

    return torch.cumsum(counts, dim=0).to(dtype=torch.int32)


def benchmark_grouped_mm(
    gmnk=None,
    a_dim=2,
    a_k_major=True,
    b_dim=3,
    b_k_major=True,
    dtype=None,
    seed=0,
    rtol=1e-2,
    atol=1e-2,
):
    torch.manual_seed(seed)

    device = "cuda"
    if dtype is None:
        dtype = torch.bfloat16
    align = 16 // dtype.itemsize

    if gmnk is None:
        gmnk = [
            [2, 5, 16, 16],
            [3, 13, 16, 32],
            [8, 128, 16, 16],
            [7, 253, 24, 24],
            [8, 512, 32, 64],
            [16, 1024, 256, 1024],
            [32, 2048, 512, 256],
            [32, 2048, 512, 2048],
            [24, 4834, 5120, 1536],
            [32, 8257, 5120, 1536],
            [24, 32768, 6144, 2048],
            [48, 32768, 6144, 2048],
            [64, 32768, 6144, 2048],
            [24, 65536, 6144, 2048],
            [32, 65536, 6144, 2048],
            [48, 65536, 6144, 2048],
            [64, 65536, 6144, 2048],
            [24, 131072, 6144, 2048],
            [32, 131072, 6144, 2048],
            [48, 131072, 6144, 2048],
            [64, 131072, 6144, 2048],
        ]
        if a_dim == 2 and b_dim == 2:
            gmnk = [[g, k, n, m] for g, m, n, k in gmnk]
        elif a_dim == 3 and b_dim == 2:
            gmnk = [[g, n, m, k] for g, m, n, k in gmnk]
        elif a_dim == 3 and b_dim == 3:
            gmnk = [[g, m // g, n, k] for g, m, n, k in gmnk]

    results = []

    for G, M, N, K in gmnk:
        K_align = (K + align - 1) // align * align
        M_align = (M + align - 1) // align * align
        N_align = (N + align - 1) // align * align

        a_is_2d = a_dim == 2
        b_is_2d = b_dim == 2

        if a_is_2d:
            if a_k_major:
                A = torch.randn(M, K_align, device=device, dtype=dtype)[:, :K]
            else:
                A = torch.randn(K, M_align, device=device, dtype=dtype).t()[:M, :]
        else:
            if a_k_major:
                A = torch.randn(G, M, K_align, device=device, dtype=dtype)[:, :, :K]
            else:
                A = torch.randn(G, K, M_align, device=device, dtype=dtype).transpose(
                    -2, -1
                )[:, :M, :]

        if b_is_2d:
            if b_k_major:
                B = torch.randn(N, K_align, device=device, dtype=dtype)[:, :K]
            else:
                B = torch.randn(K, N_align, device=device, dtype=dtype).t()[:N, :]
        else:
            if b_k_major:
                B = torch.randn(G, N, K_align, device=device, dtype=dtype)[:, :, :K]
            else:
                B = torch.randn(G, K, N_align, device=device, dtype=dtype).transpose(
                    -2, -1
                )[:, :N, :]

        if a_is_2d and b_is_2d:
            offs_align = 1 if (not a_k_major and not b_k_major) else align
            offs = _generate_offsets(K, G, device, align=offs_align)
        elif a_is_2d and not b_is_2d:
            offs_align = 1 if a_k_major else align
            offs = _generate_offsets(M, G, device, align=offs_align)
        elif not a_is_2d and b_is_2d:
            offs = _generate_offsets(N, G, device, align=align)
        else:
            offs = None

        print(f"G={G}, M={M}, N={N}, K={K}")

        flops = 2 * M * N * K
        result = {
            "G": G,
            "M": M,
            "N": N,
            "K": K,
            "A dim": a_dim,
            "B dim": b_dim,
            "A layout": _major_label(a_k_major, "m"),
            "B layout": _major_label(b_k_major, "n"),
        }

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
            fn_triton = lambda: compiled_triton(  # noqa: E731
                A, B.transpose(-2, -1), offs
            )
            us_triton = triton.testing.do_bench(fn_triton, warmup=2, rep=20) * 1e3
            tflops_triton = flops * 1e-12 / (us_triton * 1e-6)
            print(f"  Triton: {us_triton:.2f} us ({tflops_triton:.2f} TFLOPS)")
            result["Triton (us)"] = us_triton
            result["Triton speedup"] = us_aten / us_triton

            try:
                C_triton = compiled_triton(A, B.transpose(-2, -1), offs)
                torch.testing.assert_close(C_triton, C_ref, rtol=rtol, atol=atol)
                print("  ✓ Triton correctness check passed")
            except AssertionError:
                print("  ✗ Triton correctness check FAILED")
        except Exception as e:
            print(f"  Triton: Failed ({e})")

        if is_blackwell():
            if a_dim == 2 and b_dim == 3:
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
                    fn_cutedsl = lambda: compiled_cutedsl(  # noqa: E731
                        A, B.transpose(-2, -1), offs
                    )
                    us_cutedsl = (
                        triton.testing.do_bench(fn_cutedsl, warmup=2, rep=20) * 1e3
                    )
                    tflops_cutedsl = flops * 1e-12 / (us_cutedsl * 1e-6)
                    print(
                        f"  CuTeDSL: {us_cutedsl:.2f} us ({tflops_cutedsl:.2f} TFLOPS)"
                    )
                    result["CuTeDSL (us)"] = us_cutedsl
                    result["CuTeDSL speedup"] = us_aten / us_cutedsl

                    try:
                        C_cutedsl = compiled_cutedsl(A, B.transpose(-2, -1), offs)
                        torch.testing.assert_close(
                            C_cutedsl, C_ref, rtol=rtol, atol=atol
                        )
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
                fn_gluon = lambda: compiled_gluon(  # noqa: E731
                    A, B.transpose(-2, -1), offs
                )
                us_gluon = triton.testing.do_bench(fn_gluon, warmup=2, rep=20) * 1e3
                tflops_gluon = flops * 1e-12 / (us_gluon * 1e-6)
                print(f"  Gluon: {us_gluon:.2f} us ({tflops_gluon:.2f} TFLOPS)")
                result["Gluon (us)"] = us_gluon
                result["Gluon speedup"] = us_aten / us_gluon

                try:
                    C_gluon = compiled_gluon(A, B.transpose(-2, -1), offs)
                    torch.testing.assert_close(C_gluon, C_ref, rtol=rtol, atol=atol)
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
    parser = argparse.ArgumentParser(
        description="Benchmark grouped MM with selectable row/col-major layouts."
    )
    parser.add_argument(
        "--input-dtype",
        dest="input_dtype",
        type=_parse_input_dtype,
        default="bf16",
        help="Input dtype: bf16.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for input and offset generation.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness checks.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness checks.",
    )
    parser.add_argument(
        "--gmnk",
        nargs="+",
        type=_parse_gmnk,
        help="Problem sizes as G,M,N,K (space-separated).",
    )
    parser.add_argument(
        "--A",
        dest="a_spec",
        type=_parse_a_spec,
        default=_parse_a_spec("2d:k-major"),
        help="A spec: <2d|3d>:<k-major|m-major>.",
    )
    parser.add_argument(
        "--B",
        dest="b_spec",
        type=_parse_b_spec,
        default=_parse_b_spec("3d:k-major"),
        help="B spec: <2d|3d>:<k-major|n-major>.",
    )
    args = parser.parse_args()
    a_dim, a_k_major = args.a_spec
    b_dim, b_k_major = args.b_spec
    dtype = torch.bfloat16 if args.input_dtype == "bf16" else torch.float16
    gmnk = args.gmnk if args.gmnk is not None else None
    benchmark_grouped_mm(
        gmnk=gmnk,
        a_dim=a_dim,
        a_k_major=a_k_major,
        b_dim=b_dim,
        b_k_major=b_k_major,
        dtype=dtype,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
    )
