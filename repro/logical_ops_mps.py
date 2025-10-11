#!/usr/bin/env python3
import argparse
import os
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--expect-failure", action="store_true")
args = parser.parse_args()

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
    print("warning: PYTORCH_ENABLE_MPS_FALLBACK is not 0", file=sys.stderr)

if not torch.backends.mps.is_available():
    raise SystemExit("MPS backend not available")

device = torch.device("mps")
ops = ("logical_and", "logical_or", "logical_xor")

base_tensors = {
    torch.bool: torch.tensor([[True, False, True], [False, True, False]], device=device),
    torch.int32: torch.tensor([[0, 1, -2], [3, 0, 5]], dtype=torch.int32, device=device),
    torch.float32: torch.tensor([[0.0, 1.5, -2.0], [3.25, 0.0, 5.5]], device=device),
}

python_scalars = {
    torch.bool: True,
    torch.int32: 1,
    torch.float32: -1.0,
}

failures = []

def to_cpu_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()

def scalar_to_cpu_tensor(value, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return to_cpu_tensor(value)
    if dtype is torch.bool:
        return torch.tensor(value, dtype=torch.bool)
    if dtype is torch.int32:
        return torch.tensor(value, dtype=torch.int32)
    if dtype is torch.float32:
        return torch.tensor(value, dtype=torch.float32)
    raise AssertionError(f"Unhandled dtype {dtype}")

for dtype, tensor in base_tensors.items():
    scalar = python_scalars[dtype]
    other_tensor = base_tensors[dtype].clone()
    for op_name in ops:
        op = getattr(torch, op_name)
        cpu_ref = getattr(torch, op_name)
        dtype_name = str(dtype).split('.')[-1]
        case_tag = f"{op_name}[{dtype_name}]"
        expected = cpu_ref(to_cpu_tensor(tensor), to_cpu_tensor(other_tensor))
        try:
            result = op(tensor, other_tensor)
        except Exception as exc:  # tensor/tensor should always work
            raise AssertionError(f"{case_tag} tensor/tensor failed: {exc}") from exc
        if result.device.type != "mps":
            raise AssertionError(f"{case_tag} tensor/tensor did not stay on MPS")
        if not torch.equal(result.cpu(), expected):
            raise AssertionError(f"{case_tag} tensor/tensor mismatch")
        # tensor, scalar
        expect_failure = args.expect_failure
        try:
            result = op(tensor, scalar)
        except Exception as exc:
            if expect_failure:
                failures.append((case_tag, "tensor,scalar", str(exc)))
            else:
                raise AssertionError(f"{case_tag} tensor,scalar failed: {exc}") from exc
        else:
            if expect_failure:
                raise AssertionError(f"{case_tag} tensor,scalar unexpectedly succeeded")
            if result.device.type != "mps":
                raise AssertionError(f"{case_tag} tensor,scalar did not stay on MPS")
            ref_tensor = cpu_ref(to_cpu_tensor(tensor), scalar_to_cpu_tensor(scalar, dtype))
            if not torch.equal(result.cpu(), ref_tensor):
                raise AssertionError(f"{case_tag} tensor,scalar mismatch")
        # scalar, tensor
        try:
            result = op(scalar, tensor)
        except Exception as exc:
            if expect_failure:
                failures.append((case_tag, "scalar,tensor", str(exc)))
            else:
                raise AssertionError(f"{case_tag} scalar,tensor failed: {exc}") from exc
        else:
            if expect_failure:
                raise AssertionError(f"{case_tag} scalar,tensor unexpectedly succeeded")
            if result.device.type != "mps":
                raise AssertionError(f"{case_tag} scalar,tensor did not stay on MPS")
            ref_tensor = cpu_ref(scalar_to_cpu_tensor(scalar, dtype), to_cpu_tensor(tensor))
            if not torch.equal(result.cpu(), ref_tensor):
                raise AssertionError(f"{case_tag} scalar,tensor mismatch")
        # out variant uses tensor input for reference shape
        out = torch.empty_like(tensor, dtype=torch.bool, device=device)
        try:
            out_result = op(tensor, other_tensor, out=out)
        except Exception as exc:
            raise AssertionError(f"{case_tag} out=tensor failed: {exc}") from exc
        if out_result.data_ptr() != out.data_ptr():
            raise AssertionError(f"{case_tag} out=tensor did not reuse storage")
        if out_result.device.type != "mps":
            raise AssertionError(f"{case_tag} out=tensor did not stay on MPS")
        if not torch.equal(out.cpu(), expected):
            raise AssertionError(f"{case_tag} out=tensor mismatch")
        out_scalar = torch.empty_like(tensor, dtype=torch.bool, device=device)
        try:
            out_result = op(tensor, scalar, out=out_scalar)
        except Exception as exc:
            if expect_failure:
                failures.append((case_tag, "out=tensor,scalar", str(exc)))
            else:
                raise AssertionError(f"{case_tag} out=tensor,scalar failed: {exc}") from exc
        else:
            if expect_failure:
                raise AssertionError(f"{case_tag} out=tensor,scalar unexpectedly succeeded")
            if out_result.data_ptr() != out_scalar.data_ptr():
                raise AssertionError(f"{case_tag} out=tensor,scalar storage issue")
            if out_scalar.device.type != "mps":
                raise AssertionError(f"{case_tag} out=tensor,scalar did not stay on MPS")
            ref_tensor = cpu_ref(to_cpu_tensor(tensor), scalar_to_cpu_tensor(scalar, dtype))
            if not torch.equal(out_scalar.cpu(), ref_tensor):
                raise AssertionError(f"{case_tag} out=tensor,scalar mismatch")

if args.expect_failure:
    if not failures:
        raise SystemExit("Expected failures but none were observed")
    print("Observed expected failures:")
    for tag, variant, message in failures:
        print(f" - {tag} {variant}: {message}")
else:
    print("All logical op scenarios succeeded on MPS without fallback.")
