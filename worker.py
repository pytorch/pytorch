import sys
import torch
import re
import io
import json
from contextlib import redirect_stderr

torch._inductor.config.force_disable_caches = True

DTYPE_MAP = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
    "i64": torch.int64,
}

def parse_tensor_spec(spec: str) -> torch.Tensor:
    # Format: f32[1000,1000][1000,1]cuda:0
    pattern = r"(\w+)\[([0-9,]+)\]\[([0-9,]+)\](\S+)"
    match = re.fullmatch(pattern, spec.replace(" ", ""))
    if not match:
        raise ValueError(f"Invalid tensor spec: {spec}")

    dtype_str, shape_str, stride_str, device_str = match.groups()
    dtype = DTYPE_MAP[dtype_str]
    shape = [int(x) for x in shape_str.split(",")]
    stride = [int(x) for x in stride_str.split(",")]
    device = torch.device(device_str)

    t = torch.empty_strided(shape, stride, device=device, dtype=dtype)
    t.uniform_()
    return t

def run_autotune(x: torch.Tensor, y: torch.Tensor) -> str:
    log = io.StringIO()

    @torch.compile(options={"max_autotune": True, "max_autotune_gemm_backends": "TRITON"})
    def foo(a, b):
        return a @ b

    with redirect_stderr(log):
        foo(x, y)

    for line in log.getvalue().splitlines():
        if line.strip().startswith("triton_mm_"):
            # Parse the line to extract configuration parameters
            config_line = line.strip()
            parts = config_line.split()

            # Extract parameters from the line
            config = {"template_id": "mm"}

            # Parse parameters (starting from the 3rd element)
            for param in parts[3:]:
                if '=' in param:
                    key, value = param.split('=')
                    # Convert string values to appropriate types
                    if value == 'True':
                        config[key] = True
                    elif value == 'False':
                        config[key] = False
                    elif value.isdigit():
                        config[key] = int(value)
                    else:
                        # Remove quotes if present
                        config[key] = value.strip("'")

            # Write the configuration to a JSON file
            with open('best_gemm_config.json', 'w') as f:
                json.dump(config, f, indent=4)

            return config_line

    raise Exception("No autotune result found")

def main():
    if len(sys.argv) != 3:
        print("Usage: python worker.py <tensor_spec1> <tensor_spec2>", file=sys.stderr)
        sys.exit(1)

    x = parse_tensor_spec(sys.argv[1])
    y = parse_tensor_spec(sys.argv[2])
    print(run_autotune(x, y))

if __name__ == "__main__":
    main()
