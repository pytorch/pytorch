import argparse
import asyncio
import datetime
import functools
import hashlib
import itertools
import json
import logging
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path


log = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

PYTORCH_NIGHTLY_CUDA_VERSIONS = ["12.6", "12.8", "13.0"]
PYTORCH_CUDA_VERSIONS = {
    "2.9.1": ["12.6", "12.8", "13.0"],
    "2.9.0": ["12.6", "12.8", "13.0"],
    "2.8.0": ["12.6", "12.8", "12.9"],
    "2.7.1": ["11.8", "12.6", "12.8"],
    "2.7.0": ["11.8", "12.6", "12.8"],
    "2.6.0": ["11.8", "12.4", "12.6"],
    "2.5.1": ["11.8", "12.1", "12.4"],
    "2.5.0": ["11.8", "12.1", "12.4"],
    "2.4.1": ["11.8", "12.1", "12.4"],
    "2.4.0": ["11.8", "12.1", "12.4"],
    "2.3.1": ["11.8", "12.1"],
    "2.3.0": ["11.8", "12.1"],
    "2.2.2": ["11.8", "12.1"],
    "2.2.1": ["11.8", "12.1"],
    "2.2.0": ["11.8", "12.1"],
    "2.1.2": ["11.8", "12.1"],
    "2.1.1": ["11.8", "12.1"],
    "2.1.0": ["11.8", "12.1"],
    "2.0.1": ["11.8"],
    "2.0.0": ["11.8"],
}

ENABLED_CONFIGS = [
    #    ("git:5f09e6a6c93e0b5bf75b635cddc03b85bbe85938", "12.8"),
    ("nightly", "13.0"),
    # ("2.9.1", "12.8"),
    # ("2.7.1", "12.8"),
]

PYTHON_VERSION = "3.11"


@dataclass
class Mode:
    compile: bool
    backend: str | None = None
    mode: str | None = None
    options: dict | None = None
    env: dict = field(default_factory=dict)

    def __call__(self, fn):
        if self.compile:
            kwargs = {}
            if self.backend is not None:
                kwargs["backend"] = self.backend
            if self.mode is not None:
                kwargs["mode"] = self.mode
            if self.options is not None:
                options = {
                    k: v
                    for k, v in self.options.items()
                    if k in torch._inductor.list_options()
                }
                kwargs["options"] = options
            return torch.compile(fn, **kwargs)
        assert self.backend is None  # noqa: S101
        assert self.mode is None  # noqa: S101
        assert self.options is None  # noqa: S101
        return fn


MODES = {
    "eager": Mode(compile=False),
    "decomp": Mode(compile=True, backend="aot_eager_decomp_partition"),
    "compile_numerics": Mode(
        compile=True,
        options={
            "emulate_precision_casts": True,
            "use_fast_math": False,
            "emulate_division_rounding": True,
            "eager_numerics.division_rounding": True,
            "eager_numerics.disable_ftz": True,
        },
    ),
    "compile": Mode(compile=True),
}

DTYPES = ["float32", "float16", "bfloat16"]


async def run(command, capture_output=True, capture_stderr=True, stderr=None, **kwargs):
    log.info("Running command: %s", command)
    proc = await asyncio.create_subprocess_exec(
        *shlex.split(command),
        stdout=asyncio.subprocess.PIPE if capture_output else None,
        stderr=asyncio.subprocess.STDOUT if capture_stderr else stderr,
        stdin=asyncio.subprocess.DEVNULL,
        **kwargs,
    )
    if capture_output:
        result = await proc.communicate()
        result = result[0].decode("utf-8")
        log.info("Command result: %s", result)
        log.info("Return code: %s", proc.returncode)
        return result
    else:
        await proc.wait()
        log.info("Return code: %s", proc.returncode)
        return


async def copy_file_to_remote(hostname, source, destination):
    await run(f"scp {source} {hostname}:{destination}")


async def execute_on_remote(hostname, command):
    await run(f"ssh {hostname} {command}", capture_output=False)


async def copy_results_from_remote(hostname, run_id):
    await run(f"mkdir -p results_{run_id}/logs")
    await run(f"scp {hostname}:/workspace/result_*.jsonl results_{run_id}/")
    await run(
        f"scp {hostname}:/workspace/logs.tar.gz results_{run_id}/logs_{hostname}.tar.gz"
    )
    await run(f"mkdir -p results_{run_id}/logs")
    await run(
        f"tar xzf results_{run_id}/logs_{hostname}.tar.gz -C results_{run_id}/logs --strip 2"
    )
    await run(
        "bash -c 'for f in *.trace; do cd $f; uvx tlparse --no-browser *.log & cd ..; done; wait'",
        cwd=f"results_{run_id}/logs",
    )
    await run(f"rm results_{run_id}/logs_{hostname}.tar.gz")
    await run(f"bash -c 'rm results_{run_id}/logs/*.trace/*.log'")
    await run(f"bash -c 'rm results_{run_id}/logs/*.trace/tl_out/raw*'")


async def do_numerics_test(args, hostname, gpu, run_id):
    await copy_file_to_remote(hostname, __file__, "/workspace/run.py")
    await execute_on_remote(hostname, f"python3 /workspace/run.py --worker --gpu {gpu}")
    await copy_results_from_remote(hostname, run_id)


async def reserve_gpu(gpu):
    stdout = await run(
        f"gpu-dev reserve -g 1 -h 24 -t {gpu} --ignore-no-persist --disk none"
    )
    # the output has a string like this: SSH Command: ssh gpu-dev-1db65ec7
    match = re.search(r"SSH Command: ssh (.*)", stdout)
    hostname = match.group(1)
    log.info(
        "Hostname: %s, Reservation ID: %s",
        hostname,
        hostname.removeprefix("gpu-dev-"),
    )
    return hostname, hostname.removeprefix("gpu-dev-")


async def cancel_reservation(reservation_id):
    await run(f"gpu-dev cancel {reservation_id}")


async def run_on_gpu(gpu, command):
    log.info("Reserving GPU %s", gpu)
    hostname, reservation_id = await reserve_gpu(gpu)
    try:
        log.info("Running command on %s", hostname)
        await command(hostname=hostname)
    except Exception:
        log.exception("Error running command on %s", gpu)
    finally:
        log.info("Cancelling reservation %s", reservation_id)
        await cancel_reservation(reservation_id)


async def launcher(args):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    await asyncio.gather(
        *[
            run_on_gpu(
                gpu,
                functools.partial(do_numerics_test, args=args, gpu=gpu, run_id=run_id),
            )
            for gpu in args.gpu
        ]
    )
    await run(f"{sys.executable} {__file__} --html {run_id}")


def format_results_to_html(run_id):
    results_dir = Path(f"results_{run_id}")
    for f in results_dir.glob("*.jsonl"):
        html_file = f.with_suffix(".html")
        lines = f.read_text().splitlines(keepends=True)
        data = [json.loads(line) for line in lines]
        KEYS = data[0].keys()
        print(KEYS)
        FILTER_KEYS = [
            "gpu",
            "pytorch_version",
            "cuda_version",
            "pytorch_mode",
            "is_golden",
            "data_type",
            "function",
            "pass_type",
            "category",
            "match_full",
            "match_normal",
        ]
        KEY_VALUES = {k: sorted({line[k] for line in data}) for k in FILTER_KEYS}
        with html_file.open("w") as html_f:
            html_f.write("<html><body style='font-family: monospace;'>\n")
            html_f.write(f"<h1>Results for {f.stem}</h1>\n")
            for key, values in KEY_VALUES.items():
                html_f.write(
                    f"<details id='section-{key}'><summary>{key} ({len(values)})</summary>\n"
                )
                html_f.write(
                    f"<a href='#' onclick='document.querySelectorAll(\"#section-{key} input\").forEach(checkbox => checkbox.checked = true);'>Enable all</a>\n"  # noqa: B950
                )
                html_f.write(
                    f"<a href='#' onclick='document.querySelectorAll(\"#section-{key} input\").forEach(checkbox => checkbox.checked = false);'>Disable all</a>\n"  # noqa: B950
                )
                for value in values:
                    safe_value = str(value).replace(".", "_")
                    html_f.write(
                        f"<label>\n"
                        f"<input type='checkbox' id='filter-{key}-{safe_value}' checked>"
                        f"{value}</label>\n"
                    )
                    html_f.write(
                        f"<style>body:has(#filter-{key}-{safe_value}:not(:checked)) .visible-{key}-{safe_value} {{ display: none; }}</style>\n"  # noqa: B950
                    )
                html_f.write("</details>")
            # one more filter: show "first" mode only
            html_f.write("<details><summary>First mode that fails only</summary>\n")
            html_f.write(
                "<label><input type='checkbox' id='filter-mode-first' checked>"  # noqa: B950
                "Exclude items where a previous mode also fails</label>\n"
            )
            html_f.write("</details>\n")
            # column filter:
            html_f.write(
                "<details id='section-columns'><summary>Column filter</summary>\n"
            )
            html_f.write(
                "<a href='#' onclick='document.querySelectorAll(\"#section-columns input\").forEach(checkbox => checkbox.checked = true);'>Enable all</a>\n"  # noqa: B950
            )
            html_f.write(
                "<a href='#' onclick='document.querySelectorAll(\"#section-columns input\").forEach(checkbox => checkbox.checked = false);'>Disable all</a>\n"  # noqa: B950
            )
            DEFAULT_HIDDEN_KEYS = [
                "gpu",
                "pytorch_version",
                "cuda_version",
                "is_golden",
            ]
            for key in KEYS:
                html_f.write(
                    f"<label><input type='checkbox' id='filter-column-{key}' {'' if key in DEFAULT_HIDDEN_KEYS else 'checked'}>{key}</label>\n"  # noqa: B950
                )
                html_f.write(
                    f"<style>body:has(#filter-column-{key}:not(:checked)) .visible-column-{key} {{ display: none; }}</style>\n"
                )
            html_f.write("</details>\n")
            html_f.write("<table>\n")
            html_f.write("<tr>\n")
            for key in KEYS:
                html_f.write(f"<th class='visible-column-{key}'>{key}</th>\n")

            html_f.write("<th>log</th>\n")
            html_f.write("<th>tlparse</th>\n")
            html_f.write("</tr>\n")
            for line in lines:
                data = json.loads(line)
                classes = [
                    f"visible-{k}-{str(data[k]).replace('.', '_')}" for k in KEY_VALUES
                ]
                html_f.write(f"<tr class='{' '.join(classes)}'>\n")
                for key in KEYS:
                    html_f.write(f"<td class='visible-column-{key}'>\n")
                    if key == "mismatch_sample":
                        html_f.write(
                            f"<input type='checkbox' id='filter-mismatch-sample-{data['identifier']}'>\n"
                        )
                        html_f.write(
                            f"<style>body:has(#filter-mismatch-sample-{data['identifier']}:not(:checked)) .visible-mismatch-sample-{data['identifier']} {{ display: none; }}</style>\n"  # noqa: B950
                        )
                    else:
                        html_f.write(f"{data[key]}\n")
                    html_f.write("</td>\n")
                html_f.write(
                    f"<td><a href='logs/{data['identifier']}.log'>log</a></td>"
                )
                html_f.write(
                    f"<td><a href='logs/{data['identifier']}.trace/tl_out/index.html'>trace</a></td>\n"
                )
                html_f.write("</tr>\n")
                if data["mismatch_sample"]:
                    html_f.write(
                        f"<tr class='visible-mismatch-sample-{data['identifier']} {' '.join(classes)}'>\n"
                    )
                    html_f.write(f"<td colspan='{len(KEYS) + 2}'><pre>\n")
                    html_f.write("<table>\n")
                    html_f.write("<tr>\n")
                    html_f.write("<th>pos</th>\n")
                    html_f.write("<th>input</th>\n")
                    html_f.write("<th>output</th>\n")
                    html_f.write("<th>golden</th>\n")
                    html_f.write("<th>rel_err</th>\n")
                    html_f.write("</tr>\n")
                    for sample in data["mismatch_sample"]:
                        html_f.write("<tr>\n")
                        html_f.write(f"<td>{sample['pos']}</td>\n")
                        html_f.write(
                            f"<td>{', '.join(map(str, sample['input']))}</td>\n"
                        )
                        html_f.write(f"<td>{sample['output']}</td>\n")
                        html_f.write(f"<td>{sample['golden']}</td>\n")
                        html_f.write(f"<td>{sample['rel_err']}</td>\n")
                        html_f.write("</tr>\n")
                    html_f.write("</table>\n")
                    html_f.write("</td>\n")
                    html_f.write("</tr>\n")

            html_f.write("</table>\n")
            html_f.write("<style>\n")
            # implement the various filters
            html_f.write("</style>\n")
            html_f.write("</body></html>\n")


global_utils_lock = asyncio.Lock()
cuda_download_lock = asyncio.Lock()
venv_creation_lock = asyncio.Lock()
pytorch_build_lock = asyncio.Lock()
numerics_test_lock = asyncio.Lock()


async def create_results(args, config):
    pytorch_version, cuda_version = config
    async with global_utils_lock:
        await ensure_global_utils()
    async with cuda_download_lock:
        await ensure_cuda(args, cuda_version)
    async with venv_creation_lock:
        venv = await create_venv(args, config)
    async with pytorch_build_lock:
        await maybe_build_pytorch(args, config, venv)
    async with numerics_test_lock:
        await run_numerics_test(args, config, venv)


async def ensure_global_utils():
    if os.environ.get("HAS_GLOBAL_UTILS", "0") == "1":
        return
    await run("mkdir global_bin")
    os.environ["PATH"] = f"/workspace/global_bin:{os.environ.get('PATH', '')}"

    os.environ["UV_INSTALL_DIR"] = "/workspace/global_bin"
    os.environ["UV_CACHE_DIR"] = "/workspace/.cache/uv"
    await run("wget --no-verbose https://astral.sh/uv/install.sh")
    await run("bash install.sh")
    await run("rm install.sh")

    await run(
        "wget --no-verbose https://github.com/seeraven/gitcache/releases/download/v1.0.29/gitcache_v1.0.29_Ubuntu22.04_x86_64"  # noqa: B950
    )
    await run("mv gitcache_v1.0.29_Ubuntu22.04_x86_64 global_bin/gitcache")
    await run("chmod +x global_bin/gitcache")
    which_gitcache = await run("which gitcache")
    await run(f"ln -s {which_gitcache.strip()} global_bin/git")
    os.environ["GITCACHE_DIR"] = "/workspace/.cache/gitcache"

    await run("sudo apt-get update")
    await run("sudo apt-get install -y ccache git-lfs")
    os.environ["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CMAKE_CUDA_COMPILER_LAUNCHER"] = "ccache"
    os.environ["CCACHE_DIR"] = "/workspace/.cache/ccache"
    os.environ["CCACHE_NOHASHDIR"] = "1"
    os.environ["CCACHE_BASEDIR"] = "/workspace"
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "20"
    os.environ["MAX_JOBS"] = "8"
    os.environ["MAKEFLAGS"] = "-j20"
    os.environ["NINJAFLAGS"] = "-j20"

    os.environ["HAS_GLOBAL_UTILS"] = "1"


async def ensure_cuda(args, cuda_version):
    if os.path.exists(f"/workspace/cuda-{cuda_version}"):
        return
    await run("sudo apt-get update")
    await run(
        f"sudo apt-get install -y cuda-toolkit-{cuda_version.split('.')[0]}-{cuda_version.split('.')[1]}"
    )


async def create_venv(args, config):
    pytorch_version, cuda_version = config
    env = os.environ.copy()
    # cuda_home = f"/usr/local/cuda-{cuda_version}"
    cuda_home = (
        f"/usr/local/cuda-{cuda_version.split('.')[0]}.{cuda_version.split('.')[1]}"
    )
    env["CUDA_HOME"] = cuda_home
    env["CUDA_ROOT"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_NVCC_EXECUTABLE"] = f"{cuda_home}/bin/nvcc"
    env["PYTORCH_NVCC"] = f"{cuda_home}/bin/nvcc"
    venv_dir = f"/workspace/venv-pytorch_{pytorch_version}-cuda_{cuda_version}"
    await run(f"mkdir -p {venv_dir}")
    await run(f"uv venv -p {PYTHON_VERSION} --managed-python", cwd=venv_dir, env=env)
    # update env with venv paths
    env["VIRTUAL_ENV"] = f"{venv_dir}/.venv"
    env["PATH"] = f"{venv_dir}/.venv/bin:{env.get('PATH', '')}"
    await run("uv pip install pip numpy", env=env, cwd=venv_dir)

    index_url = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
    if pytorch_version == "nightly":
        await run(
            f"uv pip install --pre torch --index-url {index_url}", env=env, cwd=venv_dir
        )
    elif not pytorch_version.startswith("git:"):
        await run(
            f"uv pip install torch=={pytorch_version} --index-url {index_url}",
            env=env,
            cwd=venv_dir,
        )
    return {"env": env, "cwd": venv_dir}


async def maybe_build_pytorch(args, config, venv):
    pytorch_version, cuda_version = config
    if pytorch_version.startswith("git:"):
        raise NotImplementedError("Building PyTorch from git is not supported yet")
    return


async def run_numerics_test(args, config, venv):
    pytorch_version, cuda_version = config
    golden_flag = ""
    if args.create_golden:
        golden_flag = " --create-golden"
    for mode in MODES:
        cmd = (
            f"python {__file__} --runner --gpu {args.gpu[0]}"
            f" --pytorch-version {pytorch_version}"
            f" --cuda-version {cuda_version}"
            f" --mode {mode}"
            f" --golden /workspace/golden{golden_flag}"
        )
        await run(cmd, capture_output=False, **venv)
        golden_flag = ""


def sortable_config_key(config):
    pytorch_version, cuda_version = config
    cuda_version = tuple(map(int, cuda_version.split(".")))

    if pytorch_version == "nightly":
        pytorch_version = (-1,)
    elif pytorch_version.startswith("git:"):
        pytorch_version = (-2,)
    else:
        pytorch_version = tuple(map(int, pytorch_version.split(".")))

    return (pytorch_version, cuda_version)


async def worker(args):
    assert len(args.gpu) == 1  # noqa: S101
    # find golden task
    # a config is a tuple (pytorch_version, cuda_version)
    # where pytorch_version can either be a version string or nightly or a git hash
    golden_config = sorted(ENABLED_CONFIGS, key=sortable_config_key)[-1]
    await run("mkdir -p /workspace/logs")
    args.create_golden = True
    await create_results(args, golden_config)
    args.create_golden = False
    await asyncio.gather(
        *[
            create_results(args, config)
            for config in ENABLED_CONFIGS
            if config != golden_config
        ]
    )
    await run("tar czf /workspace/logs.tar.gz /workspace/logs")


CATEGORIES = {
    "reduction": ["torch.sum", "torch.mean", "torch.softmax"],
    "matrix": ["torch.matmul"],
    "normalization": [
        "torch.nn.functional.layer_norm",
        "torch.nn.functional.rms_norm",
    ],
    "activation": [
        "torch.nn.functional.relu",
        "torch.nn.functional.sigmoid",
        "torch.nn.functional.tanh",
        "torch.nn.functional.gelu",
        "torch.nn.functional.silu",
    ],
    "elementary": [
        "torch.sin",
        "torch.cos",
        "torch.tan",
        "torch.sigmoid",
        "torch.exp",
        "torch.exp2",
        "torch.log",
        "torch.log2",
        "torch.sqrt",
        "torch.erf",
        "torch.reciprocal",
        "torch.rsqrt",
    ],
    "binary": [
        "torch.add",
        "torch.sub",
        "torch.mul",
        "torch.div",
        "torch.pow",
    ],
}

PASS_TYPES = {
    "reduction": ["fwd", "bwd_0"],
    "matrix": ["fwd", "bwd_0", "bwd_1"],
    "normalization": ["fwd", "bwd_0", "bwd_1"],
    "activation": ["fwd", "bwd_0"],
    "elementary": ["fwd", "bwd_0"],
    "binary": ["fwd", "bwd_0", "bwd_1"],
}

PASSES = {}


def register_pass(pass_type):
    def decorator(func):
        PASSES[pass_type] = func
        return func

    return decorator


@register_pass("fwd")
def pass_fwd(callable):
    def wrapper(*args):
        return callable(*args)

    return wrapper


@register_pass("bwd_0")
def pass_bwd_0(callable):
    def wrapper(*args):
        args = [arg.detach().requires_grad_(idx == 0) for idx, arg in enumerate(args)]
        output = callable(*args)
        output.sum().backward()
        return args[0].grad.detach()

    return wrapper


@register_pass("bwd_1")
def pass_bwd_1(callable):
    def wrapper(*args):
        args = [arg.detach().requires_grad_(idx == 1) for idx, arg in enumerate(args)]
        output = callable(*args)
        output.sum().backward()
        return args[1].grad.detach()

    return wrapper


def filter_nan(a):
    return torch.where(torch.isnan(a), 0, a)


def mantissa_bits(dtype) -> int:
    return {torch.float32: 23, torch.bfloat16: 5, torch.float16: 10}[dtype]


def float_to_int_type(dtype):
    return {
        torch.float32: torch.uint32,
        torch.bfloat16: torch.uint16,
        torch.float16: torch.uint16,
    }[dtype]


def generate_test_tensor(dtype, slow=False):
    if dtype in [torch.float16, torch.bfloat16]:
        return filter_nan(
            torch.arange(0, 2**16, dtype=torch.int32, device="cuda")
            .to(torch.uint16)
            .view(dtype)
        )
    if dtype == torch.float32:
        if slow:
            return filter_nan(
                torch.arange(0, 2**32, dtype=torch.int64, device="cuda")
                .to(torch.uint32)
                .view(dtype)
            )
        else:
            # E8M5, E5M10, E8M23
            result = []
            for t in [
                generate_test_tensor(torch.bfloat16),
                generate_test_tensor(torch.float16),
            ]:
                orig_dtype = t.dtype
                t = t.to(dtype).view(torch.uint32)
                missing_mantissa_bits = mantissa_bits(dtype) - mantissa_bits(orig_dtype)
                r = torch.randint(
                    0, 2**missing_mantissa_bits, t.shape, dtype=t.dtype, device="cuda"
                )
                result.append(t.view(dtype))
                result.append((t.view(torch.int32) | r.view(torch.int32)).view(dtype))
            return filter_nan(torch.cat(result))


def make_input(dtype, category):
    dtype = getattr(torch, dtype)
    from torch.testing import make_tensor

    if category in ["matrix"]:
        return make_tensor(1024, 1024, dtype=dtype, device="cuda"), make_tensor(
            1024, 1024, dtype=dtype, device="cuda"
        )
    if category == "normalization":
        return make_tensor(1024, 1024, dtype=dtype, device="cuda"), make_tensor(
            1024, dtype=dtype, device="cuda"
        )
    if category == "reduction":
        return (make_tensor(1024, 1024, dtype=dtype, device="cuda"),)
    if category == "binary":
        test_tensor = generate_test_tensor(dtype)
        # randomly permute twice:
        return (
            test_tensor[torch.randperm(test_tensor.shape[0])],
            test_tensor[torch.randperm(test_tensor.shape[0])],
        )
    return (generate_test_tensor(dtype),)


def make_function(function, category):
    if category == "reduction":
        return eval(f"lambda x: {function}(x, dim=-1)")
    elif category == "matrix":
        return eval(f"lambda x, y: {function}(x, y)")
    elif category == "normalization":
        return eval(f"lambda x, y: {function}(x, [x.shape[-1]], y, eps=1e-5)")
    elif category == "binary":
        return eval(f"lambda x, y: {function}(x, y)")
    elif category == "activation":
        return eval(f"lambda x: {function}(x)")
    elif category == "elementary":
        return eval(f"lambda x: {function}(x)")
    else:
        raise ValueError(f"Unknown category: {category}")


def rel_err_ulp(a, b, dtype):
    return (a - b).abs() / (b.abs() * torch.finfo(dtype).eps + torch.finfo(dtype).tiny)


def evaluate_output(input, output, golden_output):
    # todo: how to make sure this handles denormals in the input well?
    # for now, only handle it if the inputs have the same shape, otherwise assume there are none
    # handle more than one input

    input = [i.flatten().float() for i in input if i.shape == output.shape]

    dtype = golden_output.dtype

    output = output.flatten().float()
    golden_output = golden_output.flatten().float()

    # we are checking subnormals separate from the rest of the numbers
    # we also need to check NaNs and Infs carefully

    subnormal_mask = golden_output.abs() < torch.finfo(dtype).smallest_normal
    for i in input:
        subnormal_mask |= i.abs() < torch.finfo(dtype).smallest_normal
    nan_mask_golden = torch.isnan(golden_output)
    golden_output = torch.where(nan_mask_golden, float("nan"), golden_output)
    nan_mask_output = torch.isnan(output)
    output = torch.where(nan_mask_output, float("nan"), output)

    equal_mask = (output == golden_output) | (nan_mask_golden & nan_mask_output)
    equal_subnormal_mask = (
        torch.where(subnormal_mask, 0, output)
        == torch.where(subnormal_mask, 0, golden_output)
    ) | equal_mask
    output_flushed = torch.where(subnormal_mask, 0.0, output)
    golden_flushed = torch.where(subnormal_mask, 0.0, golden_output)
    output_flushed = torch.where(~torch.isfinite(output_flushed), 0.0, output_flushed)
    golden_flushed = torch.where(~torch.isfinite(golden_flushed), 0.0, golden_flushed)

    num_nonequal = (~equal_mask).sum().item()
    num_nonequal_subnormal = (~equal_subnormal_mask).sum().item()

    err = rel_err_ulp(output_flushed, golden_flushed, dtype)
    max_ulp_to_golden = err.max().item()
    avg_ulp_to_golden = err.mean().item()
    mismatch_sample = []
    pos = (~equal_mask).nonzero().squeeze(1)
    log.info(
        "pos.shape: %s, input: %s, golden_output.shape: %s, output.shape: %s",
        pos.shape,
        [i.shape for i in input],
        golden_output.shape,
        output.shape,
    )
    ordered = torch.argsort(err[pos], descending=True)
    random = torch.randperm(pos.shape[0])
    for sampling in [ordered, random]:
        for i in range(min(pos.shape[0], 5)):
            sample_idx = pos[sampling[i]]
            mismatch_sample.append(
                {
                    "pos": sample_idx.item(),
                    "input": [inp[sample_idx].item() for inp in input],
                    "output": output[sample_idx].item(),
                    "golden": golden_output[sample_idx].item(),
                    "rel_err": err[sample_idx].item(),
                }
            )

    # use hashlib.sha256 to hash the tensors
    # this i
    return {
        "normal_hash": hashlib.sha256(
            output_flushed.cpu().numpy().tobytes()
        ).hexdigest()[:8],
        "full_hash": hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest()[:8],
        "max_ulp_to_golden": max_ulp_to_golden,
        "avg_ulp_to_golden": avg_ulp_to_golden,
        "num_nonequal": num_nonequal,
        "num_nonequal_subnormal": num_nonequal_subnormal,
        "num_total": output.shape[0],
        "match_full": num_nonequal == 0,
        "match_normal": num_nonequal_subnormal == 0,
        "mismatch_sample": mismatch_sample,
    }


def create_golden(args):
    golden_inputs = {
        dtype: {category: make_input(dtype, category) for category in CATEGORIES}
        for dtype in DTYPES
    }
    golden_outputs = {
        dtype: {
            category: {
                function: {
                    pass_type: MODES[args.mode](
                        PASSES[pass_type](make_function(function, category))
                    )(*golden_inputs[dtype][category])
                    for pass_type in PASS_TYPES[category]
                }
                for function in CATEGORIES[category]
            }
            for category in CATEGORIES
        }
        for dtype in DTYPES
    }
    return golden_inputs, golden_outputs


def get_metadata(args, dtype, category, function, pass_type):
    metadata = {
        "gpu": args.gpu[0],
        "pytorch_version": args.pytorch_version,
        "cuda_version": args.cuda_version,
        "pytorch_mode": args.mode,
        "is_golden": args.create_golden,
        "data_type": dtype,
        "function": function,
        "pass_type": pass_type,
        "category": category,
    }
    identifier = hashlib.sha256(json.dumps(metadata).encode()).hexdigest()[:8]
    metadata["identifier"] = identifier
    return metadata


def run_test_case(args):
    golden = torch.load(args.golden)
    golden_inputs, golden_outputs = golden
    metadata = get_metadata(
        args, args.dtype, args.category, args.function, args.pass_type
    )
    input = golden_inputs[args.dtype][args.category]
    callable = make_function(args.function, args.category)
    callable = PASSES[args.pass_type](callable)
    callable = MODES[args.mode](callable)
    output = callable(*input)
    evaluation = evaluate_output(
        input,
        output,
        golden_outputs[args.dtype][args.category][args.function][args.pass_type],
    )
    data = json.dumps(metadata | evaluation)
    log.info(data)
    print(data)


test_concurrency = asyncio.Semaphore(16)


async def launch_test_case(args, dtype, category, function, pass_type):
    async with test_concurrency:
        env = os.environ | MODES[args.mode].env
        metadata = get_metadata(args, dtype, category, function, pass_type)
        identifier = metadata["identifier"]
        env["TORCH_TRACE"] = f"/workspace/logs/{identifier}.trace"
        Path(env["TORCH_TRACE"]).mkdir(parents=True, exist_ok=True)
        env["TORCH_LOGS"] = "+all"
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        env["TRITON_ALWAYS_COMPILE"] = "1"
        cmd_args = [
            "--test",
            f"--gpu {args.gpu[0]}",
            f"--pytorch-version {args.pytorch_version}",
            f"--cuda-version {args.cuda_version}",
            f"--mode {args.mode}",
            f"--golden {args.golden}",
            f"--dtype {dtype}",
            f"--category {category}",
            f"--function {function}",
            f"--pass-type {pass_type}",
        ]
        if args.create_golden:
            cmd_args.append("--create-golden")
        return await run(
            f"{sys.executable} {__file__} {' '.join(cmd_args)}",
            env=env,
            capture_stderr=False,
            stderr=open(f"/workspace/logs/{identifier}.log", "w"),
        )


async def runner(args):
    log.info("Runner called! %s", args)

    if args.create_golden:
        golden = create_golden(args)
        torch.save(golden, args.golden)

    results = await asyncio.gather(
        *[
            launch_test_case(
                args,
                dtype,
                category,
                function,
                pass_type,
            )
            for dtype, category in itertools.product(DTYPES, CATEGORIES)
            for function, pass_type in itertools.product(
                CATEGORIES[category], PASS_TYPES[category]
            )
        ]
    )

    with open(
        f"/workspace/result_{args.gpu[0]}_{args.pytorch_version}_{args.cuda_version}.jsonl",
        "a+",
    ) as result:
        for r in results:
            result.write(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launcher",
        default=False,
        action="store_true",
        help="launch workers across machines",
    )
    parser.add_argument(
        "--worker",
        default=False,
        action="store_true",
        help="launches jobs on a single machine",
    )
    parser.add_argument(
        "--runner",
        default=False,
        action="store_true",
        help="runs tests in a specific environment",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="runs a single test case",
    )
    parser.add_argument("--gpu", nargs="+", type=str, default=["t4"])
    parser.add_argument("--html", type=str, default=None)
    parser.add_argument("--golden", type=str, default=None)
    parser.add_argument("--create-golden", action="store_true")
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--function", type=str)
    parser.add_argument("--pass-type", type=str)
    parser.add_argument("--pytorch-version", type=str)
    parser.add_argument("--cuda-version", type=str)
    parser.add_argument("--mode")
    args = parser.parse_args()
    if args.runner:
        assert torch is not None  # noqa: S101
        torch.set_default_device("cuda")
        (gpu,) = args.gpu
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s - runner:{gpu}/{args.pytorch_version}/{args.cuda_version}/{args.mode} - %(message)s",  # noqa: B950
        )
        asyncio.run(runner(args))
    if args.worker:
        (gpu,) = args.gpu
        logging.basicConfig(
            level=logging.INFO, format=f"%(asctime)s - worker:{gpu} - %(message)s"
        )
        asyncio.run(worker(args))
    if args.launcher:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - launcher - %(message)s"
        )
        try:
            asyncio.run(launcher(args))
        except KeyboardInterrupt:
            log.error("Cancelled by user")
    if args.test:
        test_id = (
            f"{args.gpu[0]}/{args.pytorch_version}/{args.cuda_version}"
            f"/{args.mode}/{args.dtype}/{args.category}"
            f"/{args.function}/{args.pass_type}"
        )
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s - test:{test_id} - %(message)s",
        )
        run_test_case(args)
    if args.html:
        format_results_to_html(args.html)


if __name__ == "__main__":
    main()
