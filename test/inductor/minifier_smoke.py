# Owner(s): ["module: inductor"]
os.environ["TORCHDYNAMO_REPRO_AFTER"] = "dynamo"
import torch
import torch._dynamo as torchdynamo
import torch._inductor.lowering
import torch._ops


def func(x):
    x = torch.sigmoid(x)
    x = torch.mul(x, torch.ones(2))
    x = torch.add(x, torch.zeros(2))
    x = torch.ops.aten.round(x)
    return x


error_injection_str = """
import torch._inductor.lowering

def inject_error():
    def throw(x):
        assert False
    # inject an error in the lowerings
    for x in list(torch._inductor.lowering.lowerings.keys()):
        if 'round' in x.__name__:
            torch._inductor.lowering.lowerings[x] = throw

inject_error()
"""

exec(error_injection_str)


def patch_launcher():
    minifier_launcher_path = torchdynamo.debug_utils.get_minifier_repro_path()
    with open(minifier_launcher_path, "r") as f:
        code = f.read()
        code = code.replace(
            torchdynamo.debug_utils.TEST_REPLACEABLE_COMMENT, error_injection_str
        )

    with open(minifier_launcher_path, "w") as f:
        f.write(code)

    return code


def run_internal_minifier():
    torchdynamo.config.debug_dir_root = "."
    try:
        f_opt = torch.compile(func)
        f_opt(torch.ones(2))
    except Exception as e:
        patch_launcher()
        raise e


run_internal_minifier()
