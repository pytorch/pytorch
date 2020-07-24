import torch
from torch.autograd import functional

import time
import argparse
from collections import namedtuple, defaultdict

import ppl_models
import vision_models
import audio_text_models

from utils import get_str

# Listing of the different tasks
FAST_TASKS_NO_DOUBLE_BACK = [
    "vjp",
]

FAST_TASKS = FAST_TASKS_NO_DOUBLE_BACK + [
    "vhp",
    "jvp",
]

ALL_TASKS = FAST_TASKS + [
    "hvp",
    "jacobian",
    "hessian"
]

DOUBLE_BACKWARD_TASKS = ["jvp", "hvp", "vhp", "hessian"]

# Model definition which contains:
# - name: a string with the model name.
# - getter: a function to get the mode. It takes as input the device on which the model
#     will run. It should return the forward function and the parameters (Tensors) used as
#     input for the forward function. Note that the forward must *not* have any side effect.
# - tasks: the list of recommended tasks that can run in a reasonable amount of time with this model.
# - unsupported: the list of tasks that this model cannot run.
ModelDef = namedtuple("ModelDef", ["name", "getter", "tasks", "unsupported"])

MODELS = [
    ModelDef("resnet18", vision_models.get_resnet18, FAST_TASKS, []),
    ModelDef("fcn_resnet", vision_models.get_fcn_resnet, FAST_TASKS, []),
    ModelDef("detr", vision_models.get_detr, FAST_TASKS, []),
    ModelDef("ppl_simple_reg", ppl_models.get_simple_regression, ALL_TASKS, []),
    ModelDef("ppl_robust_reg", ppl_models.get_robust_regression, ALL_TASKS, []),
    ModelDef("wav2letter", audio_text_models.get_wav2letter, FAST_TASKS, []),
    ModelDef("deepspeech", audio_text_models.get_deepspeech, FAST_TASKS_NO_DOUBLE_BACK, DOUBLE_BACKWARD_TASKS),
    ModelDef("transfo", audio_text_models.get_transformer, FAST_TASKS, []),
    ModelDef("multiheadattn", audio_text_models.get_multiheadattn, FAST_TASKS, []),
]

def run_once(model, inp, task, extra=None):
    func = getattr(functional, task)

    # Cache the creation of the `v` argument
    if extra is None:
        if task in ["vjp"]:
            out = model(*inp)
            v = torch.rand_like(out)
        elif task in ["jvp", "hvp", "vhp"]:
            if isinstance(inp, tuple):
                v = tuple(torch.rand_like(i) for i in inp)
            else:
                v = torch.rand_like(inp)
        else:
            v = None
    else:
        v = extra

    if v is not None:
        res = func(model, inp, v=v, strict=True)
    else:
        res = func(model, inp, strict=True)

    if extra is None:
        return v


def run_model(model_getter, args, task):
    if args.gpu == -1:
        device = "cpu"

        def noop(*args):
            return args
        do_sync = noop
    else:
        device = "cuda:{}".format(args.gpu)
        do_sync = torch.cuda.synchronize

    model, inp = model_getter(device)

    # Warmup
    extra = run_once(model, inp, task)

    elapsed = []
    for it in range(args.num_iters):
        do_sync()
        start = time.time()
        run_once(model, inp, task, extra)
        do_sync()
        elapsed.append(time.time() - start)

    return elapsed

def main():
    parser = argparse.ArgumentParser("Main script to benchmark functional API of the autograd.")
    parser.add_argument("--output", type=str, default="", help="Text file where to write the output")
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=-2, help="GPU to use, -1 for CPU and -2 for auto-detect")
    parser.add_argument("--run-slow-tasks", action="store_true", help="Run even the slow tasks")
    parser.add_argument("--model-filter", type=str, default="", help="Only run the models in this filter")
    parser.add_argument("--task-filter", type=str, default="", help="Only run the tasks in this filter")
    parser.add_argument("--num-threads", type=int, default=10,
                        help="Number of concurrent threads to use when running on cpu")
    args = parser.parse_args()

    results = defaultdict(defaultdict)
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    if args.gpu == -2:
        args.gpu = 0 if torch.cuda.is_available() else -1

    for name, model_getter, reco_tasks, unsupported_tasks in MODELS:
        if not args.model_filter or name in args.model_filter:
            tasks = ALL_TASKS if args.run_slow_tasks else reco_tasks
            for task in tasks:
                if task in unsupported_tasks:
                    continue
                if not args.task_filter or task in args.task_filter:
                    runtimes = run_model(model_getter, args, task)

                    runtimes = torch.tensor(runtimes)
                    mean, var = runtimes.mean(), runtimes.var()
                    results[name][task] = (mean.item(), var.item())
                    print("Results for model {} on task {}: {}s (var: {})".format(name, task, mean, var))

    if args.output:
        with open(args.output, "w") as f:
            f.write(get_str(results))

if __name__ == "__main__":
    main()
