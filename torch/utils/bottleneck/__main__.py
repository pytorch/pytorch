import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict

import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info


def redirect_argv(new_argv):
    sys.argv[:] = new_argv[:]


def compiled_with_cuda(sysinfo):
    if sysinfo.cuda_compiled_version:
        return f'compiled w/ CUDA {sysinfo.cuda_compiled_version}'
    return 'not compiled w/ CUDA'


env_summary = """
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch {pytorch_version}{debug_str} {cuda_compiled}
Running with Python {py_version} and {cuda_runtime}

`{pip_version} list` truncated output:
{pip_list_output}
""".strip()


def run_env_analysis():
    print('Running environment analysis...')
    info = get_env_info()

    result: Dict[str, str] = {}

    debug_str = ''
    if info.is_debug_build:
        debug_str = ' DEBUG'

    cuda_avail = ''
    if info.is_cuda_available:
        cuda = info.cuda_runtime_version
        if cuda is not None:
            cuda_avail = 'CUDA ' + cuda
    else:
        cuda = 'CUDA unavailable'

    pip_version = info.pip_version
    pip_list_output = info.pip_packages
    if pip_list_output is None:
        pip_list_output = 'Unable to fetch'

    result = {
        'debug_str': debug_str,
        'pytorch_version': info.torch_version,
        'cuda_compiled': compiled_with_cuda(info),
        'py_version': f'{sys.version_info[0]}.{sys.version_info[1]}',
        'cuda_runtime': cuda_avail,
        'pip_version': pip_version,
        'pip_list_output': pip_list_output,
    }

    return env_summary.format(**result)


def run_cprofile(code, globs, launch_blocking=False):
    print('Running your script with cProfile')
    prof = cProfile.Profile()
    prof.enable()
    exec(code, globs, None)
    prof.disable()
    return prof


cprof_summary = """
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
""".strip()


def print_cprofile_summary(prof, sortby='tottime', topk=15):
    print(cprof_summary)
    cprofile_stats = pstats.Stats(prof).sort_stats(sortby)
    cprofile_stats.print_stats(topk)


def run_autograd_prof(code, globs):
    def run_prof(use_cuda=False):
        with profiler.profile(use_cuda=use_cuda) as prof:
            exec(code, globs, None)
        return prof

    print('Running your script with the autograd profiler...')
    result = [run_prof(use_cuda=False)]
    if torch.cuda.is_available():
        result.append(run_prof(use_cuda=True))
    else:
        result.append(None)

    return result


autograd_prof_summary = """
--------------------------------------------------------------------------------
  autograd profiler output ({mode} mode)
--------------------------------------------------------------------------------
        {description}
{cuda_warning}
{output}
""".strip()


def print_autograd_prof_summary(prof, mode, sortby='cpu_time', topk=15):
    valid_sortby = ['cpu_time', 'cuda_time', 'cpu_time_total', 'cuda_time_total', 'count']
    if sortby not in valid_sortby:
        warn = ('WARNING: invalid sorting option for autograd profiler results: {}\n'
                'Expected `cpu_time`, `cpu_time_total`, or `count`. '
                'Defaulting to `cpu_time`.')
        print(warn.format(sortby))
        sortby = 'cpu_time'

    if mode == 'CUDA':
        cuda_warning = ('\n\tBecause the autograd profiler uses the CUDA event API,\n'
                        '\tthe CUDA time column reports approximately max(cuda_time, cpu_time).\n'
                        '\tPlease ignore this output if your code does not use CUDA.\n')
    else:
        cuda_warning = ''

    sorted_events = sorted(prof.function_events,
                           key=lambda x: getattr(x, sortby), reverse=True)
    topk_events = sorted_events[:topk]

    result = {
        'mode': mode,
        'description': f'top {topk} events sorted by {sortby}',
        'output': torch.autograd.profiler_util._build_table(topk_events),
        'cuda_warning': cuda_warning
    }

    print(autograd_prof_summary.format(**result))


descript = """
`bottleneck` is a tool that can be used as an initial step for debugging
bottlenecks in your program.

It summarizes runs of your script with the Python profiler and PyTorch\'s
autograd profiler. Because your script will be profiled, please ensure that it
exits in a finite amount of time.

For more complicated uses of the profilers, please see
https://docs.python.org/3/library/profile.html and
https://pytorch.org/docs/main/autograd.html#profiler for more information.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(description=descript)
    parser.add_argument('scriptfile', type=str,
                        help='Path to the script to be run. '
                        'Usually run with `python path/to/script`.')
    parser.add_argument('args', type=str, nargs=argparse.REMAINDER,
                        help='Command-line arguments to be passed to the script.')
    return parser.parse_args()


def cpu_time_total(autograd_prof):
    return sum(event.cpu_time_total for event in autograd_prof.function_events)


def main():
    args = parse_args()

    # Customizable constants.
    scriptfile = args.scriptfile
    scriptargs = [] if args.args is None else args.args
    scriptargs.insert(0, scriptfile)
    cprofile_sortby = 'tottime'
    cprofile_topk = 15
    autograd_prof_sortby = 'cpu_time_total'
    autograd_prof_topk = 15

    redirect_argv(scriptargs)

    sys.path.insert(0, os.path.dirname(scriptfile))
    with open(scriptfile, 'rb') as stream:
        code = compile(stream.read(), scriptfile, 'exec')
    globs = {
        '__file__': scriptfile,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    print(descript)

    env_summary = run_env_analysis()

    if torch.cuda.is_available():
        torch.cuda.init()
    cprofile_prof = run_cprofile(code, globs)
    autograd_prof_cpu, autograd_prof_cuda = run_autograd_prof(code, globs)

    print(env_summary)
    print_cprofile_summary(cprofile_prof, cprofile_sortby, cprofile_topk)

    if not torch.cuda.is_available():
        print_autograd_prof_summary(autograd_prof_cpu, 'CPU', autograd_prof_sortby, autograd_prof_topk)
        return

    # Print both the result of the CPU-mode and CUDA-mode autograd profilers
    # if their execution times are very different.
    cuda_prof_exec_time = cpu_time_total(autograd_prof_cuda)
    if len(autograd_prof_cpu.function_events) > 0:
        cpu_prof_exec_time = cpu_time_total(autograd_prof_cpu)
        pct_diff = (cuda_prof_exec_time - cpu_prof_exec_time) / cuda_prof_exec_time
        if abs(pct_diff) > 0.05:
            print_autograd_prof_summary(autograd_prof_cpu, 'CPU', autograd_prof_sortby, autograd_prof_topk)

    print_autograd_prof_summary(autograd_prof_cuda, 'CUDA', autograd_prof_sortby, autograd_prof_topk)

if __name__ == '__main__':
    main()
