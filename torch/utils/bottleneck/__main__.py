import argparse
import cProfile
import pstats
import subprocess
import sys
import os
import re

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import torch
from torch.autograd import profiler

PY3 = sys.version_info >= (3, 0)


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    if PY3:
        output = output.decode("ascii")
        err = err.decode("ascii")
    return (rc, output, err)


def check_running_cuda_version():
    (rc, out, err) = run('nvcc --version')
    if rc is not 0:
        return None
    m = re.search(r'V(.*)$', out)
    assert m is not None
    return m.group(1)


def check_pip_packages():
    # People generally have `pip` as `pip` or `pip3`
    def run_with_pip(pip):
        (rc, out, err) = run(pip + ' list --format=legacy | grep torch')
        if rc is 0:
            return '`{}` list truncated output:\n{}'.format(pip, out)
        return None

    result = []
    out = run_with_pip('pip')
    if out is not None:
        result.append(out)
    out_pip3 = run_with_pip('pip3')
    if out_pip3 is not None:
        result.append(out_pip3)

    return '\n'.join(result)


def compiled_with_cuda():
    if torch.version.cuda:
        return 'compiled w/ CUDA {}'.format(torch.version.cuda)
    return 'not compiled w/ CUDA'


def run_env_analysis():
    print('Running environment analysis...')
    result = []

    debug_str = ''
    if torch.version.debug:
        debug_str = ' DEBUG'
    result.append('PyTorch {}{} {}'.format(
        torch.__version__, debug_str,
        compiled_with_cuda()))

    avail = 'Running with python {}.{}, '.format(sys.version_info[0], sys.version_info[1])
    if torch.cuda.is_available():
        cuda = check_running_cuda_version()
        if cuda is None:
            cuda = ''
        avail += 'CUDA {}'.format(cuda)
    else:
        avail += 'CUDA unavailable'
    result.append(avail)

    result.append('')

    pip = check_pip_packages()
    if pip is not None:
        result.append(check_pip_packages())

    return '\n'.join(result)


def set_env_variables(gpu_device):
    # Override CUDA_LAUNCH_BLOCKING by default for more accurate profiling.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Only profile on one GPU for simplicity
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)


def get_env_variables_description(gpu_device, prefix=' '):
    if torch.cuda.is_available():
        return '{}with environment variables CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES={}'.format(
            prefix, gpu_device)
    else:
        return ''


def run_cprofile(code, globs, gpu_device=0):
    set_env_variables(gpu_device)
    env_str = get_env_variables_description(gpu_device)
    print('Running your script with cProfile{}...'.format(env_str))
    prof = cProfile.Profile()
    prof.enable()
    exec(code, globs, None)
    prof.disable()
    return prof


def print_line(width=80):
    print('-' * width)


def print_cprofile_summary(prof, gpu_device=0, sortby='tottime', topk=15):
    print_line()
    env_str = get_env_variables_description(gpu_device, '\n')
    print('cProfile output{}'.format(env_str))
    print_line()
    cprofile_stats = pstats.Stats(prof).sort_stats(sortby)
    cprofile_stats.print_stats(topk)


def run_autograd_prof(code, globs, gpu_device=0):
    set_env_variables(gpu_device)
    env_str = get_env_variables_description(gpu_device)
    print('Running your script with the autograd profiler{}...'.format(env_str))
    with profiler.profile() as prof:
        exec(code, globs, None)
    return prof


def print_autograd_prof_summary(prof, gpu_device=0, sortby='cpu_time', topk=15):
    valid_sortby = ['cpu_time', 'cuda_time', 'cpu_time_total', 'cuda_time_total', 'count']
    if sortby not in valid_sortby:
        warn = ('WARNING: invalid sorting option for autograd profiler results: {}\n'
                'Expected `cpu_time`, `cpu_time_total`, or `count`. '
                'Defaulting to `cpu_time`.')
        print(warn.format(autograd_prof_sortby))
        sortby = 'cpu_time'

    print_line()
    env_str = get_env_variables_description(gpu_device, '\n')
    print('autograd profiler output{}'.format(env_str))
    print_line()
    print('\ttop {} events sorted by {}'.format(topk, sortby))
    ex = ('    Note that because CUDA_LAUNCH_BLOCKING=1 is set, the reported CPU time\n'
          '    includes the CUDA time. Ignore the empty CUDA time column here.\n')
    if torch.cuda.is_available():
        print('')
        print(ex)

    sorted_events = sorted(prof.function_events,
                           key=lambda x: getattr(x, sortby), reverse=True)
    topk_events = sorted_events[:topk]
    print(torch.autograd.profiler.build_table(topk_events))


descript = ('`bottleneck` is a tool that can be used as an initial step for debugging '
            'bottlenecks in your program.\n\n'
            'It summarizes runs of your script with the Python profiler '
            'and PyTorch\'s autograd profiler. For ease of use and intepretability of '
            'results, `bottleneck` runs multi-GPU code on only one GPU device. '
            'Because your script will be profiled, please ensure that it exits '
            'in a finite amount of time. \n\n'
            'For more complicated uses of the profilers (like in a multi-GPU case), '
            'please see https://docs.python.org/3/library/profile.html '
            'and http://pytorch.org/docs/master/autograd.html#profiler '
            'for more information. \n')


def parse_args():
    parser = argparse.ArgumentParser(description=descript)
    parser.add_argument('scriptfile', type=str,
                        help='Path to the script to be run. '
                        'Usually run with `python path/to/script`.')
    parser.add_argument('--gpu', dest='gpu_device', type=int, default=0,
                        help='If applicable, which GPU device to run '
                        'on. Default: 0.')
    return parser.parse_args()
    return parser.parse_args()


def main():
    args = parse_args()

    # Customizable constants.
    scriptfile = args.scriptfile
    cprofile_sortby = 'tottime'
    cprofile_topk = 15
    autograd_prof_sortby = 'cpu_time'
    autograd_prof_topk = 15
    gpu_device = args.gpu_device

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
    autograd_prof = run_autograd_prof(code, globs, gpu_device)
    cprofile_prof = run_cprofile(code, globs, gpu_device)

    print_line()
    print('Environment Summary')
    print_line()
    print(env_summary)

    print_cprofile_summary(cprofile_prof, gpu_device, cprofile_sortby, cprofile_topk)
    print_autograd_prof_summary(autograd_prof, gpu_device, autograd_prof_sortby, autograd_prof_topk)

if __name__ == '__main__':
    main()
