import argparse
import subprocess
from collections import namedtuple

run = subprocess.check_output
srun = run

CPUInfo = namedtuple('CPUInfo', ['processor', 'physical_id', 'core_id'])


def get_cpus():
    with open('/proc/cpuinfo', 'r') as f:
        raw_out = f.read()
    relevant_lines = [l for l in raw_out.split('\n')
                      if 'processor' in l or 'physical id' in l or 'core id' in l]
    assert len(relevant_lines) % 3 == 0
    line_data = [int(l[l.index(':') + 1:].strip()) for l in relevant_lines]

    cpus = [CPUInfo(*line_data[i:i + 3]) for i in range(0, len(line_data), 3)]
    assert len(cpus) % 2 == 0
    return cpus


def set_cpu_state(cpu, enabled):
    with open('/sys/devices/system/cpu/cpu{}/online'.format(cpu.processor), 'w') as f:
        f.write('1' if enabled else '0')


################################################################################
# HyperThreading
################################################################################

def disable_ht():
    print('> Disabling HyperThreading...')

    cpus = get_cpus()

    seen_phys_cores = set()
    cpus_to_disable = set()
    for cpu in cpus:
        key = (cpu.physical_id, cpu.core_id)
        if key in seen_phys_cores:
            cpus_to_disable.add(cpu)
        else:
            seen_phys_cores.add(key)

    if len(cpus_to_disable) == 0:
        print('    No cores with HyperThreading enabled found.')
    elif len(cpus_to_disable) == (len(cpus) // 2):
        print('    Disabling {} CPUs.'.format(len(cpus_to_disable)))
    else:
        raise RuntimeError('Expected to disable either exactly half or no CPUs. This might be a bug.')

    for cpu in cpus_to_disable:
        set_cpu_state(cpu, False)

    return [cpu for cpu in cpus if cpu not in cpus_to_disable]


def enable_ht():
    for cpu in get_cpus():
        set_cpu_state(cpu, True)


################################################################################
# cpusets
################################################################################

def shield_cpus(bench_cpus, bg_cpus):
    bench_cpuspec = ','.join(str(cpu.processor) for cpu in bench_cpus)
    bg_cpuspec = ','.join(str(cpu.processor) for cpu in bg_cpus)
    # Set up our cpusets
    srun(['cset', 'set', '--set=bg', '--cpu=' + bg_cpuspec, '--mem=1'])
    srun(['cset', 'set', '--set=bench', '--cpu=' + bench_cpuspec, '--mem=0'])
    # Move as many tasks (both userspace and kernel) as we can to the bg cpuset
    srun(['cset', 'proc', '--move', '--fromset=root', '--toset=bg', '--kthread'])


def remove_shield():
    srun(['cset', 'set', '--destroy', '--set=bg'])
    srun(['cset', 'set', '--destroy', '--set=bench'])

################################################################################
# CPU Turbo Mode
################################################################################


def set_turbo(value):
    with open('/sys/devices/system/cpu/intel_pstate/no_turbo', 'w') as f:
        f.write('0' if value else '1')

################################################################################
# Helpers
################################################################################


def isolate_bench_subset(cpus):
    bench_cpus = [cpu for cpu in cpus if cpu.physical_id == 0]
    bg_cpus = [cpu for cpu in cpus if cpu.physical_id != 0]
    assert len(bench_cpus) > 0, "No CPUs on NUMA node 0!"
    assert len(bg_cpus) > 0, "Expected at least two NUMA nodes!"
    return bench_cpus, bg_cpus

################################################################################
# Setup/Teardown
################################################################################


def setup_benchmark_env():
    set_turbo(False)
    all_active_cpus = disable_ht()
    bench_cpus, bg_cpus = isolate_bench_subset(all_active_cpus)
    shield_cpus(bench_cpus, bg_cpus)
    with open('bench_cpus', 'w') as f:
        f.write(','.join(str(cpu.processor) for cpu in bench_cpus))


def teardown_benchmark_env():
    remove_shield()
    enable_ht()
    set_turbo(True)


def main():
    parser = argparse.ArgumentParser(description='Configure benchmarking environment')
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--teardown', action='store_true')
    args = parser.parse_args()

    assert args.setup ^ args.teardown

    if args.setup:
        setup_benchmark_env()
    else:
        teardown_benchmark_env()

if __name__ == '__main__':
    main()
