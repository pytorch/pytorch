import collections
import json
import multiprocessing
import multiprocessing.dummy
import os
import queue
import pickle
import statistics
import subprocess
import sys
import textwrap
import threading
import traceback
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
OP_BENCHMARK_ROOT = os.path.split(os.path.split(ROOT)[0])[0]

HEAD = "head"
VERSIONS = (HEAD, "1.6", "1.5", "1.4")
ENV_TEMPLATE = "historic_microbenchmark_{version}"

Task = collections.namedtuple("Task", ("version", "num_cores", "device", "tag_filter"))

CPU_QUEUE = queue.Queue()
for i in range(0, multiprocessing.cpu_count() - 6, 4):
    CPU_QUEUE.put(i)

GPU_QUEUE = queue.Queue()
GPU_QUEUE.put(0)
GPU_QUEUE.put(1)

RESULT_QUEUE = queue.Queue()


def make_env(version):
    assert version in VERSIONS
    env_name = ENV_TEMPLATE.format(version=version)

    cmd = textwrap.dedent(f"""
        conda env remove --name {env_name} 2> /dev/null || true
        conda create --no-default-packages -yn {env_name} python=3
        source activate {env_name}
        conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi hypothesis
        conda install -y -c pytorch magma-cuda102
    """).strip().replace("\n", " && ")

    print(f"Making clean env: {env_name}")
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert not result.returncode

    if version == HEAD:
        cmd = (
            f"cd {ROOT} && cd $(git rev-parse --show-toplevel) "
            f"&& source activate {env_name} && python setup.py clean && "
            "python setup.py install"
        )
        print("Building PyTorch:")
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert not result.returncode
    else:
        print(f"Installing pytorch=={version} and patching benchmark utilities.")
        cmd = (
            f"source activate {env_name} && conda install -y -c pytorch pytorch=={version} && "
            f"cd {OP_BENCHMARK_ROOT}/pt_extension && python setup.py install &&"
            "cp -r $(git rev-parse --show-toplevel)/torch/utils/_benchmark "
            "$(python -c 'import torch;import os;print(os.path.dirname(os.path.abspath(torch.__file__)))')/utils/"
        )
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert not result.returncode


def launch_subtask(t: Task):
    cpu = None
    gpu = None
    try:
        cpu = CPU_QUEUE.get()
        if t.device == "cuda":
            gpu = GPU_QUEUE.get()

        cpu_list = str(cpu) if t.num_cores == 1 else f"{cpu}-{cpu + t.num_cores - 1}"
        cmd = (
            f"cd {OP_BENCHMARK_ROOT} && "
            f"source activate {ENV_TEMPLATE.format(version=t.version)} && "
            f"taskset --cpu-list {cpu_list} "
            f"python -m pt.benchmark_all_test --framework PyTorch "
            f"--tag_filter {t.tag_filter} --ai_pep_format --device {t.device} "
            f"--omp_num_threads {t.num_cores} --mkl_num_threads  {t.num_cores}"
        )
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)},
        )
        if not result.returncode:
            RESULT_QUEUE.put((t, result.stdout.decode("utf-8")))
        else:
            RESULT_QUEUE.put((None, None))
            print(
                f"Run failed: {t}\n"
                f"stdout:\n{result.stdout.decode('utf-8')}\n"
                f"stderr:\n{result.stderr.decode('utf-8')}")

    except KeyboardInterrupt:
        pass

    finally:
        if cpu is not None:
            CPU_QUEUE.put(cpu)
        if gpu is not None:
            GPU_QUEUE.put(gpu)


def parse_output():
    t, stdout = RESULT_QUEUE.get(timeout=3600)
    if t is None:
        return None, None
    results = []
    for l in stdout.splitlines():
        if l.startswith("# Benchmarking PyTorch"):
            results.append([])
            continue

        if l.startswith("PyTorchObserver "):
            l = l[len("PyTorchObserver "):]
            data = json.loads(l.strip())
            results[-1].append(data)

    return t, results


def process(results):
    structured_results = collections.defaultdict(list)
    for t, r in results:
        if t is None:
            continue

        for ri in r:
            types = {i["type"] for i in ri}
            assert len(types) == 1
            run_type = types.pop()

            assert all(i["metric"] == "latency" for i in ri)

            times = tuple(float(j["value"]) * {"ms": 1e-3}[j["unit"]] for j in ri)
            key = (t.num_cores, t.device, run_type)
            structured_results[key].append((t.version, times))

    sorted_results = {}
    for k in sorted(structured_results.keys()):
        v = sorted(structured_results[k])
        assert v[-1][0] == HEAD
        sorted_results[k] = v

    return sorted_results


def run():
    cpu_tasks, gpu_tasks = [], []
    for v in VERSIONS:
        cpu_tasks.extend([
            Task(v, 1, "cpu", "short"),
            Task(v, 1, "cpu", "long"),
            Task(v, 2, "cpu", "short"),
            Task(v, 2, "cpu", "long"),
            Task(v, 4, "cpu", "short"),
            Task(v, 4, "cpu", "long"),
        ])
        gpu_tasks.append(Task(v, 2, "cuda", "all"))

    gpu_pool = multiprocessing.dummy.Pool(GPU_QUEUE.qsize())
    cpu_pool = multiprocessing.dummy.Pool(CPU_QUEUE.qsize())

    gpu_work = gpu_pool.map_async(launch_subtask, gpu_tasks, 1)
    time.sleep(0.5)
    cpu_work = cpu_pool.map_async(launch_subtask, cpu_tasks, 1)

    results = []
    n_tasks = len(cpu_tasks) + len(gpu_tasks)
    for i in range(1, n_tasks + 1):
        results.append(parse_output())
        print(f"\r{i} / {n_tasks}", end="")
    print("\r")

    # Snapshot results.
    with open("/tmp/microbenchmarks.pkl", "wb") as f:
        pickle.dump(results, f)

    gpu_work.wait()
    cpu_work.wait()

    parsed_results = process(results)
    with open("/tmp/microbenchmarks_parsed.pkl", "wb") as f:
        pickle.dump(parsed_results, f)


def main():
    # for v in VERSIONS:
    #     make_env(v)
    run()


if __name__ == "__main__":
    main()
