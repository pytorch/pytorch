import os
import subprocess
import textwrap


ROOT = os.path.dirname(os.path.abspath(__file__))

HEAD = "head"
VERSIONS = (HEAD, "1.6", "1.5", "1.4")
ENV_TEMPLATE = "historic_microbenchmark_{version}"

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
            f"cd $(git rev-parse --show-toplevel)/benchmarks/operator_benchmark/pt_extension && "
            "python setup.py install &&"
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


def main():
    for v in VERSIONS:
        make_env(v)


if __name__ == "__main__":
    main()
