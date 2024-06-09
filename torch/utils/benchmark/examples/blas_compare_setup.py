import collections
import os
import shutil
import subprocess

try:
    # no type stub for conda command line interface
    import conda.cli.python_api  # type: ignore[import]
    from conda.cli.python_api import Commands as conda_commands
except ImportError:
    # blas_compare.py will fail to import these when it's inside a conda env,
    # but that's fine as it only wants the constants.
    pass


WORKING_ROOT = "/tmp/pytorch_blas_compare_environments"
MKL_2020_3 = "mkl_2020_3"
MKL_2020_0 = "mkl_2020_0"
OPEN_BLAS = "open_blas"
EIGEN = "eigen"


GENERIC_ENV_VARS = ("USE_CUDA=0", "USE_ROCM=0")
BASE_PKG_DEPS = (
    "cmake",
    "hypothesis",
    "ninja",
    "numpy",
    "pyyaml",
    "setuptools",
    "typing_extensions",
)


SubEnvSpec = collections.namedtuple(
    "SubEnvSpec", (
        "generic_installs",
        "special_installs",
        "environment_variables",

        # Validate install.
        "expected_blas_symbols",
        "expected_mkl_version",
    ))


SUB_ENVS = {
    MKL_2020_3: SubEnvSpec(
        generic_installs=(),
        special_installs=("intel", ("mkl=2020.3", "mkl-include=2020.3")),
        environment_variables=("BLAS=MKL",) + GENERIC_ENV_VARS,
        expected_blas_symbols=("mkl_blas_sgemm",),
        expected_mkl_version="2020.0.3",
    ),

    MKL_2020_0: SubEnvSpec(
        generic_installs=(),
        special_installs=("intel", ("mkl=2020.0", "mkl-include=2020.0")),
        environment_variables=("BLAS=MKL",) + GENERIC_ENV_VARS,
        expected_blas_symbols=("mkl_blas_sgemm",),
        expected_mkl_version="2020.0.0",
    ),

    OPEN_BLAS: SubEnvSpec(
        generic_installs=("openblas",),
        special_installs=(),
        environment_variables=("BLAS=OpenBLAS",) + GENERIC_ENV_VARS,
        expected_blas_symbols=("exec_blas",),
        expected_mkl_version=None,
    ),

    # EIGEN: SubEnvSpec(
    #     generic_installs=(),
    #     special_installs=(),
    #     environment_variables=("BLAS=Eigen",) + GENERIC_ENV_VARS,
    #     expected_blas_symbols=(),
    # ),
}


def conda_run(*args):
    """Convenience method."""
    stdout, stderr, retcode = conda.cli.python_api.run_command(*args)
    if retcode:
        raise OSError(f"conda error: {str(args)}  retcode: {retcode}\n{stderr}")

    return stdout


def main():
    if os.path.exists(WORKING_ROOT):
        print("Cleaning: removing old working root.")
        shutil.rmtree(WORKING_ROOT)
    os.makedirs(WORKING_ROOT)

    git_root = subprocess.check_output(
        "git rev-parse --show-toplevel",
        shell=True,
        cwd=os.path.dirname(os.path.realpath(__file__))
    ).decode("utf-8").strip()

    for env_name, env_spec in SUB_ENVS.items():
        env_path = os.path.join(WORKING_ROOT, env_name)
        print(f"Creating env: {env_name}: ({env_path})")
        conda_run(
            conda_commands.CREATE,
            "--no-default-packages",
            "--prefix", env_path,
            "python=3",
        )

        print("Testing that env can be activated:")
        base_source = subprocess.run(
            f"source activate {env_path}",
            shell=True,
            capture_output=True,
            check=False,
        )
        if base_source.returncode:
            raise OSError(
                "Failed to source base environment:\n"
                f"  stdout: {base_source.stdout.decode('utf-8')}\n"
                f"  stderr: {base_source.stderr.decode('utf-8')}"
            )

        print("Installing packages:")
        conda_run(
            conda_commands.INSTALL,
            "--prefix", env_path,
            *(BASE_PKG_DEPS + env_spec.generic_installs)
        )

        if env_spec.special_installs:
            channel, channel_deps = env_spec.special_installs
            print(f"Installing packages from channel: {channel}")
            conda_run(
                conda_commands.INSTALL,
                "--prefix", env_path,
                "-c", channel, *channel_deps
            )

        if env_spec.environment_variables:
            print("Setting environment variables.")

            # This does not appear to be possible using the python API.
            env_set = subprocess.run(
                f"source activate {env_path} && "
                f"conda env config vars set {' '.join(env_spec.environment_variables)}",
                shell=True,
                capture_output=True,
                check=False,
            )
            if env_set.returncode:
                raise OSError(
                    "Failed to set environment variables:\n"
                    f"  stdout: {env_set.stdout.decode('utf-8')}\n"
                    f"  stderr: {env_set.stderr.decode('utf-8')}"
                )

            # Check that they were actually set correctly.
            actual_env_vars = subprocess.run(
                f"source activate {env_path} && env",
                shell=True,
                capture_output=True,
                check=True,
            ).stdout.decode("utf-8").strip().splitlines()
            for e in env_spec.environment_variables:
                assert e in actual_env_vars, f"{e} not in envs"

        print(f"Building PyTorch for env: `{env_name}`")
        # We have to re-run during each build to pick up the new
        # build config settings.
        build_run = subprocess.run(
            f"source activate {env_path} && "
            f"cd {git_root} && "
            "python setup.py install --cmake",
            shell=True,
            capture_output=True,
            check=True,
        )

        print("Checking configuration:")
        check_run = subprocess.run(
            # Shameless abuse of `python -c ...`
            f"source activate {env_path} && "
            'python -c "'
            "import torch;"
            "from torch.utils.benchmark import Timer;"
            "print(torch.__config__.show());"
            "setup = 'x=torch.ones((128, 128));y=torch.ones((128, 128))';"
            "counts = Timer('torch.mm(x, y)', setup).collect_callgrind(collect_baseline=False);"
            "stats = counts.as_standardized().stats(inclusive=True);"
            "print(stats.filter(lambda l: 'blas' in l.lower()))\"",
            shell=True,
            capture_output=True,
            check=False,
        )
        if check_run.returncode:
            raise OSError(
                "Failed to set environment variables:\n"
                f"  stdout: {check_run.stdout.decode('utf-8')}\n"
                f"  stderr: {check_run.stderr.decode('utf-8')}"
            )
        check_run_stdout = check_run.stdout.decode('utf-8')
        print(check_run_stdout)

        for e in env_spec.environment_variables:
            if "BLAS" in e:
                assert e in check_run_stdout, f"PyTorch build did not respect `BLAS=...`: {e}"

        for s in env_spec.expected_blas_symbols:
            assert s in check_run_stdout

        if env_spec.expected_mkl_version is not None:
            assert f"- Intel(R) Math Kernel Library Version {env_spec.expected_mkl_version}" in check_run_stdout

        print(f"Build complete: {env_name}")


if __name__ == "__main__":
    main()
