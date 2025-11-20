import glob
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
import spin


def file_digest(file, algorithm: str):
    try:
        return hashlib.file_digest(file, algorithm)
    except AttributeError:
        pass  # Fallback to manual implementation below
    hash = hashlib.new(algorithm)
    while chunk := file.read(8192):
        hash.update(chunk)
    return hash


def _hash_file(file):
    with open(file, "rb") as f:
        hash = file_digest(f, "sha256")
    return hash.hexdigest()


def _hash_files(files):
    hashes = {file: _hash_file(file) for file in files}
    return hashes


def _read_hashes(hash_file: Path):
    if not hash_file.exists():
        return {}
    with hash_file.open("r") as f:
        lines = f.readlines()
    hashes = {}
    for line in lines:
        hash = line[:64]
        file = line[66:].strip()
        hashes[file] = hash
    return hashes


def _updated_hashes(hash_file, files_to_hash):
    old_hashes = _read_hashes(hash_file)
    new_hashes = _hash_files(files_to_hash)
    if new_hashes != old_hashes:
        return new_hashes
    return None


@click.command()
def regenerate_version():
    """Regenerate version.py."""
    cmd = [
        sys.executable,
        "-m",
        "tools.generate_torch_version",
        "--is-debug=false",
    ]
    spin.util.run(cmd)


TYPE_STUBS = [
    (
        "Pytorch type stubs",
        Path(".lintbin/.pytorch-type-stubs.sha256"),
        [
            "aten/src/ATen/native/native_functions.yaml",
            "aten/src/ATen/native/tags.yaml",
            "tools/autograd/deprecated.yaml",
        ],
        [
            sys.executable,
            "-m",
            "tools.pyi.gen_pyi",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--deprecated-functions-path",
            "tools/autograd/deprecated.yaml",
        ],
    ),
    (
        "Datapipes type stubs",
        None,
        [],
        [
            sys.executable,
            "torch/utils/data/datapipes/gen_pyi.py",
        ],
    ),
]


@click.command()
def regenerate_type_stubs():
    """Regenerate type stubs."""
    for name, hash_file, files_to_hash, cmd in TYPE_STUBS:
        if hash_file:
            if hashes := _updated_hashes(hash_file, files_to_hash):
                click.echo(
                    f"Changes detected in type stub files for {name}. Regenerating..."
                )
                spin.util.run(cmd)
                hash_file.parent.mkdir(parents=True, exist_ok=True)
                with hash_file.open("w") as f:
                    for file, hash in hashes.items():
                        f.write(f"{hash}  {file}\n")
                click.echo("Type stubs and hashes updated.")
            else:
                click.echo(f"No changes detected in type stub files for {name}.")
        else:
            click.echo(f"No hash file for {name}. Regenerating...")
            spin.util.run(cmd)
            click.echo("Type stubs regenerated.")


@click.command()
def regenerate_clangtidy_files():
    """Regenerate clang-tidy files."""
    cmd = [
        sys.executable,
        "-m",
        "tools.linter.clang_tidy.generate_build_files",
    ]
    spin.util.run(cmd)


#: These linters are expected to need less than 3s cpu time total
VERY_FAST_LINTERS = {
    "ATEN_CPU_GPU_AGNOSTIC",
    "BAZEL_LINTER",
    "C10_NODISCARD",
    "C10_UNUSED",
    "CALL_ONCE",
    "CMAKE_MINIMUM_REQUIRED",
    "CONTEXT_DECORATOR",
    "COPYRIGHT",
    "CUBINCLUDE",
    "DEPLOY_DETECTION",
    "ERROR_PRONE_ISINSTANCE",
    "EXEC",
    "HEADER_ONLY_LINTER",
    "IMPORT_LINTER",
    "INCLUDE",
    "LINTRUNNER_VERSION",
    "MERGE_CONFLICTLESS_CSV",
    "META_NO_CREATE_UNBACKED",
    "NEWLINE",
    "NOQA",
    "NO_WORKFLOWS_ON_FORK",
    "ONCE_FLAG",
    "PYBIND11_INCLUDE",
    "PYBIND11_SPECIALIZATION",
    "PYPIDEP",
    "PYPROJECT",
    "RAWCUDA",
    "RAWCUDADEVICE",
    "ROOT_LOGGING",
    "TABS",
    "TESTOWNERS",
    "TYPEIGNORE",
    "TYPENOSKIP",
    "WORKFLOWSYNC",
}


#: These linters are expected to take a few seconds, but less than 10s cpu time total
FAST_LINTERS = {
    "CMAKE",
    "DOCSTRING_LINTER",
    "GHA",
    "NATIVEFUNCTIONS",
    "RUFF",
    "SET_LINTER",
    "SHELLCHECK",
    "SPACES",
}


#: These linters are expected to take more than 10s cpu time total;
#: some need more than 1 hour.
SLOW_LINTERS = {
    "ACTIONLINT",
    "CLANGFORMAT",
    "CLANGTIDY",
    "CODESPELL",
    "FLAKE8",
    "GB_REGISTRY",
    "PYFMT",
    "PYREFLY",
    "TEST_DEVICE_BIAS",
    "TEST_HAS_MAIN",
}


ALL_LINTERS = VERY_FAST_LINTERS | FAST_LINTERS | SLOW_LINTERS


LINTRUNNER_CACHE_INFO = (
    Path(".lintbin/.lintrunner.sha256"),
    [
        "requirements.txt",
        "pyproject.toml",
        ".lintrunner.toml",
    ],
)


LINTRUNNER_BASE_CMD = [
    "uvx",
    "--python",
    "3.10",
    "lintrunner@0.12.7",
]


@click.command()
def setup_lint():
    """Set up lintrunner with current CI version."""
    cmd = LINTRUNNER_BASE_CMD + ["init"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _check_linters():
    cmd = LINTRUNNER_BASE_CMD + ["list"]
    ret = spin.util.run(cmd, output=False, stderr=subprocess.PIPE)
    linters = {l.strip() for l in ret.stdout.decode().strip().split("\n")[1:]}
    unknown_linters = linters - ALL_LINTERS
    missing_linters = ALL_LINTERS - linters
    if unknown_linters:
        click.secho(
            f"Unknown linters found; please add them to the correct category "
            f"in .spin/cmds.py: {', '.join(unknown_linters)}",
            fg="yellow",
        )
    if missing_linters:
        click.secho(
            f"Missing linters found; please update the corresponding category "
            f"in .spin/cmds.py: {', '.join(missing_linters)}",
            fg="yellow",
        )
    return unknown_linters, missing_linters


@spin.util.extend_command(
    setup_lint,
    doc=f"""
        If configuration has changed, update lintrunner.

        Compares the stored old hashes of configuration files with new ones and
        performs setup via setup-lint if the hashes have changed.
        Hashes are stored in {LINTRUNNER_CACHE_INFO[0]}; the following files are
        considered: {", ".join(LINTRUNNER_CACHE_INFO[1])}.
        """,
)
@click.pass_context
def lazy_setup_lint(ctx, parent_callback, **kwargs):
    if hashes := _updated_hashes(*LINTRUNNER_CACHE_INFO):
        click.echo(
            "Changes detected in lint configuration files. Setting up linting tools..."
        )
        parent_callback(**kwargs)
        hash_file = LINTRUNNER_CACHE_INFO[0]
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with hash_file.open("w") as f:
            for file, hash in hashes.items():
                f.write(f"{hash}  {file}\n")
        click.echo("Linting tools set up and hashes updated.")
    else:
        click.echo("No changes detected in lint configuration files. Skipping setup.")
    click.echo("Regenerating version...")
    ctx.invoke(regenerate_version)
    click.echo("Regenerating type stubs...")
    ctx.invoke(regenerate_type_stubs)
    click.echo("Done.")
    _check_linters()


@click.command()
@click.option("-a", "--apply-patches", is_flag=True)
@click.pass_context
def lint(ctx, apply_patches, **kwargs):
    """Lint all files."""
    ctx.invoke(lazy_setup_lint)
    all_files_linters = VERY_FAST_LINTERS | FAST_LINTERS
    changed_files_linters = SLOW_LINTERS
    cmd = LINTRUNNER_BASE_CMD
    if apply_patches:
        cmd += ["--apply-patches"]
    all_files_cmd = cmd + [
        "--take",
        ",".join(all_files_linters),
        "--all-files",
    ]
    spin.util.run(all_files_cmd)
    changed_files_cmd = cmd + [
        "--take",
        ",".join(changed_files_linters),
    ]
    spin.util.run(changed_files_cmd)


@click.command()
@click.pass_context
def fixlint(ctx, **kwargs):
    """Autofix all files."""
    ctx.invoke(lint, apply_patches=True)


@click.command()
@click.option("-a", "--apply-patches", is_flag=True)
@click.pass_context
def quicklint(ctx, apply_patches, **kwargs):
    """Lint changed files."""
    ctx.invoke(lazy_setup_lint)
    cmd = LINTRUNNER_BASE_CMD
    if apply_patches:
        cmd += ["--apply-patches"]
    spin.util.run(cmd)


@click.command()
@click.pass_context
def quickfix(ctx, **kwargs):
    """Autofix changed files."""
    ctx.invoke(quicklint, apply_patches=True)


@click.command()
def clean():
    """Clean, that is remove all files in .gitignore except in the NOT-CLEAN-FILES section."""
    ignores = Path(".gitignore").read_text(encoding="utf-8")
    for wildcard in filter(None, ignores.splitlines()):
        if wildcard.strip().startswith("#"):
            if "BEGIN NOT-CLEAN-FILES" in wildcard:
                # Marker is found and stop reading .gitignore.
                break
            # Ignore lines which begin with '#'.
        else:
            # Don't remove absolute paths from the system
            wildcard = wildcard.lstrip("./")
            for filename in glob.iglob(wildcard):
                try:
                    os.remove(filename)
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)


@click.command()
def regenerate_github_workflows():
    """Regenerate GitHub workflows from templates."""
    cmd = [sys.executable, "scripts/generate_ci_workflows.py"]
    spin.util.run(cmd, cwd="./.github")
