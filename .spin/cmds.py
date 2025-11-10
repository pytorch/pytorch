import hashlib
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


def _regenerate_version():
    cmd = [
        sys.executable,
        "-m",
        "tools.generate_torch_version",
        "--is-debug=false",
    ]
    spin.util.run(cmd)


@click.command()
def regenerate_version():
    """Regenerate version.py."""
    _regenerate_version()


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


def _regenerate_type_stubs():
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
def regenerate_type_stubs():
    """Regenerate type stubs."""
    _regenerate_type_stubs()


@click.command()
def regenerate_clangtidy_files():
    """Regenerate clang-tidy files."""
    cmd = [
        sys.executable,
        "-m",
        "tools.linter.clang_tidy.generate_build_files",
    ]
    spin.util.run(cmd)


LINTRUNNER_CACHE_INFO = (
    Path(".lintbin/.lintrunner.sha256"),
    [
        "requirements.txt",
        "pyproject.toml",
        ".lintrunner.toml",
    ],
)


@click.command()
def setup_lint():
    """Set up lintrunner with current CI version."""
    cmd = ["uvx", "--python", "3.10", "lintrunner@0.12.7", "init"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


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
def lazy_setup_lint(*, parent_callback, **kwargs):
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
    _regenerate_version()
    click.echo("Regenerating type stubs...")
    _regenerate_type_stubs()
    click.echo("Done.")


@spin.util.extend_command(
    lazy_setup_lint,
    doc="Lint all files.",
)
def lint(*, parent_callback, **kwargs):
    parent_callback(**kwargs)
    cmd = ["uvx", "lintrunner", "--all-files"]
    spin.util.run(cmd)


@spin.util.extend_command(
    lazy_setup_lint,
    doc="Lint changed files.",
)
def quicklint(*, parent_callback, **kwargs):
    """Run linting tools."""
    parent_callback(**kwargs)
    cmd = ["uvx", "lintrunner"]
    spin.util.run(cmd)


@spin.util.extend_command(
    lazy_setup_lint,
    doc="Autofix changed files.",
)
def quickfix(*, parent_callback, **kwargs):
    """Run linting tools."""
    parent_callback(**kwargs)
    cmd = ["uvx", "lintrunner", "--apply-patches"]
    spin.util.run(cmd)
