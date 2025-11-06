import hashlib
import subprocess
from pathlib import Path

import click
import spin


DEFAULT_HASH_FILE = Path(".lintbin/.lintrunner.sha256")

DEFAULT_FILES_TO_HASH = [
    "requirements.txt",
    "pyproject.toml",
    ".lintrunner.toml",
]


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


def _updated_hashes():
    new_hashes = _hash_files(DEFAULT_FILES_TO_HASH)
    old_hashes = _read_hashes(DEFAULT_HASH_FILE)
    if new_hashes != old_hashes:
        return new_hashes
    return None


@click.command()
def setup_lint():
    """Set up latest version of lintrunner."""
    cmd = ["uvx", "lintrunner@latest", "init"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@spin.util.extend_command(
    setup_lint,
    doc=f"""
        If configuration has changed, update lintrunner.

        Compares the stored old hashes of configuration files with new ones and
        performs setup via setup-lint if the hashes have changed.
        Hashes are stored in {DEFAULT_HASH_FILE}; the following files are
        considered: {", ".join(DEFAULT_FILES_TO_HASH)}.
        """,
)
def lazy_setup_lint(*, parent_callback, **kwargs):
    if hashes := _updated_hashes():
        click.echo(
            "Changes detected in lint configuration files. Setting up linting tools..."
        )
        parent_callback(**kwargs)
        hash_file = DEFAULT_HASH_FILE
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with hash_file.open("w") as f:
            for file, hash in hashes.items():
                f.write(f"{hash}  {file}\n")
        click.echo("Linting tools set up and hashes updated.")
    else:
        click.echo("No changes detected in lint configuration files. Skipping setup.")


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
