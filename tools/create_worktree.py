#!/usr/bin/env python3
"""Create or remove git worktrees with submodules cloned locally.

This avoids fetching submodules from remote, which is slow for large repos
like PyTorch. Instead, each submodule is cloned directly from the local
checkout's git object store, so no network access is needed.

If the source checkout was built with a Python from a virtualenv (detected
via build/CMakeCache.txt PYTHON_EXECUTABLE), the venv is cloned into the
new worktree using hardlinks where possible.

Usage:
    python tools/create_worktree.py                  # pytorch-worktree-1
    python tools/create_worktree.py my-worktree      # custom name
    python tools/create_worktree.py remove my-worktree  # force-remove
"""

import argparse
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_existing_worktrees(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    names = []
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            names.append(Path(line.split(" ", 1)[1]).name)
    return names


def next_worktree_name(repo_root: Path) -> str:
    existing = get_existing_worktrees(repo_root)
    n = 1
    while True:
        name = f"pytorch-worktree-{n}"
        if name not in existing:
            return name
        n += 1


def parse_gitmodules(root: Path) -> list[dict[str, str]]:
    gitmodules = root / ".gitmodules"
    if not gitmodules.exists():
        return []
    config = configparser.ConfigParser()
    config.read(gitmodules)
    modules = []
    for section in config.sections():
        if section.startswith('submodule "'):
            path = config.get(section, "path")
            url = config.get(section, "url")
            modules.append({"path": path, "url": url})
    return modules


def get_submodule_commit(parent_repo: Path, submodule_path: str) -> str | None:
    """Get the commit hash a parent repo expects for a submodule."""
    result = subprocess.run(
        ["git", "ls-tree", "HEAD", submodule_path],
        capture_output=True,
        text=True,
        cwd=parent_repo,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # format: <mode> <type> <hash>\t<path>
    parts = result.stdout.strip().split()
    return parts[2] if len(parts) >= 3 else None


def resolve_git_dir(submodule_worktree: Path) -> Path | None:
    """Resolve the actual git object directory for a submodule.

    Submodules can have .git as either a directory (standalone clone) or a
    file containing 'gitdir: <path>' (gitlink). This resolves to the actual
    git directory in both cases.
    """
    git_path = submodule_worktree / ".git"
    if not git_path.exists():
        return None
    if git_path.is_dir():
        return git_path
    # gitlink: read the target
    content = git_path.read_text().strip()
    if content.startswith("gitdir: "):
        target = content[len("gitdir: ") :]
        resolved = (submodule_worktree / target).resolve()
        if resolved.exists():
            return resolved
    return None


def clone_submodule_recursive(
    worktree_root: Path,
    source_root: Path,
    submodule_path: str,
    url: str,
    depth: int = 0,
) -> None:
    prefix = "  " * depth
    source_sub = source_root / submodule_path
    worktree_sub = worktree_root / submodule_path

    git_dir = resolve_git_dir(source_sub)
    if git_dir is None:
        print(f"{prefix}  skipping {submodule_path} (not initialized in source)")
        return

    commit = get_submodule_commit(source_root, submodule_path)

    print(f"{prefix}  cloning {submodule_path}...")

    # The worktree checkout creates empty dirs for submodule paths; remove them
    # so git clone can create the directory itself.
    if worktree_sub.exists():
        try:
            worktree_sub.rmdir()
        except OSError:
            pass

    # Clone from the resolved git directory - works for both standalone .git
    # dirs and gitlinks pointing into .git/modules/. No network needed.
    subprocess.run(
        ["git", "clone", str(git_dir), str(worktree_sub)],
        check=True,
        capture_output=True,
        text=True,
    )

    # Checkout the exact commit the parent repo expects.
    if commit:
        subprocess.run(
            ["git", "checkout", "--quiet", commit],
            cwd=worktree_sub,
            check=True,
            capture_output=True,
            text=True,
        )

    # Point the remote back to the real URL so future fetches work.
    subprocess.run(
        ["git", "remote", "set-url", "origin", url],
        cwd=worktree_sub,
        check=True,
        capture_output=True,
        text=True,
    )

    # Register nested submodules in this repo's config so that
    # `git submodule status --recursive` recognizes them.
    subprocess.run(
        ["git", "submodule", "init"],
        cwd=worktree_sub,
        capture_output=True,
        text=True,
    )

    # Recurse into nested submodules
    nested = parse_gitmodules(worktree_sub)
    for mod in nested:
        try:
            clone_submodule_recursive(
                worktree_sub, source_sub, mod["path"], mod["url"], depth + 1
            )
        except subprocess.CalledProcessError as e:
            print(
                f"{prefix}    WARNING: nested {mod['path']}: {e}",
                file=sys.stderr,
            )


def get_build_python(repo_root: Path) -> Path | None:
    """Read PYTHON_EXECUTABLE from build/CMakeCache.txt."""
    cache = repo_root / "build" / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text().splitlines():
        m = re.match(r"^PYTHON_EXECUTABLE:\w+=(.+)$", line)
        if m:
            p = Path(m.group(1))
            if p.exists():
                return p
    return None


def get_venv_info(python: Path) -> tuple[Path, Path] | None:
    """If *python* lives inside a venv, return (venv_prefix, base_python).

    Returns None if it's not a venv.
    """
    # Walk up from e.g. /home/dev/py3.11/bin/python to /home/dev/py3.11
    # Don't resolve() - venv bin/ contains symlinks to the base Python,
    # and resolving would jump out of the venv directory.
    prefix = python.absolute().parent.parent
    cfg = prefix / "pyvenv.cfg"
    if not cfg.exists():
        return None
    # Parse pyvenv.cfg to find the base Python
    home = None
    for line in cfg.read_text().splitlines():
        key, _, val = line.partition("=")
        if key.strip() == "home":
            home = Path(val.strip())
            break
    if home is None:
        return None
    # The base Python is the one in the "home" directory with the same name
    base_python = home / python.name
    if not base_python.exists():
        # Try common fallbacks
        for name in ("python3", "python"):
            candidate = home / name
            if candidate.exists():
                base_python = candidate
                break
    return (prefix, base_python)


def get_editable_packages(site_packages: Path) -> set[str]:
    """Return names of files/dirs in site-packages that belong to editable installs.

    Handles both PEP 660 editable installs (direct_url.json with editable=True)
    and legacy setup.py develop installs (.egg-link + easy-install.pth).
    """
    editables: set[str] = set()

    # PEP 660 editable installs
    for dist_info in site_packages.glob("*.dist-info"):
        du = dist_info / "direct_url.json"
        if not du.exists():
            continue
        try:
            data = json.loads(du.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("dir_info", {}).get("editable", False):
            continue
        editables.add(dist_info.name)
        top_level = dist_info / "top_level.txt"
        if top_level.exists():
            for pkg in top_level.read_text().splitlines():
                pkg = pkg.strip()
                if pkg:
                    editables.add(pkg)
                    editables.add(f"{pkg}.py")
        record = dist_info / "RECORD"
        if record.exists():
            for line in record.read_text().splitlines():
                top = line.split(",")[0].split("/")[0]
                if top and top != "..":
                    editables.add(top)

    # Legacy setup.py develop installs (.egg-link files)
    for egg_link in site_packages.glob("*.egg-link"):
        editables.add(egg_link.name)
    # easy-install.pth references editable source directories
    easy_install = site_packages / "easy-install.pth"
    if easy_install.exists():
        editables.add("easy-install.pth")

    return editables


def find_site_packages(venv_prefix: Path) -> Path | None:
    """Find the site-packages directory inside a venv."""
    lib = venv_prefix / "lib"
    if not lib.exists():
        return None
    for pydir in sorted(lib.iterdir()):
        sp = pydir / "site-packages"
        if sp.is_dir():
            return sp
    return None


def clone_venv(source_venv: Path, base_python: Path, dest: Path) -> None:
    """Clone a virtualenv to *dest*, hardlinking site-packages where possible.

    Editable installs (like torch itself) are excluded since they point back
    to the source tree and the worktree will need its own build.
    """
    print(f"Creating venv at {dest} (base: {base_python})")
    subprocess.run(
        [str(base_python), "-m", "venv", str(dest)],
        check=True,
    )

    source_sp = find_site_packages(source_venv)
    dest_sp = find_site_packages(dest)
    if source_sp is None or dest_sp is None:
        print("  WARNING: could not locate site-packages, skipping package clone")
        return

    editables = get_editable_packages(source_sp)
    copied = 0
    skipped = 0

    for entry in sorted(source_sp.iterdir()):
        if entry.name in editables:
            skipped += 1
            continue
        # Skip __pycache__ at the top level
        if entry.name == "__pycache__":
            continue
        dest_entry = dest_sp / entry.name
        # Don't overwrite things the fresh venv already has (pip, setuptools, etc)
        if dest_entry.exists():
            shutil.rmtree(dest_entry) if dest_entry.is_dir() else dest_entry.unlink()
        try:
            if entry.is_dir():
                # Try hardlink tree (fast, no extra disk), fall back to copy
                _copytree_hardlink(entry, dest_entry)
            else:
                _link_or_copy(entry, dest_entry)
            copied += 1
        except OSError as e:
            print(f"  WARNING: failed to clone {entry.name}: {e}", file=sys.stderr)

    print(f"  Cloned {copied} packages ({skipped} editable skipped)")


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copytree_hardlink(src: Path, dst: Path) -> None:
    """Copy a directory tree using hardlinks where possible."""
    shutil.copytree(
        src,
        dst,
        copy_function=_link_or_copy_shutil,
    )


def _link_or_copy_shutil(src: str, dst: str) -> None:
    """shutil.copytree-compatible copy function that tries hardlinks first."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def cmd_create(args: argparse.Namespace) -> None:
    repo_root = get_repo_root()
    parent_dir = Path(args.parent_dir) if args.parent_dir else repo_root.parent
    name = args.name or next_worktree_name(repo_root)
    worktree_path = parent_dir / name

    print(f"Creating worktree at {worktree_path}")

    cmd = ["git", "worktree", "add", str(worktree_path), args.commit]
    subprocess.run(cmd, check=True, cwd=repo_root)

    modules = parse_gitmodules(repo_root)
    if modules:
        print(f"Cloning {len(modules)} submodules from local checkout...")
        for mod in modules:
            try:
                clone_submodule_recursive(
                    worktree_path, repo_root, mod["path"], mod["url"]
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"  WARNING: failed to clone {mod['path']}: {e}",
                    file=sys.stderr,
                )
                if e.stderr:
                    print(f"    {e.stderr.strip()}", file=sys.stderr)

    if not args.no_venv:
        build_python = get_build_python(repo_root)
        if build_python is not None:
            venv_info = get_venv_info(build_python)
            if venv_info is not None:
                source_venv, base_python = venv_info
                venv_dest = worktree_path / ".venv"
                print(f"\nDetected build venv: {source_venv}")
                try:
                    clone_venv(source_venv, base_python, venv_dest)
                except (subprocess.CalledProcessError, OSError) as e:
                    print(f"WARNING: venv clone failed: {e}", file=sys.stderr)
            else:
                print(
                    f"\nBuild python ({build_python}) is not a venv, skipping venv clone"
                )
        else:
            print("\nNo build/CMakeCache.txt found, skipping venv clone")

    print(f"\nWorktree ready at {worktree_path}")


def cmd_remove(args: argparse.Namespace) -> None:
    repo_root = get_repo_root()
    parent_dir = Path(args.parent_dir) if args.parent_dir else repo_root.parent
    worktree_path = parent_dir / args.name

    print(f"Removing worktree at {worktree_path}")
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        check=True,
        cwd=repo_root,
    )
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or remove git worktrees with locally-cloned submodules."
    )
    subparsers = parser.add_subparsers(dest="command")

    # "create" subcommand (also the default when no subcommand given)
    create_parser = subparsers.add_parser(
        "create", help="Create a new worktree (default)"
    )
    create_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Worktree directory name (default: pytorch-worktree-$n)",
    )
    create_parser.add_argument(
        "--commit",
        default="HEAD",
        help="Commit/branch to check out in the worktree (default: HEAD)",
    )
    create_parser.add_argument(
        "--parent-dir",
        default=None,
        help="Parent directory for the worktree (default: parent of repo root)",
    )
    create_parser.add_argument(
        "--no-venv",
        action="store_true",
        default=False,
        help="Skip cloning the build virtualenv into the worktree",
    )

    # "remove" subcommand
    remove_parser = subparsers.add_parser("remove", help="Force-remove a worktree")
    remove_parser.add_argument(
        "name",
        help="Worktree directory name to remove",
    )
    remove_parser.add_argument(
        "--parent-dir",
        default=None,
        help="Parent directory of the worktree (default: parent of repo root)",
    )

    args = parser.parse_args()

    # Default to "create" when no subcommand is given
    if args.command is None:
        args.command = "create"
        args.name = None
        args.commit = "HEAD"
        args.parent_dir = None
        args.no_venv = False

    if args.command == "create":
        cmd_create(args)
    elif args.command == "remove":
        cmd_remove(args)


if __name__ == "__main__":
    main()
