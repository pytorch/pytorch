#!/usr/bin/env python3
"""Create or remove git worktrees with submodules cloned locally.

This avoids fetching submodules from remote, which is slow for large repos
like PyTorch. Instead, each submodule is cloned directly from the local
checkout's git object store, so no network access is needed.

Usage:
    python tools/create_worktree.py                  # pytorch-worktree-1
    python tools/create_worktree.py my-worktree      # custom name
    python tools/create_worktree.py remove my-worktree  # force-remove
"""

import argparse
import configparser
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

    # Clone from the resolved git directory — works for both standalone .git
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

    if args.command == "create":
        cmd_create(args)
    elif args.command == "remove":
        cmd_remove(args)


if __name__ == "__main__":
    main()
