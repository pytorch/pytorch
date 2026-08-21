#!/usr/bin/env python3
"""Extract a source-file -> final-build-artifact map from a ninja build dir.

Combines two sources of edges:
  * `ninja -t deps` -- compiler-discovered header/source -> object edges.
  * `build.ninja` build statements -- every explicit input -> output edge,
    which covers object -> .so links and codegen edges (native_functions.yaml,
    torchgen/**, templates -> generated .cpp -> object -> .so).

We then compute, for each input file, the set of *terminal* artifacts it can
reach (shared libs, static libs, and linked executables). Terminals are what
target determination cares about: a changed file that reaches libtorch_cpu.so
has global blast radius; one reaching only a leaf artifact is cheap to select.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict


def _unescape(tok):
    return tok.replace("$:", ":").replace("$ ", " ").replace("$$", "$")


def _join_continuations(text):
    out = []
    buf = ""
    for line in text.splitlines():
        if line.endswith("$") and not line.endswith("$$"):
            buf += line[:-1]
        else:
            out.append(buf + line)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _split_tokens(s):
    """Split a ninja path list on unescaped spaces."""
    toks, cur, i = [], "", 0
    while i < len(s):
        c = s[i]
        if c == "$" and i + 1 < len(s):
            cur += s[i : i + 2]
            i += 2
            continue
        if c == " ":
            if cur:
                toks.append(cur)
                cur = ""
        else:
            cur += c
        i += 1
    if cur:
        toks.append(cur)
    return toks


def parse_build_ninja(build_dir, edges):
    """Add input -> {outputs} edges from every `build` statement."""
    path = os.path.join(build_dir, "build.ninja")
    if not os.path.exists(path):
        return
    with open(path, errors="replace") as f:
        lines = _join_continuations(f.read())
    for line in lines:
        if not line.startswith("build "):
            continue
        rest = line[len("build ") :]
        # Outputs and inputs are separated by the first unescaped ':'.
        i, colon = 0, -1
        while i < len(rest):
            if rest[i] == "$" and i + 1 < len(rest):
                i += 2
                continue
            if rest[i] == ":":
                colon = i
                break
            i += 1
        if colon < 0:
            continue
        outs = [_unescape(t) for t in _split_tokens(rest[:colon]) if t not in ("|",)]
        # Input side starts after the rule name; drop rule + separators.
        in_toks = _split_tokens(rest[colon + 1 :])
        ins = [
            _unescape(t)
            for t in in_toks[1:]  # first token is the rule name
            if t not in ("|", "||")
        ]
        for src in ins:
            edges[src].update(outs)


def parse_ninja_deps(build_dir, edges):
    """Add dep-file -> object edges from `ninja -t deps`."""
    try:
        proc = subprocess.run(
            ["ninja", "-C", build_dir, "-t", "deps"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return
    obj = None
    for line in proc.stdout.splitlines():
        if not line:
            obj = None
        elif not line.startswith(" ") and line.rstrip().endswith(")"):
            # e.g. "caffe2/CMakeFiles/torch_cpu.dir/foo.cpp.o: #deps ... (VALID)"
            obj = line.split(":", 1)[0].strip()
        elif line.startswith(" ") and obj:
            edges[line.strip()].add(obj)


TERMINAL_SUFFIXES = (".so", ".a", ".dylib", ".dll")


def _is_terminal(path):
    base = path.rsplit("/", 1)[-1]
    if any(path.endswith(s) or ".so." in base for s in TERMINAL_SUFFIXES):
        return True
    # Linked executables live under bin/ with no extension.
    return "/bin/" in path and "." not in base


def reachable_terminals(edges):
    """Memoized forward DFS: node -> frozenset of terminal artifacts reachable."""
    memo = {}
    WORKING = object()

    def visit(node):
        cached = memo.get(node)
        if cached is not None and cached is not WORKING:
            return cached
        if cached is WORKING:
            return frozenset()  # cycle guard; ninja graphs are DAGs in practice
        memo[node] = WORKING
        acc = set()
        if _is_terminal(node):
            acc.add(node)
        for succ in edges.get(node, ()):
            if succ != node:
                acc |= visit(succ)
        result = frozenset(acc)
        memo[node] = result
        return result

    for node in list(edges.keys()):
        visit(node)
    return memo


def main():
    import sys

    sys.setrecursionlimit(1 << 20)
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    build_dir = os.path.realpath(args.build_dir)
    repo_root = os.path.realpath(args.repo_root)

    edges = defaultdict(set)
    parse_build_ninja(build_dir, edges)
    parse_ninja_deps(build_dir, edges)

    memo = reachable_terminals(edges)

    def repo_rel(node):
        p = node if os.path.isabs(node) else os.path.join(build_dir, node)
        p = os.path.realpath(p)
        if p.startswith(repo_root):
            rel = os.path.relpath(p, repo_root)
            if not rel.startswith(".."):
                return rel
        return None

    mapping = {}
    for node, terminals in memo.items():
        if not terminals:
            continue
        rel = repo_rel(node)
        if rel is None:
            continue
        mapping[rel] = sorted(os.path.basename(t) for t in terminals)

    out = {
        "sources": mapping,
        "meta": {
            "num_edges": sum(len(v) for v in edges.values()),
            "num_sources": len(mapping),
            "build_dir": build_dir,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}: {out['meta']}")


if __name__ == "__main__":
    main()
