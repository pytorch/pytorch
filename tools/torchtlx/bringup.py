#!/usr/bin/env python3
"""TorchTLX bring-up helper: swap the Triton provider and run the TLX tests.

This fork drives Inductor with FBTriton (facebookexperimental/triton) rather
than upstream Triton, because torchTLX lives in FBTriton -- see
torch/_inductor/heuristics/template/tlx.py, which imports
triton.language.extra.tlx.inductor.registry.

That import is wrapped in `except ImportError: pass`, so a Triton without the
TLX Inductor registry does not fail -- TLX just never engages and the tests
quietly measure stock Inductor. `doctor` exists to turn that silent no-op into
a loud one.

Usage:
    python tools/torchtlx/bringup.py doctor
    python tools/torchtlx/bringup.py switch fbtriton
    python tools/torchtlx/bringup.py switch fbtriton --from-source <fbtriton_repo_path>
    python tools/torchtlx/bringup.py switch oai
    python tools/torchtlx/bringup.py test --mode allow
    python tools/torchtlx/bringup.py test --full

`test` runs the torchTLX tests matched by TLX_TEST_PATTERNS; with none present
it falls back to sanity.py, a fast plumbing check. --full adds the Inductor
Triton suite in FULL_TESTS.

Switching providers does NOT require rebuilding torch; Triton is a pure
runtime dependency of Inductor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Every distribution that installs a top-level `triton` package. They collide,
# and uninstalling one after another has overwritten its files leaves a stale
# dist-info with no module, so a swap must remove all of them first.
TRITON_DISTRIBUTIONS = [
    "fbtriton",
    "triton",
    "pytorch-triton",
    "pytorch-triton-rocm",
    "triton-rocm",
]

TLX_REGISTRY = "triton.language.extra.tlx.inductor.registry"

# torchTLX needs triton/language/extra/tlx/inductor, which is absent from the
# fbtriton 3.7.x releases.
FBTRITON_MIN_VERSION = "3.8"

TLX_TEST_PATTERNS = [
    "test/inductor/test_torchtlx*.py",
]

# --full only: 425 tests, ~2.5 min. Too heavy for a plumbing check, but the
# right thing to run when validating a provider swap properly.
FULL_TESTS = ["test/inductor/test_triton_kernels.py"]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


def uv_bin() -> str:
    """uv is required rather than optional.

    The flags used here (--reinstall, --index-strategy) have no pip equivalent,
    so a silent fallback to pip would fail after the uninstall step has already
    removed every Triton.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit(
            "uv is required (https://docs.astral.sh/uv/). `build[uv]` in "
            "requirements.txt pulls it in, so `pip install -r requirements.txt` "
            "is usually enough."
        )
    return uv


def uv_pip(subcommand: str, *args: str) -> list[str]:
    """`uv pip <subcommand> --python <this interpreter> ...`.

    --python pins the target to the interpreter running this script, which is
    also what _probe.py inspects. Without it uv resolves the target from
    VIRTUAL_ENV/CONDA_PREFIX, so `switch` could install into one environment
    while `doctor` reports on another. It is a flag of the subcommand, not of
    `uv pip`.
    """
    return [uv_bin(), "pip", subcommand, "--python", sys.executable, *args]


def clear_caches() -> None:
    """Drop the Inductor and Triton compile caches.

    Cached kernels are not keyed by which Triton produced them, so reusing a
    cache across a provider swap causes a large, confusing wave of failures on
    the first run afterwards (observed: 311 failures that vanished on a second
    run). Done unconditionally on every switch; there is no reason to skip it,
    so it is not exposed as a separate command.

    Paths come from torch rather than being re-derived here: default_cache_dir()
    handles tempfile.gettempdir() vs /var/tmp under fbcode and username
    sanitisation, and Inductor puts the Triton cache *inside* that directory
    (cache_dir()/triton/<device>) unless TRITON_CACHE_DIR overrides it. Guessing
    the paths risks clearing the wrong thing, which is the exact failure this is
    meant to prevent.

    The LLVM toolchain download cache is deliberately left alone -- unrelated to
    compiled kernels and expensive to refetch.
    """
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    targets = [Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", default_cache_dir()))]
    triton_cache = os.environ.get("TRITON_CACHE_DIR")
    if triton_cache:
        # Only a separate target when overridden; otherwise it is nested in the
        # Inductor cache dir above and already covered.
        targets.append(Path(triton_cache))
    for path in targets:
        if path.is_dir():
            print(f"    clearing {path}")
            shutil.rmtree(path, ignore_errors=True)
        else:
            print(f"    (absent) {path}")


def probe() -> dict:
    """Collect environment facts by running _probe.py in a subprocess.

    A subprocess so `doctor` reflects what is installed now, not what this
    process imported before a swap. cwd is set outside the repo so `import
    torch` resolves the installed package rather than the source directory.
    """
    res = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_probe.py")],
        capture_output=True,
        text=True,
        cwd="/",
    )
    for line in res.stdout.splitlines():
        if line.startswith("<<<JSON>>>"):
            return json.loads(line[len("<<<JSON>>>") :])
    raise RuntimeError(f"probe failed:\n{res.stdout}\n{res.stderr}")


def cmd_doctor(args: argparse.Namespace) -> int:
    info = probe()
    dists = info.get("dists", {})

    print("=" * 72)
    print(f"torch          : {info['torch']}")
    print(f"  from         : {info['torch_file']}")
    if info.get("hip"):
        print(f"  backend      : ROCm {info['hip']}")
    elif info.get("cuda"):
        print(f"  backend      : CUDA {info['cuda']}")
    else:
        print("  backend      : CPU-only")
    print(f"  devices      : {info['device_count']} x {info.get('device_name')}")
    if info.get("gcn_arch"):
        print(f"  arch         : {info['gcn_arch']}")
    version = info.get("triton")
    # FBTriton stamps every release with a "+fb" local suffix on the module
    # version. That is the only reliable signal: the PyPI wheel installs as the
    # `fbtriton` distribution, but a build from a checkout installs as `triton`
    # (setup.py defaults TRITON_WHEEL_NAME to "triton"), so the distribution
    # name alone cannot tell the two providers apart.
    is_fbtriton = bool(version) and version.endswith("+fb")
    print(f"triton         : {version or 'NOT IMPORTABLE'}")
    print(f"  provider     : {'FBTriton' if is_fbtriton else 'upstream Triton'}")
    print(f"  from         : {info.get('triton_file')}")
    print(f"  backends     : {info.get('backends')}")
    print(f"  distributions: {dists or 'none'}")
    if info.get("triton_error"):
        print(f"  error        : {info['triton_error']}")
    print("=" * 72)

    problems = []
    if len(dists) > 1:
        problems.append(
            f"multiple triton distributions installed ({', '.join(sorted(dists))}); "
            "they collide on the `triton` package -- run `switch` to clean up"
        )
    if not version:
        problems.append("triton is not importable")
    elif not is_fbtriton:
        problems.append(
            "the active Triton is not FBTriton (no '+fb' version suffix); "
            "torchTLX only exists in FBTriton"
        )
    if info.get("tlx_registry"):
        print(f"TLX registry   : OK ({TLX_REGISTRY})")
    else:
        print(f"TLX registry   : MISSING ({TLX_REGISTRY})")
        print(f"  error        : {info.get('tlx_error')}")
        problems.append(
            "the active Triton has no TLX Inductor registry, so torchTLX will "
            "silently never engage (tlx.py swallows the ImportError). Install "
            f"an FBTriton >={FBTRITON_MIN_VERSION}, which ships "
            "triton/language/extra/tlx/inductor. Either:  "
            "`bringup.py switch fbtriton` (PyPI wheel)  or  "
            "`bringup.py switch fbtriton --from-source <checkout>`"
        )

    if problems:
        print("\nNOT READY for torchTLX:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nREADY for torchTLX.")
    return 0


def resolves(spec: str, extra: list[str] | None = None) -> bool:
    """Check a spec resolves before we uninstall what is already working.

    The install has to be uninstall-then-install because the providers collide
    on the `triton` package, so an unresolvable target would otherwise leave
    the environment with no Triton at all.
    """
    res = run(
        uv_pip("install", "--dry-run", *(extra or []), spec),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"error: cannot resolve {spec!r}; leaving the current install alone")
        print((res.stderr or res.stdout).strip()[-500:])
    return res.returncode == 0


def cmd_switch(args: argparse.Namespace) -> int:
    extra: list[str] = []
    from_source = False

    if args.provider == "fbtriton":
        if args.from_source:
            src = Path(args.from_source).expanduser().resolve()
            if not (src / "setup.py").exists():
                print(f"error: {src} does not look like an fbtriton checkout")
                return 1
            spec = str(src)
            from_source = True
            if args.editable:
                extra = ["-e"]
            label = f"FBTriton from source: {src} (compiles Triton + LLVM, slow)"
        else:
            spec = (
                f"fbtriton=={args.version}"
                if args.version
                else f"fbtriton>={FBTRITON_MIN_VERSION}"
            )
            label = f"FBTriton from PyPI: {spec}"
    else:
        info = probe()
        if info.get("hip"):
            rocm = ".".join(info["hip"].split(".")[:2])
            spec = "pytorch-triton-rocm"
            extra = [
                "--index-url",
                f"https://download.pytorch.org/whl/nightly/rocm{rocm}",
                "--index-strategy",
                "unsafe-best-match",
            ]
            label = f"upstream Triton for ROCm {rocm}: {spec}"
        elif info.get("cuda"):
            cuda = info["cuda"].replace(".", "")
            spec = "pytorch-triton"
            extra = [
                "--index-url",
                f"https://download.pytorch.org/whl/nightly/cu{cuda}",
                "--index-strategy",
                "unsafe-best-match",
            ]
            # pytorch-triton, not plain `triton`: a dev torch expects the
            # PyTorch-built wheel matching ci_commit_pins/triton.txt, and a
            # release Triton from PyPI may not be compatible with it.
            label = f"upstream Triton for CUDA {info['cuda']}: {spec}"
        else:
            spec = f"triton=={args.version}" if args.version else "triton"
            label = f"upstream Triton from PyPI: {spec}"

    # Everything below the uninstall is unrecoverable if it fails, because the
    # providers collide on the `triton` package and all of them are removed at
    # once. So validate first, by whichever means fits the source.
    tmpdir = None
    if from_source and not args.editable:
        # Build the wheel before uninstalling. A source build is the slow,
        # failure-prone path (compiles Triton + LLVM) and currently the only way
        # to get a TLX-capable FBTriton, so it must not be the unguarded one.
        tmpdir = tempfile.mkdtemp(prefix="fbtriton-wheel-")
        print(f"--- building FBTriton wheel from {spec} (compiles Triton + LLVM, slow)")
        build = [uv_bin(), "build", "--wheel", "--out-dir", tmpdir, spec]
        if run(build).returncode != 0:
            print("error: build failed; leaving the current install alone")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return 1
        wheels = sorted(Path(tmpdir).glob("*.whl"))
        if not wheels:
            print(f"error: build produced no wheel in {tmpdir}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return 1
        spec = str(wheels[0])
        label = f"FBTriton wheel built from source: {wheels[0].name}"
    elif from_source:
        print(
            "warning: --editable cannot be pre-validated, so a failed build will "
            "leave this environment with no Triton. Drop --editable to build the "
            "wheel first."
        )
    else:
        print(f"--- checking {spec} resolves")
        if not resolves(spec, extra):
            return 1

    print(f"--- removing all triton distributions: {', '.join(TRITON_DISTRIBUTIONS)}")
    run(uv_pip("uninstall", *TRITON_DISTRIBUTIONS))

    print(f"--- installing {label}")
    res = run(uv_pip("install", "--reinstall", *extra, spec))
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if res.returncode != 0:
        print("error: install failed -- the environment now has NO Triton")
        return res.returncode

    print("--- clearing compile caches (stale kernels survive a provider swap)")
    clear_caches()

    print()
    return cmd_doctor(args)


def discover_tlx_tests() -> list[str]:
    found: list[str] = []
    for pattern in TLX_TEST_PATTERNS:
        parent, _, glob = pattern.rpartition("/")
        base = REPO_ROOT / parent
        if base.is_dir():
            found.extend(sorted(str(p.relative_to(REPO_ROOT)) for p in base.glob(glob)))
    return found


def cmd_test(args: argparse.Namespace) -> int:
    if not args.skip_doctor and cmd_doctor(args) != 0:
        print(
            "\nrefusing to run: TLX would not engage, so the results would be "
            "stock Inductor wearing a TLX label. Re-run with --skip-doctor to "
            "override."
        )
        return 1

    env = dict(os.environ)
    env["TORCHINDUCTOR_TLX_MODE"] = args.mode
    print(f"\n--- TORCHINDUCTOR_TLX_MODE={args.mode}")

    tests = discover_tlx_tests()
    if args.full:
        tests += FULL_TESTS
    if not tests:
        # Keep the default fast: this is a plumbing check, not a correctness
        # suite. --full adds the 425-test Inductor/Triton suite (~2.5 min).
        print("no torchTLX tests found; running the plumbing sanity check")
        print(f"  (looked for: {', '.join(TLX_TEST_PATTERNS)}; use --full for the suite)")
        return run(
            [sys.executable, str(Path(__file__).parent / "sanity.py")],
            cwd=REPO_ROOT,
            env=env,
        ).returncode

    cmd = [sys.executable, "-m", "pytest", *tests, "-q", "--no-header"]
    if args.k:
        cmd += ["-k", args.k]
    return run(cmd, cwd=REPO_ROOT, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TorchTLX bring-up: swap Triton provider and run TLX tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="report the environment and whether TLX can engage")

    p_switch = sub.add_parser("switch", help="switch the Triton provider")
    p_switch.add_argument("provider", choices=["fbtriton", "oai"])
    p_switch.add_argument("--version", help="pin an exact version")
    p_switch.add_argument(
        "--from-source", metavar="PATH", help="build FBTriton from a local checkout"
    )
    p_switch.add_argument(
        "--editable", action="store_true", help="with --from-source, install as editable"
    )

    p_test = sub.add_parser("test", help="run the torchTLX unit tests")
    p_test.add_argument("--mode", choices=["allow", "force"], default="allow")
    p_test.add_argument("-k", help="pytest -k filter")
    p_test.add_argument(
        "--skip-doctor", action="store_true", help="run even if TLX cannot engage"
    )
    p_test.add_argument(
        "--full",
        action="store_true",
        help=f"also run the full Inductor/Triton suite ({' '.join(FULL_TESTS)}, ~2.5 min)",
    )

    args = parser.parse_args()
    return {"doctor": cmd_doctor, "switch": cmd_switch, "test": cmd_test}[args.command](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
