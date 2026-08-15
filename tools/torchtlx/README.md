# torchTLX bring-up

This fork drives Inductor with [FBTriton](https://github.com/facebookexperimental/triton)
instead of upstream Triton, because torchTLX lives in FBTriton.

**`bringup.py` is the only interface you need.** It handles both things:
swapping the Triton provider, and running the torchTLX tests against it.

```bash
python tools/torchtlx/bringup.py switch fbtriton # upstream Triton -> FBTriton
python tools/torchtlx/bringup.py doctor          # can TLX engage here?
python tools/torchtlx/bringup.py test            # run the torchTLX tests
python tools/torchtlx/bringup.py switch oai      # back to upstream, for A/B
python tools/torchtlx/bringup.py test --full     # ... plus the full Inductor/Triton suite
```

There is no environment variable to set and nothing to rebuild: Triton is a
pure runtime dependency of Inductor, so swapping to a published wheel is a
~30s pip operation. `--from-source` is much slower -- it compiles Triton and
LLVM, a few minutes. (`FBTRITON=1` exists further down, but it is
build-infrastructure only and does not apply to you as a user.)

## What `test` runs

torchTLX tests (`test/inductor/test_torchtlx*.py`). Today that is
`test_torchtlx_templates.py`, which asserts the contract between this repo and
FBTriton: every TLX template Inductor proposes must have a heuristic
registered for the running device. Those tests skip when the active Triton has
no TLX registry, so they are inert on upstream Triton.

With no torchTLX tests present it falls back to `sanity.py`, a deliberately
small plumbing check -- eager matmul
plus compiled pointwise, reduction and backward, asserting Inductor emitted a
Triton kernel and the numerics match. It finishes in under 10s; it is not a
correctness suite. `--full` adds `test/inductor/test_triton_kernels.py`
(425 tests, ~2.5 min), which is what to run when validating a provider swap.

## Tests for the wiring itself

`tools/test/test_torchtlx_wiring.py` asserts the claim this fork rests on --
that FBTriton is opt-in and the default build path is unchanged -- by
evaluating the Triton block of `.ci/pytorch/binary_populate_env.sh` and
comparing the emitted requirement strings. It also guards that nothing lands
under `.ci/docker/`, whose tree hash gates every CI Docker image rebuild.

It lives in `tools/test/` rather than here so that CI collects it: the lint
workflow runs `pytest tools/test`. That job uses the linter image, so the test
imports neither torch nor triton and needs no GPU and no network.

```bash
python tools/test/test_torchtlx_wiring.py
```

## Compile caches

`switch` always clears the Inductor and Triton compile caches. This is not
cosmetic: cached kernels are not keyed by which Triton built them, so running
the test suite immediately after a swap against a warm cache produced 311
spurious failures that disappeared on a second run. There is no reason to skip
it, so there is no flag for it. The LLVM download cache (`~/.triton/llvm`) is
left alone -- it holds the toolchain, not compiled kernels.

## Why `doctor` exists

`torch/_inductor/heuristics/template/tlx.py` enables torchTLX by importing
`triton.language.extra.tlx.inductor.registry`, and that import is wrapped in
`except ImportError: pass`. With a Triton that lacks the registry, nothing
fails -- TLX simply never engages, and a green test run is really measuring
stock Inductor. `doctor` turns that silent no-op into an explicit failure, and
`test` refuses to run until it passes (override with `--skip-doctor`).

## FBTriton >= 3.8 is required

torchTLX needs `triton/language/extra/tlx/inductor/`, which is absent from the
fbtriton 3.7.x releases. Either source works:

```bash
# published wheel (installs as the `fbtriton` distribution)
python tools/torchtlx/bringup.py switch fbtriton

# your own checkout (installs as the `triton` distribution)
git clone https://github.com/facebookexperimental/triton ~/fbtriton
python tools/torchtlx/bringup.py switch fbtriton --from-source ~/fbtriton
```

`doctor` reports `TLX registry : OK` once a capable build is installed.

## Release wheels only: `FBTRITON=1`

Not for users -- `bringup.py switch fbtriton` covers that and needs no
environment variable. This applies to exactly one thing: what a **published
torch wheel declares as its Triton dependency**.

`.ci/pytorch/binary_populate_env.sh` defaults to upstream, unchanged:
`triton~=<ver>` for cuda, `triton-rocm~=<ver>` for rocm. With `FBTRITON=1` it
emits `fbtriton~=<ver>` for both -- one FBTriton wheel carries the nvidia and
amd backends, so they share a name. XPU is unaffected.

No workflow sets `FBTRITON`, so this is **manual invocation only** today --
wiring it into a release job is a separate change. `TRITON_VERSION` is
deliberately shared with upstream via `.ci/docker/triton_version.txt`, because
FBTriton mirrors upstream's `X.Y.Z` (its `release/3.8.x` reports `3.8.0+fb`);
a separate version file would live under `.ci/docker/` and invalidate every CI
Docker image hash for a value that is identical anyway.

There is **no FBTriton commit pin**, and nothing to keep up to date. Upstream
dev builds request `<ver>+git<shorthash>` because CI builds that wheel itself
from `ci_commit_pins/triton.txt`, so the pin and the wheel version agree by
construction. This fork does not build FBTriton -- FBTriton publishes its own
wheels, and has never used a `+git` local version (its dev releases are
`<ver>.devYYYYMMDD`) -- so requesting one would be unsatisfiable. FBTriton
therefore uses the plain `~=<ver>` form for dev builds too, and
`install_triton.sh` / `build_triton_wheel.py` / `nightly.yml` are left exactly
as upstream.

### Why the provider is not in `requirements.txt`

`requirements.txt` is left untouched, so a contributor who has no interest in
torchTLX never encounters it there. Installing the provider is `bringup.py`'s
job.

Two things would make a requirement line awkward anyway. No published fbtriton
wheel carries torchTLX yet -- that needs >=3.8, and the TLX Inductor registry
landed after `release/3.7.x` was cut. And it could not cover everyone even
then: a contributor who builds FBTriton from a checkout gets the `triton`
distribution rather than `fbtriton`, because `setup.py` defaults
`TRITON_WHEEL_NAME` to `triton`, and one PEP 508 requirement cannot name both.
That is why `doctor` identifies the provider by the `+fb` suffix on
`triton.__version__` rather than by distribution name -- that signal holds for
both paths.

## Enabling TLX

`TORCHINDUCTOR_TLX_MODE` (equivalently `torch._inductor.config.triton.tlx_mode`)
takes `allow` (TLX competes via autotuning) or `force` (TLX templates only).
Any other value, including unset, leaves TLX off. `bringup.py test --mode`
sets it for you.

## ROCm note

On ROCm, `switch oai` installs `pytorch-triton-rocm` from the PyTorch nightly
index matching your ROCm version. `fbtriton`, `triton`, and
`pytorch-triton-rocm` all install a top-level `triton` package and collide, so
`switch` uninstalls every known provider before installing the requested one.
