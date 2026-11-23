# torchTLX bring-up

This fork drives Inductor with [FBTriton](https://github.com/facebookexperimental/triton)
instead of upstream Triton, because torchTLX lives in FBTriton.

**`dev.py` is the only interface you need.** It handles both things:
swapping the Triton provider, and running the checks against it.

```bash
python tools/torchtlx/dev.py switch fbtriton    # upstream Triton -> FBTriton
python tools/torchtlx/dev.py doctor             # can TLX engage here?
python tools/torchtlx/dev.py test --mode allow  # compile + numerics check
python tools/torchtlx/dev.py switch triton      # back to upstream, for A/B
```

There is no environment variable to set and no torch rebuild: Triton is a pure
runtime dependency of Inductor, so a swap is only a pip operation -- ~30s for a
published wheel. Getting TLX today means `--from-source`, which is slower: it
compiles Triton and LLVM, a few minutes. (`FBTRITON=1` exists further down, but
it is build-infrastructure only and does not apply to you as a user.)

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
fbtriton 3.7.x releases. Today a source build is the only one that gets you a
registry -- no published fbtriton wheel carries it yet:

```bash
# your own checkout (installs as the `triton` distribution)
git clone https://github.com/facebookexperimental/triton ~/fbtriton
python tools/torchtlx/dev.py switch fbtriton --from-source ~/fbtriton

# published wheel (installs as the `fbtriton` distribution) -- once one ships
python tools/torchtlx/dev.py switch fbtriton
```

`doctor` reports `TLX registry : OK` once a capable build is installed.

## Release wheels only: `FBTRITON=1`

Not for users -- `dev.py switch fbtriton` covers that and needs no
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
torchTLX never encounters it there. Installing the provider is `dev.py`'s job.

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
Any other value, including unset, leaves TLX off. `dev.py test --mode` sets it
for you.

## ROCm note

On ROCm, `switch triton` installs `triton-rocm` from the PyTorch nightly index
matching your ROCm version. That is the name this repo publishes -- see
`build_triton_wheel.py` and `RELEASE.md`; the older `pytorch-triton-rocm` is
frozen at 3.6.0 and would silently give you a Triton far behind your torch.

Every provider -- `fbtriton`, `triton`, `triton-rocm`, and the legacy
`pytorch-triton*` names -- installs a top-level `triton` package, so they
collide. `switch` therefore uninstalls all of them before installing the
requested one.
