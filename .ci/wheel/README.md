# CD wheel build pipelines

Python-driven pipelines that build the nightly / release PyTorch **wheels**
for the three CD platforms. Each platform has its own subdirectory; the
per-stage scripts are small and orchestration-free, and the cross-platform
helpers live in a single `_common.py`.

```
.ci/wheel/
  _common.py     shared helpers (see "Shared code")
  linux/         manylinux (x86_64, aarch64) -- patchelf repair
  mac/           macOS arm64 -- delocate repair
  windows/       Windows (amd64) -- no repair (DLLs resolved via PATH)
```

## Stage sequence

Every platform runs the same ordered stages. A shell orchestrator drives
them and threads environment forward through an `--env-out` file that each
Python stage writes and the orchestrator `source`s before the next stage:

1. `set_desired_python.sh` -- select/install the requested interpreter
   (`DESIRED_PYTHON`, e.g. `3.13` or the free-threaded `3.13t`) and expose it
   to the later stages.
2. `build_env_setup.py --env-out F` -- compute the build-flag / toolchain
   environment (USE_CUDA, USE_DISTRIBUTED, vcvars, CUDA/XPU, ...) and write
   `export` lines to `F`.
3. `build_install_deps.py` -- install build-time pip dependencies (the
   `numpy` pin, `requirements-build.txt`, `requirements.txt`) plus any
   platform bits (libomp on macOS, libuv on Windows), then run `spin clean`.
4. `build_wheel.py RAW_DIR` -- run
   `python -m build --wheel --no-isolation --outdir RAW_DIR`
   (scikit-build-core reads the repo-root `pyproject.toml`).
5. `repair_wheel.py RAW_DIR FINAL_DIR` -- bundle the external shared
   libraries into the wheel (patchelf on Linux, delocate on macOS). Windows
   has no repair stage.

## Per-platform orchestration

| Platform | Entry point | Topology | Repair |
|----------|-------------|----------|--------|
| Linux   | `linux/build_all.sh` -> `linux/build.sh` | one runner loops over `DESIRED_PYTHONS`; `libtorch_cpu` is reused across Pythons via `SKIP_SETUP_CLEAN` | patchelf |
| macOS   | `mac/build_all_macos_wheels.sh` -> `mac/build.sh` | one runner loops (`uv python install`), a venv per Python | delocate |
| Windows | `.ci/pytorch/binary_windows_build.sh` (**outside this tree**) | CI matrix -- one Python per job, no loop | none |

`build.sh` also routes on `GPU_ARCH_TYPE` (cpu / cuda / rocm / xpu).

## Shared code -- `_common.py`

Imported by every stage via
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`:

- `numpy_pin()` / `NUMPY_PINS` -- the single build-time numpy pin table
  (oldest wheel-bearing numpy *minor* per Python, at the newest vetted
  *patch*; an unsupported Python fails loudly rather than picking a stale
  fallback).
- `retry()`, `pip_install()` -- retrying command / pip wrappers.
- `write_env_exports()`, `shell_quote()` -- serialize env for the
  `--env-out` hand-off (POSIX PATH conversion + bash-identifier filtering).
- `download()` -- streaming download with backoff.

## Dependencies outside `.ci/wheel/`

- **CI wiring.** The workflows are generated, not hand-written. Edit
  `.github/templates/{linux,windows,macos}_binary_build_workflow.yml.j2`
  (and `.github/scripts/generate_binary_build_matrix.py`, which also
  references `linux/build_env_setup.py`), then regenerate with
  `python .github/scripts/generate_ci_workflows.py`. The generated
  `.github/workflows/generated-*-binary-*-nightly.yml` files are the callers.
- **Windows workspace `.ci/pytorch/windows/`.** The Windows pipeline is
  launched by `.ci/pytorch/binary_windows_build.sh` and treats
  `.ci/pytorch/windows/` as its runtime workspace (`WIN_CI_DIR`): the Python
  install, `tmp_bin/`, magma, libuv, and the shared `internal/*.bat`
  installers (`cuda_install`, `xpu_install`, `vc_install_helper`,
  `install_python`, ...) live there, as does the separate `arm64/` build
  path. These stay outside this tree because non-CD Windows CI shares them.
- **Repo root.** `pyproject.toml` (the actual scikit-build-core build),
  `requirements-build.txt` + `requirements.txt` (build-time deps),
  `tools/amd_build/build_amd.py` (Linux ROCm hipify), and `spin` /
  `tools/clean.py` (the `spin clean` step).
- **libtorch.** There is no separate libtorch build: the nightly libtorch
  zips are extracted from the wheels built here by
  `.ci/libtorch/extract_libtorch_from_wheel.py` in a follow-on workflow job.
  `PACKAGE_TYPE` (`manywheel` / `wheel` / `libtorch`) is a semantic label
  consumed by the env-population, test, and upload scripts -- it is never
  resolved as a directory.
