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

Every platform runs the same ordered stages, bar one Linux-CUDA-only step.
A shell orchestrator drives them and threads environment forward through an
`--env-out` file that each Python stage writes and the orchestrator
`source`s before the next stage:

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
5. Linux CUDA only: native-AOT stage 2 (`tools/native_aot/build_stage2.py
   --wheel`), inline in `linux/build.sh`. Installs the raw wheel, exports the
   AOT kernels, relinks `libtorch_cuda`, and patches it back into the raw
   wheel so the repair below bundles the relinked library.
6. `repair_wheel.py RAW_DIR FINAL_DIR` -- bundle the external shared
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

- `install_numpy()` / `NUMPY_PINS` -- the single build-time numpy pin table
  (oldest wheel-bearing numpy *minor* per Python, at the newest vetted
  *patch*). Every Python in the CD matrix has an entry. A Python with no
  entry -- a newly added one, before its pin lands -- builds against a numpy
  the CI image pre-provisioned, and fails loudly when there is none, rather
  than silently picking a stale default.
- `retry()`, `pip_install()` -- retrying command / pip wrappers.
- `write_env_exports()`, `shell_quote()` -- serialize env for the
  `--env-out` hand-off (POSIX PATH conversion + bash-identifier filtering).
- `download()` -- streaming download with backoff.

## Linux wheel customization

`linux/repair_wheel.py` supports downstream customization through
`PYTORCH_RUNTIME_DEPENDENCY_POLICY`. The default value, `default`, preserves
the built-in dependency copying, accelerator runtime paths, and manylinux
platform tagging. It does not load a downstream hook. The `custom` policy
replaces dependency copying with a user-provided function and uses the runtime
paths and platform tag behavior described below. Other policy values are rejected.

### Dependency hook

Set these variables in the environment used by the Linux repair stage:

```bash
export PYTORCH_RUNTIME_DEPENDENCY_POLICY=custom
export PYTORCH_CUSTOM_WHEEL_DEPENDENCY_HOOK=/opt/downstream/copy_dependencies.py
```

In custom mode, the dependency hook is required. If it is unset or empty,
repair fails without falling back to the default copier.

The hook path points to a Python file defining `copy_runtime_dependencies`.
For example, a hook that bundles only a downstream copy of libgomp can contain:

```python
import shutil
from pathlib import Path


def copy_runtime_dependencies(
    torch_dir: Path,
    arch: str,
    use_cuda: bool,
    gpu_arch_type: str,
    gpu_arch_version: str,
) -> None:
    shutil.copy("/opt/downstream/lib/libgomp.so.1", torch_dir / "lib/libgomp.so.1")
```

The file is executed before repairing wheels. The function receives five
positional arguments and runs once per unpacked wheel; `torch_dir` points to
that wheel's `torch/` directory. The hook is checked to ensure it accepts these
five positional arguments. Parameter type annotations are not validated or
enforced at runtime.

The hook completely replaces the default copier. A hook that copies nothing
can simply return. The hook returns `None` on success (including an implicit
return) and raises an exception on failure. A non-`None` return value is an error.

The hook owns dependency-specific ELF rewrites and checks. After it returns,
repair applies RPATHs, handles platform tags, and repacks the wheel. If copying,
RPATH updates, or retagging fails, repair skips repacking the current wheel.

### Runtime paths

In `custom` mode, repair sets `$ORIGIN:$ORIGIN/lib` on `torch/*.so*` and
`$ORIGIN` on `torch/lib/*.so*`, using `patchelf --force-rpath` to write
`DT_RPATH`. Accelerator-specific package paths are not added.

Hooks copying libraries into nested directories must handle those libraries'
runtime paths themselves, for example, within `copy_runtime_dependencies()`.
Downstream deployments must provide compatible versions of any dependencies
not bundled by the hook and make them available to the dynamic loader.

### Platform tags

In `custom` mode, leaving `PYTORCH_CUSTOM_WHEEL_PLATFORM_TAG` unset preserves
the input wheel's platform tags. To replace them, set a single platform tag:

```bash
export PYTORCH_CUSTOM_WHEEL_PLATFORM_TAG=linux_x86_64
```

In `default` mode, this variable is ignored and repair uses the architecture's
built-in manylinux tag.

Tag validation checks syntax only; it does not verify ABI or manylinux
compatibility. Downstream builds are responsible for the compatibility claimed
by their chosen tag.

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
