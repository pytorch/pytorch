# Diffusion compile benchmark

`compile_benchmark.py` is the entry point for end-to-end diffusion pipeline
benchmarks. Each requested mode runs in a fresh process with a private Inductor
cache directory, and with the local and remote FX graph, AOT autograd, autotune,
and remote Dynamo PGO caches disabled; the worker records the cache settings in
effect. Workers otherwise inherit the caller's environment, so settings such as
`TORCHINDUCTOR_MAX_AUTOTUNE` apply; the worker records `PYTORCH_*`, `TORCH_*`,
`TORCHDYNAMO_*`, `TORCHINDUCTOR_*`, `TRITON_*`, and `CUDA_VISIBLE_DEVICES`.
`TORCHINDUCTOR_FORCE_DISABLE_CACHES` is not used because it gives every
graph a fresh cache directory, so kernels shared between graphs would be
recompiled, unlike in a real cold start. A run writes a CSV summary, JSON-lines
records for the PyTorch OSS benchmark database, and one log per worker.

Dashboard backends include compilation mode and CUDA graph configuration, for
example `inductor_regional_no_cudagraphs`; the eager baseline uses `eager`. The
CSV `backend` column uses the same names. Dashboard records carry only
identifying fields (device, mode, CUDA graphs, output boundary, checkpoint
revision, and torch version) and, for failures, only the status; the CSV
`provenance` column and the worker logs keep full provenance and error details.

`--output` names the CSV file, `diffusion_benchmark.csv` in the current
directory by default; JSON records use the same path with a `.json` suffix, and
worker logs go to a sibling `<stem>_logs` directory, whose old logs are removed at
the start of a run. Output paths with a `.json`
suffix (case-insensitive), and paths that cannot be written, are rejected before
workers start.

```bash
python benchmarks/diffusion/compile_benchmark.py \
  --model tiny_stable_diffusion --mode all --check-outputs \
  --output results.csv
```

Pass `--mode` more than once to run a subset. The original single `--model` and
single `--mode` invocation is still accepted, but `--mode full` no longer enables
CUDA graphs implicitly; use `--mode full --cudagraphs` for the former
`mode="reduce-overhead"` configuration. Each scenario runs one first request,
`--warmups` warmup requests, and `--repetitions` measured requests, seven
requests by default. `--cudagraphs` is independent of the compilation
granularity and requires `--backend inductor`, a CUDA device, at least one
warmup request, and at least one compiled mode; these are checked before any
worker starts. Eager scenarios keep CUDA graphs disabled, allowing
`--mode all --cudagraphs --check-outputs` to compare compiled execution against
an eager reference on the same CUDA device. `--timeout` applies to each
scenario and includes checkpoint downloads, which can be long on a cold Hugging
Face cache. Interrupting or terminating the runner (SIGINT, SIGTERM, or SIGHUP)
kills the running worker.

## Measurement boundaries

- Checkpoint downloads and device runtime startup happen before model setup
  timing. Recipe-specific filters exclude unused weight variants and replacement
  components; AuraFlow loads its transformer configuration from the same pinned
  snapshot.
- `model_setup_s` covers construction and device placement.
- `compile_wrapper_setup_s` covers installing the requested compile wrappers;
  compilation itself remains lazy.
- `compile_time_s` is the total Dynamo and Inductor compile wall time,
  `torch._dynamo.utils.calculate_time_spent()["total_wall_time"]`, which
  excludes runtime autotuning and CUDA graph recording. It is zero for eager.
- `first_request_s` includes compilation for compiled modes. It is not labeled
  compilation time.
- Per-phase compiler timings are kept in the CSV `compiler_times_s` column.
  They overlap and must not be summed, and their names are internal function
  names, so the dashboard receives only `compile_time_s`.
- Steady-state samples follow warmup and are retained individually. The summary
  reports their median and median absolute deviation. Warmup request durations,
  which may include CUDA graph recording, are not reported, so these measurements
  cannot reconstruct total time to steady state.
- Accelerator synchronization brackets every wall-time sample.
- Allocated and reserved peaks of the accelerator's caching allocator are
  reported separately for setup and requests. Request peaks are absolute
  process peaks measured from the end of warmup through the measured requests,
  so they exclude compilation, autotuning, and CUDA graph recording transients
  but include resident model state and CUDA graph memory pools. Device memory
  is unavailable on CPU.
- Output checks happen after timing and compare against the eager result. A
  compiled output fails if its shape or dtype differs from eager or, for
  floating-point outputs such as latents, if it has non-finite values;
  postprocessed images are uint8 and cannot hold them. Otherwise its relative
  RMS difference from eager is reported in `output_check` without a bound:
  rounding differences compound over many low-precision denoising steps, and
  postprocessed images and videos are quantized, so no fixed tolerance
  separates correct compilations from incorrect ones. An eager output that is
  non-finite or all zeros (a NaN latent decodes to a black image) marks the
  eager scenario failed.
  If the eager reference is unavailable, compiled results are marked failed with
  `output_check="reference unavailable"`; their CSV timings are retained, but
  only failure status is exported to the dashboard.

Every request gets a newly seeded generator and a scheduler reconstructed from
its original config. With CUDA graphs, each complete request starts one graph
iteration before timing, preserving denoiser outputs across scheduler steps.
The pilot emits latents, so VAE decoding is excluded. A recipe marked
`postprocessed_output` includes VAE decode and postprocessing even though only
the named transformer component is compiled.

## Recipe audit

| Recipe | Checkpoint | Output | Validation and known limitations |
| --- | --- | --- | --- |
| `tiny_stable_diffusion` | `diffusers/tiny-stable-diffusion-torch@c05d78754bd90a3ffea8fd3a0823ac17329cd1a6` | latent | Validated CPU pilot. The repository is ungated and only publishes pickle-format weights; the exact revision is pinned. It is a functional integration workload, not a representative performance model. |
| `auraflow` | `fal/AuraFlow-v0.3@2cd8588f04c886002be4571697d84654a50e3af3` plus `city96/AuraFlow-v0.3-gguf@f747743ef0dcc57229879fe54340dd20e907c456` | postprocessed image | Not executed here; requires CUDA and the optional GGUF dependency. Decode and postprocessing are timed, but the compiled target is only `pipeline.transformer`. |
| `wan` | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers@b184e23a8a16b20f108f727c902e769e873ffc73` | postprocessed video | Not executed here; the 14B pipeline requires suitable CUDA memory. The input image is pinned separately. Decode and postprocessing are timed, but the compiled target is only `pipeline.transformer`. |
| `ltx` | `Lightricks/LTX-Video-0.9.7-dev@2101082f5eb5540770a2df43747feadb6f69b889` | latent | Not executed here; requires CUDA. VAE decode is excluded. |
| `flux` | `black-forest-labs/FLUX.1-dev@3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` | postprocessed image | Not executed here; gated checkpoint access and suitable CUDA memory are required. Decode and postprocessing are timed, but the compiled target is only `pipeline.transformer`. |

The former script timed one one-step warmup request followed by one 50-step
request, so its output was neither a compile-time measurement nor repeated
steady-state latency. It also left random inputs uncontrolled, applied different
compiler settings by strategy, imported AuraFlow dependencies for every model,
mutated repeated-block classes permanently, and produced no machine-readable
provenance or failure records. Historical numbers from that script should not be
compared with this runner's output.

Regional mode prefers the selected Diffusers version's
`compile_repeated_blocks` entry point and verifies every configured class is
present. Hierarchical mode uses `torch.compiler.nested_compile_region`, marks a
class at most once, and restores class methods after the scenario. Diagnostics
require at least one compiled graph for compiled modes, and for hierarchical
mode a traced nested region (an `invoke_subgraph_trace` compiler phase); they
deliberately do not infer reuse from a shared subgraph identifier.
When CUDA graphs are requested, the worker additionally requires a recorded
CUDA graph and fails if Inductor skipped CUDA graphs for any graph, at compile
time or at runtime. `cudagraph_capture_count` counts recorded graphs retained
across all tree nodes, excluding kernel-free nodes; a compiled function called
several times per request records one node per call.
