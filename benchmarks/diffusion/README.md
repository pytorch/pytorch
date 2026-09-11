# Diffusion compile benchmark

`compile_benchmark.py` is the entry point for end-to-end diffusion pipeline
benchmarks. Each requested mode runs in a fresh process with a private Inductor
cache. A run writes a CSV summary, JSON-lines records for the PyTorch OSS
benchmark database, and one log per worker.

`--output` names the CSV file; JSON records use the same path with a `.json`
suffix. Output paths with a `.json` suffix (case-insensitive) are rejected before
workers start.

```bash
python benchmarks/diffusion/compile_benchmark.py \
  --model tiny_stable_diffusion --mode all --check-outputs \
  --output results.csv
```

Pass `--mode` more than once to run a subset. The original single `--model` and
single `--mode` invocation remains supported. `--cudagraphs` is independent of
the compilation granularity and requires Inductor, CUDA, and at least one
compiled mode. Eager scenarios keep CUDA graphs disabled, allowing
`--mode all --cudagraphs --check-outputs` to compare compiled execution against
an eager reference on the same CUDA device.

## Measurement boundaries

- Checkpoint downloads happen before model setup timing. Recipe-specific filters
  exclude unused weight variants and replacement components; AuraFlow loads its
  transformer configuration from the same pinned snapshot.
- `model_setup_s` covers construction and device placement.
- `compile_wrapper_setup_s` covers installing the requested compile wrappers;
  compilation itself remains lazy.
- `first_request_s` includes compilation for compiled modes. It is not labeled
  compilation time.
- Compiler-reported phases are emitted individually. They overlap and must not
  be summed.
- Steady-state samples follow warmup and are retained individually. The summary
  reports their median and median absolute deviation.
- Accelerator synchronization brackets every wall-time sample.
- CUDA allocated and reserved peaks are reported separately for setup and
  requests. Request peaks are absolute process peaks and include resident model
  state. Device memory is unavailable for CPU runs.
- Output checks happen after timing and use `torch.testing.assert_close`
  defaults against the eager result.

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
| `auraflow` (`auroflow` alias) | `fal/AuraFlow-v0.3@2cd8588f04c886002be4571697d84654a50e3af3` plus `city96/AuraFlow-v0.3-gguf@f747743ef0dcc57229879fe54340dd20e907c456` | postprocessed image | Not executed here; requires CUDA and the optional GGUF dependency. Decode and postprocessing are timed, but the compiled target is only `pipeline.transformer`. |
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
require at least one compiled graph for compiled modes. They report nested-region
activity but deliberately do not infer reuse from a shared subgraph identifier.
When CUDA graphs are requested, the worker additionally requires a recorded
CUDA graph. `cudagraph_capture_count` counts recorded graphs retained across all
tree nodes, excluding kernel-free nodes.
