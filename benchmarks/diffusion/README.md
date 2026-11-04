# Diffusion compile benchmark

`compile_benchmark.py` is the entry point for diffusion compile benchmarks. It
times either complete pipeline requests or isolated denoiser forwards. Each
requested mode runs in a fresh process with a private Inductor cache directory,
and with the local and remote FX graph, AOT autograd, autotune, and remote Dynamo
PGO caches disabled; the worker records the cache settings in effect. Workers
otherwise inherit the caller's environment, so settings such as
`TORCHINDUCTOR_MAX_AUTOTUNE` apply; the worker records `PYTORCH_*`, `TORCH_*`,
`TORCHDYNAMO_*`, `TORCHINDUCTOR_*`, `TRITON_*`, and `CUDA_VISIBLE_DEVICES`.
`TORCHINDUCTOR_FORCE_DISABLE_CACHES` is not used because it gives every graph a
fresh cache directory, so kernels shared between graphs would be recompiled,
unlike in a real cold start. A run writes a CSV summary, JSON-lines records for
the PyTorch OSS benchmark database, and one log per worker.

Dashboard backends include compilation mode and CUDA graph configuration, for
example `inductor_regional_no_cudagraphs`; the eager baseline uses `eager`. The
CSV `backend` column uses the same names. Dashboard model names include the
workload, for example `sd35_medium:denoise_step`, so pipeline and denoiser
measurements never share a series. Scenario ids have the form
`model:workload:mode:backend:cudagraphs`. Workload parameters overridden on the
command line are appended to the workload in both, sorted by workload parameter
name, for example `sd35_medium:denoise_step[height=512,num_images_per_prompt=2]`
for `--height 512 --batch-size 2`. Dashboard records carry only identifying
fields (device, mode, CUDA graphs, output boundary, checkpoint revision, and
torch version) and, for failures, only the status; the CSV `provenance` column
and the worker logs keep full provenance and error details.

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
python benchmarks/diffusion/compile_benchmark.py \
  --model sd35_medium --workload denoise_step --mode all --cudagraphs \
  --check-outputs --output sd35_denoise.csv
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
an eager reference on the same CUDA device. `--check-outputs` requires the eager
mode and at least one compiled mode. `--timeout` applies to each
scenario and includes checkpoint downloads, which can be long on a cold Hugging
Face cache. Interrupting or terminating the runner (SIGINT, SIGTERM, or SIGHUP)
kills the running worker.

## Workloads

`--workload` selects what one request is; `--mode` selects only the compilation
strategy. Both workloads support every mode.

- `pipeline` (default) times one complete pipeline call. For `sd35_medium` this
  includes CLIP and T5 text encoding, latent preparation, every scheduler step,
  and classifier-free guidance (CFG) combination; the pipeline returns latents,
  so VAE decoding is excluded. Only `pipeline.transformer` is compiled.
- `denoise_step` times one transformer forward and nothing else. Of the model
  weights, only the transformer's are loaded. Latents, conditioning, and pooled
  projections are drawn once from a seeded CPU generator, so values do not
  depend on the device, and timesteps come from the recipe's scheduler. This
  preparation runs during model setup, before timing. Text encoding, scheduler
  updates, CFG combination, and VAE decoding are excluded. Each request uses the
  next timestep of the schedule, so timestep values change while shapes and
  strides stay fixed; steady-state compilation is rejected as for pipelines.

The denoiser inputs follow `StableDiffusion3Pipeline`: latents have
`transformer.config.in_channels` channels at `1 / vae_scale_factor` of the
image size, conditioning has `tokenizer.model_max_length + max_sequence_length`
tokens (CLIP then T5) of width `joint_attention_dim`, pooled projections have
width `pooled_projection_dim`, and timesteps are float32 tensors expanded to the
transformer batch. With `guidance_scale > 1` the transformer batch packs
unconditional and conditional rows: latents are repeated and the batch doubles.
Results record the requested batch size and this effective batch size. The
conditioning values are seeded standard-normal tensors, not text-encoder
outputs, and the unconditional rows are not encodings of an empty prompt; they
match shapes and dtypes only.

`--height`, `--width`, `--batch-size` (images per prompt), and
`--guidance-scale` override the corresponding workload parameters for
`sd35_medium` and `tiny_sd3`. Height and width must be multiples of 16 (the VAE
scale factor times the patch size), which is checked before any worker starts.
`--guidance-scale 1` disables CFG. Values equal to the recipe default are not
treated as overrides. For `denoise_step`, a forward only depends on whether CFG
is enabled, so every scale above 1 is treated as the recipe default and every
scale of at most 1 as 1.

`sd35_medium` requires `diffusers` and `transformers`; `tiny_sd3` requires only
`diffusers`. Timesteps come from the recipe scheduler's `set_timesteps`, as in
`StableDiffusion3Pipeline`; schedulers with dynamic shifting, which SD3.5
Medium does not use, are rejected. The integration tests skip unless both
packages are installed. CI images do not install `diffusers`, so SD3 and
denoiser coverage comes from local runs only; CI exercises the runner,
compilation modes, and CUDA graphs through the dependency-free `toy` recipe.

## Measurement boundaries

- Checkpoint downloads and device runtime startup happen before model setup
  timing. Recipe-specific filters exclude unused weight variants and replacement
  components; AuraFlow loads its transformer configuration from the same pinned
  snapshot.
- `model_setup_s` covers construction and device placement, and for
  `denoise_step` also input and timestep preparation.
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
- Denoiser predictions are checked against a higher-precision forward of the
  final measured request, which the eager worker computes after timing in
  float32 (float64 for float32 runs) with TF32 disabled when `--check-outputs`
  is set. Devices without float64 support, such as MPS, compute the float64
  reference on the CPU. Diffusers computes some operations in float32
  regardless of the module dtype, including the timestep embedding and the
  RMSNorm statistics used by SD3.5's QK normalization, so a float64 reference
  is only float32-accurate there. A compiled prediction passes if its relative
  RMS error against that reference is at most a multiple of eager's, plus a
  floor of a tenth of `torch.testing`'s default rtol for the dtype. The
  multiplier matches the RMSE check in `torch._dynamo.utils.same`: 3 for float16
  and bfloat16 or for predictions with fewer than 1000 elements, otherwise 2.
  Unlike `same`, the error and floor are relative to the reference norm, and
  half-precision runs use a float32 rather than float64 reference. Compiled,
  eager, and reference shapes must match, compiled and eager dtypes must
  match, the compiled prediction must be finite, and eager and compiled
  scenarios must have run the same number of requests, so that both end on the
  same timestep.
  A fixed bound on the compiled-to-eager difference does not work for deep
  low-precision models: on a 24-layer SD3.5-structured transformer with random
  weights in bfloat16 under CPU Inductor, compiled and eager differed by 1.8%
  relative RMS while compiled was closer to float32 than eager (0.9% vs 1.4%).
  If the higher-precision forward raises, for example by running out of device
  memory, or returns non-finite values, the eager result keeps its timings,
  records `reference_error`, and reports `output_check="reference failed"`;
  compiled results are marked failed with `output_check="reference
  unavailable"`. A worker that is killed or times out during the reference
  forward loses its timings as usual.
- Pipeline output checks happen after timing and compare against the eager
  result. A compiled output fails if its shape or dtype differs from eager or,
  for floating-point outputs such as latents, if it has non-finite values;
  postprocessed images are uint8 and cannot hold them. Otherwise its relative
  RMS difference from eager is reported in `output_check` without a bound:
  rounding differences compound over many low-precision denoising steps, such
  as the 40-step `sd35_medium` bfloat16 latents, and postprocessed images and
  videos are quantized, so no fixed tolerance separates correct compilations
  from incorrect ones. Use `--workload denoise_step` for a bounded check.
- For both workloads, an eager output that is non-finite or all zeros (a NaN
  latent decodes to a black image) marks the eager scenario failed.
  If the eager reference is unavailable, compiled results are marked failed with
  `output_check="reference unavailable"`; their CSV timings are retained, but
  only failure status is exported to the dashboard.

Every pipeline request gets a newly seeded generator and a scheduler
reconstructed from its original config. With CUDA graphs, each complete request
starts one graph iteration before timing, preserving denoiser outputs across
scheduler steps. For `denoise_step`, each forward is one graph iteration. In
`full` and `hierarchical` modes, its prepared inputs live for the whole scenario
and are copied into the graph's static buffers inside the timed forward, and each
prediction is released before the next iteration overwrites it. In `regional`
mode, the inputs pass through the eager embedding layers, each repeated block
replays its own graph, and the eager output projection returns the prediction.
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
| `sd35_medium` (`pipeline`) | `stabilityai/stable-diffusion-3.5-medium@b940f670f0eda2d07fbb75229e779da1ad11eb80` | latent | Not executed here. The checkpoint is gated: accept the Stability AI Community License on Hugging Face and authenticate with `HF_TOKEN` or `hf auth login`, otherwise workers fail with an explicit gated-access error. CUDA/bfloat16, 1024x1024, one image per prompt, 40 steps, guidance 4.5, 256 T5 tokens; the transformer batch is 2 with CFG. Loads the transformer, VAE, and all three text encoders (non-fp16-variant weights); the single-file and ComfyUI text-encoder files are not downloaded. Regional and hierarchical modes target `JointTransformerBlock`, the only repeated block class in `SD3Transformer2DModel`; it covers the dual-attention, plain, and final `context_pre_only` blocks, which compile to different graphs. |
| `sd35_medium` (`denoise_step`) | same | denoiser prediction | Not executed here; same access requirement. Downloads only the transformer, scheduler config, CLIP tokenizer, and VAE config. Defaults match the pipeline recipe. |
| `tiny_sd3` (`denoise_step`) | built in | denoiser prediction | Download-free CPU test workload: a randomly initialized three-layer `SD3Transformer2DModel` with RMS QK norm that covers SD3.5's dual-attention, plain, and final `context_pre_only` blocks. Not a performance model. |

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
