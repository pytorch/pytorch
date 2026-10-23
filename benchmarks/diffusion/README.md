# Diffusion compile benchmark

`compile_benchmark.py` is the entry point for diffusion compile benchmarks. It
times either complete pipeline requests or isolated denoiser forwards. Each
requested mode runs in a fresh process with a private Inductor cache. A run
writes a CSV summary, JSON-lines records for the PyTorch OSS benchmark database,
and one log per worker.

Dashboard backends include compilation mode and CUDA graph configuration, for
example `inductor_regional_no_cudagraphs`; the eager baseline uses `eager`.
Dashboard model names include the workload, for example
`sd35_medium:denoise_step`, so pipeline and denoiser measurements never share a
series. Scenario ids have the form `model:workload:mode:backend:cudagraphs`.
Workload parameters overridden on the command line are appended to the workload
in both, sorted by workload parameter name, for example
`sd35_medium:denoise_step[height=512,num_images_per_prompt=2]` for
`--height 512 --batch-size 2`.

`--output` names the CSV file; JSON records use the same path with a `.json`
suffix. Output paths with a `.json` suffix (case-insensitive) are rejected before
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
single `--mode` invocation remains supported. `--cudagraphs` is independent of
the compilation granularity and requires Inductor, CUDA, and at least one
compiled mode. Eager scenarios keep CUDA graphs disabled, allowing
`--mode all --cudagraphs --check-outputs` to compare compiled execution against
an eager reference on the same CUDA device.

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
`sd35_medium` and `tiny_sd3`. `--guidance-scale 1` disables CFG. Values equal to
the recipe default are not treated as overrides.

Both denoiser recipes require `diffusers` and `transformers`, including
`tiny_sd3`: the timestep helpers come from the SD3 pipeline module, which imports
`transformers`, even though text encoders are not loaded. Their integration
tests skip when either package is unavailable.

## Measurement boundaries

- Checkpoint downloads happen before model setup timing. Recipe-specific filters
  exclude unused weight variants and replacement components; AuraFlow loads its
  transformer configuration from the same pinned snapshot.
- `model_setup_s` covers construction and device placement, and for
  `denoise_step` also input and timestep preparation.
- `compile_wrapper_setup_s` covers installing the requested compile wrappers;
  compilation itself remains lazy.
- `first_request_s` includes compilation for compiled modes. It is not labeled
  compilation time.
- Compiler-reported phases are emitted individually. They overlap and must not
  be summed.
- Steady-state samples follow warmup and are retained individually. The summary
  reports their median and median absolute deviation. Warmup request durations,
  which may include CUDA graph recording, are not reported, so these measurements
  cannot reconstruct total time to steady state.
- Accelerator synchronization brackets every wall-time sample.
- CUDA allocated and reserved peaks are reported separately for setup and
  requests. Request peaks are absolute process peaks and include resident model
  state. Device memory is unavailable for CPU runs.
- Denoiser predictions are checked against a higher-precision forward of the
  final measured request, which the eager worker computes after timing in
  float32 (float64 for float32 runs) when `--check-outputs` is set. A compiled
  prediction passes if its relative RMS error against that reference is at most
  a multiple of eager's, plus a dtype floor. The multiplier follows
  `torch._dynamo.utils.same`: 3 for float16 and bfloat16 or for predictions with
  fewer than 1000 elements, otherwise 2. Unlike `same`, the error is relative
  and half-precision runs use a float32 rather than float64 reference. Compiled
  and eager prediction dtypes and shapes must match and values must be finite.
  A fixed bound on the compiled-to-eager difference does not work for deep
  low-precision models: on a 24-layer SD3.5-structured transformer with random
  weights in bfloat16 under CPU Inductor, compiled and eager differed by 1.8%
  relative RMS while compiled was closer to float32 than eager (0.9% vs 1.4%).
  If the higher-precision forward raises, for example by running out of device
  memory, the eager result keeps its timings, records `reference_error`, and
  reports `output_check="reference failed"`; compiled results are marked failed
  with `output_check="reference unavailable"`. A worker that is killed or times
  out during the reference forward loses its timings as usual.
- Pipeline output checks happen after timing and use `torch.testing.assert_close`
  defaults against the eager result. Tolerances follow the extracted output
  dtype: PIL images require exact pixel equality, and NumPy outputs use their
  array dtype, even when the model computes in lower precision. These strict
  checks can flag small compilation differences in postprocessed outputs, and in
  low-precision latents after many denoising steps, such as the 40-step
  `sd35_medium` bfloat16 latents, where rounding differences compound. Use
  `--workload denoise_step` for a check that tolerates rounding error.
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
require at least one compiled graph for compiled modes. They report nested-region
activity but deliberately do not infer reuse from a shared subgraph identifier.
When CUDA graphs are requested, the worker additionally requires a recorded
CUDA graph. `cudagraph_capture_count` counts recorded graphs retained across all
tree nodes, excluding kernel-free nodes.
