# Adversarial MPS softmax and log-softmax benchmarks

Measured on 2026-11-16, Apple M4, 24 GiB RAM, macOS 15.6.1.

**The 5% per-shape latency limit does not hold across the expanded shapes.**
Three FP32 softmax cases regress by 6.70-9.48% in repeated public-operator
measurements, with independent paired GPU measurements supporting the result.
A CLIP-shaped FP32 log-softmax case regresses by 9.24% in screening and 12.00%
in paired GPU measurements. Several larger apparent regressions did not reproduce.

**No allocator-memory regression was found in 474 comparisons.** Static
threadgroup memory does increase: 256 to 384 bytes for each contiguous split
stage, and 4096 to 8192 bytes for the generic kernel. This is on-chip memory
per threadgroup, not a doubling of tensor or model RAM.

Baseline: local `main` at `760f1c8da40f395aca97825bc2e387d7c5afaab1`.
Candidate: `perf/mps-softmax-parity` at
`f3a56e96a08d1724cae76a6e62a82f88c600712b`. This compares the complete local
branches, not just the last commit, and does not compare with current upstream
main. No kernels were changed during this investigation. The candidate branch
and installed build were restored afterward.

## Counterexamples that reproduced

All rows below are FP32 and reduce the last dimension. Times are microseconds
per public operator call with a preallocated output. Positive changes are
slower. **C3** pools three longer confirmation runs; **S2** pools the two
screening runs. Each version uses the same protocol within a row.

| Model workload | Operator | Shape | Protocol | Main us | Candidate us | Change | Paired GPU change [95% interval] |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BERT attention | softmax | `(1, 12, 128, 128)` | C3 | 8.286 | 8.841 | **+6.70%** | +7.94% [+7.69%, +8.16%] |
| Qwen3 MoE routing | softmax | `(2048, 128)` | C3 | 10.442 | 11.431 | **+9.48%** | +9.55% [+9.26%, +9.88%] |
| Qwen2.5 decode attention | softmax | `(1, 28, 1, 2049)` | C3 | 5.803 | 6.235 | **+7.46%** | +12.19% [+10.53%, +15.16%] |
| CLIP contrastive loss | log_softmax | `(128, 128)` | S2 | 2.514 | 2.746 | **+9.24%** | +12.00% [+10.78%, +13.12%] |

The individual public-run medians make the repeatability visible:

| Workload | Main run medians us | Candidate run medians us |
| --- | --- | --- |
| BERT attention | 8.275, 8.316, 8.229 | 8.796, 8.860, 8.872 |
| Qwen3 MoE routing | 10.440, 10.524, 10.391 | 11.348, 11.462, 11.454 |
| Qwen2.5 decode attention | 5.546, 5.854, 5.830 | 6.130, 6.235, 6.404 |
| CLIP contrastive loss | 2.492, 2.532 | 2.719, 2.774 |

The Qwen2.5 `(1, 28, 1, 2049)` case is a concrete joint regression in
softmax latency and **static shader memory**: it dispatches the contiguous
split kernels, whose shared-memory allocation rises by 50%. Its tensor
allocation and allocator peaks match main. BERT and Qwen3 routing use the row
kernel, whose reported static threadgroup memory remains zero.

CLIP forms a `[batch, batch]` image/text similarity matrix and applies
cross-entropy in both directions. At the selected batch size 128, the
contiguous direction uses a `(128, 128)` log-softmax reduction. The model's
loss and matrix dimensions are explicit in
[Transformers v4.57.1 CLIP](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/clip/modeling_clip.py);
[PyTorch cross-entropy](../../aten/src/ATen/native/LossNLL.cpp) calls log-softmax.
This shape was already measured under the raw identifier
`qwen3_moe_router_t128`. Its CLIP use was identified afterward, then checked
with both archived kernels in paired measurements. The original identifiers
and role flags remain unchanged in the raw data. No additional shape is counted.
This is a benchmark of the reduction, not the complete CLIP loss or model.

The paired column uses GPU counters with both compiled libraries in the same
process and identical buffers. It is corroborating shader evidence, not the
same metric as the public-operator column. Compute-pass instrumentation changes
absolute timings and can also affect the relative cost. The interval is a
within-run bootstrap interval for the median paired change, not a guarantee
about other runs or GPUs. The CLIP FP32 case has two public screening runs;
it was not part of the three longer public confirmation runs.

## Outliers and limits of the findings

| Case | Screening change | Longer confirmation | Paired GPU change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5 decode, 63 keys, FP16 softmax | +35.50% | -17.43% | -8.10% | Largest screening outlier rejected |
| BERT `(8, 30522)`, FP32 log-softmax | +13.33% | -2.84% | -1.69% | Regression rejected |
| GPT-2 `(32, 50257)`, FP32 log-softmax | +7.81% | +3.31% | +0.38% | Not a confirmed >5% regression |
| Qwen3 `(512, 151936)`, FP32 log-softmax | +5.60% | +1.42% | -0.29% | Not a confirmed >5% regression |
| Qwen2.5 `(4, 152064, 32)`, dim 1, FP32 log-softmax | +1.06% | +6.71% | +0.08% | Public-run variation; not corroborated |

For the last row, candidate run medians were 3047.821, 3145.811, and
2876.016 us versus main's 2855.555, 2857.229, and 2857.904 us. Its paired
95% interval was [-0.48%, +0.83%]. Two neighboring strided log-softmax
controls also stayed near parity. The pooled +6.71% is retained in the data;
it is not presented as a stable kernel regression.

The selected vocabulary log-softmax cases did not establish a >5% regression
in both public and paired measurements. CLIP provides a better-supported
log-softmax counterexample. Log-softmax also regressed on several attention
and router shapes, but those models use softmax there; those results are
counterpart stress tests, not evidence of their actual attention operation.

Lower-precision softmax confirmations include BERT attention `(1,12,128,128)`
BF16 (+7.71%), Qwen3 routing `(2048,128)` FP16 (+8.02%), and Qwen2.5 prefill
`(1,28,128,128)` FP16 (+8.00%). Paired measurements supported these slowdowns.
Where the reference model explicitly upcasts its softmax to FP32, including
Qwen/Mixtral routing and Qwen/ViT eager attention, these lower-precision results
are dtype sensitivity probes. They are not claims about the default model path.

## Memory

All **474 operator/shape/dtype comparisons** matched for allocated bytes,
reserved bytes, peak allocated bytes, and peak reserved bytes at every recorded
stage. Peaks were reset with the input resident. The largest allocated peak
was 608 MiB and the largest reserved peak was 1032 MiB for both versions and
both operators. Allocated and reserved bytes returned to zero after cleanup.

One original live driver snapshot was 16 KiB larger on the candidate, for
FP16 log-softmax on `(1,28,128,4096)`. The difference was already present
with only the input resident, before the operator. Repeating the full FP16
log-softmax memory suite twice per version produced both values on main and
the lower value on both candidate repeats:

| Version/run | Driver bytes before operator, input live | Driver bytes with output live |
| --- | ---: | ---: |
| Main original / repeat 1 | 1074216960 | 1074216960 |
| Main repeat 2 | 1074233344 | 1074233344 |
| Candidate original | 1074233344 | 1074233344 |
| Candidate repeats 1 / 2 | 1074216960 | 1074216960 |

Every one of these probes returned to 475136 driver bytes after cleanup.
This is setup variability, not verified operator-added memory. Across the
original suite, driver differences ranged from -16 KiB to +16 KiB. Driver
snapshots do not measure peak driver usage.

Compiled pipeline queries gave the following values for both operators and
all three dtypes (72 pipeline queries across the two versions):

| Kernel path | Main static bytes/threadgroup | Candidate | Change |
| --- | ---: | ---: | ---: |
| Row | 0 | 0 | 0 |
| Contiguous split partial | 256 | 384 | +50% |
| Contiguous split finalize | 256 | 384 | +50% |
| Generic | 4096 | 8192 | +100% |
| Strided partial / finalize | 8192 | 8192 | 0% |

The additional count-reduction arrays explain the increased static allocations.
[Metal's `staticThreadgroupMemoryLength`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/staticthreadgroupmemorylength.md)
reports this per-threadgroup resource. It does not include tensor allocations,
registers, or spill storage. Register/spill usage, occupancy, and peak driver
memory were not measured, so the latency change cannot be attributed solely
to shared-memory growth.

## Model provenance and coverage

Dimensions come from published configurations and implementation code.
Batch sizes, sequence lengths, and row counts are selected workloads, not
captured execution traces. Inputs are seeded normal logits, not model outputs;
no checkpoints or full inference workloads were run.

| Model | Fixed dimensions used | Source |
| --- | --- | --- |
| Qwen/Qwen2.5-7B-Instruct | 28 attention heads; vocabulary 152064; maximum positions 32768 | [Pinned configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json) |
| Qwen/Qwen3-30B-A3B | 128 routing experts; vocabulary 151936 | [Pinned configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) |
| mistralai/Mixtral-8x7B-v0.1 | 8 routing experts; vocabulary 32000 | [Pinned configuration](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/fc7ac94680e38d7348cfa806e51218e6273104b0/config.json) |
| openai-community/gpt2 | Vocabulary 50257 | [Pinned configuration](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json) |
| google-bert/bert-base-uncased | 12 attention heads; vocabulary 30522; maximum positions 512 | [Pinned configuration](https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/config.json) |
| google-t5/t5-base | Vocabulary 32128 | [Pinned configuration](https://huggingface.co/google-t5/t5-base/blob/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1/config.json) |
| google/vit-base-patch16-224 | 224-pixel image, 16-pixel patches, 12 heads: 197 tokens including the class token | [Pinned configuration](https://huggingface.co/google/vit-base-patch16-224/blob/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3/config.json) |
| microsoft/swin-tiny-patch4-window7-224 | 224-pixel image, patch size 4, window size 7, stage-one heads 3: 64 windows of 49 tokens | [Pinned configuration](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/blob/d00d478bfaf1f417d34a1186673ad4b4be264c51/config.json) |


The source implementations specify the reduction axes and FP32 upcasts:
[Qwen2 attention](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen2/modeling_qwen2.py),
[Qwen3 routing](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py),
[Mixtral routing](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/mixtral/modeling_mixtral.py),
[ViT](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/vit/modeling_vit.py),
and [Swin](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/swin/modeling_swin.py).
Vocabulary shapes represent probability normalization, beam scores, or flattened
training logits. [Generation code](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py)
uses softmax for sampling, log-softmax for beam scores, and optional log-softmax
normalization of transition scores. [BERT's masked-LM head](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/bert/modeling_bert.py)
uses cross-entropy on flattened vocabulary logits; these are not BERT generation traces.

The 79 shapes comprise:

- 10 router shapes: 8 or 128 experts, with 1, 8, 128, 2048, or 8192 token rows.
- 7 vision/BERT attention shapes: ViT batches 1/8, Swin windows, and BERT lengths 64/128/256/512.
- 15 Qwen2.5 decode shapes, with key lengths 63 through 32768, including neighbors of 64, 128, 1024, and 2048.
- 5 Qwen2.5 chunked-prefill shapes: 128 queries and 128/512/1024/2048/4096 keys.
- 36 vocabulary shapes: six vocabulary sizes and 1/8/32/128/256/512 rows.
- 6 noncontiguous transition-score shapes `[B,V,T]`, reduced over dimension 1, with input strides `(V,1,B*V)` from a contiguous `[T,B,V]` buffer.

CLIP supplies a ninth model application for the existing `(128,128)` shape.
The full catalog and source snapshots are linked below. The search probes
both dispatch boundaries and changing numbers of independent rows, rather
than extrapolating from an overall average.

Screening flagged 90/474 combinations above 5%; these are candidates for
investigation, not 90 confirmed real-model regressions. No strided case crossed
5% in the initial screen. The 40 selected longer confirmations had 21 pooled
changes above 5%, including counterpart operators and the uncorroborated
strided result. The complete selected table follows so negative and marginal
results remain visible. All reductions use the last dimension except the
transition-score row, which uses dimension 1.

| Case | Shape | Dtype | Softmax main us | Candidate us | Change | Log-softmax main us | Candidate us | Change |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bert_vocab_r8 | `(8, 30522)` | float32 | 13.237 | 12.449 | -5.96% | 12.783 | 12.419 | -2.84% |
| gpt2_vocab_r32 | `(32, 50257)` | float32 | 169.718 | 174.178 | +2.63% | 169.800 | 175.416 | +3.31% |
| qwen25_decode_s2049 | `(1, 28, 1, 2049)` | float32 | 5.803 | 6.235 | +7.46% | 5.651 | 6.251 | +10.63% |
| bert_attention_s128 | `(1, 12, 128, 128)` | float32 | 8.286 | 8.841 | +6.70% | 7.812 | 9.011 | +15.35% |
| qwen3_moe_router_t2048 | `(2048, 128)` | float32 | 10.442 | 11.431 | +9.48% | 9.833 | 11.490 | +16.85% |
| qwen3_moe_vocab_r512 | `(512, 151936)` | float32 | 12891.651 | 13282.984 | +3.04% | 12917.323 | 13100.328 | +1.42% |
| gpt2_vocab_r512 | `(512, 50257)` | float32 | 4470.917 | 4538.535 | +1.51% | 4481.307 | 4558.918 | +1.73% |
| qwen25_decode_s1536 | `(1, 28, 1, 1536)` | float32 | 6.232 | 6.278 | +0.74% | 6.142 | 6.360 | +3.54% |
| qwen25_decode_s63 | `(1, 28, 1, 63)` | float16 | 1.473 | 1.216 | -17.43% | 1.470 | 1.333 | -9.33% |
| qwen3_moe_router_t1 | `(1, 128)` | float16 | 1.737 | 1.860 | +7.03% | 1.728 | 1.896 | +9.76% |
| mixtral_vocab_r8 | `(8, 32000)` | float16 | 13.254 | 12.832 | -3.19% | 12.627 | 12.568 | -0.47% |
| qwen25_prefill_q128_k128 | `(1, 28, 128, 128)` | float16 | 16.146 | 17.438 | +8.00% | 15.026 | 17.684 | +17.69% |
| bert_attention_s128 | `(1, 12, 128, 128)` | bfloat16 | 7.569 | 8.152 | +7.71% | 7.181 | 8.398 | +16.95% |
| qwen3_moe_router_t2048 | `(2048, 128)` | float16 | 9.669 | 10.445 | +8.02% | 9.016 | 10.594 | +17.49% |
| qwen3_moe_router_t128 | `(128, 128)` | bfloat16 | 1.917 | 1.950 | +1.71% | 1.895 | 2.080 | +9.72% |
| qwen25_decode_s2049 | `(1, 28, 1, 2049)` | float16 | 5.427 | 6.069 | +11.82% | 5.269 | 6.096 | +15.70% |
| vit_attention_b1 | `(1, 12, 197, 197)` | bfloat16 | 17.403 | 18.465 | +6.10% | 16.175 | 18.321 | +13.27% |
| qwen25_vocab_r512 | `(512, 152064)` | float16 | 6896.990 | 6970.773 | +1.07% | 6899.486 | 6969.617 | +1.02% |
| qwen25_transition_b4_t32 | `(4, 152064, 32)` | float32 | 2855.668 | 3033.587 | +6.23% | 2856.229 | 3047.821 | +6.71% |
| qwen25_vocab_r1 | `(1, 152064)` | float32 | 10.186 | 9.345 | -8.26% | 9.764 | 9.330 | -4.44% |

## Measurement and verification

Both builds used the existing Release configuration with `USE_DISTRIBUTED=0`
and `USE_MKLDNN=OFF`, through `pip install -e . -v --no-build-isolation`.
Benchmark GPU work did not overlap builds or another benchmark job.

Screening used 10 warmups, six trials, and a pilot selecting 1-200 calls per
trial toward 20 ms. Two shuffled orders (seeds 1 and 2) ran on each version.
Only the first trial was excluded, by design; all remaining trials were retained.
The comparison checks matching input fingerprints, shapes, dimensions, strides,
harness hashes, and catalog hashes.

Confirmation selected 20 shape/dtype pairs and both operators. It used new
input seeds (original seed +100000), 50 warmups, 12 trials, and 1-3000 calls
per trial toward 100 ms, in three shuffled orders (101, 102, 103). Each pooled
median therefore includes 33 retained trials per version. Timing outliers were
not removed. MPS event intervals include GPU queue gaps; separately saved wall
times include submission and synchronization. These measure batched calls with
preallocated outputs, not output allocation or isolated shader latency.

Memory used allocating calls in fresh processes per operator and dtype, with
before/input-live/output-live/deleted/clean snapshots. Only one shape's tensors
were resident at a time. Two additional full FP16 log-softmax memory runs per
version checked the driver discrepancy.

Paired shader measurements loaded both archived libraries in one process,
alternated version order over 32 rounds, and shuffled case order. Row cases
used 64 calls per batch; split/generic/strided cases used 1, 16, or 64 according
to tensor size. Both versions used identical buffers and launch geometry.
Every measured batch completed before the next. A logical split sample spans
both stages, including their intervening gap. This M4 exposes compute-pass
timestamps rather than dispatch timestamps. Calibration follows
[Apple's CPU/GPU timestamp procedure](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time.md).
The paired statistic is the median of per-round candidate/main batch-median
changes, with 4000 bootstrap resamples of the 32 round-level changes. It is not
the ratio of independently pooled medians. Row/CLIP checks use screening
inputs; split/generic/strided checks use the new confirmation inputs.

Each rebuilt revision passed 40 selected checks covering both public operators
against CPU float64, sampling three complete reduction rows per case. Paired-library checks
passed for nine row, seven split/generic, three strided, and three CLIP dtype
cases before timing. The row checks compare complete outputs; the other paths
sample three reduction rows. These checks validate the benchmark bindings and
ordinary logits, not an additional full-model or collapse-correctness suite.

Source and binary identities:

| Artifact | SHA-256 |
| --- | --- |
| Main `SoftMax.metal` | `1187df4a103c3b031ea8e2b432d0217cabd6b71aa98f4a8be151b6277e426f74` |
| Candidate `SoftMax.metal` | `cb9e8317fac79d0ad1947a15debd47356f542959c1842d387ed82ee3469ca848` |
| Main embedded Metal library | `9c20e793cb1872870f03cf5ad78631689c497f48b491ff212eebaf03a7d7104c` |
| Candidate embedded Metal library | `7899be0f8916b57faf678175f6177cd8cb58527f017203c71e1df0eaf200d3a3` |

Initial candidate screening used a build carrying the earlier `ffe08491847`
version string, with the final source and binary hashes above. Rebuilding the
final candidate commit reproduced the identical embedded library before the
longer confirmations. Rebuilding main likewise reproduced its archived library.

## Reproduction and local artifacts

Run from the repository root. The frozen investigation harnesses, catalogs,
raw JSON, source snapshots, and build/test logs are retained under the
**git-ignored** `agent_space/softmax_adversarial/` directory in this workspace.
They are not included by copying or committing this Markdown file alone.

- [All 79 shapes](../../agent_space/softmax_adversarial/all_cases.json) and [source manifest](../../agent_space/softmax_adversarial/sources/manifest.json).
- [All 474 screening comparisons, CSV](../../agent_space/softmax_adversarial/screen_comparison.csv).
- [All 40 longer confirmations, CSV](../../agent_space/softmax_adversarial/confirmation_comparison.csv), including individual run medians.
- [All memory comparisons](../../agent_space/softmax_adversarial/memory_comparison.json) and [pipeline resources](../../agent_space/softmax_adversarial/resources.json).
- Paired [row](../../agent_space/softmax_adversarial/paired_rows.json), [split/generic](../../agent_space/softmax_adversarial/paired_pipelines.json), [strided](../../agent_space/softmax_adversarial/paired_strided.json), and [CLIP](../../agent_space/softmax_adversarial/paired_clip.json) data.

Build each revision with the same saved configuration, then run each comparison
in fresh processes with the matching revision label. Preserve the harnesses
while changing revisions. For example, for candidate confirmation:

```sh
source venv/bin/activate
USE_DISTRIBUTED=0 CMAKE_BUILD_TYPE=Release pip install -e . -v --no-build-isolation
for bench_run in 101 102 103; do
  python agent_space/softmax_adversarial/confirm.py \
    --case-file agent_space/softmax_adversarial/all_cases.json \
    --selection agent_space/softmax_adversarial/confirmation_selection.json \
    --label candidate --output "agent_space/softmax_adversarial/candidate_confirm${bench_run}.json" \
    --order-seed "$bench_run" --trials 12 --warmup 50 --target-ms 100 --max-steps 3000
done
python agent_space/softmax_adversarial/test_selected.py
```

`run_main.sh` records the exact screening/memory commands; the primary suite
uses frozen `run_v1.py`, and the strided supplement uses `run.py`. Repeated
confirmations are compared with:

```sh
python agent_space/softmax_adversarial/analyze.py \
  --baseline agent_space/softmax_adversarial/main_confirm{101,102,103}.json \
  --candidate agent_space/softmax_adversarial/candidate_confirm{101,102,103}.json \
  --output agent_space/softmax_adversarial/confirmation_comparison.json
python agent_space/softmax_adversarial/paired_rows.py \
  --selection agent_space/softmax_adversarial/clip_counter_selection.json \
  --output agent_space/softmax_adversarial/paired_clip.json
python agent_space/softmax_adversarial/paired_pipelines.py
python agent_space/softmax_adversarial/paired_strided.py
```
