# AOTAutograd Post-Compile Wrapper Audit (pythonify)

**Source references:**
- `torch/_functorch/_aot_autograd/graph_compile.py`
- `torch/_functorch/_aot_autograd/runtime_wrappers.py`

This note enumerates the post-compile wrappers that pythonify must model, the
order they are applied, and the metadata required to reproduce their
calling conventions. It captures both the dispatch-level wrappers applied
around compiled callables and the forward/backward wrapper stack used by
AOTAutograd.

## Wrapper ordering

### Dispatch wrappers (around compiled autograd/inference callable)
Wrappers created in `_create_wrappers_for_dispatch` run `post_compile` in
**reverse order** via `post_compile(wrappers, ...)`:
1. **AOTSyntheticBaseWrapper** → unpacks synthetic bases and reapplies metadata
   mutations on aliased inputs at runtime.
2. **AOTDedupeWrapper** → re-inserts duplicated arguments according to the
   recorded map.

### Forward/inference compilation (`_aot_stage2b_fw_compile` / `_aot_stage2b_compile_forward_or_inference`)
Post-compile wrappers are applied **in this order** to the compiled forward:
1. **EffectTokensWrapper** → prepends effect tokens to args and strips them
   from outputs.
2. **AOTDispatchSubclassWrapper** → unwraps tensor subclasses before calling
   the compiled fn, then re-wraps outputs; honors
   `runtime_metadata.subclass_inp_meta` and
   `runtime_metadata.subclass_fw_graph_out_meta` plus
   `num_fw_outs_saved_for_bw`.
3. **FunctionalizedRngRuntimeWrapper** → appends/consumes RNG state tuples when
   `runtime_metadata.is_rng_op_functionalized` is set; updates CUDA RNG offset
   using `runtime_metadata.num_outputs_rng_offset` and
   `runtime_metadata.num_forward_returns`.
4. **FakifiedOutWrapper** → re-fakifies outputs using
   `out_metas` (captured during tracing) and `fwd_output_strides`
   (from `TracingContext.report_output_strides`).

### Autograd assembly (`_aot_stage2c_make_autograd_function`)
`AOTDispatchAutograd.post_compile` stitches forward/backward into a
`torch.autograd.Function`, then wraps with runtime epilogue and dispatch
wrappers:
1. **AOTDispatchAutograd.post_compile** → wires fw/bw callables, handles
   saved tensors/symints ordering, RNG state pairing, lazy backward caching,
   and subclass tangent processing.
2. **RuntimeWrapper** → runtime alias/mutation epilogue; detaches inputs per
   `indices_of_inps_to_detach`, restores grad/autocast state, and rebuilds
   aliasing described by `runtime_metadata`.
3. **DebugAssertWrapper** (only when `config.debug_assert`) → validates
   `requires_grad` expectations captured in `flat_requires_grad`.
4. **AOTSyntheticBaseWrapper** (reverse dispatch wrapper order).
5. **AOTDedupeWrapper** (reverse dispatch wrapper order).

## Per-wrapper metadata and IO effects

### EffectTokensWrapper
- **Runtime inputs:** number of effect tokens = `len(runtime_metadata.tokens)`.
- **Behavior:** injects `None` tokens at the front of args; drops the same
  number of leading outputs.
- **Metadata to capture:** token count.

### AOTDispatchSubclassWrapper
- **Runtime inputs:** compiled function is boxed; expects unwrapped args.
- **Behavior:**
  - Unwraps tensor subclasses using `runtime_metadata.subclass_inp_meta`.
  - Re-wraps outputs using `runtime_metadata.subclass_fw_graph_out_meta` and
    `num_fw_outs_saved_for_bw` (backward saves).
- **Metadata to capture:**
  - `subclass_inp_meta`, `subclass_fw_graph_out_meta` from ViewAndMutationMeta.
  - `num_fw_outs_saved_for_bw` (autograd fw outputs saved for bw).
  - `maybe_subclass_meta` presence to gate wrapper creation.

### FunctionalizedRngRuntimeWrapper
- **Runtime inputs:** optionally appends CUDA RNG seed/offset when
  `runtime_metadata.is_rng_op_functionalized`.
- **Behavior:**
  - Adds `(seed, offset)` to args; after call, consumes an RNG offset output
    and updates CUDA RNG state.
  - May strip RNG offset from outputs when `return_new_outs` is True (inference).
- **Metadata to capture:**
  - `is_rng_op_functionalized`, `num_outputs_rng_offset`,
    `num_forward_returns`, `num_graphsafe_rng_states`,
    `graphsafe_rng_state_index`.

### FakifiedOutWrapper
- **Runtime inputs:** none additional.
- **Behavior:** re-fakifies outputs based on traced `out_metas`; uses
  `fwd_output_strides` to restore strides when available.
- **Metadata to capture:**
  - `out_metas` (from tracing context) and `fwd_output_strides` collected
    during forward compilation.

### RuntimeWrapper
- **Runtime inputs:** raw outputs of compiled fw/bw.
- **Behavior:**
  - Detaches inputs (`indices_of_inps_to_detach`).
  - Restores grad/autocast context based on `aot_config.keep_inference_input_mutations`.
  - Reconstructs aliasing for outputs using `runtime_metadata.output_info`,
    `num_mutated_inp_runtime_indices`, `num_outputs_aliased`,
    `aliased_out_indices`, and view metadata.
- **Metadata to capture:**
  - `indices_of_inps_to_detach`, `trace_joint`, `disable_amp`, and full
    `ViewAndMutationMeta` (output/alias info).

### AOTDispatchAutograd.post_compile
- **Runtime inputs:** compiled fw, compiled bw, saved metadata.
- **Behavior:**
  - Handles saved tensors/symints ordering based on
    `runtime_metadata.num_outputs`, `num_outputs_aliased`,
    `num_mutated_inp_runtime_indices`, `num_forward_returns`, and
    donation metadata.
  - Manages RNG pairing using `num_graphsafe_rng_states` and
    `graphsafe_rng_state_index` plus lazy backward info.
  - Processes tensor subclasses for tangents using
    `maybe_subclass_meta.grad_input_metas`.
- **Metadata to capture:**
  - `maybe_subclass_meta`, `num_symints_saved_for_bw`, `backward_state_indices`,
    `disable_amp`, `indices_of_inps_to_detach`, `lazy_backward_info`,
    `runtime_metadata` (ViewAndMutationMeta), and cache entry hook if present.
- **IR node mapping:** `AOTAutogradWrapperNode` with these fields:
  - Lazy backward: `has_lazy_backward`, `lazy_bw_module`, `lazy_bw_placeholder_list`,
    `lazy_bw_saved_context`, `lazy_bw_saved_compile_context`
  - Saved tensor slices: `tensors_saved_for_bw_with_vc_check_slice`,
    `tensors_saved_for_bw_no_vc_check_slice`, `symints_saved_for_bw_slice`,
    `num_symints_saved_for_bw`, `dynamic_saved_tensors_idxs`
  - RNG pairing: `num_graphsafe_rng_states`, `graphsafe_rng_state_index`,
    `is_rng_op_functionalized`
  - Autograd assembly: `backward_state_indices`, `indices_of_inps_to_detach`,
    `disable_amp`, `maybe_subclass_meta`, `fw_metadata`, `try_save_cache_entry_present`

### AOTDedupeWrapper
- **Runtime inputs:** deduped args.
- **Behavior:** removes duplicate args pre-compile and re-inserts them at
  runtime when needed.
- **Metadata to capture:**
  - `keep_arg_mask`, `add_dupe_map`, `needs_post_compile` flag.
  - `old_input_metadata` may be retained for validation.

### AOTSyntheticBaseWrapper
- **Runtime inputs:** args may include synthetic bases rather than views.
- **Behavior:**
  - Reconstructs original views from synthetic bases; reapplies metadata
    mutations for aliased inputs.
- **Metadata to capture:**
  - `synthetic_base_info`, `aliased_arg_idx_with_metadata_mutations`,
    `old_input_info`, `needs_post_compile`, `trace_joint`.

### DebugAssertWrapper
- **Runtime inputs:** args passed through.
- **Behavior:** asserts `requires_grad` expectations per input when
  `config.debug_assert` is enabled.
- **Metadata to capture:**
  - `flat_requires_grad` list aligned with runtime args.

## Summary of required pythonify metadata
To faithfully model wrapper behavior, pythonify must capture at least:
- Token count (`len(runtime_metadata.tokens)`).
- Subclass metadata (`subclass_inp_meta`, `subclass_fw_graph_out_meta`,
  `maybe_subclass_meta`, `num_fw_outs_saved_for_bw`).
- RNG functionalization flags and counts (`is_rng_op_functionalized`,
  `num_outputs_rng_offset`, `num_forward_returns`, `num_graphsafe_rng_states`,
  `graphsafe_rng_state_index`).
- Fakified outputs (`out_metas`, `fwd_output_strides`).
- Dedupe maps (`keep_arg_mask`, `add_dupe_map`, `needs_post_compile`).
- Synthetic bases (`synthetic_base_info`, `aliased_arg_idx_with_metadata_mutations`,
  `old_input_info`, `needs_post_compile`, `trace_joint`).
- Runtime epilogue (`indices_of_inps_to_detach`, `disable_amp`, full
  `ViewAndMutationMeta`).
- Autograd assembly (`num_symints_saved_for_bw`, `backward_state_indices`,
  `lazy_backward_info`, `maybe_subclass_meta.grad_input_metas`, cache entry hook
  presence).
- Debug asserts (`flat_requires_grad`).

This audit should be kept in sync with changes to AOTAutograd wrapper logic.
