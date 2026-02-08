Perfect — Phase 2 / M11 is a clean pivot point. Based on where M10 landed, here’s a **Cursor-ready M11 plan** that stays squarely inside your refactoring posture and Phase II goals.

Below is a **drop-in milestone plan** you can hand directly to Cursor.

---

# **M11_plan — Distributed Protocol Version Guardrail**

## 1. Intent / Target

Introduce an **explicit, machine-checkable protocol version guardrail** for `torch.distributed` to prevent **silent cross-version incompatibility** between workers (e.g., old worker ↔ new parameter server).

This milestone converts an *implicit assumption* (“distributed versions are compatible”) into an **explicit invariant with enforcement and tests**.

**Why this exists:**

* Distributed training is a **critical external surface**
* Cross-version mismatches currently fail:

  * late
  * opaquely
  * sometimes silently
* The baseline audit and system surface map explicitly flag this as a **missing guardrail**

This milestone does **not** redesign distributed internals. It adds **minimal protocol version signaling + validation**.

---

## 2. Scope Boundaries

### In Scope

* `torch.distributed` initialization path
* Protocol version declaration (new internal module)
* Early runtime validation during `init_process_group`
* Targeted distributed tests (multi-process, mismatch case)
* Documentation comments explaining intent and invariants

### Explicitly Out of Scope

* No backend protocol redesign (NCCL/Gloo/MPI untouched)
* No wire-format changes beyond version handshake
* No behavior change for **matching versions**
* No performance optimization
* No changes to RPC, DDP, FSDP logic
* No CI restructuring beyond adding tests

---

## 3. Refactor Posture

**Behavior-Preserving (Strict)**

* Matching-version behavior must remain **bit-for-bit identical**
* Only new behavior:

  * **early, explicit failure** when protocol versions mismatch
* Any externally observable behavior change **must be justified by mismatch scenario**

---

## 4. Declared Invariants

These must hold after M11:

1. **Distributed correctness invariant**

   * Matching-version distributed runs behave identically to baseline

2. **Failure clarity invariant**

   * Version mismatches fail early with a clear, deterministic error message

3. **Compatibility invariant**

   * No existing user code breaks when all nodes run the same version

4. **Isolation invariant**

   * No protocol version logic leaks into non-distributed code paths

---

## 5. Verification Plan

### Tests

Add **explicit regression coverage**:

* New test file:

  * `test/distributed/test_protocol_version.py`
* Scenarios:

  1. Matching versions → success
  2. Mismatched versions → deterministic failure
* Multi-process execution (2 ranks minimum)

### Validation Signals

* CI distributed test suite passes
* New test fails on baseline (pre-M11), passes post-M11
* No existing distributed tests regress

### Negative Proof

* Remove version check → test must fail (proves guardrail effectiveness)

---

## 6. Implementation Steps (Ordered, Reversible)

1. **Introduce protocol version constant**

   * New file: `torch/distributed/_protocol_version.py`
   * Single source of truth (e.g., integer or semantic tuple)

2. **Wire version exchange into initialization**

   * Hook into `init_process_group`
   * Exchange protocol version across ranks
   * Compare before backend initialization proceeds

3. **Define failure semantics**

   * Explicit exception type
   * Clear error message:

     * local version
     * remote version(s)
     * guidance (“all nodes must run compatible versions”)

4. **Add targeted tests**

   * Minimal distributed harness
   * Explicit mismatch simulation

5. **Documentation comments**

   * Inline rationale (why protocol version exists)
   * Clear note that this is a *guardrail*, not feature logic

---

## 7. Risk & Rollback Plan

### Risks

* False positives if version detection is wired incorrectly
* Test flakiness if process coordination is brittle

### Mitigations

* Keep version type simple (integer or tuple)
* Fail **before** backend-specific logic
* Limit scope strictly to init path

### Rollback

* Single revert removes:

  * `_protocol_version.py`
  * version check
  * new test file
* No data migrations or persistent artifacts involved

---

## 8. Deliverables

**Code**

* `torch/distributed/_protocol_version.py`
* Modified `init_process_group` logic
* `test/distributed/test_protocol_version.py`

**Docs / Governance**

* M11 milestone artifacts:

  * `docs/refactor/milestones/M11/M11_summary.md`
  * `docs/refactor/milestones/M11/M11_audit.md`
  * Toolcall log update

---

## 9. Exit Criteria

M11 may be closed when:

* All CI is green
* New mismatch test fails on baseline, passes post-change
* No distributed behavior changes observed for matching versions
* Audit verdict: **🟢 Safe, guardrail added, no drift**

---

## 10. Authorized Next Step

Upon M11 closure:

* Proceed within **Phase 2** to the next **protocol / compatibility guardrail** or
* Transition to next Phase per Phase Map, **without reopening distributed internals**

---

If you want, next we can:

* sanity-check Phase 2’s remaining milestones for overlap risk, or
* pre-draft the **M11 audit expectations** so Cursor knows exactly what “good” looks like before implementation starts.
