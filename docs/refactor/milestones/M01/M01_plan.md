Below is a **Cursor-ready M01 plan + execution prompt** that stays consistent with the M00 baseline and governance posture (docs-only in M00; execution begins in M01). It explicitly references M00 as the locked baseline and keeps blast radius minimal.   

---

# M01_plan — Import Smoke Test Foundation (No Build Required)

## Intent / Target

Establish the **first executable verification harness** that can run **without a local C++ build** by validating **Python import-graph integrity** and **Python syntax compilation** for key `torch.*` packages. This creates a minimal “truth surface” for subsequent refactors. 

## Scope Boundaries

**In scope**

* Add a **static import-graph smoke check** (no execution of compiled bindings).
* Add a lightweight **syntax compile** check (`compileall`) for a targeted subset of packages.
* Add minimal docs/log updates to keep governance current (`REFACTOR.md`, toolcalls, M01 plan doc).

**Out of scope**

* No production refactors.
* No dependency upgrades.
* No formatting sweeps.
* No local build attempts (`setup.py`, CMake, Bazel).
* No modifications to existing PyTorch CI workflows (only an optional *new*, isolated workflow if needed).

## Invariants

Must not change:

* Runtime behavior / public APIs (Python/C++).
* Build system behavior.
* Existing CI workflows and gating semantics.
* Repository output formats, serialization formats, or protocol behavior.

M01 adds **new checks only**; it does not modify shipped code paths. (M00 invariant posture remains authoritative.) 

## Verification Plan

### Local (must work)

* `python -m tools.refactor.import_smoke_static --targets torch,torch.nn,torch.optim,torch.utils,torch.distributed,torch.fx,torch._dynamo,torch._inductor,torch.onnx`
* `python -m compileall torch torchgen -q` *(or targeted subpaths for speed)*

**Success criteria**

* Exit code 0.
* Output includes a short, deterministic summary:

  * targets checked
  * number of modules scanned
  * unresolved imports (should be 0, excluding an explicit allowlist of known compiled/dynamic modules)
  * elapsed time

### CI (optional, but recommended if Actions are enabled on your fork)

* Add a new isolated workflow `.github/workflows/refactor-smoke.yml` that runs only this static check on PRs.
* This workflow should be **non-invasive**: no touching existing workflow files; no job-filter integration.

## Implementation Steps (ordered, reversible)

1. **Create tooling module**

   * `tools/refactor/__init__.py`
   * `tools/refactor/import_smoke_static.py`

2. **Implement static import resolver**

   * For each target package:

     * locate package/module on disk in repo tree
     * AST-parse `.py` files
     * collect `import` / `from ... import ...`
     * resolve **internal** imports (`torch.*` and relative imports)
     * verify referenced modules exist on disk
   * Maintain a small **allowlist** for known non-Python/compiled/dynamic modules that should *not* be resolved (e.g., `torch._C`, `torch._C.*`, and other known generated/compiled names).
   * Deterministic output ordering (sorted lists).

3. **Add a tiny test wrapper**

   * Add `test/test_import_smoke_static.py` that calls the tool for a small default target set and asserts success.
   * Keep it standalone (stdlib `unittest` OK) so it runs without installing extras.

4. **Optional: add isolated CI workflow (fork-safe)**

   * `.github/workflows/refactor-smoke.yml`
   * Trigger: `pull_request` (and optionally `workflow_dispatch`)
   * Steps:

     * checkout
     * setup-python (3.12)
     * run the static smoke tool
   * No secrets, no caches required.

5. **Docs updates (minimal)**

   * Update `REFACTOR.md` M01 section to reflect **static import-graph check** (avoid wording about mocking `torch._C`). 
   * Add `docs/refactor/milestones/M01/M01_plan.md` capturing this plan.
   * Update `docs/refactor/toolcalls.md` with M01 actions as they occur.

## Risk & Rollback Plan

**Risks**

* False positives from optional/platform imports.
* Resolver overreaches into generated/compiled surfaces.

**Mitigations**

* Keep target list small and explicit.
* Use a conservative allowlist for compiled/dynamic modules.
* Only assert on what can be proven from the source tree.

**Rollback**

* `git revert` the M01 commit(s) (docs + tool + optional workflow).
* Since M01 adds new files, rollback is clean and low-risk.

## Deliverables

* `tools/refactor/import_smoke_static.py` (+ package init)
* `test/test_import_smoke_static.py`
* *(optional)* `.github/workflows/refactor-smoke.yml`
* `docs/refactor/milestones/M01/M01_plan.md`
* `REFACTOR.md` updated M01 wording (scope/verification alignment)
* `docs/refactor/toolcalls.md` updated

---

# Cursor Handoff Prompt — Implement M01 (Static Import Smoke Test)

You are Cursor AI acting as a **Refactoring Milestone Implementer**.

## Context

M00 is complete and locked. The audit pack is the immutable baseline; `REFACTOR.md` is the living log. M01 is the first execution milestone and must remain **low blast radius**.   

## Objective (M01)

Implement a **static import-graph smoke test + syntax compile check** that runs without a local C++ build.

## Hard Constraints

* **No production code refactors**
* **No dependency changes**
* **No formatting sweeps**
* **No local build attempts**
* Do not modify existing `.github/workflows/*` files (optional: add a brand new workflow only)
* Output must be deterministic and conservative

## Tasks

1. Create `tools/refactor/import_smoke_static.py`:

   * CLI entrypoint (`python -m tools.refactor.import_smoke_static`)
   * Accept `--targets` (comma-separated module roots)
   * Scan relevant `.py` files; AST-parse imports
   * Resolve internal imports conservatively
   * Allowlist known compiled/dynamic modules to avoid false positives
   * Print deterministic summary and return exit code 0/1

2. Add `test/test_import_smoke_static.py`:

   * Minimal test invoking the tool against a small default target list
   * Use stdlib `unittest` or lightweight pytest style without new deps

3. (Optional) Add `.github/workflows/refactor-smoke.yml`:

   * Runs only the static smoke tool on PRs
   * Self-contained; no changes to existing workflows

4. Documentation updates:

   * Add `docs/refactor/milestones/M01/M01_plan.md` (copy the plan structure)
   * Update `REFACTOR.md` M01 section to reflect *static import-graph check* (remove/avoid “mock torch._C” language)
   * Update `docs/refactor/toolcalls.md` entries for the changes you made

## Verification

* Run the tool locally in the repo and show the output summary in the PR description (or capture in docs).
* Ensure tests pass for the new test file (if runnable).
* Ensure the workflow (if added) is green on the fork.

## Deliverable Definition

M01 is “done” when:

* The static smoke tool runs deterministically and exits 0 on the repo tree
* The test wrapper exists and passes where executed
* Docs updated (M01 plan + REFACTOR.md alignment + toolcalls)
* No scope violations occurred

Proceed with minimal, conservative changes.

---

If you want, I can also generate a **matching PR description template** for M01 (with invariants + verification evidence), but the above is enough to hand off and implement cleanly.
