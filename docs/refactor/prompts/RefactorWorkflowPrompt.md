

Updated `workflowprompt.md` (Refactoring Projects)
--------------------------------------------------

🧭 **CI / Workflow Run Analysis Prompt — Refactoring Posture (Enterprise-Capable, Project-Agnostic)**

### Purpose

Analyze a **single CI workflow run** and extract **ground-truth evidence** about refactor safety, behavioral stability, and governance integrity.

This analysis must by default:

* Treat CI as a primary **truth signal**, not mearly a success badge

* Assume refactoring is **behavior-preserving by default**

* Distinguish **signal** from **noise**

* Produce **audit-defensible conclusions**

* Be suitable for **merge gating**, **milestone closure**, or **contract certification**

Do **not** assume failure.  
Do **not** assume green means “safe.”  
Do **not** optimize for speed over correctness.

* * *

Inputs (Mandatory)
------------------

Before analysis begins, identify and record:

### 1) Workflow identity

* Workflow name

* Run ID

* Trigger (PR, push, manual, scheduled)

* Branch + commit SHA

* PR number (if applicable)

### 2) Change context (Refactor-specific)

* Milestone / phase / objective (if applicable)

* Declared intent of the change

* **Refactor target surface** (module/subsystem/contract boundary/repo split)

* Whether the milestone posture is:
  
  * **Behavior-preserving**
  
  * **Behavior-changing (explicitly authorized)**
  
  * **Mixed / UNKNOWN**

* Whether this run is:
  
  * exploratory
  
  * corrective
  
  * hardening
  
  * release-related
  
  * consumer-certification

### 3) Baseline reference

* Last known **trusted green** commit/tag

* **Declared invariants** vs baseline (if any), such as:
  
  * public API compatibility
  
  * CLI output stability
  
  * schema/contract stability
  
  * determinism / golden outputs
  
  * packaging/build artifact compatibility

If any mandatory input cannot be determined, mark **UNKNOWN** explicitly.

* * *

Step 1 — Workflow Inventory
---------------------------

Enumerate **all jobs and checks** executed in this run.

For each job, record:

| Job / Check | Required? | Purpose | Pass/Fail | Notes |
| ----------- | --------- | ------- | --------- | ----- |

Explicitly identify:

* Which checks are **merge-blocking**

* Which checks are **informational**

* Which checks use `continue-on-error` (and why)

* Which checks are **new**, **removed**, or **reclassified** vs baseline

If any required checks are muted, weakened, bypassed, or conditionally skipped, **flag immediately**.

* * *

Step 2 — Refactor Signal Integrity
----------------------------------

For each category, state **what it truly measures** and whether it covers the **changed surface**.

### A) Tests

* What tiers ran (unit / integration / contract / e2e / smoke)?

* Did tests cover the **refactor target surface** (the code you touched)?

* Are failures:
  
  * real correctness failures
  
  * contract violations
  
  * test instability/flakiness
  
  * environment/tooling drift

* If behavior-preserving refactor: were any **golden/snapshot/regression** tests present?

* Are key tests missing for the touched surface?

### B) Coverage (and whether it is meaningful)

* What coverage is enforced (line/branch/trace/mutation)?

* Is coverage scoped correctly (changed packages included)?

* Any exclusions introduced/expanded? Are they justified?

* If coverage changed materially, is it expected given refactor mechanics, or does it suggest **untested new pathways**?

### C) Static / Policy Gates

* Linting, formatting, typing, architecture boundaries, doc checks

* Are these gates enforcing **current reality**, or legacy assumptions?

* Did any refactor cause import boundary breaks, circular deps, or layering violations?

### D) Security / Supply Chain Signals (if present)

* SAST (bandit), dependency audit, secret scan, SBOM, scorecard

* Are failures true findings vs tool drift?

* Any new risky dependency or insecure pattern introduced by the refactor?

### E) Performance / Benchmarks (if present)

* Are benchmarks isolated from correctness signals?

* If performance regressed, is it due to refactor structure (e.g., new abstraction overhead)?

* Do benchmarks still run deterministically and without muting important gates?

* * *

Step 3 — Delta Analysis (Change Impact vs Baseline)
---------------------------------------------------

Analyze **what changed vs baseline** and what CI proved about it.

1. **Change inventory**
* Which files/packages/contracts were modified?

* Were any public surfaces touched (CLI, API endpoints, schemas, serialized formats)?
2. **Expected vs observed deltas**
* Expected: what should have changed (structure, boundaries, wiring)?

* Observed: what CI reveals changed (new failures, new passes, skipped checks)
3. Refactor-specific drift detection (call out explicitly)
* **Signal drift:** tests skipped, coverage misleading, gates silently bypassed

* **Coupling revealed:** refactor triggered failures in “unrelated” components

* **Hidden dependencies:** import cycles, runtime side effects, implicit ordering

* * *

Step 4 — Failure Analysis (If Any)
----------------------------------

For each failure:

1. Classify:
* Correctness bug

* Contract mismatch / schema break

* Behavioral drift (unintended output change)

* Test fragility

* CI misconfiguration

* Tooling/environment drift

* Policy violation (lint/type/security)
2. Determine:
* Is this **in-scope** for the milestone?

* Is it **blocking**, **deferrable**, or **informational**?
3. If deferring:
* Why deferrable

* Where tracked (issue/registry/doc)

* What guardrail prevents silent regression

* Whether deferral is compatible with “behavior-preserving” posture

* * *

Step 5 — Invariants & Guardrails Check (Mandatory for Refactors)
----------------------------------------------------------------

Explicitly assert whether these held:

* Required checks remain enforced (no weakening)

* Refactor did not expand scope into feature work

* Public surfaces remained compatible (or breaks were explicitly authorized)

* Schema/contract outputs remain valid

* Determinism/golden outputs preserved (if required)

* No “green but misleading” path (skips, silent continues, missing tiers)

If any invariant was violated:

* Describe the violation

* Assess blast radius (what could break downstream)

* Recommend containment, rollback, or an explicit “behavior change” decision

* * *

Step 6 — Verdict (Single Paragraph + One Flag)
----------------------------------------------

Provide a one-paragraph verdict:

> **Verdict:**  
> (e.g., “Safe to merge,” “Green but misleading,” “Refactor introduced behavior drift,” “CI missing required coverage on changed surface.”)

Then explicitly state **exactly one** recommended outcome:

* ✅ Merge approved

* ⛔ Merge blocked

* ⚠️ Merge allowed with documented debt (must by default specify the debt + guardrail)

* 🔁 Re-run required (with reason)

* * *

Step 7 — Next Actions (Minimal & Explicit)
------------------------------------------

List only concrete next actions. Each must by default include:

* Owner (Cursor / human)

* Scope (precise)

* Whether it fits this milestone or requires a **new milestone**

* Any required guardrail additions

Avoid speculative refactors.  
Avoid “nice-to-have” cleanups.  
Prefer the smallest action that restores truth and preserves invariants.

* * *

Output Requirements
-------------------

The final analysis must by default be:

* Structured

* Auditable

* Copy/pasteable into milestone logs, audits, PR comments, and release records

* Neutral, technical, authoritative

* Explicit about UNKNOWNs

* * *


