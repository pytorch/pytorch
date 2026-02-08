

Refactoring Enhancements Standard (Universal) — v1
==================================================

0) Intent

---------

This document defines **enterprise-Capable refactoring enhancements** that improve:

* **Correctness preservation** (invariants + regression proof)

* **Auditability** (machine-readable evidence, SBOM/provenance, job summaries)

* **Security & supply chain** (dependency scanning, secret scanning, Scorecard)

* **Maintainability** (format/lint/types/docs/complexity gates)

* **Operational readiness** (docs + “QA & Evidence” publishing)

**Refactoring posture:** default is **behavior-preserving** change. Enhancements must by default be introduced in **small, non-overlapping phases**, each ending with **end-to-end verification** and **CI evidence artifacts**.

* The enhancements below are applied selectively and proportionally based on project size, risk profile, and client priorities.

* * *

1) Applicability Rules (Pre-Flight Detection)

---------------------------------------------

Before proposing enhancements, detect:

1. **Project shape**
   
   * Python-only library/service
   
   * Full-stack (backend + frontend)

2. **Packaging**
   
   * If Python packaging exists: OK to build wheel/sdist
   
   * If no packaging exists: **do not change build backend**; only add tool configs

3. **Runtime surfaces**
   
   * CLI? API? library import surfaces?
   
   * Dockerfile present?

4. **Test baseline**
   
   * Existing tests vs zero tests

**Output of pre-flight:** a short “Detected Surfaces & Constraints” section included in the milestone plan.

(Pre-flight requirements are explicitly called out as Phase 0 in the audit doc.)

* * *

2) Refactor Safety Rules

------------------------

These rules apply across all phases:

### 2.1 Behavior Invariants

For each milestone, explicitly state:

* **What must by default not change** (inputs/outputs, API contract, CLI output, file formats, model artifact formats, etc.)

* **How it will be proven** (tests, golden files, schema validation, snapshot checks)

### 2.2 No CI Weakening

Restoring green must by default not come from loosening checks unless explicitly approved and documented with rationale + compensating controls.

### 2.3 Evidence Over Opinion

Every enhancement adds either:

* a **hard gate** (CI failure on violation), or

* a **warn-first signal** that becomes a hard gate later (Scorecard is explicitly “warn-first”).

* * *

3) Baseline Tooling Standards

-----------------------------

### 3.1 Python Quality Gates (recommended baseline)

* Ruff for lint + format (single tool)

* mypy (strict posture)

* pydocstyle (Google convention)

* radon complexity gate (no CC grade > C)

* pytest + coverage (recommended targets: lines ≥ 85%, branches ≥ 80%, adjusted as appropriate to baseline)

* Hypothesis available (recommended; not required gate)

* Bandit (fail on ≥ HIGH)

* pip-audit strict

* Gitleaks secret scanning

* CycloneDX SBOM

These are listed as “core quality tools” and acceptance criteria.

### 3.2 Full-Stack Additions (conditional)

If a frontend exists:

* Vitest + Testing Library

* Playwright E2E + basic a11y checks (axe)

* PR deploy preview E2E gating (if Netlify or equivalent exists)

* CI artifacts: JUnit, coverage XML/HTML, Playwright reports/traces

* Concurrency guard per-branch to prevent overlapping runs/deploys

(These are captured in your TestingEnhancements + Enhancements docs.)

* * *

4) Audit & Supply-Chain Standards

---------------------------------

### 4.1 Governance References

* Map practices to **NIST SSDF SP 800-218** (include a short mapping table)

* If ML/AI code exists, include SSDF 800-218A profile note reference

* Use **OWASP ASVS v5.0 L2** as web/API reference when applicable

### 4.2 Supply Chain Requirements

* GitHub Dependency Review on PRs

* OpenSSF Scorecard (warn-first, non-blocking at first)

* SBOM generated and uploaded every CI run

* SLSA provenance attestations for built artifacts (when packageable)

* If Dockerfile exists: Trivy scan + Cosign keyless signing (conditional path)

(All of these are explicitly required by the audit enhancements doc.)

* * *

5) CI Evidence Requirements

---------------------------

Every CI run should produce:

1. **GitHub Actions job summary** (one-page “audit snapshot”)

2. **Machine-readable artifacts** uploaded (as applicable):
   
   * coverage.xml
   
   * bandit.json
   
   * pip_audit.json
   
   * gitleaks.json
   
   * SBOM (CycloneDX JSON)
   
   * Scorecard output / SARIF
   
   * radon output (or captured summary)

3. A **QA & Evidence** documentation page published (GitHub Pages or equivalent)

(Directly specified as “CI evidence & dashboards”.)

* * *

6) Phased Implementation Plan (Refactor-First)

----------------------------------------------

### Phase 0 — Pre-Flight + Baseline Guardrails (E2E)

**Goal:** detect constraints; add minimal safe wiring.

* Detect layout (tests/, package roots, Dockerfile)

* Add missing `.gitignore` patterns for common artifacts

* If zero tests: add a **sanity test** to prove CI wiring without inventing behavior

(Pre-flight and sanity-test requirements are called out explicitly.)

* * *

### Phase 1 — Local Dev Guardrails + Static Discipline (E2E)

**Goal:** tighten the developer loop without altering runtime behavior.

* Add/merge tool configs (do not clobber existing)

* Add pre-commit hooks (ruff, ruff-format, mypy, pydocstyle, local gitleaks)

* Add pinned dev tool lockfile using pip-tools (requirements-dev.in → requirements-dev.txt w/ hashes)

* Add CODEOWNERS if absent

(Exact examples and tool list come from audit enhancements.)

* * *

### Phase 2 — Tests + Coverage + Complexity Gates (E2E)

**Goal:** lock behavior and prevent silent regressions.

* Enable coverage gating (line threshold)

* Add branch coverage gating when feasible

* Add radon CC gate (> C fails)

* Make CI run: ruff, mypy, pydocstyle, radon, pytest+coverage

* Upload coverage artifact(s) and include summary output in job summary

(CI steps and thresholds are explicitly spelled out.)

* * *

### Phase 3 — Security + Supply Chain (E2E)

**Goal:** make refactoring safe to adopt and easy to audit.

* Bandit report (fail on HIGH)

* pip-audit strict JSON output

* gitleaks detect (repo scan)

* Dependency Review action (PR gate)

* Scorecard (warn-first, continue-on-error initially)

* SBOM generation (CycloneDX)

* Provenance attestation if buildable (wheel/sdist)

* Optional container scanning/signing if Dockerfile exists

(These jobs are enumerated in the audit doc.)

* * *

### Phase 4 — Docs + QA Evidence Publishing (E2E)

**Goal:** ensure the repo explains and proves itself.

* Publish minimal docs (Sphinx if Python-centric)

* Add a single **QA & Evidence** page linking to CI artifacts and describing how to reproduce locally

* Publish to GitHub Pages (or project-standard equivalent)

(“Docs & Evidence publish” + QA page requirement is explicit.)

* * *

7) Full-Stack Optional Track (Only When Applicable)

---------------------------------------------------

If the repo includes a frontend and deploy preview is possible:

* Vitest coverage thresholds (≥85 across key dimensions)

* Playwright E2E suite + a11y checks

* PR deploy preview → run Playwright against preview URL → gate merge on pass

* Main deploy hooks (Render/Netlify or project equivalents)

* CI artifacts: JUnit, coverage, Playwright traces/reports

* Concurrency guard

(From TestingEnhancements + Enhancements docs.)

* * *

8) Acceptance Criteria (Definition of Done)

-------------------------------------------

**Core (Python)**

* Ruff, mypy, pydocstyle pass

* Radon shows no CC grade > C

* Coverage meets threshold (minimum: lines ≥ 85; branches ≥ 80; prefer ≥85/85 where feasible)

* Bandit: no HIGH

* pip-audit strict passes

* gitleaks passes

* SBOM artifact uploaded each run

* If packageable: build artifacts + provenance attestation uploaded

* If Dockerfile exists: Trivy passes; Cosign signature created

* Job summary includes coverage totals + links to artifacts

* QA & Evidence page published

(These items are proposed as hard gates, subject to project constraints and client approval.)

**Full-stack (conditional)**

* Frontend coverage thresholds satisfied

* Playwright E2E passes (including deploy preview gating if enabled)

* Contract tests pass (Schemathesis or Pact where applicable)

* Trace/span IDs appear in logs and redaction tests pass

(From TestingEnhancements.)

* * *

9) Standard PR Hygiene

----------------------

* Conventional commits

* Prefer “one milestone = one PR”

* Keep changes minimally invasive: “add/merge without clobbering existing config”

* Avoid post-close commits; carry follow-ups into next milestone branch (per your workflow discipline)




