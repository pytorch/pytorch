According to a document from (2026-02-08), the next milestone after **M06 (Pin Actions)** in your Phase 1 “CI Health” sequence is **M07 — Add Dependabot**, and it explicitly depends on M06.

Also on your earlier question: **it’s unlikely this whole refactoring program lands in “official PyTorch releases” wholesale**, but **individual, tightly-scoped, low-risk improvements** (like CI hardening / supply-chain guardrails) *can* be upstreamed as discrete PRs if they’re maintainers-aligned and evidence-backed. Your milestone design (small, reversible, audit-heavy) is exactly the shape that *could* get accepted upstream.

---

## M07 Plan for Cursor: Dependabot for GitHub Actions Updates

### Milestone Identity

* **Milestone:** M07 — Add Dependabot (for action updates)
* **Phase:** Phase 1 — CI Health
* **Change Class:** Verification / Maintenance Automation (config-only)
* **Risk:** Low
* **Dependency:** M06 complete (action pins exist)

### Intent

Enable **Dependabot version updates** for the **`github-actions`** ecosystem so pinned actions can be updated through reviewable PRs (i.e., “automation for maintaining the pins”). This is explicitly called out as the follow-on mitigation after action pinning.

### Scope Boundaries

**In scope**

* Add `.github/dependabot.yml` enabling **version updates for `github-actions`**
* Keep it **minimal**: one ecosystem, one directory (`/`), conservative schedule
* Document behavior + evidence constraints in milestone artifacts

**Out of scope**

* No action upgrades by hand
* No workflow logic edits
* No pin-format changes (keep full SHAs where already pinned by M06)
* No SBOM work (M08) / third-party audit scripting (M09)

### Invariants

* **INV-080 (Action Immutability)** remains enforced from M06 (don’t reintroduce `@main` / mutable tags).
* Introduce **INV-090 (Action Update Channel Exists)**: “There is a repo-native automated mechanism that proposes updates to GitHub Actions dependencies via PRs.” (Observational until we see it operate on the repo.)

### Evidence Constraints (pre-commit honesty)

* You can validate the **YAML file presence + structure** locally, but **you cannot prove Dependabot execution** without GitHub-side scheduling/permissions and time elapsing.
* Therefore:

  * **Proof Type A:** “Config exists, minimal, conservative, reviewed.”
  * **Proof Type B (Deferred):** “Dependabot opened at least one PR (or shows as enabled in Insights/Security UI).”

### Deliverables

**Code/config**

* `.github/dependabot.yml` (new)

**Docs**

* `docs/refactor/milestones/M07/M07_toolcalls.md`
* `docs/refactor/milestones/M07/M07_audit.md`
* `docs/refactor/milestones/M07/M07_summary.md`
* `REFACTOR.md` updated with M07 entry + score/progress + any deferrals (as your governance requires)

### Implementation Steps (Cursor execution checklist)

#### 0) Recovery + baseline hygiene

1. Confirm clean working tree.
2. Create branch: `m07-dependabot-actions`
3. Create `docs/refactor/milestones/M07/` folder and start `M07_toolcalls.md`.
4. Log the exact starting commit SHA and branch name in toolcalls.

#### 1) Add Dependabot config (minimal, conservative)

Create `.github/dependabot.yml` using **Dependabot v2 config** for `github-actions`. GitHub docs: ecosystem `github-actions`, directory `/`, schedule block required. ([GitHub Docs][1])

Suggested baseline config (Cursor can paste verbatim):

```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "ci"
      - "refactor-program"
    commit-message:
      prefix: "deps(actions)"
    # Optional noise control while M06-B remains deferred:
    ignore:
      - dependency-name: "pytorch/pytorch"
      - dependency-name: "pytorch/test-infra"
```

Notes:

* The `ignore` block is optional, but it cleanly prevents churn on the **PyTorch-owned reusable workflows** until you explicitly do M06-B. (M06-B being the “pin internal @main actions” work you deferred.)

#### 2) Local validation (truthful)

* Run `actionlint` on workflows (M05 installed the pattern; actionlint won’t validate dependabot, but it ensures you didn’t accidentally break workflows while touching repo config).
* Sanity check Dependabot YAML formatting (basic): ensure file exists, `version: 2`, `updates:` list present.

Record results in `M07_audit.md` as “structural validation” + explicitly call out the “runtime proof deferred” constraint.

#### 3) Documentation pack (M07_audit + M07_summary)

**M07_audit.md should include**

* Exact diff (file added)
* Config rationale (why weekly, why PR limit, why labels)
* Evidence constraints & deferral entry: *“M07-V01: Dependabot runtime behavior unobservable locally; verify post-merge via GitHub UI/PR arrival.”*

**M07_summary.md should include**

* What changed, why, how it reduces risk
* How it connects to M06 (maintenance automation for pinned SHAs)

#### 4) Update REFACTOR.md (in-PR)

* Add M07 milestone entry
* Update “Milestone Progress” and score trend
* Add Active Risk / Deferral line for M07-V01 if you’re keeping a registry

#### 5) PR + merge gate

* Push branch
* Open PR against `main` (fork)
* Stop for explicit merge approval per your governance

### Rollback Plan

* Single revert commit removing `.github/dependabot.yml` (and reverting docs updates) restores prior behavior.

### “Done means”

* `.github/dependabot.yml` exists and is minimal + reviewed
* Milestone docs created (toolcalls/audit/summary) + REFACTOR.md updated
* Deferral recorded for GitHub-side runtime confirmation (if not yet observed)

---

If you want the plan to be even more “Cursor-native,” tell Cursor: *“Implement exactly what’s in the ‘Suggested baseline config’ block, then produce the three milestone docs and REFACTOR.md update, then stop at merge permission.”*

[1]: https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/keeping-your-actions-up-to-date-with-dependabot?utm_source=chatgpt.com "Keeping your actions up to date with Dependabot"
