

Updated `summaryprompt.md` (Refactoring Projects)
-------------------------------------------------

You are generating a **Comprehensive Milestone Summary** for a **refactoring program**.

This summary is a **canonical project artifact**, not a retrospective narrative.  
It must by default be suitable for:

* governance records

* phase reviews

* future audits

* onboarding new contributors

* anchoring subsequent milestones

* demonstrating **safe refactoring evidence** to external reviewers

The tone must by default be:

* neutral

* factual

* evidence-based

* non-promotional

Avoid speculation; mark UNKNOWN explicitly when evidence is insufficient. 
Do NOT infer intent unless it is explicitly documented.  
Do NOT include future plans beyond what is explicitly authorized by this milestone.

**Refactoring posture:** The default assumption is **behavior preservation** unless the milestone explicitly authorized behavioral change.

* * *

REQUIRED INPUTS (Infer from context if not explicitly provided)
---------------------------------------------------------------

Identify and record:

* Project name

* Milestone identifier (ID + name)

* Phase (if applicable)

* Date range

* Status (Open / Closed / Blocked / Rolled Back)

* Baseline reference (prior milestone, phase exit, tag, or commit)

* Scope boundaries (in-scope vs out-of-scope)

* **Refactor target surface** (what was being refactored: module, subsystem, API, repo boundary, pipeline)

* **Invariants** (explicit “must by default not change” surfaces, if stated)

* **Evidence sources** (CI runs, artifact links, logs, reports)

If any required input cannot be determined, explicitly mark it as **UNKNOWN**.

* * *

OUTPUT FORMAT (STRICT — follow exactly)
---------------------------------------

📌 Milestone Summary — :
========================

**Project:**  
**Phase:**  
**Milestone:** <ID + descriptive name>  
**Timeframe:** <start date → end date>  
**Status:** <Closed / Open / Blocked / Rolled Back>  
**Baseline:** <commit / tag / prior milestone or UNKNOWN>  
**Refactor Posture:** <Behavior-Preserving / Behavior-Changing / Mixed / UNKNOWN>

* * *

1. Milestone Objective

----------------------

State **why this milestone existed** in refactoring terms.

* Identify the specific risk, coupling, drift, debt, or boundary problem it addressed

* Tie the objective to a baseline or governance need

* Keep this section short and precise

> Answer: “What would remain unsafe, brittle, or ungoverned if this refactor did not occur?”

* * *

2. Scope Definition

-------------------

### In Scope

Explicitly list:

* components / modules touched

* entrypoints affected (CLI/API/library)

* contracts / schemas / interfaces involved

* CI workflows or gates impacted

* documentation artifacts updated

### Out of Scope

Explicitly list:

* areas intentionally untouched

* features explicitly not added

* performance work not attempted (unless authorized)

* dependency upgrades excluded (unless authorized)

* “nice-to-have” cleanup deferred

If scope changed during execution, document **when and why**.

* * *

3. Refactor Classification

--------------------------

Classify the work with a bias toward auditability.

### Change Type

Mark one:

* **Mechanical refactor** (rename, move, extract helpers, reorganize files)

* **Boundary refactor** (package split, adapter introduction, API surface isolation)

* **Semantic refactor** (logic change without intended behavior change)

* **Behavior change** (intended externally observable change)

### Observability

State what could be externally observed:

* API responses, CLI output, model outputs, file formats, integration behavior, performance, etc.

If unknown, mark **UNKNOWN** and explain why.

* * *

4. Work Executed

----------------

Summarize what actually happened (not the plan).

Include:

* key actions (extraction, decomposition, module split, adapter insertion, contract hardening)

* counts where meaningful (files changed, modules added/moved, new tests)

* any migration steps (old path preserved, shim added, call sites updated)

* explicit note if **no functional logic changed** (when true)

Avoid implementation trivia unless it impacts risk, governance, or reversibility.

* * *

5. Invariants & Compatibility

-----------------------------

This section is mandatory for refactoring projects.

### Declared Invariants (must by default Not Change)

List invariants explicitly, such as:

* public API contract

* CLI output shape

* deterministic outputs / golden files

* schema compatibility

* integration call patterns

* model artifact formats

If none were declared, state:

* “No invariants were explicitly declared; refactor assumed behavior preservation.”

### Compatibility Notes

* backward compatibility preserved? (Yes/No/Unknown)

* breaking changes introduced? (Yes/No/Unknown)

* deprecations introduced? (Yes/No/Unknown)

* * *

6. Validation & Evidence

------------------------

Describe **how correctness and non-regression were verified**.

Include:

* tests run (CI + local + integration)

* coverage impact (if known)

* contract/schema validation

* golden file comparisons / snapshot tests

* lint/type/security gates invoked

* failures encountered and how they were resolved

* evidence that validation is meaningful (not just “green”)

If validation is incomplete, explicitly state what is missing and why.

Prefer a table:

| Evidence Type | Tool/Workflow | Result | Notes |
| ------------- | ------------- | ------ | ----- |

* * *

7. CI / Automation Impact

-------------------------

Document the milestone’s interaction with automation.

Include:

* workflows affected

* checks added/removed/reclassified

* enforcement changes (stricter/looser/unchanged)

* any signal drift observed (false green, missing coverage, flaky tests)

State whether CI:

* blocked incorrect changes

* validated correct changes

* failed to observe relevant risk

* * *

8. Issues, Exceptions, and Guardrails

-------------------------------------

List all notable issues encountered.

For each:

* description

* root cause (if known)

* resolution status (resolved / deferred / unchanged)

* tracking reference (issue ID, registry entry, doc)

* guardrail added (if applicable)

If no issues occurred, explicitly state:

> “No new issues were introduced during this milestone.”

* * *

9. Deferred Work

----------------

Enumerate deferred items surfaced or touched.

For each deferred item:

* what was deferred

* why it was deferred

* whether it pre-existed the milestone

* whether its status changed

This section must by default not introduce new, untracked debt.

* * *

10. Governance Outcomes

-----------------------

State what changed in governance posture.

Examples:

* invariants became explicit

* interfaces became contract-validated

* boundaries clarified

* CI truthfulness improved

* risk isolated via adapters / shims

> Answer: “What is now provably true that was not provably true before?”

* * *

11. Exit Criteria Evaluation

----------------------------

Evaluate against success criteria.

For each criterion:

* Met / Partially Met / Not Met

* evidence or rationale

If criteria were adjusted, document the change and justification.

* * *

12. Final Verdict

-----------------

Provide a concise, authoritative conclusion.

Examples:

* “Milestone objectives met. Refactor verified safe. Proceed.”

* “Milestone partially complete; missing evidence prevents closure.”

* “Milestone invalidated; rollback recommended.”

Unambiguous.

* * *

13. Authorized Next Step

------------------------

Document only what is explicitly authorized:

* next milestone(s), phase transition, or pause

* constraints or conditions on proceeding

If no next step is authorized, state that explicitly.

* * *

14. Canonical References

------------------------

List authoritative references sufficient for reconstruction:

* commits (SHAs)

* pull requests

* CI run URLs

* documents

* audit artifacts

* issue tracker entries

* * *

FORMATTING RULES
----------------

* Use markdown

* Prefer tables for inventories and comparisons

* Use bullet points over prose

* Mark UNKNOWN explicitly

* Do not embed opinions or future planning unless authorized

* * *

COMPLETION RULE
---------------

Stop after generating the summary.  
Do not propose improvements or additional work unless explicitly requested.

* * *


