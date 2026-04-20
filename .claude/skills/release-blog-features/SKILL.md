---
name: release-blog-features
description: Produce a draft "Features" section for a PyTorch release blog post by comparing two release branches (e.g., release/2.11 vs release/2.12). Use when the user asks to generate a release blog, list new features for a release, compare release branches, or draft content for pytorch.org/blog. Organizes features similar to https://pytorch.org/blog/pytorch-2-11-release-blog/ with Highlights and per-component sections (Dynamo, Inductor, Distributed, Export, MPS, ROCm, XPU, CPU, etc.), each tagged with stability (Stable / Beta / Prototype / API-Unstable).
---

# PyTorch Release Blog Features Skill

Produce a draft list of features for a PyTorch release blog post by diffing
two release branches and organizing the noteworthy changes into the format
used on pytorch.org/blog.

## Usage

The user invokes the skill with two release branches (a "base" and a
"target"), for example:

```
Generate features for the 2.12 release blog (release/2.11 → release/2.12)
```

Default behavior when the user only gives the target version:
- base = `release/<previous-minor>` (e.g. `release/2.11` for target 2.12)
- target = `release/<target>` (e.g. `release/2.12`)

Always confirm the two refs you'll compare before doing any heavy work.

## What to produce

A Markdown document with the same structure as recent release blogs
(see [reference-structure.md](reference-structure.md) for a template and an
example based on the 2.11 blog). At a high level:

1. **Header** — version, link to release notes, commit/contributor stats.
2. **Highlights** — 5-10 bullet points of the most notable features.
3. **Feature sections** — one subsection per feature, tagged with
   `[Stable]`, `[Beta]`, `[Prototype]`, or `[API-Unstable]`, and grouped
   by component (Dynamo, Inductor, Distributed, Export, Quantization,
   MPS, ROCm, XPU, CPU, Arm, Release Engineering, etc.).
4. **Deprecations / BC-breaking** — only if any are found.
5. **Non-feature updates** — version bumps (CUDA default, Python min),
   release-cadence changes, etc.

Each feature entry should contain:
- Short title and stability tag
- 1-3 sentence description written for a blog audience (not a commit log)
- Links to representative PRs (`#12345`) and docs where relevant
- Platform/backend notes if applicable

Do **not** invent features, metrics, or API names. If something is
ambiguous, mark it with `TODO:` so the release manager can resolve it.

## Workflow

### Step 1 — Resolve the refs

```bash
# Make sure we have both release branches locally
git fetch origin release/<base> release/<target> --no-tags

# Determine the merge-base — this is the effective "cut point"
git merge-base origin/release/<base> origin/release/<target>
```

Prefer `origin/release/<x>` over bare `release/<x>` when fetching, and use
the actual tag (e.g. `v2.11.0`) over the branch when it exists, because
the branch may contain post-release cherry-picks.

### Step 2 — Collect the commit list

Two viable options — pick based on what is available:

**Option A: reuse existing tooling (preferred if `scripts/release_notes/`
already has a commitlist for this release).**

```bash
cd scripts/release_notes
python commitlist.py --create_new v<base>.0 <target-commit-hash>
# Produces results/commitlist.csv with labels, categories, files changed.
```

See `scripts/release_notes/README.md` for details on cherry-pick handling
and the classifier.

**Option B: lightweight `git log` + `gh` lookup** (use when the tooling is
not set up or when a quick draft is enough):

```bash
# Titles + PR numbers, one per commit
git log --pretty=format:'%H %s' \
    origin/release/<base>..origin/release/<target> \
    > /tmp/release-commits.txt

# For any commit, pull the PR metadata including labels
gh pr view <num> --json number,title,labels,body,author,files
```

### Step 3 — Filter to "blog-worthy" changes

A release has thousands of commits. The blog only highlights a small
fraction. Use these signals, in order:

1. **`release notes:` labels** — PRs labelled `release notes: <area>` are
   the canonical source of user-facing changes. Prefer these.
2. **"New feature" / "Beta" / "Prototype" labels** and wording in the PR
   body (e.g. `## Summary` mentions a new public API).
3. **New public APIs** — PRs that add to `torch/`, `torch/nn/`,
   `torch/distributed/`, `torch/export/`, etc., especially new `.py`
   modules or new entries in `__all__`.
4. **Dashboard / perf callouts** — PRs referencing the PyTorch HUD or
   numerical speedups.
5. **Platform enablement** — new backends, new hardware, new wheel
   variants (ROCm, XPU, CUDA major bump, aarch64).

Ignore: pure refactors, test infra, lint fixes, typo fixes, internal
dispatcher churn, reverts, and cherry-picks that land on both branches.

### Step 4 — Categorize

Group the survivors into components. The canonical set used by recent
blogs (use these exact names when they apply):

- **Dynamo / torch.compile**
- **Inductor**
- **Distributed** (FSDP, DTensor, DeviceMesh, symmetric memory,
  pipelining, collectives)
- **Export** (`torch.export`, AOTInductor)
- **Quantization**
- **Profiler**
- **ONNX**
- **NN / Frontend** (FlexAttention, SDPA, new modules)
- **MPS** (Apple Silicon)
- **ROCm** (AMD)
- **XPU** (Intel)
- **CPU / Arm**
- **CUDA** (default version change, cuDNN, CUTLASS integrations)
- **libtorch / C++ ABI** (`torch::stable`)
- **Release Engineering** (wheel variants, Python/CUDA support matrix)

Tag each entry with a stability level. If the PR body explicitly says
"prototype" or "experimental", use `[Prototype]` / `[API-Unstable]`.
Otherwise, infer from the PR's release-notes label and from whether the
API is gated behind a private namespace (`torch._*`).

### Step 5 — Draft the blog

Open [reference-structure.md](reference-structure.md) and follow the
template. Keep descriptions blog-voice, not commit-voice: write for a
PyTorch user deciding whether to upgrade, not for a reviewer.

### Step 6 — Gather stats for the header

```bash
# Commit count between the two refs (excludes merges)
git log --no-merges --oneline \
    origin/release/<base>..origin/release/<target> | wc -l

# Unique contributors
git log --format='%an' \
    origin/release/<base>..origin/release/<target> | sort -u | wc -l
```

## Output

Save the draft to `agent_space/release-<version>-blog-draft.md` (per
CLAUDE.md, `agent_space/` is the git-ignored scratch directory). Do not
commit the draft — the release manager will review and move it into the
pytorch.org blog repo.

## Best practices

- **Confirm scope before you dig in.** Thousands of commits; the user
  cares about ~30 features.
- **Let the PR author's words do the work.** Lift phrasing from the PR
  body's "Summary" section rather than re-describing the change from the
  diff.
- **Flag uncertainty.** `TODO(release-manager): is this beta or
  prototype?` is more useful than a confident guess.
- **Don't overclaim performance.** Only cite speedups that appear in the
  PR body or linked dashboard — never invent numbers.
- **Deduplicate ghstack.** A feature often lands as 5-15 ghstack PRs.
  Collapse them to one blog entry citing the user-facing PR (usually the
  top of the stack, which adds the public API or docs).
- **Skip reverts and follow-ups.** If PR #A landed and PR #B reverted it
  before the release branch was cut, neither belongs in the blog.
