# CI Runner Naming Guidelines

This document covers the naming standard for self-hosted CI runner labels in PyTorch PR reviews. Apply it whenever a PR adds new CI (a workflow, job, matrix or matrix entry) or changes the runner label of existing CI, or the prefix written in front of it.

Every self-hosted runner label that a PR adds or changes is checked against the standard. A non-conforming label that no other CI in this repo uses is a blocker; one that other CI already uses gets a short non-blocking heads-up. The canonical standard is [runner_naming_convention.md](https://github.com/pytorch/ci-infra/blob/main/osdc/docs/runner_naming_convention.md) in pytorch/ci-infra; the grammar and fields below are copied from it at commit ca6864a9 (last changed 2026-08-04), and the procedure is this repo's own.

## Which Labels This Covers

The rule covers every self-hosted label: OSDC ARC labels, legacy EC2 labels (`linux.*`, `windows.*`), partner fleets (ROCm, XPU, TPU, DGX, s390x), and the self-hosted macOS fleets (`macos-m1-stable`, `macos-m1-15`, `macos-m2-15`, `macos-m2-26`).

GitHub-hosted runners are exempt. These label families are GitHub-hosted:

- GitHub images: `ubuntu-*`, `windows-latest`, `windows-20*`, `windows-11-*`, `macos-latest*` and `macos-<digit>*`. The self-hosted macOS fleets start with `macos-m`, so they are not in these families.
- The org's GitHub-hosted larger runners `linux.24_04.<n>x`, such as `linux.24_04.4x`.
- Meta's enterprise GitHub-hosted larger runners `<n>-core-ubuntu*` and `<n>-core-windows*`, such as `8-core-ubuntu`, `4-core-ubuntu-24.04`, `2-core-ubuntu-arm` and `4-core-windows-gpu-t4`, and `windows-<n>-core*`, such as `windows-8-core`.

Every other label is self-hosted.

## The Naming Standard

### Grammar

`[c-]{provider}-[rel-]{os}-[b]{arch}{vendor}{features}-{vcpu}-{memory}[-{gpu_type}[-{gpu_count}]]`

Square brackets mark optional parts. Lowercase only and no leading zeros are this repo's reading of the grammar; the canonical doc does not state them.

### Fields

| Field | Required | Values | Rules |
|-------|----------|--------|-------|
| `c-` | No | `c-` | Canary / staging. Production labels omit it |
| provider | Yes | `mt` (Meta), `lf` (Linux Foundation), `am` (AMD), `in` (Intel), `nv` (NVIDIA), `ib` (IBM) | The organization that operates and funds the fleet. The canonical doc reserves every code but `mt` for provider-funded fleets; `lf` is in use anyway: this repo's runner determinator emits `lf-` and ci-infra deploys `lf-` clusters |
| `rel-` | No | `rel-` | Right after the provider. Release runner: dedicated `release-runners` group with node isolation |
| os | Yes | `l` Linux, `w` Windows, `m` macOS | |
| `b` | No | `b` | Directly before arch. Bare metal / dedicated instance: the job gets the whole node |
| arch | Yes | `x86`, `arm64` | |
| vendor | Yes | x86: `i` (Intel-style ISA), `a` (AMD-style ISA); arm64: `g2`, `g3`, `g4` (Graviton generation) | Bound to arch. ISA family or generation, not the silicon vendor: `i`-named AVX-512 runners commonly run on AMD c7a/r7a |
| features | x86 only | `avx2`, `avx512`, `amx` | Required on x86, forbidden on arm64 |
| vcpu | Yes | integer | Plain positive integer: no unit, no leading zero |
| memory | Yes | integer | GiB; same rules as vcpu |
| gpu_type | No | `t4`, `a10g`, `l4`, `a100`, `h100`, `b200` | |
| gpu_count | No | integer >= 2 | Only after gpu_type; omitted when the count is 1 |

The `rel-` marker is load-bearing. test-infra's release runner group tooling (`tools/scripts/release_manage_runner_groups.py`) allow-lists a workflow for the protected release runner group when its job text contains a `rel-` token. A missing marker means release jobs never get release runners; a stray one lets a non-release workflow onto the protected group.

### Length

For every label, the part after the `[c-]{provider}-` prefix is at most 37 characters. Staging deploys every runner with the 5-character `c-mt-` prefix, so this keeps the staging name within the standard's ~42 characters; ARC rejects names over 45.

### Examples

| Label | Verdict |
|-------|---------|
| `mt-l-x86iavx512-8-16` | Valid: Meta, Linux, x86 Intel-style AVX-512, 8 vCPU, 16 GiB |
| `mt-l-arm64g3-16-62` | Valid: arm64 Graviton 3 takes no features |
| `c-mt-l-x86iavx512-8-16` | Valid: canary |
| `mt-rel-l-x86iavx512-44-340` | Valid: release runner |
| `mt-l-bx86iamx-176-1800-h100-8` | Valid: bare metal with 8 H100 |
| `nv-l-x86aavx2-48-192-a10g-4` | Valid: NVIDIA-funded fleet with 4 A10G |
| `${{ needs.get-label-type.outputs.label-type }}l-x86iavx512-16-128` | Valid: both values the determinator yields, `mt-` and `lf-`, give conforming labels |
| `linux.2xlarge` | Invalid: legacy EC2 scheme |
| `mt-windows.4xlarge` | Invalid: ARC prefix on a legacy label |
| `l-mt-x86iavx512-8-16` | Invalid: the provider comes first |
| `mt-l-x86aavx2-11-41-a10g-1` | Invalid: a GPU count of 1 is written out |
| `mt-l-arm64g3avx512-16-62` | Invalid: features on arm64 |
| `mt-l-x86g3avx2-8-16` | Invalid: Graviton vendor on x86 |
| `mt-l-x86i-8-16` | Invalid: x86 without features |
| `mt-l-x86iavx512-8-16gb` | Invalid: unit on memory |
| `mt-l-x86iavx512-08-16` | Invalid: leading zero |
| `mt-l-x86iamx-22-225-gb300-4` | Invalid: GPU type not in the standard |
| `mt-l-x86iamx-44-450-h100-fab-2` | Invalid: `-fab` is not in the grammar; already used here, so a heads-up, not a blocker |
| `MT-L-X86IAVX512-8-16` | Invalid: uppercase |
| `aws-l-x86iavx512-8-16` | Invalid: unknown provider |
| `c-mt-rel-l-bx86iavx512-1000-10000-a100-1280` | Invalid: 38 characters after the provider |

### Full Form

The full form, as a regex. It does not encode length; apply the length rule separately.

```
^(c-)?(mt|lf|am|in|nv|ib)-(rel-)?[lwm]-b?(x86(i|a)(avx2|avx512|amx)|arm64g[234])-[1-9][0-9]*-[1-9][0-9]*(-(t4|a10g|l4|a100|h100|b200)(-([2-9]|[1-9][0-9]+))?)?$
```

### Hardware the Standard Cannot Express

Non-NVIDIA GPUs (ROCm, XPU), TPUs, NVIDIA Grace or other non-Graviton ARM hosts, Apple Silicon, arm64 CPU features, and the multi-node fabric `-fab` variant. A runner that needs one of these cannot conform.

## How Labels Are Written in This Repo

A label is a runner value in any of these forms:

- `runs-on:`, often a pass-through such as `${{ matrix.runner }}`; follow it to where the value is set.
- `runner:` entries inside `test-matrix: |` blocks, the largest group. YAML sees each block as one string, so only a text search finds these entries.
- `runner:`, `runs_on:`, `runner_label:` and `runner-type:` inputs, and reusable-workflow input defaults; a default counts as another CI's use only through callers that omitted that input before the PR, each with its own prefix, and never through callers the PR adds.
- `runner_prefix:`, prepended only to the `runner:` input of `_linux-build.yml`, the `runs_on:` input of `_binary-build-linux.yml` and `_binary-test-linux.yml`, and the two matrix labels of `_docs.yml`. Test-matrix rows never get it: the test workflows run them exactly as written.
- `echo "runner=..." >> "$GITHUB_OUTPUT"`, and the `x86=` and `arm64=` outputs of `_select-release-runner.yml`.
- Dispatch inputs: the fallback in `${{ github.event.inputs.runner || '<label>' }}`, and the `default:` and `options:` values of a `workflow_dispatch` input whose value a runner value uses as its label, read through `github.event.inputs.<name>` or `inputs.<name>`.
- Ternaries: `${{ cond && '<a>' || '<b>' }}`. Each branch is a label.
- `.github/templates/*.j2`: `runs-on:` and `{%- set ... %}` assignments such as `_runs_on`. Each generated workflow, `.github/workflows/generated-*.yml`, copies its template's labels.
- `.github/scripts/generate_ci_workflows.py`, which sets the macOS labels.

Prefix sources:

- The runner determinator: the outputs `label-type`, `amd-sandbox-label-type`, `amd-dpx-label-type` and `scale-config-label-type` of a job whose `uses:` is exactly `pytorch/pytorch/.github/workflows/_runner-determinator.yml@main` (`@release/<X.Y>` on release branches) or `./.github/workflows/_runner-determinator.yml`, such as `${{ needs.get-label-type.outputs.label-type }}`. From any other job, these outputs are an unknown prefix scheme. `label-type` yields `mt-`, the default and error fallback, or `lf-` (`.github/scripts/runner_determinator.py`); in a workflow that passes `opt_out_experiments: lf` to the determinator, it yields only `mt-`.
- A literal `runner_prefix: "mt-"` or `runner_prefix: ""`.
- `${{ inputs.runner_prefix }}` resolves to each caller's `runner_prefix:` value, or to the input default for a caller that omits it; each value gives a separate label. Both callers of `_docs.yml` pass the determinator.
- `scale-config-label-type` yields an empty string, `wincanary.` or `wincanarylf.` in front of `windows.*` labels.
- `amd-sandbox-label-type` and `amd-dpx-label-type` yield an empty string, or `amd-sandbox-` and `amd-dpx-`, in front of `linux.rocm.*` labels.
- `rel-` is never a prefix output. It is a literal chosen by branch or tag: `_select-release-runner.yml`, the templates' `release_runner_if`, and ternaries such as those in `build-vllm-wheel.yml`.
- `c-` is not used in this repo.

## Checking a Label

1. **Resolve the effective labels.** Check every label that new CI runs on (a new workflow, job, matrix or matrix entry such as a test-matrix row or strategy-matrix entry, or a generated or templated job), and every runner label or prefix the PR changes on existing CI. A label counts only if the PR raises the number of places that request it. A place is a job, a job's test-matrix `config:` value, or a strategy-matrix entry; compare the places before and after the PR, with effective labels resolved on both sides, counting only runner values in the forms above (lines in `.github/actionlint.yaml`, `.github/arc.yaml`, docs or test fixtures never count). Deleting a runner input (`runner:`, `runs_on:`), a `runner_prefix:` or a matrix key from existing CI moves that place to the label it now falls back to. Skip a label the PR only re-touches, such as rows whose shard count changed, new shard rows of a config that already runs on that label, or a job moved to another file; a new config, job or matrix, and a prefix flip (the effective label changes), still count. New CI's labels include the ones it gets without writing them: the input defaults of a reusable workflow for runner inputs it omits (a new `_linux-build.yml` caller without `runner_prefix:` and `runner:` runs on `""` plus the `runner` default, a bare label), the fallback a runner value uses when a new matrix entry omits its key (as in `${{ matrix.runner_label || '<label>' }}`), and the `default:` and `options:` values of a `workflow_dispatch` input whose value a runner value uses as its label, through `github.event.inputs.<name>` or `inputs.<name>`. For each value, compose the prefix written before it: a literal, a determinator expression, or a `runner_prefix:` value. Every ternary branch, every default, and every value a prefix expression can produce (see Prefix sources) is a separate label. A change to only the prefix creates a new effective label: adding `runner_prefix: "mt-"` in front of `linux.s390x` requests `mt-linux.s390x`. A PR that changes a prefix value itself (a `runner_prefix:` literal or input default, or the prefixes the determinator emits) must leave a valid `[c-]{provider}-` part, or the empty prefix that legacy labels use; check every label it is prepended to.
2. **Skip GitHub-hosted labels.**
3. **Validate.** Check each label against the length rule and:
   - literal, or with a literal prefix: the whole label against the full form;
   - prefixed by a prefix expression: each value the expression yields (see Prefix sources), as a literal prefix, plus the rest against the full form; if the PR changes what the determinator emits (`.github/scripts/runner_determinator.py` or `.github/workflows/_runner-determinator.yml`), use the values the PR leaves there;
   - anything else (legacy dot labels, partner labels, unknown prefix schemes) does not conform.

   A literal provider after a prefix expression, as in `${{ needs.get-label-type.outputs.label-type }}mt-l-x86iavx512-16-128`, does not conform. A label that conforms needs no finding: stop. Run steps 4-8 only for a label that does not conform, to decide whether other CI in this repo already uses it.
4. **Take the core.** Drop either a leading prefix expression or a literal prefix (`c-mt-`, `mt-`, `c-lf-`, `lf-`, `amd-sandbox-`, `amd-dpx-`, `wincanary.` or `wincanarylf.`), never both, and keep `rel-` and everything after it: `${{ needs.get-label-type.outputs.amd-dpx-label-type }}linux.rocm.gpu.gfx950.1` has the core `linux.rocm.gpu.gfx950.1`, and `mt-linux.s390x` has the core `linux.s390x`. A core with any character outside `[A-Za-z0-9._-]` is invalid: report it as a blocker, without searching.
5. **Search.** Use the Grep tool with path set to the PR checkout's `.github` directory (never the repo root: too slow), output mode `content` with line numbers, glob `*.{yml,yaml,j2,py}`, and this pattern, with each `.` in CORE escaped as `\.`:

   ```
   (^|[^A-Za-z0-9_.-])((c-)?(mt|lf)-|amd-(sandbox|dpx)-|wincanary(lf)?\.)?CORE([^A-Za-z0-9_.-]|$)
   ```

   Search all of a PR's labels in one alternation, for example `(^|[^A-Za-z0-9_.-])((c-)?(mt|lf)-|amd-(sandbox|dpx)-|wincanary(lf)?\.)?(rel-l-x86iavx512-44-340|l-arm64g3-61-463|linux\.s390x)([^A-Za-z0-9_.-]|$)`. Keep the boundaries: a plain substring or `\b` search matches `l-x86iamx-22-225` inside `l-x86iamx-22-225-h100`, and `l-x86iavx512-44-340` inside `rel-l-x86iavx512-44-340`, so a new label looks existing. If a Grep result is truncated or cut short, re-run it narrower (fewer labels, or output mode `files_with_matches` first); never conclude that a label is new from a partial result.
6. **Keep only prior uses.** A prior use is a runner value in one of the forms above, not a comment or doc, in `.github/workflows/`, `.github/templates/`, `.github/actions/` or `.github/scripts/generate_ci_workflows.py`. It is either:
   - a deleted (`-`) diff line that used the label; look for it in the diff itself, since a search of the checkout cannot find deleted lines; or
   - a hit on a line the diff does not show as added (`+`). A line of a changed file outside every hunk's new-side range (`@@ ... +c,d @@`) is unchanged, so a pure rename leaves every line unchanged; but treat every line of a file whose diff entry reads `Binary files ... differ` as added.

   Lines in `.github/actionlint.yaml` (a lint allowlist), `.github/arc.yaml` (a reference mapping that no code reads), `*.md` files or test fixtures never count. To save time, check hits in files the PR does not change first (`/tmp/pr-files.txt` in the In Progress PR Review, or the PR's changed-file list), and stop at the first one that passes steps 6 and 7.
7. **Match the prefix.** Resolve each prior use's prefix as in step 1, as it stood before the PR: when the prefix is set on another line, such as the job's `runner_prefix:`, read it there (test-matrix rows never get one), and if the PR changed that line, use its `-` version from the diff. Each label from step 1 needs a prior use with the same core whose prefix gives the same prefix value. No prefix means the literal `""`, which matches any expression that can produce `""`. A literal prefix matches a prefix expression when the literal is one of the expression's values, in either direction: a literal `mt-` prior use covers `mt-<core>` but not the `lf-<core>` that a determinator prefix also requests; a determinator prior use covers a literal `mt-` label, and a literal `lf-` one unless its workflow opts out of `lf`. Any other prefix matches only itself.
8. **Decide.** No prior use left means no other CI in this repo uses the label: it is new, and a blocker. Otherwise other CI already uses it, and it gets a heads-up.

## What to Report

Anchor a finding on the label line, or on the changed prefix line or determinator line when only the prefix changed, or, for a label new CI gets from an input or matrix key it omits, on the new job's `uses:` line or the new matrix entry; if the label repeats, on its first added occurrence. For a label existing CI falls back to after the PR deletes an input, prefix or matrix key, anchor on the new-side line right after the deletion, inside the hunk that removed it. In /pr-review, file findings under Infrastructure. When one line yields labels in both tiers, such as a determinator prefix whose `mt-` label other CI uses while its `lf-` label is new, report only the blocker for that line and leave its already-used label out of the heads-up.

- **Blocker: no other CI in this repo uses the label.** One finding per distinct label; the labels one line yields through a prefix expression, such as `mt-X` and `lf-X`, are one finding. Always Request Changes. Name the label and the field or fields that break the standard, and suggest a conforming name: describe its shape from the intended hardware, or name a conforming label already used in this repo with the same resources. The runner must be defined under that name in pytorch/ci-infra. For hardware the standard cannot express, say the canonical doc must be extended first. State the consequence, so the finding is not mistaken for a naming preference: every runner introduced in this repo must follow the standard, and maintainers send back PRs that do not. Tooling keys off the structure: the `rel-` marker and the length limit above, and the `l-` segment, which HUD queue-time queries and queue alerts use to treat a job as ARC.
- **Heads-up: other CI in this repo already uses the label.** One short finding for the whole PR that lists every such label and the field or fields each breaks, anchored on the first one's line. It does not block approval; it is a deliberate exception to pr-review's "Everything is a must-fix".
