# TorchFuzz Bug Fix Workflow

## Overview

This skill provides a structured, step-by-step workflow for fixing bugs discovered by TorchFuzz - PyTorch's fuzzing infrastructure. It ensures thorough testing, proper validation, and clean PR submission using a test-driven approach where you first reproduce the fuzz-discovered bug with a failing test before implementing the fix.

## When to Use This Skill

Use this skill when:
- Fixing a bug discovered by TorchFuzz fuzzing
- Addressing a crash or assertion failure found during fuzzing
- Fixing numerical inconsistencies between eager and compiled execution detected by TorchFuzz
- Working on edge cases in tensor operations uncovered by fuzzing
- Fixing bugs in `torch.compile`, inductor, or dynamo discovered through fuzzing

## Prerequisites

> **IMPORTANT:** Before running any tests, ask the user:
> "Is there a conda environment name I need to activate before running tests?"

> **IMPORTANT:** Use `git` for source control commands, not `sl` (Sapling).

> **IMPORTANT:** Always activate conda environment before running `ghstack` (e.g., `conda activate <env_name> && ghstack`).

---

## TorchFuzz Bug Fix Workflow

### Step 0: Get the Failing Fuzz Test Information (MANDATORY)

> **IMPORTANT:** Before proceeding, ask the user:
> "Please provide one of the following:
> 1. **The seed** for the failing fuzz test, OR
> 2. **The fuzzed program** that is failing (the generated code/test case)"

**If the user provides a seed:**

1. Use the seed to regenerate and reproduce the issue
2. Run TorchFuzz with the specific seed using:
   ```bash
   python tools/experimental/torchfuzz/torchfuzz.py --seed <SEED>
   ``` 
   conda activat first. 
3. Capture the output and failing program for analysis
4. Proceed to Step 1 once you have the failing program and error details

**If the user provides the fuzzed program:**

1. Review the provided program to understand what operations and inputs are involved
2. Proceed directly to Step 1 to create a unit test from the failing program

---

### Step 1: Reproduce the Bug with a Failing Test (MANDATORY)

> **IMPORTANT:** Always be explicit about which step you are currently on (e.g., "**Step 1: Reproduce the Bug** - Creating unit test...").

1. Use the fuzz bug information provided (seed, operations, tensor shapes, etc.) to understand the issue
2. If no logs are provided, run the reproduction steps to capture the logs and identify the issue
3. Create a unit test that reproduces the exact bug using the fuzz-generated inputs
4. Run the test and **verify it FAILS with the same error** described in the fuzz report
5. **DO NOT proceed to Step 2 until the test fails as expected**
6. **I repeat: DO NOT proceed to Step 2 until the test fails as expected**
7. Add `@unittest.expectedFailure` decorator to the test, then run the test to ensure it fails
   
   > **Caveat:** Sometimes a test runs for both CPU and CUDA, but only one device fails. This can break CI because `@unittest.expectedFailure` is applied to both but only one actually fails. Ensure your test setup handles this case (e.g., skip the test on devices where it doesn't fail, or use device-specific expected failure decorators).

8. Run the test again and **verify it PASSES** (shows as "expected failure" or "xfail")
9. Give the user a command to run and ask them to verify the test fails as expected before proceeding
10. Ask the user if they want to create a PR with a repro for the failure using the unit test you generated. If so, use `ghstack` to create one.
11. Only after confirming the test passes with expectedFailure, proceed to Step 2


---

### Step 2: Fix the Bug

1. Only after Step 1 is complete, implement the fix
2. Run the test again and verify it now PASSES (make sure u removed @unittest.expectedFailure)!
3. Run any related tests to ensure no regressions
4. Perform these quality checks:

| Check | Question to Ask | Action if Yes |
|-------|-----------------|---------------|
| **Complexity** | What is the runtime complexity? Is this the most efficient way? | Find a more efficient approach |
| **DRY** | Am I duplicating the same code/logic? | Refactor to avoid repetition |
| **Early Exit** | Do I have deeply nested conditions or long prologue branches? | Refactor to use early exits/guard clauses |
| **Correctness** | Does this handle ALL cases and corner cases? | Ensure theoretical correctness |
| **Root Cause** | Am I fixing the problem or just avoiding it? | Fix the root cause, not a workaround |
| **Edge Cases** | Does this fix similar edge cases that fuzzing might find? | Consider broader fixes |

**After Step 2, write a summary that answers:**
- What was the root cause?
- What is the runtime complexity of the fix?
- Were there any refactoring opportunities (duplication, early exits)?
- What edge cases does the fix handle?
- Why is this fix correct?
- Are there other similar operations that might have the same bug?

---

### Step 3: Validate

1. **Confirm test behavior:** Make sure the test PASSES with changes and FAILS without changes
2. **Comments review:** Ensure comments are not too long, not too short, and easy to understand
3. **Fix lint issues:** Run `lintrunner -a` to automatically fix all lint issues caused by this PR
5. **Generate PR title and summary:** Create a clear, concise PR title and summary describing the fix

---

### Step 4: Ask User if They Are Satisfied

Ask the user if they are satisfied with the summary and fix, or if they have concerns and/or need more information. Work with them until they are ready. When they are ready, ask if you should proceed to the next step, which is publishing.

---

### Step 5: Publish

1. Create a new branch/commit and add the unit test to it (mark it as expected failure) (if you did not do that in step 0).
2. Create another branch/commit on top of that and add the fix to it (removing the expected failure)
   
   Include in the commit:
   - The fix summary (from Step 2)
   - PR Title
   - PR Summary
   - Reference to the TorchFuzz seed/configuration that found the bug

3. **Run `lintrunner -a` on each commit before pushing**
4. Ask the user if they want you to push the commits as PRs
5. Use `ghstack` to push the two commits as PRs

---

## Best Practices

✅ **DO** work through the steps one by one to avoid confusion
✅ **DO** always verify the test fails before implementing the fix
✅ **DO** include the fuzz seed and configuration in test comments
✅ **DO** run lint checks before each commit
✅ **DO** ask the user for confirmation before publishing
✅ **DO** consider if similar operations might have the same bug

## Common Mistakes to Avoid

❌ **DO NOT** skip the failing test step
❌ **DO NOT** proceed to the fix before verifying test failure
❌ **DO NOT** forget to activate the conda environment
❌ **DO NOT** use `sl` (Sapling) - use `git` instead
❌ **DO NOT** push commits without user confirmation
❌ **DO NOT** skip lint checks before pushing
