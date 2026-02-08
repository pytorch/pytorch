# M06 Tool Calls Log

## Session: 2026-02-08

---

### Step 1: Clean Baseline

| # | Tool | Purpose | Outcome |
|---|------|---------|---------|
| 1 | `git status --porcelain` | Check uncommitted changes | Found 5 modified files + untracked M02 folder |
| 2 | `git diff --stat` | Analyze changes | M03/M04/M05 files = line-ending only; M06_plan.md = real content |
| 3 | `git ls-tree HEAD docs/refactor/milestones/M02/` | Check if M02 tracked | Not tracked (empty output) |
| 4 | `grep M02 REFACTOR.md` | Check if M02 referenced | Referenced in closeout artifacts |
| 5 | `git restore` | Restore line-ending-only files | M03/M04/M05 files restored |
| 6 | `Remove-Item pr_body_m05.txt` | Delete local debris | Removed |
| 7 | `git add + commit` | Add M02 artifacts | 7ab7c20bb16 |
| 8 | `git add + commit` | Update M06 plan | 2ea6594df06 |
| 9 | `git status` | Verify clean baseline | Clean |
| 10 | `git rev-parse HEAD` | Record baseline commit | 2ea6594df06 saved to temp |
| 11 | `git checkout -b m06-action-pinning` | Create M06 branch | Branch created |

---

### Step 2: Generate Action Inventory (Before)

| # | Tool | Purpose | Outcome |
|---|------|---------|---------|
| 12 | `Get-ChildItem + Measure-Object` | Count workflow files | 143 files |
| 13 | `grep uses:` | Count uses: statements | ~1,600+ total |
| 14 | `Select-String + Get-Unique` | Extract unique action refs | 104 unique refs |
| 15 | PowerShell classification | Categorize by pinning type | 48 local, 21 SHA, 13 tag, 20 branch |
| 16 | `write` | Create inventory.before.md | Created |

---

### Step 3: Governance Checkpoint — SHA Resolution Blocked

| # | Tool | Purpose | Outcome |
|---|------|---------|---------|
| 17 | `web_search` (x5) | Resolve tag→SHA mappings | ❌ Tool malfunction (returned conversation context) |
| 18 | (manual) | Stop and report | Correct governance decision |

**Decision:** Split M06 into M06-A (external actions) and M06-B (PyTorch @main, deferred).

---

### Step 4: SHA Mappings Received

| # | Tool | Purpose | Outcome |
|---|------|---------|---------|
| 19 | User input | SHA mappings provided | 13 mappings verified (40-char hex, 6 cross-validated) |

---

### Step 5: Apply Pins Mechanically

| # | Tool | Purpose | Outcome |
|---|------|---------|---------|
| 20 | PowerShell + git | Pin actions/checkout@v4 | 18 files, commit e8069162844 |
| 21 | PowerShell + git | Pin actions/download-artifact | 10 files, commit 37e69f139b4 |
| 22 | PowerShell + git | Pin actions/setup-python | 3 files, commit 63f098311f3 |
| 23 | PowerShell + git | Pin actions/upload-artifact | 11 files, commit 50435672308 |
| 24 | PowerShell + git | Pin anthropics/claude-code-action@v1 | 1 file, commit 738fdfc8c13 |
| 25 | PowerShell + git | Pin aws-actions/configure-aws-credentials@v4 | 5 files, commit d3932a4c5e3 |
| 26 | PowerShell + git | Pin remaining third-party actions | 4 files, commit 0b1e7c6a38c |

---

### Step 6: Generate After-Inventory

| # | Tool | Purpose | Outcome |
|---|------|---------|---------|
| 27 | PowerShell | Count action categories | 26 SHA, 0 tag, 20 branch |
| 28 | write | Create after-inventory | action_uses_inventory.after.md |

---

### Step 7: Documentation

| # | Tool | Purpose | Outcome |
|---|------|---------|---------|
| 29 | write | Create M06_audit.md | Created |
| 30 | write | Create M06_summary.md | Created |
| 31 | (pending) | Update REFACTOR.md | Pending |

---

