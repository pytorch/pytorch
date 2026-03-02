# Documentation Preview Notification - Implementation Summary

## What Was Created

A GitHub Action workflow that automatically notifies PR authors when their documentation preview links are ready.

### Files Created

1. **`.github/workflows/doc-preview-notification.yml`** - The main workflow file
2. **`.github/workflows/README-doc-preview-notification.md`** - Documentation explaining how it works

## How It Works

### The Problem
- Documentation previews take time to build and upload to S3
- Users don't know when the preview link is actually ready
- The link is posted early but doesn't work until build completes

### The Solution
This workflow:

1. **Triggers** after the main "pull" workflow completes successfully
2. **Waits and polls** the preview URL (checking every 10 seconds for up to 10 minutes)
3. **Posts a comment** on the PR when the docs are ready with clickable links
4. **Avoids duplicates** by updating existing comments if they exist

## Key Features

### ✅ Intelligent Polling
- Checks the preview URL every 30 seconds
- Waits up to 40 minutes (configurable)
- Uses HTTP status codes to detect when docs are ready

### ✅ Supports Both Doc Types
- Checks Python docs first
- Also checks for C++ docs if available
- Posts links to all available documentation

### ✅ User-Friendly Comments
Example comment when ready:
```markdown
## 📄 Documentation preview is ready! 🎉

Your documentation changes have been built and are ready for review:

- **Python docs**: https://docs-preview.pytorch.org/pytorch/pytorch/12345/index.html
- **C++ docs**: https://docs-preview.pytorch.org/pytorch/pytorch/12345/cppdocs/index.html

---
This preview will be available for 14 days.
```

### ✅ Handles Edge Cases
- No duplicate comments (updates existing ones)
- Graceful timeout handling
- Clear error messages if preview doesn't appear

## Technical Details

### Workflow Trigger
```yaml
workflow_run:
  workflows: ["pull"]
  types:
    - completed
```

### Preview URL Pattern
Based on the `_docs.yml` workflow, docs are uploaded to:
- **S3**: `s3://doc-previews/pytorch/pytorch/{PR_NUMBER}/`
- **Public URL**: `https://docs-preview.pytorch.org/pytorch/pytorch/{PR_NUMBER}/index.html`

### Configuration Variables

```bash
MAX_ATTEMPTS=80   # Number of polling attempts
SLEEP_TIME=30      # Seconds between attempts
# Total wait time = 180 × 10 = 1800 seconds (30 minutes)
```

## Integration with Existing Workflow

This workflow integrates seamlessly with the existing PyTorch CI/CD:

1. **`_docs.yml`** (lines 190-208) uploads docs to S3:
   - Python docs: `s3://doc-previews/pytorch/pytorch/{PR_NUMBER}/`
   - C++ docs: `s3://doc-previews/pytorch/pytorch/{PR_NUMBER}/cppdocs/`

2. **This workflow** waits for those uploads to be accessible via HTTPS

3. **Notifies users** exactly when the links become clickable

## Why This Approach?

### Alternative Approaches Considered:

1. **❌ Comment immediately after upload**: Users would click broken links
2. **❌ S3 event notification**: Complex infrastructure changes required
3. **✅ Poll the public URL**: Simple, reliable, no infrastructure changes needed

### Advantages:

- ✅ No changes to existing build infrastructure
- ✅ Works with current S3 upload mechanism
- ✅ Easy to maintain and debug
- ✅ Clear logging in GitHub Actions
- ✅ Configurable timeout and polling intervals

## Testing Recommendations

To test this workflow:

1. **Create a test PR** with documentation changes
2. **Monitor the Actions tab** to see:
   - The "pull" workflow complete
   - This notification workflow start
   - Polling attempts in the logs
3. **Check the PR** for the notification comment
4. **Click the links** to verify they work

## Customization Options

### Adjust Timeout
Change these values in the workflow:
```yaml
MAX_ATTEMPTS=120  # Change the wait time to 20 minutes
SLEEP_TIME=5      # Check more frequently (every 5 seconds)
```

### Change URL Pattern
If preview URLs change, update:
```bash
PREVIEW_URL="https://docs-preview.pytorch.org/pytorch/pytorch/${PR_NUMBER}/index.html"
CPP_PREVIEW_URL="https://docs-preview.pytorch.org/pytorch/pytorch/${PR_NUMBER}/cppdocs/index.html"
```

### Customize Comment Format
Modify the `commentBody` in the GitHub Script action (lines 118-130)

## Permissions Required

The workflow needs these permissions (already configured):
```yaml
permissions:
  pull-requests: write  # To comment on PRs
  issues: write         # PRs are treated as issues in GitHub API
```

## Monitoring & Debugging

### Successful Run Logs Will Show:
```
Found PR #12345
Checking preview URL: https://docs-preview.pytorch.org/pytorch/pytorch/12345/index.html
Attempt 1/60: Checking if preview is ready...
⏳ Preview not ready yet (HTTP 404). Waiting 10 seconds...
Attempt 2/60: Checking if preview is ready...
✅ Preview is ready! HTTP 200
✅ C++ Preview is also ready! HTTP 200
Created new comment
```

### Failed/Timeout Logs Will Show:
```
Found PR #12345
Checking preview URL: https://docs-preview.pytorch.org/pytorch/pytorch/12345/index.html
[... multiple attempts ...]
❌ Preview did not become ready within the timeout period
Posted timeout notification
```

## Success Metrics

Track these metrics to measure effectiveness:
- % of PRs where notification is posted
- Average time from build complete to docs ready
- User satisfaction (fewer "link doesn't work" issues)
- False positive rate (timeout notifications when docs are actually ready)

---
