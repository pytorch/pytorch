## Summary

This is a skill for auto-triaging issues. This is the human side of things :)
There are 4 pieces to this skill;

1. `SKILL.md` this is the main description of what to do/ directions to follow. If you notice a weird anti pattern in triaging
this is the file you should update. The basic workflow is that there is a static list of labels in `labels.json` that the agent will read w/ their descriptions in order to make decisions. *NOTE* This is static and if new labels are added to `pytorch/pytorch` we should
bump this list w/ their description. I made this static because the full set of labels is too big/quite stale. And I wanted to add more color to certain descriptions. The mechanics for actually interacting w/ gh issues is through the official mcp server. For V1, we always apply
`bot-triaged` whenever any triage action is taken; you can filter those decisions here: https://fburl.com/pt-bot-triaged
2. `templates.json`: This is basically where we want to put canned responses. It includes `redirect_to_forum` (for usage questions) and
`request_more_info` (when classification is unclear). There are likely others we should add here as we notice more patterns.
3. There are hooks in `/scripts`: a pre-hook (`validate_labels.py`) that filters out labels we never want the bot to add, and a post-hook (`add_bot_triaged.py`) that automatically applies `bot-triaged` after any issue mutation.
4. The gh action uses a **two-stage workflow** to support issues opened by OSS users:
   - **Stage 1** (`.github/workflows/claude-issue-triage.yml`): Triggers on `issues: opened`, captures the issue number, and uploads it as an artifact. This stage has no protected environment, so OSS actors can run it.
   - **Stage 2** (`.github/workflows/claude-issue-triage-run.yml`): Triggers on `workflow_run` completion of Stage 1. Runs in the protected `bedrock` environment with AWS/Bedrock access. Downloads the artifact, reads the issue number, and runs the actual triage.

   **Why two stages?** GitHub environment protection blocks jobs before they start if the triggering actor isn't authorized. By using `workflow_run`, Stage 2 is triggered by GitHub itself (trusted context), allowing it to enter the protected environment regardless of who opened the issue.  We use sonnet-4.5, since from testing it is much cheaper and appears to do a more than adequate job at triaging.
5. To disable the flow, disable the GitHub Actions workflow in the repo settings or remove/disable `.github/workflows/claude-issue-triage.yml`.
6. If you would like to test updates before committing them upstream to pytorch, you can do that here: https://github.com/pytorch/ciforge @lint-ignore