## Summary

This is a skill for auto-triaging issues. This is the human side of things :)
The pieces of this skill:

1. `SKILL.md` this is the main description of what to do/ directions to follow. If you notice a weird anti pattern in triaging
this is the file you should update. The basic workflow is that there is a static list of labels in `labels.json` that the agent will read w/ their descriptions in order to make decisions. *NOTE* This is static and if new labels are added to `pytorch/pytorch` we should
bump this list w/ their description. I made this static because the full set of labels is too big/quite stale. And I wanted to add more color to certain descriptions. For V1, we always apply
`bot-triaged` whenever any triage action is taken; you can filter those decisions here: https://fburl.com/pt-bot-triaged
2. `templates.json`: This is basically where we want to put canned responses. It includes `redirect_to_forum` (for usage questions) and
`request_more_info` (when classification is unclear). There are likely others we should add here as we notice more patterns. These are the only comments the bot can post.
3. The label rules (forbidden prefixes, the `labels.json` allowlist, redundant pairs) live in `scripts/issue_triage_pi/labels.py`, which the triage workflow's apply step enforces.
4. The gh action uses a **two-stage workflow** to support issues opened by OSS users:
   - **Stage 1** (`.github/workflows/issue-triage.yml`): Triggers on `issues: opened`, captures the issue number, and uploads it as an artifact. This stage has no protected environment, so OSS actors can run it.
   - **Stage 2** (`.github/workflows/issue-triage-pi.yml`): Triggers on `workflow_run` completion of Stage 1. A [pi](https://github.com/earendil-works/pi) agent in the protected `bedrock` environment reads the issue with read-only tools and submits a plan; a separate job without model access applies it (`scripts/issue_triage_pi/apply_plan.py`), adds `bot-triaged`, and hands `oncall: distributed` issues to the distributed triage workflow. See `.github/actions/pi-agent/README.md`.

   **Why two stages?** GitHub environment protection blocks jobs before they start if the triggering actor isn't authorized. By using `workflow_run`, Stage 2 is triggered by GitHub itself (trusted context), allowing it to enter the protected environment regardless of who opened the issue. The model and thinking level default to the workflow's dispatch inputs.
5. To disable the flow, disable the GitHub Actions workflow in the repo settings or remove/disable `.github/workflows/issue-triage.yml`.
6. To check a change before it lands, dispatch `issue-triage-pi.yml` in `replay` mode on already-triaged issues: it shows each issue as it was before triage and compares the plan with the labels added since. Changes can also be tried first in https://github.com/pytorch/ciforge @lint-ignore
