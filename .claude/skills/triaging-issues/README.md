## Summary

This is a skill for auto-triaging issues. This is the human side of things :)
There are 4 pieces to this skill;

1. `SKILL.md` this is the main description of what to do/ directions to follow. If you notice a weird anti pattern in triaging
this is the file you should update. The basic workflow is that there is a static list of labels in `labels.json` that the agent will read w/ their descriptions in order to make decisions. *NOTE* This is static and if new labels are added to `pytorch/pytorch` we should
bump this list w/ their description. I made this static because the full set of labels is too big/quite stale. And I wanted to add more color to certain descriptions. The mechanics for actually interacting w/ gh issues is through the official mcp server. For V1, we always apply
`bot-triaged` whenever any triage action is taken; you can filter those decisions here: https://fburl.com/pt-bot-triaged
2. `templates.json`: This is basically where we want to put canned responses. It includes `redirect_to_forum` (for usage questions) and
`request_more_info` (when classification is unclear). There are likely others we should add here as we notice more patterns.
3. There is a pre-tool use hook in `/scripts` that filters out any labels that we never want the bot to add.
4. The gh action is here: `.github/workflows/claude-issue-triage.yml`. It sets up roles, checks forks, and logs usage. We are using sonnet-4.5 since from testing it is much cheaper and appears to do a more than adequate job at triaging.
