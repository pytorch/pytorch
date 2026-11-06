We support the use of AI tools to help prepare issues, pull
requests, reviews, or comments. We expect everyone interacting with
this repo to follow the below policy whenever they use AI tools.

*AI-generated content in comments, issues, or PRs must be clearly disclosed and
contained*, either under a section heading that labels it as AI-generated (e.g.
the PR template's "Agent details" section) or in a code or quote block.
AI-generated content must
be accompanied by human commentary explaining its relevance. For example:
"Codex produced the following analysis: `<insert codex output here>`, so I
believe that \<insert human analysis here\>". The only exceptions to this rule
are the pytorchbot automations.

*Please do not respond to comments+questions by pasting raw or lightly reviewed
AI-generated text.* Other users' comments+questions are requests for your
understanding, reasoning, and judgment. Unreviewed AI output shifts the burden of
verification onto the other party and can make discussions slower, noisier, or
misleading.

*For pull requests, please carefully read the code before submitting it for
review*, especially if AI tools helped write or rewrite it. Make sure the
implementation is something you understand, that the important ideas are clear,
and that the code is high quality. AI-generated code can sometimes be overly
complex, indirect, inconsistent, or include artifacts that obscure the main idea.
Before submitting, simplify where possible and make sure the core change is easy
for maintainers to review. If your PR is not ready for review, please use the
GitHub draft PR feature.

A human may explicitly ask an AI agent to open or update a draft PR before
reviewing its exact patch or description. That request authorizes publication
in draft mode without an additional content-approval step; all disclosure and
human-commentary requirements still apply. Keep the PR in draft mode until the
human has reviewed and understood the code and explicitly asks to mark it ready.
This exception does not authorize comments, reviews, or merging.

All contributions should involve a human who understands and can take
responsibility for the work produced with AI assistance. *We do not accept
contributions created by fully autonomous agents*, and we may close pull requests
that appear to have been generated without meaningful human involvement.
