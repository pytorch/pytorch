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

*Please keep stacks of pull requests under 10 PRs.* Large stacks can use up
shared CI resources and block other contributors. Land the bottom of a stack
before adding more on top, and use `[no-ci]` on PRs that don't need CI yet
(see the [ghstack-ci skill](.agents/skills/ghstack-ci/SKILL.md)). To protect
our CI infrastructure, we reserve the right to stop running CI on large
stacks, as in [#185290](https://github.com/pytorch/pytorch/pull/185290).

All contributions should involve a human who understands and can take
responsibility for the work produced with AI assistance. *We do not accept
contributions created by fully autonomous agents*, and we may close pull requests
that appear to have been generated without meaningful human involvement.
