# Directory of .github

```sh
.github
├── ISSUE_TEMPLATE
├── ISSUE_TEMPLATE.md
├── PULL_REQUEST_TEMPLATE.md
├── REAME.md
├── labels.yml                     # auto sync github labels (for issues and pr)
├── pytorch-circleci-labels.yml    # probot config to enable circleci label dispatch
├── pytorch-probot.yml             # probot config to enable auto-CC'ing based on labels
├── scale-config.yml               # custom github runner specs
├── scripts                        # scripts that generate the workflows (using tempaltes)
├── templates                      # jinja2 templates for workflows
└── workflows                      # github action workflows (usually for CI)
```

## Sync Labels

Labels are essential to certain github actions, pytorch's triage workflows, and many bots
like probot and oss bots. We now make all the labels of issues and pull requests declared
in `.github/labels.yml`, and there's a action to make sure the labels are in sync
with `.github/workflows/sync_labels.yml`.

The initial set of the labels are crawled by `label-exporter`. No need to run again as
`.github/labels.yml` will be the source of truth going forward.

```sh
# FYI doc to crawl and setup the initial set of labels

curl -sf https://gobinaries.com/github.com/micnncim/label-exporter/cmd/label-exporter | sh
GITHUB_TOKEN=<GITHUB_TOKEN> label-exporter pytorch pytorch --yaml > .github/labels.yml
```
