# PyTorch CI Stats

We track various stats about each CI job.

1. Jobs upload their artifacts to an intermediate data store (either GitHub
   Actions artifacts or S3, depending on what permissions the job has). Example:
   https://github.com/pytorch/pytorch/blob/a9f6a35a33308f3be2413cc5c866baec5cfe3ba1/.github/workflows/_linux-build.yml#L144-L151
2. When a workflow completes, a `workflow_run` event [triggers
   `upload-test-stats.yml`](https://github.com/pytorch/pytorch/blob/d9fca126fca7d7780ae44170d30bda901f4fe35e/.github/workflows/upload-test-stats.yml#L4).
3. `upload-test-stats` downloads the raw stats from the intermediate data store
   and uploads them as JSON to s3, which then uploads to our database backend

```mermaid
graph LR
    J1[Job with AWS creds<br>e.g. linux, win] --raw stats--> S3[(AWS S3)]
    J2[Job w/o AWS creds<br>e.g. mac] --raw stats--> GHA[(GH artifacts)]

    S3 --> uts[upload-test-stats.yml]
    GHA --> uts

    uts --json--> s3[(s3)]
    s3 --> DB[(database)]
```

Why this weird indirection? Because writing to the database requires special
permissions which, for security reasons, we do not want to give to pull request
CI. Instead, we implemented GitHub's [recommended
pattern](https://securitylab.github.com/research/github-actions-preventing-pwn-requests/)
for cases like this.

For more details about what stats we export, check out
[`upload-test-stats.yml`](https://github.com/pytorch/pytorch/blob/d9fca126fca7d7780ae44170d30bda901f4fe35e/.github/workflows/upload-test-stats.yml)
