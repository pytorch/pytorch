# PyTorch GitHub Actions Workflows

This folder hosts all the GitHub Actions run for pytorch/pytorch.

## Intro

GitHub Actions are arranged by **workflows**, which contain specific **jobs**, which contain **steps**. Check out [GitHub's intro docs](https://docs.github.com/en/github-ae@latest/actions/learn-github-actions/understanding-github-actions?learn=getting_started) to learn more. An example in our repository is the pull.yml
workflow, which defines all build + tests jobs run when a pull request is created or modified. Each build and test job
(like the Linux CUDA builds and tests) consists of modular steps for setting up, building/testing, and tearing down.

## Job Naming Convention
GHA jobs have both job names and string job IDs (they also have number IDs but those aren't included in the GitHub context yet, sadly). Job names are mostly for display (what shows up in the signal box to users), but the job id is
a unique reference to a particular job.
