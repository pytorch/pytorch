# CI Integration

## Background

Out-of-tree (OOT) accelerator backends need to maintain compatibility with PyTorch's evolving codebase. As PyTorch continues to develop rapidly, changes in the upstream repository can potentially break downstream accelerator integrations. To address this challenge, PyTorch provides a Cross-Repository CI Relay (CRCR) mechanism that enables automatic CI coordination between the PyTorch repository and downstream accelerator repositories.

This chapter guides third-party accelerator vendors through the process of integrating their repositories with PyTorch's CI ecosystem, ensuring continuous compatibility validation.

## Why CI/CD Integration Matters

Integrating with PyTorch's CI ecosystem provides several key benefits:

* **Early Detection**: Catch compatibility issues before they reach production, reducing debugging effort and user impact.
* **Automated Validation**: Automatically test your accelerator against PyTorch PRs without manual intervention.
* **Reduced Maintenance Burden**: Proactive testing reduces the need for reactive fixes when compatibility breaks.

## How It Works

The CRCR system consists of four components: a **GitHub App** that bridges authentication and events, the **PyTorch repository** as the upstream event source, a **Relay Server** that dispatches events to eligible downstream repos, and **downstream repositories** that receive events and optionally report results back.

When a PR is opened or updated in PyTorch, GitHub notifies the Relay Server via the GitHub App. The Relay Server verifies the event, reads the allowlist, and dispatches a ``repository_dispatch`` event to each registered downstream repository. Downstream repos can optionally report CI results back to the Relay Server, which surfaces them in the PyTorch HUD or as PR check runs.

```{mermaid}
flowchart TD
    PyTorch["PyTorch\n(PR Event)"] -->|webhook| RS["Relay Server\n(Allowlist/Dispatch/Callback)"]
    GH["GitHub APP\n(Auth&Bridge)"] <--> RS
    RS <--> HUD["HUD\n(Dashboard)"]
    RS -->|repo_dispatch| DA["Downstream A\n(e.g. Ascend)"]
    RS -->|repo_dispatch| DB[Downstream B]
    RS -->|repo_dispatch| DC[Downstream C]
    DA -->|callback| RS
    DB -->|callback| RS
    DC -->|callback| RS
```

Participation is governed by a four-tier allowlist:

* **L1**: Events are forwarded to the downstream repo; no results are reported upstream.
* **L2**: Results are displayed on dedicated HUD pages for the downstream repo.
* **L3**: Non-blocking check runs appear on PyTorch PRs, triggered by maintainer labels.
* **L4**: Blocking check runs run on all PyTorch PRs (reserved for critical accelerators).

Downstream repos advance through levels by meeting documented requirements around hardware verification, CI reliability, and success rates.

For a deeper dive into the architecture and design decisions, see the [RFC-0050: Cross-Repository CI Relay for PyTorch Out-of-Tree Backends](https://github.com/fffrog/rfcs/blob/5e138470e962b0f9c5092e564f35bd7fb13b0b2f/RFC-0050-Cross-Repository-CI-Relay-for-PyTorch-Out-of-Tree-Backends.md).

```{note}
The CRCR currently supports **L1 (Silent)** integration only.
```

## Integration Steps

### Step 1: Install the GitHub App

Install the [PyTorch Cross-Repo CI Relay](https://github.com/apps/pytorch-fdn-cross-repo-ci-relay) GitHub App on your repository by clicking the **Configure** button and selecting your repository.

### Step 2: Add Your Repository to the Allowlist

Submit a pull request to ``pytorch/pytorch`` adding your repository to ``.github/allowlist.yml`` under the ``L1`` key:

```{eval-rst}
.. code-block:: yaml

   L1:
     - your-org/your-accelerator
```

See [#180352](https://github.com/pytorch/pytorch/pull/180352) for a reference example. The PyTorch team will review and merge the PR to complete the onboarding.

### Step 3: Create the Workflow File

Create a GitHub Actions workflow in your repository to receive ``repository_dispatch`` events:

```{eval-rst}
.. code-block:: yaml
   :caption: .github/workflows/pytorch_ci.yml

   name: PyTorch CI

   run-name: >-
     PyTorch CI -
     ${{
       github.event.client_payload.event_type == 'pull_request' &&
       format('PR #{0} ({1})',
         github.event.client_payload.payload.pull_request.number,
         github.event.client_payload.payload.action) ||
       format('Push {0}', github.event.client_payload.payload.after)
     }}

   on:
     repository_dispatch:
       types: [pull_request, push]

   concurrency:
     group: >-
       pytorch-ci-${{ github.event.client_payload.payload.repository.full_name }}-${{
       github.event.client_payload.payload.pull_request.number || github.run_id }}
     cancel-in-progress: true

   permissions:
     contents: read

   jobs:
     cancel-pr:
       if: ${{ github.event.client_payload.payload.action == 'closed' }}
       runs-on: ubuntu-latest
       steps:
         - run: echo "PR closed, canceling in-progress runs"

     ci:
       if: ${{ github.event.client_payload.payload.action != 'closed' }}
       runs-on: ubuntu-latest
       steps:
         - name: Checkout downstream repo
           uses: actions/checkout@v4

         - name: Checkout PyTorch at triggered commit
           uses: actions/checkout@v4
           with:
             repository: pytorch/pytorch
             ref: >-
               ${{ github.event.client_payload.event_type == 'pull_request' &&
               github.event.client_payload.payload.pull_request.head.sha ||
               github.event.client_payload.payload.after }}
             path: pytorch

         - name: Build and test
           run: |
             # Your build and test commands
             echo "Running tests against PyTorch..."
```

### Step 4: Test the Integration

Verify your integration works correctly:

1. Create a test PR in PyTorch (or ask maintainers to trigger a test dispatch)
2. Confirm your workflow triggers correctly

## Event Payload

The CRCR relay is a stateless pass-through: it forwards the complete GitHub webhook payload as ``client_payload`` in the ``repository_dispatch`` event. There is no simplified intermediary schema.

The ``client_payload`` has two top-level fields:

* ``event_type``: either ``pull_request`` or ``push``
* ``payload``: the raw GitHub webhook payload for that event type

Commonly used fields:

```{eval-rst}
.. code-block:: yaml

   github.event.client_payload.event_type                       # "pull_request" or "push"
   github.event.client_payload.payload.action                   # "opened", "synchronize", "reopened" or "closed" only
   github.event.client_payload.payload.pull_request.number      # PR number (pull_request events only)
   github.event.client_payload.payload.pull_request.head.sha    # Head commit SHA to checkout
   github.event.client_payload.payload.after                    # Commit SHA (push events only)
```

Supported ``action`` values for ``pull_request`` events:

| Action | Description |
| ------ | ----------- |
| ``opened`` | New PR created |
| ``synchronized`` | New commits pushed to an existing PR |
| ``reopened`` | Previously closed PR reopened |
| ``closed`` | PR closed or merged; triggers the ``cancel-pr`` job to stop in-progress runs |

## Troubleshooting

### Workflow Not Triggering

1. Confirm your onboarding with the PyTorch team is complete
2. Check that your workflow file is on the default branch
3. Ensure the ``repository_dispatch`` event type in your workflow matches what the relay sends

## Resources

* [RFC-0050: Cross-Repository CI Relay for PyTorch Out-of-Tree Backends](https://github.com/pytorch/rfcs/blob/master/RFC-0050-Cross-Repository-CI-Relay-for-PyTorch-Out-of-Tree-Backends.md)
