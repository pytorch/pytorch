# Security Policy

 - [**Reporting a Vulnerability**](#reporting-a-vulnerability)
 - [**Using Pytorch Securely**](#using-pytorch-securely)
   - [Untrusted models](#untrusted-models)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)
   - [Using distributed features](#using-distributed-features)
- [**CI/CD security principles**](#cicd-security-principles)
## Reporting Security Issues

Beware that none of the topics under [Using Pytorch Securely](#using-pytorch-securely) are considered vulnerabilities of Pytorch.

However, if you believe you have found a security vulnerability in PyTorch, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/pytorch/pytorch/security/advisories/new

Please refer to the following page for our responsible disclosure policy, reward guidelines, and those things that should not be reported:

https://www.facebook.com/whitehat


## Using Pytorch Securely
**Pytorch models are programs**, so treat its security seriously -- running untrusted models is equivalent to running untrusted code. In general we recommend that model weights and the python code for the model are distributed independently. That said, be careful about where you get the python code from and who wrote it (preferentially check for a provenance or checksums, do not run any pip installed package).

### Untrusted models
Be careful when running untrusted models. This classification includes models created by unknown developers or utilizing data obtained from unknown sources[^data-poisoning-sources].

**Prefer to execute untrusted models within a secure, isolated environment such as a sandbox** (e.g., containers, virtual machines). This helps protect your system from potentially malicious code. You can find further details and instructions in [this page](https://developers.google.com/code-sandboxing).

**Be mindful of risky model formats**. Give preference to share and load weights with the appropriate format for your use case. [safetensors](https://huggingface.co/docs/safetensors/en/index) gives the most safety but is the most restricted in what it supports. [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load) with `weights_only=True` is also secure to our knowledge even though it offers significantly larger surface of attack. Loading un-trusted checkpoint with `weights_only=False` MUST never be done.

**Implement sandboxing for untrusted models and inputs to mitigate risks**. This helps protect your system from potentially malicious code. You can find further details and instructions in [this page](https://developers.google.com/code-sandboxing).

**Add input validation and sanitation to prevent prompt injections and other malicious inputs**. This helps protect your system from potentially malicious code. You can find further details and instructions in [this page](https://developers.google.com/code-sandboxing).

Important Note: The trustworthiness of a model is not binary. You must always determine the proper level of caution depending on the specific model and how it matches your use case and risk tolerance.

[^data-poisoning-sources]: To understand risks of utilization of data from unknown sources, read the following Cornell papers on Data poisoning:
    https://arxiv.org/abs/2312.04748
    https://arxiv.org/abs/2401.05566

### Untrusted inputs during training and prediction

If you plan to open your model to untrusted inputs, be aware that inputs can also be used as vectors by malicious agents. To minimize risks, make sure to give your model only the permissions strictly required, and keep your libraries updated with the latest security patches.

If applicable, prepare your model against bad inputs and prompt injections. Some recommendations:
- Pre-analysis: check how the model performs by default when exposed to prompt injection (e.g. using fuzzing for prompt injection).
- Input Sanitation: Before feeding data to the model, sanitize inputs rigorously. This involves techniques such as:
    - Validation: Enforce strict rules on allowed characters and data types.
    - Filtering: Remove potentially malicious scripts or code fragments.
    - Encoding: Convert special characters into safe representations.
    - Verification: Run tooling that identifies potential script injections (e.g. [models that detect prompt injection attempts](https://python.langchain.com/docs/guides/safety/hugging_face_prompt_injection)).

### Data privacy

**Take special security measures if your model if you train models with sensitive data**. Prioritize [sandboxing](https://developers.google.com/code-sandboxing) your models and:
- Do not feed sensitive data to untrusted model (even if runs in a sandboxed environment)
- If you consider publishing a model that was partially trained with sensitive data, be aware that data can potentially be recovered from the trained weights (especially if model overfits).

### Using distributed features

PyTorch can be used for distributed computing, and as such there is a `torch.distributed` package. PyTorch Distributed features are intended for internal communication only. They are not built for use in untrusted environments or networks.

For performance reasons, none of the PyTorch Distributed primitives (including c10d, RPC, and TCPStore) include any authorization protocol and will send messages unencrypted. They accept connections from anywhere, and execute the workload sent without performing any checks. Therefore, if you run a PyTorch Distributed program on your network, anybody with access to the network can execute arbitrary code with the privileges of the user running PyTorch.

## CI/CD security principles
_Audience_: Contributors and reviewers, especially if modifying the workflow files/build system.

PyTorch CI/CD security philosophy is based on finding a balance between open and transparent CI pipelines while keeping the environment efficient and safe.

PyTorch testing requirements are complex, and a large part of the code base can only be tested on specialized powerful hardware, such as GPU, making it a lucrative target for resource misuse. To prevent this, we require workflow run approval for PRs from non-member contributors. To keep the volume of those approvals relatively low, we easily extend write permissions to the repository to regular contributors.

More widespread write access to the repo presents challenges when it comes to reviewing changes, merging code into trunk, and creating releases. [Protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) are used to restrict the ability to merge to the trunk/release branches only to the repository administrators and merge bot. The merge bot is responsible for mechanistically merging the change and validating reviews against the path-based rules defined in [merge_rules.yml](https://github.com/pytorch/pytorch/blob/main/.github/merge_rules.yaml). Once a PR has been reviewed by person(s) mentioned in these rules, leaving a `@pytorchbot merge` comment on the PR will initiate the merge process. To protect merge bot credentials from leaking, merge actions must be executed only on ephemeral runners (see definition below) using a specialized deployment environment.

To speed up the CI system, build steps of the workflow rely on the distributed caching mechanism backed by [sccache](https://github.com/mozilla/sccache), making them susceptible to cache corruption compromises. For that reason binary artifacts generated during CI should not be executed in an environment that contains an access to any sensitive/non-public information and should not be published for use by general audience. One should not have any expectation about the lifetime of those artifacts, although in practice they likely remain accessible for about two weeks after the PR has been closed.

To speed up CI system setup, PyTorch relies heavily on Docker to pre-build and pre-install the dependencies. To prevent a potentially malicious PR from altering ones that were published in the past, ECR has been configured to use immutable tags.

To improve runner availability and more efficient resource utilization, some of the CI runners are non-ephemeral, i.e., workflow steps from completely unrelated PRs could be scheduled sequentially on the same runner, making them susceptible to reverse shell attacks. For that reason, PyTorch does not rely on the repository secrets mechanism, as these can easily be compromised in such attacks.

### Release pipelines security

To ensure safe binary releases, PyTorch release pipelines are built on the following principles:
 - All binary builds/upload jobs must be run on ephemeral runners, i.e., on a machine that is allocated from the cloud to do the build and released back to the cloud after the build is finished. This protects those builds from interference from external actors, who potentially can get reverse shell access to a non-ephemeral runner and wait there for a binary build.
 - All binary builds are cold-start builds, i.e., distributed caching/incremental builds are not permitted. This renders builds much slower than incremental CI builds but isolates them from potential compromises of the intermediate artifacts caching systems.
 - All upload jobs are executed in a [deployment environments](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment) that are restricted to protected branches
 - Security credentials needed to upload binaries to PyPI/conda or stable indexes `download.pytorch.org/whl` are never uploaded to repo secrets storage/environment. This requires an extra manual step to publish the release but ensures that access to those would not be compromised by deliberate/accidental leaks of secrets stored in the cloud.
 - No binary artifacts should be published to GitHub releases pages, as these are overwritable by anyone with write permission to the repo.
