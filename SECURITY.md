# Security Policy

 - [**Reporting a Vulnerability**](#reporting-a-vulnerability)
 - [**Issues That Are Not Security Vulnerabilities**](#issues-that-are-not-security-vulnerabilities)
 - [**Using PyTorch Securely**](#using-pytorch-securely)
   - [Untrusted models](#untrusted-models)
   - [TorchScript models](#torchscript-models)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)
   - [Using distributed features](#using-distributed-features)
- [**Backporting Security Fixes**](#security-fixes-and-old-releases)
- [**CI/CD security principles**](#cicd-security-principles)
## Reporting Security Issues

Beware that none of the topics under [Using PyTorch Securely](#using-pytorch-securely) are considered vulnerabilities of PyTorch.

However, if you believe you have found a security vulnerability in PyTorch, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/pytorch/pytorch/security/advisories/new

All reports submitted through the security advisories mechanism would **either be made public or dismissed by the team within 90 days of the submission**. If advisory has been closed on the grounds that it is not a security issue, please do not hesitate to create an [new issue](https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml) as it is still likely a valid issue within the framework.

Please refer to the following page for our responsible disclosure policy, reward guidelines, and those things that should not be reported:

https://www.facebook.com/whitehat

## Issues That Are Not Security Vulnerabilities

PyTorch is a framework that executes user-provided code, including model definitions, custom operators, and training scripts. Like many low-level computational libraries, PyTorch generally does not validate all inputs to every function - the responsibility for providing valid arguments lies with the calling code. An attacker who already has the ability to execute arbitrary code locally, or to modify files on the system, does not gain any additional capability by exploiting PyTorch. The following categories of reports should be filed as regular bugs, **not** as security vulnerabilities:

- **Crashes and out-of-bounds access**: Passing invalid arguments (wrong shapes, dtypes, device placement, etc.) may result in crashes, segfaults, or out-of-bounds memory access. These are often caused by missing input validation and are bugs worth reporting as regular issues, but not security vulnerabilities — the caller is already running native code and could achieve the same effect without PyTorch.

- **Numerical accuracy and stability**: Differences in numerical results between devices (CPU vs. CUDA), dtypes, platforms, or library versions are expected behavior in floating-point computing. This includes precision loss (CWE-682), non-determinism across runs, and results that diverge from a reference implementation.

- **Internal or "unsafe" functions**: Some PyTorch functions are intentionally low-level or unchecked (e.g., functions prefixed with `_`, internal C++ APIs, or APIs documented as unsafe). These exist for performance or flexibility and are not intended to be hardened against adversarial callers. If an attacker can invoke such functions with arbitrary arguments, they likely have local execution access already.

- **Denial of service via resource consumption**: PyTorch is designed to run computationally intensive workloads that fully utilize CPU, GPU, and memory. Crafting inputs that cause high resource consumption or long-running operations is trivial and expected — this is not a security vulnerability.

- **Local filesystem and cache trust**: PyTorch heavily uses local caching for performance (e.g., compiled kernel artifacts, Triton cache). These cache should be located in local user-only accessible locations and those are trusted by design. If an attacker has write access to the local filesystem, they can already execute arbitrary code — poisoning a PyTorch cache does not grant any additional capability.

If your security advisory is closed because it falls into one of these categories, please don't be discouraged — these are still valuable reports. We encourage you to re-file them as a [regular issue](https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml) so they can be tracked and fixed as bugs.

## Using PyTorch Securely
**PyTorch models are programs**, so treat its security seriously -- running untrusted models is equivalent to running untrusted code. In general we recommend that model weights and the python code for the model are distributed independently. That said, be careful about where you get the python code from and who wrote it (preferentially check for a provenance or checksums, do not run any pip installed package).

### Untrusted models
Be careful when running untrusted models. This classification includes models created by unknown developers or utilizing data obtained from unknown sources[^data-poisoning-sources].

**Prefer to execute untrusted models within a secure, isolated environment such as a sandbox** (e.g., containers, virtual machines). This helps protect your system from potentially malicious code. You can find further details and instructions in [this page](https://developers.google.com/code-sandboxing).

**Be mindful of risky model formats**. Give preference to share and load weights with the appropriate format for your use case. [Safetensors](https://huggingface.co/docs/safetensors/en/index) gives the most safety but is the most restricted in what it supports. [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load) has a significantly larger surface of attack but is more flexible in what it can serialize. See the documentation for more details.

Even for more secure serialization formats, unexpected inputs to the downstream system can cause diverse security threats (e.g. denial of service, out of bound reads/writes) and thus we recommend extensive validation of any untrusted inputs.

Important Note: The trustworthiness of a model is not binary. You must always determine the proper level of caution depending on the specific model and how it matches your use case and risk tolerance.

[^data-poisoning-sources]: To understand risks of utilization of data from unknown sources, read the following Cornell papers on Data poisoning:
    https://arxiv.org/abs/2312.04748
    https://arxiv.org/abs/2401.05566

### TorchScript models

TorchScript models should be treated the same way as locally executable code from an unknown source. Only run TorchScript models if you trust the provider. Please note, that tools for introspecting TorchScript models (such as `torch.utils.model_dump`) may also execute partial or full code stored in those models, therefore they should be used only if you trust the provider of the binary you are about to load.

PyTorch mobile models (`.ptl` files) are TorchScript models optimized for mobile deployment and should be treated with the same level of caution. Only load PyTorch mobile models from trusted sources.

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

**Take special security measures if you train your models with sensitive data**. Prioritize [sandboxing](https://developers.google.com/code-sandboxing) your models and:
- Do not feed sensitive data to an untrusted model (even if runs in a sandboxed environment)
- If you consider publishing a model that was partially trained with sensitive data, be aware that data can potentially be recovered from the trained weights (especially if the model overfits).

### Using distributed features

PyTorch can be used for distributed computing, and as such there is a `torch.distributed` package. PyTorch Distributed features are intended for internal communication only. They are not built for use in untrusted environments or networks.

For performance reasons, none of the PyTorch Distributed primitives (including c10d, RPC, and TCPStore) include any authorization protocol and will send messages unencrypted. They accept connections from anywhere, and execute the workload sent without performing any checks. Therefore, if you run a PyTorch Distributed program on your network, anybody with access to the network can execute arbitrary code with the privileges of the user running PyTorch.

## Backporting Security Fixes

Security fixes are only applied to the current release. Fixes are never backported to older versions of PyTorch. The only security concern applicable to previously published releases is tampering with the published binaries themselves (e.g., unauthorized modification of packages on PyPI or download indexes). If you believe a published binary has been compromised, please report it through the [security advisory process](#reporting-security-issues).

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
