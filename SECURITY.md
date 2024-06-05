# Security Policy

 - [**Reporting a Vulnerability**](#reporting-a-vulnerability)
 - [**Using Pytorch Securely**](#using-pytorch-securely)
   - [Untrusted models](#untrusted-models)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)

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



Important Note: The trustworthiness of a model is not binary. You must always determine the proper level of caution depending on the specific model and how it matches your use case and risk tolerance.

[^data-poisoning-sources]: To understand risks of utilization of data from unknown sources, read the following Cornell papers on Data poisoning:
    https://arxiv.org/abs/2312.04748
    https://arxiv.org/abs/2401.05566

### Untrusted inputs during training and prediction

If you plan to open your model to untrusted inputs, be aware that inputs can also be used as vectors by malicious agents. To minimize risks, make sure to give your model only the permisisons strictly required, and keep your libraries updated with the lates security patches.

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
