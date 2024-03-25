# Security Policy

 - [**Using Pytorch Securely**](#using-pytorch-securely)
   - [Untrusted models](#untrusted-models)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)
   - [Untrusted environments or networks](#untrusted-environments-or-networks)
   - [Multi-Tenant environments](#multi-tenant-environments)
 - [**Reporting a Vulnerability**](#reporting-a-vulnerability)

## Using Pytorch Securely
**Consider Pythor models as programs**. Machine Learning models are mathematical vectors and weights in their essence, but as they grow in complexity they can require privileges or be partially composed by programmatic code. That said, running untrusted models is equivelent to running untrusted code.

### Untrusted models
Be careful when running untrusted models. This classification includes models created by unknown developers or utilizing data obtained from unknown sources.

**Prefer to execute untrusted models within a secure, isolated environment such as a sandbox** (e.g., containers, virtual machines). This helps protect your system from potentially malicious code. You can find further details and instructions in [this page](https://developers.google.com/code-sandboxing).

**Be mindful of risky model formats**. Give preference to run models with `.safetensors` format, and if you need to run load models on the [picle](https://docs.python.org/3/library/pickle.html) format, make sure to use the `weights_only` option to `torch.load`.

Important Note: The trustworthiness of a model is not binary. You must always determine the proper level of caution depending on the specific model and how it matches your use case and risk tolerance.

### Untrusted inputs during training and prediction

Some models accept various input formats (text, images, audio, etc.) and pre-process them by modifying and/or converting the input. Most of this work is done by a variaty of libraries that provide different levels of security. Based on the security history of these libraries, it's considered safe to work with untrusted inputs for PNG, BMP, GIF, WAV, RAW, RAW_PADDED, CSV and PROTO formats. If you process untrusted data in any other format, we you should employ the following recommendations:

* Sandboxing: Isolate the model process. You can follow the same instructions as given on the section [Untrusted models](#untrusted-models).
* Updates: Keep your model and libraries updated with the latest security patches.

### Data privacy

**Take special security measures if your model if you train models with sensitive data**. Prioritize [sandboxing](https://developers.google.com/code-sandboxing) your models and:
- Do not feed sensitive data to untrusted model (even if runs in a sandboxed environment)
- If you consider publishing a model that was partially trained with sensitive data, be aware that data can potentially be recovered from the trained weights (especially if model overfits).

### Untrusted environments or networks

If you can't run your models in a secure and isolated environment or if it must be exposed to an untrusted network, make sure to take the following security precautions:
* Confirm the hash of any downloaded artifact (e.g. pre-trained model weights) matches a known-good value
* Encrypt your data if sending it over the network.

### Multi-Tenant environments

If you intend to run multiple models in parallel with shared memory, it is your responsibility to ensure the models do not interact or access each other's data. The primary areas of concern are tenant isolation, resource allocation, model sharing and hardware attacks.

#### Tenant Isolation

You must make sure that models run separately. Since models can run code, it's important to use strong isolation methods to prevent unwanted access to the data from other tenants.

Separating networks is also a big part of isolation. If you keep model network traffic separate, you not only prevent unauthorized access to data or models, but also prevent malicious users or tenants sending graphs to execute under another tenant’s identity.

#### Resource Allocation

A denial of service caused by one model can impact the overall system health. Implement safeguards like rate limits, access controls, and health monitoring.

#### Model Sharing

In a multitenant design that allows sharing models, it's crucial to ensure that tenants and users fully understand the potential security risks involved. They must be aware that they will essentially be running code provided by other users. Unfortunately, there are no reliable methods available to detect malicious models, graphs, or checkpoints. To mitigate this risk, the recommended approach is to sandbox the model execution, effectively isolating it from the rest of the system.

#### Hardware Attacks

Besides the virtual environment, the hardware (GPUs or TPUs) can also be attacked. [Research](https://scholar.google.com/scholar?q=gpu+side+channel) has shown that side channel attacks on GPUs are possible, which can make data leak from other models or processes running on the same system at the same time.

## Reporting Security Issues

Beware that none of the topics under [Using Pytorch Securely](#using-pytorch-securely) are considered vulnerabilities of Pytorch.

However, if you believe you have found a security vulnerability in PyTorch, we encourage you to let us know right away. We will investigate all legitimate reports and do our best to quickly fix the problem.

Please report security issues using https://github.com/pytorch/pytorch/security/advisories/new

Please refer to the following page for our responsible disclosure policy, reward guidelines, and those things that should not be reported:

https://www.facebook.com/whitehat
