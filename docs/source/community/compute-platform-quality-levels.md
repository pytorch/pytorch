# PyTorch Compute Platform Quality Levels

As a leading deep learning framework PyTorch capabilities are regularly expanded with the support for new compute platforms and device backends. This document provides scoring criteria to assess quality levels of PyTorch device backends to help developers identify gaps and equip them with the tool to make decisions whether certain compute platforms are ready for specific milestones.

## Quality Levels

Compute Platform is hardware and software environment where computations are executed. In this document the compute platform is understood as a platform which consists of PyTorch backend implementation, underlying software stack (compilers, runtime libraries, drivers and other components) and hardware. The compute platform must be versioned to reflect respective changes in the platform ingredients such as updated versions of software components or new hardware generations being supported. Minor changes of the platform ingredients which are done within the same software or hardware architecture paragidms correspond to different versions of the same platform. It is recommended however to differentiate platforms per the type of supported operating system as different OS typically comes with significantly different driver stacks and often has different set of supported features. Each compute platform can be assessed according to the scoring tables defined below in this document. Scoring covers requirements for platform hardware and software availability, maturity, support obligations, platform features, ci coverage, etc. Each requirement is marked with its relative priority (P0, P1 or P2) and the Score.

Compute Platforms quality levels are defined as follows:

* **Engineering** compute platforms
* **Unstable** compute platforms
* **Stable** compute platforms

**Engineering** platforms are platforms which are under active development and might not be ready for adoption. **Unstable** and **Stable** platforms are more mature platforms which are ready for adoption with different levels of maturity. P0 requirements are mandatory for the platform to qualify as **Stable**. P1 and P2 set relative priorities to help decide which requirements should be considered first. The following table provides requirements for each quality level:

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Compute Platform Level
     - Requirements
   * - **Stable**
     - * Satisfy all 100% P0 requirements
       * Reach 80% overall score (99 points out of 124)
   * - **Unstable**
     - * Reach 70% overall score (86 points out of 124)
   * - **Engineering**
     - * Less than 70% overall score
```

## Documentation update guideline

To first introduce a new compute platform (as **Engineering** Platform) to PyTorch:

* Raise an RFC [issue](https://github.com/pytorch/pytorch/issues) at PyTorch Github repository. Describe proposed platform software, hardware, its availability and plans to develop respective support in PyTorch. Emphasize if a new PyTorch backend is being proposed and what are development plans.

* Request review of the RFC by [Accelerator Working Group](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html) and PyTorch [Technical Advisory Council](https://pytorch.org/tac/). Upon reviewing the proposal and associated materials, the Working Group will provide recommendations and notify PyTorch [Core Maintainers](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html#core-maintainers) of the proposal.

* Request approval for your RFC from the PyTorch [Core Maintainers](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html#core-maintainers). Proceed with the next steps upon approval.

* For the in-tree platforms, submit PR(s) to add support for the new platform

* Submit PR(s) or documentation change requests to update PyTorch side documentation with the descriptions of the new platform

* Once support for the platform has landed, submit a [New Feature for Release](https://github.com/pytorch/pytorch/issues) issue to request announcement of the new platform in the PyTorch release blog and marketing materials

To change existing platform level (to **Unstable** or **Stable**) and reflect that in PyTorch documentation:

* Raise a [New Feature for Release](https://github.com/pytorch/pytorch/issues) issue describing required change

* Attach assessment results following the criteria outlined in this document. Assessment must include evidence for each item.

* Request review of the RFC by [Accelerator Working Group](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html) and PyTorch [Technical Advisory Council](https://pytorch.org/tac/). Upon reviewing the proposal and associated materials, the Working Group will provide recommendations and notify PyTorch [Core Maintainers](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html#core-maintainers) of the proposal.

* Request approval for your RFC from the PyTorch [Core Maintainers](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html#core-maintainers). Proceed with the next steps upon approval.

* Post PR(s) to modify PyTorch documentation as needed and link them to the raised issue

Note that with the evolution of PyTorch some platforms might need to be added and some removed from the PyTorch documentation. It is required to periodically assess quality levels of the compute platforms supported by PyTorch. Assessment results must be provided by the respective team supporting the compute platform per request from [Accelerator Working Group](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html) or PyTorch [Core Maintainers](https://docs.pytorch.org/docs/stable/community/persons_of_interest.html#core-maintainers). Accelerator Working Group owns overall tracking of the compute platforms quality levels and issues recommendations to the PyTorch Core Maintainers. Ultimate decision on the compute platform quality level is owned by PyTorch Core Maintainers. Decision to add or remove a platform from documentation follows the next guideline:

* If adding platform:
  * Require platform to grade as Unstable platform for 2 consecutive PyTorch releases to be added to documentation as Unstable platform
  * Require platform to grade as Stable platform for 2 consecutive PyTorch releases to be added to documentation as Stable platform

* If removing platform:
  * Require platform to grade as Unstable platform for 2 consecutive PyTorch releases to be downgraded to Unstable platform in documentation
  * Require platform to grade as Engineering platform for 2 consecutive PyTorch releases to be downgraded to Engineering platform in documentation

## Prerequisite requirements

As the most popular deep learning framework currently, PyTorch primary goal is to provide users with stable and secure software releases. Therefore, basic function stability and security serve as two fundamental prerequisites criteria for compute platforms.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - Prerequisite Stability
     - Current source and binary release for the given compute platform pass the essential PyTorch smoke testing as defined by `smoke_test.py <https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/smoke_test/smoke_test.py>`_ on the following environments:

       * Full environment setup for compute stack with the access to hardware
       * In case of HW accelerators, full environment setup for compute stack without the access to hardware (``--runtime-error-check disabled`` might be needed for ``smoke_test.py``)
       * Minimal OS environment without fully installed compute stack (``--runtime-error-check disabled`` might be needed for ``smoke_test.py``)

       As overall guidance, PyTorch users should be able at a baseline to ``import torch`` and run basic things like a ``print(torch.__version__)`` after installing PyTorch regardless if the compute platform was configured or not.
     - **P0**
     - 8
   * - 2
     - Prerequisite Security
     - No CVE issues by PyTorch `Security Policy <https://github.com/pytorch/pytorch/?tab=security-ov-file#readme>`_ for currently released source and binaries.
     - **P0**
     - 8
   * - Total
     -
     -
     -
     - 16
```

## Compute Platform Hardware

Compute platform should be publicly released for sale, or accessible as a data center cloud resource by end-users or developers of the open source community.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - Accessibility
     - Access to compute platform hardware is publicly obtained by end users or developers.
     - P1
     - 4
   * - 2
     - Stability
     - Compute platform hardware is of product quality (i.e. hardware is released by the vendor for mass production and long-term customer use) and is under active support.
     - **P0**
     - 6
   * - Total
     -
     -
     -
     - 10
```

## Compute Platform Software Stack

The software stack of a compute platform should be publicly available with complete installation and usage documentation. Stack must provide a stable user interface, regular releases and release updates with the fixes for critical issues and CVEs.

Compute platform software stack includes:

* Compilers
* Runtime libraries
* Device drivers (for platforms with HW accelerators)
* Operating system

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - Accessibility
     - Compute platform software stack is obtained by end users or developers thru publicly available documentation which included complete installation and usage manuals.
     - P1
     - 4
   * - 2
     - Stability
     - Compute platform stack provides:

       * Stable features and user interface for all cases used in pytorch and claimed as supported
       * Compile and run time API versioning for each release
       * ABI versioning for each release
       * **Backward compatible** API and ABI within major release series
     - **P0**
     - 4
   * - 3
     - Maintainability
     - Compute platform stack provides:

       * Regular releases with new features and APIs
       * Release updates with targeted fixes for critical and CVE issues (targeted fixes are bug fixes which do NOT bring in new features, refactoring and non-essential changes - these are NOT permitted)
     - **P0**
     - 4
   * - Total
     -
     -
     -
     - 12
```

##  Compute Platform Feature Support

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 2
     - Usability
     - Compute platform supports data type:

       * torch.float64 : \+1
       * torch.float32 : \+1
       * torch.float16 : \+1
       * torch.bfloat16 : \+1
       * torch.float8 : \+1
       * torch.float4: \+1
       * torch.int64 : \+1
       * torch.int32 : \+1
       * torch.int16 : \+1
       * torch.int8 : \+1
       * torch.int4 : \+1

       Support for the data type can be implemented via simulation on the hardware which does not support specific precision.
     - P1
     - 11
   * - 3
     - Usability
     - Compute platform supports on reasonable performance level:

       * Eager mode : \+1
       * Graph mode : \+1

       Performance level to achieve highly depends on compute platform type. In some cases rough expectations can be given:

       * In case of accelerators, not less than pytorch performance level on CPUs which can be used with the platform
       * In case of CPUs, not worse than 50% performance of other CPUs from the domain given CPU belongs to
     - P1
     - 2
   * - 4
     - Usability
     - Compute platform supports on reasonable performance level:

       * Inference : \+1
       * Training : \+1

       Performance level to achieve highly depends on compute platform type. In some cases rough expectations can be given:

       * In case of accelerators, not less than pytorch performance level on CPUs which can be used with the platform
       * In case of CPUs, not worse than 50% performance of other CPUs from the domain given CPU belongs to
     - P1
     - 2
   * - 5
     - Usability
     - Compute platform supports distributed backend
     - P1
     - 1
   * - 6
     - Usability
     - Compute platform supports CPP Extension API to implement custom operations
     - P1
     - 1
   * - 5
     - Debugability
     - Compute platform supports profiling
     - P1
     - 1
   * - Total
     -
     -
     -
     - 18
```

## Testing Coverage

Testing includes but is not limited to:

* Unit testing
* Integration testing
* E2E validation testing

Test suites, benchmarks and test cases should take PyTorch native unit testing, integration testing and benchmark testing as baseline, and out-of-tree testing results can serve as an auxiliary proof. The test result should be publicly available as result evidence.

Test coverage for a new compute platform should vary depending on the positioning of the attested compute platform. For example, if the compute platform is targeted for inference only in the single node, the training and distributed test scope must be exempted.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - UT
     - PyTorch UT test cases for adapting new compute platforms. Reach the following percentile of test coverage (excluding deprecated cases):

       * 50%: \+2
       * 80%: \+4
       * 100%: \+6

       Deprecated test cases should not be enabled for the new compute platforms.
     - P1
     - 6
   * - 2
     - Benchmarks
     - PyTorch test benchmarks (dynamo benchmarks) for adapting new compute platforms. Reach the following percentile of test coverage (excluding deprecated cases):

       * 50%: \+2
       * 80%: \+4
       * 100%: \+6

       Deprecated test cases should not be enabled for the new compute platforms.
     - P1
     - 6
   * - 3
     - Mode support
     - One of the Pytorch basic modes, eager and/or graph mode is tested:

       * Eager mode: \+2
       * Graph mode: \+2
     - P1
     - 4
   * - 4
     - Ops support
     - Ops test coverage for adapting new compute platform. Reach the following percentile of test coverage (excluding deprecated cases):

       * 50%: \+2
       * 80%: \+4
       * 100%: \+6

       Deprecated test cases should not be enabled for the new compute platforms.
     - P1
     - 6
   * - 5
     - Distributed
     - Distributed mode is tested (only applicable to new compute platforms targeting distributed usage scenarios).
     - P1
     - 4
   * - Total
     -
     -
     -
     - 26
```

## User Experience

The PyTorch backend implementation for the proposed compute platform must align with PyTorch common practice to provide better user experience on binary distribution, environment setup, installation, documentation, debugging ability and profiler for better performance.


```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - PyTorch binary distribution
     - At least one kind of PyPI, Conda or offline download package (similar to LibTorch) is provided. \+2

       Binary size is comparable with PyTorch default wheel to allow user quick download: \+2

       Binary release date of the PyTorch variant with backend implementation for the proposed compute platform is on the same day as the PyTorch release: \+2
     - P2
     - 6
   * - 2
     - Easy Software setup
     - PyTorch release for compute platform contains valid links to vendor documentation describing compute stack prerequisites, environment setup and installation instructions.

       Vendor side documentation remains valid and available by the same link for the whole live time of pytorch release.

       Compute platform initialization logs contain actionable user friendly feedback.
     - P2
     - 2
   * - 3
     - Documentation/examples starting points
     - PyTorch release for compute platform contains:

       * Getting started guide
       * Environment setup guide
       * Installation manual
       * User tutorials with examples for the advertised features
     - P2
     - 2
   * - 4
     - Debugging ability
     - Pytorch release for compute platform provides:

       * Debug build options
       * Runtime error log messages
       * Compatibility with PyTorch debugging utilities (PyTorch Profiler, Flight Recorder, etc.)
     - P2
     - 2
   * - 5
     - Profiler ability
     - Pytorch release for compute platform provides performance profiling tools:

       * Pytorch native profiling support (``torch.profile``): +2
       * Compute platform custom profiling tools for profiling and debug on per-operator level: +2

       (Performance is always an important factor that users care about in PyTorch.)
     - P2
     - 4
   * - Total
     -
     -
     -
     - 14
```


## CI/CD

CI/CD is an effective mechanism in the PyTorch open source community to ensure continuous integration and release. Enabled CI/CD is a prerequisite for maturity of the pytorch compute platform.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - Accessibility
     - CI/CD is publicly accessible. Test results are transparent to the PyTorch open source community.
     - P0
     - 2
   * - 2
     - CI
     - CI gates every PR relevant for the proposed compute platform by executing:

       * Proposed compute platform specific tests: \+2
       * Tests for other affected platforms (if affected by the change): \+2

       Tests or part of the tests for each PR for the particular platform can be replaced by the cadence testing (nightly, weekly) if running tests on the given platform is time and/or infrastructure expensive. This should be done with the agreement from PyTorch maintainers if PRs are done for one of the upstream PyTorch repositories.
     - P1
     - 4
   * - 3
     - CD
     - CD supports preview (nightly builds) and release builds.
     - P1
     - 4
   * - 4
     - Tests stability
     - CI/CD configuration has less than 5% tests marked as flaky
     - P1
     - 4
   * - 5
     - CI/CD stability
     - CI/CD has minimal outages with the recovery period not longer than 3 days
     - P1
     - 4
   * - Total
     -
     -
     -
     - 18
```

## Feedback adopters

Feedback is required for the proposed compute platform by multiple channels. Need to list users/teams that have tried the compute platforms and provided public feedback. Feedback might include but not limited to blog posts evaluating the platform, list of submitted issues associated with the specific project, engineering articles discussing scientific results obtained using the platform, etc.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - Feedback adopters
     - Users/teams have tried the compute platform, provided feedback and the feedback was addressed.

       * Address 50%: \+2
       * Address 80%: \+4
       * Address 100%: \+6
     - P1
     - 6
   * - Total
     -
     -
     -
     - 6
```

## Maintenance

Support should be provided for the new compute platform in line with the existing PyTorch support policies, including [Releasing PyTorch Policy](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) and [Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md).

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - ID
     - Category
     - As measured by
     - Priority
     - Score
   * - 1
     - Support
     - Critical and CVE issues (“Security”) fixes are committed to support following [Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
     - P1
     - 4
   * - Total
     -
     -
     -
     - 4
```
