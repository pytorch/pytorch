# PyTorch ONNX Exporter Diagnostics

NOTE: This feature is underdevelopment and is subject to change.

Summary of source tree:
- [OVERVIEW.md](OVERVIEW.md): Technical overview of the diagnostics infrastructure.
- [generated/](generated): Generated diagnostics rules from [rules.yaml](rules.yaml).
- [infra/](infra): Generic diagnostics infrastructure built on top of [SARIF](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html).
- [_diagnostic.py](diagnostic.py): Python API for diagnostics.
- [rules.yaml](rules.yaml): Single source of truth for diagnostics rules. Used to generate C++ and Python interfaces, and documentation pages.
- [tools/onnx/](/tools/onnx): Scripts for generating source code and documentation for diagnostics rules.

## Table of Contents

<!-- toc -->

- [Introduction](#introduction)
  - [Motivation](#motivation)
    - [Diagnostics as documentation](#diagnostics-as-documentation)
    - [Different context and background](#different-context-and-background)
    - [Machine parsable](#machine-parsable)
  - [Design](#design)
    - [Adopting SARIF for diagnostic structure](#adopting-sarif-for-diagnostic-structure)
    - [Single source of truth for diagnostic rules](#single-source-of-truth-for-diagnostic-rules)
- [Internal Details](#internal-details)
  - [Rules](#rules)
  - [Infrastructure](#infrastructure)
  - [Documentation](#documentation)
- [Usage](#usage)
  - [Python](#python)
  - [C++](#c)

<!-- tocstop -->

# Introduction

The goal is to improve the diagnostics to help users debug and improve their model export.
* The diagnostics are emitted in machine parsable [Static Analysis Results Interchange Format (SARIF)](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html).
* A new clearer, structured way to add new and keep track of diagnostic rules.
* Serve as foundation for more future improvements consuming the diagnostics.

## Motivation ##

The previous diagnostics were only scattered warning or error messages. They are not structured and are not machine parsable. This makes it hard to consume the diagnostics in a systematic way. This is a blocker for improving the diagnostics and for building tools on top of them. The diagnostics are also not very helpful for users to debug their model export. They are often not actionable and do not provide enough context to help users debug their model export. Some unsupported patterns or code are documented in the [PyTorch ONNX doc](https://pytorch.org/docs/stable/onnx.html#limitations). The information is scattered, hard to find, and hard to maintain and thus often outdated. The new diagnostics system aim to address these issues with the following key properties.

### Diagnostics as documentation

The diagnostics are the source of truth for the documentation of export issues. Documentations are no longer separated. Any changes are directly reflected as the diagnostic progress. The diagnostic itself serves as the means to track the history and progress of any specific issue. Linking the source code, the issues, the PRs, the fix, the docs, etc together through this single entity.

### Different context and background

There are two very different audiences: users and converter developers. The users care more about where the error is coming from the model, and how to resolve it for a successful export. They are not experts in the internal of exporter or JIT. The converter developers on the other hand need more info of the internal state of the converter to debug the issue. The diagnostics should be actionable for users and provide enough context for converter developers to debug and fix the issues. It should display the right information and context to the right audience, in a clean and concise way.

### Machine parsable

The diagnostics are emitted in machine parsable SARIF format. This opens the door for the diagnostics to be consumed by tools and systems. Future applications like auto fixing, formatted displaying, auto reporting, etc are possible.

## Design ##

### Adopting SARIF for diagnostic structure

The diagnostics are emitted in machine parsable [Static Analysis Results Interchange Format (SARIF)](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html), with [python classes for SARIF object model](https://github.com/microsoft/sarif-python-om) as starting point. This is a standard format for the output of static analysis tools, and can be consumed by the SARIF Viewer, [VSCode extension](https://marketplace.visualstudio.com/items?itemName=MS-SarifVSCode.sarif-viewer) for example. The diagnostics are also emitted in a human readable format for users to read. The human readable format is a subset of the SARIF format. The human readable format is emitted to stdout and the SARIF format is emitted to a file. [Authoring rule metadata and result messages](https://github.com/microsoft/sarif-tutorials/blob/main/docs/Authoring-rule-metadata-and-result-messages.md) is a good starting point for understanding the SARIF format.

### Single source of truth for diagnostic rules

The diagnostic rules are defined in a single location, in [SARIF `reportingDescriptor` format](https://docs.oasis-open.org/sarif/sarif/v2.1.0/os/sarif-v2.1.0-os.html#_Toc34317836). From it, respective C++, python and documentation files are generated during build. With a bit of redundancy, this approach makes all the rules statically accessible under both Python and C++, while maintaining a single source of truth.

# Internal Details

## Rules ##


## Infrastructure ##


## Documentation ##


# Usage

## Python ##

## C++ ##
