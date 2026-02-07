# OpenReg Standardized Testing Patterns Documentation

This directory contains comprehensive documentation for OpenReg testing patterns, intended for backend authors, maintainers, and contributors.

## Contents

- **[test_instantiation.md](test_instantiation.md)** — Core guide explaining how tests are generated and expanded using `DeviceTypeTestBase` and parametrization.
- **[failure_interpretation.md](failure_interpretation.md)** — Guide to understanding, categorizing, and debugging test failures.
- **[operator_coverage.md](operator_coverage.md)** — Explains operator coverage expectations and backend maturity stages.
- **[skip_patterns.md](skip_patterns.md)** — Reference for when and how to use test skips (approved decorators and best practices).
- **[adding_tests.md](adding_tests.md)** — Step-by-step recipe for contributing new tests to OpenReg.

## Quick Links

- **Issue/Parent Reference:** OpenReg Testing Patterns Documentation (issue #169597)
- **Related PR:** #158644 (OpenReg documentation work)
- **Related RFC:** RFC-0045 — PyTorch Accelerator Integration Enhancements
- **PyTorch Accelerator Docs:** https://docs.pytorch.org/main/accelerator/device.html

## For New Contributors

Start with [test_instantiation.md](test_instantiation.md) to understand how tests work, then proceed to [adding_tests.md](adding_tests.md) for a practical step-by-step workflow.

## For Maintainers

See [operator_coverage.md](operator_coverage.md) to understand backend readiness evaluation, and [failure_interpretation.md](failure_interpretation.md) to triage and categorize test results.

## Getting Help

- Run a local test: see [failure_interpretation.md](failure_interpretation.md#running-openreg-tests-locally)
- Debug a failure: see [failure_interpretation.md](failure_interpretation.md#common-failure-categories-and-triage)
- Understand test expansion: see [test_instantiation.md](test_instantiation.md#how-tests-expand-from-one-definition-to-many)
