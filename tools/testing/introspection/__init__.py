"""Test-introspection tool: enumerate the concrete tests (incl. generated/synthetic)
and per-platform run/skip status of PyTorch test files, by importing them under a
simulated platform descriptor rather than requiring the target hardware.

See platforms.py for the platform descriptors and collector.py for the engine.
"""
