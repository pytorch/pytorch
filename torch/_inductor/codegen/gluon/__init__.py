"""Gluon (Triton's low-level frontend) template support for Inductor.

``gluon_template`` holds the target-neutral frontend: ``@gluon.jit`` codegen and
the op lowerings Gluon's explicit layouts force. Every other module here binds
one Gluon target namespace and takes its name -- ``cdna4`` wraps
``gluon.language.amd.cdna4``. That split follows the frontend itself, which is
namespaced per target (``gluon.language.amd`` has cdna3, cdna4, gfx1250, rdna3
and rdna4; ``gluon.language.nvidia`` has ampere through rubin) because the
families share no matmul or async-copy primitives.

A target module carries only the imports its kernel bodies need. The geometry
those bodies are written against -- matmul tile, wavefront width, staging
layouts -- stays with the kernel that offers them, as in
``inductor/kernel/flex/gluon_flex.py``.
"""
