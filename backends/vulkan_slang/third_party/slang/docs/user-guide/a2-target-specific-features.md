---
layout: user-guide
---

# Target-Specific Features

Slang can produce code for a variety of targets. When producing code for a target, Slang attempts to translate HLSL intrinsics to the closest functionality provided by the target. In addition, Slang also supports target-specific intrinsics and language extensions that allow users to make best use of the target. This chapter documents all the important target-specific behaviors.

In this chapter:

1. [SPIR-V target specific](./a2-01-spirv-target-specific.md)
2. [Metal target specific](./a2-02-metal-target-specific.md)
3. [WGSL target specific](./a2-03-wgsl-target-specific.md)

<!-- RTD-TOC-START
```{toctree}
:titlesonly:
:hidden:

SPIR-V target specific <a2-01-spirv-target-specific>
Metal target specific <a2-02-metal-target-specific>
WGSL target specific <a2-03-wgsl-target-specific>
```
RTD-TOC-END -->