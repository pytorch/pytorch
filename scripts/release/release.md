* Highlights
* Backwards Incompatible Change
* Deprecations
* New Features
* Improvements
* Bug fixes
* Performance
* Documentation
* Developers

## **Highlights**


We are excited to announce the release of PyTorch® 2.6 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.6.0))! This release features multiple improvements for PT2: `torch.compile` can now be used with Python 3.13; new performance-related knob `torch.compiler.set_stance`; several AOTInductor enhancements. Besides the PT2 improvements, another highlight is FP16 support on X86 CPUs.

NOTE: Starting with this release we are not going to publish on Conda, please see [[Announcement] Deprecating PyTorch’s official Anaconda channel](https://github.com/pytorch/pytorch/issues/138506) for the details.

For this release the experimental Linux binaries shipped with CUDA 12.6.3 (as well as Linux Aarch64,  Linux ROCm 6.2.4, and Linux XPU binaries) are built with CXX11_ABI=1 and are [using the Manylinux 2.28 build platform](https://dev-discuss.pytorch.org/t/pytorch-linux-wheels-switching-to-new-wheel-build-platform-manylinux-2-28-on-november-12-2024/2581). If you build PyTorch extensions with custom C++ or CUDA extensions, please update these builds to use CXX_ABI=1 as well and report any issues you are seeing. For the next PyTorch 2.7 release we plan to switch all Linux builds to Manylinux 2.28 and CXX11_ABI=1, please see [[RFC] PyTorch next wheel build platform: manylinux-2.28](https://github.com/pytorch/pytorch/issues/123649) for the details and discussion.

Also in this release as an important security improvement measure we have changed the default value for `weights_only` parameter of `torch.load`. This is a backward compatibility-breaking change, please see [this forum post](https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573) for more details.

This release is composed of 3892 commits from 520 contributors since PyTorch 2.5. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve PyTorch. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/pytorch-2.0/) page.


<table>
  <tr>
   <td>Beta
   </td>
   <td>Prototype
   </td>
  </tr>
  <tr>
   <td>torch.compiler.set_stance
   </td>
   <td>Improved PyTorch user experience on Intel GPUs
   </td>
  </tr>
  <tr>
   <td>torch.library.triton_op
   </td>
   <td>FlexAttention support on X86 CPU for LLMs
   </td>
  </tr>
  <tr>
   <td>torch.compile support for Python 3.13
   </td>
   <td>Dim.AUTO
   </td>
  </tr>
  <tr>
   <td>New packaging APIs for AOTInductor
   </td>
   <td>CUTLASS and CK GEMM/CONV Backends for AOTInductor
   </td>
  </tr>
  <tr>
   <td>AOTInductor: minifier
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>AOTInductor: ABI-compatible mode code generation
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>FP16 support for X86 CPUs
   </td>
   <td>
   </td>
  </tr>
</table>


*To see a full list of public feature submissions click [here](https://docs.google.com/spreadsheets/d/1TzGkWuUMF1yTe88adz1dt2mzbIsZLd3PBasy588VWgk/edit?usp=sharing).


### BETA FEATURES


#### **[Beta] torch.compiler.set_stance**

This feature enables the user to specify different behaviors (“stances”) that `torch.compile` can take between different invocations of compiled functions. One of the stances, for example, is 

“eager_on_recompile”, that instructs PyTorch to code eagerly when a recompile is necessary, reusing cached compiled code when possible.

For more information please refer to the [set_stance documentation](https://pytorch.org/docs/2.6/generated/torch.compiler.set_stance.html#torch.compiler.set_stance) and the [Dynamic Compilation Control with torch.compiler.set_stance](https://pytorch.org/tutorials/recipes/torch_compiler_set_stance_tutorial.html) tutorial.

**[Beta] torch.library.triton_op**

`torch.library.triton_op` offers a standard way of creating custom operators that are backed by user-defined triton kernels. 

When users turn user-defined triton kernels into custom operators, `torch.library.triton_op` allows `torch.compile` to peek into the implementation, enabling `torch.compile` to optimize the triton kernel inside it.

For more information please refer to the [triton_op documentation](https://pytorch.org/docs/2.6/library.html#torch.library.triton_op) and the[ Using User-Defined Triton Kernels with torch.compile](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) tutorial.

**[Beta] torch.compile support for Python 3.13**

`torch.compile` previously only supported Python up to version 3.12. Users can now optimize models with `torch.compile` in Python 3.13. 

**[Beta] New packaging APIs for AOTInductor**

A new package format, “[PT2 archive](https://docs.google.com/document/d/1RQ4cmywilnFUT1VE-4oTGxwXdc8vowCSZsrRgo3wFA8/edit?usp=sharing)”, has been introduced. This essentially contains a zipfile of all the files that need to be used by AOTInductor, and allows users to send everything needed to other environments. There is also functionality to package multiple models into one artifact, and to store additional metadata inside of the package.

For more details please see the updated [torch.export AOTInductor Tutorial for Python runtime](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html).

**[Beta] AOTInductor: minifier**

If a user encounters an error while using AOTInductor APIs, AOTInductor Minifier allows creation of a minimal nn.Module that reproduces the error.

For more information please see the [AOTInductor Minifier documentation](https://pytorch.org/docs/2.6/torch.compiler_aot_inductor_minifier.html).

**[Beta] AOTInductor: ABI-compatible mode code generation**

AOTInductor-generated model code has dependency on Pytorch cpp libraries. As Pytorch evolves quickly, it’s important to make sure previously AOTInductor compiled models can continue to run on newer Pytorch versions, i.e. AOTInductor is backward compatible. 

In order to guarantee application binary interface (ABI) backward compatibility, we have carefully defined a set of stable C interfaces in libtorch and make sure AOTInductor generates code that only refers to the specific set of APIs and nothing else in libtorch. We will keep the set of C APIs stable across Pytorch versions and thus provide backward compatibility guarantees for AOTInductor-compiled models.

**[Beta] FP16 support for X86 CPUs (both eager and Inductor modes)**

Float16 datatype is commonly used for reduced memory usage and faster computation in AI inference and training. CPUs like the recently launched [Intel® Xeon® 6 with P-Cores](https://www.intel.com/content/www/us/en/products/details/processors/xeon/xeon6-p-cores.html) support Float16 datatype with native accelerator [AMX](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html). Float16 support on X86 CPUs was introduced in PyTorch 2.5 as a prototype feature, and now it has been further improved for both eager mode and Torch.compile + Inductor mode, making it Beta level feature with both functionality and performance verified with a broad scope of workloads.


### PROTOTYPE FEATURES

**[Prototype] Improved PyTorch user experience on Intel GPUs**

PyTorch user experience on Intel GPUs is further improved with simplified installation steps, Windows release binary distribution and expanded coverage of supported GPU models including the latest Intel® Arc™ B-Series discrete graphics. Application developers and researchers seeking to fine-tune, inference and develop with PyTorch models on [Intel® Core™ Ultra AI PCs ](https://www.intel.com/content/www/us/en/products/docs/processors/core-ultra/ai-pc.html)and [Intel® Arc™ discrete graphics](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) will now be able to directly install PyTorch with binary releases for Windows, Linux and Windows Subsystem for Linux 2.



* Simplified Intel GPU software stack setup to enable one-click installation of the torch-xpu PIP wheels to run deep learning workloads in an out of the box fashion, eliminating the complexity of installing and activating Intel GPU development software bundles.
* Windows binary releases for torch core, torchvision and torchaudio have been made available for Intel GPUs, and the supported GPU models have been expanded from Intel® Core™ Ultra Processors with Intel® Arc™ Graphics, [Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html) and [Intel® Arc™ A-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/a-series/overview.html) to the latest GPU hardware [Intel® Arc™ B-Series graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/b-series/overview.html).
* Further enhanced coverage of Aten operators on Intel GPUs with SYCL* kernels for smooth eager mode execution, as well as bug fixes and performance optimizations for torch.compile on Intel GPUs.

For more information regarding Intel GPU support, please refer to [Getting Started Guide](https://pytorch.org/docs/main/notes/get_start_xpu.html).

**[Prototype] FlexAttention support on X86 CPU for LLMs**

FlexAttention was initially introduced in PyTorch 2.5 to provide optimized implementations for Attention variants with a flexible API. In PyTorch 2.6, X86 CPU support for FlexAttention was added through TorchInductor CPP backend. This new feature leverages and extends current CPP template abilities to support broad attention variants (e.x.: PageAttention, which is critical for LLMs inference) based on the existing FlexAttention API, and brings optimized performance on x86 CPUs. With this feature, it’s easy to use FlexAttention API to compose Attention solutions on CPU platforms and achieve good performance.

**[Prototype] Dim.AUTO**

`Dim.AUTO` allows usage of automatic dynamic shapes with `torch.export`. Users can export with `Dim.AUTO `and “discover” the dynamic behavior of their models, with min/max ranges, relations between dimensions, and static/dynamic behavior being automatically inferred.

This is a more user-friendly experience compared to the existing named-Dims approach for specifying dynamic shapes, which requires the user to fully understand the dynamic behavior of their models at export time. `Dim.AUTO` allows users to write generic code that isn’t model-dependent, increasing ease-of-use for exporting with dynamic shapes.

Please see [torch.export tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes) for more information.

**[Prototype] CUTLASS and CK GEMM/CONV Backends for AOTInductor**

The CUTLASS and CK backend adds kernel choices for GEMM autotuning in Inductor. This is now also available in AOTInductor which can run in C++ runtime environments. A major improvement to the two backends is improved compile-time speed by eliminating redundant kernel binary compilations and dynamic shapes support.




## **Backwards Incompatible changes**

### Flip default torch.load to weights_only ([#137602](https://github.com/pytorch/pytorch/pull/137602), [#138225](https://github.com/pytorch/pytorch/pull/138225), [#138866](https://github.com/pytorch/pytorch/pull/138866), [#139221](https://github.com/pytorch/pytorch/pull/139221), [#140304](https://github.com/pytorch/pytorch/pull/140304), [#138936](https://github.com/pytorch/pytorch/pull/138936), [#139541](https://github.com/pytorch/pytorch/pull/139541), [#140738](https://github.com/pytorch/pytorch/pull/140738), [#142153](https://github.com/pytorch/pytorch/pull/142153), [#139433](https://github.com/pytorch/pytorch/pull/139433))

We are closing the loop on the deprecation that started in 2.4 and flipped `torch.load` to use `weights_only=True` by default.


When this flag is set, instead of using the usual pickle module, `torch.load` uses a custom unpickler constrained to call only functions and classes needed for loading state dictionaries and basic types.


While this change is disruptive for users serializing more than basic types, we expect the increased security by default is a tradeoff that is worth it. Do note that, even though this default is safer, we still recommend only loading trusted checkpoints and rely on more constrained (and even safer) formats like [safetensors](https://github.com/huggingface/safetensors) for un-trusted checkpoints.


For full details, please refer to [this dev-discuss post](https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573).


### Anaconda deprecation in CD. Remove anaconda dependency in Magma builds ([#141024](https://github.com/pytorch/pytorch/pull/141024)) ([#141281](https://github.com/pytorch/pytorch/pull/141281)) ([#140157](https://github.com/pytorch/pytorch/pull/140157)) ([#139888](https://github.com/pytorch/pytorch/pull/139888)) ([#140141](https://github.com/pytorch/pytorch/pull/140141))  ([#139924](https://github.com/pytorch/pytorch/pull/139924)) ([#140158](https://github.com/pytorch/pytorch/pull/140158)) ([#142019](https://github.com/pytorch/pytorch/pull/142019))  ([#142276](https://github.com/pytorch/pytorch/pull/142276)) ([#142277](https://github.com/pytorch/pytorch/pull/142277))  ([#142282](https://github.com/pytorch/pytorch/pull/142282))

PyTorch will stop publishing Anaconda packages that depend on Anaconda’s default packages. We are directing users to utilize our official wheel packages from download.pytorch.org or PyPI, or switch to utilizing conda-forge (pytorch) packages if they would like to continue to use conda. For more details refer to [this announcement](https://github.com/pytorch/pytorch/issues/138506)

### Added Manylinux 2.28 prototype support and CXX11_ABI=1 for following binaries: Linux CUDA 12.6, Linux aarch64 CPU, Linux aarch64 GPU CUDA 12.6, ROCm 6.2.4, Linux XPU  ([#139894](https://github.com/pytorch/pytorch/pull/139894)) ([#139631](https://github.com/pytorch/pytorch/pull/139631)) ([#139636](https://github.com/pytorch/pytorch/pull/139636)) ([#140743](https://github.com/pytorch/pytorch/pull/140743)) ([#137696](https://github.com/pytorch/pytorch/pull/137696)) ([#141565](https://github.com/pytorch/pytorch/pull/141565)) ([#140681](https://github.com/pytorch/pytorch/pull/140681)) ([#141609](https://github.com/pytorch/pytorch/pull/141609)) ([#141704](https://github.com/pytorch/pytorch/pull/141704)) ([#141423](https://github.com/pytorch/pytorch/pull/141423)) ([#141609](https://github.com/pytorch/pytorch/pull/141609))

The PyTorch binaries shipped with CUDA 12.6.3 are built with CXX11_ABI=1 and are using the Manylinux 2.28 build platform. If you are building PyTorch extensions with custom C++ or CUDA extensions, please update these builds to use CXX_ABI=1 as well and report any issues you are seeing. For the next PyTorch 2.7 release we plan to switch all Linux builds to Manylinux 2.28 and CXX11_ABI=1, please see [[RFC] PyTorch next wheel build platform: manylinux-2.28](https://github.com/pytorch/pytorch/issues/123649) for the details and discussion.


## **Deprecations**
### Releng

### Removed CUDA 12.1 support in CI/CD ([#141271](https://github.com/pytorch/pytorch/pull/141271)) ([#142177](https://github.com/pytorch/pytorch/pull/142177))
The full release compatibility matrix matrix can be found in [release.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)

### Deprecated `c10d::onCompletionHook` ([#142390](https://github.com/pytorch/pytorch/pull/142390))

* In PT 2.5 and before, users can do:
  ```py
  pg = dist.init_process_group()
  def hook(work_info: torch._C._distributed_c10d.WorkInfo):
    # do something
  pg._register_on_completion_hook(hook)

  # The hook will be triggered after the collective complete
  pg.broadcast([tensor]).wait()
  ```
* Starting from PT 2.6, when users write the code above, they will get get a warning message “ProcessGroupNCCL OnCompletion hook will be deprecated in favor of Flight Recorder”

## **New features**

### Python Frontend

* Introduce a device-agnostic runtime API design ([#132204](https://github.com/pytorch/pytorch/pull/132204))
* Add validation for ambiguous behavior in `Tensor.dim_order()` ([#141632](https://github.com/pytorch/pytorch/pull/141632))
* Add type check for `ord` argument for `torch.linalg.{vector,matrix}_norm()` ([#137463](https://github.com/pytorch/pytorch/pull/137463))
* FlexAttention support for NJT ([#136792](https://github.com/pytorch/pytorch/pull/136792), [#140723](https://github.com/pytorch/pytorch/pull/140723))

### Miscellaneous

* Enable forward AD in `functional.affine_grid` ([#135494](https://github.com/pytorch/pytorch/pull/135494))
* Added SVE support for ARM CPUs ([#119571](https://github.com/pytorch/pytorch/pull/119571))
* User buffer registration via MemPool API ([#133603](https://github.com/pytorch/pytorch/pull/133603))
* Add in_order flag for data loader, allowing out-of-order dataloading ([#141833](https://github.com/pytorch/pytorch/pull/141833))


### Optim

* Add Support for Tracking Parameter Names (named_parameters) in Optimizer State Dict ([#134107](https://github.com/pytorch/pytorch/pull/134107))
* Support tensor betas in Adam and AdamW ([#134171](https://github.com/pytorch/pytorch/pull/134171))


### Distributed

* c10d
    * Made ProcessGroup initialization non-blocking when `device_id` is given [#138527](https://github.com/pytorch/pytorch/pull/138527))
    * Allowed sub group to be eagerly inited even if default one is not ([#138665](https://github.com/pytorch/pytorch/pull/138665))
    * Supported `group_dst`/`group_src` in c10d collectives ([#140460](https://github.com/pytorch/pytorch/pull/140460), [#139677](https://github.com/pytorch/pytorch/pull/139677), [#140827](https://github.com/pytorch/pytorch/pull/140827), [#140843](https://github.com/pytorch/pytorch/pull/140843), [#140847](https://github.com/pytorch/pytorch/pull/140847))
    * Enabled Flight Recorder buffer for all users ([#142260](https://github.com/pytorch/pytorch/pull/142260))
    * Registered Intel distributed Backend (`XCCL`) in PyTorch distributed package ([#141856](https://github.com/pytorch/pytorch/pull/141856))
* Pipeline
    * Performed shape inference at runtime using user-provided real tensors ([#136912](https://github.com/pytorch/pytorch/pull/136912))
    * Added ZBV schedule ([#142084](https://github.com/pytorch/pytorch/pull/142084))
* FSDP2
    * Moved FSDP2 to public ([#141868](https://github.com/pytorch/pytorch/pull/141868))


### Dynamo

* Add `torch.compiler.set_stance` to dynamically change `torch.compile` behavior without needing to re-apply `torch.compile`. ([#137504](https://github.com/pytorch/pytorch/pull/137504))
* Profile guided optimization for `automatic_dynamic` - automatically save and load automatic dynamic decisions to reuse on future runs ([#139001](https://github.com/pytorch/pytorch/pull/139001))
* `skip_guard_eval_unsafe` compiler stance option for power users - skip guard checks when it is known to be safe to do so ([#140251](https://github.com/pytorch/pytorch/pull/140251))


### Releng

* Added support for CUDA 12.6 in CI/CD ([#142335](https://github.com/pytorch/pytorch/pull/142335)) ([#136321](https://github.com/pytorch/pytorch/pull/136321)) ([#138417](https://github.com/pytorch/pytorch/pull/138417)) ([#138563](https://github.com/pytorch/pytorch/pull/138563)) ([#138562](https://github.com/pytorch/pytorch/pull/138562))  ([#139909](https://github.com/pytorch/pytorch/pull/139909)) ([#138899](https://github.com/pytorch/pytorch/pull/138899)) ([#141365](https://github.com/pytorch/pytorch/pull/141365)) ([#141433](https://github.com/pytorch/pytorch/pull/141433))  ([#141805](https://github.com/pytorch/pytorch/pull/141805)) ([#141976](https://github.com/pytorch/pytorch/pull/141976)) ([#139988](https://github.com/pytorch/pytorch/pull/139988))  ([#140143](https://github.com/pytorch/pytorch/pull/140143)) ([#141377](https://github.com/pytorch/pytorch/pull/141377)) ([#142064](https://github.com/pytorch/pytorch/pull/142064))
* Intel GPU enablement in CI/CD. Upgrade XPU support packages to Intel® Deep Learning Essentials 2025.0. Add prototype Linux and Windows binary builds with XPU runtime pypi packages dependencies. ([#138189](https://github.com/pytorch/pytorch/pull/138189)) ([#139050](https://github.com/pytorch/pytorch/pull/139050)) ([#139604](https://github.com/pytorch/pytorch/pull/139604)) ([#139775](https://github.com/pytorch/pytorch/pull/139775)) ([#140373](https://github.com/pytorch/pytorch/pull/140373)) ([#141546](https://github.com/pytorch/pytorch/pull/141546)) ([#141775](https://github.com/pytorch/pytorch/pull/141775)) ([#141135](https://github.com/pytorch/pytorch/pull/141135)) ([#142210](https://github.com/pytorch/pytorch/pull/142210)) ([#135638](https://github.com/pytorch/pytorch/pull/135638)) ([#142298](https://github.com/pytorch/pytorch/pull/142298))
* Added Python 3.13 in CI/CD support and prototype support for Python 3.13t in CD (Only Linux and Linux aarch64 torch binaries)  ([#136001](https://github.com/pytorch/pytorch/pull/136001)) ([#137396](https://github.com/pytorch/pytorch/pull/137396)) ([#138037](https://github.com/pytorch/pytorch/pull/138037)) ([#138629](https://github.com/pytorch/pytorch/pull/138629)) ([#140137](https://github.com/pytorch/pytorch/pull/140137)) ([#138095](https://github.com/pytorch/pytorch/pull/138095)) ([#141572](https://github.com/pytorch/pytorch/pull/141572)) ([#140733](https://github.com/pytorch/pytorch/pull/140733)) ([#141264](https://github.com/pytorch/pytorch/pull/141264)) ([#142294](https://github.com/pytorch/pytorch/pull/142294)) ([#137142](https://github.com/pytorch/pytorch/pull/137142)) ([#137127](https://github.com/pytorch/pytorch/pull/137127)) ([#139533](https://github.com/pytorch/pytorch/pull/139533)) ([#140733](https://github.com/pytorch/pytorch/pull/140733))


### ROCM

* Added AMDSMI support for UUID input ([#129741](https://github.com/pytorch/pytorch/pull/129741))
* Added faster HW support for packed bfloat16 and fp16 for MI300 ([#135770](https://github.com/pytorch/pytorch/pull/135770))
* Improved performance of reductions on 1D and 2D tensors. ([#137737](https://github.com/pytorch/pytorch/pull/137737))


### XPU

* Add `torch.xpu.mem_get_info` API: Introduces a new API to retrieve memory information for XPU devices. ([#141230](https://github.com/pytorch/pytorch/pull/141230))
* Add architecture property to XPU device: Adds new properties to XPU devices to query architecture details. ([#138186](https://github.com/pytorch/pytorch/pull/138186))
* Add `elapsed_time` method for XPU events: Introduces a method to measure elapsed time between XPU events. ([#140865](https://github.com/pytorch/pytorch/pull/140865))
* Add `torch.xpu.get_arch_list` and `torch.xpu.get_gencode_flags`: Introduces new APIs to retrieve architecture lists and code generation flags for XPU. ([#137773](https://github.com/pytorch/pytorch/pull/137773))
* Add quantized convolution support for XPU backend ([#133080](https://github.com/pytorch/pytorch/pull/133080))
* Enable XPU device support for LSTMCell operators ([#140246](https://github.com/pytorch/pytorch/pull/140246))


### Profiler

* Hide ProfilerStep Alignment behind Experimental Config ([#137668](https://github.com/pytorch/pytorch/pull/137668))
* Add functionality to call dump function of NCCL profiler plugin ([#137523](https://github.com/pytorch/pytorch/pull/137523))


### Export

* Add `torch.export.export_for_training()` API to perform export that can run training. Note that this replaces the non-documented `capture_pre_autograd_graph` feature ([#135374](https://github.com/pytorch/pytorch/pull/135374), [#135918](https://github.com/pytorch/pytorch/pull/135918), [#135549](https://github.com/pytorch/pytorch/pull/135549), [#143224](https://github.com/pytorch/pytorch/pull/143224))
* New packaging APIs for AOTInductor `torch._inductor.aoti_compile_and_package` 
    * Previously, AOTInductor (through `torch._export.aot_compile`), would return a path to a .so. However, this does not have a great user experience as actually there are other files that are used along with the .so, for example .cubin files and serialized extern kernels. So, we introduce a new package format, “[PT2 archive](https://docs.google.com/document/d/1RQ4cmywilnFUT1VE-4oTGxwXdc8vowCSZsrRgo3wFA8/edit#heading=h.v2y2jgnwc56a)”, which is what we intend to have AOTInductor return. This essentially contains a zipfile of all the files that need to be used by AOTInductor, and allows users to send to other environments. There is also functionality to package multiple models into one artifact, and to store additional metadata inside of the package.
* [AOTInductor Minifier](https://pytorch.org/docs/main/torch.compiler_aot_inductor_minifier.html). If you encounter an error while using AOT Inductor APIs such as `torch._inductor.aoti_compile_and_package`, `torch._indcutor.aoti_load_package`, or running the loaded model of aoti_load_package on some inputs, you can use the AOTInductor Minifier to create a minimal nn.Module that reproduces the error. ([#139351](https://github.com/pytorch/pytorch/pull/139351),[#140999](https://github.com/pytorch/pytorch/pull/140999), [#141159](https://github.com/pytorch/pytorch/pull/141159), [#141156](https://github.com/pytorch/pytorch/pull/141156))
* AOTInductor: ABI-compatible mode code generation. In order to guarantee ABI backward compatibility, we have carefully defined a set of stable C interfaces in libtorch and make sure AOTInductor generates code that only refers to the specific set of APIs and nothing else in libtorch. We will keep the set of C APIs stable across Pytorch versions and thus provide BC guarantees for AOTInductor-compiled models.
* `export.export_for_inference` and `export.exported_program.core_aten_decompositions` API. `export_for_inference` returns a functional, post-dispatch ATen IR. ([#135912](https://github.com/pytorch/pytorch/pull/135912)).

### Inductor

* Move stack allocation related configs in AOTI ([#139093](https://github.com/pytorch/pytorch/pull/139093)). All configs now have a aot_inductor prefix, so `torch.compile(options={"use_minimal_arrayref_interface": True})(foo)` is now `torch.compile(options={"aot_inductor.use_minimal_arrayref_interface": True})(foo)` and `torch.compile(options={"allow_stack_allocation": True})(foo)` is now `torch.compile(options={"aot_inductor.allow_stack_allocation": True})(foo)`.
* Move `torch._utils.is_compiling` to `torch.compiler.is_compiling` ([#127690](https://github.com/pytorch/pytorch/pull/127690)) Rewrite `torch._utils.is_compiling()` to `torch.compiler.is_compiling()`.
* Added option `​​autotune_num_choices_displayed` to control number of kernel options displayed ([#138788](https://github.com/pytorch/pytorch/pull/138788))
* Added option `force_pointwise_cat` concat support through inductor using pointwise kernels ([#141966](https://github.com/pytorch/pytorch/pull/141966)). This forces concat to be generated as a pointwise op with masked loads.
* New config option `annotate_training` that adds Inductor annotations to NVTX.  ([#130429](https://github.com/pytorch/pytorch/pull/130429))
* Introduces an option `triton_kernel_default_layout_constraint` to tweak stride settings for user-defined Triton kernels, enhancing customization and flexibility ([#135530](https://github.com/pytorch/pytorch/pull/135530)).
* User can patch inductor config to enable strict custom kernel layout constraints by changing `torch.compile(options={"triton_kernel_default_layout_constraint": "needs_fixed_stride_order"})(foo)` for torch._inductor.config ([#135581](https://github.com/pytorch/pytorch/pull/135581)). 
* External callable registration API `register_external_matmul` for Matmul tuning candidates in Inductor ([#130774](https://github.com/pytorch/pytorch/pull/130774)).
* Adds support for Windows Arm64 to enhance platform compatibility ([#133088](https://github.com/pytorch/pytorch/pull/133088)).
* Integrates support for AMD triton stream pipeliner in ROCm to enhance performance ([#139881](https://github.com/pytorch/pytorch/pull/139881)).
* Adds support for TRITON_INTERPRET in Inductor ([#140841](https://github.com/pytorch/pytorch/pull/140841)).
* Adds update_constant_buffer pybind support in AOTInductor ([#140755](https://github.com/pytorch/pytorch/pull/140755)).
* Provides an option `package_constants_in_so` to exclude weights from .so files in AOTInductor ([#141997](https://github.com/pytorch/pytorch/pull/141997)).
* Adds `load_constants` to the package API ([#142246](https://github.com/pytorch/pytorch/pull/142246)).
* Enables auto functionalize v2 by default ([#136685](https://github.com/pytorch/pytorch/pull/136685)).
* Adds raise_error_on_ignored_optimization to the aoti config ([#138035](https://github.com/pytorch/pytorch/pull/138035)).
* Adds stats summary (mean/min/max, etc) for jit inductor tensor value printing ([#135887](https://github.com/pytorch/pytorch/pull/135887)).

## **Improvements**

### Python Frontend

* Add support for fp16 and bf16 to `torch.special.i1` ([#137899](https://github.com/pytorch/pytorch/pull/137899))
* Add option to disable checksum computation in `torch.save` ([#137735](https://github.com/pytorch/pytorch/pull/137735))
* Speed up fp16 tensors printing ([#141927](https://github.com/pytorch/pytorch/pull/141927))
* Add support for fp16 for `torch.adaptive_pool3d` on cpu ([#136091](https://github.com/pytorch/pytorch/pull/136091))
* Add support for fp8* to `torch.masked_select` ([#141928](https://github.com/pytorch/pytorch/pull/141928))
* Add support for complex fp16 to fill_empty_deterministic_ ([#137488](https://github.com/pytorch/pytorch/pull/137488))
* Remove dependency on numpy for serialization for XLA/open registration devices without numpy ([#137444](https://github.com/pytorch/pytorch/pull/137444), [#137600](https://github.com/pytorch/pytorch/pull/137600))
* Fix `torch.{linalg.}norm` complex half support ([#133661](https://github.com/pytorch/pytorch/pull/133661))



### NN Frontend

* Allow global module hook to accept keyword arguments ([#137403](https://github.com/pytorch/pytorch/pull/137403))
* Add APIs to separate norm calculation and gradient scaling in ``nn.utils.clip_grad_norm_`` ([#139662](https://github.com/pytorch/pytorch/pull/139662))
* Add Half support for reflection and replication padding on CPU ([#135931](https://github.com/pytorch/pytorch/pull/135931))
* Add `weight` argument to MSELoss, HuberLoss and L1Loss ([#132049](https://github.com/pytorch/pytorch/pull/132049))
* Gaussian nll loss scalar variance support ([#138931](https://github.com/pytorch/pytorch/pull/138931))
* Added validation for input types for `torch.nn.Linear` and `torch.nn.Bilinear` ([#135596](https://github.com/pytorch/pytorch/pull/135596))


### Optim



* Improve `ReduceLROnPlateau` and `Optimizer.add_param_group` interaction by auto-updating `min_lrs` ([#137637](https://github.com/pytorch/pytorch/pull/137637))
* Allow `SequentialLR` to include `ChainedScheduler` ([#133450](https://github.com/pytorch/pytorch/pull/133450))


### Composability


##### **Decompositions, FakeTensor and meta tensors**

Operator decompositions, FakeTensors and meta tensors are used to trace out a graph in `torch.compile` and `torch.export`. They received several improvements:


* Several operator decomps received improvements/bugfixes:
    * `aten.split_with_sizes` ([#135728](https://github.com/pytorch/pytorch/pull/135728))
    * `aten.max_unpool2d/aten.max_unpool3d` ([#133146](https://github.com/pytorch/pytorch/pull/133146))
    * `aten.dot` ([#138596](https://github.com/pytorch/pytorch/pull/138596))
    * `aten.layer_norm` ([#140557](https://github.com/pytorch/pytorch/pull/140557))
    * `aten.scaled_dot_product_attention` ([#135297](https://github.com/pytorch/pytorch/pull/135297))
    * `aten.matmul` ([#134568](https://github.com/pytorch/pytorch/pull/134568))
    * `aten._embedding_bag`  ([#136774](https://github.com/pytorch/pytorch/pull/136774))
    * `aten.native_group_norm/aten.native_layer_norm` ([#137079](https://github.com/pytorch/pytorch/pull/137079))
    * `aten.to(..., non_blocking=True)` ([#136513](https://github.com/pytorch/pytorch/pull/136513))
    * `Aten.addmm` ([#138520](https://github.com/pytorch/pytorch/pull/138520))
    * General fixes:
        * `out= dtype` checks for unary ops  ([#140288](https://github.com/pytorch/pytorch/pull/140288))
* New decompositions for a few pytorch operators:
    * `aten.diagonal_copy` ([#136730](https://github.com/pytorch/pytorch/pull/136730))
* Several meta implementations of operators received improvements/bugfixes:
    * `Aten.triangular_solve` ([#140186](https://github.com/pytorch/pytorch/pull/140186))
    * `Aten.log_softmax` ([#140289](https://github.com/pytorch/pytorch/pull/140289))
* New meta tensor implementations for a few pytorch operators:
    * `aten._segment_reduce_backward` ([#137442](https://github.com/pytorch/pytorch/pull/137442))
    * `Aten._add_relu` ([#140009](https://github.com/pytorch/pytorch/pull/140009))

**Dynamic shapes**

We made many improvements and bugfixes to dynamic shapes in `torch.compile`


* Minor error message improvements ([#136671](https://github.com/pytorch/pytorch/pull/136671), [#138310](https://github.com/pytorch/pytorch/pull/138310))
* Make `native_layer_norm_backward` work with unbacked SymInts ([#136798](https://github.com/pytorch/pytorch/pull/136798))
* Make `masked_fill` work with unbacked SymIntsl ([#137060](https://github.com/pytorch/pytorch/pull/137060))
* Improve tracing speed of `torch.cat` with large numbers of symbolic variables ([#139653](https://github.com/pytorch/pytorch/pull/139653))
* Improve performance of `canonicalize_bool_expr` ([#135621](https://github.com/pytorch/pytorch/pull/135621))
* Improve performance of `sympy_generic_le` ([#135622](https://github.com/pytorch/pytorch/pull/135622))
* Simplify expr before getting implications in `_maybe_evaluate_static` ([#135499](https://github.com/pytorch/pytorch/pull/135499))
* use a fast expand algorithm ([#135999](https://github.com/pytorch/pytorch/pull/135999), [#136163](https://github.com/pytorch/pytorch/pull/136163))
* Fix calling `Add._from_args` and `Mul._from_args` ([#136143](https://github.com/pytorch/pytorch/pull/136143))
* Dynamic shape logging improvements in tlparse ([#136508](https://github.com/pytorch/pytorch/pull/136508), [#141068](https://github.com/pytorch/pytorch/pull/141068), [#140867](https://github.com/pytorch/pytorch/pull/140867))
* Avoid some quadratic behavior of dynamic shapes involving aliasing + mutation of graph inputs ([#136857](https://github.com/pytorch/pytorch/pull/136857))
* Tensorify compute on Python scalars ([#136674](https://github.com/pytorch/pytorch/pull/136674))
* Delay mul/pow expansion for `_SympyT` to enable more folding ([#138235](https://github.com/pytorch/pytorch/pull/138235))
* Fix bug in unbacked_bindings for a*u0 ([#138136](https://github.com/pytorch/pytorch/pull/138136))
* Remove parallel_and and parallel_or ([#138135](https://github.com/pytorch/pytorch/pull/138135))
* Explicitly avoid recording when should_record_events is false in record_shapeenv_event ([#138965](https://github.com/pytorch/pytorch/pull/138965))
* Better support for dynamic shapes with tensor subclasses ([#125941](https://github.com/pytorch/pytorch/pull/125941))
* support symfloats in translation validation ([#139457](https://github.com/pytorch/pytorch/pull/139457))
* Add trunc to z3 validator ([#140886](https://github.com/pytorch/pytorch/pull/140886))
* Refactor ShapeGuardPrinter for future C++ additon ([#140968](https://github.com/pytorch/pytorch/pull/140968))
* Fix another item memo loss location + bool specialization bug ([#139587](https://github.com/pytorch/pytorch/pull/139587))
* Optimize increment summations ([#140822](https://github.com/pytorch/pytorch/pull/140822))
* Only compute new_untracked_symbols and `new_unbacked_bindings` if needed. ([#140083](https://github.com/pytorch/pytorch/pull/140083))
* Use `has_free_unbacked_symbols` instead of `bool(free_unbacked_symbols)` ([#140027](https://github.com/pytorch/pytorch/pull/140027))
* Try to simplify FloorDiv axioms implications when needed during evaluations. ([#141267](https://github.com/pytorch/pytorch/pull/141267))
* Fix AttributeError: 'int' object has no attribute 'node' due to constant prop ([#141250](https://github.com/pytorch/pytorch/pull/141250))
* Update tensorify pass to specialize symfloats we didn't tensorify away ([#139564](https://github.com/pytorch/pytorch/pull/139564))
* Add `TORCHDYNAMO_EXTENDED_ADVICE` ([#137159](https://github.com/pytorch/pytorch/pull/137159)) ([#137196](https://github.com/pytorch/pytorch/pull/137196))
* Do not try to optimize new implications in `get_implications` ([#139738](https://github.com/pytorch/pytorch/pull/139738))


**Custom operators**

We improved the existing `torch.library` APIs and added new ones.


* Add new `torch.library.triton_op` API ([#141880](https://github.com/pytorch/pytorch/pull/141880))
* Fix partitioner behavior on user triton kernels ([#136878](https://github.com/pytorch/pytorch/pull/136878))
* Add links to new Custom Ops Landing Page ([#137933](https://github.com/pytorch/pytorch/pull/137933), [#139634](https://github.com/pytorch/pytorch/pull/139634))
* Fix `torch.library.register_vmap` to work with nested vmap ([#137306](https://github.com/pytorch/pytorch/pull/137306))
* No-op `torch.library.custom_op` APIs on `torch.deploy` ([#139509](https://github.com/pytorch/pytorch/pull/139509))
* Optimize mutable `torch.library.custom_op` overhead ([#139513](https://github.com/pytorch/pytorch/pull/139513))
* Improve `torch.library.opcheck` and `register_autograd` docs ([#141883](https://github.com/pytorch/pytorch/pull/141883))


### Distributed

* c10d
    * Added FP8 support to NaN checker ([#135891](https://github.com/pytorch/pytorch/pull/135891), [#135961](https://github.com/pytorch/pytorch/pull/135961), [#136115](https://github.com/pytorch/pytorch/pull/136115))
    * Added support for `cuStreamWriteValue32` ([#136488](https://github.com/pytorch/pytorch/pull/136488))
    * Improved the detection robustness in `CudaDMAConnectivityDetector` ([#137530](https://github.com/pytorch/pytorch/pull/137530))
    * Simplified barrier implementation and further decouple CPU/GPU synchronization ([#137516](https://github.com/pytorch/pytorch/pull/137516))
    * Threw value error if passing `world_size=0` to `TCPStore` ([#137792](https://github.com/pytorch/pytorch/pull/137792))
    * Performed retry connection timeout failures in socket ([#138003](https://github.com/pytorch/pytorch/pull/138003))
    * Added an API to get the future result(success or failure) of a collective and customized error handling ([#137799](https://github.com/pytorch/pytorch/pull/137799))
    * Disabled watchdog thread in blockingWait mode ([#138001](https://github.com/pytorch/pytorch/pull/138001))
    * Added default value for ``nccl_nonblocking_timeout`` ([#138374](https://github.com/pytorch/pytorch/pull/138374))
    * Ensured nccl comm is ready before all accesses ([#138384](https://github.com/pytorch/pytorch/pull/138384))
    * Used a promise to delay watchdog shutdown ([#138828](https://github.com/pytorch/pytorch/pull/138828))
    * Supported optional backend if `device_id` provided ([#140963](https://github.com/pytorch/pytorch/pull/140963))
    * Supported group ranks in `P2POp` and `batch_isend_irecv` ([#141054](https://github.com/pytorch/pytorch/pull/141054))
    * Enabled `CudaEventCache` by default and add multi device support ([#140975](https://github.com/pytorch/pytorch/pull/140975))
    * Added an API to retrieve default distributed backend from device ([#140536](https://github.com/pytorch/pytorch/pull/140536))
    * Supported rank, world size, group name/desc overrides for `PyProcessGroup` ([#141529](https://github.com/pytorch/pytorch/pull/141529))
    * Added the detect of accelerator type when backend is not specified ([#142216](https://github.com/pytorch/pytorch/pull/142216))
    * Used task submitter TLS in gloo working threads ([#142184](https://github.com/pytorch/pytorch/pull/142184))
    * Added ``_reduce_scatter_base`` to ``c10d::ProcessGroupUCC`` ([#138021](https://github.com/pytorch/pytorch/pull/138021))
* DDP
    * Made `DDPOptimizer` work with HOPs ([#138787](https://github.com/pytorch/pytorch/pull/138787))
    * Made DDP Quantization hooks backend Agnostic ([#138816](https://github.com/pytorch/pytorch/pull/138816))
    * Used device-agnostic runtime API in DDP/FSDP instead of `cuda` device specific. ([#137678](https://github.com/pytorch/pytorch/pull/137678))
* FSDP
    * Updates real device in FSDP `state_dict_utils` ([#134994](https://github.com/pytorch/pytorch/pull/134994))
    * Generalized of FSDP common for non-cuda execution ([#133209](https://github.com/pytorch/pytorch/pull/133209))
* FSDP2
    * Added ``_set_unshard_async_op`` ([#135523](https://github.com/pytorch/pytorch/pull/135523))
    * Added module, mp policy to ``fsdp_pre_all_gather`` ([#136129](https://github.com/pytorch/pytorch/pull/136129))
    * Added check for contiguous parameters ([#137000](https://github.com/pytorch/pytorch/pull/137000))
    * Relaxed even sharding requirement for `all-gather` extensions ([#137005](https://github.com/pytorch/pytorch/pull/137005))
    * Used stream and event based on device ([#136843](https://github.com/pytorch/pytorch/pull/136843))
    * Added ``shard_placement_fn`` arg ([#137496](https://github.com/pytorch/pytorch/pull/137496))
    * Added ``set_unshard_in_backward(bool)`` ([#137922](https://github.com/pytorch/pytorch/pull/137922))
    * Made module-to-state mapping use weakrefs ([#139650](https://github.com/pytorch/pytorch/pull/139650))
    * Removed CUDA-like device check in fsdp2. ([#139539](https://github.com/pytorch/pytorch/pull/139539))
* DTensor
    * Allowed user to manual_seed different seed on device mesh and only synced RNG state in WORLD when manual_seed has not been called ([#141223](https://github.com/pytorch/pytorch/pull/141223))
    * Supported `matmul` in inference_mode ([#142197](https://github.com/pytorch/pytorch/pull/142197))
* Pipeline
    * Made `PipelineStage` support meta initialization ([#136243](https://github.com/pytorch/pytorch/pull/136243))
    * Allowed non-0 stages to accept kwargs ([#136416](https://github.com/pytorch/pytorch/pull/136416))
    * added schedule simulator and chrometrace dump ([#138134](https://github.com/pytorch/pytorch/pull/138134))
    * Supported separate dI / dW and V-schedules ([#131762](https://github.com/pytorch/pytorch/pull/131762))
    * Updated schedules to use I, B actions. ([#138886](https://github.com/pytorch/pytorch/pull/138886))
    * Added type checking to _backward functions ([#140019](https://github.com/pytorch/pytorch/pull/140019))
    * Allowed multiple backward grads ([#140981](https://github.com/pytorch/pytorch/pull/140981))
    * Improved schedule csv loading ([#142009](https://github.com/pytorch/pytorch/pull/142009))
* TorchElastic
    * Added TryExcept when decoding healthcheck port ([#136574](https://github.com/pytorch/pytorch/pull/136574))
    * Skipped store barrier and store get in host assign ([#136865](https://github.com/pytorch/pytorch/pull/136865))
* Checkpoint
    * Throw an error when state_dict and saved tensors are different sizes ([#141571](https://github.com/pytorch/pytorch/pull/141571))

### Profiler



* Create Auto-Trace Frontend for Trace ID ([#139310](https://github.com/pytorch/pytorch/pull/139310))
* Add skip_first_wait to profiler.schedule ([#141512](https://github.com/pytorch/pytorch/pull/141512))
* Add CUDA Overhead to Auto-trace ([#142271](https://github.com/pytorch/pytorch/pull/142271))


### Nested Tensor

* Added NJT operator support: `rms_norm()`, `embedding_bag()`, `record_stream()`, `rad2deg()`, `embedding()` backward, activation functions ([#135872](https://github.com/pytorch/pytorch/pull/135872), [#135888](https://github.com/pytorch/pytorch/pull/135888), [#140736](https://github.com/pytorch/pytorch/pull/140736), [#138627](https://github.com/pytorch/pytorch/pull/138627), [#137099](https://github.com/pytorch/pytorch/pull/137099), [#140290](https://github.com/pytorch/pytorch/pull/140290))
* Mixed NJT, dense binary pointwise broadcasting support ([#133021](https://github.com/pytorch/pytorch/pull/133021))
* Allow any single non-batch dim to be ragged for NJT ([#137125](https://github.com/pytorch/pytorch/pull/137125))
* Add bfloat16 support to `torch.bmm(NST, NST)` ([#141380](https://github.com/pytorch/pytorch/pull/141380))
* Add missing fp classification functions for NST ([#139890](https://github.com/pytorch/pytorch/pull/139890))


### Functorch

* Add vmap support for `torch.scatter_reduce` ([#135547](https://github.com/pytorch/pytorch/pull/135547))
* Add vmap support for `native_dropout_backward` ([#140140](https://github.com/pytorch/pytorch/pull/140140))
* Allow optional positional arguments for `torch.func.functional_call` ([#134643)](https://github.com/pytorch/pytorch/pull/134643))


### Quantization



* Add uint16 support for observer ([#136238](https://github.com/pytorch/pytorch/pull/136238))
* change flatten recipe for `X86InductorQuantizer` ([#136298](https://github.com/pytorch/pytorch/pull/136298))
* Update choose_qparams_per_token op to output correct shape for scales and zp ([#136807](https://github.com/pytorch/pytorch/pull/136807))
* Make QAT Fused modules torchscriptable ([#136285](https://github.com/pytorch/pytorch/pull/136285))
* Add missing mappings to support `torch.uint16` in quantization and export ([#136547](https://github.com/pytorch/pytorch/pull/136547))
* Default to use training IR ([#137804](https://github.com/pytorch/pytorch/pull/137804))
* Remove Redundant Method in X86 Quantizer ([#139161](https://github.com/pytorch/pytorch/pull/139161))
* Add bfloat16 support for per tensor/channel cpu/cuda fake quantize ops ([#139306](https://github.com/pytorch/pytorch/pull/139306))
* add `linear_dynamic_fp16` ops for OneDNN ([#140376](https://github.com/pytorch/pytorch/pull/140376))
* annotate and convert for `linear_dynamic_fp16` for x86 ([#141480](https://github.com/pytorch/pytorch/pull/141480))


### Releng

* Updated CUDNN to 9.5.1.17 for CUDA 12.6 builds, Linux and Windows  ([#137978](https://github.com/pytorch/pytorch/pull/137978))
* upgrade CI/CD to 6.2.4 for ROCm ([#141423](https://github.com/pytorch/pytorch/pull/141423))




### Cuda

* Extend `cuda_flip` to unsigned types ([#137781](https://github.com/pytorch/pytorch/pull/137781))
* SDPA Priority Manager accepts ordering ([#140467](https://github.com/pytorch/pytorch/pull/140467))
* cuDNN Attention memory layout handling improvements ([#141147](https://github.com/pytorch/pytorch/pull/141147)) ([#138354](https://github.com/pytorch/pytorch/pull/138354))


### Mps



* Add native im2col ([#135706](https://github.com/pytorch/pytorch/pull/135706))
* Add `upsample_bicubic2d` as Metal op ([#136123](https://github.com/pytorch/pytorch/pull/136123))
* Add `scatter_reduce.two` ([#141948](https://github.com/pytorch/pytorch/pull/141948))
* Add i0 op ([#137849](https://github.com/pytorch/pytorch/pull/137849))
* Add `torch.special.i1` op ([#140196](https://github.com/pytorch/pytorch/pull/140196))
* Add `unfold_backward` on MPS ([#135411](https://github.com/pytorch/pytorch/pull/135411))
* Add `isposinf` and `isneginf` ([#136689](https://github.com/pytorch/pytorch/pull/136689))
* Add `MetalShaderLibrary::getFunctionNames()` ([#141499](https://github.com/pytorch/pytorch/pull/141499))
* Add `tri[lu]_indices` ([#137648](https://github.com/pytorch/pytorch/pull/137648))
* Fix Gamma for bfloat16 dtypes ([#136981](https://github.com/pytorch/pytorch/pull/136981))
* Extend `fmin`/`fmax`/`copysign` and `nextafter` to bfloat16 ([#136982](https://github.com/pytorch/pytorch/pull/136982))
* Enable bucketization for bfloat16 ([#136983](https://github.com/pytorch/pytorch/pull/136983))
* Fix bfloat16 to complex casts ([#137070](https://github.com/pytorch/pytorch/pull/137070))
* Enable `arange` to bfloat16 ([#136754](https://github.com/pytorch/pytorch/pull/136754))
* Enable `torch.linalg.cross` for bfloat16 ([#136984](https://github.com/pytorch/pytorch/pull/136984))
* Enable Renorm for bfloat16 ([#136985](https://github.com/pytorch/pytorch/pull/136985))
* Enable `nan_to_num` for bfloat16 ([#136986](https://github.com/pytorch/pytorch/pull/136986))
* Add support for bfloat16 autocast ([#139390](https://github.com/pytorch/pytorch/pull/139390))
* Eliminate `c10::value_or_else` ([#138818](https://github.com/pytorch/pytorch/pull/138818))
* Compile kernels into Metallib ([#138636](https://github.com/pytorch/pytorch/pull/138636))
* Write/Invoke Metal shaders from C++ ([#141547](https://github.com/pytorch/pytorch/pull/141547))
* Support `torch.Event` for MPS ([#142468](https://github.com/pytorch/pytorch/pull/142468))
* Add CompileShader method ([#141478](https://github.com/pytorch/pytorch/pull/141478))
* Reintroduce support for convolutions with output_channels > 65536 ([#140726](https://github.com/pytorch/pytorch/pull/140726))


### ROCM

* Improve PyTorch build speed in ROCm environment by Downloading AOTriton from GitHub unless AOTRITON_INSTALL_FROM_SOURCE=1 is set ([#136603](https://github.com/pytorch/pytorch/pull/136603))
* enable gfx110x architecture for hipblaslt ([#137317](https://github.com/pytorch/pytorch/pull/137317))


### XPU

* Improves the device index bound checking mechanism for XPU. ([#120768](https://github.com/pytorch/pytorch/pull/120768))
* Use default context on Windows for Intel GPU: Improves XPU device handling on Windows by using the default context. ([#138049](https://github.com/pytorch/pytorch/pull/138049))
* Add device guard for XPU structured operators in torchgen ([#138802](https://github.com/pytorch/pytorch/pull/138802))
* Generalize device-bias code to align XPU unroll reduction with CUDA ([#142348](https://github.com/pytorch/pytorch/pull/142348))
* Generalize CUDA C++ wrapper for reuse by XPU ([#135312](https://github.com/pytorch/pytorch/pull/135312))

### Miscellaneous

* Add `torch.float8e4m3fn` dtype support to semi-structured sparse ([#136397](https://github.com/pytorch/pytorch/pull/136397))
* Faster BatchSampler ([#137423](https://github.com/pytorch/pytorch/pull/137423))
* Init threadpool with user defined `num_threads` before default ([#136793](https://github.com/pytorch/pytorch/pull/136793), [#137051](https://github.com/pytorch/pytorch/pull/137051))


### Dynamo


* `automatic_dynamic_shapes_mark_as` - adds an option to cause automatic dynamic shapes to trigger unbacked SymInts rather than backed SymInts ([#141415](https://github.com/pytorch/pytorch/pull/141415))
* Propagate detailed source location information of shape guards to guards/recompiles output ([#136917](https://github.com/pytorch/pytorch/pull/136917))
* `torch.compile` support for Python 3.13 ([#139533](https://github.com/pytorch/pytorch/pull/139533))
* Trace through dynamic callables on tensor variables ([#137940](https://github.com/pytorch/pytorch/pull/137940))
* Trace through dataclasses ([#141294](https://github.com/pytorch/pytorch/pull/141294))
* Graph region tracking for deduplication (i.e. common subgraph extraction) ([#141381](https://github.com/pytorch/pytorch/pull/141381))
* Scan higher order op ([#134102](https://github.com/pytorch/pytorch/pull/134102))
* Trace subclasses of namedtuple type ([#140534](https://github.com/pytorch/pytorch/pull/140534))
* Trace dict subclasses ([#143548](https://github.com/pytorch/pytorch/pull/143548))


### Export



* Preserve preserve the call signature for a module when it was called multiple times ([#137999](https://github.com/pytorch/pytorch/pull/137999), [#138669](https://github.com/pytorch/pytorch/pull/138669))
* Let `export` preserves `node.meta["custom"]` field ([#138266](https://github.com/pytorch/pytorch/pull/138266))
* Add `neg` and `pos` operator to `serde/serialize` ([#138309](https://github.com/pytorch/pytorch/pull/138309), [#143343](https://github.com/pytorch/pytorch/pull/143343))
* Update min_val and max_val to Optional[int] in serialization and allow the schema to express infinity ([#139394](https://github.com/pytorch/pytorch/pull/139394))



### Fx



* Bypass custom **setattr** in Node.**init** ([#135733](https://github.com/pytorch/pytorch/pull/135733))
* Add new replacement_callback to materialize a replacement just in time ([#135553](https://github.com/pytorch/pytorch/pull/135553))
* Minor optimization in create_arg ([#135821](https://github.com/pytorch/pytorch/pull/135821))
* Replace _snake_case with a regexp ([#135822](https://github.com/pytorch/pytorch/pull/135822))
* Update `_inline_module` util function to work with both args and kwargs ([#136631](https://github.com/pytorch/pytorch/pull/136631))
* Fx graph always return tuple in fuse_as_graphmodule ([#139236](https://github.com/pytorch/pytorch/pull/139236))
* Change fx graph `_replace_hook` to a list of Callable ([#142006](https://github.com/pytorch/pytorch/pull/142006))
* Avoid generation of empty merge cpu submodule by splitter v2 ([#140794](https://github.com/pytorch/pytorch/pull/140794))
* Make split_module work with `keep_original_order=True` and no-op graph ([#141340](https://github.com/pytorch/pytorch/pull/141340))
* Add output_node util function to `fx.Graph` ([#139770](https://github.com/pytorch/pytorch/pull/139770))
* Fix `stride` in TensorMetadata to always be a `Tuple[int, ...]` ([#141106](https://github.com/pytorch/pytorch/pull/141106))
* Enhance `from_node` node meta to track source recursively ([#142066](https://github.com/pytorch/pytorch/pull/142066))
* Support linear/BN fusion and follow the API guideline ([#141585](https://github.com/pytorch/pytorch/pull/141585))
* Enable `fuse_by_partitions` to always return output as tuple ([#142056](https://github.com/pytorch/pytorch/pull/142056))
* Add safer check for isatty in fx/_utils.py ([#140876](https://github.com/pytorch/pytorch/pull/140876))


### Inductor


* Switch GPU codegen to one-pass in AOTI ([#141980](https://github.com/pytorch/pytorch/pull/141980))
* Fix multi-kernel codegen when using one-pass in AOTI ([#142333](https://github.com/pytorch/pytorch/pull/142333))
* Fix an issue when fallback op does not return a value in AOTI ([#142339](https://github.com/pytorch/pytorch/pull/142339))
* Improve the stride preservation logic of user-visible outputs ([#136732](https://github.com/pytorch/pytorch/pull/136732))
* Add workspace to TritonTemplates ([#138050](https://github.com/pytorch/pytorch/pull/138050))
* Enable Cpp wraper for Intel GPU. ([#135318](https://github.com/pytorch/pytorch/pull/135318))
* Flip `custom_op_default_layout_constraint` in Inductor to optimize tensor layout ([#135239](https://github.com/pytorch/pytorch/pull/135239)).
* Enables coordinate descent tuning with max-autotune in Inductor ([#136867](https://github.com/pytorch/pytorch/pull/136867)).
* Adds `relu_nan_to_num` option for handling NaNs in pre-grad passes in AOTInductor ([#138545](https://github.com/pytorch/pytorch/pull/138545)).
* Enables cooperative and persistent reductions in Inductor ([#138533](https://github.com/pytorch/pytorch/pull/138533)).
* Introduces multi-kernel support alongside cooperative reductions in Inductor ([#138893](https://github.com/pytorch/pytorch/pull/138893)).
* Adds new configs `env_name_default` and `env_name_force` for better configuration management ([#138956](https://github.com/pytorch/pytorch/pull/138956)).
* Adjusts loop split optimization heuristic ([#137550](https://github.com/pytorch/pytorch/pull/137550)).
* Enhances numerical precision for fp32 in FlexAttention on ROCm devices using IEEE ([#135702](https://github.com/pytorch/pytorch/pull/135702)).
* Enables SDPA pattern matching in Inductor for CUDA, enhancing optimization capabilities ([#137085](https://github.com/pytorch/pytorch/pull/137085)).
* Updates Inductor's support for Triton AttrsDescriptor ([#137757](https://github.com/pytorch/pytorch/pull/137757)).
* Update C++ runner API to take a const vector ([#139955](https://github.com/pytorch/pytorch/pull/139955))



## **Bug fixes**

### Python Frontend



* Fix `torch.mean(..., out=)` for fp16 and bf16 on CPU ([#135174](https://github.com/pytorch/pytorch/pull/135174))
* Fix serialization for `torch.uint16`, `torch.uint32`, `torch.uint64` ([#137184](https://github.com/pytorch/pytorch/pull/137184))
* Fix Tensor preservation logic to not lose user-defined attributes in some cases ([#137267](https://github.com/pytorch/pytorch/pull/137267))
* Fix memory leak in `torch.utils.module_tracker.ModuleTracker` ([#141960](https://github.com/pytorch/pytorch/pull/141960))



### NN Frontend



* Fix `nn.functional.softshrink` returning 0 on NAN input  ([#138421](https://github.com/pytorch/pytorch/pull/138421))
* Fix flex_decode to build offsets off of strides ([#139516](https://github.com/pytorch/pytorch/pull/139516))




### Autograd Frontend



* Fix `torch.nn.EmbeddingBag` when per_sample_weights is differentiable but embedding weights are not ([#142338](https://github.com/pytorch/pytorch/pull/142338))
* Determine autograd engine ready queue based on InputMetadata instead of InputBuffer ([#135633](https://github.com/pytorch/pytorch/pull/135633))


### Composability



* Fixed a correctness issue when `torch.compiling` `torch.scaled_dot_product_attention`, in the case where the scale argument is a dynamic shape ([#141728](https://github.com/pytorch/pytorch/pull/141728))
* Fixed a correctness issue when `torch.compiling` `torch.rrelu`, in the case where it mutates any module buffers ([#136008](https://github.com/pytorch/pytorch/pull/136008))





### Distributed



* c10d
    * Fixed extra context on device 0 ([#135273](https://github.com/pytorch/pytorch/pull/135273))
    * Fixed bugs in non-blocking mode ([#137741](https://github.com/pytorch/pytorch/pull/137741))
    * Fixed P2P data corruption in non-blocking mode ([#138860](https://github.com/pytorch/pytorch/pull/138860))
    * Made sure not use split for P2P comm creation ([#139013](https://github.com/pytorch/pytorch/pull/139013))
    * Used long/short wait for different non-blocking calls ([#142291](https://github.com/pytorch/pytorch/pull/142291))
    * Recorded device index for GPU guarding during NCCLComm method calls ([#141270](https://github.com/pytorch/pytorch/pull/141270))
    * Fixed the behavior of `destroy_process_group` ([#141510](https://github.com/pytorch/pytorch/pull/141510))
    * Reworked NCCLComm destructor to avoid clash with CUDA driver shutdown ([#141511](https://github.com/pytorch/pytorch/pull/141511))
    * Removed Option for `ProcessGroup` and Expose backend `Options` to reflect the correct code structure ([#132931](https://github.com/pytorch/pytorch/pull/132931)) ([#135653](https://github.com/pytorch/pytorch/pull/135653))
    * Fixed prefix store segmentation fault ([#136872](https://github.com/pytorch/pytorch/pull/136872))
    * Fixed a race condition in one-shot `all-reduce` ([#137257](https://github.com/pytorch/pytorch/pull/137257))
    * Enforced contiguity for `all-reduce` ([#137345](https://github.com/pytorch/pytorch/pull/137345))
    * Fixed data corruption bug after `CUDAEventCache` is enabled ([#138040](https://github.com/pytorch/pytorch/pull/138040))
    * Enforced contiguity for `alltoall` ([#141816](https://github.com/pytorch/pytorch/pull/141816))
    * Fixed sequence numbers for coalesced operations ([#135132](https://github.com/pytorch/pytorch/pull/135132))
    * Fixed color value for comm split being negative ([#137855](https://github.com/pytorch/pytorch/pull/137855))
    * Fixed a logic of using `ncclCommSplit` ([#138781](https://github.com/pytorch/pytorch/pull/138781))
    * Caught tensor.numel() == 0 in NaN detector ([#140741](https://github.com/pytorch/pytorch/pull/140741))
    * Fixed a breakage in `IntraNodeComm::rendezvous()` ([#141200](https://github.com/pytorch/pytorch/pull/141200))
    * Fixed `_are_we_tracing()` in dynamo for functional collectives ([#142075](https://github.com/pytorch/pytorch/pull/142075))
* DeviceMesh
    * Fixed ``from_group`` when passing a tensor `mesh` ([#137713](https://github.com/pytorch/pytorch/pull/137713))
* DTensor
    * Fixed 2D DTensor `mm` with mesh_shape (1, n) or (n, 1) ([#139134](https://github.com/pytorch/pytorch/pull/139134))
    * Removed the adhoc DTensor RNG tracker `TensorParallelRNGTracker` since it does not match FSDP2+TP ([#141220](https://github.com/pytorch/pytorch/pull/141220))
* DistributedStateDict (DSD)
    * Initialize lr as a tensor if it is originally a tensor ([#141620](https://github.com/pytorch/pytorch/pull/141620))
* FSDP2
    * Fixed 2D mismatched grad placements ([#136237](https://github.com/pytorch/pytorch/pull/136237))
    * Fixed ``test_all_gather_extensions_monkey_patch`` ([#136130](https://github.com/pytorch/pytorch/pull/136130))
    * Fixed mistargeted backward prefetch ([#137348](https://github.com/pytorch/pytorch/pull/137348))
    * Fixed incorrect tensor meta after `.to(dtype)` ([#137593](https://github.com/pytorch/pytorch/pull/137593))
    * Gated dynamo import for torch deploy ([#137203](https://github.com/pytorch/pytorch/pull/137203))
    * Fixed CUDA sync for bf16 HSDP AR, fp32 params ([#140044](https://github.com/pytorch/pytorch/pull/140044))
    * Fixed backward-compatible imports ([#142419](https://github.com/pytorch/pytorch/pull/142419))
    * Gated PT2 code for torch deploy ([#142456](https://github.com/pytorch/pytorch/pull/142456))
* Pipeline
    * Fixed py ref cycle in `stage_backward` ([#136507](https://github.com/pytorch/pytorch/pull/136507))
    * Fixed more leaks and check leaks in tests ([#136584](https://github.com/pytorch/pytorch/pull/136584))
    * Removed modifications to autograd nodes in Zero Bubble schedule ([#136678](https://github.com/pytorch/pytorch/pull/136678))
    * Fixed extra memory usage in Zero Bubble ([#138119](https://github.com/pytorch/pytorch/pull/138119))
    * Fixed last backward counting for dI / dW ([#139415](https://github.com/pytorch/pytorch/pull/139415))
    * Forward fixed for `_validate_schedule` ([#142211](https://github.com/pytorch/pytorch/pull/142211))
    * Allowed schedules to run with single stage ([#138925](https://github.com/pytorch/pytorch/pull/138925))
    * Freed memory usage earlier in last stage ([#138504](https://github.com/pytorch/pytorch/pull/138504))
* TorchElastic
    * Fixed store prefix race in `rendezvous` ([#136768](https://github.com/pytorch/pytorch/pull/136768))
    * Fixed rendezvous error due to `EtcdStore` get method not waiting in some cases ([#137056](https://github.com/pytorch/pytorch/pull/137056))
    * Fixed the bug caused by wrong host address in creating `TCPStore` server inside dynamic rendezvous ([#139702](https://github.com/pytorch/pytorch/pull/139702))
* Checkpoint
    * Fix fsspec transaction failure cleanup in multithreaded environments ([#135541](https://github.com/pytorch/pytorch/pull/135541))



### Dynamo



* Fix tracing of NumPy 2 ops ([#138686](https://github.com/pytorch/pytorch/pull/138686))
* Don’t graph break on inner `torch.compile` ([#135819](https://github.com/pytorch/pytorch/pull/135819))
* Various closure/cell variable/mutation related fixes ([#136891](https://github.com/pytorch/pytorch/pull/136891), [#139339](https://github.com/pytorch/pytorch/pull/139339), [#140155](https://github.com/pytorch/pytorch/pull/140155))
* Stop importing some third party libraries ([#136334](https://github.com/pytorch/pytorch/pull/136334), [#142502](https://github.com/pytorch/pytorch/pull/142502), [#142503](https://github.com/pytorch/pytorch/pull/142503))



### Nested Tensor Frontend



* Fix NJT operator support: `sum()`, `unsqueeze()`, `to()` on non-contiguous NJTs, `where()`, `select()`, `chunk()`, reductions ([#131945](https://github.com/pytorch/pytorch/pull/131945), [#141392](https://github.com/pytorch/pytorch/pull/141392), [#137124](https://github.com/pytorch/pytorch/pull/137124), [#141500](https://github.com/pytorch/pytorch/pull/141500), [#139317](https://github.com/pytorch/pytorch/pull/139317), [#141506](https://github.com/pytorch/pytorch/pull/141506), [#141604](https://github.com/pytorch/pytorch/pull/141604))
* Fix NJT `linear_backward()` memory usage using a more efficient formula ([#141163](https://github.com/pytorch/pytorch/pull/141163))
* Fix NJT serialization ([#137031](https://github.com/pytorch/pytorch/pull/137031))

### Cuda



* Add missing boundary checks to cunn_SoftMaxForward ([#140682](https://github.com/pytorch/pytorch/pull/140682))
* Fix CTC cuda backend out-of-bound access ([#141607](https://github.com/pytorch/pytorch/pull/141607))
* Fixed cuda sanitizer and as_subclass calls ([#138218](https://github.com/pytorch/pytorch/pull/138218))


### Mps



* Allow nan mean reduction in `nll_loss` ([#135434](https://github.com/pytorch/pytorch/pull/135434))
* Fix AvgPool2d for float16 ([#136822](https://github.com/pytorch/pytorch/pull/136822))
* Error checking/bfloat16 support for `torch.normal` ([#136863](https://github.com/pytorch/pytorch/pull/136863))
* Fix reduction ops outputs for empty tensors ([#139446](https://github.com/pytorch/pytorch/pull/139446))
* Restrict MSELoss to floating types ([#139960](https://github.com/pytorch/pytorch/pull/139960))
* Fix conv backward pass for channels last ([#141009](https://github.com/pytorch/pytorch/pull/141009))
* Add autocast rule for SDPA ([#141776](https://github.com/pytorch/pytorch/pull/141776))
* Release MetalShaderLibrary cached resources ([#142053](https://github.com/pytorch/pytorch/pull/142053))
* Fixes SiLU on non-contiguous tensors ([#139006](https://github.com/pytorch/pytorch/pull/139006))
* Fix `channels_last_3d` in `nn.Conv3d` ([#141780](https://github.com/pytorch/pytorch/pull/141780))
* Guard on flash attention SymFloat scale instead of incorrectly casting to float ([#141725](https://github.com/pytorch/pytorch/pull/141725))
* Fix memory leak from unreleased NSProcessInfo ([#142052](https://github.com/pytorch/pytorch/pull/142052))

### ROCM



* Fixed out of memory errors on AMD triton backend ([#139883](https://github.com/pytorch/pytorch/pull/139883))
* Correct numerical issues in layer norm backwards kernel ([#140259](https://github.com/pytorch/pytorch/pull/140259))

### XPU



* Resolves an issue with duplicated build environments in XPU Linux CI. ([#141546](https://github.com/pytorch/pytorch/pull/141546))
* Fix XPU support packages version: Corrects the versioning of XPU support packages. ([#138189](https://github.com/pytorch/pytorch/pull/138189))
* Fix `c10::Event` unit test failure on XPU backend ([#141800](https://github.com/pytorch/pytorch/pull/141800))
* Fix mismatched tensor metadata between FakeTensor and XPU concrete tensor in `F.logsigmoid` ([#141333](https://github.com/pytorch/pytorch/pull/141333))
* Fix memory stats error on XPU: Corrects an error in memory statistics for XPU devices. ([#135818](https://github.com/pytorch/pytorch/pull/135818))
* Fix XPU CMake typo: Corrects a typo in XPU CMake configuration. ([#140374](https://github.com/pytorch/pytorch/pull/140374))
* Fix an issue causing endless code regeneration in non-XPU environments. ([#140438](https://github.com/pytorch/pytorch/pull/140438))
* Fix incorrect device check before skipping concat linear in Inductor XPU. ([#140916](https://github.com/pytorch/pytorch/pull/140916))



### Profiler



* Clear Out Dangling AppendOnlyLists after collection ([#137450](https://github.com/pytorch/pytorch/pull/137450))
* Fix `UnicodeDecodeError: 'utf-8' codec can't decode byte` ([#139062](https://github.com/pytorch/pytorch/pull/139062))
* Fix ASAN Overflow Issues ([#140441](https://github.com/pytorch/pytorch/pull/140441))
* Fix devices Parameter Type in benchmark_utilization Function ([#138774](https://github.com/pytorch/pytorch/pull/138774))[ ](https://github.com/pytorch/pytorch/pull/138774)


### Quantization



* Pass `ideep:lowp_kind` to` matmul_forward::compute` on cache misses ([#135058](https://github.com/pytorch/pytorch/pull/135058))
* fix re-export custom metadata ([#135282](https://github.com/pytorch/pytorch/pull/135282), [#135634](https://github.com/pytorch/pytorch/pull/135634), [#135720](https://github.com/pytorch/pytorch/pull/135720))
* moving eps in `torchao/quantization/utils.py` to targeted device to avoid device mismatch issue ([#135204](https://github.com/pytorch/pytorch/pull/135204))
* Add type check for `dilation` in `torch.quantized_max_pool3d()` ([#137845](https://github.com/pytorch/pytorch/pull/137845))
* Pass all arguments when quantizing embedding bag from float ([#137697](https://github.com/pytorch/pytorch/pull/137697))
* Fix for split gates enabled quantizable LSTM subclass ([#140818](https://github.com/pytorch/pytorch/pull/140818))
* Fix ReLU fusion when conv/linear has > 1 user for XNNPACK ([#140846](https://github.com/pytorch/pytorch/pull/140846))
* Fix RecursionError when `prepare_pt2e` graph with concat of the same node ([#141651](https://github.com/pytorch/pytorch/pull/141651))



### Sparse Frontend



* Fix memory leak in MaskedTensor when using autograd ([#137890](https://github.com/pytorch/pytorch/pull/137890))
* Fix `bmm(COO, dense)` illegal memory access for some shapes ([#131977](https://github.com/pytorch/pytorch/pull/131977))
* Fix MaskedTensor binary ops for `sparse_csr` layout ([#134335](https://github.com/pytorch/pytorch/pull/134335))


### Miscellaneous



* Fix PyBind 2.10.4 compatibility issue ([#141456](https://github.com/pytorch/pytorch/pull/141456))
* correctly keep track of processed tensors for foreach reductions (norm, max) ([#140103](https://github.com/pytorch/pytorch/pull/140103))
* Fixes to `torch.package` for 3.13 ([#141409](https://github.com/pytorch/pytorch/pull/141409))


### Export



* Do not deserialize arguments with default values as kwargs ([#136036](https://github.com/pytorch/pytorch/pull/136036))
* Fix `_get_non_persistent_buffers` for duplicate submodules ([#136552](https://github.com/pytorch/pytorch/pull/136552))
* Fix lifted constants order for 0-input graphs ([#136658](https://github.com/pytorch/pytorch/pull/136658))
* Handle attribute assignment detection and registered buffer assignments in `make_fx`([#137240](https://github.com/pytorch/pytorch/pull/137240))
* Fix specialization bug in unflatten + preserve_module_call_signature ([#137363](https://github.com/pytorch/pytorch/pull/137363))
* Fix `export` for constant outputs ([#137547](https://github.com/pytorch/pytorch/pull/137547), [#137993](https://github.com/pytorch/pytorch/pull/137993))
* Fix param and buffer mapping for state_dict when there are state_dict hooks ([#137609](https://github.com/pytorch/pytorch/pull/137609))
* Fix export retracing ([#137733](https://github.com/pytorch/pytorch/pull/137733))
* Fix non-strict retracing with kwargs ([#138927](https://github.com/pytorch/pytorch/pull/138927))
* Fix assigning tensor with requires_grad as constant in export ([#137997](https://github.com/pytorch/pytorch/pull/137997))
* Fix issue with runtime_assertions with in `export_for_training` ([#138292](https://github.com/pytorch/pytorch/pull/138292))
* Fix issue in move pass for copying Parameter ([#138855](https://github.com/pytorch/pytorch/pull/138855))
* Fix unflatten with HOPs ([#138978](https://github.com/pytorch/pytorch/pull/138978))
* Fix unflattening to handle multiple specialized graphs corresponding to multiple calls to the same submodule ([#137013](https://github.com/pytorch/pytorch/pull/137013))
* Allow autocast in training ir export ([#137287](https://github.com/pytorch/pytorch/pull/137287))
* Fix unlift to preserve aliased constants ([#137310](https://github.com/pytorch/pytorch/pull/137310))
* Handle` AttrProxy._modules` when module is overwritten as None ([#139957](https://github.com/pytorch/pytorch/pull/139957))
* Fix joint graph metadata ([#136011](https://github.com/pytorch/pytorch/pull/136011))
* Fix mapping issue with `torch.Size` ([#137465](https://github.com/pytorch/pytorch/pull/137465))
* Fix `test_lazy_module_kwargs` ([#137705](https://github.com/pytorch/pytorch/pull/137705))
* Propagate ShapeEnv during lowering ([#138362](https://github.com/pytorch/pytorch/pull/138362))
* Plumb `is_export` flag to `FunctionalTensorMode` in analysis pass ([#138836](https://github.com/pytorch/pytorch/pull/138836))


### Fx



* Add  `__init__.py` to shape inference folder. ([#135461](https://github.com/pytorch/pytorch/pull/135461))
* Handle `sympy.oo` in bitwise_and/or value_ranges ([#141522](https://github.com/pytorch/pytorch/pull/141522))
* Fixes issue with enums in a tuple for dynamo ([#133123](https://github.com/pytorch/pytorch/pull/133123))
* Add output node to split_module subgraphs ([#139275](https://github.com/pytorch/pytorch/pull/139275))
* Fix deep copy of empty graph ([#141660](https://github.com/pytorch/pytorch/pull/141660))


### Inductor



* Fix a bug with not enabling the Python dispatcher in AOTInductor ([#135933](https://github.com/pytorch/pytorch/pull/135933))
* Don't run reshape pattern match on dynamic shape size tensor ([#136100](https://github.com/pytorch/pytorch/pull/136100))
* Make DtypeView work with cpp_wrapper without `abi_compatible` ([#136233](https://github.com/pytorch/pytorch/pull/136233))
* Check size hints to determine indexing dtype in Triton ([#137234](https://github.com/pytorch/pytorch/pull/137234))
* Fix an error in `_dynamo.compiled_autograd.reset()` ([#137889](https://github.com/pytorch/pytorch/pull/137889))
* Fix out-of-bounds array access in `atomic_add_vec` ([#138744](https://github.com/pytorch/pytorch/pull/138744))
* Update zero size computation in `clone_preserve_strides` ([#139224](https://github.com/pytorch/pytorch/pull/139224), [#139458](https://github.com/pytorch/pytorch/pull/139458))
* Fix for gcc10 `torch.compile` compiler error when `march=aarch64+sve` ([#137795](https://github.com/pytorch/pytorch/pull/137795))
* Fix a cubin file path issue ([#139848](https://github.com/pytorch/pytorch/pull/139848))
* Fix caching issue with AOTI packaging ([#140022](https://github.com/pytorch/pytorch/pull/140022))
* Fix a two-pass kernel missmatch in AOTI ([#141041](https://github.com/pytorch/pytorch/pull/141041))
* Fix performance bug by removing `copy_misaligned_inputs` from AOTI ([#142136](https://github.com/pytorch/pytorch/pull/142136))
* Fix mask bug in `torch.cat` kernel ([#140838](https://github.com/pytorch/pytorch/pull/140838))
* Fixed max-autotune in FlexAttention to reset kernel options appropriately ([#138733](https://github.com/pytorch/pytorch/pull/138733))
* Don't set XBLOCK larger than xnumel ([#138730](https://github.com/pytorch/pytorch/pull/138730))
* Fix inductor CPU `masked()` body codegen when result dtype is bool and operator is where ([#138486](https://github.com/pytorch/pytorch/pull/138486))
* Fix typo in `codegen_dynamic_scalar` ([#138760](https://github.com/pytorch/pytorch/pull/138760))
* Fix `ReinterpretView` call in `TMADescriptor` IR ([#138759](https://github.com/pytorch/pytorch/pull/138759))
* Fix free symbol handling in FlexAttention ([#138794](https://github.com/pytorch/pytorch/pull/138794))
* Fix codegen for `tl.constexpr` globals ([#138757](https://github.com/pytorch/pytorch/pull/138757))
* Force strides for efficient attention backward ([#138879](https://github.com/pytorch/pytorch/pull/138879))
* Make AOT inductor treat None args correctly ([#139114](https://github.com/pytorch/pytorch/pull/139114))
* Fix a bug with arg ordering in handling dynamic shapes ([#139777](https://github.com/pytorch/pytorch/pull/139777))
* Fixing missing ck package warning when the backend is disabled ([#139790](https://github.com/pytorch/pytorch/pull/139790))
* Force contiguous layout for implicit fallback ([#140996](https://github.com/pytorch/pytorch/pull/140996))
* Fix another IMA with captured buffers ([#141164](https://github.com/pytorch/pytorch/pull/141164))
* Inductor dtype propagation fixes ([#141495](https://github.com/pytorch/pytorch/pull/141495))
* Fix broadcast logic for Triton ([#141027](https://github.com/pytorch/pytorch/pull/141027)) ([#141693](https://github.com/pytorch/pytorch/pull/141693))
* Fix grid codegen for configs with empty kwargs ([#141824](https://github.com/pytorch/pytorch/pull/141824))
* Fix issue in CPP GEMM Template Prune Tensor ([#141798](https://github.com/pytorch/pytorch/pull/141798))
* Fix max-autotune bug with captured buffer grads ([#141531](https://github.com/pytorch/pytorch/pull/141531))
* TritonTemplate dtype fixes ([#141991](https://github.com/pytorch/pytorch/pull/141991))
* Fix device error for `NopKernelSchedulerNode` ([#141372](https://github.com/pytorch/pytorch/pull/141372))
* Resolves an issue where `try_solve` fails when both symbols are unknown and their product is zero ([#137919](https://github.com/pytorch/pytorch/pull/137919)).
* Resolves an issue where a fallback operation returned `None`, preventing potential errors in AOTI initialization ([#135997](https://github.com/pytorch/pytorch/pull/135997)).
* Resolves test failures following the update of pybind11 to version 2.13.6 ([#136280](https://github.com/pytorch/pytorch/pull/136280)).
* Corrects the maximum autotuning for single-thread dynamic shapes in Inductor ([#136418](https://github.com/pytorch/pytorch/pull/136418)).
* Fixes FMA codegen for Halide backend to ensure correct operation behavior ([#136810](https://github.com/pytorch/pytorch/pull/136810)).
* Corrects `max-autotune` behavior when dealing with View nodes in FlexAttention ([#137204](https://github.com/pytorch/pytorch/pull/137204)).
* Adjust BlockMask handling when reused from a larger sequence length ([#137255](https://github.com/pytorch/pytorch/pull/137255)).
* Corrects `triton_reshape` by properly expanding the Min keyword in code generation ([#137357](https://github.com/pytorch/pytorch/pull/137357)).
* Corrects `reduction_hint` behavior for single-element sums ([#137754](https://github.com/pytorch/pytorch/pull/137754)).
* Resolves a codecache `write_atomic` issue on Windows ([#138331](https://github.com/pytorch/pytorch/pull/138331)).
* Fixes AOTI data type codegen for symbolic integers ([#138106](https://github.com/pytorch/pytorch/pull/138106)).
* Resolves an issue where passing `None` arguments to user-defined Triton kernels caused errors ([#138472](https://github.com/pytorch/pytorch/pull/138472)).
* Correctly sets keyword arguments when creating Buffers in ROCmTemplate for proper initialization ([#138521](https://github.com/pytorch/pytorch/pull/138521)).


### Jit



* Unbreak vec128_half_neon comparison without FP16 hardware support ([#139558](https://github.com/pytorch/pytorch/pull/139558))
* Isolate the locale for NNC’s IRPrinter ([#136458](https://github.com/pytorch/pytorch/pull/136458))
* Fix misuse of offset param in seek ([#140633](https://github.com/pytorch/pytorch/pull/140633))



## **Performance**


### Dynamo



* Attempt to use previously compiled code when Dynamo cache limit is hit ([#136655](https://github.com/pytorch/pytorch/pull/136655))
* Don’t convert Python frame local C buffer into Python dict until necessary [#140063](https://github.com/pytorch/pytorch/pull/140063)


### Mps



* Dispatch to SDP-math-mps for non-contiguous Tensors ([#139791](https://github.com/pytorch/pytorch/pull/139791))
* Avoid creating spurious instances of `FUSED_ADAM_OPS` ([#141090](https://github.com/pytorch/pytorch/pull/141090))


### ROCM



* Improve `torch.sum` performance by increasing max_values_per_thread ([#135397](https://github.com/pytorch/pytorch/pull/135397))
* Turn on fast path for index_put on new ROCm version ([#136136](https://github.com/pytorch/pytorch/pull/136136))

### Sparse Frontend



* Speedup broadcasting of sparse_coo Tensors ([#142364](https://github.com/pytorch/pytorch/pull/142364))
* Speedup addmm(dense, BSR) for some int8 shapes on A100 ([#136088](https://github.com/pytorch/pytorch/pull/136088))
* Fuse scaling with addmm(dense, BSR) for some int8 shapes on A100 ([#136104](https://github.com/pytorch/pytorch/pull/136104))
* Fuse dtype conversion with addmm(dense, BSR) for some int8 shapes on A100 ([#136626](https://github.com/pytorch/pytorch/pull/136626))



### Miscellaneous


* Speed up fp16/bf16 AMP casts on H100+ ([#137053](https://github.com/pytorch/pytorch/pull/137053))
* c10d
    * Improved efficiency of NaN checker ([#135414](https://github.com/pytorch/pytorch/pull/135414))
* Improves performance by avoiding atomic add operations in `scatter_add` for XPU. ([#137966](https://github.com/pytorch/pytorch/pull/137966))




### Inductor



* Turn on TORCHINDUCTOR_REORDER_FOR_PEAK_MEMORY by default ([#137205](https://github.com/pytorch/pytorch/pull/137205)).  If old behavior is desired, add `"reorder_for_peak_memory": False` to options in your `torch.compile` call.
* Cache weight tiles in L1D for AMX int8 WoQ GEMM ([#136688](https://github.com/pytorch/pytorch/pull/136688))
* Add and use `borrow_arrayref_tensor_as_tensor` ([#142183](https://github.com/pytorch/pytorch/pull/142183))
* Support for accelerated sorting with x86-simd-sort ([#127936](https://github.com/pytorch/pytorch/pull/127936))
* Enable extended MMA shapes in CUTLASS. ([#133686](https://github.com/pytorch/pytorch/pull/133686))
* Port ExecuTorch bfdot improvement back to ATen BlasKernel ([#136331](https://github.com/pytorch/pytorch/pull/136331), [#137377](https://github.com/pytorch/pytorch/pull/137377))
* Build `ReducedPrecisionFloatGemvFastPathKernel` & entry points for non-ARM architectures too ([#137917](https://github.com/pytorch/pytorch/pull/137917))
* Hook up `fp16_gemv_trans` to gemv fast path for non-aarch64 architectures ([#138005](https://github.com/pytorch/pytorch/pull/138005))
* Add `Vectorizedc10::BFloat16` specialization for ARM ([#139090](https://github.com/pytorch/pytorch/pull/139090))
* Build bf16 gemv fast path & entry points for non-ARM architectures too ([#139208](https://github.com/pytorch/pytorch/pull/139208))
* Hook up `bf16_gemv_trans` to x86 bf16 GEMM ([#139220](https://github.com/pytorch/pytorch/pull/139220))
* Don't go through dispatch for *_dot_with_fp32_arith ([#140834](https://github.com/pytorch/pytorch/pull/140834))
* Add efficient isnan for NEON float/half ([#139082](https://github.com/pytorch/pytorch/pull/139082), [#139083](https://github.com/pytorch/pytorch/pull/139083))
* Hook up `fp16_gemv_trans` to x86 fp16 GEMM ([#137918](https://github.com/pytorch/pytorch/pull/137918))
* Support non-zero beta in `fp16_gemv_trans` ([#138275](https://github.com/pytorch/pytorch/pull/138275))
* Port X86_F16 from executorch half to PyTorch half ([#140720](https://github.com/pytorch/pytorch/pull/140720))
* Reserve vector for NT GEMM Matmul ([#141130](https://github.com/pytorch/pytorch/pull/141130))
* add CK grouped conv2d fwd kernels to ROCm codegen ([#137947](https://github.com/pytorch/pytorch/pull/137947))
* expand quantization conv-binary(-unary) pattern fusion inside inductor ([#138051](https://github.com/pytorch/pytorch/pull/138051))
* Stop force realizing to prevent recursion errors unless it's much bigger ([#138881](https://github.com/pytorch/pytorch/pull/138881))
* Constant folding for lifted graph ([#135060](https://github.com/pytorch/pytorch/pull/135060))
* Add host-side TMA support to AOTInductor ([#138878](https://github.com/pytorch/pytorch/pull/138878))
* Allow inplacing buffer when other users are inconsequential ([#138383](https://github.com/pytorch/pytorch/pull/138383))
* Don't fuse two nodes if likely increase peak memory ([#138756](https://github.com/pytorch/pytorch/pull/138756))
* Add oneDNN BRGEMM config for Half cpp gemm template ([#136255](https://github.com/pytorch/pytorch/pull/136255))
* Enable the oneDNN Linear fusion for special case ([#139172](https://github.com/pytorch/pytorch/pull/139172))
* Remove uses of deleted operations ([#139447](https://github.com/pytorch/pytorch/pull/139447))
* Enable scaled mm with bias in gemm max autotune with CK backend ([#140674](https://github.com/pytorch/pytorch/pull/140674))
* Support linear+binary folding for freezing path ([#138807](https://github.com/pytorch/pytorch/pull/138807))
* Simplify & rectify dequantized B buffer loading for AMX GEMM micro-kernel for WoQ int8 case ([#140258](https://github.com/pytorch/pytorch/pull/140258))
* Improve parallelization by collapsing vectorized loop ([#128812](https://github.com/pytorch/pytorch/pull/128812))
* qconv at XPU backend ([#133080](https://github.com/pytorch/pytorch/pull/133080))
* Dont use constant mask if y numel potentially overflows y grids ([#139751](https://github.com/pytorch/pytorch/pull/139751))
* Add batched gemms into gemm max autotune with CK backend ([#141520](https://github.com/pytorch/pytorch/pull/141520))
* Adding lowering to persistent-tma device kernel for `_scaled_mm` ([#142045](https://github.com/pytorch/pytorch/pull/142045))
* Add fusion pass for `linear_dynamic_fp16` with RELU ([#141556](https://github.com/pytorch/pytorch/pull/141556))
* Reverts runtime numeric check in Inductor to reduce compilation time ([#137324](https://github.com/pytorch/pytorch/pull/137324)).
* Optimizes ARM64 performance by utilizing 128-bit vectors ([#137426](https://github.com/pytorch/pytorch/pull/137426)).
* Adjusts `score_fusion_memory_threshold` application strategy in Inductor ([#138970](https://github.com/pytorch/pytorch/pull/138970)).
* Enhances reduction operations with cooperative multi-kernel support in Inductor ([#138893](https://github.com/pytorch/pytorch/pull/138893138893)).
* Disables `sanitize_overflow` in Inductor kernels ([#139502](https://github.com/pytorch/pytorch/pull/139502)).
* Implements caching for `get_operation_names` and `get_buffer_names` ([#135446](https://github.com/pytorch/pytorch/pull/135446)).
* Reorders scheduler nodes after fusion to reduce peak memory usage ([#134874](https://github.com/pytorch/pytorch/pull/134874)).
* Optimize WOQ INT8 weight dequantization in AMX GEMM template ([#136630](https://github.com/pytorch/pytorch/pull/136630)).
* Uses scalar for f64 constants in Triton codegen ([#136858](https://github.com/pytorch/pytorch/pull/136858)).
* Reduces block sizes for improved performance when using the Triton CPU backend ([#136612](https://github.com/pytorch/pytorch/pull/136612)).
* Optimizes CPU copies during autotuning by restricting them to CUDA devices ([#137509](https://github.com/pytorch/pytorch/pull/137509)).
* Adds host-side Triton TMA support ([#137950](https://github.com/pytorch/pytorch/pull/137950)).
* Optimizes the `can_fuse_vertical()` function ([#135788](https://github.com/pytorch/pytorch/pull/135788)).



## **Documentation**



### Distributed



* c10d
    * Added some code documents for `TCPStore` and `TCPStoreLibUvBackend` code ([#130496](https://github.com/pytorch/pytorch/pull/130496))
    * Added more examples for c10d collectives `gather` and `scatter` ([#130427](https://github.com/pytorch/pytorch/pull/130427))
    * Fixed comments in `ProcessGroupGloo` ([#137746](https://github.com/pytorch/pytorch/pull/137746))
    * Added more inline comments to `CUDAEventCache` code ([#138079](https://github.com/pytorch/pytorch/pull/138079))
    * Added documentations for PG APIs with some cleanups ([#140853](https://github.com/pytorch/pytorch/pull/140853))
    * Updated `backend` arg documentation ([#142404](https://github.com/pytorch/pytorch/pull/142404))
* DTensor
    * Updated DTensor readme to use the new import path ([#138625](https://github.com/pytorch/pytorch/pull/138625))
* FSDP2
    * Better error msg for cpu offloading ([#135156](https://github.com/pytorch/pytorch/pull/135156))
    * Added current FSDP2 path to old composable FSDP1 warning ([#139759](https://github.com/pytorch/pytorch/pull/139759))
* Pipeline
    * Added small comments and variable renames ([#138735](https://github.com/pytorch/pytorch/pull/138735))
* c10d
    * Added some code documents for `TCPStore` and `TCPStoreLibUvBackend` code ([#130496](https://github.com/pytorch/pytorch/pull/130496))
    * Added more examples for c10d collectives `gather` and `scatter` ([#130427](https://github.com/pytorch/pytorch/pull/130427))
    * Fixed comments in `ProcessGroupGloo` ([#137746](https://github.com/pytorch/pytorch/pull/137746))
    * Added more inline comments to `CUDAEventCache` code ([#138079](https://github.com/pytorch/pytorch/pull/138079))
    * Added documentations for PG APIs with some cleanups ([#140853](https://github.com/pytorch/pytorch/pull/140853))
    * Updated `backend` arg documentation ([#142404](https://github.com/pytorch/pytorch/pull/142404))
* DTensor
    * Updated DTensor readme to use the new import path ([#138625](https://github.com/pytorch/pytorch/pull/138625))
* FSDP2
    * Better error msg for cpu offloading ([#135156](https://github.com/pytorch/pytorch/pull/135156))
    * Added current FSDP2 path to old composable FSDP1 warning ([#139759](https://github.com/pytorch/pytorch/pull/139759))
* Pipeline
    * Added small comments and variable renames ([#138735](https://github.com/pytorch/pytorch/pull/138735))
* TP
    * Updated link in distributed.tensor.parallel.rst ([#136103](https://github.com/pytorch/pytorch/pull/136103))
* Checkpoints
    * Add links to tutorial and TorchTitan checkpointing to DCP docs ([#139776](https://github.com/pytorch/pytorch/pull/139776))


### Inductor



* Update the OSS tutorial ([#139956](https://github.com/pytorch/pytorch/pull/139956))
* Add README for `torch._inductor.runtime` ([#141492](https://github.com/pytorch/pytorch/pull/141492))
* Improve OSSProxyExecutor error messages ([#141501](https://github.com/pytorch/pytorch/pull/141501))
* Enhances documentation for the bundled autotune cache to provide clearer guidance ([#138298](https://github.com/pytorch/pytorch/pull/138298)).


### Mps



* Update `MPS_ERROR_RUNTIME_TOO_LOW` message ([#139427](https://github.com/pytorch/pytorch/pull/139427))
* Fixing MPS conv1d error message for output 2**16 ([#134770](https://github.com/pytorch/pytorch/pull/134770))
* Modify missing op message ([#141314](https://github.com/pytorch/pytorch/pull/141314))
* Update error message for supported autocast type ([#139192](https://github.com/pytorch/pytorch/pull/139192))


### NN Frontend



* Fix formula in RMSNorm documentation ([#136727](https://github.com/pytorch/pytorch/pull/136727))
* Remove incorrect bias initialization in RMSNorm documentation ([#139620](https://github.com/pytorch/pytorch/pull/139620))
* Add reference to `pad_packed_sequence` in `pack_padded_sequence` documentation ([#137294](https://github.com/pytorch/pytorch/pull/137294))
* Improve documentation of `register_module_forward_hook` ([#140379](https://github.com/pytorch/pytorch/pull/140379))
* Correct reference link for triplet margin loss ([#142071](https://github.com/pytorch/pytorch/pull/142071))
* Changed 'standard-deviation' to 'variance' in normalization documentation ([#141982](https://github.com/pytorch/pytorch/pull/141982))
* Fix broadcasting error in example in `nn.functional.scaled_dot_product_attention` documentation ([#135427](https://github.com/pytorch/pytorch/pull/135427))
* Point to transformer building blocks tutorial in transformer documentation ([#144425](https://github.com/pytorch/pytorch/pull/144425))

### Optim



* Removes confusing note about closure grad modification ([#137535](https://github.com/pytorch/pytorch/pull/137535))
* Minorly reorder optim kwargs in docs ([#137531](https://github.com/pytorch/pytorch/pull/137531), [#137528](https://github.com/pytorch/pytorch/pull/137528))
* RMSprop docs: add missing input "epsilon" ([#137854](https://github.com/pytorch/pytorch/pull/137854))
* Add missing input "eps" to adam docs ([#135191](https://github.com/pytorch/pytorch/pull/135191))
* Corrected AMSGrad max equation in Adam and AdamW ([#142051](https://github.com/pytorch/pytorch/pull/142051))
* Documentation Update: Fix Missing Whitespace in Optimizer Docs ([#138321](https://github.com/pytorch/pytorch/pull/138321))


### Python Frontend



* Fix return type of `torch.nansum` example. ([#135435](https://github.com/pytorch/pytorch/pull/135435))
* Fix `torch.cat` doc ([#135698](https://github.com/pytorch/pytorch/pull/135698))
* Fix multiple function parameters docstring ([#136097](https://github.com/pytorch/pytorch/pull/136097), [#140089](https://github.com/pytorch/pytorch/pull/140089))
* Clarify that NaNs are not equal to each other ([#137386](https://github.com/pytorch/pytorch/pull/137386))
* Fix description in `torch.save` docs to show default for pickle_protocol instead of variable name ([#138153](https://github.com/pytorch/pytorch/pull/138153))
* Fix docs for logcumsumexp formula ([#139768](https://github.com/pytorch/pytorch/pull/139768))
* Clarify meaning of rate parameter in Gamma distribution ([#134847](https://github.com/pytorch/pytorch/pull/134847))
* Updated docstrings referring to `torch.expand` to point to `torch.Tensor.expand` ([#140045](https://github.com/pytorch/pytorch/pull/140045))
* Update documentation for `torch.mean()` to note behavior with empty tensors ([#142039](https://github.com/pytorch/pytorch/pull/142039))
* Improve `torch.squeeze` parameter type in docstring ([#137485](https://github.com/pytorch/pytorch/pull/137485))
* Improve `torch.isclose` docstring ([#138459](https://github.com/pytorch/pytorch/pull/138459), [#139724](https://github.com/pytorch/pytorch/pull/139724))
* Clarify `torch.sum` dtype promotion behavior ([#140939](https://github.com/pytorch/pytorch/pull/140939))
* Clarify `torch.arang`e floating-point rounding behavior ([#141655](https://github.com/pytorch/pytorch/pull/141655))
* Fix `torch.trapezoid` docstring ([#141459](https://github.com/pytorch/pytorch/pull/141459))
* Clarify when the optional opt-einsum dependency is used ([#137596](https://github.com/pytorch/pytorch/pull/137596))
* Clarify `torch.linalg.vector_norm` input aliasing behavior ([#136921](https://github.com/pytorch/pytorch/pull/136921))
* Fix `torch.linalg.svd` V* shape ([#142037](https://github.com/pytorch/pytorch/pull/142037))


### Miscellaneous



* Small rendering fix to our `torch.compile` FakeTensor documentation ([#138281](https://github.com/pytorch/pytorch/pull/138281))
* Document that load_inline requires having a C++ compiler installed ([#137521](https://github.com/pytorch/pytorch/pull/137521))
* Fix error message in `torch._scaled_mm` ([#140343](https://github.com/pytorch/pytorch/pull/140343))
* Revamp `torch.compile` troubleshooting doc ([#138620](https://github.com/pytorch/pytorch/pull/138620))
* Fix doc for export.export() API ([#135551](https://github.com/pytorch/pytorch/pull/135551))
* Fix the example in fx/interpreter ([#139368](https://github.com/pytorch/pytorch/pull/139368))
* Add new PT2 troubleshooting doc ([#138620](https://github.com/pytorch/pytorch/pull/138620))
* Update "Getting Started with XPU" documentation. ([#137479](https://github.com/pytorch/pytorch/pull/137479))


## **Developers**


### Composability



* Make `maybe_aliasing_or_mutating` proper tag ([#131990](https://github.com/pytorch/pytorch/pull/131990))


### Distributed



* c10d
    * Added wait counter for nccl abort ([#136067](https://github.com/pytorch/pytorch/pull/136067))
    * Added wait counter for time spent in object to tensor and tensor to object ([#140414](https://github.com/pytorch/pytorch/pull/140414))
    * Added trace operations for `TCPStoreLibUvBackend` ([#136320](https://github.com/pytorch/pytorch/pull/136320))
    * Cast device index to int before logging ([#135405](https://github.com/pytorch/pytorch/pull/135405))
    * Logged `WorkNCCL` exception string to `C10dLogger` ([#137736](https://github.com/pytorch/pytorch/pull/137736))
    * Made Formatter avoid throwing exceptions in `socket.cpp` ([#137745](https://github.com/pytorch/pytorch/pull/137745))
    * Recorded world size in the log of flight recorder ([#138044](https://github.com/pytorch/pytorch/pull/138044))
    * Differentiated timeout errors from nccl errors ([#138240](https://github.com/pytorch/pytorch/pull/138240))
    * Added more appropriate socket errors and debug messages ([#130347](https://github.com/pytorch/pytorch/pull/130347))
    * Reordered cpp stack dump and FR dump and add log prefix to loggings ([#138368](https://github.com/pytorch/pytorch/pull/138368))
    * Reordered GIL checker and c++ stack trace print with comments ([#138734](https://github.com/pytorch/pytorch/pull/138734))
    * Enabled watchdog to print call-time traceback when reporting NCCL watchdog timeout ([#139659](https://github.com/pytorch/pytorch/pull/139659))
    * Added type information for `FakeProcessGroup` ([#133211](https://github.com/pytorch/pytorch/pull/133211))
    * Added a wait counter for dump function ([#140823](https://github.com/pytorch/pytorch/pull/140823))
    * Switched all timer logging in c10d to wait_counter ([#141154](https://github.com/pytorch/pytorch/pull/141154))
    * Improved Flight Recorder efficacy ([#142178](https://github.com/pytorch/pytorch/pull/142178))
    * Changed back `vlog(2)` to `LOG(INFO)` for Flight Recorder ([#142441](https://github.com/pytorch/pytorch/pull/142441))
    * Added better profiling title for “NCCL barrier, nccl:all_reduce” to “nccl:all_reduce_barrier” ([#140785](https://github.com/pytorch/pytorch/pull/140785))
    * Adopted better error message for flight recorder status ([#142505](https://github.com/pytorch/pytorch/pull/142505))
    * Fixed the wrong error msg in `ProcessGroupNCCL` ([#135423](https://github.com/pytorch/pytorch/pull/135423))
    * Added some missing spaces in barrier msg ([#137721](https://github.com/pytorch/pytorch/pull/137721))
    * Added thread-safety initialization warning ([#139638](https://github.com/pytorch/pytorch/pull/139638))
    * Added the log of started work numel ([#139773](https://github.com/pytorch/pytorch/pull/139773))
    * Improved messaging of `ProcessGroupNCCL` destructor ([#142297](https://github.com/pytorch/pytorch/pull/142297))
* TorchElastic
    * Passed `FileTimerRequests.to_json()` to `log_debug_info_for_expired_timers` for a better debugging experience ([#135913](https://github.com/pytorch/pytorch/pull/135913))


### Export



* Prototype `_swap_modules` API that can be used to swap submodules of an exported program ([#136190](https://github.com/pytorch/pytorch/pull/136190), [#139126](https://github.com/pytorch/pytorch/pull/139126))
* Avoid debug name crash for dim hints ([#139104](https://github.com/pytorch/pytorch/pull/139104))


### Inductor



* Remove the non-ABI-compatible mode ([#138009](https://github.com/pytorch/pytorch/pull/138009), [#138047](https://github.com/pytorch/pytorch/pull/138047))
* Move `use_minimal_arrayref_interface` logic ([#138250](https://github.com/pytorch/pytorch/pull/138250))
* Refactor `ir.Layout` into `ir.OutputSpec` ([#140910](https://github.com/pytorch/pytorch/pull/140910))
* Refactor `dependencies.extract_loop_body_with_args` ([#141404](https://github.com/pytorch/pytorch/pull/141404))
* Modest code motion in compile_fx ([#141574](https://github.com/pytorch/pytorch/pull/141574))
* Move post compile steps into post_compile1/post_compile2 method ([#141656](https://github.com/pytorch/pytorch/pull/141656))
* Inline `FxGraphCache.load` into its sole call site ([#141681](https://github.com/pytorch/pytorch/pull/141681))
* Hoist `set_feature_use` out of conditional, rename some variables ([#141683](https://github.com/pytorch/pytorch/pull/141683))
* Unify cache disable and cache bypass paths ([#141685](https://github.com/pytorch/pytorch/pull/141685))
* Unify `post_compile1` and `CompiledFxGraph` constructor ([#141689](https://github.com/pytorch/pytorch/pull/141689))
* Inline `compile_to_fn` at its only call site ([#141691](https://github.com/pytorch/pytorch/pull/141691))
* move block pointer analysis to a new module ([#141733](https://github.com/pytorch/pytorch/pull/141733))
* Factor `_fx_graph_cache_key` and _time_taken_ns to common base class ([#141878](https://github.com/pytorch/pytorch/pull/141878))
* codecache: pull out some Graph serialization code into common helpers ([#141502](https://github.com/pytorch/pytorch/pull/141502))
* Refactor optional graph module into `CompiledFxGraphConstants` ([#141897](https://github.com/pytorch/pytorch/pull/141897))
* Adds a compiler bisector tool to aid in debugging and development processes within PyTorch ([#131936](https://github.com/pytorch/pytorch/pull/131936)).


### Optim



* Add back optim type hints that were lost when `*.pyi` files were removed ([#136185](https://github.com/pytorch/pytorch/pull/136185))
* Ensure SWA boundary conditions w.r.t. definition ([#133773](https://github.com/pytorch/pytorch/pull/133773))


### Quantization



* Add unaligned attributes to `q8gemm`/`4x4c2-sse2.c` ([#140188](https://github.com/pytorch/pytorch/pull/140188))
* Adding more support QuantizedPrivateuse1 backends ([#139860](https://github.com/pytorch/pytorch/pull/139860))
* Make `move_exported_model_to_train`/`eval` idempotent ([#142239](https://github.com/pytorch/pytorch/pull/142239))


### Releng



* Deprecate usage of pytorch/builder repository ([#142156](https://github.com/pytorch/pytorch/pull/142156)) ([#142277](https://github.com/pytorch/pytorch/pull/142277)) ([#142282](https://github.com/pytorch/pytorch/pull/142282)) ([#142482](https://github.com/pytorch/pytorch/pull/142482)) ([#138103](https://github.com/pytorch/pytorch/pull/138103)) ([#139815](https://github.com/pytorch/pytorch/pull/139815)) ([#140020](https://github.com/pytorch/pytorch/pull/140020)) ([#142382](https://github.com/pytorch/pytorch/pull/142382))
* Add inductor micro benchmark on x86 metal runner ([#135042](https://github.com/pytorch/pytorch/pull/135042)) ([#136052](https://github.com/pytorch/pytorch/pull/136052)) ([#135780](https://github.com/pytorch/pytorch/pull/135780))
* Migrated PyTorch Dev Infra Runners to Amazon Linux 2023 ([#136540](https://github.com/pytorch/pytorch/pull/136540)) ([#136544](https://github.com/pytorch/pytorch/pull/136544))
* Migrated HUD backend database from Rockset to Clickhouse ([#139296](https://github.com/pytorch/pytorch/pull/139296)) ([#139322](https://github.com/pytorch/pytorch/pull/139322)) ([#137207](https://github.com/pytorch/pytorch/pull/137207)) ([#139922](https://github.com/pytorch/pytorch/pull/139922)) ([#140574](https://github.com/pytorch/pytorch/pull/140574))
* Release engineering tooling, CI fixes and additional CI tests . Workflows, Trymerge, Bot Labeler, Mergebot  ([#136060](https://github.com/pytorch/pytorch/pull/136060)) ([#140185](https://github.com/pytorch/pytorch/pull/140185)) ([#135582](https://github.com/pytorch/pytorch/pull/135582))  ([#135644](https://github.com/pytorch/pytorch/pull/135644)) ([#136061](https://github.com/pytorch/pytorch/pull/136061))  ([#135342](https://github.com/pytorch/pytorch/pull/135342)) ([#136043](https://github.com/pytorch/pytorch/pull/136043)) ([#134356](https://github.com/pytorch/pytorch/pull/134356)) ([#136208](https://github.com/pytorch/pytorch/pull/136208)) ([#136610](https://github.com/pytorch/pytorch/pull/136610)) ([#136791](https://github.com/pytorch/pytorch/pull/136791)) ([#136239](https://github.com/pytorch/pytorch/pull/136239)) ([#135342](https://github.com/pytorch/pytorch/pull/135342)) ([#136794](https://github.com/pytorch/pytorch/pull/136794)) ([#137104](https://github.com/pytorch/pytorch/pull/137104)) ([#137168](https://github.com/pytorch/pytorch/pull/137168)) ([#137176](https://github.com/pytorch/pytorch/pull/137176)) ([#137170](https://github.com/pytorch/pytorch/pull/137170)) ([#137169](https://github.com/pytorch/pytorch/pull/137169))  ([#135390](https://github.com/pytorch/pytorch/pull/135390)) ([#137614](https://github.com/pytorch/pytorch/pull/137614)) ([#137802](https://github.com/pytorch/pytorch/pull/137802)) ([#137791](https://github.com/pytorch/pytorch/pull/137791)) ([#138178](https://github.com/pytorch/pytorch/pull/138178)) ([#138054](https://github.com/pytorch/pytorch/pull/138054))  ([#138232](https://github.com/pytorch/pytorch/pull/138232)) ([#138263](https://github.com/pytorch/pytorch/pull/138263)) ([#138178](https://github.com/pytorch/pytorch/pull/138178))  ([#138752](https://github.com/pytorch/pytorch/pull/138752)) ([#138204](https://github.com/pytorch/pytorch/pull/138204)) ([#138714](https://github.com/pytorch/pytorch/pull/138714))  ([#138874](https://github.com/pytorch/pytorch/pull/138874))


### XPU



* Remove unnecessary Triton dependencies for XPU wheel builds. ([#143983](https://github.com/pytorch/pytorch/pull/143983))
* Update Docker builds workflow with a new XPU image name. ([#142298](https://github.com/pytorch/pytorch/pull/142298))
* Restore Triton build support for XPU. ([#141775](https://github.com/pytorch/pytorch/pull/141775))
* Update Triton XPU version pinning. ([#135638](https://github.com/pytorch/pytorch/pull/135638))
* Improve exception handling for XPU device initialization. ([#141658](https://github.com/pytorch/pytorch/pull/141658))
* Enhance unit tests for XPU memory allocation. ([#141325](https://github.com/pytorch/pytorch/pull/141325))
* Make XPU libraries publicly accessible for developers. ([#136974](https://github.com/pytorch/pytorch/pull/136974))
* Improve code formatting for XPU oneDNN integration. ([#139721](https://github.com/pytorch/pytorch/pull/139721))
* Make XPU oneDNN headers publicly available for documentation purposes. ([#139177](https://github.com/pytorch/pytorch/pull/139177))
* Ensure XPU compiler version control in CMake for backward compatibility. Users should align their XPU compiler version with supported versions in PyTorch. ([#139258](https://github.com/pytorch/pytorch/pull/139258))
