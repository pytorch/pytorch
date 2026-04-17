(export.pt2_archive)=

# PT2 Archive Spec

The following specification defines the archive format which can be produced
through the following methods:

* {ref}`torch.export <torch.export>` through calling {func}`torch.export.save`
* {ref}`AOTInductor <torch.compiler_aot_inductor>` through calling {func}`torch._inductor.aoti_compile_and_package`

The archive is a zipfile, and can be manipulated using standard zipfile APIs.

The following is a sample archive. We will walk through the archive folder by folder.

```
.
├── archive_format
├── byteorder
├── .data
│   ├── serialization_id
│   └── version
├── data
│   ├── aotinductor
│   │   └── model1
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.kernel_metadata.json
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.kernel.cpp
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.wrapper_metadata.json
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.wrapper.cpp
│   │       ├── cf5ez6ifexr7i2hezzz4s7xfusj4wtisvu2gddeamh37bw6bghjw.wrapper.so
│   │       ├── cg7domx3woam3nnliwud7yvtcencqctxkvvcafuriladwxw4nfiv.cubin
│   │       └── cubaaxppb6xmuqdm4bej55h2pftbce3bjyyvljxbtdfuolmv45ex.cubin
│   ├── weights
│   │  ├── model1_weights_config.json
│   │  ├── model2_weights_config.json
│   │  ├── weight_0
│   │  ├── weight_1
│   │  ├── weight_2
│   └── constants
│   │  ├── model1_constants_config.json
│   │  ├── model2_constants_config.json
│   │  ├── tensor_0
│   │  ├── tensor_1
│   │  ├── custom_obj_0
│   │  ├── custom_obj_1
│   └── sample_inputs
│       ├── model1.pt
│       └── model2.pt
├── extra
│   └── ....json
└── models
    ├── model1.json
    └── model2.json
```

## Contents

### Archive Headers

* `archive_format` declares the format used by this archive. Currently, it can only be “pt2”.
* `byteorder`. One of “little” or “big”, used by zip file reader
* `/.data/version` contains the archive version. (Notice that this is neither export serialization’s schema version, nor Aten Opset Version).
* `/.data/serialization_id` is a hash generated for the current archive, used for verification.


### AOTInductor Compiled Artifact

Path: `/data/aotinductor/<model_name>-<backend>/`

AOTInductor compilation artifacts are saved for each model-backend pair. For
example, compilation artifacts for the `model1` model on A100 and H100 will be
saved in `model1-a100` and `model1-h100` folders separately.

The folder typically contains
* `<uuid>.wrapper.so`: Dynamic library compiled from <uuid>.cpp.
* `<uuid>.wrapper.cpp`: AOTInductor generated cpp wrapper file.
* `<uuid>.kernel.cpp`: AOTInductor generated cpp kernel file.
* `*.cubin`: Triton kernels compiled from triton codegen kernels
* `<uuid>.wrapper_metadata.json`: Metadata which was passed in from the `aot_inductor.metadata` inductor config
* (optional) `<uuid>.json`: External fallback nodes for custom ops to be executed by `ProxyExecutor`, serialized according to `ExternKernelNode` struct. If the model doesn’t use custom ops/ProxyExecutor, this file would be omitted.

### Weights

Path: `/data/weights/*`

Model parameters and buffers are saved in the `/data/weights/` folder. Each
tensor is saved as a separated file. The file only contains the raw data blob,
tensor metadata and mapping from model weight FQN to saved raw data blob are saved separately in the
`<model_name>_weights_config.json`.

### Constants

Path: `/data/constants/*`

TensorConstants, non-persistent buffers and TorchBind objects are saved in the
`/data/constants/` folder. Metadata and mapping from model constant FQN to saved raw data blob are saved separately in the
`<model_name>_constants_config.json`

### Sample Inputs

Path: `/data/sample_inputs/<model_name>.pt`

The `sample_input` used by `torch.export` could be included in the archive for
downstream use. Typically, it’s a flattened list of Tensors, combining both args
and kwargs of the forward() function.

The .pt file is produced by `torch.save(sample_input)`, and can be loaded by
`torch.load()` in python and `torch::pickle_load()` in c++.

When the model has multiple copies of sample input, it would be packaged as
`<model_name>_<index>.pt`.

### Models Definitions

Path: `/models/<model_name>.json`

Model definition is the serialized json of the ExportedProgram from
`torch.export.save`, and other model-level metadata.

## Multiple Models

This archive spec supports multiple model definitions coexisting in the same
file, with `<model_name>` serving as a unique identifier for the models, and
will be used as reference in other folders of the archive.

Lower level APIs like {func}`torch.export.pt2_archive._package.package_pt2` and
{func}`torch.export.pt2_archive._package.load_pt2` allow you to have
finer-grained control over the packaging and loading process.
