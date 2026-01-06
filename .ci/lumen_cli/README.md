# 🔧 Lumen_cli
A Python CLI tool for building and testing PyTorch-based components, using a YAML configuration file for structured, repeatable workflows.


## Features
- **Build**
    - external projects (e.g. vLLM)

## 📦 Installation
at the root of the pytorch repo
```bash
pip install -e .ci/lumen_cli
```

## Run the cli tool
The cli tool must be used at root of pytorch repo, as example to run build external vllm:
```bash
python -m cli.run build external vllm
```
this will run the build steps with default behaviour for vllm project.

to see help messages, run
```bash
python3 -m cli.run --help
```

## Add customized external build logics
To add a new external build, for instance, add a new external build logics:
1. create the build function in cli/lib folder
2. register your target and the main build function at  EXTERNAL_BUILD_TARGET_DISPATCH in `cli/build_cli/register_build.py`
3. [optional] create your ci config file in .github/ci_configs/${EXTERNAL_PACKAGE_NAME}.yaml
