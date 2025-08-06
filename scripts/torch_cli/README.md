# 🔧 Torch Cli
A  Python CLI tool for building and testing PyTorch-based components, using a YAML configuration file for structured, repeatable workflows.

## Features
- 🏗️ **Build**
    - external projects (e.g. vLLM)
- ✅ **Test** components with custom or CI-based logic
- ⚙️ **Main input: YAML-configurable**: define complex build/test steps via input config

## 📦 Installation
at the root of the pytorch repo
```bash
pip install -e scripts/torch_cli
```

## Run the cli tool
The cli tool must be installed and used at root of pytorch repo, as example to run build external vllm:
```bash
python3 -m cli.run build external vllm
```
this will run the build steps defined in the default config file for vllm project

with config file:
```bash
python3 -m cli.run --config ".github/ci_configs/vllm.yaml" build external vllm
```
this will run the build steps defined in the config file for vllm project

to see help message:
```bash
python3 -m cli.run --help
```
