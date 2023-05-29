![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

---

- [English](README_en.md)
- [Português](README_pt.md)
- [عربي](README_ar.md)
- [Türkçe](README_tr.md)
- [Deutsch](README_de.md)

PyTorch é um pacote Python que fornece dois recursos de alto nível:

- Computação de tensor (como NumPy) com forte aceleração de GPU
- Redes neurais profundas construídas em um sistema de autograduação baseado em fita

Você pode reutilizar seus pacotes Python favoritos, como NumPy, SciPy e Cython, para estender o PyTorch quando necessário.

Nossa integridade (sinais de Continuous Integration) pode ser encontrada em [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main).

<!-- toc -->

- [Mais Sobre PyTorch](#more-about-pytorch)
  - [Biblioteca de Tensores Pronta para GPU](#a-gpu-ready-tensor-library)
  - [Redes Neurais Dinâmicas: Autogradiente Baseado em Fita](#dynamic-neural-networks-tape-based-autograd)
  - [Python Primeiro](#python-first)
  - [Experiências Imperativas](#imperative-experiences)
  - [Rápido e Leve](#fast-and-lean)
  - [Extensões sem Dor](#extensions-without-pain)
- [Instalação](#installation)
  - [Binaries](#binaries)
    - [Plataformas NVIDIA Jetson](#nvidia-jetson-platforms)
  - [Da Fonte](#from-source)
    - [Pré-requisitos](#prerequisites)
    - [Dependências de instalação](#install-dependencies)
    - [Obter a Fonte PyTorch](#get-the-pytorch-source)
    - [Instalar PyTorch](#install-pytorch)
      - [Adjust Build Options (Optional)](#adjust-build-options-optional)
  - [Imagem Docker](#docker-image)
    - [Usando imagens pré-prontas](#using-pre-built-images)
    - [Criando a própria imagem](#building-the-image-yourself)
  - [Montando Documentação](#building-the-documentation)
  - [Versões Anteriores](#previous-versions)
- [Começando](#getting-started)
- [Recursos](#resources)
- [Comunicação](#communication)
- [Lançamentos e Contribuição](#releases-and-contributing)
- [O Time](#the-team)
- [Licença](#license)

<!-- tocstop -->

## Mais Sobre PyTorch

At a granular level, PyTorch is a library that consists of the following components:

| Component                                                                         | Description                                                                                                                             |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)                           | A Tensor library like NumPy, with strong GPU support                                                                                    |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html)               | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch                              |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)                         | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code                                       |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)                           | A neural networks library deeply integrated with autograd designed for maximum flexibility                                              |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)                      | DataLoader and other utility functions for convenience                                                                                  |

Usually, PyTorch is used either as:

- A replacement for NumPy to use the power of GPUs.
- A deep learning research platform that provides maximum flexibility and speed.

Elaborating Further:

### Biblioteca de Tensores Pronta para GPU

If you use NumPy, then you have used Tensors (a.k.a. ndarray).

![Tensor illustration](../docs/source/_static/img/tensor_illustration.png)

PyTorch provides Tensors that can live either on the CPU or the GPU and accelerates the
computation by a huge amount.

We provide a wide variety of tensor routines to accelerate and fit your scientific computation needs
such as slicing, indexing, mathematical operations, linear algebra, reductions.
And they are fast!

### Redes Neurais Dinâmicas: Autogradiente Baseado em Fita

PyTorch has a unique way of building neural networks: using and replaying a tape recorder.

Most frameworks such as TensorFlow, Theano, Caffe, and CNTK have a static view of the world.
One has to build a neural network and reuse the same structure again and again.
Changing the way the network behaves means that one has to start from scratch.

With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows you to
change the way your network behaves arbitrarily with zero lag or overhead. Our inspiration comes
from several research papers on this topic, as well as current and past work such as
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

While this technique is not unique to PyTorch, it's one of the fastest implementations of it to date.
You get the best of speed and flexibility for your crazy research.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

### Python Primeiro

PyTorch is not a Python binding into a monolithic C++ framework.
It is built to be deeply integrated into Python.
You can use it naturally like you would use [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) etc.
You can write your new neural network layers in Python itself, using your favorite libraries
and use packages such as [Cython](https://cython.org/) and [Numba](http://numba.pydata.org/).
Our goal is to not reinvent the wheel where appropriate.

### Experiências Imperativas

PyTorch is designed to be intuitive, linear in thought, and easy to use.
When you execute a line of code, it gets executed. There isn't an asynchronous view of the world.
When you drop into a debugger or receive error messages and stack traces, understanding them is straightforward.
The stack trace points to exactly where your code was defined.
We hope you never spend hours debugging your code because of bad stack traces or asynchronous and opaque execution engines.

### Rápido e Leve

PyTorch has minimal framework overhead. We integrate acceleration libraries
such as [Intel MKL](https://software.intel.com/mkl) and NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl)) to maximize speed.
At the core, its CPU and GPU Tensor and neural network backends
are mature and have been tested for years.

Hence, PyTorch is quite fast — whether you run small or large neural networks.

The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives.
We've written custom memory allocators for the GPU to make sure that
your deep learning models are maximally memory efficient.
This enables you to train bigger deep learning models than before.

### Extensões sem Dor

Writing new neural network modules, or interfacing with PyTorch's Tensor API was designed to be straightforward
and with minimal abstractions.

You can write new neural network layers in Python using the torch API
[or your favorite NumPy-based libraries such as SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate.
No wrapper code needs to be written. You can see [a tutorial here](https://pytorch.org/tutorials/advanced/cpp_extension.html) and [an example here](https://github.com/pytorch/extension-cpp).

## Instalação

### Binaries

Os comandos para instalar binários via Conda ou pip wheels estão em nosso site: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### Plataformas NVIDIA Jetson

Python wheels para NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, e Jetson AGX Orin são fornecidos [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) e o contêiner L4T é publicado [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

Eles exigem o JetPack 4.2 e superior e[@dusty-nv](https://github.com/dusty-nv) e [@ptrblck](https://github.com/ptrblck) e estão mantendo.

### Da Fonte

#### Pré-requisitos

Se você estiver instalando a partir da fonte, precisará de:

- Python 3.8 ou mais tarde (para Linux, Python 3.8.1+ é preciso).
- A C++17 compilador compatível, como clang.

Recomendamos a instalação de um [Anaconda](https://www.anaconda.com/distribution/#download-section) ambiente. Você obterá uma BLAS library (MKL) de alta qualidade e obterá versões de dependência controladas, independentemente da sua distro Linux.

Se você deseja compilar com suporte CUDA, instale o seguinte (observe que CUDA não é compatível com macOS)

- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.0 ou mais atual
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 ou mais atual
- [Compiler](https://gist.github.com/ax3l/9489132) compatível com CUDA

Nota: Você pode consultar o [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) para versões cuDNN com os vários CUDA suportados, CUDA driver e NVIDIA hardware.

Se você deseja desabilitar o suporte CUDA, exporte a variável de ambiente `USE_CUDA=0`.
Outras variáveis de ambiente potencialmente úteis podem ser encontradas em `setup.py`.

Se você está criando para as plataformas Jetson da NVIDIA (Jetson Nano, TX1, TX2, AGX Xavier), Instruções para instalar o PyTorch para Jetson Nano são [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

Se você deseja compilar com suporte a ROCm, instale:

- [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) 4.0 e acima da instalação
- Atualmente, o ROCm é suportado apenas para sistemas Linux.

Se você deseja desabilitar o suporte ROCM, exporte a variável de ambiente `USE_ROCM=0`.
Outras variáveis de ambiente potencialmente úteis podem ser encontradas em `setup.py`.

#### Dependências de Instalação

**Comum**

```bash
conda install cmake ninja
# Execute este comando no diretório PyTorch após clonar o código-fonte usando a seção "Obter a fonte PyTorch" abaixo
pip install -r requirements.txt
```

**No Linux**

```bash
conda install mkl mkl-include
# Somente CUDA: add suporte LAPACK para a GPU, se necessário.
conda install -c pytorch magma-cuda110  # ou o magma-cuda* que corresponde à sua versão CUDA de https://anaconda.org/pytorch/repo

# (opcional) Se estiver usando torch.compile com inductor/triton, instale a versão correspondente do triton.
# Execute a partir do diretório pytorch após a clonagem.
make triton
```

**No MacOS**

```bash
# Adicione este pacote apenas em máquinas com processador intel x86.
conda install mkl mkl-include
# Adicione esses pacotes se torch.distributed é preciso.
conda install pkg-config libuv
```

**No Windows**

```bash
conda install mkl mkl-include
# Adicione esses pacotes se torch.distributed é preciso.
# O suporte a pacotes distribuídos no Windows é um recurso de protótipo e está sujeito a alterações.
conda install -c conda-forge libuv=1.39
```

#### Obter a Fonte PyTorch

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# se você estiver atualizando um checkout existente então:
git submodule sync
git submodule update --init --recursive
```

#### Instalar PyTorch

**No Linux**

Se você estiver compilando para AMD ROCm, primeiro execute este comando:

```bash
# Só execute isso se você estiver compilando para ROCm
python tools/amd_build/build_amd.py
```

Instalar PyTorch

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

> _Aside:_ Se você estiver usando [Anaconda](https://www.anaconda.com/distribution/#download-section), você pode ter um erro causado pelo vinculador(linker):
>
> ```plaintext
> build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized
> collect2: error: ld returned 1 exit status
> error: command 'g++' failed with exit status 1
> ```
>
> Isso é causado por `ld` do ambiente Conda sombreando o sistema `ld`. Você deve usar uma versão mais recente do Python que corrige esse problema. A versão Python recomendada é 3.8.1+.

**No macOS**

```bash
python3 setup.py develop
```

**No Windows**

Escolha Correto a versão do Visual Studio.

PyTorch CI usa Visual C++ BuildTools, que vêm com Visual Studio Enterprise,
Professional, ou Community Editions. Você também pode instalar as ferramentas de compilação de
https://visualstudio.microsoft.com/visual-cpp-build-tools/. As ferramentas de construção _do not_
venha com Visual Studio Code by default.

Se você quer construir legacy python code, por favor consulte [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**Builds somente de CPU**

Nesse modo, os cálculos do PyTorch serão executados em sua CPU, não em sua GPU

```cmd
conda activate
python setup.py develop
```

Nota sobre o OpenMP: A implementação OpenMP desejada é Intel OpenMP (iomp). Para vincular ao iomp, você precisará baixar manualmente a biblioteca e configurar o ambiente de construção ajustando `CMAKE_INCLUDE_PATH` e `LIB`. A instrução [here](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) é um exemplo para configurar MKL e Intel OpenMP. Sem essas configurações para CMake, Microsoft Visual C OpenMP runtime (vcomp) será usado.

**Build baseada em CUDA**

Nesse modo, os cálculos do PyTorch aproveitarão sua GPU via CUDA para processamento de números mais rápido

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) é necessário para o build o Pytorch com CUDA.
NVTX é uma parte de CUDA distributive, onde é chamado "Nsight Compute".Para instalá-lo em um CUDA já instalado, execute a instalação do CUDA novamente e marque a caixa de seleção correspondente.
Certifique-se de que CUDA com Nsight Compute é instalado depois do Visual Studio.

Atualmente, VS 2017 / 2019, e Ninja são suportados como o gerador de CMake. Se `ninja.exe` é detectado em `PATH`, então Ninja será usado como gerador padrão, caso contrário, ele usará VS 2017 / 2019.
<br/> Se Ninja for selecionado como o gerador, o MSVC mais recente será selecionado como a cadeia de ferramentas subjacente(toolchain).

Bibliotecas adicionais, como[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) muitas vezes são necessários. Por favor, consulte o [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) para instalá-los.

Você pode consultar o [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) script para algumas outras configurações de variáveis de ambiente.

```cmd
cmd

:: Defina as variáveis de ambiente depois de baixar e descompactar o pacote mkl,
:: caso contrário, o CMake lançaria um erro como `Could NOT find OpenMP`.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Leia o conteúdo da seção anterior cuidadosamente antes de prosseguir.
:: [Opcional] Se você deseja substituir o conjunto de ferramentas subjacente usado por Ninja e Visual Studio com CUDA, execute o seguinte bloco de script.
:: "Visual Studio 2019 Developer Command Prompt" será executado automaticamente.
:: Certifique-se de ter CMake >= 3.12 antes de fazer isso ao usar o gerador do Visual Studio.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Opcional] Se você deseja substituir o compilador de host CUDA
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python setup.py develop

```

##### Ajustar Opções de Build (Opcional)

Você pode ajustar a configuração das variáveis cmake opcionalmente (sem compilar primeiro), fazendo o seguinte. Por exemplo, o ajuste dos diretórios pré-detectados para CuDNN ou BLAS pode ser feito como a passo a baixo.

No Linux

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

No macOS

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

### Imagem Docker

#### Usando imagens pré-prontas

Você pode também baixar uma imagem Docker pré-pronta do Docker Hub e executar com docker v19.03+

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Repare que PyTorch usa memória compartilhada para compartilhar dados entre processos, então se torch multiprocessing for usado (por exemplo, para carregar dados multithreads), o tamanho padrão do segmento de memória compartilhada não é suficiente, e você deve aumentar o tamanho da memória com as opções `--ipc=host` ou `--shm-size` do `nvidia-docker run`.

#### Criando a própria imagem

**OBSERVAÇÃO:** Deve ser criada com uma versão docker > 18.06

O `Dockerfile` é fornecido para criar imagens com suporte CUDA 11.1 e cuDNN v8.
Você pode passar a variável `PYTHON_VERSION=x.y` para especificar qual versão do Python será usada pelo Miniconda, ou deixá-la indefinida para usar o padrão.

```bash
make -f docker.Makefile
# imagens are rotuladas como docker.io/${your_docker_username}/pytorch
```

Você também pode passar a variável de ambiente `CMAKE_VARS="..."` para especificar variáveis CMake adicionais a serem passadas ao CMake durante a build. Veja [setup.py](./setup.py) para a lista de variáveis disponíveis.

```bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker.Makefile
```

### Montando Documentação

Para montar documentação em vários formatos, você vai precisar do [Sphinx](http://www.sphinx-doc.org) e do tema readthedocs.

```bash
cd docs/
pip install -r requirements.txt
```
Você pode então montar a documentação executando `make <format>` na pasta `docs/`. Execute `make` para receber uma lista de todos os formatos de saída disponíveis.

Se você receber um erro katex, execute `npm install katex`.  Se o erro persistir, execute `npm install -g katex`

> Observação: se você instalou `nodejs` com outro gerenciador de pacotes (por exemplo, `conda`) então `npm` provavelmente irá instalar uma versão de `katex` que não é compatível com a sua versão de `nodejs` e a criação do doc vai falhar. Uma combinação de versões sabidamente funcional é `node@6.13.1` e `katex@0.13.18`. Para instalar esta última com `npm` você pode rodar ```npm install -g katex@0.13.18```

### Versões Anteriores

Instruções de instalação e binaries de versões anteriores do PyTorch podem ser encontradas no [nosso site](https://pytorch.org/previous-versions).

## Começando

Três dicas para começar:
- [Tutoriais: para você começar a entender e usar o PyTorch](https://pytorch.org/tutorials/)
- [Exemplos: códigos PyTorch fáceis de entender em todos os domínios](https://github.com/pytorch/examples)
- [Referência da API](https://pytorch.org/docs/)
- [Glossário](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Recursos

* [PyTorch.org](https://pytorch.org/)
* [PyTorch Tutoriais](https://pytorch.org/tutorials/)
* [PyTorch Exemplos](https://github.com/pytorch/examples)
* [PyTorch Modelos](https://pytorch.org/hub/)
* [Introdução a Deep Learning com PyTorch pela Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Introdução a Machine Learning com PyTorch pela Udacity](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Deep Neural Networks com PyTorch pela Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch Blog](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Comunicação
* Fóruns: Discuta implementações, pesquisas, etc. https://discuss.pytorch.org
* GitHub Issues: Reports de bugs, requisições de features, problemas de instalação, RFCs, pensamentos, etc.
* Slack: O [PyTorch Slack](https://pytorch.slack.com/) recebe um público principal de usuários e desenvolvedores intermediários a experientes do PyTorch para bate-papo geral, discussões, colaboração, etc. Se você é um iniciante buscando por ajuda, o meio principal são os [Fóruns PyTorch](https://discuss.pytorch.org). Se você precisa de um convide para o Slack, preencha esse formulário: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* Notícias: Um email de notícias com anúncios importantes sobre PyTorch (sem SPAM). Você pode se inscrever aqui: https://eepurl.com/cbG0rv
* Página do Facebook: Anúncios importantes sobre PyTorch. https://www.facebook.com/pytorch
* Para obter as diretrizes da marca, visite nosso site em [pytorch.org](https://pytorch.org/)

## Lançamentos e Contribuição

PyTorch possui tipicamente três lançamentos principais por ano. Caso encontre um bug, por favor nos avise [criando uma issue](https://github.com/pytorch/pytorch/issues).

Nós apreciamos todas as contribuições. Se você planeja contribuir através de correções de bugs, por favor o faça sem discussões.

Se você planeja contribuir com novas features, funções utilitárias ou extensões de núcleo, por favor abra primeiro uma issue e discuta a feature conosco. Enviar um pull request sem discussão pode terminar em um PR rejeitado, porque podemos estar levando o núcleo em uma direção diferente da que você pode imaginar.

Para saber mais sobre como contribuir com PyTorch, por favor veja nossa [página de Contribuições](CONTRIBUTING.md). Para mais informações sobre lançamentos do PyTorch, veja a [página de Lançamentos](RELEASE.md).

## O Time

PyTorch é um projeto voltado para a comunidade com vários engenheiros e pesquisadores habilidosos contribuindo para ele.

PyTorch é atualmente administrado por [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), e [Nikita Shulga](https://github.com/malfet) com grandes contribuições vindas de centenas de indivíduos talentosos em várias formas e meios.
Menções honrosas devem ser feitas para: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

Observação: este projeto não está relacionado a [hughperkins/pytorch](https://github.com/hughperkins/pytorch) com o mesmo nome. Hugh é um colaborador valioso para a comunidade do Torch e ajudou em muitas coisas do Torch e do PyTorch.

## Licença

PyTorch tem uma licença BSD, conforme descrito no arquivo [LICENSE](LICENSE).
