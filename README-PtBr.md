![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------

PyTorch é um pacote de Python  que fornece dois recursos de alto nível:
- Cálculo do tensor (Parecido com NumPy) com uma forte aceleração de GPU
- Redes neurais profundas construídas em um sistema autograd baseado em tape

Você pode usar sueus pacotes favoritos de Python por exemplo NumPy, SciPy, e Cython para aumentar PyTorch quando necessario.

Nossa Trunk health (Sinais de Integração Contínua) pode ser encontrado em [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/master).

<!-- toc -->

- [Mais sobre PyTorch](#more-about-pytorch)
  - [Uma biblioteca de tensores pronta para GPU](#a-gpu-ready-tensor-library)
  - [Redes Neurais Dinâmicas: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
  - [Python primeiro](#python-first)
  - [Experiências Imperativas](#imperative-experiences)
  - [Rapido e leve](#fast-and-lean)
  - [Extensões Sem Dor](#extensions-without-pain)
- [Instalações](#installation)
  - [Binarias](#binaries)
    - [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
  - [Sobre a Fonte](#from-source)
    - [Pré-requisitos](#prerequisites)
    - [Instalar dependências](#install-dependencies)
    - [Consiga a Fonte do PyTorch](#get-the-pytorch-source)
    - [Instalar PyTorch](#install-pytorch)
      - [Ajuste as opções de Build (Opcional)](#adjust-build-options-optional)
  - [Imagem do Docker](#docker-image)
    - [Usando imagens pré-construídas](#using-pre-built-images)
    - [Construindo a imagem você mesmo](#building-the-image-yourself)
  - [Construindo a Documentação](#building-the-documentation)
  - [Versões prévias](#previous-versions)
- [Começando](#getting-started)
- [Recursos](#resources)
- [Comunicação](#communication)
- [Lançamentos e contribuições](#releases-and-contributing)
- [O Time](#the-team)
- [Licença](#license)

<!-- tocstop -->

## Mais sobre o PyTorch

Em um nível granular, PyTorch é uma biblioteca que consiste nos seguintes componentes:

| Componente | Descrição |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | Uma biblioteca Tensor como NumPy, com forte suporte a GPU |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Uma tape-based biblioteca de diferenciação automáticaque suporta todas as operações diferenciáveis do tensor em torch |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | Uma pilha de compilação (TorchScript) para criar modelos serializáveis e otimizáveis a partir do códico do PyTorch |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | Uma biblioteca de redes neurais profundamente integrada com autograd projetada para máxima flexibilidade |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Multiprocessamento dePython , mas com compartilhamento de magico de memoria  de torch Tensors entre os processos. Útil para carregamento de dados e treinamento de Hogwild |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader e outras funções utilitárias para conveniência |

Usualmente, PyTorch é usado como:

- Um substituto para o NumPy para usar o poder das GPUs.
- Uma plataforma de pesquisa de aprendizado profundo que fornece flexibilidade e velocidade máximas.

Elaborando mais:

### Uma biblioteca de tensores pronta para a GPU

Se você usa NumPy, então você usou Tensores (a.k.a. ndarray).

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch fornece tensores que podem residir na CPU ou na GPU e acelera o
cálculo por uma quantidade enorme.

Fornecemos uma ampla variedade de rotinas de tensores para acelerar e atender às suas necessidades de computação científica
como separação, indexação, operações matemáticas, álgebra Linear, reduções.
E eles são rápidos!

### Redes Neurais Dinâmicas: Tape-Based Autograd

PyTorch tem uma maneira única de construir redes neurais: usando e reproduzindo um gravador tape.

A maioria dos frameworks como TensorFlow, Theano, Caffe, e CNTK tem uma visão estática do mundo.
Alguem precisou construir uma rede neural e reutilizar a mesma estrutura repetidamente.
Mudando a forma como a rede se comporta significando que é preciso começar do zero.

Com PyTorch,usamos uma técnica chamada autodiferenciação de modo reverso,o que lhe permite
mude a maneira como sua rede se comporta arbitrariamente com zero atraso ou sobrecarga. Nossa inspiração vem
de vários trabalhos de pesquisa sobre este tópico, bem como trabalhos atuais e anteriores, como:
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

Embora esta técnica não seja exclusiva do PyTorch, é uma das implementações mais rápidas até hoje.
Você obtém o melhor de velocidade e flexibilidade para sua pesquisa maluca.

![Gráfico dinâmico](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch não é um ligação Python em um monolítico C++ framework.
Ele é construído para ser profundamente integrado ao Python.
Você pode usá-lo naturalmente como você usaria [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) etc.
Você pode escrever suas novas camadas de rede neural no próprio Python, usando suas bibliotecas favoritas
e usar pacotes como [Cython](https://cython.org/) e [Numba](http://numba.pydata.org/).
Nosso objetivo é não reinventar a roda quando apropriado.

### Experiências Imperativas

O PyTorch foi projetado para ser intuitivo, linear em pensamento e fácil de usar.
Quando você executa uma linha de código, ela é executada. Não há uma visão assíncrona do mundo.
Quando você entra em um depurador ou recebe mensagens de erro e rastreamentos de pilha, entendê-los é simples.
O rastreamento de pilha aponta exatamente para onde seu código foi definido.
Esperamos que você nunca gaste horas depurando seu código por causa de rastreamentos de pilha ruins ou mecanismos de execução assíncronos e opacos.

### Rapido e leve

PyTorch tem sobrecarga mínima de estrutura.Integramos bibliotecas de aceleração
tal como [Intel MKL](https://software.intel.com/mkl) e NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl)) para maximizar a velocidade.
No centro, sua CPU e GPU Tensor e back-ends de rede neural
são maduros e foram testados por anos.

Hence, PyTorch é bem rápido – se você executa pequenas ou grandes redes neurais.

O uso de memória no PyTorch é extremamente eficiente em comparação com o Torch ou algumas das alternativas.
Escrevemos alocadores de memória personalizados para a GPU para garantir que
seus modelos de Deep learning são extremamente eficientes em termos de memória.
Isso permite que você treine modelos de Deep learning maiores do que antes.

### Extensões Sem Dor

Escrevendo novos módulos de rede neural, ou a interface com a API do Tensor do PyTorch foi projetada para ser direta
e com abstrações mínimas.

Você pode escrever novas camadas de rede neural em Python usando a API da tocha
[ou suas bibliotecas favoritas baseadas em NumPy, como SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

Se você deseja escrever suas camadas em C/C++, fornecemos uma API de extensão conveniente que é eficiente e com o mínimo de clichê.
Nenhum código wrapper precisa ser escrito. Você pode ver [um tutorial aqui](https://pytorch.org/tutorials/advanced/cpp_extension.html) e [um exemplo aqui](https://github.com/pytorch/extension-cpp).


## Instalação

### Binários
Comandos para instalar binários via Conda ou pip wheels estão em seu website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


#### Plataforma NVIDIA Jetson 

Rodas Python para NVIDIA Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, e Jetson AGX Orin são fornecidos [aqui](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

Eles requerem JetPack 4.2 ou maiores, e [@dusty-nv](https://github.com/dusty-nv) e [@ptrblck](https://github.com/ptrblck) estão mantendo-os.


### Da fonte

#### Pré-requisitos
Se você estiver instalando a partir da fonte, precisará:
- Python 3.7 ou superior (para Linux, Python 3.7.6+ or 3.8.1+ é necessario)
- Um compilador compativel com C++14 , como estrondo

Recomendamos a instalação de um [Anaconda](https://www.anaconda.com/distribution/#download-section) o meio.Você obterá uma biblioteca BLAS de alta qualidade (MKL) e obterá versões de dependência controladas, independentemente da sua distribuição Linux.

Se você deseja compilar com suporte CUDA, instale o seguinte (observe que CUDA não é compatível com macOS)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 10.2 ou maior
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 ou maior
- [Compilador](https://gist.github.com/ax3l/9489132) compativel com CUDA

Nota: Você pode consultar o [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf)para versões cuDNN com os vários CUDA, driver CUDA e hardware NVIDIA suportados

Se você deseja desabilitar o suporte CUDA, exporte a variável de ambiente `USE_CUDA=0`.
Outras variáveis de ambiente potencialmente úteis podem ser encontradas em `setup.py`.

Se você está criando para as plataformas NVIDIA's Jetson (Jetson Nano, TX1, TX2, AGX Xavier), Instruções para instalar o PyTorch para Jetson Nano estão [disponiveis aqui](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

Se você deseja compilar com suporte a ROCm, instale
- [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) 4.0 ou uma instalação superior
- ROCm atualmente é suportado apenas para sistemas Linux.

Se você deseja desabilitar o suporte ROCM, exporte a variável de ambiente `USE_ROCM=0`.
Outras variáveis de ambiente potencialmente úteis podem ser encontradas em `setup.py`.

#### Instalar dependências

**Comum**

```bash
conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
```

**No Linux**

```bash
conda install mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```

**No MacOS**

```bash
# Add this package on intel x86 processor machines only
conda install mkl mkl-include
# Add these packages if torch.distributed is needed
conda install pkg-config libuv
```

**No Windows**

```bash
conda install mkl mkl-include
# Add these packages if torch.distributed is needed.
# Distributed package support on Windows is a prototype feature and is subject to changes.
conda install -c conda-forge libuv=1.39
```

#### Obtenha a fonte do PyTorch
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive --jobs 0
```

#### Instalar PyTorch
**no Linux**

Se você estiver compilando para AMD ROCm, primeiro execute este comando:
```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
```

Install PyTorch
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

Note que se você estiver usando [Anaconda](https://www.anaconda.com/distribution/#download-section), você pode ter um erro causado pelo vinculador:

```plaintext
build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized
collect2: error: ld returned 1 exit status
error: command 'g++' failed with exit status 1
```

This is caused by `ld` from the Conda environment shadowing the system `ld`. You should use a newer version of Python that fixes this issue. The recommended Python version is 3.7.6+ and 3.8.1+.

**no macOS**

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py develop
```

**no Windows**

Escolha a versão correta do Visual Studio.

Às vezes, há regressões em novas versões do Visual Studio, então
é melhor usar a mesma versão do Visual Studio [16.8.5](https://github.com/pytorch/pytorch/blob/master/.circleci/scripts/vs_install.ps1) como Pytorch CI's.

PyTorch CI usa visual C++ BuildTools, que vêm com o Visual Studio Enterprise,
Profissional, ou edições da comunidade. Você também pode instalar as ferramentas de compilação de
https://visualstudio.microsoft.com/visual-cpp-build-tools/. As ferramentas de construção *do not*
vêm com o Visual Studio Code por padrão.

Se você deseja criar código python herdado, consulte [Construindo em código legado e CUDA](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**Compilações somente de CPU**

Nesse modo, os cálculos do PyTorch serão executados em sua CPU, não em sua GPU

```cmd
conda activate
python setup.py develop
```

Nota sobre OpenMP: A implementação OpenMP desejada é Intel OpenMP (iomp). Para vincular ao iomp, você precisará baixar manualmente a biblioteca e configurar o ambiente de construção ajustando `CMAKE_INCLUDE_PATH` e `LIB`. As instruções [aqui](https://github.com/pytorch/pytorch/blob/master/docs/source/notes/windows.rst#building-from-source) é um exemplo para configurar MKL e Intel OpenMP. Sem essas configurações para o CMake, o tempo de execução do Microsoft Visual C OpenMP (vcomp) será usado.

**Compilação baseada em CUDA**

Nesse modo, os cálculos do PyTorch aproveitarão sua GPU via CUDA para processamento de números mais rápido

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) é necessário para construir Pytorch com CUDA.
NVTX faz parte da distribuição CUDA, onde é chamado "Nsight Compute". Para instalá-lo em um CUDA já instalado, execute a instalação do CUDA novamente e marque a caixa de seleção correspondente.
Certifique-se de que o CUDA com Nsight Compute esteja instalado após o Visual Studio.

Atualmente, VS 2017 / 2019, e Ninja são suportados como o gerador de CMake. Se `ninja.exe` é detectado em `PATH`, então o Ninja será usado como gerador padrão, caso contrário, usará o VS 2017 / 2019.
<br/> Se Ninja for selecionado como gerador, o MSVC mais recente será selecionado como a cadeia de ferramentas subjacente.

Bibliotecas adicionais, como
[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), e [Sccache](https://github.com/mozilla/sccache) muitas vezes são necessários. Por favor, consulte o [installation-helper](https://github.com/pytorch/pytorch/tree/master/.jenkins/pytorch/win-test-helpers/installation-helpers) para instalá-los.

Você pode consultar o [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/win-test-helpers/build_pytorch.bat) script para algumas outras configurações de variáveis de ambiente


```cmd
cmd

:: Set the environment variables after you have downloaded and unzipped the mkl package,
:: else CMake would throw an error as `Could NOT find OpenMP`.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Read the content in the previous section carefully before you proceed.
:: [Optional] If you want to override the underlying toolset used by Ninja and Visual Studio with CUDA, please run the following script block.
:: "Visual Studio 2019 Developer Command Prompt" will be run automatically.
:: Make sure you have CMake >= 3.12 before you do this when you use the Visual Studio generator.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Optional] If you want to override the CUDA host compiler
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python setup.py develop

```

##### Ajustar opções de construção (Opcional)

Você pode ajustar a configuração das variáveis cmake opcionalmente (sem compilar primeiro), fazendo
a seguir. Por exemplo, ajustar os diretórios pré-detectados para CuDNN ou BLAS pode ser feito
com tal passo.

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

### Imagem do Docker

#### Usando imagens pré-construídas

Você também pode extrair uma imagem do docker pré-criada do Docker Hub e executar com o docker v19.03+

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Observe que o PyTorch usa memória compartilhada para compartilhar dados entre processos,portanto, se o multiprocessamento da tocha for usado (por exemplo.
para carregadores de dados multithread) o tamanho do segmento de memória compartilhada padrão com o qual o contêiner é executado não é suficiente, e você
deve aumentar o tamanho da memória compartilhada com `--ipc=host` ou `--shm-size` opções de linha de comando para `nvidia-docker run`.

#### Construindo a imagem você mesmo

**NOTA:** Deve ser construído com uma versão do docker > 18.06

O `Dockerfile` é fornecido para criar imagens com suporte CUDA 11.1 e cuDNN v8.
Você pode passar a variável `PYTHON_VERSION=x.y` para especificar qual versão do Python deve ser usada pelo Miniconda, ou deixá-la
desdefinido para usar o padrão.
```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

### Construindo a Documentação

Para criar documentação em vários formatos, você precisará do [Sphinx](http://www.sphinx-doc.org) e do
leia o tema dos documentos.

```bash
cd docs/
pip install -r requirements.txt
```
Você pode então construir a documentação executando `make <format>` de
`docs/` Arquivo. Inicie `make` para obter uma lista de todos os formatos de saída disponíveis.

Se você receber um erro katex, execute `npm install katex`.  Se persistir, tente
`npm install -g katex`

> Note: if you installed `nodejs` with a different package manager (e.g.,
`conda`) then `npm` will probably install a version of `katex` that is not
compatible with your version of `nodejs` and doc builds will fail.
A combination of versions that is known to work is `node@6.13.1` and
`katex@0.13.18`. To install the latter with `npm` you can run
```npm install -g katex@0.13.18```

### Versões prévias

Instruções de instalação e binários para versões anteriores do PyTorch podem ser encontradas
sobre [nosso website](https://pytorch.org/previous-versions).


## Começando

Três ponteiros para você começar:
- [Tutorial: começar a entender e usar PyTorch](https://pytorch.org/tutorials/)
- [Exemplos: código PyTorch fácil de entender em todos os domínios](https://github.com/pytorch/examples)
- [A referencia API](https://pytorch.org/docs/)
- [Glossario](https://github.com/pytorch/pytorch/blob/master/GLOSSARY.md)

## Recursos

* [PyTorch.org](https://pytorch.org/)
* [PyTorch Tutoriais]=(https://pytorch.org/tutorials/)
* [PyTorch Examplos](https://github.com/pytorch/examples)
* [PyTorch Modelos](https://pytorch.org/hub/)
* [Iniciar no Deep Learning com PyTorch para Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Iniciar no Machine Learning com PyTorch para Udacity](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Deep Neural Networks com PyTorch para Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch Blog](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Comunicação
* Forums: Discutir implementações, pesquisas, etc. https://discuss.pytorch.org
* GitHub Problemas: Relatórios de bugs, solicitações de recursos, problemas de instalação, RFCs, pensamentos, etc.
* Slack: O [PyTorch Slack](https://pytorch.slack.com/) hospeda um público principal de usuários moderados a experientes do PyTorch e desenvolvedores para bate-papo geral, discussões online, colaboração, etc. Se você é um iniciante procurando ajuda, o meio principal é [PyTorch Forums](https://discuss.pytorch.org). Se você precisa de um convite para Slack, preencha este formulário: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* Boletim de Notícias: No-noise, um boletim informativo por e-mail unidirecional com anúncios importantes sobre o PyTorch. Você pode se inscrever aqui: https://eepurl.com/cbG0rv
* Pagina no Facebook: Anúncios importantes sobre o PyTorch. https://www.facebook.com/pytorch
* Para obter as diretrizes da marca, visite nosso site em [pytorch.org](https://pytorch.org/)

## Lançamentos e contribuições

O PyTorch tem um ciclo de lançamento de 90 dias (versões principais). Por favor, deixe-nos saber se você encontrar um bug [apresentando o problema](https://github.com/pytorch/pytorch/issues).

Agradecemos todas as contribuições. Se você planeja contribuir com correções de bugs, faça-o sem mais discussões.

Se você planeja contribuir com novos recursos, funções utilitárias ou extensões para o núcleo, primeiro abra um problema e discuta o recurso conosco.
Enviar um PR sem discussão pode acabar resultando em um PR rejeitado porque podemos estar levando o núcleo em uma direção diferente da que você pode estar ciente.

Para saber mais sobre como fazer uma contribuição para o Pytorch, consulte nossa [Página de contribuição](CONTRIBUTING.md).

## O Time

PyTorch é um projeto voltado para a comunidade com vários engenheiros e pesquisadores habilidosos contribuindo para isso.

Atualmente, o PyTorch é mantido por [Adam Paszke](https://apaszke.github.io/), [Sam Gross](https://github.com/colesbury), [Soumith Chintala](http://soumith.ch) e [Gregory Chanan](https://github.com/gchanan) com grandes contribuições vindas de centenas de indivíduos talentosos em várias formas e meios.
Uma lista não exaustiva, mas crescente, precisa mencionar: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

Nota: Este projeto não está relacionado com [hughperkins/pytorch](https://github.com/hughperkins/pytorch) com o mesmo nome.Hugh é um colaborador valioso para a comunidade Torch e ajudou em muitas coisas Torch e PyTorch.

## Licença

O PyTorch possui uma licença no estilo BSD, conforme encontrado no [LICENÇA](LICENSE) file.
