![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

---
#### _Leia em outras línguas:_ 
<kbd>[<img title="English" alt="English" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/us.svg" width="30">](../README.md)</kbd>
<kbd>[<img title="Português" alt="Português" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/br.svg" width="30">](README_pt.md)</kbd>
<kbd>[<img title="عربى" alt="عربى" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/sa.svg" width="30">](README_ar.md)</kbd>
<kbd>[<img title="Türkçe" alt="Türkçe" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/tr.svg" width="30">](README_tr.md)</kbd>
<kbd>[<img title="Deutsch" alt="Deutsch" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/de.svg" width="30">](README_de.md)</kbd>

**Essa tradução não é atualizada automaticamente. Alterações no [README.md](../README.md) não são realizadas aqui.**

Este documento pertence a [essa versão](https://github.com/atalman/pytorch/blob/93b27acd035cbfadeae96759db523594b6e6ee92/README.md).
Última atualização: 18/05/2023.

---

PyTorch é um pacote Python que fornece dois recursos de alto nível:

- Computação de tensor (como NumPy) com forte aceleração de GPU
- Redes neurais profundas construídas em um sistema de autograduação baseado em fita

Você pode reutilizar seus pacotes Python favoritos, como NumPy, SciPy e Cython, para estender o PyTorch quando necessário.

Nossa integridade (sinais de Continuous Integration) pode ser encontrada em [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main).

<!-- toc -->

- [Mais Sobre PyTorch](#mais-sobre-pytorch)
  - [Biblioteca de Tensores Pronta para GPU](#biblioteca-de-tensores-pronta-para-gpu)
  - [Redes Neurais Dinâmicas: Autogradiente Baseado em Fita](#redes-neurais-dinâmicas-autogradiente-baseado-em-fita)
  - [Python Primeiro](#python-primeiro)
  - [Experiências Imperativas](#experiências-imperativas)
  - [Rápido e Leve](#rápido-e-leve)
  - [Extensões sem Dor](#extensões-sem-dor)
- [Instalação](#instalação)
  - [Binaries](#binaries)
    - [Plataformas NVIDIA Jetson](#plataformas-nvidia-jetson)
  - [Da Fonte](#da-fonte)
    - [Pré-requisitos](#pré-requisitos)
    - [Dependências de instalação](#dependências-de-instalação)
    - [Obter a Fonte PyTorch](#obter-a-fonte-pytorch)
    - [Instalar PyTorch](#instalar-pytorch)
      - [Ajustar Opções de Build (Opcional)](#ajustar-opções-de-build-opcional)
  - [Imagem Docker](#imagem-docker)
    - [Usando imagens pré-prontas](#usando-imagens-pré-prontas)
    - [Criando a própria imagem](#criando-a-própria-imagem)
  - [Montando Documentação](#montando-documentação)
  - [Versões Anteriores](#versões-anteriores)
- [Começando](#começando)
- [Recursos](#recursos)
- [Comunicação](#comunicação)
- [Lançamentos e Contribuição](#lançamentos-e-contribuição)
- [O Time](#o-time)
- [Licença](#licença)

<!-- tocstop -->

## Mais Sobre PyTorch

Em um nível granular, o PyTorch é uma biblioteca que consiste nos seguintes componentes:

| Componente                                                                        | Descrição                                                                                                                            |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)                           | Uma biblioteca Tensor como NumPy, com forte suporte a GPU                                                                                    |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html)               | Uma biblioteca de diferenciação automática baseada em fita (tape-based) que suporta todas as operações diferenciáveis do Tensor em torch                              |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)                         |Uma pilha de compilação (TorchScript) para criar modelos serializáveis e otimizáveis a partir do código PyTorch                                       |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)                           | Uma biblioteca de redes neurais profundamente integrada com autograd projetada para máxima flexibilidade                                              |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Multiprocessamento Python, mas com compartilhamento de memória mágica de tensores de torch entre processos. Útil para carregamento de dados e treinamento Hogwild |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)                      | DataLoader e outras funções utilitárias para conveniência                                                                                  |

Normalmente, o PyTorch é usado como:

- Um substituto para o NumPy para usar o poder das GPUs.
- Uma plataforma de pesquisa de aprendizado profundo que fornece flexibilidade e velocidade máximas.

Elaborando mais:

### Biblioteca de Tensores Pronta para GPU

Se você usa NumPy, então você usou Tensores (a.k.a. ndarray).

![Tensor illustration](../docs/source/_static/img/tensor_illustration.png)

O PyTorch fornece tensores que podem residir na CPU ou na GPU e acelera bastante a computação.

Fornecemos uma ampla variedade de rotinas de tensor para acelerar e atender às suas necessidades de computação científica, como divisão, indexação, operações matemáticas, álgebra linear e reduções.
E eles são rápidos!

### Redes Neurais Dinâmicas: Autogradiente Baseado em Fita

O PyTorch tem uma maneira única de construir redes neurais: usando e reproduzindo um gravador.

A maioria das estruturas, como TensorFlow, Theano, Caffe e CNTK, tem uma visão estática do mundo.
É preciso construir uma rede neural e reutilizar a mesma estrutura repetidamente.
Mudar a maneira como a rede se comporta significa que é preciso começar do zero.

Com o PyTorch, usamos uma técnica chamada diferenciação automática de modo reverso, que permite
mude a maneira como sua rede se comporta arbitrariamente com atraso ou sobrecarga zero. Nossa inspiração vem
de vários trabalhos de pesquisa sobre este tópico, bem como trabalhos atuais e anteriores, como
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

Embora essa técnica não seja exclusiva do PyTorch, é uma das implementações mais rápidas até hoje.
Você obtém o melhor em velocidade e flexibilidade para sua pesquisa maluca.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

### Python Primeiro

PyTorch não é uma ligação Python em um monolítico C++ framework.
Ele foi desenvolvido para ser profundamente integrado ao Python.
Você pode usá-lo naturalmente como você usaria [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) etc.
Você pode escrever suas novas camadas de rede neural no próprio Python, usando suas bibliotecas favoritas
e usar pacotes como [Cython](https://cython.org/) e [Numba](http://numba.pydata.org/).
Nosso objetivo é não reinventar a roda quando apropriado.

### Experiências Imperativas

O PyTorch foi projetado para ser intuitivo, linear em pensamento e fácil de usar.
Quando você executa uma linha de código, ela é executada. Não existe uma visão assíncrona do mundo.
Quando você entra em um depurador ou recebe mensagens de erro e rastreamentos de pilha, é fácil entendê-los.
O rastreamento de pilha aponta exatamente para onde seu código foi definido.
Esperamos que você nunca gaste horas depurando seu código por causa de rastreamentos de pilha ruins ou mecanismos de execução assíncronos e opacos.

### Rápido e Leve

O PyTorch tem sobrecarga mínima de estrutura. Integramos bibliotecas de aceleração
como [Intel MKL](https://software.intel.com/mkl) e NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl)) para maximizar a velocidade.
No núcleo, sua CPU e GPU Tensor e back-ends de rede neural são maduros e foram testados por anos.

Portanto, o PyTorch é bastante rápido - independentemente de você executar redes neurais pequenas ou grandes.

O uso de memória no PyTorch é extremamente eficiente em comparação com o Torch ou algumas das alternativas.
Escrevemos alocadores de memória personalizados para a GPU para garantir que seus modelos de aprendizado profundo são extremamente eficientes em termos de memória. Isso permite que você treine modelos de aprendizado profundo maiores do que antes.

### Extensões sem Dor

Escrever novos módulos de rede neural ou fazer interface com a API Tensor do PyTorch foi projetado para ser direto e com abstrações mínimas.

Você pode escrever novas camadas de rede neural em Python usando a API da torch [ou suas bibliotecas favoritas baseadas em NumPy, como SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

Se você deseja escrever suas camadas em C/C++, fornecemos uma API de extensão conveniente que é eficiente e com o mínimo de clichê.
Nenhum código wrapper precisa ser escrito. Você pode ver um tutorial [aqui](https://pytorch.org/tutorials/advanced/cpp_extension.html) e um exemplo [aqui](https://github.com/pytorch/extension-cpp).

## Instalação

### Binaries

Os comandos para instalar binários via Conda ou pip wheels estão em nosso site: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### Plataformas NVIDIA Jetson

Python wheels para NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, e Jetson AGX Orin são fornecidos [aqui](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) e o contêiner L4T é publicado [aqui](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

Eles exigem o JetPack 4.2 e superior, e [@dusty-nv](https://github.com/dusty-nv) e [@ptrblck](https://github.com/ptrblck) estão mantendo elas.

### Da Fonte

#### Pré-requisitos

Se você estiver instalando a partir da fonte, precisará de:

- Python 3.8 ou mais tarde (para Linux, Python 3.8.1+ é preciso).
- A C++17 compilador compatível, como clang.

Recomendamos a instalação de um [Anaconda](https://www.anaconda.com/distribution/#download-section) ambiente. Você obterá uma BLAS library (MKL) de alta qualidade e obterá versões de dependência controladas, independentemente da sua distro Linux.

Se você deseja compilar com suporte CUDA, instale o seguinte (observe que CUDA não é compatível com macOS):

- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.0 ou mais atual
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 ou mais atual
- [Compiler](https://gist.github.com/ax3l/9489132) compatível com CUDA

Nota: Você pode consultar o [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) para versões cuDNN com os vários CUDA suportados, CUDA driver e NVIDIA hardware.

Se você deseja desabilitar o suporte CUDA, exporte a variável de ambiente `USE_CUDA=0`.
Outras variáveis de ambiente potencialmente úteis podem ser encontradas em `setup.py`.

Se você está criando para as plataformas Jetson da NVIDIA (Jetson Nano, TX1, TX2, AGX Xavier), instruções para instalar o PyTorch para Jetson Nano estão [disponívei aqui](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

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

> _Obs:_ Se você estiver usando [Anaconda](https://www.anaconda.com/distribution/#download-section), você pode ter um erro causado pelo vinculador(linker):
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

Escolha corretamente a versão do Visual Studio.

PyTorch CI usa Visual C++ BuildTools, que vêm com Visual Studio Enterprise,
Professional, ou Community Editions. Você também pode instalar as ferramentas de compilação de
https://visualstudio.microsoft.com/visual-cpp-build-tools/. As ferramentas de construção _não vêm_ com Visual Studio Code por padrão.

Se você quer construir legacy python code, por favor consulte [Construindo legacy code e CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**Builds somente de CPU**

Nesse modo, os cálculos do PyTorch serão executados em sua CPU, não em sua GPU

```cmd
conda activate
python setup.py develop
```

Nota sobre o OpenMP: A implementação OpenMP desejada é Intel OpenMP (iomp). Para vincular ao iomp, você precisará baixar manualmente a biblioteca e configurar o ambiente de construção ajustando `CMAKE_INCLUDE_PATH` e `LIB`. A instrução [aqui](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) é um exemplo para configurar MKL e Intel OpenMP. Sem essas configurações para CMake, Microsoft Visual C OpenMP runtime (vcomp) será usado.

**Build baseada em CUDA**

Nesse modo, os cálculos do PyTorch aproveitarão sua GPU via CUDA para processamento de números mais rápido

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) é necessário para o build o Pytorch com CUDA.
NVTX é uma parte de CUDA distributive, onde é chamado "Nsight Compute".Para instalá-lo em um CUDA já instalado, execute a instalação do CUDA novamente e marque a caixa de seleção correspondente.
Certifique-se de que CUDA com Nsight Compute é instalado depois do Visual Studio.

Atualmente, VS 2017 / 2019, e Ninja são suportados como o gerador de CMake. Se `ninja.exe` é detectado em `PATH`, então Ninja será usado como gerador padrão, caso contrário, ele usará VS 2017 / 2019.
<br/> Se Ninja for selecionado como o gerador, o MSVC mais recente será selecionado como a cadeia de ferramentas subjacente(toolchain).

Bibliotecas adicionais, como [Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), e [Sccache](https://github.com/mozilla/sccache) muitas vezes são necessárias. Por favor, consulte o [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) para instalá-los.

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
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^] -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

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

**NOTA:** Deve ser criada com uma versão docker > 18.06

O `Dockerfile` é fornecido para criar imagens com suporte CUDA 11.1 e cuDNN v8.
Você pode passar a variável `PYTHON_VERSION=x.y` para especificar qual versão do Python será usada pelo Miniconda, ou deixá-la indefinida para usar o padrão.

```bash
make -f docker.Makefile
# imagens are rotuladas como docker.io/${your_docker_username}/pytorch
```

Você também pode passar a variável de ambiente `CMAKE_VARS="..."` para especificar variáveis CMake adicionais a serem passadas ao CMake durante a build. Veja [setup.py](../setup.py) para a lista de variáveis disponíveis.

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
* [Redes Neurais Profundas com PyTorch pela Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
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

Para saber mais sobre como contribuir com PyTorch, por favor veja nossa [página de Contribuições](../CONTRIBUTING.md). Para mais informações sobre lançamentos do PyTorch, veja a [página de Lançamentos](../RELEASE.md).

## O Time

PyTorch é um projeto voltado para a comunidade com vários engenheiros e pesquisadores habilidosos contribuindo para ele.

PyTorch é atualmente administrado por [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), e [Nikita Shulga](https://github.com/malfet) com grandes contribuições vindas de centenas de indivíduos talentosos em várias formas e meios.
Menções honrosas devem ser feitas para: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

Observação: este projeto não está relacionado a [hughperkins/pytorch](https://github.com/hughperkins/pytorch) com o mesmo nome. Hugh é um colaborador valioso para a comunidade do Torch e ajudou em muitas coisas do Torch e do PyTorch.

## Licença

PyTorch tem uma licença BSD, conforme descrito no arquivo [LICENSE](../LICENSE).
