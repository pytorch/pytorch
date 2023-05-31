![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------
#### _Read this in other languages:_

<kbd>[<img title="English" alt="English" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/us.svg" width="30">](../README.md)</kbd>
<kbd>[<img title="Português" alt="Português" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/br.svg" width="30">](README_pt.md)</kbd>
<kbd>[<img title="عربى" alt="عربى" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/sa.svg" width="30">](README_ar.md)</kbd>
<kbd>[<img title="Türkçe" alt="Türkçe" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/tr.svg" width="30">](README_tr.md)</kbd>
<kbd>[<img title="Deutsch" alt="Deutsch" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/de.svg" width="30">](README_de.md)</kbd>

PyTorch, iki yüksek düzeyli özellik sağlayan bir Python paketidir.:
- Güçlü GPU hızlandırma ile Tensör hesaplama (NumPy gibi)
- Bant tabanlı bir otograd sistemi üzerine inşa edilmiş derin sinir ağları

İhtiyaç duyduğunuzda, NumPy, SciPy ve Cython gibi favori Python paketlerinizi PyTorch'u genişletmek için yeniden kullanabilirsiniz.

Bağlantı yolu iyileştirmelerini (Sürekli Entegrasyon işaretleri) hud.pytorch.org adresinde bulabilirsiniz..

<!-- toc -->

- [PyTorch Hakkında Daha Fazlası](#more-about-pytorch)
  - [GPU'ya Hazır Bir Python Kütüphanesi](#a-gpu-ready-tensor-library)
  - [Dinamik Sinir Ağları: Bant Tabanlı Otograd](#dynamic-neural-networks-tape-based-autograd)
  - [Python First](#python-first)
  - [Zorunlu Deneyimler](#imperative-experiences)
  - [Hızlı ve Yalın](#fast-and-lean)
  - [Sorunsuz Uzantılar](#extensions-without-pain)
- [Kurulum](#installation)
  - [Binaries](#binaries)
    - [NVIDIA Jetson Platformları](#nvidia-jetson-platforms)
  - [Kaynak Kodundan](#from-source)
    - [Ön Koşullar](#prerequisites)
    - [Bağımlılıkları Kurun](#install-dependencies)
    - [PyTorch Kaynak Kodunu Edinin](#get-the-pytorch-source)
    - [PyTorch'u Kurma](#install-pytorch)
      - [Yapılandırma Seçeneklerini Ayarlama (İsteğe Bağlı)](#adjust-build-options-optional)
  - [Docker Image](#docker-image)
    - [Önceden oluşturulmuş görüntüleri kullanma](#using-pre-built-images)
    - [Görüntüyü kendiniz oluşturma](#building-the-image-yourself)
  - [Dokümantasyonu Oluşturma](#building-the-documentation)
  - [Önceki Sürümler](#previous-versions)
- [Buradan Başlayın](#getting-started)
- [Kaynaklar](#resources)
- [İletişim](#communication)
- [Releases and Contributing](#releases-and-contributing)
- [Takım](#the-team)
- [Lisans](#license)

<!-- tocstop -->

## PyTorch Hakkında Daha Fazlası

Ayrıntılı bir düzeyde, PyTorch aşağıdaki bileşenlerden oluşan bir kütüphanedir:

| Bileşen | Tanımlama |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | Güçlü GPU desteğine sahip NumPy gibi bir Tensör kütüphanesi |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Torch'taki tüm farklılaştırılabilir Tensör işlemlerini destekleyen bant tabanlı otomatik farklılaştırma kütüphanesi |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | PyTorch kodundan serileştirilebilir ve optimize edilebilir modeller oluşturmak için bir derleme yığını (TorchScript)  |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | Maksimum esneklik için tasarlanmış otograd ile derinlemesine entegre edilmiş bir sinir ağları kütüphanesi |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing ile, PyTorch Tensor'ların süreçler arasında fevkalade bir şekilde bellek paylaşımı yapabilen bir kütüphane. Veri yükleme ve Hogwild eğitimi gibi durumlarda kullanışlıdır. |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | Kolaylık için DataLoader ve diğer yardımcı işlevler |

PyTorch genellikle şu şekilde kullanılır:

- GPU'ların gücünü kullanmak için NumPy'nin yerine geçecek bir araç.
- Maksimum esneklik ve hız sağlayan bir derin öğrenme araştırma platformu.

Daha Fazla Detaylandırma:

### GPU'ya Hazır Bir Tensör Kütüphanesi

NumPy kullanıyorsanız, Tensörleri kullanmışsınızdır (a.k.a. ndarray).

![Tensor gösterimi](../docs/source/_static/img/tensor_illustration.png)

PyTorch, CPU veya GPU üzerinde yaşayabilen Tensörler sağlar ve büyük miktarda hesaplama.

Birçok tensor rutini sunarak bilimsel hesaplama ihtiyaçlarınızı hızlandırmak ve uyumlu hale getirmek için çeşitli seçenekler sunarız.
Bu rutinler arasında dilimleme, indeksleme, matematiksel operasyonlar, lineer cebir ve azaltma gibi işlemler bulunur.
Ve hızlılardır!

### Dinamik Sinir Ağları: Bant Tabanlı Otograd

PyTorch'un sinir ağları oluşturma konusunda benzersiz bir yöntemi vardır: bir bant kaydediciyi kullanma ve tekrar oynatma.

TensorFlow, Theano, Caffe ve CNTK gibi çoğu çerçeve statik bir dünya görüşüne sahiptir.
Bir sinir ağı inşa etmek ve aynı yapıyı tekrar tekrar kullanmak gerekir.
Ağın davranış şeklini değiştirmek, sıfırdan başlamak gerektiği anlamına gelir.

PyTorch ile, ağınızın davranış şeklini sıfır gecikme veya ek yük ile keyfi olarak değiştirmenize olanak tanıyan ters mod otomatik farklılaştırma adı verilen bir teknik kullanıyoruz. İlham kaynağımız gibi güncel ve geçmiş çalışmaların yanı sıra bu konudaki çeşitli araştırma makalelerinden
[torch-otograd](https://github.com/twitter/torch-autograd),
[otograd](https://github.com/HIPS/autograd),
[Zincirleme](https://chainer.org), vb.

Bu teknik PyTorch'a özgü olmasa da, bugüne kadarki en hızlı uygulamalardan biridir.
Çılgın araştırmanız için en iyi hız ve esnekliğe sahip olursunuz.

![Dinamik grafik](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch, monolitik bir C++ çerçevesine bağlanmış bir Python değildir.
Python'a derinlemesine entegre olacak şekilde tasarlanmıştır.
Kullandığınız gibi, doğal bir şekilde kullanabilirsiniz [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) vb.
Yeni sinir ağı katmanlarınızı Python'da yazabilir, favori kütüphanelerinizi kullanabilir ve şu gibi paketleri kullanabilirsiniz [Cython](https://cython.org/) ve [Numba](http://numba.pydata.org/).
Amacımız, uygun olduğu yerlerde tekerleği yeniden icat etmemektir.

### Zorunlu Deneyimler

PyTorch, sezgisel, düşünce açısından doğrusal ve kullanımı kolay olacak şekilde tasarlanmıştır.
Bir kod satırını çalıştırdığınızda, o satır hemen çalışır. Asenkron bir görünüm yoktur.
Bir hata ayıklayıcıya girdiğinizde veya hata mesajları ve yığın izleri aldığınızda, bunları anlamak kolaydır.
Yığın izi, kodunuzun tam olarak nerede tanımlandığını işaret eder.
Kötü yığın izleri veya asenkron ve anlaşılması zor yürütme motorları nedeniyle saatlerce kod hata ayıklama yapmanıza gerek kalmayacağını umuyoruz.

### Hızlı ve Yalın

PyTorch minimum çerçeve ek yüküne sahiptir. Hızlandırma kütüphanelerini entegre ediyoruz örneğin  [Intel MKL](https://software.intel.com/mkl) ve NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl))  hızı en üst düzeye çıkarmak için.
Çekirdekte, CPU ve GPU Tensör ve sinir ağı arka uçları (backend) olgunlaşmıştır ve yıllardır test edilmektedir.

Bu nedenle, PyTorch oldukça hızlıdır - küçük veya büyük sinir ağları bile çalıştırsanız.

PyTorch'ta bellek kullanımı, Torch veya diğer bazı alternatiflere göre son derece verimlidir.
Derin öğrenme modellerinizin maksimum bellek verimliliği sağlanması için GPU için özel bellek ayırıcıları yazdık.
Bu, daha öncekinden daha büyük derin öğrenme modellerini eğitebilmenizi sağlar.

### Sorunsuz Uzantılar

Yeni sinir ağı modülleri yazmak veya PyTorch'un Tensor API'si ile arayüz oluşturmak basit ve minimum soyutlama ile olacak şekilde tasarlanmıştır.

Torch API'sini kullanarak Python'da yeni sinir ağı katmanları yazabilirsiniz
[veya SciPy gibi favori NumPy tabanlı kütüphanelerini](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

Katmanlarınızı C/C++ dilinde yazmak istiyorsanız, verimli ve minimum şablon içeren kullanışlı bir uzantı API'si sunuyoruz.
Sarmalayıcı kod yazılmasına gerek yoktur.Burada [bir öğretici](https://pytorch.org/tutorials/advanced/cpp_extension.html) ve [bir örnek](https://github.com/pytorch/extension-cpp) görebilirsiniz.


## Kurulum

### Binaries
Conda veya pip wheels aracılığıyla ikili dosyaları yüklemek için komutlar web sitemizde bulunmaktadır. [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


#### NVIDIA Jetson Platformları

Conda veya pip wheels aracılığıyla ikili dosyaları yüklemek için komutlar web sitemizde bulunmaktadır. [burada](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) ve L4T kapsayıcıları yayınlanmıştır. [burada](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

JetPack 4.2 ve üstünü gerektirirler ve[@dusty-nv](https://github.com/dusty-nv) ve [@ptrblck](https://github.com/ptrblck) bunları sürdürmektedir.


### Kaynak Kodundan

#### Ön Koşullar
Eğer kaynaktan kurulum yapıyorsanız, şunlara ihtiyacınız olacaktır:
- Python 3.8 veya üstü (Linux için Python 3.8.1+ gereklidir)
- Clang gibi C++17 uyumlu bir derleyici

[Anaconda](https://www.anaconda.com/distribution/#download-section) ortamını kurmanızı şiddetle öneririz. Yüksek kaliteli bir BLAS kütüphanesi (MKL) alırsınız ve Linux dağıtımınız ne olursa olsun kontrol edilen bağımlılık sürümleri elde edersiniz.

CUDA desteği ile derlemek istiyorsanız, aşağıdakileri yükleyin. (CUDA'nın macOS'ta desteklenmediğini unutmayın.)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.0 veya üzeri
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 veya üzeri
- [Derleyici](https://gist.github.com/ax3l/9489132) CUDA ile uyumlu

Not: Desteklenen çeşitli CUDA, CUDA sürücüsü ve NVIDIA donanımına sahip cuDNN sürümleri için [cuDNN Destek Matrisine](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) başvurabilirsiniz.

CUDA desteğini devre dışı bırakmak istiyorsanız, ortam değişkenini dışa aktarın `USE_CUDA=0`.
Diğer potansiyel olarak yararlı ortam değişkenleri şurada bulunabilir `setup.py`.

NVIDIA'nın Jetson platformları (Jetson Nano, TX1, TX2, AGX Xavier),için geliştirme yapıyorsanız, Jetson Nano için PyTorch yükleme talimatları [burada mevcuttur](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

ROCm desteği ile derlemek istiyorsanız
- [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) 4.0 ve üzeri kurulum
- ROCm şu anda yalnızca Linux sistemlerinde desteklenmektedir.

ROCm desteğini devre dışı bırakmak istiyorsanız, ortam değişkenini dışa aktarın `USE_ROCM=0`.
Diğer potansiyel olarak yararlı ortam değişkenleri şurada bulunabilir `setup.py`.

#### Bağımlılıkları Kurun

**Yaygın**

```bash
conda install cmake ninja
# Aşağıdaki "PyTorch Kaynak Kodunu Edinin" bölümünü kullanarak kaynak kodu klonladıktan sonra PyTorch dizininden bu komutu çalıştırın
pip install -r requirements.txt
```

**On Linux**

```bash
conda install mkl mkl-include
# Sadece CUDA için: Gerekirse GPU için LAPACK desteğini ekleyin.
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

# (isteğe bağlı) Eğer inductor/triton ile torch.compile kullanıyorsanız, uyumlu triton sürümünü yükleyin.
# Klonlamadan sonra pytorch dizininden çalıştırın.
make triton
```

**On MacOS**

```bash
# Bu paketi yalnızca Intel x86 işlemcili makinelerde ekleyin.
conda install mkl mkl-include
# Eğer torch.distributed gerekiyorsa, bu paketleri ekleyin.
conda install pkg-config libuv
```

**On Windows**

```bash
conda install mkl mkl-include
# Eğer torch.distributed gerekiyorsa, aşağıdaki paketleri ekleyin.
# Windows üzerinde dağıtılmış paket desteği prototip bir özelliktir ve değişikliklere tabidir.
conda install -c conda-forge libuv=1.39
```

#### PyTorch Kaynak Kodunu Edinin
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# Mevcut bir kontrol dışı güncelleme yapıyorsanız.
git submodule sync
git submodule update --init --recursive
```

#### Pytorch'u Kurma
**On Linux**

Eğer AMD ROCm için derleme yapıyorsanız, önce bu komutu çalıştırın.:
```bash
# Bunu sadece ROCm için derleme yapıyorsanız çalıştırın.
python tools/amd_build/build_amd.py
```

Pytorchu Kurma
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

> _Bunun yanı sıra:_ Eğer [Anaconda](https://www.anaconda.com/distribution/#download-section) kullanıyorsanız, bağlayıcıdan (linker) kaynaklanan bir hata ile karşılabilirsiniz:
>
> ```plaintext
> build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized
> collect2: error: ld returned 1 exit status
> error: command 'g++' failed with exit status 1
> ```
>
> This is caused by `ld` from the Conda environment shadowing the system `ld`. You should use a newer version of Python that fixes this issue. The recommended Python version is 3.8.1+.

**On macOS**

```bash
python3 setup.py develop
```

**On Windows**

Choose Correct Visual Studio Version.

PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
Professional, or Community Editions. You can also install the build tools from
https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools *do not*
come with Visual Studio Code by default.

If you want to build legacy python code, please refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**CPU-only builds**

In this mode PyTorch computations will run on your CPU, not your GPU

```cmd
conda activate
python setup.py develop
```

Note on OpenMP: The desired OpenMP implementation is Intel OpenMP (iomp). In order to link against iomp, you'll need to manually download the library and set up the building environment by tweaking `CMAKE_INCLUDE_PATH` and `LIB`. The instruction [here](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) is an example for setting up both MKL and Intel OpenMP. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.

**CUDA based build**

In this mode PyTorch computations will leverage your GPU via CUDA for faster number crunching

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) is needed to build Pytorch with CUDA.
NVTX is a part of CUDA distributive, where it is called "Nsight Compute". To install it onto an already installed CUDA run CUDA installation once again and check the corresponding checkbox.
Make sure that CUDA with Nsight Compute is installed after Visual Studio.

Currently, VS 2017 / 2019, and Ninja are supported as the generator of CMake. If `ninja.exe` is detected in `PATH`, then Ninja will be used as the default generator, otherwise, it will use VS 2017 / 2019.
<br/> If Ninja is selected as the generator, the latest MSVC will get selected as the underlying toolchain.

Additional libraries such as
[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) are often needed. Please refer to the [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) to install them.

You can refer to the [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) script for some other environment variables configurations


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
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^] -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Optional] If you want to override the CUDA host compiler
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python setup.py develop

```

##### Yapılandırma Seçenekleri Ayarlama (İsteğe bağlı)

You can adjust the configuration of cmake variables optionally (without building first), by doing
the following. For example, adjusting the pre-detected directories for CuDNN or BLAS can be done
with such a step.

On Linux
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

On macOS
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

### Docker Image

#### Using pre-built images

You can also pull a pre-built docker image from Docker Hub and run with docker v19.03+

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Please note that PyTorch uses shared memory to share data between processes, so if torch multiprocessing is used (e.g.
for multithreaded data loaders) the default shared memory segment size that container runs with is not enough, and you
should increase shared memory size either with `--ipc=host` or `--shm-size` command line options to `nvidia-docker run`.

#### Building the image yourself

**NOTE:** Must be built with a docker version > 18.06

The `Dockerfile` is supplied to build images with CUDA 11.1 support and cuDNN v8.
You can pass `PYTHON_VERSION=x.y` make variable to specify which Python version is to be used by Miniconda, or leave it
unset to use the default.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

You can also pass the `CMAKE_VARS="..."` environment variable to specify additional CMake variables to be passed to CMake during the build.
See [setup.py](../setup.py) for the list of available variables.

```bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker.Makefile
```

### Building the Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org) and the
readthedocs theme.

```bash
cd docs/
pip install -r requirements.txt
```
You can then build the documentation by running `make <format>` from the
`docs/` folder. Run `make` to get a list of all available output formats.

If you get a katex error run `npm install katex`.  If it persists, try
`npm install -g katex`

> Note: if you installed `nodejs` with a different package manager (e.g.,
`conda`) then `npm` will probably install a version of `katex` that is not
compatible with your version of `nodejs` and doc builds will fail.
A combination of versions that is known to work is `node@6.13.1` and
`katex@0.13.18`. To install the latter with `npm` you can run
```npm install -g katex@0.13.18```

### Previous Versions

Installation instructions and binaries for previous PyTorch versions may be found
on [our website](https://pytorch.org/previous-versions).


## Getting Started

Three-pointers to get you started:
- [Tutorials: get you started with understanding and using PyTorch](https://pytorch.org/tutorials/)
- [Examples: easy to understand PyTorch code across all domains](https://github.com/pytorch/examples)
- [The API Reference](https://pytorch.org/docs/)
- [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Resources

* [PyTorch.org](https://pytorch.org/)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Examples](https://github.com/pytorch/examples)
* [PyTorch Models](https://pytorch.org/hub/)
* [Intro to Deep Learning with PyTorch from Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Intro to Machine Learning with PyTorch from Udacity](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Deep Neural Networks with PyTorch from Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch Blog](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Communication
* Forums: Discuss implementations, research, etc. https://discuss.pytorch.org
* GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.
* Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* Newsletter: No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: https://eepurl.com/cbG0rv
* Facebook Page: Important announcements about PyTorch. https://www.facebook.com/pytorch
* For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

Typically, PyTorch has three major releases a year. Please let us know if you encounter a bug by [filing an issue](https://github.com/pytorch/pytorch/issues).

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

To learn more about making a contribution to Pytorch, please see our [Contribution page](../CONTRIBUTING.md). For more information about PyTorch releases, see [Release page](../RELEASE.md).

## The Team

PyTorch is a community-driven project with several skillful engineers and researchers contributing to it.

PyTorch is currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions coming from hundreds of talented individuals in various forms and means.
A non-exhaustive but growing list needs to mention: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch) with the same name. Hugh is a valuable contributor to the Torch community and has helped with many things Torch and PyTorch.

## License

PyTorch has a BSD-style license, as found in the [LICENSE](../LICENSE) file.
