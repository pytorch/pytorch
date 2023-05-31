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
  - [Dokümantasyon Oluşturma](#building-the-documentation)
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

**Linux için**

```bash
conda install mkl mkl-include
# Sadece CUDA için: Gerekirse GPU için LAPACK desteğini ekleyin.
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

# (isteğe bağlı) Eğer inductor/triton ile torch.compile kullanıyorsanız, uyumlu triton sürümünü yükleyin.
# Klonlamadan sonra pytorch dizininden çalıştırın.
make triton
```

**MacOS için**

```bash
# Bu paketi yalnızca Intel x86 işlemcili makinelerde ekleyin.
conda install mkl mkl-include
# Eğer torch.distributed gerekiyorsa, bu paketleri ekleyin.
conda install pkg-config libuv
```

**Windows için**

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
**Linux için**

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
> Bu, Conda ortamından `ld`'nin sistem `ld`'yi gölgelemesinden kaynaklanır. Bu sorunu düzeltmek için daha yeni bir Python sürümü kullanmalısınız. Önerilen Python sürümü 3.8.1 veya üzeridir.

**macOS için**

```bash
python3 setup.py develop
```

**Windows için**

Doğru Visual Studio sürümünü seçin.


PyTorch CI, Visual Studio Enterprise, Professional veya Community Sürümleri ile birlikte gelen Visual C++ BuildTools'u kullanır. Derleme araçları şu adresten de yükleyebilirsiniz
https://visualstudio.microsoft.com/visual-cpp-build-tools/. Derleme araçları *yapmaz* varsayılan olarak Visual Studio Code ile birlikte gelir.

Eski Python kodunu derlemek isterseniz, lütfen [Eski kod ve CUDA üzerinde oluşturma ](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**Sadece CPU Derlemeleri**

Bu modda PyTorch hesaplamaları GPU'nuzda değil CPU'nuzda çalışacaktır.

```cmd
conda activate
python setup.py develop
```

OpenMP hakkında not: İstenilen OpenMP uygulaması Intel OpenMP (iomp) kullanmaktır. iomp ile bağlantı kurmak için kütüphaneyi manuel olarak indirmeniz ve `CMAKE_INCLUDE_PATH` ve `LIB` ayarlarını yapılandırarak derleme ortamını ayarlamanız gerekecektir. [Burada](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) yer alan yönerge hem MKL hem de Intel OpenMP kurulumu için bir örnektir. CMake için bu yapılandırmalar olmadan, Microsoft Visual C OpenMP çalışma zamanı (vcomp) kullanılacaktır.

**CUDA tabanlı derleme**

Bu modda PyTorch hesaplamaları, CUDA aracılığıyla GPU'nuzu kullanarak daha hızlı bir şekilde gerçekleştirilir.

PyTorch'un CUDA desteğiyle derlemek için [NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) gereklidir.
NVTX, CUDA dağıtımının bir parçasıdır ve "Nsight Compute" olarak adlandırılır. Mevcut bir CUDA yüklemesine NVTX'i yüklemek için CUDA kurulumunu tekrar çalıştırın ve ilgili onay kutusunu işaretleyin.
Visual Studio'dan sonra Nsight Compute ile CUDA'nın yüklendiğinden emin olun.

Şu anda CMake'in üreticisi olarak VS 2017 / 2019 ve Ninja desteklenmektedir. PATH` içinde `ninja.exe` tespit edilirse, Ninja varsayılan oluşturucu olarak kullanılacaktır, aksi takdirde VS 2017 / 2019'u kullanacaktır.
<br/> Eğer Ninja üretici olarak seçilirse, en son MSVC sürümü temel araç seti olarak seçilecektir.

Genellikle
[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN veya DNNL](https://github.com/oneapi-src/oneDNN), ve [Sccache](https://github.com/mozilla/sccache) gibi ek kütüphanelere ihtiyaç duyulabilir. Bunları yüklemek için lütfen [kurulum-yardımcısı](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) adresine bakın.

Diğer bazı ortam değişkenleri yapılandırmaları için  [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) betiğine başvurabilirsiniz.


```cmd
cmd

:: mkl paketini indirdikten ve açtıktan sonra ortam değişkenlerini ayarlayın,
:: Aksi takdirde CMake `AçıkMP bulunamadı` şeklinde bir hata verir.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Devam etmeden önce bir önceki bölümdeki içeriği dikkatlice okuyun.
:: [İsteğe bağlı] Ninja ve Visual Studio tarafından kullanılan temel araç setini CUDA ile geçersiz kılmak istiyorsanız, lütfen aşağıdaki kod bloğunu çalıştırın.
:: "Visual Studio 2019 Geliştirici Komut İstemi" otomatik olarak çalıştırılacaktır.
:: Visual Studio oluşturucusunu kullanmadan önce CMake >= 3.12'ye sahip olduğunuzdan emin olun.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^] -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [İsteğe bağlı] CUDA host derleyicisini geçersiz kılmak istiyorsanız
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python setup.py develop

```

##### Yapılandırma Seçenekleri Ayarlama (İsteğe bağlı)

Tercihinize bağlı olarak, cmake değişkenlerinin yapılandırmasını (derleme yapmadan önce) aşağıdaki adımları izleyerek ayarlayabilirsiniz. Örneğin, CuDNN veya BLAS için önceden tespit edilen dizinleri ayarlamak bu şekilde yapılabilir.

Linux için
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

macOS için
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

### Docker Görüntüsü

#### Önceden oluşturulmuş görüntüleri kullanma

Docker Hub'dan önce oluşturulmuş bir docker görüntüsünü çekebilir ve docker v19.03+ ile çalıştırabilirsiniz

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

PyTorch'un süreçler arasında veri paylaşmak için paylaşılan bellek kullandığını lütfen unutmayın, bu nedenle torch çoklu işlem kullanılıyorsa (örn. çoklu iş parçacıklı veri yükleyicileri için) kapsayıcıların birlikte çalıştığı varsayılan paylaşılan bellek segmenti boyutu yeterli değildir ve paylaşılan bellek boyutunu `nvidia-docker run` komut satırındaki `--ipc=host` veya `--shm-size` seçenekleriyle artırmalıdır.

#### Görüntüyü kendiniz oluşturma

**NOT:** Docker sürümü > 18.06 olmalıdır.

`Dockerfile` dosyası, CUDA 11.1 ve cuDNN v8 desteği olan görüntülerin oluşturulması için sağlanır.
Miniconda tarafından hangi Python sürümünün kullanılacağını belirtmek için `PYTHON_VERSION=x.y` değişkeni yapılır veya varsayılanı kullanmak için ayarlanmamış olarak bırakabilirsiniz.

```bash
make -f docker.Makefile
# Görüntüler docker.io/${your_docker_username}/pytorch olarak etiketlenir.
```

Ayrıca derleme sırasında CMake'e aktarılacak ek CMake değişkenlerini belirtmek için `CMAKE_VARS="..."` ortam değişkenini de aktarabilirsiniz.
Kullanılabilir değişkenlerin listesi için [setup.py](../setup.py) dosyasına bakabilirsiniz.

```bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker.Makefile
```

### Dökümantasyon Oluşturma

Çeşitli formatlarda dokümantasyon oluşturmak için [Sphinx](http://www.sphinx-doc.org) ve readthedocs temasına ihtiyacınız olacaktır.

```bash
cd docs/
pip install -r requirements.txt
```
Ardından, belgelendirmeyi `docs/` klasöründen `make <format>` komutunu çalıştırarak oluşturabilirsiniz. Tüm mevcut çıktı formatlarının listesini almak için `make` komutunu çalıştırın.

Eğer bir katex hatası alırsanız `npm install katex`.  komutunu çalıştırın.  Eğer hata devam ederse, şu komutu deneyin
`npm install -g katex`

> Not: Eğer `nodejs` i farklı bir paket yöneticisiyle (örneğin,
`conda`) yüklediyseniz `npm` muhtemelen `nodejs` ürümünüzle uyumlu
olmayan bir `katex` sürümü yükler ve belge derlemesi başarısız olur.
Bilinen çalışan bir kombinasyon, `node@6.13.1` ve `katex@0.13.18` sürümleridir.
`npm` ile bunları yüklemek için aşağıdaki komutu çalıştırabilirsiniz:
```npm install -g katex@0.13.18```

### Önceki Sürümler

Eski PyTorch sürümlerinin kurulum talimatları ve binary dosyaları
[website'mizde](https://pytorch.org/previous-versions) bulunabilir.


## Buradan Başlayın

Başlamanız için üç ipucu:
- [Öğreticiler: PyTorch'u anlamaya ve kullanmaya başlamanızı sağlar](https://pytorch.org/tutorials/)
- [Örnekler: tüm alanlarda(domainlerde) anlaşılması kolay PyTorch kodu](https://github.com/pytorch/examples)
- [API Referansı](https://pytorch.org/docs/)
- [Sözlük](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Kaynaklar

* [PyTorch.org](https://pytorch.org/)
* [PyTorch Öğreticileri](https://pytorch.org/tutorials/)
* [PyTorch Örnekleri](https://github.com/pytorch/examples)
* [PyTorch Modelleri](https://pytorch.org/hub/)
* [Udacity'den PyTorch ile Derin Öğrenmeye Giriş](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Udacity'den PyTorch ile Makine Öğrenimine Giriş](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Coursera'dan PyTorch ile Derin Sinir Ağları](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch Blog](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## İletişim
* Forumlar: Uygulamaları, araştırmaları vb. konuları tartışın https://discuss.pytorch.org
* GitHub Sorunları: Hata raporları, özellik talepleri, kurulum sorunları, RFC'ler, düşünceler, vb.
* Slack: [PyTorch Slack](https://pytorch.slack.com/) genel sohbet, çevrimiçi tartışmalar, işbirliği vb. için orta ve deneyimli PyTorch kullanıcıları ve geliştiricilerinden oluşan birincil kitleye ev sahipliği yapmaktadır. [PyTorch Forumlar](https://discuss.pytorch.org). Eğer bir Slack davetiyesine ihtiyacınız varsa, lütfen bu formu doldurun: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* Haber Bülteni: Gereksiz bilgiler olmadan, PyTorch hakkında önemli duyurular içeren tek yönlü bir e-posta bülteni. Buradan kayıt olabilirsiniz: https://eepurl.com/cbG0rv
* Facebook Sayfası: PyTorch hakkında önemli duyurular için https://www.facebook.com/pytorch
* Marka yönergeleri için lütfen web sitemizi ziyaret edin [pytorch.org](https://pytorch.org/)

## Releases and Contributing

Genellikle, PyTorch yılda üç büyük sürüm çıkarır. Eğer bir hata ile karşılaşırsanız lütfen [sorun bildirerek](https://github.com/pytorch/pytorch/issues).

 Tüm katkılarınızı takdir ediyoruz. Eğer hata düzeltmeleriyle katkıda bulunmayı planlıyorsanız, lütfen başka bir tartışma olmaksızın bunu yapın.

Eğer yeni özellikler, yardımcı fonksiyonlar veya çekirdeğe uzantılar eklemeyi planlıyorsanız, lütfen önce bir konu açın ve özelliği bizimle tartışın.
Tartışma yapmadan bir PR göndermek, çekirdeği farklı bir yöne taşıyor olabileceğimiz için reddedilen bir PR ile sonuçlanabilir, çünkü sizin farkında olabileceğinizden farklı bir yol izliyor olabiliriz.

PyTorch'a katkıda bulunmak hakkında daha fazla bilgi edinmek için [Katkı Sayfası'na](../CONTRIBUTING.md) göz atabilirsiniz. PyTorch sürümleri hakkında daha fazla bilgi için [Sürüm Sayfası'n](../RELEASE.md) ziyaret edebilirsiniz..

## The Team

PyTorch is a community-driven project with several skillful engineers and researchers contributing to it.

PyTorch is currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions coming from hundreds of talented individuals in various forms and means.
A non-exhaustive but growing list needs to mention: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch) with the same name. Hugh is a valuable contributor to the Torch community and has helped with many things Torch and PyTorch.

## License

PyTorch has a BSD-style license, as found in the [LICENSE](../LICENSE) file.
