![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

---
#### _اقرأ هذا بلغات أخرى:_

<kbd>[<img title="English" alt="English" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/us.svg" width="30">](../README.md)</kbd>
<kbd>[<img title="Português" alt="Português" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/br.svg" width="30">](README_pt.md)</kbd>
<kbd>[<img title="عربى" alt="عربى" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/sa.svg" width="30">](README_ar.md)</kbd>
<kbd>[<img title="Türkçe" alt="Türkçe" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/tr.svg" width="30">](README_tr.md)</kbd>
<kbd>[<img title="Deutsch" alt="Deutsch" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/de.svg" width="30">](README_de.md)</kbd>

**لا يتم تحديث هذه الترجمة تلقائيًا. لم يتم إجراء التغييرات على [README.md](../README.md) هنا.**

ينتمي هذا المستند إلى [هذا الإصدار](https://github.com/atalman/pytorch/blob/93b27acd035cbfadeae96759db523594b6e6ee92/README.md).
آخر تحديث:18/5/2023

---

مكتبة PyTorch:
هي حزمة برمجية في لغة Python توفر ميزتين على مستوى عالٍ:
- حساب التنسورات (مثل Numpy) بتسارع قوي باستخدام وحدة المعالجة الرسومية (GPU).
- شبكات عصبية عميقة مبنية على نظام تلقائي للتفاضل يعتمد على تقنية الشريط (autograd system)، وهي تقنية تساعد على تحسين أداء الشبكة وتعلمها.

يمكنك إعادة استخدام حزم بايثون المفضلة لديك مثل (NumPy/SciPyan/Cython) لتوسيع حزمة PyTorch عند الحاجة.
 
 
يمكن العثور على مؤشرات صحة جذعنا (تكامل مستمر) في الرابط التالي[hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main).

<!-- toc -->

- [المزيد عن PyTorch](#المزيد-عن-pytorch)
  - [مكتبة Tensor جاهزة للعمل على وحدة معالجة الرسوميات GPU](#مكتبة-tensor-جاهزة-للعمل-على-وحدة-معالجة-الرسوميات-gpu)
  - [شبكات عصبية ديناميكية: نظام تلقائي للتفاضل يعتمد على التسجيل بواسطة الشريط](#شبكات-عصبية-ديناميكية)
  - [بايثون في المقام الأول](#بايثون-في-المقام-الأول)
  - [التجارب الحتمية](#التجارب-الحتمية)
  - [سريع وخفيف](#سريع-وخفيف)
  - [سهولة في التوسع](#سهولة-في-التوسع)
- [التثبيت](#التثبيت)
  - [الملفات التنفيذية](#الملفات-التنفيذية)
    - [منصات NVIDIA Jetson](#منصات-nvidia-jetson)
  - [من المصدر](#من-المصدر)
    - [المتطلبات الأساسية](#المتطلبات-الأساسية)
    - [تثبيت التبعيات أو الاعتماديات](#تثبيت-التبعيات-أو-الاعتماديات)
    - [الحصول على مصدر PyTorch](#الحصول-على-مصدر-PyTorch)
    - [تثبيت PyTorch](#تثبيت-PyTorch)
      - [ضبط خيارات البناء (اختياري)](#ضبط-خيارات-البناء-اختياري)
  - [صورة دوكر (Docker Image)](#صورة-دوكر-docker-image)
    - [استخدام صور مبنية مسبقًا](#استخدام-صور-مبنية-مسبقًا)
    - [بناء الصورة بنفسك](#بناء-الصورة-بنفسك)
  - [إنشاء الوثائق](#إنشاء-الوثائق)
  - [الإصدارات السابقة](#الإصدارات-السابقة)
- [البدء بالاستخدام](#البدء-بالاستخدام)
- [موارد](#موارد)
- [التواصل](#التواصل)
- [الإصدارات والمساهمة](#الإصدارات-والمساهمة)
- [الفريق](#الفريق)
- [الترخيص](#الترخيص)

<!-- tocstop -->

## المزيد عن PyTorch

على المستوى التفصيلي، تعد PyTorch مكتبة تتكون من العناصر التالية:
| العنصر | الوصف |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | مع دعم قوي لوحدة المعالجة الرسومية Numpy مكتبة تينسور مثل |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | torch القابلة للتفاضل في Tensor مكتبة تفاضل تلقائية قائمة على الأشرطة تدعم جميع عمليات |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | PyTorch لإنشاء نماذج قابلة للتسلسل والتحسين من رمز TorchScript مجموعة تراكيب |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | مصممة لتوفير أقصى قدر من المرونة autograd مكتبة شبكات عصبية متكاملة بشكل عميق مع |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Hogwild  لتحميل البيانات وتدريب torch  مع مشاركة ذاكرة سحرية لتوزيع التينسورات في multiprocessing مكتبة من نظام|
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | ووظائف أخرى للراحة والاستخدام الملائم DataLoader هي |


عادةً يتم استخدام PyTorch إما كـ:

- بديل لـ NumPy للاستفادة من قوة وحدات المعالجة الرسومية (GPUs).
- منصة بحث التعلم العميق (Deeb lerning)التي توفر أقصى قدر من المرونة والسرعة.

مزيد من التفصيل:

### مكتبة Tensor جاهزة للعمل على وحدة معالجة الرسوميات GPU

إذا كنت تستخدم NumPy ، فهذا يعني أنك استخدمت Tensors (المعروف أيضًا باسم ndarray).

![Tensor illustration](../docs/source/_static/img/tensor_illustration.png)

توفر PyTorch تينسورات (Tensors) التي يمكن أن تعيش إما على وحدة المعالجة المركزية أو وحدة معالجة الرسومات (GPU) وتسريع الحساب بمقدار ضخم.

نحن نقدم مجموعة متنوعة من إجراءات التينسور لتسريع وتناسب احتياجاتك الحسابية العلمية مثل التقطيع والفهرسة والعمليات الحسابية والجبر الخطي والتخفيضات.
وهم سريعون!

### شبكات عصبية ديناميكية

تمتلك PyTorch طريقة فريدة لبناء الشبكات العصبية: استخدام مسجل شرائط وإعادة تشغيله.

معظم أطر العمل مثل TensorFlow و Theano و Caffe و CNTK لديها رؤية ثابتة للعالم.
يتعين على المرء بناء شبكة عصبية وإعادة استخدام نفس البنية مرارًا وتكرارًا.
يعني تغيير الطريقة التي تتصرف بها الشبكة أن على المرء أن يبدأ من نقطة الصفر.

باستخدام PyTorch ، نستخدم تقنية تسمى التمايز التلقائي للوضع العكسي ، والتي تتيح لك بذلك
تغيير الطريقة التي تتصرف بها شبكتك بشكل تعسفي بدون أي تأخير أو زيادة في النفقات. يأتي إلهامنا
من عدة أوراق بحثية حول هذا الموضوع ، بالإضافة إلى الأعمال الحالية والسابقة مثل
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), إلخ.

على الرغم من أن هذه التقنية ليست فريدة من نوعها في PyTorch ، إلا أنها واحدة من أسرع تطبيقاتها حتى الآن.
سوف تحصل على أفضل سرعة ومرونة لبحثك المجنون.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

### بايثون في المقام الأول

هذه المكتبة ليست ارتباطًا بلغة Python في إطار عمل C ++ أحادي.
تم تصميمه ليتم دمجه بعمق في Python.
يمكنك استخدامه بشكل طبيعي كما تستخدم [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) إلخ.
يمكنك كتابة طبقات الشبكة العصبية الجديدة في Python نفسها ، باستخدام مكتباتك المفضلة
واستخدام حزم مثل [Cython](https://cython.org/) و [Numba](http://numba.pydata.org/).
هدفنا هو عدم إعادة اختراع العجلة عند الاقتضاء.

### التجارب الحتمية

تم تصميم PyTorch ليكون بديهيًا وخطيًا في التفكير وسهل الاستخدام.
عند تنفيذ سطر من التعليمات البرمجية ، يتم تنفيذه. لا توجد رؤية غير متزامنة للعالم.
عندما تسقط في مصحح أخطاء أو تتلقى رسائل خطأ وتتبعات مكدسة ، يكون فهمها أمرًا سهلاً.
يشير stack trace إلى المكان الذي تم فيه تعريف التعليمات البرمجية الخاصة بك بالضبط.
نأمل ألا تقضي ساعات في تصحيح أخطاء التعليمات البرمجية الخاصة بك أبدًا بسبب آثار ال stack traces أو محركات التنفيذ غير المتزامنة وغير الشفافة.

### سريع وخفيف

مكتبتنا لديها الحد الأدنى من حمل الإطار. نقوم بدمج مكتبات التسريع
مثل [Intel MKL](https://software.intel.com/mkl) و [cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl) لتعظيم السرعة.
في جوهرها ، خلفية وحدة المعالجة المركزية ووحدة معالجة الرسومات (GPU) والشبكة العصبية 
ناضجة وتم اختبارها لسنوات.

وبالتالي ، فإن PyTorch سريع جدًا - سواء كنت تدير شبكات عصبية صغيرة أو كبيرة.
استخدام الذاكرة في PyTorch فعال للغاية مقارنة بـ Torch أو بعض البدائل.
لقد كتبنا مخصصات ذاكرة مخصصة لوحدة معالجة الرسومات للتأكد من ذلك 
تعد نماذج التعلم العميق الخاصة بك فعالة إلى أقصى حد في استخدام الذاكرة.
يمكّنك هذا من تدريب نماذج التعلم العميق بشكل أكبر من ذي قبل.

### سهولة في التوسع

تم تصميم كتابة وحدات شبكة عصبية جديدة ، أو التفاعل مع Tensor API من PyTorch ليكون مباشرًا
وبأقل قدر من التجريد.

يمكنك كتابة طبقات شبكة عصبية جديدة في Python باستخدام واجهة برمجة تطبيقات torch
[أو مكتبات NumPy المفضلة لديك مثل SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

إذا كنت ترغب في كتابة طبقاتك في C / C ++ ، فنحن نوفر واجهة برمجة تطبيقات تمديد ملائمة تتسم بالكفاءة وبأقل قدر ممكن من النماذج المعيارية. 
لا يلزم كتابة أي رمز مجمّع. يمكنك مشاهدة [برنامج تعليمي هنا](https://pytorch.org/tutorials/advanced/cpp_extension.html) و [مثال هنا](https://github.com/pytorch/extension-cpp).


## التثبيت

### الملفات التنفيذية
توجد أوامر لتثبيت الثنائيات عبر Conda أو عجلات الأنابيب على موقعنا الإلكتروني: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


#### منصات NVIDA Jeston


تتوفر عجلات Python لـ Jetson Nano من NVIDIA و Jetson TX1 / TX2 و Jetson Xavier NX / AGX و Jetson AGX Orin [هنا](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) وتم نشر حاوية L4T [هنا](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
إنها تتطلب JetPack 4.2 وما فوق ، ويقوم [@dusty-nv](https://github.com/dusty-nv) و [ptrblck](https://github.com/ptrblck) بالحفاظ عليها.

### من المصدر

#### المتطلبات الأساسية
إذا كنت تقوم بالتثبيت من المصدر ، فستحتاج إلى:
- نسخة Python 3.8 أو أحدث (لنظام التشغيل Linux ، يلزم استخدام Python 3.8.1+)
- مترجم متوافق مع C ++ 17 ، مثل clang

نوصي بشدة بتثبيت بيئة [Anaconda](https://www.anaconda.com/distribution/#download-section). ستحصل على مكتبة BLAS عالية الجودة (MKL) وستحصل على إصدارات تبعية خاضعة للرقابة بغض النظر عن توزيعات Linux الخاصة بك. 
إذا كنت تريد التحويل البرمجي باستخدام دعم CUDA ، فقم بتثبيت ما يلي (لاحظ أن CUDA غير مدعوم على macOS)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.0 وما فوق
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 وما فوق
- [Compiler](https://gist.github.com/ax3l/9489132) متوافق مع CUDA

ملاحظة: يمكنك الرجوع إلى [مصفوفة دعم cuDNN](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) لإصدارات cuDNN مع مختلف CUDA و CUDA driver و NVIDIA المعدات.

إذا كنت ترغب في تعطيل دعم CUDA ، فقم بتصدير متغير البيئة `USE_CUDA = 0`.
يمكن العثور على متغيرات البيئة المفيدة الأخرى في `setup.py`.

إذا كنت تقوم بالتصميم لمنصات Jetson من NVIDIA (Jetson Nano، TX1، TX2، AGX Xavier) ، فإن تعليمات تثبيت PyTorch لـ Jetson Nano [متوفرة هنا](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

إذا كنت تريد التحويل البرمجي باستخدام دعم ROCm ، فقم بتثبيت

- [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) 4.0 وما فوق 
- ROCm مدعوم حاليًا لأنظمة Linux فقط.

إذا كنت تريد تعطيل دعم ROCm ، فقم بتصدير متغير البيئة `USE_ROCM = 0`.
يمكن العثور على متغيرات البيئة المفيدة الأخرى `setup.py`.

#### تثبيت التبعيات أو الاعتماديات

**عام**

```bash
conda install cmake ninja
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below
pip install -r requirements.txt
```

**على Linux**

```bash
conda install mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

# (optional) If using torch.compile with inductor/triton, install the matching version of triton
# Run from the pytorch directory after cloning
make triton
```

**على MacOS**

```bash
# Add this package on intel x86 processor machines only
conda install mkl mkl-include
# Add these packages if torch.distributed is needed
conda install pkg-config libuv
```

**على Windows**

```bash
conda install mkl mkl-include
# Add these packages if torch.distributed is needed.
# Distributed package support on Windows is a prototype feature and is subject to changes.
conda install -c conda-forge libuv=1.39
```

#### الحصول على مصدر PyTorch
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

#### تثبيت PyTorch
**على Linux**

إذا كنت تقوم بالتجميع لـ AMD ROCm ، فقم أولاً بتشغيل هذا الأمر:
```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
```

قم بتثبيت PyTorch
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

> _Aside:_ If you are using [Anaconda](https://www.anaconda.com/distribution/#download-section), you may experience an error caused by the linker:
>
> ```plaintext
> build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized
> collect2: error: ld returned 1 exit status
> error: command 'g++' failed with exit status 1
> ```
>
> This is caused by `ld` from the Conda environment shadowing the system `ld`. You should use a newer version of Python that fixes this issue. The recommended Python version is 3.8.1+.

**على macOS**

```bash
python3 setup.py develop
```

**على Windows**

اختر إصدار Visual Studio الصحيح.

يقوم PyTorch CI باستخدام  Visual C++ BuildTools التي تأتي مع Visual Studio Enterprise طبعات احترافية أو مجتمعية. 
يمكنك أيضًا تثبيت أدوات البناء من https://visualstudio.microsoft.com/visual-cpp-build-tools/. أدوات البناء *لا*
تأتي مع Visual Studio Code افتراضيًا.

إذا كنت ترغب في إنشاء كود بيثون قديم ، يرجى الرجوع إلى [البناء على الكود القديم و CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**لبناء CPU فقط**
 
في هذا الوضع ، سيتم تشغيل حسابات PyTorch على وحدة المعالجة المركزية الخاصة بك ، وليس على وحدة معالجة الرسومات الخاصة بك

```cmd
conda activate
python setup.py develop
```
ملاحظة حول OpenMP: تطبيق OpenMP المطلوب هو Intel OpenMP (iomp). للارتباط بـ iomp ، ستحتاج إلى تنزيل المكتبة يدويًا وإعداد بيئة المبنى عن طريق تعديل `CMAKE_INCLUDE_PATH` و` LIB`. التعليمات [هنا](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) هي مثال على إعداد كل من MKL و Intel OpenMP. بدون هذه التكوينات لـ CMake ، سيتم استخدام وقت تشغيل Microsoft Visual C OpenMP (vcomp).

**بناء على أساس CUDA**

في هذا الوضع ، ستستفيد حسابات PyTorch من وحدة معالجة الرسومات الخاصة بك عبر CUDA لسحق الأرقام بشكل أسرع إن
[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) 
مطلوب لبناء Pytorch مع CUDA.
كما أن NVTX هو جزء من CUDA التوزيعي ، حيث يطلق عليه "Nsight Compute". لتثبيته على CUDA مثبت بالفعل ، قم بتشغيل تثبيت CUDA مرة أخرى وتحقق من مربع الاختيار المقابل.
تأكد من تثبيت CUDA مع Nsight Compute بعد Visual Studio.

حاليًا ، يتم دعم VS 2017/2019 و Ninja كمنشئ لـ CMake. إذا تم اكتشاف `ninja.exe` في `PATH` ، فسيتم استخدام Ninja كمولد افتراضي ، وإلا فسيستخدم VS 2017/2019.
ولكن <br/> إذا تم تحديد Ninja كمنشئ ، فسيتم تحديد أحدث MSVC باعتباره سلسلة الأدوات الأساسية.

مكتبات إضافية مثل [Magma](https://developer.nvidia.com/magma) و [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN) و [Sccache](https://github.com/mozilla/sccache) غالبًا ما تكون مطلوبة. يرجى الرجوع إلى [install-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) لتثبيتها.  

إن[Magma](https://developer.nvidia.com/magma) و [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN) و [Sccache](https://github.com/mozilla/sccache) غالبًا ما تكون مطلوبة. يرجى الرجوع إلى [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) لتثبيتها.

يمكنك الرجوع إلى البرنامج النصي [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) لبعض تكوينات متغيرات البيئة الأخرى

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


##### ضبط خيارات البناء اختياري

يمكنك ضبط تكوين متغيرات cmake اختياريًا (بدون البناء أولاً) ، عن طريق القيام بذلك
الأتى. على سبيل المثال ، يمكن ضبط الدلائل المكتشفة مسبقًا لـ CuDNN أو BLAS
بهذه الخطوة.

على Linux
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

على macOS
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only
ccmake build  # or cmake-gui build
```

### صورة دوكر Docker Image

#### استخدام صور مبنية مسبقًا

يمكنك أيضًا سحب صورة عامل إرساء مسبقة الصنع من Docker Hub وتشغيلها باستخدام docker v19.03 +

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

يرجى ملاحظة أن PyTorch تستخدم ذاكرة مشتركة لمشاركة البيانات بين العمليات ، لذلك إذا تم استخدام المعالجة المتعددة للشعلة (على سبيل المثال
بالنسبة إلى برامج تحميل البيانات متعددة مؤشرات الترابط) ، لا يكفي حجم مقطع الذاكرة المشتركة الافتراضي الذي تعمل به الحاوية ، وأنت
يجب زيادة حجم الذاكرة المشتركة إما باستخدام خيارات سطر الأوامر `--ipc = host` أو` --shm-size` إلى `nvidia-docker run`.

#### بناء الصورة بنفسك

**ملاحظة:** يجب أن تُبنى بإصدار عامل إرساء> 18.06

يتم توفير "Dockerfile" لإنشاء صور بدعم CUDA 11.1 و cuDNN v8.
يمكنك تمرير `PYTHON_VERSION = x.y` لعمل متغير لتحديد إصدار Python الذي سيتم استخدامه بواسطة Miniconda ، أو اتركه
غير مضبوط لاستخدام الافتراضي.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

You can also pass the `CMAKE_VARS="..."` environment variable to specify additional CMake variables to be passed to CMake during the build.
See [setup.py](./setup.py) for the list of available variables.

```bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker.Makefile
```

### إنشاء الوثائق

لإنشاء وثائق بتنسيقات مختلفة ، ستحتاج إلى [Sphinx](http://www.sphinx-doc.org) و
موضوع readthedocs.

```bash
cd docs/
pip install -r requirements.txt
```
يمكنك بعد ذلك إنشاء التوثيق عن طريق تشغيل `make <format>` من ملف
مجلد مستندات / `. قم بتشغيل `make` للحصول على قائمة بجميع تنسيقات الإخراج المتاحة.
إذا حصلت على خطأ katex ، قم بتشغيل `npm install katex`. إذا استمرت ، حاول
تثبيت npm -g katex`


ملاحظة: إذا قمت بتثبيت `nodejs` مع مدير حزم مختلف (على سبيل المثال conda)
ثم من المحتمل أن يقوم `npm` بتثبيت إصدار من `katex` ليس كذلك
متوافق مع إصدارك من `nodejs` وستفشل إصدارات doc.
مجموعة من الإصدارات المعروفة للعمل هي `node @ 6.13.1` و
كاتكس @ 0.13.18`. لتثبيت الأخير مع `npm` يمكنك تشغيل
```npm install -g katex@0.13.18```

### الإصدارات السابقة

 
يمكن العثور على تعليمات التثبيت والثنائيات لإصدارات PyTorch السابقة
على [موقعنا](https://pytorch.org/previous-versions).

## البدء بالاستخدام

ثلاث مؤشرات لتبدأ بها:
- [دروس: ابدأ بفهم واستخدام PyTorch](https://pytorch.org/tutorials/)
- [أمثلة: سهولة فهم كود PyTorch عبر جميع المجالات](https://github.com/pytorch/examples)
- [مرجع واجهة برمجة التطبيقات](https://pytorch.org/docs/)
- [المسرد](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)
 
## موارد

* [PyTorch.org](https://pytorch.org/)
* [دروس PyTorch](https://pytorch.org/tutorials/)
* [أمثلة PyTorch](https://github.com/pytorch/examples)
* [نماذج PyTorch](https://pytorch.org/hub/)
* [مقدمة عن التعلم العميق باستخدام PyTorch من Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [مقدمة عن التعلم الآلي باستخدام PyTorch من Udacity](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [الشبكات العصبية العميقة مع PyTorch من Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [مدونة PyTorch](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## التواصل

* المنتديات: ناقش عمليات التنفيذ والبحث وما إلى ذلك https://discuss.pytorch.org
* مشكلات GitHub: تقارير الأخطاء وطلبات الميزات ومشكلات التثبيت و RFCs والأفكار وما إلى ذلك.
* اما Slack: يستضيف [PyTorch Slack](https://pytorch.slack.com/) جمهورًا أساسيًا من مستخدمي ومطوري PyTorch المعتدلين وذوي الخبرة للدردشة العامة والمناقشات عبر الإنترنت والتعاون وما إلى ذلك. إذا كنت مبتدئًا تبحث عن help ، الوسيط الأساسي هو [منتديات PyTorch](https://discuss.pytorch.org). إذا كنت بحاجة إلى دعوة Slack ، فيرجى ملء هذا النموذج: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* النشرة الإخبارية: No-No-No-No-No-No-No-No-ضجيج ، وهي رسالة إخبارية أحادية الاتجاه تحتوي على إعلانات مهمة حول PyTorch. يمكنك التسجيل هنا: https://eepurl.com/cbG0rv
* صفحة الفيسبوك: إعلانات مهمة حول PyTorch. https://www.facebook.com/pytorch
* للحصول على إرشادات العلامة التجارية ، يرجى زيارة موقعنا على الويب على [pytorch.org](https://pytorch.org/) 

## الإصدارات والمساهمة

عادةً ما تصدر PyTorch ثلاثة إصدارات رئيسية سنويًا. يُرجى إخبارنا إذا واجهتك خطأ عن طريق [تقديم مشكلة](https://github.com/pytorch/pytorch/issues).

نحن نقدر جميع المساهمات. إذا كنت تخطط للمساهمة في إصلاحات الأخطاء ، فالرجاء القيام بذلك دون أي مناقشة أخرى.

إذا كنت تخطط للمساهمة بميزات جديدة أو وظائف أدوات مساعدة أو ملحقات للجوهر ، فيرجى أولاً فتح مشكلة ومناقشة الميزة معنا.
قد يؤدي إرسال العلاقات العامة دون مناقشة إلى رفض العلاقات العامة لأننا قد نأخذ جوهر في اتجاه مختلف عما قد تكون على دراية به.

لمعرفة المزيد حول تقديم مساهمة إلى Pytorch ، يرجى الاطلاع على [صفحة المساهمة](CONTRIBUTING.md). لمزيد من المعلومات حول إصدارات PyTorch ، راجع [صفحة الإصدار](ELEASE.md).

## الفريق

إن PyTorch هو مشروع يحركه المجتمع مع العديد من المهندسين والباحثين الماهرين الذين يساهمون فيه.

تتم صيانة PyTorch حاليًا بواسطة [Soumith Chintala](http://soumith.ch) و [Gregory Chanan](https://github.com/gchanan) و [Dmytro Dzhulgakov](https://github.com/dzhulgakov) و [إدوارد يانغ](https://github.com/ezyang) و [نيكيتا شولجا](https://github.com/malfet) مع مساهمات كبيرة من مئات الأفراد الموهوبين بأشكال ووسائل مختلفة.
قائمة غير شاملة ولكن متنامية تحتاج إلى ذكر: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

ملاحظة: هذا المشروع ليس له علاقة بـ [hughperkins/pytorch](https://github.com/hughperkins/pytorch) بنفس الاسم. هو مساهم قيم في مجتمع Torch وساعد في العديد من الأشياء Torch و PyTorch.

## الترخيص

تمتلك PyTorch ترخيصًا على طراز BSD ، كما هو موجود في ملف [LICENSE](LICENSE).
