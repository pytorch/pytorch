![PyTorch Logo](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

--------------------------------------------------------------------------------
#### _Read this in other languages:_

<kbd>[<img title="English" alt="English" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/us.svg" width="30">](../README.md)</kbd>
<kbd>[<img title="Português" alt="Português" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/br.svg" width="30">](README_pt.md)</kbd>
<kbd>[<img title="عربى" alt="عربى" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/sa.svg" width="30">](README_ar.md)</kbd>
<kbd>[<img title="Türkçe" alt="Türkçe" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/tr.svg" width="30">](README_tr.md)</kbd>
<kbd>[<img title="Deutsch" alt="Deutsch" src="https://cdn.staticaly.com/gh/hjnilsson/country-flags/master/svg/de.svg" width="30">](README_de.md)</kbd>

PyTorch ist ein Python-Paket, das zwei Funktionen auf hohem Niveau bietet:
- Tensor-Berechnungen (wie NumPy) mit starker GPU-Beschleunigung
- Tiefe neuronale Netze, die auf einem bandbasierten Autograd-System aufgebaut sind

Sie können Ihre bevorzugten Python-Pakete wie NumPy, SciPy und Cython wiederverwenden, um PyTorch bei Bedarf zu erweitern.

Unseren Stammzustand (Signale für die kontinuierliche Integration) finden Sie unter [hud.pytorch.org](https://hud.pytorch.org/ci/pytorch/pytorch/main).

<!-- toc -->

- [Mehr über PyTorch](#mehr-über-pytorch)
  - [Eine GPU-fähige Tensor-Bibliothek](#eine-gpu-fähige-tensor-bibliothek)
  - [Dynamische neuronale Netze: Bandgestütztes Autograd](#dynamische-neuronale-netze-bandgestütztes-autograd)
  - [Python zuerst](#python-zuerst)
  - [Zwingende Erlebnisse](#zwingende-erlebnisse)
  - [Fast and Lean](#fast-and-lean)
  - [Extensions Without Pain](#extensions-without-pain)
- [Installation](#installation)
  - [Binaries](#binaries)
    - [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
  - [From Source](#from-source)
    - [Prerequisites](#prerequisites)
    - [Install Dependencies](#install-dependencies)
    - [Get the PyTorch Source](#get-the-pytorch-source)
    - [Install PyTorch](#install-pytorch)
      - [Adjust Build Options (Optional)](#adjust-build-options-optional)
  - [Docker Image](#docker-image)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image yourself](#building-the-image-yourself)
  - [Building the Documentation](#building-the-documentation)
  - [Previous Versions](#previous-versions)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Communication](#communication)
- [Releases and Contributing](#releases-and-contributing)
- [The Team](#the-team)
- [License](#license)

<!-- tocstop -->

## Mehr über PyTorch

Grundsätzlich besteht PyTorch aus den folgenden Komponenten, wodurch es sich um eine Bibliothek handelt:

| Komponent | Beschreibung |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | Eine Tensor-Bibliothek wie NumPy, mit starker GPU-Unterstützung |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Eine bandbasierte automatische Differenzierungsbibliothek, die alle differenzierbaren Tensor-Operationen in Torch unterstützt |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | Ein Kompilierungsstack (TorchScript) zur Erstellung von serialisierbaren und optimierbaren Modellen aus PyTorch-Code  |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | Eine tief in autograd integrierte Bibliothek für neuronale Netze, die für maximale Flexibilität ausgelegt ist |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python-Multiprocessing, aber mit magischer Speicherfreigabe von Torch-Tensoren über Prozesse hinweg. Nützlich für das Laden von Daten und Hogwild-Training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader und andere nützliche Funktionen für mehr Komfort |

Normalerweise wird PyTorch entweder als:

- Ein Ersatz für NumPy, um die Leistung von GPUs zu nutzen.
- Eine Deep-Learning-Forschungsplattform, die maximale Flexibilität und Geschwindigkeit bietet.

Weiter ausarbeiten:

### Eine GPU-fähige Tensor-Bibliothek

Wenn Sie NumPy verwenden, haben Sie wahrscheinlich Tensoren (auch bekannt als ndarray) verwendet.

![Tensor illustration](../docs/source/_static/img/tensor_illustration.png)

PyTorch bietet Tensoren, die entweder auf der CPU oder der GPU laufen können und die
Berechnungen um ein Vielfaches.

Wir bieten eine Vielzahl von Tensor-Routinen zur Beschleunigung und Anpassung an Ihre wissenschaftlichen Berechnungsanforderungen
wie Slicing, Indexierung, mathematische Operationen, lineare Algebra, Reduktionen.
Und sie sind schnell!

### Dynamische neuronale Netze: Bandgestütztes Autograd

PyTorch verfügt über eine einzigartige Methode zum Aufbau neuronaler Netze: die Verwendung und Wiedergabe eines Tonbandgeräts.

Die meisten Frameworks wie TensorFlow, Theano, Caffe und CNTK haben eine statische Sicht auf die Welt.
Man muss ein neuronales Netz aufbauen und die gleiche Struktur immer wieder verwenden.
Wenn man die Art und Weise, wie sich das Netzwerk verhält, ändert, muss man bei Null anfangen.

Mit PyTorch verwenden wir eine Technik namens Reverse-Mode-Autodifferenzierung, die es Ihnen ermöglicht
die Art und Weise, wie sich Ihr Netzwerk verhält, beliebig und ohne Verzögerung oder Overhead zu ändern. Unsere Inspiration stammt
von mehreren Forschungsarbeiten zu diesem Thema, sowie von aktuellen und früheren Arbeiten wie
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), usw.

Diese Technik ist zwar nicht einzigartig für PyTorch, aber es ist eine der schnellsten Implementierungen, die es bisher gab.
Sie erhalten das Beste aus Geschwindigkeit und Flexibilität für Ihre verrückte Forschung.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/dynamic_graph.gif)

### Python zuerst

PyTorch ist keine Python-Einbindung in ein monolithisches C++ -Framework.
Es ist so aufgebaut, dass es tief in Python integriert werden kann.
Sie können es natürlich verwenden, wie Sie es auch bei [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) usw.
Sie können Ihre neuen neuronalen Netzschichten in Python selbst schreiben, indem Sie Ihre bevorzugten Bibliotheken verwenden
und verwenden Pakete wie [Cython](https://cython.org/) und [Numba](http://numba.pydata.org/).
Unser Ziel ist es, das Rad gegebenenfalls nicht neu zu erfinden.

### Zwingende Erlebnisse

PyTorch wurde mit dem Ziel entwickelt, intuitiv, geradlinig im Denken und einfach zu bedienen zu sein.
Wenn Sie eine Code-Zeile ausführen, wird sie ausgeführt. Es gibt keine asynchrone Sicht auf die Welt.
Wenn Sie einen Debugger verwenden oder Fehlermeldungen und Stack Traces erhalten, ist es einfach, diese zu verstehen.
Der Stack-Trace zeigt genau an, wo Ihr Code definiert wurde.
Wir hoffen, dass Sie nie Stunden mit der Fehlersuche in Ihrem Code verbringen, weil die Stack Traces schlecht oder die Ausführungsengines asynchron und undurchsichtig sind.

### Schnell und schlank

PyTorch hat einen minimalen Framework-Overhead. Wir integrieren Beschleunigungsbibliotheken
wie zum Beispiel [Intel MKL](https://software.intel.com/mkl) und NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl)) um die Geschwindigkeit zu maximieren.
Im Kern sind die CPU- und GPU-Tensor- und neuronalen Netzwerk-Backends
sind ausgereift und werden seit Jahren getestet.

Daher ist PyTorch recht schnell - egal, ob Sie kleine oder große neuronale Netze betreiben.

Die Speichernutzung in PyTorch ist im Vergleich zu Torch oder einigen der Alternativen extrem effizient.
Wir haben benutzerdefinierte Speicherallokatoren für die GPU geschrieben, um sicherzustellen, dass
Ihre Deep-Learning-Modelle maximal speichereffizient sind.
Dadurch können Sie größere Deep-Learning-Modelle als bisher trainieren.

### Verlängerungen ohne Schmerzen

Das Schreiben neuer Module für neuronale Netze oder das Anbinden an die Tensor-API von PyTorch wurde so konzipiert, dass es einfach ist
und mit minimalen Abstraktionen.

Mit der Torch-API können Sie neue Schichten für neuronale Netze in Python schreiben
[oder Ihre bevorzugten NumPy-basierten Bibliotheken wie SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

Wenn Sie Ihre Schichten in C/C++ schreiben möchten, bieten wir Ihnen eine bequeme Erweiterungs-API, die effizient und mit minimaler Boilerplate ist.
Es muss kein Wrapper-Code geschrieben werden. Sie können sehen [ein Tutorial hier](https://pytorch.org/tutorials/advanced/cpp_extension.html) und [ein Beispiel hier](https://github.com/pytorch/extension-cpp).


## Einrichtung

### Binaries
Befehle zur Installation von Binaries über Conda oder pip wheels finden Sie auf unserer Website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


#### NVIDIA Jetson Plattformen

Python-Räder für NVIDIAs Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX und Jetson AGX Orin werden bereitgestellt [hier](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) und der L4T-Container wird veröffentlicht [hier](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

Sie benötigen JetPack 4.2 und höher, und [@dusty-nv](https://github.com/dusty-nv) und [@ptrblck](https://github.com/ptrblck) erhalten sie.


### Von der Quelle

#### Voraussetzungen
Wenn Sie vom Quellcode installieren, benötigen Sie:
- Python 3.8 oder höher (für Linux wird Python 3.8.1+ benötigt)
- Einen C++17-kompatiblen Compiler, z. B. clang

Wir empfehlen dringend die Installation eines [Anaconda](https://www.anaconda.com/distribution/#download-section) Umgebung. Sie erhalten eine hochwertige BLAS-Bibliothek (MKL) und Sie erhalten kontrollierte Abhängigkeitsversionen unabhängig von Ihrer Linux-Distribution.

Wenn Sie mit CUDA-Unterstützung kompilieren möchten, installieren Sie Folgendes (beachten Sie, dass CUDA unter macOS nicht unterstützt wird)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11,0 oder 
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 oder 
- [Compiler](https://gist.github.com/ax3l/9489132) kompatibel mit CUDA

Hinweis: Sie können sich auf die [cuDNN-Unterstützungsmatrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) für cuDNN Versionen mit den verschiedenen unterstützten CUDA, CUDA Treibern und NVIDIA Hardware

Wenn Sie die CUDA-Unterstützung deaktivieren wollen, exportieren Sie die Umgebungsvariable `USE_CUDA=0`.
Andere potentiell nützliche Umgebungsvariablen können in `setup.py` gefunden werden.

Wenn Sie für NVIDIAs Jetson-Plattformen (Jetson Nano, TX1, TX2, AGX Xavier) bauen, lautet die Anleitung zur Installation von PyTorch für Jetson Nano [hier verfügbar](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

Wenn Sie mit ROCm-Unterstützung kompilieren wollen, installieren Sie
- [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) 4.0 und obige Installation
- ROCm wird derzeit nur für Linux-Systeme unterstützt.

Wenn Sie die ROCm-Unterstützung deaktivieren wollen, exportieren Sie die Umgebungsvariable `USE_ROCM=0`.
Andere potentiell nützliche Umgebungsvariablen können in `setup.py` gefunden werden.

#### Abhängigkeiten installieren

**Allgemeine**

```bash
conda install cmake ninja
#Führen Sie diesen Befehl aus dem PyTorch-Verzeichnis aus, nachdem Sie den Quellcode mit Hilfe des Abschnitts "Get the PyTorch Source" unten geklont haben
pip install -r requirements.txt
```

**Bei Linux**

```bash
conda install mkl mkl-include
# Nur CUDA: Fügen Sie bei Bedarf LAPACK-Unterstützung für die GPU hinzu
conda install -c pytorch magma-cuda110 # oder das magma-cuda*, das Ihrer CUDA-Version entspricht, von https://anaconda.org/pytorch/repo

# (optional) Wenn Sie torch.compile mit inductor/triton verwenden, installieren Sie die passende Version von triton.
# Nach dem Klonen aus dem pytorch-Verzeichnis ausführen
make triton
```

**Bei MacOS**

```bash
# Fügen Sie dieses Paket nur auf Maschinen mit intel x86-Prozessoren hinzu.
conda install mkl mkl-include
# Fügen Sie diese Pakete hinzu, wenn torch.distributed benötigt wird
conda install pkg-config libuv
```

**Bei Windows**

```bash
conda install mkl mkl-include
# Fügen Sie diese Pakete hinzu, wenn torch.distributed benötigt wird.
# Die Unterstützung verteilter Pakete unter Windows ist ein Prototyp und kann sich noch ändern.
conda install -c conda-forge libuv=1.39
```

#### Die PyTorch-Quelle abrufen
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# wenn Sie einen bestehenden Checkout aktualisieren
git submodule sync
git submodule update --init --rekursiv
```

#### PyTorch installieren
**Bei Linux**

Wenn Sie für AMD ROCm kompilieren, führen Sie zuerst diesen Befehl aus:
```bash
# Führen Sie dies nur aus, wenn Sie für ROCm kompilieren
python tools/amd_build/build_amd.py
```

PyTorch installieren
``bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py entwickeln
```

> _Aside:_ Wenn Sie [Anaconda](https://www.anaconda.com/distribution/#download-section) verwenden, kann ein Fehler auftreten, der durch den Linker verursacht wird:
>
> ``'Plaintext
> build/temp.linux-x86_64-3.7/torch/csrc/stub.o: Datei nicht erkannt: Dateiformat nicht erkannt
> collect2: Fehler: ld gab 1 Exit-Status zurück
> error: Befehl 'g++' schlug fehl mit Exit-Status 1
> ```
>
> Dies wird durch `ld` aus der Conda-Umgebung verursacht, das das System `ld` überschattet. Sie sollten eine neuere Version von Python verwenden, die dieses Problem behebt. Die empfohlene Python-Version ist 3.8.1+.

**Unter macOS**

``bash
python3 setup.py entwickeln
```

**Bei Windows**

Wählen Sie die korrekte Visual Studio-Version.

PyTorch CI verwendet Visual C++ BuildTools, die mit Visual Studio Enterprise,
Professional- oder Community-Editionen enthalten sind. Sie können die BuildTools auch installieren von
https://visualstudio.microsoft.com/visual-cpp-build-tools/. Die Build-Tools werden *nicht*
werden standardmäßig mit Visual Studio Code geliefert.

Wenn Sie Legacy-Python-Code bauen möchten, lesen Sie bitte [Bauen auf Legacy-Code und CUDA] (https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**Nur-CPU-Builds**

In diesem Modus werden die PyTorch-Berechnungen auf der CPU und nicht auf der GPU ausgeführt.

```cmd
conda activate
python setup.py develop
```

Anmerkung zu OpenMP: Die gewünschte OpenMP-Implementierung ist Intel OpenMP (iomp). Um gegen iomp zu linken, müssen Sie die Bibliothek manuell herunterladen und die Bauumgebung einrichten, indem Sie `CMAKE_INCLUDE_PATH` und `LIB` anpassen. Die Anleitung [hier] (https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) ist ein Beispiel für die Einrichtung von MKL und Intel OpenMP. Ohne diese Konfigurationen für CMake wird die Microsoft Visual C OpenMP-Laufzeit (vcomp) verwendet.

**CUDA-basierte Erstellung**

In diesem Modus nutzen PyTorch-Berechnungen Ihren Grafikprozessor über CUDA für schnelleres Number Crunching

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) wird benötigt, um Pytorch mit CUDA zu erstellen.
NVTX ist ein Teil der CUDA-Distribution, wo es "Nsight Compute" genannt wird. Um es auf einem bereits installierten CUDA zu installieren, führen Sie die CUDA-Installation noch einmal aus und aktivieren Sie die entsprechende Checkbox.
Stellen Sie sicher, dass CUDA mit Nsight Compute nach Visual Studio installiert wird.

Aktuell werden VS 2017 / 2019 und Ninja als Generator von CMake unterstützt. Falls `ninja.exe` im `PATH` gefunden wird, wird Ninja als Standardgenerator verwendet, ansonsten wird VS 2017 / 2019 verwendet.
<br/> Falls Ninja als Generator ausgewählt ist, wird die neueste MSVC als zugrunde liegende Toolchain ausgewählt.

Zusätzliche Bibliotheken wie
[Magma](https://developer.nvidia.com/magma), [oneDNN, auch bekannt als MKLDNN oder DNNL](https://github.com/oneapi-src/oneDNN), und [Sccache](https://github.com/mozilla/sccache) werden oft benötigt. Bitte beachten Sie die [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers), um sie zu installieren.

Sie können das Skript [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) für einige andere Umgebungsvariablen-Konfigurationen heranziehen


```cmd
cmd

:: Setzen Sie die Umgebungsvariablen, nachdem Sie das mkl-Paket heruntergeladen und entpackt haben,
:: sonst würde CMake eine Fehlermeldung ausgeben: `Konnte OpenMP nicht finden`.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Lesen Sie den Inhalt des vorherigen Abschnitts sorgfältig durch, bevor Sie fortfahren.
:: [Optional] Wenn Sie das von Ninja und Visual Studio verwendete Toolset mit CUDA überschreiben möchten, führen Sie bitte den folgenden Skriptblock aus.
:: "Visual Studio 2019 Developer Command Prompt" wird automatisch ausgeführt.
:: Stellen Sie sicher, dass Sie CMake >= 3.12 haben, bevor Sie dies tun, wenn Sie den Visual Studio Generator verwenden.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^] -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Optional] Wenn Sie den CUDA-Host-Compiler außer Kraft setzen möchten
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python setup.py develop

```

##### Build-Optionen anpassen (optional)

Sie können die Konfiguration der cmake-Variablen optional anpassen (ohne vorher zu bauen), indem Sie Folgendes tun
die folgenden Schritte. Zum Beispiel können Sie die vordefinierten Verzeichnisse für CuDNN oder BLAS anpassen
mit einem solchen Schritt erfolgen.

Unter Linux
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py build --cmake-only
ccmake build # oder cmake-gui build
```

Unter macOS
```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build --cmake-only
ccmake build # oder cmake-gui build
```

### Docker Image

#### Vorgefertigte Bilder verwenden

Sie können auch ein vorgefertigtes Docker-Image von Docker Hub beziehen und mit Docker v19.03+ ausführen

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Bitte beachten Sie, dass PyTorch Shared Memory verwendet, um Daten zwischen Prozessen auszutauschen, wenn also Torch Multiprocessing verwendet wird (z.B.
(z.B. für Multithreading-Datenlader) ist die Standardgröße des gemeinsamen Speichers, mit dem der Container läuft, nicht ausreichend und Sie
Sie sollten die Größe des gemeinsamen Speichers entweder mit den Kommandozeilenoptionen `--ipc=host` oder `--shm-size` für `nvidia-docker run` erhöhen.

#### Das Bild selbst erstellen

**Hinweis:** Muss mit einer Docker-Version > 18.06 erstellt werden.

Das `Dockerfile` wird mitgeliefert, um Images mit CUDA 11.1 Unterstützung und cuDNN v8 zu bauen.
Sie können die Make-Variable `PYTHON_VERSION=x.y` übergeben, um anzugeben, welche Python-Version von Miniconda verwendet werden soll, oder sie
nicht setzen, um die Standardversion zu verwenden.

```bash
make -f docker.Makefile
# Bilder werden als docker.io/${Ihr_docker_benutzername}/pytorch getaggt
```

Sie können auch die Umgebungsvariable `CMAKE_VARS="..."` übergeben, um zusätzliche CMake-Variablen zu spezifizieren, die während des Builds an CMake übergeben werden. Siehe [setup.py](../setup.py) für die Liste der verfügbaren Variablen.

```bash
CMAKE_VARS="BUILD_CAFFE2=ON BUILD_CAFFE2_OPS=ON" make -f docker.Makefile
```

### Erstellung der Dokumentation

Um Dokumentation in verschiedenen Formaten zu erstellen, benötigen Sie [Sphinx](http://www.sphinx-doc.org) und das
readthedocs-Thema.

```bash
cd docs/
pip install -r requirements.txt
```
Sie können dann die Dokumentation erstellen, indem Sie `make <format>` aus dem
Ordner `docs/` ausführen. Führen Sie `make` aus, um eine Liste aller verfügbaren Ausgabeformate zu erhalten.

Wenn Sie einen katex-Fehler erhalten, führen Sie `npm install katex` aus.  Wenn der Fehler weiterhin besteht, versuchen Sie
npm install -g katex".

> Anmerkung: Wenn Sie `nodejs` mit einem anderen Paketmanager installiert haben (z.B.,
`conda`), dann wird `npm` wahrscheinlich eine Version von `katex` installieren, die nicht
die nicht mit Ihrer Version von `nodejs` kompatibel ist, und Doc-Builds werden fehlschlagen.
Eine Kombination von Versionen, von der bekannt ist, dass sie funktioniert, ist `node@6.13.1` und
`katex@0.13.18`. Um die letztere mit `npm` zu installieren, können Sie folgendes ausführen
``npm install -g katex@0.13.18```

### Vorherige Versionen

Installationsanweisungen und Binärdateien für frühere PyTorch-Versionen finden Sie
auf [unserer Website](https://pytorch.org/previous-versions).


## Einstieg in das Thema

Drei Anhaltspunkte für den Einstieg:
- [Tutorien: für den Einstieg in das Verständnis und die Verwendung von PyTorch](https://pytorch.org/tutorials/)
- [Beispiele: einfach zu verstehender PyTorch-Code in allen Bereichen](https://github.com/pytorch/examples)
- [Die API-Referenz](https://pytorch.org/docs/)
- [Glossar](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Ressourcen

* [PyTorch.org](https://pytorch.org/)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Beispiele](https://github.com/pytorch/examples)
* [PyTorch Modelle](https://pytorch.org/hub/)
* [Einführung in Deep Learning mit PyTorch von Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
* [Einführung in maschinelles Lernen mit PyTorch von Udacity](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
* [Tiefe neuronale Netze mit PyTorch von Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
* [PyTorch Twitter](https://twitter.com/PyTorch)
* [PyTorch Blog](https://pytorch.org/blog/)
* [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Kommunikation
* Foren: Diskussionen über Implementierungen, Forschung usw. https://discuss.pytorch.org
* GitHub Issues: Fehlerberichte, Funktionsanfragen, Installationsprobleme, RFCs, Gedanken, usw.
* Slack: Der [PyTorch Slack] (https://pytorch.slack.com/) beherbergt ein primäres Publikum von moderaten bis erfahrenen PyTorch-Nutzern und -Entwicklern für allgemeinen Chat, Online-Diskussionen, Zusammenarbeit usw. Wenn Sie ein Anfänger sind und Hilfe suchen, ist das primäre Medium [PyTorch Foren](https://discuss.pytorch.org). Wenn Sie eine Slack-Einladung benötigen, füllen Sie bitte dieses Formular aus: https://goo.gl/forms/PP1AGvNHpSaJP8to1
* Newsletter: No-noise, ein einseitiger E-Mail-Newsletter mit wichtigen Ankündigungen zu PyTorch. Sie können sich hier anmelden: https://eepurl.com/cbG0rv
* Facebook-Seite: Wichtige Ankündigungen über PyTorch. https://www.facebook.com/pytorch
* Für Markenrichtlinien besuchen Sie bitte unsere Website unter [pytorch.org](https://pytorch.org/)

## Freisetzung und Beitrag

In der Regel gibt es drei Hauptversionen von PyTorch pro Jahr. Bitte lassen Sie uns wissen, wenn Sie auf einen Fehler stoßen, indem Sie [ein Problem einreichen] (https://github.com/pytorch/pytorch/issues).

Wir schätzen alle Beiträge. Wenn Sie vorhaben, Fehlerbehebungen beizusteuern, tun Sie dies bitte ohne weitere Diskussion.

Wenn Sie vorhaben, neue Features, Utility-Funktionen oder Erweiterungen zum Kern beizusteuern, öffnen Sie bitte zuerst ein Issue und diskutieren Sie das Feature mit uns.
Das Senden eines PR ohne Diskussion kann zu einer Ablehnung des PR führen, da wir den Kern in eine andere Richtung lenken könnten, als Ihnen vielleicht bewusst ist.

Um mehr darüber zu erfahren, wie Sie einen Beitrag zu Pytorch leisten können, lesen Sie bitte unsere [Beitragsseite](../CONTRIBUTING.md). Weitere Informationen über PyTorch-Releases finden Sie auf der [Release-Seite](../RELEASE.md).

## Das Team

PyTorch ist ein von der Gemeinschaft getragenes Projekt, zu dem mehrere erfahrene Ingenieure und Forscher beitragen.

PyTorch wird derzeit von [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang) und [Nikita Shulga](https://github.com/malfet) betreut, wobei Hunderte von talentierten Personen in unterschiedlicher Form und mit unterschiedlichen Mitteln wichtige Beiträge leisten.
Eine nicht erschöpfende, aber wachsende Liste muss erwähnt werden: Trevor Killeen, Sasank Chilamkurthy, Sergey Zagoruyko, Adam Lerer, Francisco Massa, Alykhan Tejani, Luca Antiga, Alban Desmaison, Andreas Koepf, James Bradbury, Zeming Lin, Yuandong Tian, Guillaume Lample, Marat Dukhan, Natalia Gimelshein, Christian Sarofeen, Martin Raison, Edward Yang, Zachary Devito.

Hinweis: Dieses Projekt hat nichts mit [hughperkins/pytorch](https://github.com/hughperkins/pytorch) mit demselben Namen zu tun. Hugh ist ein wertvoller Mitwirkender der Torch-Gemeinschaft und hat bei vielen Dingen mit Torch und PyTorch geholfen.

## Lizenz

PyTorch hat eine BSD-ähnliche Lizenz, wie sie in der Datei [LICENSE](../LICENSE) zu finden ist.
