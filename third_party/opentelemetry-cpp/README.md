This folder contains vendored copy of opentelemetry-cpp API folder.
PyTorch depends just on opentelemetry-cpp API.
We took a dependency on version v1.14.2 of the API.
To update to a newer version:
```
cd /tmp
git clone https://github.com/open-telemetry/opentelemetry-cpp.git
cd opentelemetry-cpp
git checkout <new opentelemetry-cpp tag>
cd <pytorch checkout dir>
cp -R /tmp/opentelemetry-cpp/api/ third_party/opentelemetry-cpp
git add third_party/opentelemetry-cpp
git commit
```
