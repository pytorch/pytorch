## TestApp

The TestApp is currently being used as a dummy app by Circle CI for nightly jobs. The challenge comes when testing the arm64 build as we don't have a way to code-sign our TestApp. This is where Fastlane came to rescue. [Fastlane](https://fastlane.tools/) is a trendy automation tool for building and managing iOS applications. It also works seamlessly with Circle CI. We are going to leverage the `import_certificate` action, which can install developer certificates on CI machines. See `Fastfile` for more details.

For simulator build, we run unit tests as the last step of our CI workflow. Those unit tests can also be run manually via the `fastlane scan` command.

## Run Simulator Test Locally
Follow these steps if you want to run the test locally.

1. Checkout PyTorch repo including all submodules

2. Build PyTorch for ios
```
USE_COREML_DELEGATE=1 IOS_PLATFORM=SIMULATOR ./scripts/build_ios.sh
```

3. Generate on-the-fly test models
```
python test/mobile/model_test/gen_test_model.py ios-test
```
You need to install regular PyTorch on your local machine to run this script.
Check https://github.com/pytorch/pytorch/tree/master/test/mobile/model_test#diagnose-failed-test to learn more.

4. Create XCode project (for lite interpreter)
```
cd ios/TestApp/benchmark
ruby setup.rb --lite 1
```

5. Open the generated TestApp/TestApp.xcodeproj in XCode and run simulator test.

## Re-generate All Test Models
1. Make sure PyTorch (not PyTorch for iOS) is installed
See https://pytorch.org/get-started/locally/

2. Re-generate models for operator test
```
python test/mobile/model_test/gen_test_model.py ios
python test/mobile/model_test/gen_test_model.py ios-test
```

3. Re-generate Core ML model
```
cd ios/TestApp/benchmark; python coreml_backend.py
```

## Run test on AWS Device Farm
The test app and its test suite could also be run on actual devices via
AWS Device Farm.

1. The following steps could only be done on MacOS with Xcode installed.
   I'm using Xcode 15.0 on MacOS M1 arm64

2. Checkout PyTorch repo including all submodules

3. Build PyTorch for iOS devices, not for simulator
```
export BUILD_LITE_INTERPRETER=1
export USE_PYTORCH_METAL=1
export USE_COREML_DELEGATE=1
export IOS_PLATFORM=OS
export IOS_ARCH=arm64

./scripts/build_ios.sh
```

4. Build the test app locally
```
# Use the pytorch nightly build to generate models
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Generate models for differnet backends
pushd ios/TestApp/benchmark
mkdir -p ../models

# This requires numpy==1.23.1
python coreml_backend.py

# NB: Also need to set the team ID with -t if you are running this locally. This
# command setups an app that could be used to launch TestAppTests on device. On
# the other hand, adding the --benchmark flag to build the one that runs benchmark
# instead.
ruby setup.rb --lite 1
popd

# Build the TestApp and its TestAppTests
ruby scripts/xcode_build.rb -i build_ios/install -x ios/TestApp/TestApp.xcodeproj -p "OS"
```

5. Prepare the artifacts
https://docs.aws.amazon.com/devicefarm/latest/developerguide/test-types-ios-xctest.html

```
export DEST_DIR="Payload"

pushd ios/TestApp/build/Release-iphoneos
mkdir "${DEST_DIR}"

cp -r TestApp.app "${DEST_DIR}"
# TestApp.ipa is just a zip file with a payload subdirectory
zip -vr TestApp.ipa "${DEST_DIR}"

pushd TestApp.app/PlugIns
# Also zip the TestAppTests.xctest test suite
zip -vr TestAppTests.xctest.zip TestAppTests.xctest
popd

cp TestApp.app/PlugIns/TestAppTests.xctest.zip .
popd
```

6. Upload the artifacts to AWS Device Farm and run the tests
```
export PYTORCH_ARN="arn:aws:devicefarm:us-west-2:308535385114:project:b531574a-fb82-40ae-b687-8f0b81341ae0"

pushd ios/TestApp
# AWS Device Farm is only available on us-west-2
AWS_DEFAULT_REGION=us-west-2 python run_on_aws_devicefarm.py \
  --project-arn "${PYTORCH_ARN}" \
  --app-file build/Release-iphoneos/TestApp.ipa \
  --xctest-file build/Release-iphoneos/TestAppTests.xctest.zip \
  --name-prefix PyTorch
popd
```

7. The script will continue polling for the outcome. A visual output of
   the test results could be view on AWS Device Farm console for [PyTorch project](https://us-west-2.console.aws.amazon.com/devicefarm/home#/mobile/projects/b531574a-fb82-40ae-b687-8f0b81341ae0/runs)

## Debug Test Failures
Make sure all models are generated. See https://github.com/pytorch/pytorch/tree/master/test/mobile/model_test to learn more.

There's no debug information in simulator test (project TestAppTests). You can copy the failed test code to
TestApp/TestApp/ViewController.mm and debug in the main TestApp.

### Benchmark

The benchmark folder contains two scripts that help you setup the benchmark project. The `setup.rb` does the heavy-lifting jobs of setting up the XCode project, whereas the `trace_model.py` is a Python script that you can tweak to generate your model for benchmarking. Simply follow the steps below to setup the project

1. In the PyTorch root directory, run `IOS_ARCH=arm64 ./scripts/build_ios.sh` to generate the custom build from **Master** branch
2. Navigate to the `benchmark` folder, run `python trace_model.py` to generate your model.
3. In the same directory, open `config.json`. Those are the input parameters you can tweak.
4. Again, in the same directory, run `ruby setup.rb` to setup the XCode project.
5. Open the `TestApp.xcodeproj`, you're ready to go.

The benchmark code is written in C++, you can use `UI_LOG` to visualize the log. See `benchmark.mm` for more details.
