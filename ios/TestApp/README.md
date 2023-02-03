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
