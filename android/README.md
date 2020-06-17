# Android

## Demo applications and tutorials

Demo applications with code walk-through can be find in [this github repo](https://github.com/pytorch/android-demo-app).

## Publishing

##### Release
Release artifacts are published to jcenter:

```
repositories {
    jcenter()
}

dependencies {
    implementation 'org.pytorch:pytorch_android:1.5.0'
    implementation 'org.pytorch:pytorch_android_torchvision:1.5.0'
}
```

##### Nightly

Nightly(snapshots) builds are published every night from `master` branch to [nexus sonatype snapshots repository](https://oss.sonatype.org/#nexus-search;quick~pytorch_android)

To use them repository must be specified explicitly:
```
repositories {
    maven {
        url "https://oss.sonatype.org/content/repositories/snapshots"
    }
}

dependencies {
    ...
    implementation 'org.pytorch:pytorch_android:1.6.0-SNAPSHOT'
    implementation 'org.pytorch:pytorch_android_torchvision:1.6.0-SNAPSHOT'
    ...
}
```
The current nightly(snapshots) version is the value of `VERSION_NAME` in `gradle.properties` in current folder, at this moment it is `1.6.0-SNAPSHOT`.

## Building PyTorch Android from Source

In some cases you might want to use a local build of pytorch android, for example you may build custom libtorch binary with another set of operators or to make local changes.

For this you can use `./scripts/build_pytorch_android.sh` script.
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
sh ./scripts/build_pytorch_android.sh
```

The workflow contains several steps:

1\. Build libtorch for android for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64)

2\. Create symbolic links to the results of those builds:
`android/pytorch_android/src/main/jniLibs/${abi}` to the directory with output libraries
`android/pytorch_android/src/main/cpp/libtorch_include/${abi}` to the directory with headers. These directories are used to build `libpytorch.so` library that will be loaded on android device.

3\. And finally run `gradle` in `android/pytorch_android` directory with task `assembleRelease`

Script requires that Android SDK, Android NDK and gradle are installed.
They are specified as environment variables:

`ANDROID_HOME` - path to [Android SDK](https://developer.android.com/studio/command-line/sdkmanager.html)

`ANDROID_NDK` - path to [Android NDK](https://developer.android.com/studio/projects/install-ndk)

`GRADLE_HOME` - path to [gradle](https://gradle.org/releases/)


After successful build you should see the result as aar file:

```
$ find pytorch_android/build/ -type f -name *aar
pytorch_android/build/outputs/aar/pytorch_android.aar
pytorch_android_torchvision/build/outputs/aar/pytorch_android.aar
```

It can be used directly in android projects, as a gradle dependency:
```
allprojects {
    repositories {
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {
    implementation(name:'pytorch_android', ext:'aar')
    implementation(name:'pytorch_android_torchvision', ext:'aar')
    ...
    implementation 'com.android.support:appcompat-v7:28.0.0'
    implementation 'com.facebook.soloader:nativeloader:0.8.0'
    implementation 'com.facebook.fbjni:fbjni-java-only:0.0.3'
}
```
We also have to add all transitive dependencies of our aars.
As `pytorch_android` [depends](https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/build.gradle#L62-L63) on `'com.android.support:appcompat-v7:28.0.0'`, `'com.facebook.soloader:nativeloader:0.8.0'` and 'com.facebook.fbjni:fbjni-java-only:0.0.3', we need to add them.
(In case of using maven dependencies they are added automatically from `pom.xml`).

You can check out [test app example](https://github.com/pytorch/pytorch/blob/master/android/test_app/app/build.gradle) that uses aars directly.

## More Details

You can find more details about the PyTorch Android API in the [Javadoc](https://pytorch.org/docs/stable/packages.html).
