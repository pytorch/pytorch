Pod::Spec.new do |s|
    s.name             = 'LibTorch'
    s.version          = '1.13.0'
    s.authors          = 'PyTorch Team'
    s.license          = { :type => 'BSD' }
    s.homepage         = 'https://github.com/pytorch/pytorch'
    s.source           = { :http => "https://ossci-ios.s3.amazonaws.com/libtorch_ios_#{s.version}.zip" }
    s.summary          = 'The PyTorch C++ library for iOS'
    s.description      = <<-DESC
        The PyTorch C++ library for iOS.
    DESC
    s.ios.deployment_target = '12.0'
    s.default_subspec = 'Core'
    s.subspec 'Core' do |ss|
        ss.dependency 'LibTorch/Torch'
        ss.source_files = 'src/*.{h,cpp,c,cc}'
        ss.public_header_files = ['src/LibTorch.h']
    end
    s.subspec 'Torch' do |ss|
        ss.header_mappings_dir = 'install/include/'
        ss.preserve_paths = 'install/include/**/*.{h,cpp,cc,c}'
        ss.vendored_libraries = 'install/lib/*.a'
        ss.libraries = ['c++', 'stdc++']
    end
    s.user_target_xcconfig = {
        'HEADER_SEARCH_PATHS' => '$(inherited) "$(PODS_ROOT)/LibTorch/install/include/"',
        'OTHER_LDFLAGS' => '-force_load "$(PODS_ROOT)/LibTorch/install/lib/libtorch.a" -force_load "$(PODS_ROOT)/LibTorch/install/lib/libtorch_cpu.a"',
        'CLANG_CXX_LANGUAGE_STANDARD' => 'c++14',
        'CLANG_CXX_LIBRARY' => 'libc++'
    }
    s.pod_target_xcconfig = {
        'HEADER_SEARCH_PATHS' => '$(inherited) "$(PODS_ROOT)/LibTorch/install/include/"',
        'VALID_ARCHS' => 'x86_64 arm64'
    }
    s.library = ['c++', 'stdc++']
    s.frameworks = 'Accelerate'
end
