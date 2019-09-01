Pod::Spec.new do |s|
    s.name             = 'Pytorch'
    s.version          = '0.0.1'
    s.authors          = 'Facebook'
    s.license          = { :type => 'BSD' }
    s.homepage         = 'https://github.com/pytorch/pytorch'
    s.source           = { :http => 'http://ossci-macos.s3.amazonaws.com/libtorch_x86_arm64.zip' }
    s.summary          = 'Pytorch for iOS'
    s.description      = <<-DESC
    An internal-only pod containing the Pytorch C++ library that the public `PytorchObjC` depends on. This pod is not
    intended to be used directly. Swift and Objective-C developers should use `PytorchObjC` pod.
    DESC

    s.default_subspec = 'Core'
    s.subspec 'Core' do |ss|
        ss.dependency 'Pytorch/Libtorch'
        ss.source_files = 'src/*.{h,cpp,cc}'
        ss.public_header_files = ['src/Pytorch.h']
    end
    
    s.subspec 'Libtorch' do |ss|
        ss.header_mappings_dir = 'install/include/'
        ss.preserve_paths = 'install/include/**/*.{h,cpp,cc,c}' 
        ss.vendored_libraries = 'install/lib/libtorch.a'
        ss.libraries = ['c++', 'stdc++']
    end
    s.user_target_xcconfig = {
        'OTHER_LDFLAGS' => '-force_load "$(PODS_ROOT)/Pytorch/install/lib/libtorch.a"',
        'CLANG_CXX_LANGUAGE_STANDARD' => 'c++11',
        'CLANG_CXX_LIBRARY' => 'libc++'
    }
    s.pod_target_xcconfig = { 
        'HEADER_SEARCH_PATHS' => '$(inherited) "$(PODS_ROOT)/Pytorch/install/include/"', 
        'VALID_ARCHS' => 'x86_64 arm64' }
    s.module_name='Pytorch'
    s.module_map = 'src/framework.modulemap'
    s.library = ['c++', 'stdc++']
end
