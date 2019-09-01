Pod::Spec.new do |s|
    s.name             = 'PytorchObjC'
    s.version          = '0.0.1'
    s.authors          = 'Facebook'
    s.license          = { :type => 'BSD' }
    s.homepage         = 'https://github.com/pytorch/pytorch'
    s.source           = { :git => 'https://github.com/pytorch/pytorch.git', :branch => "master" }
    s.summary          = 'Pytorch for Objective-C and Swift'
    s.description      = <<-DESC
   Pytorch for Objective-C and Swift developers. 
                         DESC
  
    s.ios.deployment_target = '10.3'
    s.module_name = 'PytorchObjC'
    s.static_framework = true

    objc_dir = 'ios/objc/'
    s.public_header_files = objc_dir + 'apis/*.h'
    s.source_files = [ objc_dir+'apis/*.{h,m,mm}', objc_dir+'src/*.{h,m,mm}' ]
    s.module_map = objc_dir+'apis/framework.modulemap'
    s.dependency 'Pytorch'
    header = ""
    s.pod_target_xcconfig = { 
      'HEADER_SEARCH_PATHS' => 
      '"${PODS_ROOT}/Pytorch/install/include" ' + 
      '"${PODS_ROOT}/PytorchObjC/' + objc_dir + 'apis"',
      'VALID_ARCHS' => 'x86_64 arm64' 
    }
    s.library = 'c++', 'stdc++'
    s.user_target_xcconfig = {
      'HEADER_SEARCH_PATHS' => '"${PODS_ROOT}/Pytorch/install/include"'
    }
    s.test_spec 'Tests' do |ts| 
      ts.source_files = objc_dir + 'Tests/*.{h,mm,m}'
      ts.resources = [ objc_dir + 'Tests/models/*.pt']
      ts.pod_target_xcconfig = {
        'OTHER_LDFLAGS' => '-force_load "$(PODS_ROOT)/Pytorch/install/lib/libtorch.a"'
      }
    end
  end