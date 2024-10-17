require 'xcodeproj'
require 'fileutils'
require 'optparse'

options = {}
option_parser = OptionParser.new do |opts|
 opts.banner = 'Script for setting up TestApp.xcodeproj'
 opts.on('-t', '--team_id ', 'development team ID') { |value|
    options[:team_id] = value
 }
 opts.on('-l', '--lite ', 'use lite interpreter') { |value|
    options[:lite] = value
 }
 opts.on('-b', '--benchmark', 'build app to run benchmark') { |value|
    options[:benchmark] = value
 }
end.parse!
puts options.inspect

puts "Current directory: #{Dir.pwd}"
install_path = File.expand_path("../../../build_ios/install")
if not Dir.exist? (install_path)
    raise "path doesn't exist:#{install_path}!"
end
xcodeproj_path = File.expand_path("../TestApp.xcodeproj")
if not File.exist? (xcodeproj_path)
    raise "path doesn't exist:#{xcodeproj_path}!"
end
puts "Setting up TestApp.xcodeproj..."
project = Xcodeproj::Project.open(xcodeproj_path)
targets = project.targets
test_target = targets.last
header_search_path      = ['$(inherited)', "#{install_path}/include"]
libraries_search_path   = ['$(inherited)', "#{install_path}/lib"]
other_linker_flags      = ['$(inherited)', "-all_load"]
# TestApp and TestAppTests
targets.each do |target|
    target.build_configurations.each do |config|
        config.build_settings['HEADER_SEARCH_PATHS']    = header_search_path
        config.build_settings['LIBRARY_SEARCH_PATHS']   = libraries_search_path
        config.build_settings['OTHER_LDFLAGS']          = other_linker_flags
        config.build_settings['ENABLE_BITCODE']         = 'No'
        if (options[:lite])
            config.build_settings['GCC_PREPROCESSOR_DEFINITIONS'] = ['$(inherited)', "BUILD_LITE_INTERPRETER"]
        else
            config.build_settings['GCC_PREPROCESSOR_DEFINITIONS'] = ['$(inherited)']
        end
        if (options[:benchmark])
            config.build_settings['GCC_PREPROCESSOR_DEFINITIONS'].append("RUN_BENCHMARK")
        end
        dev_team_id = options[:team_id]
        if dev_team_id
            config.build_settings['DEVELOPMENT_TEAM'] = dev_team_id
        end
    end
end
group = project.main_group.find_subpath(File.join('TestApp'),true)
group.set_source_tree('SOURCE_ROOT')
group.files.each do |file|
    if (file.name.to_s.end_with?(".pt") ||
        file.name.to_s.end_with?(".ptl") ||
        file.name == "config.json")
        group.remove_reference(file)
        targets.each do |target|
            target.resources_build_phase.remove_file_reference(file)
        end
    end
end

config_path = File.expand_path("./config.json")
if not File.exist?(config_path)
    raise "config.json can't be found!"
end
config_file_ref = group.new_reference(config_path)

file_refs = []
# collect models
puts "Installing models..."
models_dir = File.expand_path("../models")
Dir.foreach(models_dir) do |model|
    if(model.end_with?(".pt") || model.end_with?(".ptl"))
      model_path = models_dir + "/" + model
      file_refs.push(group.new_reference(model_path))
    end
end

targets.each do |target|
    target.resources_build_phase.add_file_reference(config_file_ref, true)
    file_refs.each do |ref|
        target.resources_build_phase.add_file_reference(ref, true)
    end
end

# add test files
puts "Adding test files..."
testTarget = targets[1]
testFilePath = File.expand_path('../TestAppTests/')
group = project.main_group.find_subpath(File.join('TestAppTests'),true)
group.files.each do |file|
    if (file.path.end_with?(".mm"))
        file.remove_from_project
    end
end

if(options[:lite])
    file = group.new_file("TestLiteInterpreter.mm")
    testTarget.add_file_references([file])
else
    file = group.new_file("TestFullJIT.mm")
    testTarget.add_file_references([file])
end

puts "Linking static libraries..."
libs = ['libc10.a', 'libclog.a', 'libpthreadpool.a', 'libXNNPACK.a', 'libeigen_blas.a', 'libcpuinfo.a', 'libpytorch_qnnpack.a', 'libtorch_cpu.a', 'libtorch.a']
frameworks = ['CoreML', 'Metal', 'MetalPerformanceShaders', 'Accelerate', 'UIKit']
targets.each do |target|
    # NB: All these libraries and frameworks have already been linked by TestApp, adding them
    # again onto the test target will cause the app to crash on actual devices
    if (target == test_target)
        next
    end
    target.frameworks_build_phases.clear
    for lib in libs do
        path = "#{install_path}/lib/#{lib}"
        if File.exist?(path)
            libref = project.frameworks_group.new_file(path)
            target.frameworks_build_phases.add_file_reference(libref)
        end
    end
     # link system frameworks
    if frameworks
        frameworks.each do |framework|
            path = "System/Library/Frameworks/#{framework}.framework"
            framework_ref = project.frameworks_group.new_reference(path)
            framework_ref.name = "#{framework}.framework"
            framework_ref.source_tree = 'SDKROOT'
            target.frameworks_build_phases.add_file_reference(framework_ref)
        end
    end

end

project.save
puts "Done."
