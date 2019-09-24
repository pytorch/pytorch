require 'optparse'
require 'xcodeproj'

options = {}
option_parser = OptionParser.new do |opts| 
 opts.banner = 'Tools for testing the PyTorch iOS x86 build on MacOS'
 opts.on('-i', '--install ', 'path to the cmake install folder') { |value|
    options[:install] = value
 }
 opts.on('-x', '--xcodeproj ', 'path to the XCode project file') { |value|
    options[:xcodeproj] = value
 }
end.parse!
puts options.inspect

install_path = File.expand_path(options[:install])
puts("Library install path: #{install_path}")
if not Dir.exist? (install_path) 
    puts "path don't exist:#{install_path}!"
    exit(false)
end
xcodeproj_path = File.expand_path(options[:xcodeproj])
puts("XCode project file path: #{xcodeproj_path}")
if not File.exist? (xcodeproj_path) 
    puts "path don't exist:#{xcodeproj_path}!"
    exit(false)
end

project = Xcodeproj::Project.open(xcodeproj_path)
target = project.targets.first
header_search_path      = ['$(inherited)', "#{install_path}/include"]
libraries_search_path   = ['$(inherited)', "#{install_path}/lib"]
other_linker_flags      = ['$(inherited)', "-all_load"]

target.build_configurations.each do |config|
    config.build_settings['HEADER_SEARCH_PATHS']    = header_search_path
    config.build_settings['LIBRARY_SEARCH_PATHS']   = libraries_search_path
    config.build_settings['OTHER_LINKER_FLAGS']     = other_linker_flags
end

# link static libraries
target.frameworks_build_phases.clear
libs = ['libc10.a', 'libclog.a', 'libeigen_blas.a', 'libcpuinfo.a', 'libpytorch_qnnpack.a', 'libtorch.a']
for lib in libs do 
    path = "#{install_path}/lib/#{lib}"
    libref = project.frameworks_group.new_file(path)
    target.frameworks_build_phases.add_file_reference(libref)
end
project.save

# run xcodebuild
exec "xcodebuild clean build  -project #{xcodeproj_path}  -target #{target.name} -sdk iphonesimulator -configuration Release"
