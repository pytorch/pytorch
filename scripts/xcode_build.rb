require 'xcodeproj'

install_path = File.expand_path(ARGV[0])
puts("Libraries install path: #{install_path}")
if not Dir.exist? (install_path) 
    puts "path don't exist:#{install_path}!"
    exit(false)
end
xcodeproj_path = File.expand_path(ARGV[1])
puts("XCode project: #{xcodeproj_path}")
if not File.exist? (xcodeproj_path) 
    puts "path don't exist:#{xcodeproj_path}!"
    exit(false)
end

project = Xcodeproj::Project.open(xcodeproj_path)
target = project.targets.first
header_search_path      = ['$(inherited)', "#{install_path}/include"]
libraries_search_path   = ['$(inherited)', "#{install_path}/lib"]
other_linker_flags      = ['$(inherited)', "-force_load #{install_path}/lib/libtorch.a"]

puts ("HEADER SEARCH PATHS: #{header_search_path}")
puts ("LIBRARIES SEARCH PATHS: #{libraries_search_path}")
puts ("OTHER_LINKER_FALGS:  #{other_linker_flags}")

target.build_configurations.each do |config|
    config.build_settings['HEADER_SEARCH_PATHS']    = header_search_path
    config.build_settings['LIBRARY_SEARCH_PATHS']   = libraries_search_path
    config.build_settings['OTHER_LINKER_FLAGS']     = other_linker_flags
end

# link static libraries
libs = ['libc10.a', 'libclog.a', 'libcpuinfo.a', 'libpytorch_qnnpack.a', 'libtorch.a']
for lib in libs do 
    path = "#{install_path}/lib/#{lib}"
    puts path
    libref = project.frameworks_group.new_file(path)
    target.frameworks_build_phases.add_file_reference(libref)
end

project.save
