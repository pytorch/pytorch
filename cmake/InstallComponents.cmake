# Install components used to control which install() rules end up in the
# wheel. scikit-build-core packages only the components listed in
# pyproject.toml's `install.components` setting, so rules tagged with an
# unlisted component are silently skipped during wheel build while still
# being available for non-wheel `cmake --install --component <name>` flows.
#
# Components:
#   libtorch     Default. Shared libraries, headers, runtime data, and
#                anything else that would belong in the libtorch C++
#                distribution.
#   torch        Python frontend: libtorch_python, the _C extension shim,
#                generated .py files, type stubs, yaml/jinja templates
#                consumed by torch's Python side, and similar artifacts.
#   dev          CMake config files consumed by downstream
#                find_package(Torch) / find_package(Caffe2) users
#                (Caffe2Config, public/*.cmake, Modules_CUDA_fix, ...).
#   third_party  Sentinel for vendored subprojects added via
#                pytorch_add_thirdparty_subdirectory(). Not included in
#                the wheel, so subproject install() rules (e.g. libfbgemm.a,
#                lib64/cmake/foo-config.cmake) become no-ops at install time.
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "libtorch")

# Wrap add_subdirectory() for vendored third-party projects so that any
# install() rules registered by the subdirectory go to the "third_party"
# component instead of the default. Build behavior is unchanged: targets
# are still configured and built; only the install rules are reassigned.
#
# Defined as a macro (not a function) so that variables and policies set
# by the third-party CMakeLists propagate to the caller as they would for
# a bare add_subdirectory().
macro(pytorch_add_thirdparty_subdirectory)
  set(_pytorch_saved_default_component "${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME}")
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "third_party")
  add_subdirectory(${ARGV})
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "${_pytorch_saved_default_component}")
  unset(_pytorch_saved_default_component)
endmacro()
