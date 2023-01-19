"""Module providing migrate_legacy_fields function.

migrate_legacy_fields takes parsed CROSSTOOL proto and migrates it (inplace) to
use only the features.

Tracking issue: https://github.com/bazelbuild/bazel/issues/5187

Since C++ rules team is working on migrating CROSSTOOL from text proto into
Starlark, we advise CROSSTOOL owners to wait for the CROSSTOOL -> Starlark
migrator before they invest too much time into fixing their pipeline. Tracking
issue for the Starlark effort is
https://github.com/bazelbuild/bazel/issues/5380.
"""

from third_party.com.github.bazelbuild.bazel.src.main.protobuf import crosstool_config_pb2

ALL_CC_COMPILE_ACTIONS = [
    "assemble", "preprocess-assemble", "linkstamp-compile", "c-compile",
    "c++-compile", "c++-header-parsing", "c++-module-compile",
    "c++-module-codegen", "lto-backend", "clif-match"
]

ALL_OBJC_COMPILE_ACTIONS = [
    "objc-compile", "objc++-compile"
]

ALL_CXX_COMPILE_ACTIONS = [
    action for action in ALL_CC_COMPILE_ACTIONS
    if action not in ["c-compile", "preprocess-assemble", "assemble"]
]

ALL_CC_LINK_ACTIONS = [
    "c++-link-executable", "c++-link-dynamic-library",
    "c++-link-nodeps-dynamic-library"
]

ALL_OBJC_LINK_ACTIONS = [
    "objc-executable", "objc++-executable",
]

DYNAMIC_LIBRARY_LINK_ACTIONS = [
    "c++-link-dynamic-library", "c++-link-nodeps-dynamic-library"
]

NODEPS_DYNAMIC_LIBRARY_LINK_ACTIONS = ["c++-link-nodeps-dynamic-library"]

TRANSITIVE_DYNAMIC_LIBRARY_LINK_ACTIONS = ["c++-link-dynamic-library"]

TRANSITIVE_LINK_ACTIONS = ["c++-link-executable", "c++-link-dynamic-library"]

CC_LINK_EXECUTABLE = ["c++-link-executable"]


def compile_actions(toolchain):
  """Returns compile actions for cc or objc rules."""
  if _is_objc_toolchain(toolchain):
    return ALL_CC_COMPILE_ACTIONS + ALL_OBJC_COMPILE_ACTIONS
  else:
    return ALL_CC_COMPILE_ACTIONS

def link_actions(toolchain):
  """Returns link actions for cc or objc rules."""
  if _is_objc_toolchain(toolchain):
    return ALL_CC_LINK_ACTIONS + ALL_OBJC_LINK_ACTIONS
  else:
    return ALL_CC_LINK_ACTIONS


def executable_link_actions(toolchain):
  """Returns transitive link actions for cc or objc rules."""
  if _is_objc_toolchain(toolchain):
    return CC_LINK_EXECUTABLE + ALL_OBJC_LINK_ACTIONS
  else:
    return CC_LINK_EXECUTABLE


def _is_objc_toolchain(toolchain):
  return any(ac.action_name == "objc-compile" for ac in toolchain.action_config)

# Map converting from LinkingMode to corresponding feature name
LINKING_MODE_TO_FEATURE_NAME = {
    "FULLY_STATIC": "fully_static_link",
    "MOSTLY_STATIC": "static_linking_mode",
    "DYNAMIC": "dynamic_linking_mode",
    "MOSTLY_STATIC_LIBRARIES": "static_linking_mode_nodeps_library",
}

def migrate_legacy_fields(crosstool):
  """Migrates parsed crosstool (inplace) to not use legacy fields."""
  crosstool.ClearField("default_toolchain")
  for toolchain in crosstool.toolchain:
    _ = [_migrate_expand_if_all_available(f) for f in toolchain.feature]
    _ = [_migrate_expand_if_all_available(ac) for ac in toolchain.action_config]
    _ = [_migrate_repeated_expands(f) for f in toolchain.feature]
    _ = [_migrate_repeated_expands(ac) for ac in toolchain.action_config]

    if (toolchain.dynamic_library_linker_flag or
        _contains_dynamic_flags(toolchain)) and not _get_feature(
            toolchain, "supports_dynamic_linker"):
      feature = toolchain.feature.add()
      feature.name = "supports_dynamic_linker"
      feature.enabled = True

    if toolchain.supports_start_end_lib and not _get_feature(
        toolchain, "supports_start_end_lib"):
      feature = toolchain.feature.add()
      feature.name = "supports_start_end_lib"
      feature.enabled = True

    if toolchain.supports_interface_shared_objects and not _get_feature(
        toolchain, "supports_interface_shared_libraries"):
      feature = toolchain.feature.add()
      feature.name = "supports_interface_shared_libraries"
      feature.enabled = True

    if toolchain.supports_embedded_runtimes and not _get_feature(
        toolchain, "static_link_cpp_runtimes"):
      feature = toolchain.feature.add()
      feature.name = "static_link_cpp_runtimes"
      feature.enabled = True

    if toolchain.needsPic and not _get_feature(toolchain, "supports_pic"):
      feature = toolchain.feature.add()
      feature.name = "supports_pic"
      feature.enabled = True

    if toolchain.supports_fission and not _get_feature(
        toolchain, "per_object_debug_info"):
      # feature {
      #   name: "per_object_debug_info"
      #   enabled: true
      #   flag_set {
      #     action: "assemble"
      #     action: "preprocess-assemble"
      #     action: "c-compile"
      #     action: "c++-compile"
      #     action: "c++-module-codegen"
      #     action: "lto-backend"
      #     flag_group {
      #       expand_if_all_available: 'is_using_fission'",
      #       flag: "-gsplit-dwarf"
      #     }
      #   }
      # }
      feature = toolchain.feature.add()
      feature.name = "per_object_debug_info"
      feature.enabled = True
      flag_set = feature.flag_set.add()
      flag_set.action[:] = [
          "c-compile", "c++-compile", "c++-module-codegen", "assemble",
          "preprocess-assemble", "lto-backend"
      ]
      flag_group = flag_set.flag_group.add()
      flag_group.expand_if_all_available[:] = ["is_using_fission"]
      flag_group.flag[:] = ["-gsplit-dwarf"]

    if toolchain.objcopy_embed_flag and not _get_feature(
        toolchain, "objcopy_embed_flags"):
      feature = toolchain.feature.add()
      feature.name = "objcopy_embed_flags"
      feature.enabled = True
      flag_set = feature.flag_set.add()
      flag_set.action[:] = ["objcopy_embed_data"]
      flag_group = flag_set.flag_group.add()
      flag_group.flag[:] = toolchain.objcopy_embed_flag

      action_config = toolchain.action_config.add()
      action_config.action_name = "objcopy_embed_data"
      action_config.config_name = "objcopy_embed_data"
      action_config.enabled = True
      tool = action_config.tool.add()
      tool.tool_path = _find_tool_path(toolchain, "objcopy")

    if toolchain.ld_embed_flag and not _get_feature(
        toolchain, "ld_embed_flags"):
      feature = toolchain.feature.add()
      feature.name = "ld_embed_flags"
      feature.enabled = True
      flag_set = feature.flag_set.add()
      flag_set.action[:] = ["ld_embed_data"]
      flag_group = flag_set.flag_group.add()
      flag_group.flag[:] = toolchain.ld_embed_flag

      action_config = toolchain.action_config.add()
      action_config.action_name = "ld_embed_data"
      action_config.config_name = "ld_embed_data"
      action_config.enabled = True
      tool = action_config.tool.add()
      tool.tool_path = _find_tool_path(toolchain, "ld")


    # Create default_link_flags feature for linker_flag
    flag_sets = _extract_legacy_link_flag_sets_for(toolchain)
    if flag_sets:
      if _get_feature(toolchain, "default_link_flags"):
        continue
      if _get_feature(toolchain, "legacy_link_flags"):
        for f in toolchain.feature:
          if f.name == "legacy_link_flags":
            f.ClearField("flag_set")
            feature = f
            _rename_feature_in_toolchain(toolchain, "legacy_link_flags",
                                         "default_link_flags")
            break
      else:
        feature = _prepend_feature(toolchain)
      feature.name = "default_link_flags"
      feature.enabled = True
      _add_flag_sets(feature, flag_sets)

    # Create default_compile_flags feature for compiler_flag, cxx_flag
    flag_sets = _extract_legacy_compile_flag_sets_for(toolchain)
    if flag_sets and not _get_feature(toolchain, "default_compile_flags"):
      if _get_feature(toolchain, "legacy_compile_flags"):
        for f in toolchain.feature:
          if f.name == "legacy_compile_flags":
            f.ClearField("flag_set")
            feature = f
            _rename_feature_in_toolchain(toolchain, "legacy_compile_flags",
                                         "default_compile_flags")
            break
      else:
        feature = _prepend_feature(toolchain)
      feature.enabled = True
      feature.name = "default_compile_flags"
      _add_flag_sets(feature, flag_sets)

    # Unfiltered cxx flags have to have their own special feature.
    # "unfiltered_compile_flags" is a well-known (by Bazel) feature name that is
    # excluded from nocopts filtering.
    if toolchain.unfiltered_cxx_flag:
      # If there already is a feature named unfiltered_compile_flags, the
      # crosstool is already migrated for unfiltered_compile_flags
      if _get_feature(toolchain, "unfiltered_compile_flags"):
        for f in toolchain.feature:
          if f.name == "unfiltered_compile_flags":
            for flag_set in f.flag_set:
              for flag_group in flag_set.flag_group:
                if flag_group.iterate_over == "unfiltered_compile_flags":
                  flag_group.ClearField("iterate_over")
                  flag_group.ClearField("expand_if_all_available")
                  flag_group.ClearField("flag")
                  flag_group.flag[:] = toolchain.unfiltered_cxx_flag
      else:
        if not _get_feature(toolchain, "user_compile_flags"):
          feature = toolchain.feature.add()
          feature.name = "user_compile_flags"
          feature.enabled = True
          flag_set = feature.flag_set.add()
          flag_set.action[:] = compile_actions(toolchain)
          flag_group = flag_set.flag_group.add()
          flag_group.expand_if_all_available[:] = ["user_compile_flags"]
          flag_group.iterate_over = "user_compile_flags"
          flag_group.flag[:] = ["%{user_compile_flags}"]

        if not _get_feature(toolchain, "sysroot"):
          sysroot_actions = compile_actions(toolchain) + link_actions(toolchain)
          sysroot_actions.remove("assemble")
          feature = toolchain.feature.add()
          feature.name = "sysroot"
          feature.enabled = True
          flag_set = feature.flag_set.add()
          flag_set.action[:] = sysroot_actions
          flag_group = flag_set.flag_group.add()
          flag_group.expand_if_all_available[:] = ["sysroot"]
          flag_group.flag[:] = ["--sysroot=%{sysroot}"]

        feature = toolchain.feature.add()
        feature.name = "unfiltered_compile_flags"
        feature.enabled = True
        flag_set = feature.flag_set.add()
        flag_set.action[:] = compile_actions(toolchain)
        flag_group = flag_set.flag_group.add()
        flag_group.flag[:] = toolchain.unfiltered_cxx_flag

    # clear fields
    toolchain.ClearField("debian_extra_requires")
    toolchain.ClearField("gcc_plugin_compiler_flag")
    toolchain.ClearField("ar_flag")
    toolchain.ClearField("ar_thin_archives_flag")
    toolchain.ClearField("gcc_plugin_header_directory")
    toolchain.ClearField("mao_plugin_header_directory")
    toolchain.ClearField("supports_normalizing_ar")
    toolchain.ClearField("supports_thin_archives")
    toolchain.ClearField("supports_incremental_linker")
    toolchain.ClearField("supports_dsym")
    toolchain.ClearField("supports_gold_linker")
    toolchain.ClearField("default_python_top")
    toolchain.ClearField("default_python_version")
    toolchain.ClearField("python_preload_swigdeps")
    toolchain.ClearField("needsPic")
    toolchain.ClearField("compilation_mode_flags")
    toolchain.ClearField("linking_mode_flags")
    toolchain.ClearField("unfiltered_cxx_flag")
    toolchain.ClearField("ld_embed_flag")
    toolchain.ClearField("objcopy_embed_flag")
    toolchain.ClearField("supports_start_end_lib")
    toolchain.ClearField("supports_interface_shared_objects")
    toolchain.ClearField("supports_fission")
    toolchain.ClearField("supports_embedded_runtimes")
    toolchain.ClearField("compiler_flag")
    toolchain.ClearField("cxx_flag")
    toolchain.ClearField("linker_flag")
    toolchain.ClearField("dynamic_library_linker_flag")
    toolchain.ClearField("static_runtimes_filegroup")
    toolchain.ClearField("dynamic_runtimes_filegroup")

    # Enable features that were previously enabled by Bazel
    default_features = [
        "dependency_file", "random_seed", "module_maps", "module_map_home_cwd",
        "header_module_compile", "include_paths", "pic", "preprocessor_define"
    ]
    for feature_name in default_features:
      feature = _get_feature(toolchain, feature_name)
      if feature:
        feature.enabled = True


def _find_tool_path(toolchain, tool_name):
  """Returns the tool path of the tool with the given name."""
  for tool in toolchain.tool_path:
    if tool.name == tool_name:
      return tool.path
  return None


def _add_flag_sets(feature, flag_sets):
  """Add flag sets into a feature."""
  for flag_set in flag_sets:
    with_feature = flag_set[0]
    actions = flag_set[1]
    flags = flag_set[2]
    expand_if_all_available = flag_set[3]
    not_feature = None
    if len(flag_set) >= 5:
      not_feature = flag_set[4]
    flag_set = feature.flag_set.add()
    if with_feature is not None:
      flag_set.with_feature.add().feature[:] = [with_feature]
    if not_feature is not None:
      flag_set.with_feature.add().not_feature[:] = [not_feature]
    flag_set.action[:] = actions
    flag_group = flag_set.flag_group.add()
    flag_group.expand_if_all_available[:] = expand_if_all_available
    flag_group.flag[:] = flags
  return feature


def _extract_legacy_compile_flag_sets_for(toolchain):
  """Get flag sets for default_compile_flags feature."""
  result = []
  if toolchain.compiler_flag:
    result.append(
        [None, compile_actions(toolchain), toolchain.compiler_flag, []])

  # Migrate compiler_flag from compilation_mode_flags
  for cmf in toolchain.compilation_mode_flags:
    mode = crosstool_config_pb2.CompilationMode.Name(cmf.mode).lower()
    # coverage mode has been a noop since a while
    if mode == "coverage":
      continue

    if (cmf.compiler_flag or
        cmf.cxx_flag) and not _get_feature(toolchain, mode):
      feature = toolchain.feature.add()
      feature.name = mode

    if cmf.compiler_flag:
      result.append([mode, compile_actions(toolchain), cmf.compiler_flag, []])

  if toolchain.cxx_flag:
    result.append([None, ALL_CXX_COMPILE_ACTIONS, toolchain.cxx_flag, []])

  # Migrate compiler_flag/cxx_flag from compilation_mode_flags
  for cmf in toolchain.compilation_mode_flags:
    mode = crosstool_config_pb2.CompilationMode.Name(cmf.mode).lower()
    # coverage mode has been a noop since a while
    if mode == "coverage":
      continue

    if cmf.cxx_flag:
      result.append([mode, ALL_CXX_COMPILE_ACTIONS, cmf.cxx_flag, []])

  return result


def _extract_legacy_link_flag_sets_for(toolchain):
  """Get flag sets for default_link_flags feature."""
  result = []

  # Migrate linker_flag
  if toolchain.linker_flag:
    result.append([None, link_actions(toolchain), toolchain.linker_flag, []])

  # Migrate linker_flags from compilation_mode_flags
  for cmf in toolchain.compilation_mode_flags:
    mode = crosstool_config_pb2.CompilationMode.Name(cmf.mode).lower()
    # coverage mode has beed a noop since a while
    if mode == "coverage":
      continue

    if cmf.linker_flag and not _get_feature(toolchain, mode):
      feature = toolchain.feature.add()
      feature.name = mode

    if cmf.linker_flag:
      result.append([mode, link_actions(toolchain), cmf.linker_flag, []])

  # Migrate linker_flags from linking_mode_flags
  for lmf in toolchain.linking_mode_flags:
    mode = crosstool_config_pb2.LinkingMode.Name(lmf.mode)
    feature_name = LINKING_MODE_TO_FEATURE_NAME.get(mode)
    # if the feature is already there, we don't migrate, lmf is not used
    if _get_feature(toolchain, feature_name):
      continue

    if lmf.linker_flag:
      feature = toolchain.feature.add()
      feature.name = feature_name
      if mode == "DYNAMIC":
        result.append(
            [None, NODEPS_DYNAMIC_LIBRARY_LINK_ACTIONS, lmf.linker_flag, []])
        result.append([
            None,
            TRANSITIVE_DYNAMIC_LIBRARY_LINK_ACTIONS,
            lmf.linker_flag,
            [],
            "static_link_cpp_runtimes",
        ])
        result.append([
            feature_name,
            executable_link_actions(toolchain), lmf.linker_flag, []
        ])
      elif mode == "MOSTLY_STATIC":
        result.append(
            [feature_name,
             CC_LINK_EXECUTABLE, lmf.linker_flag, []])
      else:
        result.append(
           [feature_name,
            link_actions(toolchain), lmf.linker_flag, []])

  if toolchain.dynamic_library_linker_flag:
    result.append([
        None, DYNAMIC_LIBRARY_LINK_ACTIONS,
        toolchain.dynamic_library_linker_flag, []
    ])

  if toolchain.test_only_linker_flag:
    result.append([
        None,
        link_actions(toolchain), toolchain.test_only_linker_flag,
        ["is_cc_test"]
    ])

  return result


def _prepend_feature(toolchain):
  """Create a new feature and make it be the first in the toolchain."""
  features = toolchain.feature
  toolchain.ClearField("feature")
  new_feature = toolchain.feature.add()
  toolchain.feature.extend(features)
  return new_feature


def _get_feature(toolchain, name):
  """Returns feature with a given name or None."""
  for feature in toolchain.feature:
    if feature.name == name:
      return feature
  return None


def _migrate_expand_if_all_available(message):
  """Move expand_if_all_available field to flag_groups."""
  for flag_set in message.flag_set:
    if flag_set.expand_if_all_available:
      for flag_group in flag_set.flag_group:
        new_vars = (
            flag_group.expand_if_all_available[:] +
            flag_set.expand_if_all_available[:])
        flag_group.expand_if_all_available[:] = new_vars
      flag_set.ClearField("expand_if_all_available")


def _migrate_repeated_expands(message):
  """Replace repeated legacy fields with nesting."""
  todo_queue = []
  for flag_set in message.flag_set:
    todo_queue.extend(flag_set.flag_group)
  while todo_queue:
    flag_group = todo_queue.pop()
    todo_queue.extend(flag_group.flag_group)
    if len(flag_group.expand_if_all_available) <= 1 and len(
        flag_group.expand_if_none_available) <= 1:
      continue

    current_children = flag_group.flag_group
    current_flags = flag_group.flag
    flag_group.ClearField("flag_group")
    flag_group.ClearField("flag")

    new_flag_group = flag_group.flag_group.add()
    new_flag_group.flag_group.extend(current_children)
    new_flag_group.flag.extend(current_flags)

    if len(flag_group.expand_if_all_available) > 1:
      expands_to_move = flag_group.expand_if_all_available[1:]
      flag_group.expand_if_all_available[:] = [
          flag_group.expand_if_all_available[0]
      ]
      new_flag_group.expand_if_all_available.extend(expands_to_move)

    if len(flag_group.expand_if_none_available) > 1:
      expands_to_move = flag_group.expand_if_none_available[1:]
      flag_group.expand_if_none_available[:] = [
          flag_group.expand_if_none_available[0]
      ]
      new_flag_group.expand_if_none_available.extend(expands_to_move)

    todo_queue.append(new_flag_group)
    todo_queue.append(flag_group)


def _contains_dynamic_flags(toolchain):
  for lmf in toolchain.linking_mode_flags:
    mode = crosstool_config_pb2.LinkingMode.Name(lmf.mode)
    if mode == "DYNAMIC":
      return True
  return False


def _rename_feature_in_toolchain(toolchain, from_name, to_name):
  for f in toolchain.feature:
    _rename_feature_in(f, from_name, to_name)
  for a in toolchain.action_config:
    _rename_feature_in(a, from_name, to_name)


def _rename_feature_in(msg, from_name, to_name):
  if from_name in msg.implies:
    msg.implies.remove(from_name)
  for requires in msg.requires:
    if from_name in requires.feature:
      requires.feature.remove(from_name)
      requires.feature.extend([to_name])
    for flag_set in msg.flag_set:
      for with_feature in flag_set.with_feature:
        if from_name in with_feature.feature:
          with_feature.feature.remove(from_name)
          with_feature.feature.extend([to_name])
        if from_name in with_feature.not_feature:
          with_feature.not_feature.remove(from_name)
          with_feature.not_feature.extend([to_name])
    for env_set in msg.env_set:
      for with_feature in env_set.with_feature:
        if from_name in with_feature.feature:
          with_feature.feature.remove(from_name)
          with_feature.feature.extend([to_name])
        if from_name in with_feature.not_feature:
          with_feature.not_feature.remove(from_name)
          with_feature.not_feature.extend([to_name])
