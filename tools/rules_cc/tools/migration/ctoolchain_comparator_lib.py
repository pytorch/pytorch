# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module providing compare_ctoolchains function.

compare_ctoolchains takes in two parsed CToolchains and compares them
"""


def _print_difference(field_name, before_value, after_value):
  if not before_value and after_value:
    print(("Difference in '%s' field:\nValue before change is not set\n"
           "Value after change is set to '%s'") % (field_name, after_value))
  elif before_value and not after_value:
    print(("Difference in '%s' field:\nValue before change is set to '%s'\n"
           "Value after change is not set") % (field_name, before_value))
  else:
    print(("Difference in '%s' field:\nValue before change:\t'%s'\n"
           "Value after change:\t'%s'\n") % (field_name, before_value,
                                             after_value))


def _array_to_string(arr, ordered=False):
  if not arr:
    return "[]"
  elif len(arr) == 1:
    return "[" + list(arr)[0] + "]"
  if not ordered:
    return "[\n\t%s\n]" % "\n\t".join(arr)
  else:
    return "[\n\t%s\n]" % "\n\t".join(sorted(list(arr)))


def _check_with_feature_set_equivalence(before, after):
  before_set = set()
  after_set = set()
  for el in before:
    before_set.add((str(set(el.feature)), str(set(el.not_feature))))
  for el in after:
    after_set.add((str(set(el.feature)), str(set(el.not_feature))))
  return before_set == after_set


def _check_tool_equivalence(before, after):
  """Compares two "CToolchain.Tool"s."""
  if before.tool_path == "NOT_USED":
    before.tool_path = ""
  if after.tool_path == "NOT_USED":
    after.tool_path = ""
  if before.tool_path != after.tool_path:
    return False
  if set(before.execution_requirement) != set(after.execution_requirement):
    return False
  if not _check_with_feature_set_equivalence(before.with_feature,
                                             after.with_feature):
    return False
  return True


def _check_flag_group_equivalence(before, after):
  """Compares two "CToolchain.FlagGroup"s."""
  if before.flag != after.flag:
    return False
  if before.expand_if_true != after.expand_if_true:
    return False
  if before.expand_if_false != after.expand_if_false:
    return False
  if set(before.expand_if_all_available) != set(after.expand_if_all_available):
    return False
  if set(before.expand_if_none_available) != set(
      after.expand_if_none_available):
    return False
  if before.iterate_over != after.iterate_over:
    return False
  if before.expand_if_equal != after.expand_if_equal:
    return False
  if len(before.flag_group) != len(after.flag_group):
    return False
  for (flag_group_before, flag_group_after) in zip(before.flag_group,
                                                   after.flag_group):
    if not _check_flag_group_equivalence(flag_group_before, flag_group_after):
      return False
  return True


def _check_flag_set_equivalence(before, after, in_action_config=False):
  """Compares two "CToolchain.FlagSet"s."""
  # ActionConfigs in proto format do not have a 'FlagSet.action' field set.
  # Instead, when construction the Java ActionConfig object, we set the
  # flag_set.action field to the action name. This currently causes the
  # CcToolchainConfigInfo.proto to generate a CToolchain.ActionConfig that still
  # has the action name in the FlagSet.action field, therefore we don't compare
  # the FlagSet.action field when comparing flag_sets that belong to an
  # ActionConfig.
  if not in_action_config and set(before.action) != set(after.action):
    return False
  if not _check_with_feature_set_equivalence(before.with_feature,
                                             after.with_feature):
    return False
  if len(before.flag_group) != len(after.flag_group):
    return False
  for (flag_group_before, flag_group_after) in zip(before.flag_group,
                                                   after.flag_group):
    if not _check_flag_group_equivalence(flag_group_before, flag_group_after):
      return False
  return True


def _check_action_config_equivalence(before, after):
  """Compares two "CToolchain.ActionConfig"s."""
  if before.config_name != after.config_name:
    return False
  if before.action_name != after.action_name:
    return False
  if before.enabled != after.enabled:
    return False
  if len(before.tool) != len(after.tool):
    return False
  for (tool_before, tool_after) in zip(before.tool, after.tool):
    if not _check_tool_equivalence(tool_before, tool_after):
      return False
  if before.implies != after.implies:
    return False
  if len(before.flag_set) != len(after.flag_set):
    return False
  for (flag_set_before, flag_set_after) in zip(before.flag_set, after.flag_set):
    if not _check_flag_set_equivalence(flag_set_before, flag_set_after, True):
      return False
  return True


def _check_env_set_equivalence(before, after):
  """Compares two "CToolchain.EnvSet"s."""
  if set(before.action) != set(after.action):
    return False
  if not _check_with_feature_set_equivalence(before.with_feature,
                                             after.with_feature):
    return False
  if before.env_entry != after.env_entry:
    return False
  return True


def _check_feature_equivalence(before, after):
  """Compares two "CToolchain.Feature"s."""
  if before.name != after.name:
    return False
  if before.enabled != after.enabled:
    return False
  if len(before.flag_set) != len(after.flag_set):
    return False
  for (flag_set_before, flag_set_after) in zip(before.flag_set, after.flag_set):
    if not _check_flag_set_equivalence(flag_set_before, flag_set_after):
      return False
  if len(before.env_set) != len(after.env_set):
    return False
  for (env_set_before, env_set_after) in zip(before.env_set, after.env_set):
    if not _check_env_set_equivalence(env_set_before, env_set_after):
      return False
  if len(before.requires) != len(after.requires):
    return False
  for (requires_before, requires_after) in zip(before.requires, after.requires):
    if set(requires_before.feature) != set(requires_after.feature):
      return False
  if before.implies != after.implies:
    return False
  if before.provides != after.provides:
    return False
  return True


def _compare_features(features_before, features_after):
  """Compares two "CToolchain.FlagFeature" lists."""
  feature_name_to_feature_before = {}
  feature_name_to_feature_after = {}
  for feature in features_before:
    feature_name_to_feature_before[feature.name] = feature
  for feature in features_after:
    feature_name_to_feature_after[feature.name] = feature

  feature_names_before = set(feature_name_to_feature_before.keys())
  feature_names_after = set(feature_name_to_feature_after.keys())

  before_after_diff = feature_names_before - feature_names_after
  after_before_diff = feature_names_after - feature_names_before

  diff_string = "Difference in 'feature' field:"
  found_difference = False
  if before_after_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List before change contains entries for the following features "
           "that the list after the change doesn't:\n%s") % _array_to_string(
               before_after_diff, ordered=True))
  if after_before_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List after change contains entries for the following features "
           "that the list before the change doesn't:\n%s") % _array_to_string(
               after_before_diff, ordered=True))

  names_before = [feature.name for feature in features_before]
  names_after = [feature.name for feature in features_after]
  if names_before != names_after:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("Features not in right order:\n"
           "* List of features before change:\t%s"
           "* List of features before change:\t%s") %
          (_array_to_string(names_before), _array_to_string(names_after)))
  for name in feature_name_to_feature_before:
    feature_before = feature_name_to_feature_before[name]
    feature_after = feature_name_to_feature_after.get(name, None)
    if feature_after and not _check_feature_equivalence(feature_before,
                                                        feature_after):
      if not found_difference:
        print(diff_string)  # pylint: disable=superfluous-parens
        found_difference = True
      print(("* Feature '%s' differs before and after the change:\n"
             "Value before change:\n%s\n"
             "Value after change:\n%s") % (name, str(feature_before),
                                           str(feature_after)))
  if found_difference:
    print("")  # pylint: disable=superfluous-parens
  return found_difference


def _compare_action_configs(action_configs_before, action_configs_after):
  """Compares two "CToolchain.ActionConfig" lists."""
  action_name_to_action_before = {}
  action_name_to_action_after = {}
  for action_config in action_configs_before:
    action_name_to_action_before[action_config.config_name] = action_config
  for action_config in action_configs_after:
    action_name_to_action_after[action_config.config_name] = action_config

  config_names_before = set(action_name_to_action_before.keys())
  config_names_after = set(action_name_to_action_after.keys())

  before_after_diff = config_names_before - config_names_after
  after_before_diff = config_names_after - config_names_before

  diff_string = "Difference in 'action_config' field:"
  found_difference = False
  if before_after_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List before change contains entries for the following "
           "action_configs that the list after the change doesn't:\n%s") %
          _array_to_string(before_after_diff, ordered=True))
  if after_before_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List after change contains entries for the following "
           "action_configs that the list before the change doesn't:\n%s") %
          _array_to_string(after_before_diff, ordered=True))

  names_before = [config.config_name for config in action_configs_before]
  names_after = [config.config_name for config in action_configs_after]
  if names_before != names_after:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("Action configs not in right order:\n"
           "* List of action configs before change:\t%s"
           "* List of action_configs before change:\t%s") %
          (_array_to_string(names_before), _array_to_string(names_after)))
  for name in config_names_before:
    action_config_before = action_name_to_action_before[name]
    action_config_after = action_name_to_action_after.get(name, None)
    if action_config_after and not _check_action_config_equivalence(
        action_config_before, action_config_after):
      if not found_difference:
        print(diff_string)  # pylint: disable=superfluous-parens
        found_difference = True
      print(("* Action config '%s' differs before and after the change:\n"
             "Value before change:\n%s\n"
             "Value after change:\n%s") % (name, str(action_config_before),
                                           str(action_config_after)))
  if found_difference:
    print("")  # pylint: disable=superfluous-parens
  return found_difference


def _compare_tool_paths(tool_paths_before, tool_paths_after):
  """Compares two "CToolchain.ToolPath" lists."""
  tool_to_path_before = {}
  tool_to_path_after = {}
  for tool_path in tool_paths_before:
    tool_to_path_before[tool_path.name] = (
        tool_path.path if tool_path.path != "NOT_USED" else "")
  for tool_path in tool_paths_after:
    tool_to_path_after[tool_path.name] = (
        tool_path.path if tool_path.path != "NOT_USED" else "")

  tool_names_before = set(tool_to_path_before.keys())
  tool_names_after = set(tool_to_path_after.keys())

  before_after_diff = tool_names_before - tool_names_after
  after_before_diff = tool_names_after - tool_names_before

  diff_string = "Difference in 'tool_path' field:"
  found_difference = False
  if before_after_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List before change contains entries for the following tools "
           "that the list after the change doesn't:\n%s") % _array_to_string(
               before_after_diff, ordered=True))
  if after_before_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List after change contains entries for the following tools that "
           "the list before the change doesn't:\n%s") % _array_to_string(
               after_before_diff, ordered=True))

  for tool in tool_to_path_before:
    path_before = tool_to_path_before[tool]
    path_after = tool_to_path_after.get(tool, None)
    if path_after and path_after != path_before:
      if not found_difference:
        print(diff_string)  # pylint: disable=superfluous-parens
        found_difference = True
      print(("* Path for tool '%s' differs before and after the change:\n"
             "Value before change:\t'%s'\n"
             "Value after change:\t'%s'") % (tool, path_before, path_after))
  if found_difference:
    print("")  # pylint: disable=superfluous-parens
  return found_difference


def _compare_make_variables(make_variables_before, make_variables_after):
  """Compares two "CToolchain.MakeVariable" lists."""
  name_to_variable_before = {}
  name_to_variable_after = {}
  for variable in make_variables_before:
    name_to_variable_before[variable.name] = variable.value
  for variable in make_variables_after:
    name_to_variable_after[variable.name] = variable.value

  variable_names_before = set(name_to_variable_before.keys())
  variable_names_after = set(name_to_variable_after.keys())

  before_after_diff = variable_names_before - variable_names_after
  after_before_diff = variable_names_after - variable_names_before

  diff_string = "Difference in 'make_variable' field:"
  found_difference = False
  if before_after_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List before change contains entries for the following variables "
           "that the list after the change doesn't:\n%s") % _array_to_string(
               before_after_diff, ordered=True))
  if after_before_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List after change contains entries for the following variables "
           "that the list before the change doesn't:\n%s") % _array_to_string(
               after_before_diff, ordered=True))

  for variable in name_to_variable_before:
    value_before = name_to_variable_before[variable]
    value_after = name_to_variable_after.get(variable, None)
    if value_after and value_after != value_before:
      if not found_difference:
        print(diff_string)  # pylint: disable=superfluous-parens
        found_difference = True
      print(
          ("* Value for variable '%s' differs before and after the change:\n"
           "Value before change:\t'%s'\n"
           "Value after change:\t'%s'") % (variable, value_before, value_after))
  if found_difference:
    print("")  # pylint: disable=superfluous-parens
  return found_difference


def _compare_cxx_builtin_include_directories(directories_before,
                                             directories_after):
  if directories_before != directories_after:
    print(("Difference in 'cxx_builtin_include_directory' field:\n"
           "List of elements before change:\n%s\n"
           "List of elements after change:\n%s\n") %
          (_array_to_string(directories_before),
           _array_to_string(directories_after)))
    return True
  return False


def _compare_artifact_name_patterns(artifact_name_patterns_before,
                                    artifact_name_patterns_after):
  """Compares two "CToolchain.ArtifactNamePattern" lists."""
  category_to_values_before = {}
  category_to_values_after = {}
  for name_pattern in artifact_name_patterns_before:
    category_to_values_before[name_pattern.category_name] = (
        name_pattern.prefix, name_pattern.extension)
  for name_pattern in artifact_name_patterns_after:
    category_to_values_after[name_pattern.category_name] = (
        name_pattern.prefix, name_pattern.extension)

  category_names_before = set(category_to_values_before.keys())
  category_names_after = set(category_to_values_after.keys())

  before_after_diff = category_names_before - category_names_after
  after_before_diff = category_names_after - category_names_before

  diff_string = "Difference in 'artifact_name_pattern' field:"
  found_difference = False
  if before_after_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List before change contains entries for the following categories "
           "that the list after the change doesn't:\n%s") % _array_to_string(
               before_after_diff, ordered=True))
  if after_before_diff:
    if not found_difference:
      print(diff_string)  # pylint: disable=superfluous-parens
      found_difference = True
    print(("* List after change contains entries for the following categories "
           "that the list before the change doesn't:\n%s") % _array_to_string(
               after_before_diff, ordered=True))

  for category in category_to_values_before:
    value_before = category_to_values_before[category]
    value_after = category_to_values_after.get(category, None)
    if value_after and value_after != value_before:
      if not found_difference:
        print(diff_string)  # pylint: disable=superfluous-parens
        found_difference = True
      print(("* Value for category '%s' differs before and after the change:\n"
             "Value before change:\tprefix:'%s'\textension:'%s'\n"
             "Value after change:\tprefix:'%s'\textension:'%s'") %
            (category, value_before[0], value_before[1], value_after[0],
             value_after[1]))
  if found_difference:
    print("")  # pylint: disable=superfluous-parens
  return found_difference


def compare_ctoolchains(toolchain_before, toolchain_after):
  """Compares two CToolchains."""
  found_difference = False
  if (toolchain_before.toolchain_identifier !=
      toolchain_after.toolchain_identifier):
    _print_difference("toolchain_identifier",
                      toolchain_before.toolchain_identifier,
                      toolchain_after.toolchain_identifier)
  if toolchain_before.host_system_name != toolchain_after.host_system_name:
    _print_difference("host_system_name", toolchain_before.host_system_name,
                      toolchain_after.host_system_name)
    found_difference = True
  if toolchain_before.target_system_name != toolchain_after.target_system_name:
    _print_difference("target_system_name", toolchain_before.target_system_name,
                      toolchain_after.target_system_name)
    found_difference = True
  if toolchain_before.target_cpu != toolchain_after.target_cpu:
    _print_difference("target_cpu", toolchain_before.target_cpu,
                      toolchain_after.target_cpu)
    found_difference = True
  if toolchain_before.target_libc != toolchain_after.target_libc:
    _print_difference("target_libc", toolchain_before.target_libc,
                      toolchain_after.target_libc)
    found_difference = True
  if toolchain_before.compiler != toolchain_after.compiler:
    _print_difference("compiler", toolchain_before.compiler,
                      toolchain_after.compiler)
    found_difference = True
  if toolchain_before.abi_version != toolchain_after.abi_version:
    _print_difference("abi_version", toolchain_before.abi_version,
                      toolchain_after.abi_version)
    found_difference = True
  if toolchain_before.abi_libc_version != toolchain_after.abi_libc_version:
    _print_difference("abi_libc_version", toolchain_before.abi_libc_version,
                      toolchain_after.abi_libc_version)
    found_difference = True
  if toolchain_before.cc_target_os != toolchain_after.cc_target_os:
    _print_difference("cc_target_os", toolchain_before.cc_target_os,
                      toolchain_after.cc_target_os)
    found_difference = True
  if toolchain_before.builtin_sysroot != toolchain_after.builtin_sysroot:
    _print_difference("builtin_sysroot", toolchain_before.builtin_sysroot,
                      toolchain_after.builtin_sysroot)
    found_difference = True
  found_difference = _compare_features(
      toolchain_before.feature, toolchain_after.feature) or found_difference
  found_difference = _compare_action_configs(
      toolchain_before.action_config,
      toolchain_after.action_config) or found_difference
  found_difference = _compare_tool_paths(
      toolchain_before.tool_path, toolchain_after.tool_path) or found_difference
  found_difference = _compare_cxx_builtin_include_directories(
      toolchain_before.cxx_builtin_include_directory,
      toolchain_after.cxx_builtin_include_directory) or found_difference
  found_difference = _compare_make_variables(
      toolchain_before.make_variable,
      toolchain_after.make_variable) or found_difference
  found_difference = _compare_artifact_name_patterns(
      toolchain_before.artifact_name_pattern,
      toolchain_after.artifact_name_pattern) or found_difference
  if not found_difference:
    print("No difference")  # pylint: disable=superfluous-parens
  return found_difference
