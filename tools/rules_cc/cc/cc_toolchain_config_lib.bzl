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

""" A library of functions creating structs for CcToolchainConfigInfo."""

def _check_is_none_or_right_type(obj, obj_of_right_type, parameter_name, method_name):
    if obj != None:
        _check_right_type(obj, obj_of_right_type, parameter_name, method_name)

def _check_right_type(obj, obj_of_right_type, parameter_name, method_name):
    if type(obj) != type(obj_of_right_type):
        fail("{} parameter of {} should be a {}, found {}."
            .format(parameter_name, method_name, type(obj_of_right_type), type(obj)))

def _check_is_nonempty_string(obj, parameter_name, method_name):
    _check_right_type(obj, "", parameter_name, method_name)
    if obj == "":
        fail("{} parameter of {} must be a nonempty string."
            .format(parameter_name, method_name))

def _check_is_nonempty_list(obj, parameter_name, method_name):
    _check_right_type(obj, [], parameter_name, method_name)
    if len(obj) == 0:
        fail("{} parameter of {} must be a nonempty list."
            .format(parameter_name, method_name))

EnvEntryInfo = provider(
    "A key/value pair to be added as an environment variable.",
    fields = ["key", "value", "type_name"],
)

def env_entry(key, value):
    """ A key/value pair to be added as an environment variable.

    The returned EnvEntry provider finds its use in EnvSet creation through
    the env_entries parameter of env_set(); EnvSet groups environment variables
    that need to be expanded for specific actions.
    The value of this pair is expanded in the same way as is described in
    flag_group. The key remains an unexpanded string literal.

    Args:
        key: a string literal representing the name of the variable.
        value: the value to be expanded.

    Returns:
        An EnvEntryInfo provider.
    """
    _check_is_nonempty_string(key, "key", "env_entry")
    _check_is_nonempty_string(value, "value", "env_entry")
    return EnvEntryInfo(key = key, value = value, type_name = "env_entry")

VariableWithValueInfo = provider(
    "Represents equality check between a variable and a certain value.",
    fields = ["name", "value", "type_name"],
)

def variable_with_value(name, value):
    """ Represents equality check between a variable and a certain value.

    The returned provider finds its use through flag_group.expand_if_equal,
    making the expansion of the flag_group conditional on the value of the
    variable.

    Args:
        name: name of the variable.
        value: the value the variable should be compared against.

    Returns:
        A VariableWithValueInfo provider.
    """
    _check_is_nonempty_string(name, "name", "variable_with_value")
    _check_is_nonempty_string(value, "value", "variable_with_value")
    return VariableWithValueInfo(
        name = name,
        value = value,
        type_name = "variable_with_value",
    )

MakeVariableInfo = provider(
    "A make variable that is made accessible to rules.",
    fields = ["name", "value", "type_name"],
)

def make_variable(name, value):
    """ A make variable that is made accessible to rules."""
    _check_is_nonempty_string(name, "name", "make_variable")
    _check_is_nonempty_string(value, "value", "make_variable")
    return MakeVariableInfo(
        name = name,
        value = value,
        type_name = "make_variable",
    )

FeatureSetInfo = provider(
    "A set of features.",
    fields = ["features", "type_name"],
)

def feature_set(features = []):
    """ A set of features.

    Used to support logical 'and' when specifying feature requirements in a
    feature.

    Args:
        features: A list of unordered feature names.

    Returns:
        A FeatureSetInfo provider.
    """
    _check_right_type(features, [], "features", "feature_set")
    return FeatureSetInfo(features = features, type_name = "feature_set")

WithFeatureSetInfo = provider(
    "A set of positive and negative features.",
    fields = ["features", "not_features", "type_name"],
)

def with_feature_set(features = [], not_features = []):
    """ A set of positive and negative features.

    This stanza will evaluate to true when every 'feature' is enabled, and
    every 'not_feature' is not enabled.

    Args:
        features: A list of feature names that need to be enabled.
        not_features: A list of feature names that need to not be enabled.

    Returns:
        A WithFeatureSetInfo provider.
    """
    _check_right_type(features, [], "features", "with_feature_set")
    _check_right_type(not_features, [], "not_features", "with_feature_set")
    return WithFeatureSetInfo(
        features = features,
        not_features = not_features,
        type_name = "with_feature_set",
    )

EnvSetInfo = provider(
    "Groups a set of environment variables to apply for certain actions.",
    fields = ["actions", "env_entries", "with_features", "type_name"],
)

def env_set(actions, env_entries = [], with_features = []):
    """ Groups a set of environment variables to apply for certain actions.

    EnvSet providers are passed to feature() and action_config(), to be applied to
    the actions they are specified for.

    Args:
        actions: A list of actions this env set applies to; each env set must
            specify at least one action.
        env_entries: A list of EnvEntry - the environment variables applied
            via this env set.
        with_features: A list of feature sets defining when this env set gets
            applied. The env set will be applied when any one of the feature
            sets evaluate to true. (That is, when when every 'feature' is
            enabled, and every 'not_feature' is not enabled.)
            If 'with_features' is omitted, the env set will be applied
            unconditionally for every action specified.

    Returns:
        An EnvSetInfo provider.
    """
    _check_is_nonempty_list(actions, "actions", "env_set")
    _check_right_type(env_entries, [], "env_entries", "env_set")
    _check_right_type(with_features, [], "with_features", "env_set")
    return EnvSetInfo(
        actions = actions,
        env_entries = env_entries,
        with_features = with_features,
        type_name = "env_set",
    )

FlagGroupInfo = provider(
    "A group of flags. Supports parametrization via variable expansion.",
    fields = [
        "flags",
        "flag_groups",
        "iterate_over",
        "expand_if_available",
        "expand_if_not_available",
        "expand_if_true",
        "expand_if_false",
        "expand_if_equal",
        "type_name",
    ],
)

def flag_group(
        flags = [],
        flag_groups = [],
        iterate_over = None,
        expand_if_available = None,
        expand_if_not_available = None,
        expand_if_true = None,
        expand_if_false = None,
        expand_if_equal = None):
    """ A group of flags. Supports parametrization via variable expansion.

    To expand a variable of list type, flag_group has to be annotated with
    `iterate_over` message. Then all nested flags or flag_groups will be
    expanded repeatedly for each element of the list.
    For example:
       flag_group(
         iterate_over = 'include_path',
         flags = ['-I', '%{include_path}'],
       )
    ... will get expanded to -I /to/path1 -I /to/path2 ... for each
    include_path /to/pathN.

    To expand a variable of structure type, use dot-notation, e.g.:
        flag_group(
          iterate_over = "libraries_to_link",
          flag_groups = [
            flag_group(
              iterate_over = "libraries_to_link.libraries",
              flags = ["-L%{libraries_to_link.libraries.directory}"],
            )
          ]
        )

    Flag groups can be nested; if they are, the flag group must only contain
    other flag groups (no flags) so the order is unambiguously specified.
    In order to expand a variable of nested lists, 'iterate_over' can be used.
    For example:
       flag_group (
         iterate_over = 'object_files',
         flag_groups = [
           flag_group (
             flags = ['--start-lib'],
           ),
           flag_group (
             iterate_over = 'object_files',
             flags = ['%{object_files}'],
           ),
           flag_group (
             flags = ['--end-lib'],
           )
        ]
     )
     ... will get expanded to
       --start-lib a1.o a2.o ... --end-lib --start-lib b1.o b2.o .. --end-lib
       with %{object_files} being a variable of nested list type
       [['a1.o', 'a2.o', ...], ['b1.o', 'b2.o', ...], ...].

    Args:
        flags: a string list, representing flags. Only one of flags and
            flag_groups can be set, as to avoid ambiguity.
        flag_groups: a list of FlagGroup entries. Only one of flags and
            flag_groups can be set, as to avoid ambiguity.
        iterate_over: a string, representing a variable of list type.
        expand_if_available: A build variable that needs to be available
            in order to expand the flag_group.
        expand_if_not_available: A build variable that needs to be
            unavailable in order for this flag_group to be expanded.
        expand_if_true: if set, this variable needs to evaluate to True in
            order for the flag_group to be expanded.
        expand_if_false: if set, this variable needs to evalate to False in
            order for the flag_group to be expanded.
        expand_if_equal: a VariableWithValue, the flag_group is expanded in
           case of equality.

    Returns:
        A FlagGroupInfo provider.
    """

    _check_right_type(flags, [], "flags", "flag_group")
    _check_right_type(flag_groups, [], "flag_groups", "flag_group")
    if len(flags) > 0 and len(flag_groups) > 0:
        fail("flag_group must not contain both a flag and another flag_group.")
    if len(flags) == 0 and len(flag_groups) == 0:
        fail("flag_group must contain either a list of flags or a list of flag_groups.")
    _check_is_none_or_right_type(expand_if_true, "string", "expand_if_true", "flag_group")
    _check_is_none_or_right_type(expand_if_false, "string", "expand_if_false", "flag_group")
    _check_is_none_or_right_type(expand_if_available, "string", "expand_if_available", "flag_group")
    _check_is_none_or_right_type(
        expand_if_not_available,
        "string",
        "expand_if_not_available",
        "flag_group",
    )
    _check_is_none_or_right_type(iterate_over, "string", "iterate_over", "flag_group")

    return FlagGroupInfo(
        flags = flags,
        flag_groups = flag_groups,
        iterate_over = iterate_over,
        expand_if_available = expand_if_available,
        expand_if_not_available = expand_if_not_available,
        expand_if_true = expand_if_true,
        expand_if_false = expand_if_false,
        expand_if_equal = expand_if_equal,
        type_name = "flag_group",
    )

FlagSetInfo = provider(
    "A set of flags to be expanded in the command line for specific actions.",
    fields = [
        "actions",
        "with_features",
        "flag_groups",
        "type_name",
    ],
)

def flag_set(
        actions = [],
        with_features = [],
        flag_groups = []):
    """ A set of flags to be expanded in the command line for specific actions.

    Args:
        actions: The actions this flag set applies to; each flag set must
            specify at least one action.
        with_features: A list of feature sets defining when this flag set gets
            applied. The flag set will be applied when any one of the feature
            sets evaluate to true. (That is, when when every 'feature' is
            enabled, and every 'not_feature' is not enabled.)
            If 'with_feature' is omitted, the flag set will be applied
            unconditionally for every action specified.
        flag_groups: A FlagGroup list - the flags applied via this flag set.

    Returns:
        A FlagSetInfo provider.
    """
    _check_right_type(actions, [], "actions", "flag_set")
    _check_right_type(with_features, [], "with_features", "flag_set")
    _check_right_type(flag_groups, [], "flag_groups", "flag_set")
    return FlagSetInfo(
        actions = actions,
        with_features = with_features,
        flag_groups = flag_groups,
        type_name = "flag_set",
    )

FeatureInfo = provider(
    "Contains all flag specifications for one feature.",
    fields = [
        "name",
        "enabled",
        "flag_sets",
        "env_sets",
        "requires",
        "implies",
        "provides",
        "type_name",
    ],
)

def feature(
        name,
        enabled = False,
        flag_sets = [],
        env_sets = [],
        requires = [],
        implies = [],
        provides = []):
    """ Contains all flag specifications for one feature.

    Args:
        name: The feature's name. It is possible to introduce a feature without
            a change to Bazel by adding a 'feature' section to the toolchain
            and adding the corresponding string as feature in the BUILD file.
        enabled: If 'True', this feature is enabled unless a rule type
            explicitly marks it as unsupported.
        flag_sets: A FlagSet list - If the given feature is enabled, the flag
            sets will be applied for the actions are specified for.
        env_sets: an EnvSet list - If the given feature is enabled, the env
            sets will be applied for the actions they are specified for.
        requires: A list of feature sets defining when this feature is
            supported by the  toolchain. The feature is supported if any of the
            feature sets fully apply, that is, when all features of a feature
            set are enabled.
            If 'requires' is omitted, the feature is supported independently of
            which other features are enabled.
            Use this for example to filter flags depending on the build mode
            enabled (opt / fastbuild / dbg).
        implies: A string list of features or action configs that are
            automatically enabled when this feature is enabled. If any of the
            implied features or action configs cannot be enabled, this feature
            will (silently) not be enabled either.
        provides: A list of names this feature conflicts with.
            A feature cannot be enabled if:
              - 'provides' contains the name of a different feature or action
            config that we want to enable.
              - 'provides' contains the same value as a 'provides' in a
            different feature or action config that we want to enable.
            Use this in order to ensure that incompatible features cannot be
            accidentally activated at the same time, leading to hard to
            diagnose compiler errors.

    Returns:
        A FeatureInfo provider.
    """
    _check_right_type(enabled, True, "enabled", "feature")
    _check_right_type(flag_sets, [], "flag_sets", "feature")
    _check_right_type(env_sets, [], "env_sets", "feature")
    _check_right_type(requires, [], "requires", "feature")
    _check_right_type(provides, [], "provides", "feature")
    _check_right_type(implies, [], "implies", "feature")
    return FeatureInfo(
        name = name,
        enabled = enabled,
        flag_sets = flag_sets,
        env_sets = env_sets,
        requires = requires,
        implies = implies,
        provides = provides,
        type_name = "feature",
    )

ToolPathInfo = provider(
    "Tool locations.",
    fields = ["name", "path", "type_name"],
)

def tool_path(name, path):
    """ Tool locations.

    Args:
        name: Name of the tool.
        path: Location of the tool; Can be absolute path (in case of non hermetic
            toolchain), or path relative to the cc_toolchain's package.

    Returns:
        A ToolPathInfo provider.

    Deprecated:
         Prefer specifying an ActionConfig for the action that needs the tool.
         TODO(b/27903698) migrate to ActionConfig.
    """
    _check_is_nonempty_string(name, "name", "tool_path")
    _check_is_nonempty_string(path, "path", "tool_path")
    return ToolPathInfo(name = name, path = path, type_name = "tool_path")

ToolInfo = provider(
    "Describes a tool associated with a crosstool action config.",
    fields = ["path", "with_features", "execution_requirements", "type_name"],
)

def tool(path, with_features = [], execution_requirements = []):
    """ Describes a tool associated with a crosstool action config.

    Args:
        path: Location of the tool; Can be absolute path (in case of non hermetic
            toolchain), or path relative to the cc_toolchain's package.
        with_features: A list of feature sets defining when this tool is
            applicable. The tool will used when any one of the feature sets
            evaluate to true. (That is, when when every 'feature' is enabled,
            and every 'not_feature' is not enabled.)
            If 'with_feature' is omitted, the tool will apply for any feature
            configuration.
        execution_requirements: Requirements on the execution environment for
            the execution of this tool, to be passed as out-of-band "hints" to
            the execution backend.
            Ex. "requires-darwin"

    Returns:
        A ToolInfo provider.
    """
    _check_is_nonempty_string(path, "path", "tool")
    _check_right_type(with_features, [], "with_features", "tool")
    _check_right_type(execution_requirements, [], "execution_requirements", "tool")
    return ToolInfo(
        path = path,
        with_features = with_features,
        execution_requirements = execution_requirements,
        type_name = "tool",
    )

ActionConfigInfo = provider(
    "Configuration of a Bazel action.",
    fields = [
        "config_name",
        "action_name",
        "enabled",
        "tools",
        "flag_sets",
        "implies",
        "type_name",
    ],
)

def action_config(
        action_name,
        enabled = False,
        tools = [],
        flag_sets = [],
        implies = []):
    """ Configuration of a Bazel action.

    An action config corresponds to a Bazel action, and allows selection of
    a tool based on activated features.
    Action config activation occurs by the same semantics as features: a
    feature can 'require' or 'imply' an action config in the same way that it
    would another feature.

    Args:
        action_name: The name of the Bazel action that this config applies to,
            ex. 'c-compile' or 'c-module-compile'.
        enabled: If 'True', this action is enabled unless a rule type
            explicitly marks it as unsupported.
        tools: The tool applied to the action will be the first Tool with a
            feature set that matches the feature configuration.  An error will
            be thrown if no tool matches a provided feature configuration - for
            that reason, it's a good idea to provide a default tool with an
            empty feature set.
        flag_sets: If the given action config is enabled, the flag sets will be
            applied to the corresponding action.
        implies: A list of features or action configs that are automatically
            enabled when this action config is enabled. If any of the implied
            features or action configs cannot be enabled, this action config
            will (silently) not be enabled either.

    Returns:
        An ActionConfigInfo provider.
    """
    _check_is_nonempty_string(action_name, "name", "action_config")
    _check_right_type(enabled, True, "enabled", "action_config")
    _check_right_type(tools, [], "tools", "action_config")
    _check_right_type(flag_sets, [], "flag_sets", "action_config")
    _check_right_type(implies, [], "implies", "action_config")
    return ActionConfigInfo(
        action_name = action_name,
        enabled = enabled,
        tools = tools,
        flag_sets = flag_sets,
        implies = implies,
        type_name = "action_config",
    )

ArtifactNamePatternInfo = provider(
    "The name for an artifact of a given category of input or output artifacts to an action.",
    fields = [
        "category_name",
        "prefix",
        "extension",
        "type_name",
    ],
)

def artifact_name_pattern(category_name, prefix, extension):
    """ The name for an artifact of a given category of input or output artifacts to an action.

    Args:
        category_name: The category of artifacts that this selection applies
            to. This field is compared against a list of categories defined
            in bazel. Example categories include "linked_output" or
            "debug_symbols". An error is thrown if no category is matched.
        prefix: The prefix for creating the artifact for this selection.
            Together with the extension it is used to create an artifact name
            based on the target name.
        extension: The extension for creating the artifact for this selection.
            Together with the prefix it is used to create an artifact name
            based on the target name.

    Returns:
        An ArtifactNamePatternInfo provider
    """
    _check_is_nonempty_string(category_name, "category_name", "artifact_name_pattern")
    _check_is_none_or_right_type(prefix, "", "prefix", "artifact_name_pattern")
    _check_is_none_or_right_type(extension, "", "extension", "artifact_name_pattern")
    return ArtifactNamePatternInfo(
        category_name = category_name,
        prefix = prefix,
        extension = extension,
        type_name = "artifact_name_pattern",
    )
