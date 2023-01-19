/*
Package crosstooltostarlarklib provides the Transform method
for conversion of a CROSSTOOL file to a Starlark rule.

https://github.com/bazelbuild/bazel/issues/5380
*/
package crosstooltostarlarklib

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
	"strings"

	crosstoolpb "third_party/com/github/bazelbuild/bazel/src/main/protobuf/crosstool_config_go_proto"
)

// CToolchainIdentifier is what we'll use to differ between CToolchains
// If a CToolchain can be distinguished from the other CToolchains
// by only one of the fields (eg if cpu is different for each CToolchain
// then only that field will be set.
type CToolchainIdentifier struct {
	cpu      string
	compiler string
}

// Writes the load statement for the cc_toolchain_config_lib
func getCcToolchainConfigHeader() string {
	return `load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "make_variable",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
`
}

var allCompileActions = []string{
	"c-compile",
	"c++-compile",
	"linkstamp-compile",
	"assemble",
	"preprocess-assemble",
	"c++-header-parsing",
	"c++-module-compile",
	"c++-module-codegen",
	"clif-match",
	"lto-backend",
}

var allCppCompileActions = []string{
	"c++-compile",
	"linkstamp-compile",
	"c++-header-parsing",
	"c++-module-compile",
	"c++-module-codegen",
	"clif-match",
}

var preprocessorCompileActions = []string{
	"c-compile",
	"c++-compile",
	"linkstamp-compile",
	"preprocess-assemble",
	"c++-header-parsing",
	"c++-module-compile",
	"clif-match",
}

var codegenCompileActions = []string{
	"c-compile",
	"c++-compile",
	"linkstamp-compile",
	"assemble",
	"preprocess-assemble",
	"c++-module-codegen",
	"lto-backend",
}

var allLinkActions = []string{
	"c++-link-executable",
	"c++-link-dynamic-library",
	"c++-link-nodeps-dynamic-library",
}

var actionNames = map[string]string{
	"c-compile":                       "ACTION_NAMES.c_compile",
	"c++-compile":                     "ACTION_NAMES.cpp_compile",
	"linkstamp-compile":               "ACTION_NAMES.linkstamp_compile",
	"cc-flags-make-variable":          "ACTION_NAMES.cc_flags_make_variable",
	"c++-module-codegen":              "ACTION_NAMES.cpp_module_codegen",
	"c++-header-parsing":              "ACTION_NAMES.cpp_header_parsing",
	"c++-module-compile":              "ACTION_NAMES.cpp_module_compile",
	"assemble":                        "ACTION_NAMES.assemble",
	"preprocess-assemble":             "ACTION_NAMES.preprocess_assemble",
	"lto-indexing":                    "ACTION_NAMES.lto_indexing",
	"lto-backend":                     "ACTION_NAMES.lto_backend",
	"c++-link-executable":             "ACTION_NAMES.cpp_link_executable",
	"c++-link-dynamic-library":        "ACTION_NAMES.cpp_link_dynamic_library",
	"c++-link-nodeps-dynamic-library": "ACTION_NAMES.cpp_link_nodeps_dynamic_library",
	"c++-link-static-library":         "ACTION_NAMES.cpp_link_static_library",
	"strip":                           "ACTION_NAMES.strip",
	"objc-compile":                    "ACTION_NAMES.objc_compile",
	"objc++-compile":                  "ACTION_NAMES.objcpp_compile",
	"clif-match":                      "ACTION_NAMES.clif_match",
// 	"objcopy_embed_data":              "ACTION_NAMES.objcopy_embed_data", // copybara-comment-this-out-please
// 	"ld_embed_data":                   "ACTION_NAMES.ld_embed_data",      // copybara-comment-this-out-please
}

func getLoadActionsStmt() string {
	return "load(\"@bazel_tools//tools/build_defs/cc:action_names.bzl\", \"ACTION_NAMES\")\n\n"
}

// Returns a map {toolchain_identifier : CToolchainIdentifier}
func toolchainToCToolchainIdentifier(
	crosstool *crosstoolpb.CrosstoolRelease) map[string]CToolchainIdentifier {
	cpuToCompiler := make(map[string][]string)
	compilerToCPU := make(map[string][]string)
	var cpus []string
	var compilers []string
	var identifiers []string
	res := make(map[string]CToolchainIdentifier)
	for _, cToolchain := range crosstool.GetToolchain() {
		cpu := cToolchain.GetTargetCpu()
		compiler := cToolchain.GetCompiler()

		cpuToCompiler[cpu] = append(cpuToCompiler[cpu], compiler)
		compilerToCPU[compiler] = append(compilerToCPU[compiler], cpu)

		cpus = append(cpus, cToolchain.GetTargetCpu())
		compilers = append(compilers, cToolchain.GetCompiler())
		identifiers = append(identifiers, cToolchain.GetToolchainIdentifier())
	}

	for i := range cpus {
		if len(cpuToCompiler[cpus[i]]) == 1 {
			// if cpu is unique among CToolchains, we don't need the compiler field
			res[identifiers[i]] = CToolchainIdentifier{cpu: cpus[i], compiler: ""}
		} else {
			res[identifiers[i]] = CToolchainIdentifier{
				cpu:      cpus[i],
				compiler: compilers[i],
			}
		}
	}
	return res
}

func getConditionStatementForCToolchainIdentifier(identifier CToolchainIdentifier) string {
	if identifier.compiler != "" {
		return fmt.Sprintf(
			"ctx.attr.cpu == \"%s\" and ctx.attr.compiler == \"%s\"",
			identifier.cpu,
			identifier.compiler)
	}
	return fmt.Sprintf("ctx.attr.cpu == \"%s\"", identifier.cpu)
}

func isArrayPrefix(prefix []string, arr []string) bool {
	if len(prefix) > len(arr) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		if arr[i] != prefix[i] {
			return false
		}
	}
	return true
}

func isAllCompileActions(actions []string) (bool, []string) {
	if isArrayPrefix(allCompileActions, actions) {
		return true, actions[len(allCompileActions):]
	}
	return false, actions
}

func isAllCppCompileActions(actions []string) (bool, []string) {
	if isArrayPrefix(allCppCompileActions, actions) {
		return true, actions[len(allCppCompileActions):]
	}
	return false, actions
}

func isPreprocessorCompileActions(actions []string) (bool, []string) {
	if isArrayPrefix(preprocessorCompileActions, actions) {
		return true, actions[len(preprocessorCompileActions):]
	}
	return false, actions
}

func isCodegenCompileActions(actions []string) (bool, []string) {
	if isArrayPrefix(codegenCompileActions, actions) {
		return true, actions[len(codegenCompileActions):]
	}
	return false, actions
}

func isAllLinkActions(actions []string) (bool, []string) {
	if isArrayPrefix(allLinkActions, actions) {
		return true, actions[len(allLinkActions):]
	}
	return false, actions
}

func getActionNames(actions []string) []string {
	var res []string
	for _, el := range actions {
		if name, ok := actionNames[el]; ok {
			res = append(res, name)
		} else {
			res = append(res, "\""+el+"\"")
		}
	}
	return res
}

func getListOfActions(name string, depth int) string {
	var res []string
	if name == "all_compile_actions" {
		res = getActionNames(allCompileActions)
	} else if name == "all_cpp_compile_actions" {
		res = getActionNames(allCppCompileActions)
	} else if name == "preprocessor_compile_actions" {
		res = getActionNames(preprocessorCompileActions)
	} else if name == "codegen_compile_actions" {
		res = getActionNames(codegenCompileActions)
	} else if name == "all_link_actions" {
		res = getActionNames(allLinkActions)
	}
	stmt := fmt.Sprintf("%s%s = %s\n\n", getTabs(depth),
		name, makeStringArr(res, depth /* isPlainString= */, false))
	return stmt
}

func processActions(actions []string, depth int) []string {
	var res []string
	var ok bool
	initLen := len(actions)
	if ok, actions = isAllCompileActions(actions); ok {
		res = append(res, "all_compile_actions")
	}
	if ok, actions = isAllCppCompileActions(actions); ok {
		res = append(res, "all_cpp_compile_actions")
	}
	if ok, actions = isPreprocessorCompileActions(actions); ok {
		res = append(res, "preprocessor_compile_actions")
	}
	if ok, actions = isCodegenCompileActions(actions); ok {
		res = append(res, "codegen_actions")
	}
	if ok, actions = isAllLinkActions(actions); ok {
		res = append(res, "all_link_actions")
	}
	if len(actions) != 0 {
		actions = getActionNames(actions)
		newDepth := depth + 1
		if len(actions) != initLen {
			newDepth++
		}
		res = append(res, makeStringArr(actions, newDepth /* isPlainString= */, false))
	}
	return res
}

func getUniqueValues(arr []string) []string {
	valuesSet := make(map[string]bool)
	for _, val := range arr {
		valuesSet[val] = true
	}
	var uniques []string
	for val, _ := range valuesSet {
		uniques = append(uniques, val)
	}
	sort.Strings(uniques)
	return uniques
}

func getRule(cToolchainIdentifiers map[string]CToolchainIdentifier,
	allowedCompilers []string) string {
	cpus := make(map[string]bool)
	shouldUseCompilerAttribute := false
	for _, val := range cToolchainIdentifiers {
		cpus[val.cpu] = true
		if val.compiler != "" {
			shouldUseCompilerAttribute = true
		}
	}

	var cpuValues []string
	for cpu := range cpus {
		cpuValues = append(cpuValues, cpu)
	}

	var args []string
	sort.Strings(cpuValues)
	args = append(args,
		fmt.Sprintf(
			`"cpu": attr.string(mandatory=True, values=["%s"]),`,
			strings.Join(cpuValues, "\", \"")))
	if shouldUseCompilerAttribute {
		// If there are two CToolchains that share the cpu we need the compiler attribute
		// for our cc_toolchain_config rule.
		allowedCompilers = getUniqueValues(allowedCompilers)
		args = append(args,
			fmt.Sprintf(`"compiler": attr.string(mandatory=True, values=["%s"]),`,
				strings.Join(allowedCompilers, "\", \"")))
	}
	return fmt.Sprintf(`cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        %s
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
`, strings.Join(args, "\n        "))
}

func getImplHeader() string {
	return "def _impl(ctx):\n"
}

func getStringStatement(crosstool *crosstoolpb.CrosstoolRelease,
	cToolchainIdentifiers map[string]CToolchainIdentifier, field string,
	depth int) string {

	identifiers := getToolchainIdentifiers(crosstool)
	var fieldValues []string
	if field == "toolchain_identifier" {
		fieldValues = getToolchainIdentifiers(crosstool)
	} else if field == "host_system_name" {
		fieldValues = getHostSystemNames(crosstool)
	} else if field == "target_system_name" {
		fieldValues = getTargetSystemNames(crosstool)
	} else if field == "target_cpu" {
		fieldValues = getTargetCpus(crosstool)
	} else if field == "target_libc" {
		fieldValues = getTargetLibcs(crosstool)
	} else if field == "compiler" {
		fieldValues = getCompilers(crosstool)
	} else if field == "abi_version" {
		fieldValues = getAbiVersions(crosstool)
	} else if field == "abi_libc_version" {
		fieldValues = getAbiLibcVersions(crosstool)
	} else if field == "cc_target_os" {
		fieldValues = getCcTargetOss(crosstool)
	} else if field == "builtin_sysroot" {
		fieldValues = getBuiltinSysroots(crosstool)
	}

	mappedValuesToIds := getMappedStringValuesToIdentifiers(identifiers, fieldValues)
	return getAssignmentStatement(field, mappedValuesToIds, crosstool,
		cToolchainIdentifiers, depth /* isPlainString= */, true /* shouldFail= */, true)
}

func getFeatures(crosstool *crosstoolpb.CrosstoolRelease) (
	map[string][]string, map[string]map[string][]string, error) {
	featureNameToFeature := make(map[string]map[string][]string)
	toolchainToFeatures := make(map[string][]string)
	for _, toolchain := range crosstool.GetToolchain() {
		id := toolchain.GetToolchainIdentifier()
		if len(toolchain.GetFeature()) == 0 {
			toolchainToFeatures[id] = []string{}
		}
		for _, feature := range toolchain.GetFeature() {
			featureName := strings.ToLower(feature.GetName()) + "_feature"
			featureName = strings.Replace(featureName, "+", "p", -1)
			featureName = strings.Replace(featureName, ".", "_", -1)
			featureName = strings.Replace(featureName, "-", "_", -1)
			stringFeature, err := parseFeature(feature, 1)
			if err != nil {
				return nil, nil, fmt.Errorf(
					"Error in feature '%s': %v", feature.GetName(), err)
			}
			if _, ok := featureNameToFeature[featureName]; !ok {
				featureNameToFeature[featureName] = make(map[string][]string)
			}
			featureNameToFeature[featureName][stringFeature] = append(
				featureNameToFeature[featureName][stringFeature], id)
			toolchainToFeatures[id] = append(toolchainToFeatures[id], featureName)
		}
	}
	return toolchainToFeatures, featureNameToFeature, nil
}

func getFeaturesDeclaration(crosstool *crosstoolpb.CrosstoolRelease,
	cToolchainIdentifiers map[string]CToolchainIdentifier,
	featureNameToFeature map[string]map[string][]string, depth int) string {
	var res []string
	for featureName, featureStringToID := range featureNameToFeature {
		res = append(res,
			getAssignmentStatement(
				featureName,
				featureStringToID,
				crosstool,
				cToolchainIdentifiers,
				depth,
				/* isPlainString= */ false,
				/* shouldFail= */ false))
	}
	return strings.Join(res, "")
}

func getFeaturesStmt(cToolchainIdentifiers map[string]CToolchainIdentifier,
	toolchainToFeatures map[string][]string, depth int) string {
	var res []string
	arrToIdentifier := make(map[string][]string)
	for id, features := range toolchainToFeatures {
		arrayString := strings.Join(features, "{arrayFieldDelimiter}")
		arrToIdentifier[arrayString] = append(arrToIdentifier[arrayString], id)
	}
	res = append(res,
		getStringArrStatement(
			"features",
			arrToIdentifier,
			cToolchainIdentifiers,
			depth,
			/* isPlainString= */ false))
	return strings.Join(res, "\n")
}

func getActions(crosstool *crosstoolpb.CrosstoolRelease) (
	map[string][]string, map[string]map[string][]string, error) {
	actionNameToAction := make(map[string]map[string][]string)
	toolchainToActions := make(map[string][]string)
	for _, toolchain := range crosstool.GetToolchain() {
		id := toolchain.GetToolchainIdentifier()
		var actionName string
		if len(toolchain.GetActionConfig()) == 0 {
			toolchainToActions[id] = []string{}
		}
		for _, action := range toolchain.GetActionConfig() {
			if aName, ok := actionNames[action.GetActionName()]; ok {
				actionName = aName
			} else {
				actionName = strings.ToLower(action.GetActionName())
				actionName = strings.Replace(actionName, "+", "p", -1)
				actionName = strings.Replace(actionName, ".", "_", -1)
				actionName = strings.Replace(actionName, "-", "_", -1)
			}
			stringAction, err := parseAction(action, 1)
			if err != nil {
				return nil, nil, fmt.Errorf(
					"Error in action_config '%s': %v", action.GetActionName(), err)
			}
			if _, ok := actionNameToAction[actionName]; !ok {
				actionNameToAction[actionName] = make(map[string][]string)
			}
			actionNameToAction[actionName][stringAction] = append(
				actionNameToAction[actionName][stringAction], id)
			toolchainToActions[id] = append(
				toolchainToActions[id],
				strings.TrimPrefix(strings.ToLower(actionName), "action_names.")+"_action")
		}
	}
	return toolchainToActions, actionNameToAction, nil
}

func getActionConfigsDeclaration(
	crosstool *crosstoolpb.CrosstoolRelease,
	cToolchainIdentifiers map[string]CToolchainIdentifier,
	actionNameToAction map[string]map[string][]string, depth int) string {
	var res []string
	for actionName, actionStringToID := range actionNameToAction {
		variableName := strings.TrimPrefix(strings.ToLower(actionName), "action_names.") + "_action"
		res = append(res,
			getAssignmentStatement(
				variableName,
				actionStringToID,
				crosstool,
				cToolchainIdentifiers,
				depth,
				/* isPlainString= */ false,
				/* shouldFail= */ false))
	}
	return strings.Join(res, "")
}

func getActionConfigsStmt(
	cToolchainIdentifiers map[string]CToolchainIdentifier,
	toolchainToActions map[string][]string, depth int) string {
	var res []string
	arrToIdentifier := make(map[string][]string)
	for id, actions := range toolchainToActions {
		var arrayString string
		arrayString = strings.Join(actions, "{arrayFieldDelimiter}")
		arrToIdentifier[arrayString] = append(arrToIdentifier[arrayString], id)
	}
	res = append(res,
		getStringArrStatement(
			"action_configs",
			arrToIdentifier,
			cToolchainIdentifiers,
			depth,
			/* isPlainString= */ false))
	return strings.Join(res, "\n")
}

func parseAction(action *crosstoolpb.CToolchain_ActionConfig, depth int) (string, error) {
	actionName := action.GetActionName()
	aName := ""
	if val, ok := actionNames[actionName]; ok {
		aName = val
	} else {
		aName = "\"" + action.GetActionName() + "\""
	}
	name := fmt.Sprintf("action_name = %s", aName)
	fields := []string{name}
	if action.GetEnabled() {
		fields = append(fields, "enabled = True")
	}
	if len(action.GetFlagSet()) != 0 {
		flagSets, err := parseFlagSets(action.GetFlagSet(), depth+1)
		if err != nil {
			return "", err
		}
		fields = append(fields, "flag_sets = "+flagSets)
	}
	if len(action.GetImplies()) != 0 {
		implies := "implies = " +
			makeStringArr(action.GetImplies(), depth+1 /* isPlainString= */, true)
		fields = append(fields, implies)
	}
	if len(action.GetTool()) != 0 {
		tools := "tools = " + parseTools(action.GetTool(), depth+1)
		fields = append(fields, tools)
	}
	return createObject("action_config", fields, depth), nil
}

func getStringArrStatement(attr string, arrValToIds map[string][]string,
	cToolchainIdentifiers map[string]CToolchainIdentifier, depth int, plainString bool) string {
	var b bytes.Buffer
	if len(arrValToIds) == 0 {
		b.WriteString(fmt.Sprintf("%s%s = []\n", getTabs(depth), attr))
	} else if len(arrValToIds) == 1 {
		for value := range arrValToIds {
			var arr []string
			if value == "" {
				arr = []string{}
			} else if value == "None" {
				b.WriteString(fmt.Sprintf("%s%s = None\n", getTabs(depth), attr))
				break
			} else {
				arr = strings.Split(value, "{arrayFieldDelimiter}")
			}
			b.WriteString(
				fmt.Sprintf(
					"%s%s = %s\n",
					getTabs(depth),
					attr,
					makeStringArr(arr, depth+1, plainString)))
			break
		}
	} else {
		first := true
		var keys []string
		for k := range arrValToIds {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, value := range keys {
			ids := arrValToIds[value]
			branch := "elif"
			if first {
				branch = "if"
			}
			first = false
			var arr []string
			if value == "" {
				arr = []string{}
			} else if value == "None" {
				b.WriteString(
					getIfStatement(
						branch, ids, attr, "None", cToolchainIdentifiers,
						depth /* isPlainString= */, true))
				continue
			} else {
				arr = strings.Split(value, "{arrayFieldDelimiter}")
			}
			b.WriteString(
				getIfStatement(branch, ids, attr,
					makeStringArr(arr, depth+1, plainString),
					cToolchainIdentifiers, depth /* isPlainString= */, false))
		}
		b.WriteString(fmt.Sprintf("%selse:\n%sfail(\"Unreachable\")\n", getTabs(depth), getTabs(depth+1)))
	}
	b.WriteString("\n")
	return b.String()
}

func getStringArr(crosstool *crosstoolpb.CrosstoolRelease,
	cToolchainIdentifiers map[string]CToolchainIdentifier, attr string, depth int) string {
	var res []string
	arrToIdentifier := make(map[string][]string)
	for _, toolchain := range crosstool.GetToolchain() {
		id := toolchain.GetToolchainIdentifier()
		arrayString := strings.Join(getArrField(attr, toolchain), "{arrayFieldDelimiter}")
		arrToIdentifier[arrayString] = append(arrToIdentifier[arrayString], id)
	}
	statement := getStringArrStatement(attr, arrToIdentifier, cToolchainIdentifiers, depth /* isPlainString= */, true)
	res = append(res, statement)
	return strings.Join(res, "\n")
}

func getArrField(attr string, toolchain *crosstoolpb.CToolchain) []string {
	var arr []string
	if attr == "cxx_builtin_include_directories" {
		arr = toolchain.GetCxxBuiltinIncludeDirectory()
	}
	return arr
}

func getTabs(depth int) string {
	var res string
	for i := 0; i < depth; i++ {
		res = res + "    "
	}
	return res
}

func createObject(objtype string, fields []string, depth int) string {
	if len(fields) == 0 {
		return objtype + "()"
	}
	singleLine := objtype + "(" + strings.Join(fields, ", ") + ")"
	if len(singleLine) < 60 {
		return singleLine
	}
	return objtype +
		"(\n" +
		getTabs(depth+1) +
		strings.Join(fields, ",\n"+getTabs(depth+1)) +
		",\n" + getTabs(depth) +
		")"
}

func getArtifactNamePatterns(crosstool *crosstoolpb.CrosstoolRelease,
	cToolchainIdentifiers map[string]CToolchainIdentifier, depth int) string {
	var res []string
	artifactToIds := make(map[string][]string)
	for _, toolchain := range crosstool.GetToolchain() {
		artifactNamePatterns := parseArtifactNamePatterns(
			toolchain.GetArtifactNamePattern(),
			depth)
		artifactToIds[artifactNamePatterns] = append(
			artifactToIds[artifactNamePatterns],
			toolchain.GetToolchainIdentifier())
	}
	res = append(res,
		getAssignmentStatement(
			"artifact_name_patterns",
			artifactToIds,
			crosstool,
			cToolchainIdentifiers,
			depth,
			/* isPlainString= */ false,
			/* shouldFail= */ true))
	return strings.Join(res, "\n")
}

func parseArtifactNamePatterns(
	artifactNamePatterns []*crosstoolpb.CToolchain_ArtifactNamePattern, depth int) string {
	var res []string
	for _, pattern := range artifactNamePatterns {
		res = append(res, parseArtifactNamePattern(pattern, depth+1))
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseArtifactNamePattern(
	artifactNamePattern *crosstoolpb.CToolchain_ArtifactNamePattern, depth int) string {
	categoryName := fmt.Sprintf("category_name = \"%s\"", artifactNamePattern.GetCategoryName())
	prefix := fmt.Sprintf("prefix = \"%s\"", artifactNamePattern.GetPrefix())
	extension := fmt.Sprintf("extension = \"%s\"", artifactNamePattern.GetExtension())
	fields := []string{categoryName, prefix, extension}
	return createObject("artifact_name_pattern", fields, depth)
}

func parseFeature(feature *crosstoolpb.CToolchain_Feature, depth int) (string, error) {
	name := fmt.Sprintf("name = \"%s\"", feature.GetName())

	fields := []string{name}
	if feature.GetEnabled() {
		fields = append(fields, "enabled = True")
	}

	if len(feature.GetFlagSet()) > 0 {
		flagSets, err := parseFlagSets(feature.GetFlagSet(), depth+1)
		if err != nil {
			return "", err
		}
		fields = append(fields, "flag_sets = "+flagSets)
	}
	if len(feature.GetEnvSet()) > 0 {
		envSets := "env_sets = " + parseEnvSets(feature.GetEnvSet(), depth+1)
		fields = append(fields, envSets)
	}
	if len(feature.GetRequires()) > 0 {
		requires := "requires = " + parseFeatureSets(feature.GetRequires(), depth+1)
		fields = append(fields, requires)
	}
	if len(feature.GetImplies()) > 0 {
		implies := "implies = " +
			makeStringArr(feature.GetImplies(), depth+1 /* isPlainString= */, true)
		fields = append(fields, implies)
	}
	if len(feature.GetProvides()) > 0 {
		provides := "provides = " +
			makeStringArr(feature.GetProvides(), depth+1 /* isPlainString= */, true)
		fields = append(fields, provides)
	}
	return createObject("feature", fields, depth), nil
}

func parseFlagSets(flagSets []*crosstoolpb.CToolchain_FlagSet, depth int) (string, error) {
	var res []string
	for _, flagSet := range flagSets {
		parsedFlagset, err := parseFlagSet(flagSet, depth+1)
		if err != nil {
			return "", err
		}
		res = append(res, parsedFlagset)
	}
	return makeStringArr(res, depth /* isPlainString= */, false), nil
}

func parseFlagSet(flagSet *crosstoolpb.CToolchain_FlagSet, depth int) (string, error) {
	var fields []string
	if len(flagSet.GetAction()) > 0 {
		actionArr := processActions(flagSet.GetAction(), depth)
		actions := "actions = " + strings.Join(actionArr, " +\n"+getTabs(depth+2))
		fields = append(fields, actions)
	}
	if len(flagSet.GetFlagGroup()) > 0 {
		flagGroups, err := parseFlagGroups(flagSet.GetFlagGroup(), depth+1)
		if err != nil {
			return "", err
		}
		fields = append(fields, "flag_groups = "+flagGroups)
	}
	if len(flagSet.GetWithFeature()) > 0 {
		withFeatures := "with_features = " +
			parseWithFeatureSets(flagSet.GetWithFeature(), depth+1)
		fields = append(fields, withFeatures)
	}
	return createObject("flag_set", fields, depth), nil
}

func parseFlagGroups(flagGroups []*crosstoolpb.CToolchain_FlagGroup, depth int) (string, error) {
	var res []string
	for _, flagGroup := range flagGroups {
		flagGroupString, err := parseFlagGroup(flagGroup, depth+1)
		if err != nil {
			return "", err
		}
		res = append(res, flagGroupString)
	}
	return makeStringArr(res, depth /* isPlainString= */, false), nil
}

func parseFlagGroup(flagGroup *crosstoolpb.CToolchain_FlagGroup, depth int) (string, error) {
	var res []string
	if len(flagGroup.GetFlag()) != 0 {
		res = append(res, "flags = "+makeStringArr(flagGroup.GetFlag(), depth+1, true))
	}
	if flagGroup.GetIterateOver() != "" {
		res = append(res, fmt.Sprintf("iterate_over = \"%s\"", flagGroup.GetIterateOver()))
	}
	if len(flagGroup.GetFlagGroup()) != 0 {
		flagGroupString, err := parseFlagGroups(flagGroup.GetFlagGroup(), depth+1)
		if err != nil {
			return "", err
		}
		res = append(res, "flag_groups = "+flagGroupString)
	}
	if len(flagGroup.GetExpandIfAllAvailable()) > 1 {
		return "", errors.New("Flag group must not have more than one 'expand_if_all_available' field")
	}
	if len(flagGroup.GetExpandIfAllAvailable()) != 0 {
		res = append(res,
			fmt.Sprintf(
				"expand_if_available = \"%s\"",
				flagGroup.GetExpandIfAllAvailable()[0]))
	}
	if len(flagGroup.GetExpandIfNoneAvailable()) > 1 {
		return "", errors.New("Flag group must not have more than one 'expand_if_none_available' field")
	}
	if len(flagGroup.GetExpandIfNoneAvailable()) != 0 {
		res = append(res,
			fmt.Sprintf(
				"expand_if_not_available = \"%s\"",
				flagGroup.GetExpandIfNoneAvailable()[0]))
	}
	if flagGroup.GetExpandIfTrue() != "" {
		res = append(res, fmt.Sprintf("expand_if_true = \"%s\"",
			flagGroup.GetExpandIfTrue()))
	}
	if flagGroup.GetExpandIfFalse() != "" {
		res = append(res, fmt.Sprintf("expand_if_false = \"%s\"",
			flagGroup.GetExpandIfFalse()))
	}
	if flagGroup.GetExpandIfEqual() != nil {
		res = append(res,
			"expand_if_equal = "+parseVariableWithValue(
				flagGroup.GetExpandIfEqual(), depth+1))
	}
	return createObject("flag_group", res, depth), nil
}

func parseVariableWithValue(variable *crosstoolpb.CToolchain_VariableWithValue, depth int) string {
	variableName := fmt.Sprintf("name = \"%s\"", variable.GetVariable())
	value := fmt.Sprintf("value = \"%s\"", variable.GetValue())
	return createObject("variable_with_value", []string{variableName, value}, depth)
}

func getToolPaths(crosstool *crosstoolpb.CrosstoolRelease,
	cToolchainIdentifiers map[string]CToolchainIdentifier, depth int) string {
	var res []string
	toolPathsToIds := make(map[string][]string)
	for _, toolchain := range crosstool.GetToolchain() {
		toolPaths := parseToolPaths(toolchain.GetToolPath(), depth)
		toolPathsToIds[toolPaths] = append(
			toolPathsToIds[toolPaths],
			toolchain.GetToolchainIdentifier())
	}
	res = append(res,
		getAssignmentStatement(
			"tool_paths",
			toolPathsToIds,
			crosstool,
			cToolchainIdentifiers,
			depth,
			/* isPlainString= */ false,
			/* shouldFail= */ true))
	return strings.Join(res, "\n")
}

func parseToolPaths(toolPaths []*crosstoolpb.ToolPath, depth int) string {
	var res []string
	for _, toolPath := range toolPaths {
		res = append(res, parseToolPath(toolPath, depth+1))
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseToolPath(toolPath *crosstoolpb.ToolPath, depth int) string {
	name := fmt.Sprintf("name = \"%s\"", toolPath.GetName())
	path := toolPath.GetPath()
	if path == "" {
		path = "NOT_USED"
	}
	path = fmt.Sprintf("path = \"%s\"", path)
	return createObject("tool_path", []string{name, path}, depth)
}

func getMakeVariables(crosstool *crosstoolpb.CrosstoolRelease,
	cToolchainIdentifiers map[string]CToolchainIdentifier, depth int) string {
	var res []string
	makeVariablesToIds := make(map[string][]string)
	for _, toolchain := range crosstool.GetToolchain() {
		makeVariables := parseMakeVariables(toolchain.GetMakeVariable(), depth)
		makeVariablesToIds[makeVariables] = append(
			makeVariablesToIds[makeVariables],
			toolchain.GetToolchainIdentifier())
	}
	res = append(res,
		getAssignmentStatement(
			"make_variables",
			makeVariablesToIds,
			crosstool,
			cToolchainIdentifiers,
			depth,
			/* isPlainString= */ false,
			/* shouldFail= */ true))
	return strings.Join(res, "\n")
}

func parseMakeVariables(makeVariables []*crosstoolpb.MakeVariable, depth int) string {
	var res []string
	for _, makeVariable := range makeVariables {
		res = append(res, parseMakeVariable(makeVariable, depth+1))
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseMakeVariable(makeVariable *crosstoolpb.MakeVariable, depth int) string {
	name := fmt.Sprintf("name = \"%s\"", makeVariable.GetName())
	value := fmt.Sprintf("value = \"%s\"", makeVariable.GetValue())
	return createObject("make_variable", []string{name, value}, depth)
}

func parseTools(tools []*crosstoolpb.CToolchain_Tool, depth int) string {
	var res []string
	for _, tool := range tools {
		res = append(res, parseTool(tool, depth+1))
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseTool(tool *crosstoolpb.CToolchain_Tool, depth int) string {
	toolPath := "path = \"NOT_USED\""
	if tool.GetToolPath() != "" {
		toolPath = fmt.Sprintf("path = \"%s\"", tool.GetToolPath())
	}
	fields := []string{toolPath}
	if len(tool.GetWithFeature()) != 0 {
		withFeatures := "with_features = " + parseWithFeatureSets(tool.GetWithFeature(), depth+1)
		fields = append(fields, withFeatures)
	}
	if len(tool.GetExecutionRequirement()) != 0 {
		executionRequirements := "execution_requirements = " +
			makeStringArr(tool.GetExecutionRequirement(), depth+1 /* isPlainString= */, true)
		fields = append(fields, executionRequirements)
	}
	return createObject("tool", fields, depth)
}

func parseEnvEntries(envEntries []*crosstoolpb.CToolchain_EnvEntry, depth int) string {
	var res []string
	for _, envEntry := range envEntries {
		res = append(res, parseEnvEntry(envEntry, depth+1))
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseEnvEntry(envEntry *crosstoolpb.CToolchain_EnvEntry, depth int) string {
	key := fmt.Sprintf("key = \"%s\"", envEntry.GetKey())
	value := fmt.Sprintf("value = \"%s\"", envEntry.GetValue())
	return createObject("env_entry", []string{key, value}, depth)
}

func parseWithFeatureSets(withFeatureSets []*crosstoolpb.CToolchain_WithFeatureSet,
	depth int) string {
	var res []string
	for _, withFeature := range withFeatureSets {
		res = append(res, parseWithFeatureSet(withFeature, depth+1))
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseWithFeatureSet(withFeature *crosstoolpb.CToolchain_WithFeatureSet,
	depth int) string {
	var fields []string
	if len(withFeature.GetFeature()) != 0 {
		features := "features = " +
			makeStringArr(withFeature.GetFeature(), depth+1 /* isPlainString= */, true)
		fields = append(fields, features)
	}
	if len(withFeature.GetNotFeature()) != 0 {
		notFeatures := "not_features = " +
			makeStringArr(withFeature.GetNotFeature(), depth+1 /* isPlainString= */, true)
		fields = append(fields, notFeatures)
	}
	return createObject("with_feature_set", fields, depth)
}

func parseEnvSets(envSets []*crosstoolpb.CToolchain_EnvSet, depth int) string {
	var res []string
	for _, envSet := range envSets {
		envSetString := parseEnvSet(envSet, depth+1)
		res = append(res, envSetString)
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseEnvSet(envSet *crosstoolpb.CToolchain_EnvSet, depth int) string {
	actionsStatement := processActions(envSet.GetAction(), depth)
	actions := "actions = " + strings.Join(actionsStatement, " +\n"+getTabs(depth+2))
	fields := []string{actions}
	if len(envSet.GetEnvEntry()) != 0 {
		envEntries := "env_entries = " + parseEnvEntries(envSet.GetEnvEntry(), depth+1)
		fields = append(fields, envEntries)
	}
	if len(envSet.GetWithFeature()) != 0 {
		withFeatures := "with_features = " + parseWithFeatureSets(envSet.GetWithFeature(), depth+1)
		fields = append(fields, withFeatures)
	}
	return createObject("env_set", fields, depth)
}

func parseFeatureSets(featureSets []*crosstoolpb.CToolchain_FeatureSet, depth int) string {
	var res []string
	for _, featureSet := range featureSets {
		res = append(res, parseFeatureSet(featureSet, depth+1))
	}
	return makeStringArr(res, depth /* isPlainString= */, false)
}

func parseFeatureSet(featureSet *crosstoolpb.CToolchain_FeatureSet, depth int) string {
	features := "features = " +
		makeStringArr(featureSet.GetFeature(), depth+1 /* isPlainString= */, true)
	return createObject("feature_set", []string{features}, depth)
}

// Takes in a list of string elements and returns a string that represents
// an array :
//     [
//         "element1",
//         "element2",
//     ]
// The isPlainString argument tells us whether the input elements should be
// treated as string (eg, flags), or not (eg, variable names)
func makeStringArr(arr []string, depth int, isPlainString bool) string {
	if len(arr) == 0 {
		return "[]"
	}
	var escapedArr []string
	for _, el := range arr {
		if isPlainString {
			escapedArr = append(escapedArr, strings.Replace(el, "\"", "\\\"", -1))
		} else {
			escapedArr = append(escapedArr, el)
		}
	}
	addQuote := ""
	if isPlainString {
		addQuote = "\""
	}
	singleLine := "[" + addQuote + strings.Join(escapedArr, addQuote+", "+addQuote) + addQuote + "]"
	if len(singleLine) < 60 {
		return singleLine
	}
	return "[\n" +
		getTabs(depth+1) +
		addQuote +
		strings.Join(escapedArr, addQuote+",\n"+getTabs(depth+1)+addQuote) +
		addQuote +
		",\n" +
		getTabs(depth) +
		"]"
}

// Returns a string that represents a value assignment
// (eg if ctx.attr.cpu == "linux":
//         compiler = "llvm"
//     elif ctx.attr.cpu == "windows":
//         compiler = "mingw"
//     else:
//         fail("Unreachable")
func getAssignmentStatement(field string, valToIds map[string][]string,
	crosstool *crosstoolpb.CrosstoolRelease,
	toCToolchainIdentifier map[string]CToolchainIdentifier,
	depth int, isPlainString, shouldFail bool) string {
	var b bytes.Buffer
	if len(valToIds) <= 1 {
		// if there is only one possible value for this field, we don't need if statements
		for val := range valToIds {
			if val != "None" && isPlainString {
				val = "\"" + val + "\""
			}
			b.WriteString(fmt.Sprintf("%s%s = %s\n", getTabs(depth), field, val))
			break
		}
	} else {
		first := true
		var keys []string
		for k := range valToIds {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, value := range keys {
			ids := valToIds[value]
			branch := "elif"
			if first {
				branch = "if"
			}
			b.WriteString(
				getIfStatement(branch, ids, field, value,
					toCToolchainIdentifier, depth, isPlainString))
			first = false
		}
		if shouldFail {
			b.WriteString(
				fmt.Sprintf(
					"%selse:\n%sfail(\"Unreachable\")\n",
					getTabs(depth), getTabs(depth+1)))
		} else {
			b.WriteString(
				fmt.Sprintf(
					"%selse:\n%s%s = None\n",
					getTabs(depth), getTabs(depth+1), field))
		}
	}
	b.WriteString("\n")
	return b.String()
}

func getCPUToCompilers(identifiers []CToolchainIdentifier) map[string][]string {
	res := make(map[string][]string)
	for _, identifier := range identifiers {
		if identifier.compiler != "" {
			res[identifier.cpu] = append(res[identifier.cpu], identifier.compiler)
		}
	}
	return res
}

func getIfStatement(ifOrElseIf string, identifiers []string, field, val string,
	toCToolchainIdentifier map[string]CToolchainIdentifier, depth int,
	isPlainString bool) string {
	usedStmts := make(map[string]bool)
	if val != "None" && isPlainString {
		val = "\"" + val + "\""
	}
	var cToolchainIdentifiers []CToolchainIdentifier
	for _, value := range toCToolchainIdentifier {
		cToolchainIdentifiers = append(cToolchainIdentifiers, value)
	}
	cpuToCompilers := getCPUToCompilers(cToolchainIdentifiers)
	countCpus := make(map[string]int)
	var conditions []string
	for _, id := range identifiers {
		identifier := toCToolchainIdentifier[id]
		stmt := getConditionStatementForCToolchainIdentifier(identifier)
		if _, ok := usedStmts[stmt]; !ok {
			conditions = append(conditions, stmt)
			usedStmts[stmt] = true
			if identifier.compiler != "" {
				countCpus[identifier.cpu]++
			}
		}
	}

	var compressedConditions []string
	usedStmtsOptimized := make(map[string]bool)
	for _, id := range identifiers {
		identifier := toCToolchainIdentifier[id]
		var stmt string
		if _, ok := countCpus[identifier.cpu]; ok {
			if countCpus[identifier.cpu] == len(cpuToCompilers[identifier.cpu]) {
				stmt = getConditionStatementForCToolchainIdentifier(
					CToolchainIdentifier{cpu: identifier.cpu, compiler: ""})
			} else {
				stmt = getConditionStatementForCToolchainIdentifier(identifier)
			}
		} else {
			stmt = getConditionStatementForCToolchainIdentifier(identifier)
		}
		if _, ok := usedStmtsOptimized[stmt]; !ok {
			compressedConditions = append(compressedConditions, stmt)
			usedStmtsOptimized[stmt] = true
		}
	}

	sort.Strings(compressedConditions)
	val = strings.Join(strings.Split(val, "\n"+getTabs(depth)), "\n"+getTabs(depth+1))
	return fmt.Sprintf(`%s%s %s:
%s%s = %s
`, getTabs(depth),
		ifOrElseIf,
		"("+strings.Join(compressedConditions, "\n"+getTabs(depth+1)+"or ")+")",
		getTabs(depth+1),
		field,
		val)
}

func getToolchainIdentifiers(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetToolchainIdentifier())
	}
	return res
}

func getHostSystemNames(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetHostSystemName())
	}
	return res
}

func getTargetSystemNames(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetTargetSystemName())
	}
	return res
}

func getTargetCpus(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetTargetCpu())
	}
	return res
}

func getTargetLibcs(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetTargetLibc())
	}
	return res
}

func getCompilers(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetCompiler())
	}
	return res
}

func getAbiVersions(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetAbiVersion())
	}
	return res
}

func getAbiLibcVersions(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		res = append(res, toolchain.GetAbiLibcVersion())
	}
	return res
}

func getCcTargetOss(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		targetOS := "None"
		if toolchain.GetCcTargetOs() != "" {
			targetOS = toolchain.GetCcTargetOs()
		}
		res = append(res, targetOS)
	}
	return res
}

func getBuiltinSysroots(crosstool *crosstoolpb.CrosstoolRelease) []string {
	var res []string
	for _, toolchain := range crosstool.GetToolchain() {
		sysroot := "None"
		if toolchain.GetBuiltinSysroot() != "" {
			sysroot = toolchain.GetBuiltinSysroot()
		}
		res = append(res, sysroot)
	}
	return res
}

func getMappedStringValuesToIdentifiers(identifiers, fields []string) map[string][]string {
	res := make(map[string][]string)
	for i := range identifiers {
		res[fields[i]] = append(res[fields[i]], identifiers[i])
	}
	return res
}

func getReturnStatement() string {
	return `
    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features,
            action_configs = action_configs,
            artifact_name_patterns = artifact_name_patterns,
            cxx_builtin_include_directories = cxx_builtin_include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = host_system_name,
            target_system_name = target_system_name,
            target_cpu = target_cpu,
            target_libc = target_libc,
            compiler = compiler,
            abi_version = abi_version,
            abi_libc_version = abi_libc_version,
            tool_paths = tool_paths,
            make_variables = make_variables,
            builtin_sysroot = builtin_sysroot,
            cc_target_os = cc_target_os
        ),
        DefaultInfo(
            executable = out,
        ),
    ]
`
}

// Transform writes a cc_toolchain_config rule functionally equivalent to the
// CROSSTOOL file.
func Transform(crosstool *crosstoolpb.CrosstoolRelease) (string, error) {
	var b bytes.Buffer

	cToolchainIdentifiers := toolchainToCToolchainIdentifier(crosstool)

	toolchainToFeatures, featureNameToFeature, err := getFeatures(crosstool)
	if err != nil {
		return "", err
	}

	toolchainToActions, actionNameToAction, err := getActions(crosstool)
	if err != nil {
		return "", err
	}

	header := getCcToolchainConfigHeader()
	if _, err := b.WriteString(header); err != nil {
		return "", err
	}

	loadActionsStmt := getLoadActionsStmt()
	if _, err := b.WriteString(loadActionsStmt); err != nil {
		return "", err
	}

	implHeader := getImplHeader()
	if _, err := b.WriteString(implHeader); err != nil {
		return "", err
	}

	stringFields := []string{
		"toolchain_identifier",
		"host_system_name",
		"target_system_name",
		"target_cpu",
		"target_libc",
		"compiler",
		"abi_version",
		"abi_libc_version",
		"cc_target_os",
		"builtin_sysroot",
	}

	for _, stringField := range stringFields {
		stmt := getStringStatement(crosstool, cToolchainIdentifiers, stringField, 1)
		if _, err := b.WriteString(stmt); err != nil {
			return "", err
		}
	}

	listsOfActions := []string{
		"all_compile_actions",
		"all_cpp_compile_actions",
		"preprocessor_compile_actions",
		"codegen_compile_actions",
		"all_link_actions",
	}

	for _, listOfActions := range listsOfActions {
		actions := getListOfActions(listOfActions, 1)
		if _, err := b.WriteString(actions); err != nil {
			return "", err
		}
	}

	actionConfigDeclaration := getActionConfigsDeclaration(
		crosstool, cToolchainIdentifiers, actionNameToAction, 1)
	if _, err := b.WriteString(actionConfigDeclaration); err != nil {
		return "", err
	}

	actionConfigStatement := getActionConfigsStmt(
		cToolchainIdentifiers, toolchainToActions, 1)
	if _, err := b.WriteString(actionConfigStatement); err != nil {
		return "", err
	}

	featureDeclaration := getFeaturesDeclaration(
		crosstool, cToolchainIdentifiers, featureNameToFeature, 1)
	if _, err := b.WriteString(featureDeclaration); err != nil {
		return "", err
	}

	featuresStatement := getFeaturesStmt(
		cToolchainIdentifiers, toolchainToFeatures, 1)
	if _, err := b.WriteString(featuresStatement); err != nil {
		return "", err
	}

	includeDirectories := getStringArr(
		crosstool, cToolchainIdentifiers, "cxx_builtin_include_directories", 1)
	if _, err := b.WriteString(includeDirectories); err != nil {
		return "", err
	}

	artifactNamePatterns := getArtifactNamePatterns(
		crosstool, cToolchainIdentifiers, 1)
	if _, err := b.WriteString(artifactNamePatterns); err != nil {
		return "", err
	}

	makeVariables := getMakeVariables(crosstool, cToolchainIdentifiers, 1)
	if _, err := b.WriteString(makeVariables); err != nil {
		return "", err
	}

	toolPaths := getToolPaths(crosstool, cToolchainIdentifiers, 1)
	if _, err := b.WriteString(toolPaths); err != nil {
		return "", err
	}

	if _, err := b.WriteString(getReturnStatement()); err != nil {
		return "", err
	}

	rule := getRule(cToolchainIdentifiers, getCompilers(crosstool))
	if _, err := b.WriteString(rule); err != nil {
		return "", err
	}

	return b.String(), nil
}
