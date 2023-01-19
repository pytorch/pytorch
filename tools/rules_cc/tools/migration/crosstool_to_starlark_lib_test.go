package crosstooltostarlarklib

import (
	"fmt"
	"strings"
	"testing"

	"log"
	crosstoolpb "third_party/com/github/bazelbuild/bazel/src/main/protobuf/crosstool_config_go_proto"
	"github.com/golang/protobuf/proto"
)

func makeCToolchainString(lines []string) string {
	return fmt.Sprintf(`toolchain {
  %s
}`, strings.Join(lines, "\n  "))
}

func makeCrosstool(CToolchains []string) *crosstoolpb.CrosstoolRelease {
	crosstool := &crosstoolpb.CrosstoolRelease{}
	requiredFields := []string{
		"major_version: '0'",
		"minor_version: '0'",
		"default_target_cpu: 'cpu'",
	}
	CToolchains = append(CToolchains, requiredFields...)
	if err := proto.UnmarshalText(strings.Join(CToolchains, "\n"), crosstool); err != nil {
		log.Fatalf("Failed to parse CROSSTOOL:", err)
	}
	return crosstool
}

func getSimpleCToolchain(id string) string {
	lines := []string{
		"toolchain_identifier: 'id-" + id + "'",
		"host_system_name: 'host-" + id + "'",
		"target_system_name: 'target-" + id + "'",
		"target_cpu: 'cpu-" + id + "'",
		"compiler: 'compiler-" + id + "'",
		"target_libc: 'libc-" + id + "'",
		"abi_version: 'version-" + id + "'",
		"abi_libc_version: 'libc_version-" + id + "'",
	}
	return makeCToolchainString(lines)
}

func getCToolchain(id, cpu, compiler string, extraLines []string) string {
	lines := []string{
		"toolchain_identifier: '" + id + "'",
		"host_system_name: 'host'",
		"target_system_name: 'target'",
		"target_cpu: '" + cpu + "'",
		"compiler: '" + compiler + "'",
		"target_libc: 'libc'",
		"abi_version: 'version'",
		"abi_libc_version: 'libc_version'",
	}
	lines = append(lines, extraLines...)
	return makeCToolchainString(lines)
}

func TestStringFieldsConditionStatement(t *testing.T) {
	toolchain1 := getSimpleCToolchain("1")
	toolchain2 := getSimpleCToolchain("2")
	toolchains := []string{toolchain1, toolchain2}
	crosstool := makeCrosstool(toolchains)

	testCases := []struct {
		field        string
		expectedText string
	}{
		{field: "toolchain_identifier",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        toolchain_identifier = "id-1"
    elif (ctx.attr.cpu == "cpu-2"):
        toolchain_identifier = "id-2"
    else:
        fail("Unreachable")`},
		{field: "host_system_name",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        host_system_name = "host-1"
    elif (ctx.attr.cpu == "cpu-2"):
        host_system_name = "host-2"
    else:
        fail("Unreachable")`},
		{field: "target_system_name",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        target_system_name = "target-1"
    elif (ctx.attr.cpu == "cpu-2"):
        target_system_name = "target-2"
    else:
        fail("Unreachable")`},
		{field: "target_cpu",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        target_cpu = "cpu-1"
    elif (ctx.attr.cpu == "cpu-2"):
        target_cpu = "cpu-2"
    else:
        fail("Unreachable")`},
		{field: "target_libc",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        target_libc = "libc-1"
    elif (ctx.attr.cpu == "cpu-2"):
        target_libc = "libc-2"
    else:
        fail("Unreachable")`},
		{field: "compiler",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        compiler = "compiler-1"
    elif (ctx.attr.cpu == "cpu-2"):
        compiler = "compiler-2"
    else:
        fail("Unreachable")`},
		{field: "abi_version",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        abi_version = "version-1"
    elif (ctx.attr.cpu == "cpu-2"):
        abi_version = "version-2"
    else:
        fail("Unreachable")`},
		{field: "abi_libc_version",
			expectedText: `
    if (ctx.attr.cpu == "cpu-1"):
        abi_libc_version = "libc_version-1"
    elif (ctx.attr.cpu == "cpu-2"):
        abi_libc_version = "libc_version-2"
    else:
        fail("Unreachable")`}}

	got, err := Transform(crosstool)
	if err != nil {
		t.Fatalf("CROSSTOOL conversion failed: %v", err)
	}

	failed := false
	for _, tc := range testCases {
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			failed = true
		}
	}
	if failed {
		t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
			strings.Join(toolchains, "\n"), got)
	}
}

func TestConditionsSameCpu(t *testing.T) {
	toolchainAA := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainAB := getCToolchain("2", "cpuA", "compilerB", []string{})
	toolchains := []string{toolchainAA, toolchainAB}
	crosstool := makeCrosstool(toolchains)

	testCases := []struct {
		field        string
		expectedText string
	}{
		{field: "toolchain_identifier",
			expectedText: `
    if (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerA"):
        toolchain_identifier = "1"
    elif (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerB"):
        toolchain_identifier = "2"
    else:
        fail("Unreachable")`},
		{field: "host_system_name",
			expectedText: `
    host_system_name = "host"`},
		{field: "target_system_name",
			expectedText: `
    target_system_name = "target"`},
		{field: "target_cpu",
			expectedText: `
    target_cpu = "cpuA"`},
		{field: "target_libc",
			expectedText: `
    target_libc = "libc"`},
		{field: "compiler",
			expectedText: `
    if (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerA"):
        compiler = "compilerA"
    elif (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerB"):
        compiler = "compilerB"
    else:
        fail("Unreachable")`},
		{field: "abi_version",
			expectedText: `
    abi_version = "version"`},
		{field: "abi_libc_version",
			expectedText: `
    abi_libc_version = "libc_version"`}}

	got, err := Transform(crosstool)
	if err != nil {
		t.Fatalf("CROSSTOOL conversion failed: %v", err)
	}

	failed := false
	for _, tc := range testCases {
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			failed = true
		}
	}
	if failed {
		t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
			strings.Join(toolchains, "\n"), got)
	}
}

func TestConditionsSameCompiler(t *testing.T) {
	toolchainAA := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainBA := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchains := []string{toolchainAA, toolchainBA}
	crosstool := makeCrosstool(toolchains)

	testCases := []struct {
		field        string
		expectedText string
	}{
		{field: "toolchain_identifier",
			expectedText: `
    if (ctx.attr.cpu == "cpuA"):
        toolchain_identifier = "1"
    elif (ctx.attr.cpu == "cpuB"):
        toolchain_identifier = "2"
    else:
        fail("Unreachable")`},
		{field: "target_cpu",
			expectedText: `
    if (ctx.attr.cpu == "cpuA"):
        target_cpu = "cpuA"
    elif (ctx.attr.cpu == "cpuB"):
        target_cpu = "cpuB"
    else:
        fail("Unreachable")`},
		{field: "compiler",
			expectedText: `
    compiler = "compilerA"`}}

	got, err := Transform(crosstool)
	if err != nil {
		t.Fatalf("CROSSTOOL conversion failed: %v", err)
	}

	failed := false
	for _, tc := range testCases {
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			failed = true
		}
	}
	if failed {
		t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
			strings.Join(toolchains, "\n"), got)
	}
}

func TestNonMandatoryStrings(t *testing.T) {
	toolchainAA := getCToolchain("1", "cpuA", "compilerA", []string{"cc_target_os: 'osA'"})
	toolchainBB := getCToolchain("2", "cpuB", "compilerB", []string{})
	toolchains := []string{toolchainAA, toolchainBB}
	crosstool := makeCrosstool(toolchains)

	testCases := []struct {
		field        string
		expectedText string
	}{
		{field: "cc_target_os",
			expectedText: `
    if (ctx.attr.cpu == "cpuB"):
        cc_target_os = None
    elif (ctx.attr.cpu == "cpuA"):
        cc_target_os = "osA"
    else:
        fail("Unreachable")`},
		{field: "builtin_sysroot",
			expectedText: `
    builtin_sysroot = None`}}

	got, err := Transform(crosstool)
	if err != nil {
		t.Fatalf("CROSSTOOL conversion failed: %v", err)
	}

	failed := false
	for _, tc := range testCases {
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			failed = true
		}
	}
	if failed {
		t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
			strings.Join(toolchains, "\n"), got)
	}
}

func TestBuiltinIncludeDirectories(t *testing.T) {
	toolchainAA := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainBA := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchainCA := getCToolchain("3", "cpuC", "compilerA",
		[]string{"cxx_builtin_include_directory: 'dirC'"})
	toolchainCB := getCToolchain("4", "cpuC", "compilerB",
		[]string{"cxx_builtin_include_directory: 'dirC'",
			"cxx_builtin_include_directory: 'dirB'"})
	toolchainDA := getCToolchain("5", "cpuD", "compilerA",
		[]string{"cxx_builtin_include_directory: 'dirC'"})

	toolchainsEmpty := []string{toolchainAA, toolchainBA}

	toolchainsOneNonempty := []string{toolchainAA, toolchainBA, toolchainCA}

	toolchainsSameNonempty := []string{toolchainCA, toolchainDA}

	allToolchains := []string{toolchainAA, toolchainBA, toolchainCA, toolchainCB, toolchainDA}

	testCases := []struct {
		field        string
		toolchains   []string
		expectedText string
	}{
		{field: "cxx_builtin_include_directories",
			toolchains: toolchainsEmpty,
			expectedText: `
    cxx_builtin_include_directories = []`},
		{field: "cxx_builtin_include_directories",
			toolchains: toolchainsOneNonempty,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"
        or ctx.attr.cpu == "cpuB"):
        cxx_builtin_include_directories = []
    elif (ctx.attr.cpu == "cpuC"):
        cxx_builtin_include_directories = ["dirC"]
    else:
        fail("Unreachable")`},
		{field: "cxx_builtin_include_directories",
			toolchains: toolchainsSameNonempty,
			expectedText: `
    cxx_builtin_include_directories = ["dirC"]`},
		{field: "cxx_builtin_include_directories",
			toolchains: allToolchains,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"
        or ctx.attr.cpu == "cpuB"):
        cxx_builtin_include_directories = []
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerA"
        or ctx.attr.cpu == "cpuD"):
        cxx_builtin_include_directories = ["dirC"]
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerB"):
        cxx_builtin_include_directories = ["dirC", "dirB"]`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func TestMakeVariables(t *testing.T) {
	toolchainEmpty1 := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainEmpty2 := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchainA1 := getCToolchain("3", "cpuC", "compilerA",
		[]string{"make_variable {name: 'A', value: 'a/b/c'}"})
	toolchainA2 := getCToolchain("4", "cpuC", "compilerB",
		[]string{"make_variable {name: 'A', value: 'a/b/c'}"})
	toolchainAB := getCToolchain("5", "cpuC", "compilerC",
		[]string{"make_variable {name: 'A', value: 'a/b/c'}",
			"make_variable {name: 'B', value: 'a/b/c'}"})
	toolchainBA := getCToolchain("6", "cpuD", "compilerA",
		[]string{"make_variable {name: 'B', value: 'a/b/c'}",
			"make_variable {name: 'A', value: 'a b c'}"})

	toolchainsEmpty := []string{toolchainEmpty1, toolchainEmpty2}

	toolchainsOneNonempty := []string{toolchainEmpty1, toolchainA1}

	toolchainsSameNonempty := []string{toolchainA1, toolchainA2}

	toolchainsDifferentOrder := []string{toolchainAB, toolchainBA}

	allToolchains := []string{
		toolchainEmpty1,
		toolchainEmpty2,
		toolchainA1,
		toolchainA2,
		toolchainAB,
		toolchainBA,
	}

	testCases := []struct {
		field        string
		toolchains   []string
		expectedText string
	}{
		{field: "make_variables",
			toolchains: toolchainsEmpty,
			expectedText: `
    make_variables = []`},
		{field: "make_variables",
			toolchains: toolchainsOneNonempty,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"):
        make_variables = []
    elif (ctx.attr.cpu == "cpuC"):
        make_variables = [make_variable(name = "A", value = "a/b/c")]
    else:
        fail("Unreachable")`},
		{field: "make_variables",
			toolchains: toolchainsSameNonempty,
			expectedText: `
    make_variables = [make_variable(name = "A", value = "a/b/c")]`},
		{field: "make_variables",
			toolchains: toolchainsDifferentOrder,
			expectedText: `
    if (ctx.attr.cpu == "cpuC"):
        make_variables = [
            make_variable(name = "A", value = "a/b/c"),
            make_variable(name = "B", value = "a/b/c"),
        ]
    elif (ctx.attr.cpu == "cpuD"):
        make_variables = [
            make_variable(name = "B", value = "a/b/c"),
            make_variable(name = "A", value = "a b c"),
        ]
    else:
        fail("Unreachable")`},
		{field: "make_variables",
			toolchains: allToolchains,
			expectedText: `
    if (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerC"):
        make_variables = [
            make_variable(name = "A", value = "a/b/c"),
            make_variable(name = "B", value = "a/b/c"),
        ]
    elif (ctx.attr.cpu == "cpuD"):
        make_variables = [
            make_variable(name = "B", value = "a/b/c"),
            make_variable(name = "A", value = "a b c"),
        ]
    elif (ctx.attr.cpu == "cpuA"
        or ctx.attr.cpu == "cpuB"):
        make_variables = []
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerA"
        or ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerB"):
        make_variables = [make_variable(name = "A", value = "a/b/c")]
    else:
        fail("Unreachable")`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func TestToolPaths(t *testing.T) {
	toolchainEmpty1 := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainEmpty2 := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchainA1 := getCToolchain("3", "cpuC", "compilerA",
		[]string{"tool_path {name: 'A', path: 'a/b/c'}"})
	toolchainA2 := getCToolchain("4", "cpuC", "compilerB",
		[]string{"tool_path {name: 'A', path: 'a/b/c'}"})
	toolchainAB := getCToolchain("5", "cpuC", "compilerC",
		[]string{"tool_path {name: 'A', path: 'a/b/c'}",
			"tool_path {name: 'B', path: 'a/b/c'}"})
	toolchainBA := getCToolchain("6", "cpuD", "compilerA",
		[]string{"tool_path {name: 'B', path: 'a/b/c'}",
			"tool_path {name: 'A', path: 'a/b/c'}"})

	toolchainsEmpty := []string{toolchainEmpty1, toolchainEmpty2}

	toolchainsOneNonempty := []string{toolchainEmpty1, toolchainA1}

	toolchainsSameNonempty := []string{toolchainA1, toolchainA2}

	toolchainsDifferentOrder := []string{toolchainAB, toolchainBA}

	allToolchains := []string{
		toolchainEmpty1,
		toolchainEmpty2,
		toolchainA1,
		toolchainA2,
		toolchainAB,
		toolchainBA,
	}

	testCases := []struct {
		field        string
		toolchains   []string
		expectedText string
	}{
		{field: "tool_paths",
			toolchains: toolchainsEmpty,
			expectedText: `
    tool_paths = []`},
		{field: "tool_paths",
			toolchains: toolchainsOneNonempty,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"):
        tool_paths = []
    elif (ctx.attr.cpu == "cpuC"):
        tool_paths = [tool_path(name = "A", path = "a/b/c")]
    else:
        fail("Unreachable")`},
		{field: "tool_paths",
			toolchains: toolchainsSameNonempty,
			expectedText: `
    tool_paths = [tool_path(name = "A", path = "a/b/c")]`},
		{field: "tool_paths",
			toolchains: toolchainsDifferentOrder,
			expectedText: `
    if (ctx.attr.cpu == "cpuC"):
        tool_paths = [
            tool_path(name = "A", path = "a/b/c"),
            tool_path(name = "B", path = "a/b/c"),
        ]
    elif (ctx.attr.cpu == "cpuD"):
        tool_paths = [
            tool_path(name = "B", path = "a/b/c"),
            tool_path(name = "A", path = "a/b/c"),
        ]
    else:
        fail("Unreachable")`},
		{field: "tool_paths",
			toolchains: allToolchains,
			expectedText: `
    if (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerC"):
        tool_paths = [
            tool_path(name = "A", path = "a/b/c"),
            tool_path(name = "B", path = "a/b/c"),
        ]
    elif (ctx.attr.cpu == "cpuD"):
        tool_paths = [
            tool_path(name = "B", path = "a/b/c"),
            tool_path(name = "A", path = "a/b/c"),
        ]
    elif (ctx.attr.cpu == "cpuA"
        or ctx.attr.cpu == "cpuB"):
        tool_paths = []
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerA"
        or ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerB"):
        tool_paths = [tool_path(name = "A", path = "a/b/c")]
    else:
        fail("Unreachable")`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func getArtifactNamePattern(lines []string) string {
	return fmt.Sprintf(`artifact_name_pattern {
  %s
}`, strings.Join(lines, "\n  "))
}

func TestArtifactNamePatterns(t *testing.T) {
	toolchainEmpty1 := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainEmpty2 := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchainA1 := getCToolchain("3", "cpuC", "compilerA",
		[]string{
			getArtifactNamePattern([]string{
				"category_name: 'A'",
				"prefix: 'p'",
				"extension: '.exe'"}),
		},
	)
	toolchainA2 := getCToolchain("4", "cpuC", "compilerB",
		[]string{
			getArtifactNamePattern([]string{
				"category_name: 'A'",
				"prefix: 'p'",
				"extension: '.exe'"}),
		},
	)
	toolchainAB := getCToolchain("5", "cpuC", "compilerC",
		[]string{
			getArtifactNamePattern([]string{
				"category_name: 'A'",
				"prefix: 'p'",
				"extension: '.exe'"}),
			getArtifactNamePattern([]string{
				"category_name: 'B'",
				"prefix: 'p'",
				"extension: '.exe'"}),
		},
	)
	toolchainBA := getCToolchain("6", "cpuD", "compilerA",
		[]string{
			getArtifactNamePattern([]string{
				"category_name: 'B'",
				"prefix: 'p'",
				"extension: '.exe'"}),
			getArtifactNamePattern([]string{
				"category_name: 'A'",
				"prefix: 'p'",
				"extension: '.exe'"}),
		},
	)
	toolchainsEmpty := []string{toolchainEmpty1, toolchainEmpty2}

	toolchainsOneNonempty := []string{toolchainEmpty1, toolchainA1}

	toolchainsSameNonempty := []string{toolchainA1, toolchainA2}

	toolchainsDifferentOrder := []string{toolchainAB, toolchainBA}

	allToolchains := []string{
		toolchainEmpty1,
		toolchainEmpty2,
		toolchainA1,
		toolchainA2,
		toolchainAB,
		toolchainBA,
	}

	testCases := []struct {
		field        string
		toolchains   []string
		expectedText string
	}{
		{field: "artifact_name_patterns",
			toolchains: toolchainsEmpty,
			expectedText: `
    artifact_name_patterns = []`},
		{field: "artifact_name_patterns",
			toolchains: toolchainsOneNonempty,
			expectedText: `
    if (ctx.attr.cpu == "cpuC"):
        artifact_name_patterns = [
            artifact_name_pattern(
                category_name = "A",
                prefix = "p",
                extension = ".exe",
            ),
        ]
    elif (ctx.attr.cpu == "cpuA"):
        artifact_name_patterns = []
    else:
        fail("Unreachable")`},
		{field: "artifact_name_patterns",
			toolchains: toolchainsSameNonempty,
			expectedText: `
    artifact_name_patterns = [
        artifact_name_pattern(
            category_name = "A",
            prefix = "p",
            extension = ".exe",
        ),
    ]`},
		{field: "artifact_name_patterns",
			toolchains: toolchainsDifferentOrder,
			expectedText: `
    if (ctx.attr.cpu == "cpuC"):
        artifact_name_patterns = [
            artifact_name_pattern(
                category_name = "A",
                prefix = "p",
                extension = ".exe",
            ),
            artifact_name_pattern(
                category_name = "B",
                prefix = "p",
                extension = ".exe",
            ),
        ]
    elif (ctx.attr.cpu == "cpuD"):
        artifact_name_patterns = [
            artifact_name_pattern(
                category_name = "B",
                prefix = "p",
                extension = ".exe",
            ),
            artifact_name_pattern(
                category_name = "A",
                prefix = "p",
                extension = ".exe",
            ),
        ]
    else:
        fail("Unreachable")`},
		{field: "artifact_name_patterns",
			toolchains: allToolchains,
			expectedText: `
    if (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerC"):
        artifact_name_patterns = [
            artifact_name_pattern(
                category_name = "A",
                prefix = "p",
                extension = ".exe",
            ),
            artifact_name_pattern(
                category_name = "B",
                prefix = "p",
                extension = ".exe",
            ),
        ]
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerA"
        or ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerB"):
        artifact_name_patterns = [
            artifact_name_pattern(
                category_name = "A",
                prefix = "p",
                extension = ".exe",
            ),
        ]
    elif (ctx.attr.cpu == "cpuD"):
        artifact_name_patterns = [
            artifact_name_pattern(
                category_name = "B",
                prefix = "p",
                extension = ".exe",
            ),
            artifact_name_pattern(
                category_name = "A",
                prefix = "p",
                extension = ".exe",
            ),
        ]
    elif (ctx.attr.cpu == "cpuA"
        or ctx.attr.cpu == "cpuB"):
        artifact_name_patterns = []
    else:
        fail("Unreachable")`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func getFeature(lines []string) string {
	return fmt.Sprintf(`feature {
  %s
}`, strings.Join(lines, "\n  "))
}

func TestFeatureListAssignment(t *testing.T) {
	toolchainEmpty1 := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainEmpty2 := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchainA1 := getCToolchain("3", "cpuC", "compilerA",
		[]string{getFeature([]string{"name: 'A'"})},
	)
	toolchainA2 := getCToolchain("4", "cpuC", "compilerB",
		[]string{getFeature([]string{"name: 'A'"})},
	)
	toolchainAB := getCToolchain("5", "cpuC", "compilerC",
		[]string{
			getFeature([]string{"name: 'A'"}),
			getFeature([]string{"name: 'B'"}),
		},
	)
	toolchainBA := getCToolchain("6", "cpuD", "compilerA",
		[]string{
			getFeature([]string{"name: 'B'"}),
			getFeature([]string{"name: 'A'"}),
		},
	)
	toolchainsEmpty := []string{toolchainEmpty1, toolchainEmpty2}

	toolchainsOneNonempty := []string{toolchainEmpty1, toolchainA1}

	toolchainsSameNonempty := []string{toolchainA1, toolchainA2}

	toolchainsDifferentOrder := []string{toolchainAB, toolchainBA}

	allToolchains := []string{
		toolchainEmpty1,
		toolchainEmpty2,
		toolchainA1,
		toolchainA2,
		toolchainAB,
		toolchainBA,
	}

	testCases := []struct {
		field        string
		toolchains   []string
		expectedText string
	}{
		{field: "features",
			toolchains: toolchainsEmpty,
			expectedText: `
    features = []`},
		{field: "features",
			toolchains: toolchainsOneNonempty,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"):
        features = []
    elif (ctx.attr.cpu == "cpuC"):
        features = [a_feature]
    else:
        fail("Unreachable")`},
		{field: "features",
			toolchains: toolchainsSameNonempty,
			expectedText: `
    features = [a_feature]`},
		{field: "features",
			toolchains: toolchainsDifferentOrder,
			expectedText: `
    if (ctx.attr.cpu == "cpuC"):
        features = [a_feature, b_feature]
    elif (ctx.attr.cpu == "cpuD"):
        features = [b_feature, a_feature]
    else:
        fail("Unreachable")`},
		{field: "features",
			toolchains: allToolchains,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"
        or ctx.attr.cpu == "cpuB"):
        features = []
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerA"
        or ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerB"):
        features = [a_feature]
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerC"):
        features = [a_feature, b_feature]
    elif (ctx.attr.cpu == "cpuD"):
        features = [b_feature, a_feature]
    else:
        fail("Unreachable")`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func getActionConfig(lines []string) string {
	return fmt.Sprintf(`action_config {
  %s
}`, strings.Join(lines, "\n  "))
}

func TestActionConfigListAssignment(t *testing.T) {
	toolchainEmpty1 := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainEmpty2 := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchainA1 := getCToolchain("3", "cpuC", "compilerA",
		[]string{
			getActionConfig([]string{"action_name: 'A'", "config_name: 'A'"}),
		},
	)
	toolchainA2 := getCToolchain("4", "cpuC", "compilerB",
		[]string{
			getActionConfig([]string{"action_name: 'A'", "config_name: 'A'"}),
		},
	)
	toolchainAB := getCToolchain("5", "cpuC", "compilerC",
		[]string{
			getActionConfig([]string{"action_name: 'A'", "config_name: 'A'"}),
			getActionConfig([]string{"action_name: 'B'", "config_name: 'B'"}),
		},
	)
	toolchainBA := getCToolchain("6", "cpuD", "compilerA",
		[]string{
			getActionConfig([]string{"action_name: 'B'", "config_name: 'B'"}),
			getActionConfig([]string{"action_name: 'A'", "config_name: 'A'"}),
		},
	)
	toolchainsEmpty := []string{toolchainEmpty1, toolchainEmpty2}

	toolchainsOneNonempty := []string{toolchainEmpty1, toolchainA1}

	toolchainsSameNonempty := []string{toolchainA1, toolchainA2}

	toolchainsDifferentOrder := []string{toolchainAB, toolchainBA}

	allToolchains := []string{
		toolchainEmpty1,
		toolchainEmpty2,
		toolchainA1,
		toolchainA2,
		toolchainAB,
		toolchainBA,
	}

	testCases := []struct {
		field        string
		toolchains   []string
		expectedText string
	}{
		{field: "action_configs",
			toolchains: toolchainsEmpty,
			expectedText: `
    action_configs = []`},
		{field: "action_configs",
			toolchains: toolchainsOneNonempty,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"):
        action_configs = []
    elif (ctx.attr.cpu == "cpuC"):
        action_configs = [a_action]
    else:
        fail("Unreachable")`},
		{field: "action_configs",
			toolchains: toolchainsSameNonempty,
			expectedText: `
    action_configs = [a_action]`},
		{field: "action_configs",
			toolchains: toolchainsDifferentOrder,
			expectedText: `
    if (ctx.attr.cpu == "cpuC"):
        action_configs = [a_action, b_action]
    elif (ctx.attr.cpu == "cpuD"):
        action_configs = [b_action, a_action]
    else:
        fail("Unreachable")`},
		{field: "action_configs",
			toolchains: allToolchains,
			expectedText: `
    if (ctx.attr.cpu == "cpuA"
        or ctx.attr.cpu == "cpuB"):
        action_configs = []
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerA"
        or ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerB"):
        action_configs = [a_action]
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerC"):
        action_configs = [a_action, b_action]
    elif (ctx.attr.cpu == "cpuD"):
        action_configs = [b_action, a_action]
    else:
        fail("Unreachable")`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func TestAllAndNoneAvailableErrorsWhenMoreThanOneElement(t *testing.T) {
	toolchainFeatureAllAvailable := getCToolchain("1", "cpu", "compiler",
		[]string{getFeature([]string{
			"name: 'A'",
			"flag_set {",
			"  action: 'A'",
			"  flag_group {",
			"    flag: 'f'",
			"    expand_if_all_available: 'e1'",
			"    expand_if_all_available: 'e2'",
			"  }",
			"}",
		})},
	)
	toolchainFeatureNoneAvailable := getCToolchain("1", "cpu", "compiler",
		[]string{getFeature([]string{
			"name: 'A'",
			"flag_set {",
			"  action: 'A'",
			"  flag_group {",
			"    flag: 'f'",
			"    expand_if_none_available: 'e1'",
			"    expand_if_none_available: 'e2'",
			"  }",
			"}",
		})},
	)
	toolchainActionConfigAllAvailable := getCToolchain("1", "cpu", "compiler",
		[]string{getActionConfig([]string{
			"config_name: 'A'",
			"action_name: 'A'",
			"flag_set {",
			"  action: 'A'",
			"  flag_group {",
			"    flag: 'f'",
			"    expand_if_all_available: 'e1'",
			"    expand_if_all_available: 'e2'",
			"  }",
			"}",
		})},
	)
	toolchainActionConfigNoneAvailable := getCToolchain("1", "cpu", "compiler",
		[]string{getActionConfig([]string{
			"config_name: 'A'",
			"action_name: 'A'",
			"flag_set {",
			"  action: 'A'",
			"  flag_group {",
			"    flag: 'f'",
			"    expand_if_none_available: 'e1'",
			"    expand_if_none_available: 'e2'",
			"  }",
			"}",
		})},
	)

	testCases := []struct {
		field        string
		toolchain    string
		expectedText string
	}{
		{field: "features",
			toolchain: toolchainFeatureAllAvailable,
			expectedText: "Error in feature 'A': Flag group must not have more " +
				"than one 'expand_if_all_available' field"},
		{field: "features",
			toolchain: toolchainFeatureNoneAvailable,
			expectedText: "Error in feature 'A': Flag group must not have more " +
				"than one 'expand_if_none_available' field"},
		{field: "action_configs",
			toolchain: toolchainActionConfigAllAvailable,
			expectedText: "Error in action_config 'A': Flag group must not have more " +
				"than one 'expand_if_all_available' field"},
		{field: "action_configs",
			toolchain: toolchainActionConfigNoneAvailable,
			expectedText: "Error in action_config 'A': Flag group must not have more " +
				"than one 'expand_if_none_available' field"},
	}

	for _, tc := range testCases {
		crosstool := makeCrosstool([]string{tc.toolchain})
		_, err := Transform(crosstool)
		if err == nil || !strings.Contains(err.Error(), tc.expectedText) {
			t.Errorf("Expected error: %s, got: %v", tc.expectedText, err)
		}
	}
}

func TestFeaturesAndActionConfigsSetToNoneWhenAllOptionsAreExausted(t *testing.T) {
	toolchainFeatureAEnabled := getCToolchain("1", "cpuA", "compilerA",
		[]string{getFeature([]string{"name: 'A'", "enabled: true"})},
	)
	toolchainFeatureADisabled := getCToolchain("2", "cpuA", "compilerB",
		[]string{getFeature([]string{"name: 'A'", "enabled: false"})},
	)

	toolchainWithoutFeatureA := getCToolchain("3", "cpuA", "compilerC", []string{})

	toolchainActionConfigAEnabled := getCToolchain("4", "cpuA", "compilerD",
		[]string{getActionConfig([]string{
			"config_name: 'A'",
			"action_name: 'A'",
			"enabled: true",
		})})

	toolchainActionConfigADisabled := getCToolchain("5", "cpuA", "compilerE",
		[]string{getActionConfig([]string{
			"config_name: 'A'",
			"action_name: 'A'",
		})})

	toolchainWithoutActionConfigA := getCToolchain("6", "cpuA", "compilerF", []string{})

	testCases := []struct {
		field        string
		toolchains   []string
		expectedText string
	}{
		{field: "features",
			toolchains: []string{
				toolchainFeatureAEnabled, toolchainFeatureADisabled, toolchainWithoutFeatureA},
			expectedText: `
    if (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerB"):
        a_feature = feature(name = "A")
    elif (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerA"):
        a_feature = feature(name = "A", enabled = True)
    else:
        a_feature = None
`},
		{field: "action_config",
			toolchains: []string{
				toolchainActionConfigAEnabled, toolchainActionConfigADisabled, toolchainWithoutActionConfigA},
			expectedText: `
    if (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerE"):
        a_action = action_config(action_name = "A")
    elif (ctx.attr.cpu == "cpuA" and ctx.attr.compiler == "compilerD"):
        a_action = action_config(action_name = "A", enabled = True)
    else:
        a_action = None
`},
	}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly convert '%s' field, expected to contain:\n%v\n",
				tc.field, tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func TestActionConfigDeclaration(t *testing.T) {
	toolchainEmpty1 := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainEmpty2 := getCToolchain("2", "cpuB", "compilerA", []string{})

	toolchainNameNotInDict := getCToolchain("3", "cpBC", "compilerB",
		[]string{
			getActionConfig([]string{"action_name: 'A-B.C'", "config_name: 'A-B.C'"}),
		},
	)
	toolchainNameInDictA := getCToolchain("4", "cpuC", "compilerA",
		[]string{
			getActionConfig([]string{"action_name: 'c++-compile'", "config_name: 'c++-compile'"}),
		},
	)
	toolchainNameInDictB := getCToolchain("5", "cpuC", "compilerB",
		[]string{
			getActionConfig([]string{
				"action_name: 'c++-compile'",
				"config_name: 'c++-compile'",
				"tool {",
				"  tool_path: '/a/b/c'",
				"}",
			}),
		},
	)
	toolchainComplexActionConfig := getCToolchain("6", "cpuC", "compilerC",
		[]string{
			getActionConfig([]string{
				"action_name: 'action-complex'",
				"config_name: 'action-complex'",
				"enabled: true",
				"tool {",
				"  tool_path: '/a/b/c'",
				"  with_feature {",
				"    feature: 'a'",
				"    feature: 'b'",
				"    not_feature: 'c'",
				"    not_feature: 'd'",
				"  }",
				"  with_feature{",
				"    feature: 'e'",
				"  }",
				"  execution_requirement: 'a'",
				"}",
				"tool {",
				"  tool_path: ''",
				"}",
				"flag_set {",
				"  flag_group {",
				"    flag: 'a'",
				"    flag: '%b'",
				"    iterate_over: 'c'",
				"    expand_if_all_available: 'd'",
				"    expand_if_none_available: 'e'",
				"    expand_if_true: 'f'",
				"    expand_if_false: 'g'",
				"    expand_if_equal {",
				"      variable: 'var'",
				"      value: 'val'",
				"    }",
				"  }",
				"  flag_group {",
				"    flag_group {",
				"      flag: 'a'",
				"    }",
				"  }",
				"}",
				"flag_set {",
				"  with_feature {",
				"    feature: 'a'",
				"    feature: 'b'",
				"    not_feature: 'c'",
				"    not_feature: 'd'",
				"  }",
				"}",
				"env_set {",
				"  action: 'a'",
				"  env_entry {",
				"    key: 'k'",
				"    value: 'v'",
				"  }",
				"  with_feature {",
				"    feature: 'a'",
				"  }",
				"}",
				"requires {",
				"  feature: 'a'",
				"  feature: 'b'",
				"}",
				"implies: 'a'",
				"implies: 'b'",
			}),
		},
	)

	testCases := []struct {
		toolchains   []string
		expectedText string
	}{
		{
			toolchains: []string{toolchainEmpty1, toolchainEmpty2},
			expectedText: `
    action_configs = []`},
		{
			toolchains: []string{toolchainEmpty1, toolchainNameNotInDict},
			expectedText: `
    a_b_c_action = action_config(action_name = "A-B.C")`},
		{
			toolchains: []string{toolchainNameInDictA, toolchainNameInDictB},
			expectedText: `
    if (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerB"):
        cpp_compile_action = action_config(
            action_name = ACTION_NAMES.cpp_compile,
            tools = [tool(path = "/a/b/c")],
        )
    elif (ctx.attr.cpu == "cpuC" and ctx.attr.compiler == "compilerA"):
        cpp_compile_action = action_config(action_name = ACTION_NAMES.cpp_compile)`},
		{
			toolchains: []string{toolchainComplexActionConfig},
			expectedText: `
    action_complex_action = action_config(
        action_name = "action-complex",
        enabled = True,
        flag_sets = [
            flag_set(
                flag_groups = [
                    flag_group(
                        flags = ["a", "%b"],
                        iterate_over = "c",
                        expand_if_available = "d",
                        expand_if_not_available = "e",
                        expand_if_true = "f",
                        expand_if_false = "g",
                        expand_if_equal = variable_with_value(name = "var", value = "val"),
                    ),
                    flag_group(flag_groups = [flag_group(flags = ["a"])]),
                ],
            ),
            flag_set(
                with_features = [
                    with_feature_set(
                        features = ["a", "b"],
                        not_features = ["c", "d"],
                    ),
                ],
            ),
        ],
        implies = ["a", "b"],
        tools = [
            tool(
                path = "/a/b/c",
                with_features = [
                    with_feature_set(
                        features = ["a", "b"],
                        not_features = ["c", "d"],
                    ),
                    with_feature_set(features = ["e"]),
                ],
                execution_requirements = ["a"],
            ),
            tool(path = "NOT_USED"),
        ],
    )`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly declare an action_config, expected to contain:\n%v\n",
				tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func TestFeatureDeclaration(t *testing.T) {
	toolchainEmpty1 := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainEmpty2 := getCToolchain("2", "cpuB", "compilerA", []string{})

	toolchainSimpleFeatureA1 := getCToolchain("3", "cpuB", "compilerB",
		[]string{
			getFeature([]string{"name: 'Feature-c++.a'", "enabled: true"}),
		},
	)
	toolchainSimpleFeatureA2 := getCToolchain("4", "cpuC", "compilerA",
		[]string{
			getFeature([]string{"name: 'Feature-c++.a'"}),
		},
	)
	toolchainComplexFeature := getCToolchain("5", "cpuC", "compilerC",
		[]string{
			getFeature([]string{
				"name: 'complex-feature'",
				"enabled: true",
				"flag_set {",
				"  action: 'c++-compile'",    // in ACTION_NAMES
				"  action: 'something-else'", // not in ACTION_NAMES
				"  flag_group {",
				"    flag: 'a'",
				"    flag: '%b'",
				"    iterate_over: 'c'",
				"    expand_if_all_available: 'd'",
				"    expand_if_none_available: 'e'",
				"    expand_if_true: 'f'",
				"    expand_if_false: 'g'",
				"    expand_if_equal {",
				"      variable: 'var'",
				"      value: 'val'",
				"    }",
				"  }",
				"  flag_group {",
				"    flag_group {",
				"      flag: 'a'",
				"    }",
				"  }",
				"}",
				"flag_set {", // all_compile_actions
				"  action: 'c-compile'",
				"  action: 'c++-compile'",
				"  action: 'linkstamp-compile'",
				"  action: 'assemble'",
				"  action: 'preprocess-assemble'",
				"  action: 'c++-header-parsing'",
				"  action: 'c++-module-compile'",
				"  action: 'c++-module-codegen'",
				"  action: 'clif-match'",
				"  action: 'lto-backend'",
				"}",
				"flag_set {", // all_cpp_compile_actions
				"  action: 'c++-compile'",
				"  action: 'linkstamp-compile'",
				"  action: 'c++-header-parsing'",
				"  action: 'c++-module-compile'",
				"  action: 'c++-module-codegen'",
				"  action: 'clif-match'",
				"}",
				"flag_set {", // all_link_actions
				"  action: 'c++-link-executable'",
				"  action: 'c++-link-dynamic-library'",
				"  action: 'c++-link-nodeps-dynamic-library'",
				"}",
				"flag_set {", // all_cpp_compile_actions + all_link_actions
				"  action: 'c++-compile'",
				"  action: 'linkstamp-compile'",
				"  action: 'c++-header-parsing'",
				"  action: 'c++-module-compile'",
				"  action: 'c++-module-codegen'",
				"  action: 'clif-match'",
				"  action: 'c++-link-executable'",
				"  action: 'c++-link-dynamic-library'",
				"  action: 'c++-link-nodeps-dynamic-library'",
				"}",
				"flag_set {", // all_link_actions + something else
				"  action: 'c++-link-executable'",
				"  action: 'c++-link-dynamic-library'",
				"  action: 'c++-link-nodeps-dynamic-library'",
				"  action: 'some.unknown-c++.action'",
				"}",
				"env_set {",
				"  action: 'a'",
				"  env_entry {",
				"    key: 'k'",
				"    value: 'v'",
				"  }",
				"  with_feature {",
				"    feature: 'a'",
				"  }",
				"}",
				"env_set {",
				"  action: 'c-compile'",
				"}",
				"env_set {", // all_compile_actions
				"  action: 'c-compile'",
				"  action: 'c++-compile'",
				"  action: 'linkstamp-compile'",
				"  action: 'assemble'",
				"  action: 'preprocess-assemble'",
				"  action: 'c++-header-parsing'",
				"  action: 'c++-module-compile'",
				"  action: 'c++-module-codegen'",
				"  action: 'clif-match'",
				"  action: 'lto-backend'",
				"}",
				"requires {",
				"  feature: 'a'",
				"  feature: 'b'",
				"}",
				"implies: 'a'",
				"implies: 'b'",
				"provides: 'c'",
				"provides: 'd'",
			}),
		},
	)

	testCases := []struct {
		toolchains   []string
		expectedText string
	}{
		{
			toolchains: []string{toolchainEmpty1, toolchainEmpty2},
			expectedText: `
    features = []
`},
		{
			toolchains: []string{toolchainEmpty1, toolchainSimpleFeatureA1},
			expectedText: `
    feature_cpp_a_feature = feature(name = "Feature-c++.a", enabled = True)`},
		{
			toolchains: []string{toolchainSimpleFeatureA1, toolchainSimpleFeatureA2},
			expectedText: `
    if (ctx.attr.cpu == "cpuC"):
        feature_cpp_a_feature = feature(name = "Feature-c++.a")
    elif (ctx.attr.cpu == "cpuB"):
        feature_cpp_a_feature = feature(name = "Feature-c++.a", enabled = True)`},
		{
			toolchains: []string{toolchainComplexFeature},
			expectedText: `
    complex_feature_feature = feature(
        name = "complex-feature",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile, "something-else"],
                flag_groups = [
                    flag_group(
                        flags = ["a", "%b"],
                        iterate_over = "c",
                        expand_if_available = "d",
                        expand_if_not_available = "e",
                        expand_if_true = "f",
                        expand_if_false = "g",
                        expand_if_equal = variable_with_value(name = "var", value = "val"),
                    ),
                    flag_group(flag_groups = [flag_group(flags = ["a"])]),
                ],
            ),
            flag_set(actions = all_compile_actions),
            flag_set(actions = all_cpp_compile_actions),
            flag_set(actions = all_link_actions),
            flag_set(
                actions = all_cpp_compile_actions +
                    all_link_actions,
            ),
            flag_set(
                actions = all_link_actions +
                    ["some.unknown-c++.action"],
            ),
        ],
        env_sets = [
            env_set(
                actions = ["a"],
                env_entries = [env_entry(key = "k", value = "v")],
                with_features = [with_feature_set(features = ["a"])],
            ),
            env_set(actions = [ACTION_NAMES.c_compile]),
            env_set(actions = all_compile_actions),
        ],
        requires = [feature_set(features = ["a", "b"])],
        implies = ["a", "b"],
        provides = ["c", "d"],
    )`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly declare a feature, expected to contain:\n%v\n",
				tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}

func TestRule(t *testing.T) {
	simpleToolchain := getSimpleCToolchain("simple")
	expected := `load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
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
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    toolchain_identifier = "id-simple"

    host_system_name = "host-simple"

    target_system_name = "target-simple"

    target_cpu = "cpu-simple"

    target_libc = "libc-simple"

    compiler = "compiler-simple"

    abi_version = "version-simple"

    abi_libc_version = "libc_version-simple"

    cc_target_os = None

    builtin_sysroot = None

    all_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.clif_match,
        ACTION_NAMES.lto_backend,
    ]

    all_cpp_compile_actions = [
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.clif_match,
    ]

    preprocessor_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.clif_match,
    ]

    codegen_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.lto_backend,
    ]

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    action_configs = []

    features = []

    cxx_builtin_include_directories = []

    artifact_name_patterns = []

    make_variables = []

    tool_paths = []


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
cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True, values=["cpu-simple"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
`
	crosstool := makeCrosstool([]string{simpleToolchain})
	got, err := Transform(crosstool)
	if err != nil {
		t.Fatalf("CROSSTOOL conversion failed: %v", err)
	}
	if got != expected {
		t.Fatalf("Expected:\n%v\nGot:\n%v\nTested CROSSTOOL:\n%v",
			expected, got, simpleToolchain)
	}
}

func TestAllowedCompilerValues(t *testing.T) {
	toolchainAA := getCToolchain("1", "cpuA", "compilerA", []string{})
	toolchainBA := getCToolchain("2", "cpuB", "compilerA", []string{})
	toolchainBB := getCToolchain("3", "cpuB", "compilerB", []string{})
	toolchainCC := getCToolchain("4", "cpuC", "compilerC", []string{})

	testCases := []struct {
		toolchains   []string
		expectedText string
	}{
		{
			toolchains: []string{toolchainAA, toolchainBA},
			expectedText: `
cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True, values=["cpuA", "cpuB"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
`},
		{
			toolchains: []string{toolchainBA, toolchainBB},
			expectedText: `
cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True, values=["cpuB"]),
        "compiler": attr.string(mandatory=True, values=["compilerA", "compilerB"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
`},
		{
			toolchains: []string{toolchainAA, toolchainBA, toolchainBB},
			expectedText: `
cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True, values=["cpuA", "cpuB"]),
        "compiler": attr.string(mandatory=True, values=["compilerA", "compilerB"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
`},
		{
			toolchains: []string{toolchainAA, toolchainBA, toolchainBB, toolchainCC},
			expectedText: `
cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True, values=["cpuA", "cpuB", "cpuC"]),
        "compiler": attr.string(mandatory=True, values=["compilerA", "compilerB", "compilerC"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
`}}

	for _, tc := range testCases {
		crosstool := makeCrosstool(tc.toolchains)
		got, err := Transform(crosstool)
		if err != nil {
			t.Fatalf("CROSSTOOL conversion failed: %v", err)
		}
		if !strings.Contains(got, tc.expectedText) {
			t.Errorf("Failed to correctly declare the rule, expected to contain:\n%v\n",
				tc.expectedText)
			t.Fatalf("Tested CROSSTOOL:\n%v\n\nGenerated rule:\n%v\n",
				strings.Join(tc.toolchains, "\n"), got)
		}
	}
}
