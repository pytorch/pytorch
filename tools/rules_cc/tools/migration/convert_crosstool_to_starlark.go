/*
The convert_crosstool_to_starlark script takes in a CROSSTOOL file and
generates a Starlark rule.

See https://github.com/bazelbuild/bazel/issues/5380

Example usage:
bazel run \
@rules_cc//tools/migration:convert_crosstool_to_starlark -- \
--crosstool=/path/to/CROSSTOOL \
--output_location=/path/to/cc_config.bzl
*/
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/user"
	"path"
	"strings"

	// Google internal base/go package, commented out by copybara
	"log"
	crosstoolpb "third_party/com/github/bazelbuild/bazel/src/main/protobuf/crosstool_config_go_proto"
	"github.com/golang/protobuf/proto"

	"tools/migration/crosstooltostarlarklib"
)

var (
	crosstoolLocation = flag.String(
		"crosstool", "", "Location of the CROSSTOOL file")
	outputLocation = flag.String(
		"output_location", "", "Location of the output .bzl file")
)

func toAbsolutePath(pathString string) (string, error) {
	usr, err := user.Current()
	if err != nil {
		return "", err
	}
	homeDir := usr.HomeDir

	if strings.HasPrefix(pathString, "~") {
		return path.Join(homeDir, pathString[1:]), nil
	}

	if path.IsAbs(pathString) {
		return pathString, nil
	}

	workingDirectory := os.Getenv("BUILD_WORKING_DIRECTORY")
	return path.Join(workingDirectory, pathString), nil
}

func main() {
	flag.Parse()

	if *crosstoolLocation == "" {
		log.Fatalf("Missing mandatory argument 'crosstool'")
	}
	crosstoolPath, err := toAbsolutePath(*crosstoolLocation)
	if err != nil {
		log.Fatalf("Error while resolving CROSSTOOL location:", err)
	}

	if *outputLocation == "" {
		log.Fatalf("Missing mandatory argument 'output_location'")
	}
	outputPath, err := toAbsolutePath(*outputLocation)
	if err != nil {
		log.Fatalf("Error resolving output location:", err)
	}

	in, err := ioutil.ReadFile(crosstoolPath)
	if err != nil {
		log.Fatalf("Error reading CROSSTOOL file:", err)
	}
	crosstool := &crosstoolpb.CrosstoolRelease{}
	if err := proto.UnmarshalText(string(in), crosstool); err != nil {
		log.Fatalf("Failed to parse CROSSTOOL:", err)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Error creating output file:", err)
	}
	defer file.Close()

	rule, err := crosstooltostarlarklib.Transform(crosstool)
	if err != nil {
		log.Fatalf("Error converting CROSSTOOL to a Starlark rule:", err)
	}

	if _, err := file.WriteString(rule); err != nil {
		log.Fatalf("Error converting CROSSTOOL to a Starlark rule:", err)
	}
	fmt.Println("Success!")
}
