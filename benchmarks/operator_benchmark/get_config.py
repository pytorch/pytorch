#!/usr/bin/env python3
"""
Helper script to extract operator benchmark configuration.

This script can be used to get the operator benchmark tests list
and configuration in various formats (JSON, YAML, shell variables).
"""

import sys
import json
import argparse
from pathlib import Path

# Add the benchmarks directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from operator_tests_config import OPERATOR_BENCHMARK_TESTS, OPERATOR_BENCHMARK_CONFIG
except ImportError:
    print("Error: Could not import operator_tests_config.py", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract operator benchmark configuration")
    parser.add_argument("--format", choices=["json", "yaml", "shell", "list"], 
                      default="list", help="Output format")
    parser.add_argument("--tests-only", action="store_true", 
                      help="Output only the test names")
    
    args = parser.parse_args()
    
    if args.tests_only:
        data = OPERATOR_BENCHMARK_TESTS
    else:
        data = {
            "operator_benchmark_tests": OPERATOR_BENCHMARK_TESTS,
            "operator_benchmark_config": OPERATOR_BENCHMARK_CONFIG
        }
    
    if args.format == "json":
        print(json.dumps(data, indent=2))
    elif args.format == "yaml":
        import yaml
        print(yaml.dump(data, default_flow_style=False))
    elif args.format == "shell":
        if args.tests_only:
            print("OP_BENCHMARK_TESTS=\"" + " ".join(data) + "\"")
        else:
            print(f"OP_BENCHMARK_TESTS=\"{' '.join(data['operator_benchmark_tests'])}\"")
            print(f"OP_BENCHMARK_TAG_FILTER=\"{data['operator_benchmark_config']['tag_filter']}\"")
            print(f"OP_BENCHMARK_NAME=\"{data['operator_benchmark_config']['benchmark_name']}\"")
            print(f"OP_BENCHMARK_USE_COMPILE=\"{data['operator_benchmark_config']['use_compile']}\"")
    else:  # list format
        if args.tests_only:
            for test in data:
                print(test)
        else:
            print("Operator Benchmark Tests:")
            for test in data["operator_benchmark_tests"]:
                print(f"  - {test}")
            print("\nConfiguration:")
            for key, value in data["operator_benchmark_config"].items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
