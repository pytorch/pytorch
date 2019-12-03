import argparse

# Usage
# -----
#
# Run the benchmark and write the result to a json file:
#   python main.py benchmark output.json
#
# Compare a new benchmark result with a baseline and render it to a web page:
#   python main.py compare baseline.json new.json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark TensorIterator')
    subs = parser.add_subparsers(dest="command")
    benchmark_parser = subs.add_parser('benchmark', description='Benchmark TensorIterator and write result to a json file')
    benchmark_parser.add_argument('--more', action='store_true', help='Run more benchmarks than just the selected ones')
    benchmark_parser.add_argument('output', help='Name of the output json file')
    compare_parser = subs.add_parser('compare', description='Compare a new benchmark result with a baseline and render it to HTML')
    compare_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    compare_parser.add_argument('baseline', help='Name of the json file used as baseline')
    compare_parser.add_argument('new', help='Name of the json file for the new result')
    args = parser.parse_args()

    if args.command == 'benchmark':
        import benchmark
        benchmark.run(args.more)
        benchmark.dump(args.output)
    else:
        assert args.command == 'compare'
        import compare
        compare.serve(args.baseline, args.new, args.port)
