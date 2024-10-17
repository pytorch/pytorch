"""
This script uses linear programming to analyze outputs of triton mm config tuning.
To generate output that can be fed into this script set the env varTORCHINDUCTOR_MM_LOGGING_FILE.

That file can be fed into this script to generate the minimizes total, weighted matmul time as a function of allowed templates.
"""

import json

import click
import pulp


def parse_log_file(file_path):
    with open(file_path) as f:
        logs = json.load(f)

    occurrence_count = {}
    benchmark_logs = {}

    # Parse the logs
    for entry in logs:
        if "invoke" in entry:
            shape = entry["invoke"]
            if shape not in occurrence_count:
                occurrence_count[shape] = 0
            occurrence_count[shape] += 1
        else:
            for shape, timings in entry.items():
                if shape not in benchmark_logs:
                    benchmark_logs[shape] = []
                benchmark_logs[shape].extend(timings)

    return occurrence_count, benchmark_logs


def optimize_templates(N, occurrence_count, benchmark_logs, verbose=False):
    # Set of all possible Triton templates keyed by their attributes
    triton_templates = set()
    for timings in benchmark_logs.values():
        for timing in timings:
            if timing["type"] == "triton":
                triton_templates.add(
                    (
                        timing["BLOCK_M"],
                        timing["BLOCK_N"],
                        timing["BLOCK_K"],
                        timing["num_stages"],
                        timing["num_warps"],
                    )
                )

    # Print the initial data
    if verbose:
        print("Occurrence Count:", occurrence_count)
        print("Triton Templates:", triton_templates)

    # Create a dictionary to store template selection variables
    template_vars = {
        template: pulp.LpVariable(f"Template_{template}", 0, 1, pulp.LpBinary)
        for template in triton_templates
    }

    # Variables to select specific timing option for each shape
    selection_vars = {
        (shape, "cublas"): pulp.LpVariable(
            f"Select_{shape}_cublas", 0, 1, pulp.LpBinary
        )
        for shape in occurrence_count
    }
    for shape in occurrence_count:
        for template in triton_templates:
            selection_vars[(shape, template)] = pulp.LpVariable(
                f"Select_{shape}_{template}", 0, 1, pulp.LpBinary
            )

    # Variables for the total time for each shape
    min_time_vars = pulp.LpVariable.dicts(
        "MinTime", occurrence_count.keys(), 0, None, pulp.LpContinuous
    )

    # Define the problem
    prob = pulp.LpProblem("MatrixMultiplicationOptimization", pulp.LpMinimize)

    # Objective: Minimize the weighted total time
    prob += pulp.lpSum(
        [occurrence_count[shape] * min_time_vars[shape] for shape in occurrence_count]
    )

    # Constraints to select exactly N templates
    prob += pulp.lpSum([template_vars[template] for template in triton_templates]) == N

    # Store triton options per shape for debugging
    triton_options_per_shape = {}

    # Constraints for the total time for each shape
    for shape in occurrence_count:
        # Get cuBLAS time
        cublas_times = [
            timing["time"]
            for timing in benchmark_logs[shape]
            if timing["type"] == "cublas"
        ]
        min_cublas_time = min(cublas_times)

        # Collect Triton options
        triton_options = []
        for template in triton_templates:
            triton_times = [
                timing["time"]
                for timing in benchmark_logs[shape]
                if timing["type"] == "triton"
                and (
                    timing["BLOCK_M"],
                    timing["BLOCK_N"],
                    timing["BLOCK_K"],
                    timing["num_stages"],
                    timing["num_warps"],
                )
                == template
            ]
            if triton_times:
                min_triton_time = min(triton_times)
                triton_options.append((min_triton_time, template))

        # Save triton options for debugging
        triton_options_per_shape[shape] = triton_options

        # Ensure exactly one timing option is selected for each shape
        prob += (
            pulp.lpSum(
                [selection_vars[(shape, "cublas")]]
                + [
                    selection_vars[(shape, template)]
                    for triton_time, template in triton_options
                ]
            )
            == 1
        )

        # Ensure min_time_vars[shape] matches the selected timing option
        prob += min_time_vars[shape] == (
            selection_vars[(shape, "cublas")] * min_cublas_time
            + pulp.lpSum(
                [
                    selection_vars[(shape, template)] * triton_time
                    for triton_time, template in triton_options
                ]
            )
        )

        # Ensure Triton templates can only be selected if they are included in the N allowed templates
        for triton_time, template in triton_options:
            prob += selection_vars[(shape, template)] <= template_vars[template]

    # Print the constraints
    if verbose:
        print("Constraints:")
        for constraint in prob.constraints.values():
            print(constraint)

    # Solve the problem with suppressed output
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Output the selected templates and their configurations
    selected_templates = [
        template
        for template in triton_templates
        if pulp.value(template_vars[template]) == 1
    ]
    total_time = sum(
        pulp.value(min_time_vars[shape]) * occurrence_count[shape]
        for shape in occurrence_count
    )

    # Print the values of the decision variables after solving
    if verbose:
        print("Decision Variable Values:")
        for var in prob.variables():
            print(f"{var.name} = {var.varValue}")

    # # Debugging information
    if verbose:
        for shape in occurrence_count:
            print(f"Shape: {shape}")
            print(f"  Min Time: {pulp.value(min_time_vars[shape])}")
            print(f"  Occurrences: {occurrence_count[shape]}")
            print(
                f"  Min CuBLAS Time: {min_cublas_time} Selected: {pulp.value(selection_vars[(shape, 'cublas')])}"
            )
            for triton_time, template in triton_options_per_shape[shape]:
                print(
                    f"  Triton Template: {template} Time: {triton_time} Selected: {pulp.value(selection_vars[(shape, template)])}"
                )

    return selected_templates, total_time


# Main code to parse the log file and optimize templates
@click.command()
@click.argument("filename")
@click.option("--min-templates", default=0, help="Minimum number of templates.")
@click.option("--max-templates", default=10, help="Maximum number of templates.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def main(filename, min_templates, max_templates, verbose):
    occurrence_count, benchmark_logs = parse_log_file(filename)
    times = []
    for N in range(min_templates, max_templates + 1):
        selected_templates, total_time = optimize_templates(
            N, occurrence_count, benchmark_logs, verbose
        )
        print(f"N = {N}")
        print(f"Selected Templates: {selected_templates}")
        print(f"Total Weighted Time: {total_time}")
        times.append(total_time)
    print(times)


if __name__ == "__main__":
    main()
