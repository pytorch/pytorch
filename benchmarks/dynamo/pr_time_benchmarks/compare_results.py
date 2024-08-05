import csv
import sys


def read_data(file_name):
    data = {}
    with open(file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            benchmark_name = row[0]
            metric = row[1]
            number = float(row[2])
            if benchmark_name not in data:
                data[benchmark_name] = {}
            data[benchmark_name][metric] = number
    return data


def compute_percentage_difference(data1, data2, output_file):
    fail = False
    output_csv = open(output_file, "a", newline="")
    writer = csv.writer(output_csv)
    intersection_benchmarks = set(data1.keys()).intersection(set(data2.keys()))
    for benchmark_name in intersection_benchmarks:
        intersection_metrics = set(data1[benchmark_name].keys()).intersection(
            set(data2[benchmark_name].keys())
        )
        for metric in intersection_metrics:
            num1 = data1[benchmark_name][metric]
            num2 = data2[benchmark_name][metric]
            diff = ((num2 - num1) / num1) * 100 if num1 != 0 else 0
            print(f"{benchmark_name}, {metric}, {diff:.2f}%")
            writer.writerow((benchmark_name, metric, diff))
            if diff > 3.0:
                print(f"{benchmark_name}, {metric}, failed")
                fail = True

    if fail:
        sys.exit(1)
    else:
        print("all benchmarks passed")


def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]

    data1 = read_data(file1)
    data2 = read_data(file2)
    compute_percentage_difference(data1, data2, output_file)


if __name__ == "__main__":
    main()
