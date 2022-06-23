#!/usr/bin/env python
import argparse
import json
import os
import logging
import pandas as pd

# process the chrome traces output by the pytorch profiler
# require the json input file's name to be in format {model_name}_chrome_trace_*.json
# the runtimes file should have format (model_name, time)

gpu_pids = []

def is_gpu_compute_event(event):
    global gpu_pids
    return "pid" in event and event["pid"] in gpu_pids and "ph" in event and event["ph"] == "X"

def get_events(filename):
    f = open(filename)
    data = json.load(f)
    events = data["traceEvents"]
    return events

def get_sorted_gpu_events(events):
    sorted_gpu_events = []
    for event in events:
        if(not is_gpu_compute_event(event)):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=lambda x: x["ts"])

def get_sorted_gpu_mm_conv_events(events):
    def is_mm_conv_event(event):
        return "name" in event and ("gemm" in event["name"] or "conv" in event["name"] 
        or "cutlass" in event["name"] or "wgrad" in event["name"])
    gpu_events = get_sorted_gpu_events(events)
    sorted_events = []
    for event in gpu_events:
        if(not is_mm_conv_event(event)):
            continue
        sorted_events.append(event)
    return sorted_events

def get_duration(sorted_gpu_events):
    event = sorted_gpu_events[0]
    current_end_time = event["ts"] + event["dur"]
    total_duration = event["dur"]
    for event in sorted_gpu_events[1:]:
        start_time = max(event["ts"], current_end_time)
        end_time = event["ts"] + event["dur"]
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration

def get_model_name(filename):
    _, tail = os.path.split(filename)
    modelname = tail[:tail.find("_chrome_trace")]
    return modelname

def get_total_length(run_times_df, modelname):
    return float(run_times_df[run_times_df["name"]==modelname]["runtime"])

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--runtime", "-runf", help="file name of the runtime file", required=True
    )
    group.add_argument(
        "--filename", "-f", action="append", help="a filename of the json file to process"
    )
    group.add_argument(
        "--folder", "-fd", help="a folder of the json files to process"
    )
    args = parser.parse_args()

    run_times_df = pd.read_csv(args.runtime)

    if args.filename:
        filenames = args.filename
    elif args.folder:
        filenames = []
        directory = args.folder
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and f.endswith(".json"):
                filenames.append(f)
    else:
        print("Please provide a filename or a folder name")

    print(f"modelname, GPU Utilization, MM and Conv time")

    for filename in filenames:
        try:
            events = get_events(filename)

            # get pids of GPU events
            global gpu_pids
            for event in events:
                if "name" not in event:
                    continue
                if event["name"] == 'process_labels' and "GPU" in event["args"]["labels"]:
                    gpu_pids.append(event["pid"])
            
            modelname = get_model_name(filename)
            total_length = get_total_length(run_times_df, modelname) * 1e6

            sorted_gpu_events = get_sorted_gpu_events(events)
            utilization = get_duration(sorted_gpu_events) / total_length 
            
            sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
            mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length

            print(f"{modelname}, {utilization}, {mm_conv_utilization}")
        except:
            logging.exception(f"{filename}, ERROR")
            print(f"{filename}, ERROR")


if __name__ == "__main__":
    main()
