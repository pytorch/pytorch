#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import subprocess
import sys

import torch.fx as fx

from common import main
from huggingface import HuggingfaceRunner
from timm_models import TimmRunner
from torchbench import setup_torchbench_cwd, TorchBenchmarkRunner


def run_model_suite(args):
    script_args = [
        f"--devices={args.device}",
        f"--{args.precision}",
        "--use-eval-mode",
        f"--output-directory={args.output_dir}",
    ]

    if args.device == "xla":
        script_args.append("--trace-on-xla")

    if args.backend and not args.print_fx:
        script_args.append(f"--backend={args.backend}")

    if args.training:
        script_args.append("--training")
    else:
        script_args.append("--inference")

    if args.performance:
        script_args.append("--performance")
    else:
        script_args.append("--accuracy")

    if args.print_fx:
        script_args.append("--print-fx")

    if args.no_skip:
        script_args.append("--no-skip")

    if args.model_suite.startswith("timm"):
        args.model_suite = "timm_models"

    script_args.insert(0, f"{args.model_suite}.py")
    script_args.insert(0, "python")
    script_args.append(f"--output={args.output}.csv")

    return script_args


def generate_hlo_files(args, model_name):
    env = os.environ
    if args.gen_hlo_files:
        hlo_dir = f"{args.output_dir}/{model_name}/hlo_files"
        os.makedirs(hlo_dir, exist_ok=True)
        env["XLA_HLO_DEBUG"] = "1"
        env["XLA_IR_DEBUG"] = "1"
        env["XLA_FLAGS"] = f"--xla_dump_to={hlo_dir}"
        env["XLA_SAVE_TENSORS_FILE"] = f"{hlo_dir}/{model_name}-tensors.hlo"
        env["XLA_SAVE_TENSORS_FMT"] = "hlo"
        env["XLA_SAVE_HLO_FILE"] = f"{hlo_dir}/{model_name}.hlo"
        env["XLA_DUMP_HLO_GRAPH"] = "1"
    return env


def run_script(script_args, mod_env, model_name, output_dir, filestring=""):
    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    with open(
        f"{output_dir}/{model_name}/{model_name}{filestring}.log", "w"
    ) as model_outfile:
        result = subprocess.run(
            script_args, env=mod_env, check=True, stdout=model_outfile
        )
    return result


def run_from_model_list(script_args, args):
    failed_models = dict()
    successful_models = dict()

    logging.info("\nStarting model training...\n")
    with open(args.model_list) as models:
        models_to_bs = dict()
        lines = models.readlines()
        for line in lines:
            split_line = line.split(",")
            models_to_bs[split_line[0]] = int(split_line[1])

        for model in models_to_bs:
            script_args.append(f"--only={model}")
            script_args.append(f"--batch-size={models_to_bs[model]}")
            logging.info("Running training on model %s", model)
            try:
                env = generate_hlo_files(args, model)
                run_script(script_args, env, model, args.output_dir)
            except Exception as e:
                failed_models[model] = (models_to_bs[model], e)
                logging.exception("Model %s failed during training", model)
                script_args = script_args[:-2]
                continue
            else:
                successful_models[model] = models_to_bs[model]
                script_args = script_args[:-2]
                continue

    logging.info("\nFinished model training.\n")
    return failed_models, successful_models


def classify_failed_models(failed_models):
    failed_cats = dict()
    for model in failed_models:
        for err in [AssertionError, AttributeError, ValueError]:
            if failed_models[model][1] == err:
                failed_cats = add_to_dict(
                    failed_cats, "accuracy", (model, failed_models[model][0])
                )
        if not isinstance(failed_models[model][1], str):
            failed_cats = add_to_dict(
                failed_cats, "compiler", (model, failed_models[model][0])
            )
    return failed_cats


def add_missed_failed_models(failed_models, successful_models, args):
    model_results = dict()
    with open(f"{args.output_dir}/{args.output}.csv") as results_file:
        lines = results_file.readlines()
        skip_first_line = True
        for line in lines:
            if not skip_first_line:
                split_line = line.split(",")
                model_results[split_line[1]] = (int(split_line[2]), split_line[3])
            else:
                skip_first_line = False

    for model in successful_models:
        if model not in model_results:
            failed_models = add_to_dict(
                failed_models, "compiler", (model, successful_models[model])
            )
        elif model in model_results:
            result = model_results[model][1]
            batch_size = model_results[model][0]
            result_tuple = (model, batch_size)
            if result == "fail" and result_tuple not in failed_models["accuracy"]:
                failed_models = add_to_dict(failed_models, "accuracy", result_tuple)
                if result_tuple in failed_models["compiler"]:
                    failed_models["compiler"].remove(result_tuple)

    return failed_models


def model_loader(args, model_name, batch_size):
    model_tuple = tuple()
    model_suite_name = args.model_suite.lower()

    args.output = f"{args.output}-model-load.csv"
    args.only = f"{model_name}"
    args.batch_size = int(batch_size)
    sys.argv = run_model_suite(args)
    sys.argv.pop(0)
    sys.argv.pop()
    sys.argv.append(f"--output={args.output}")
    sys.argv.append(f"--only={model_name}")
    sys.argv.append(f"--batch_size={batch_size}")

    if model_suite_name == "huggingface":
        runner = HuggingfaceRunner()
        main(runner)
    elif model_suite_name.startswith("timm"):
        runner = TimmRunner()
        main(runner)
    elif model_suite_name.startswith("torch"):
        torchbench_dir = setup_torchbench_cwd()
        runner = TorchBenchmarkRunner()
        main(runner, torchbench_dir)
    else:
        raise Exception(f"Invalid model model suite specified: {model_suite_name}.")

    try:
        model_tuple = runner.load_model(args.device, model_name, batch_size)
    except Exception as e:
        pass

    return model_tuple


def add_to_dict(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]
    return d


def extract_fx_graph(graph_module, args):
    if args.verbose:
        graph_module.graph.print_tabular()
    return graph_module.forward


def write_fx_to_file(args, model_name, batch_size):
    if args.print_fx:
        logging.info("Attempting to write FX IR for %s to file...", model_name)
        args.only = model_name
        args.batch_size = int(batch_size)
        model_characteristics = model_loader(args, model_name, batch_size)
        model = model_characteristics[2]
        inputs = model_characteristics[3]

        for module in model.modules():
            try:
                fx_graph = extract_fx_graph(fx.GraphModule(module), args)
                minified_models = add_to_dict(minified_models, model_name, fx_graph)
            except Exception as e:
                logging.exception("Could not generate fx graph for %s", model_name)
                continue
            else:
                continue

        if model_name in minified_models:
            with open(
                f"{args.output_dir}/{model_name}/{model_name}_fxir.log", "w"
            ) as fxir_file:
                for op in minified_models[model_name]:
                    fxir_file.write(f"{minified_models[model_name][op]}\n")
            logging.info("Successfully wrote FX IR for %s to file.", model_name)
        else:
            logging.error("Could not trace FX IR. Unable to write it to file.")


def fx_minifier(models, successful_models, script_args, args):
    cat_failed_models = classify_failed_models(models)
    cat_failed_models = add_missed_failed_models(
        cat_failed_models, successful_models, args
    )
    minified_models = dict()

    for category in cat_failed_models:
        error_type = category
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCHDYNAMO_REPRO_AFTER"] = "dynamo"
        os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "4" if error_type == "accuracy" else "1"
        for model_index in range(0, len(cat_failed_models[category])):
            model_name = cat_failed_models[category][model_index][0]
            batch_size = cat_failed_models[category][model_index][1]

            script_args.append(f"--only={model_name}")
            script_args.append(f"--batch-size={batch_size}")

            logging.info("Minifying model: %s", model_name)
            try:
                env = generate_hlo_files(args, model_name)
                run_script(script_args, env, model_name, args.output_dir, "-minified")
            except Exception as e:
                logging.exception("Minification failed for model: %s", model_name)
                script_args = script_args[:-2]
                continue
            else:
                if os.path.exists("repro.py"):
                    logging.info("Minification succeeded for model: %s", model_name)
                    shutil.move(
                        "repro.py",
                        f"{args.output_dir}/{model_name}/{model_name}_repro.py",
                    )
                    write_fx_to_file(args, model_name, batch_size)
                else:
                    logging.error("Minification failed for model: %s", model_name)
                script_args = script_args[:-2]
                continue


def start_minifier(failed_models, successful_models, script_args, args):
    logging.info("\nStarting minification...\n")
    fx_minifier(failed_models, successful_models, script_args, args)
    logging.info("\nFinished minification.")


def minifier_runner():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-suite",
        type=str,
        default="huggingface",
        choices=["huggingface", "timm", "torchbench"],
        help="Model suite name to run Dynamo benchmarks.",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        default="huggingface_models_list.txt",
        help="Path to list of models to run model training or inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="xla",
        choices=["xla", "cpu", "gpu"],
        help="Device type to run inference or training on.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=[None, "eager", "aot_eager", "inductor"],
        help="Non-XLA backend to run training on.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type used when running training. Default value is float32.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Specify to run models in inference mode.",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Specify to run models in training mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dynamo-benchmark-results",
        help="Name of output csv file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_output_dir",
        help="Name of output directory.",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run inference or training in performance mode",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run inference training in accuracy mode",
    )
    parser.add_argument(
        "--print-fx",
        action="store_true",
        help="Dump FX IR",
    )
    parser.add_argument(
        "--gen-hlo-files",
        action="store_true",
        help="Generate HLO files for each model",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Run models that are in the global skip list",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug printouts",
    )

    args = parser.parse_args()
    args.device = "cuda" if args.device.startswith("gpu") else args.device

    script_args = run_model_suite(args)
    failed_models, successful_models = run_from_model_list(script_args, args)
    start_minifier(failed_models, successful_models, script_args, args)


if __name__ == "__main__":
    minifier_runner()
