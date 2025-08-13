import argparse
import os
import sys
import textwrap

import pandas as pd


# Hack to have something similar to DISABLED_TEST. These models are flaky.

flaky_models = {
    "yolov3",
    "gluon_inception_v3",
    "detectron2_maskrcnn_r_101_c4",
    "timm_efficientnet",  # see https://github.com/pytorch/pytorch/issues/148699
    "XGLMForCausalLM",  # discovered in https://github.com/pytorch/pytorch/pull/128148
    "moondream",  # discovered in https://github.com/pytorch/pytorch/pull/159291
}


def get_field(csv, model_name: str, field: str):
    try:
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception:
        return None


def check_accuracy(actual_csv, expected_csv, expected_filename):
    failed = []
    improved = []

    if "rocm" in expected_filename:
        flaky_models.update(
            {
                "alexnet",
                "cait_m36_384",
                "demucs",
                "densenet121",
                "detectron2_fcos_r_50_fpn",
                "doctr_det_predictor",
                "doctr_reco_predictor",
                "hf_BigBird",
                "hf_Longformer",
                "hf_Reformer",
                "hf_Roberta_base",
                "hf_T5",
                "hf_T5_base",
                "levit_128",
                "llava",
                "microbench_unbacked_tolist_sum",
                "mnasnet1_0",
                "mobilenet_v2",
                "pytorch_CycleGAN_and_pix2pix",
                "pytorch_stargan",
                "resnet152",
                "resnet18",
                "resnet50",
                "resnext50_32x4d",
                "sam",
                "sam_fast",
                "shufflenet_v2_x1_0",
                "squeezenet1_1",
                "stable_diffusion_text_encoder",
                "stable_diffusion_unet",
                "timm_efficientdet",
                "timm_efficientnet",
                "timm_nfnet",
                "timm_regnet",
                "timm_resnest",
                "timm_vovnet",
                "torchrec_dlrm",
                "vgg16",
            }
        )

    for model in actual_csv["name"]:
        accuracy = get_field(actual_csv, model, "accuracy")
        expected_accuracy = get_field(expected_csv, model, "accuracy")

        if accuracy == expected_accuracy:
            status = "PASS" if expected_accuracy == "pass" else "XFAIL"
            print(f"{model:34}  {status}")
            continue
        elif model in flaky_models:
            if accuracy == "pass":
                # model passed but marked xfailed
                status = "PASS_BUT_FLAKY:"
            else:
                # model failed but marked passe
                status = "FAIL_BUT_FLAKY:"
        elif accuracy != "pass":
            status = "FAIL:"
            failed.append(model)
        else:
            status = "IMPROVED:"
            improved.append(model)
        print(
            f"{model:34}  {status:9} accuracy={accuracy}, expected={expected_accuracy}"
        )

    msg = ""
    if failed or improved:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have accuracy status regressed:
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have accuracy status improved:
                {" ".join(improved)}

            """
            )
        sha = os.getenv("SHA1", "{your CI commit sha}")
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
        )
    return failed or improved, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)

    failed, msg = check_accuracy(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
