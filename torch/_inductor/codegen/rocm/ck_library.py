import logging
import os
import subprocess
from dataclasses import fields, replace
from functools import lru_cache, partial
from typing import List

from torch._inductor import config

from torch._inductor.codegen.rocm.ck_universal_gemm_op import CKGemmOperation

log = logging.getLogger(__name__)


def parse_instances(str_instances: List[str]) -> List[CKGemmOperation]:
    """
    Parse the lines containing Universal Gemm template instances into `CKGemmOperation` instances
    """

    def maybe_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    op_instances = []
    for line in str_instances:
        s_template_args = line.split("DeviceGemm_Xdl_CShuffleV3")[-1].strip("<>, ")
        template_args = []
        i_current = 0
        while i_current < len(s_template_args):
            if s_template_args[i_current] == " ":
                # skip whitespace
                i_current += 1
                continue
            elif s_template_args[i_current : i_current + 2] == "S<":
                # parse template S<Index...>
                i_next = s_template_args.find(">", i_current)
                template_args.append(
                    tuple(map(int, s_template_args[i_current + 2 : i_next].split(",")))
                )
                i_current = i_next + 2
            else:
                # all string attributes must be either type aliases or global constants in C++
                i_next = s_template_args.find(",", i_current)
                template_args.append(
                    maybe_int(
                        s_template_args[i_current : i_next if i_next != -1 else None]
                    )
                )
                if i_next != -1:
                    i_current = i_next + 1
            if i_next == -1:
                break
        # pad with `None`s for the fields which are not defined in the instance
        new_instance = CKGemmOperation(
            *template_args,  # type: ignore[arg-type]
            *((None,) * (len(fields(CKGemmOperation)) - len(template_args))),
        )
        # the last 2 template parameters are optional
        # if they are absent, substitute them with default values from Universal Gemm C++ template declaration
        if new_instance.a_compute_dtype is None:
            new_instance.a_compute_dtype = new_instance.c_element_dtype
        if new_instance.b_compute_dtype is None:
            new_instance.b_compute_dtype = new_instance.c_element_dtype

        op_instances.append(new_instance)
    return op_instances


def default_instances() -> List[CKGemmOperation]:
    # fallback: known working op instance for problem size M=2240 K=256 N=2048
    # all string attributes must be either type aliases or global constants in C++

    return [
        CKGemmOperation(
            a_layout="Row",
            b_layout="Row",
            c_layout="Row",
            a_element_dtype="F16",
            b_element_dtype="F16",
            c_element_dtype="F16",
            a_compute_dtype="F16",
            b_compute_dtype="F16",
            acc_dtype="F32",
            c_shuffle_dtype="F16",
            a_elementwise_op="PassThrough",
            b_elementwise_op="PassThrough",
            c_elementwise_op="PassThrough",
            gemm_specialization="GemmSpecialization::Default",
            block_size=256,
            m_per_block=224,
            n_per_block=256,
            k_per_block=64,
            a_k1=8,
            b_k1=2,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=7,
            n_xdl_per_wave=8,
            a_block_transfer_thread_cluster_lengths_ak0_m_ak1=(8, 32, 1),
            a_block_transfer_thread_cluster_arrange_order=(1, 0, 2),
            a_block_transfer_src_access_order=(1, 0, 2),
            a_block_transfer_src_vector_dim=2,
            a_block_transfer_src_scalar_per_vector=8,
            a_block_transfer_dst_scalar_per_vector_ak1=8,
            a_block_lds_extra_m=0,  # type: ignore[arg-type]
            b_block_transfer_thread_cluster_lengths_bk0_n_bk1=(8, 32, 1),
            b_block_transfer_thread_cluster_arrange_order=(0, 2, 1),
            b_block_transfer_src_access_order=(0, 2, 1),
            b_block_transfer_src_vector_dim=1,
            b_block_transfer_src_scalar_per_vector=8,
            b_block_transfer_dst_scalar_per_vector_bk1=2,
            b_block_lds_extra_n=0,  # type: ignore[arg-type]
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=2,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                32,
                1,
                8,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v3",
        )
    ]


@lru_cache(None)
def gen_ops_library() -> List[CKGemmOperation]:
    """
    Parse the Universal Gemm instances defined in the composable kernel library folder.
    """
    grep_result = subprocess.run(
        [
            "grep",
            "-inR",
            "DeviceGemm_Xdl_CShuffleV3",
            os.path.join(config.rocm.ck_dir, "library"),
        ],
        capture_output=True,
        text=True,
    )

    op_instances = parse_instances(grep_result.stdout.strip().split("\n"))

    log.debug("ck instances from library: %d", len(op_instances))

    schedulers = [
        "BlockGemmPipelineScheduler::Intrawave",
        "BlockGemmPipelineScheduler::Interwave",
    ]
    gemm_specs = [
        "GemmSpecialization::Default",
        "GemmSpecialization::MPadding",
        "GemmSpecialization::NPadding",
        "GemmSpecialization::KPadding",
        "GemmSpecialization::MNPadding",
        "GemmSpecialization::MKPadding",
        "GemmSpecialization::NKPadding",
        "GemmSpecialization::MNKPadding",
    ]

    # substitute templated args by looping through their domains
    substitute_instances = []
    for instance in op_instances:
        sub_scheduler = instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched"
        sub_spec = instance.gemm_specialization == "GemmSpec"
        schedulers_range = (
            schedulers if sub_scheduler else [instance.block_gemm_pipeline_scheduler]
        )
        spec_range = gemm_specs if sub_spec else [instance.gemm_specialization]
        for scheduler in schedulers_range:
            for spec in spec_range:
                substitute_instances.append(
                    replace(
                        instance,
                        block_gemm_pipeline_scheduler=scheduler,
                        gemm_specialization=spec,
                    )
                )

    return substitute_instances


@lru_cache(None)
def gen_ops_preselected() -> List[CKGemmOperation]:
    """
    Manually selected (through benchmarking) F16/F16/F16 Row/Col/Row instances
    """
    ck_gemm_f16_rcr = partial(
        CKGemmOperation,
        a_layout="Row",
        b_layout="Col",
        c_layout="Row",
        a_element_dtype="F16",
        b_element_dtype="F16",
        c_element_dtype="F16",
        acc_dtype="F32",
        c_shuffle_dtype="F16",
        a_elementwise_op="PassThrough",
        b_elementwise_op="PassThrough",
        c_elementwise_op="PassThrough",
        k_per_block=64,
        a_k1=8,
        b_k1=8,
        a_block_transfer_thread_cluster_arrange_order=(1, 0, 2),
        a_block_transfer_src_access_order=(1, 0, 2),
        a_block_transfer_src_vector_dim=2,
        a_block_transfer_src_scalar_per_vector=8,
        a_block_transfer_dst_scalar_per_vector_ak1=8,
        a_block_lds_extra_m=0,
        b_block_transfer_thread_cluster_arrange_order=(1, 0, 2),
        b_block_transfer_src_access_order=(1, 0, 2),
        b_block_transfer_src_vector_dim=2,
        b_block_transfer_src_scalar_per_vector=8,
        b_block_transfer_dst_scalar_per_vector_bk1=8,
        b_block_lds_extra_n=0,
        a_compute_dtype="F16",
        b_compute_dtype="F16",
    )
    ck_gemm_f16_rcr_compute_friendly = partial(
        ck_gemm_f16_rcr,
        block_size=256,
        a_block_transfer_thread_cluster_lengths_ak0_m_ak1=(8, 32, 1),
        b_block_transfer_thread_cluster_lengths_bk0_n_bk1=(8, 32, 1),
        c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
            1,
            32,
            1,
            8,
        ),
        c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
    )
    ck_gemm_f16_rcr_memory_friendly = partial(
        ck_gemm_f16_rcr,
        block_size=128,
        a_block_transfer_thread_cluster_lengths_ak0_m_ak1=(8, 16, 1),
        b_block_transfer_thread_cluster_lengths_bk0_n_bk1=(8, 16, 1),
        block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Interwave",
        block_gemm_pipeline_version="BlockGemmPipelineVersion::v2",
    )
    ck_gemm_f16_rcr_latency_friendly = partial(
        ck_gemm_f16_rcr,
        gemm_specialization="GemmSpecialization::Default",
        block_size=128,
        m_per_xdl=16,
        n_per_xdl=16,
        m_xdl_per_wave=1,
        n_xdl_per_wave=1,
        a_block_transfer_thread_cluster_lengths_ak0_m_ak1=(8, 16, 1),
        b_block_transfer_thread_cluster_lengths_bk0_n_bk1=(8, 16, 1),
        c_shuffle_m_xdl_per_wave_per_shuffle=1,
        c_shuffle_n_xdl_per_wave_per_shuffle=1,
        c_shuffle_block_transfer_scalar_per_vector_n_per_block=4,
        block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
        block_gemm_pipeline_version="BlockGemmPipelineVersion::v1",
    )
    return [
        ck_gemm_f16_rcr_compute_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=224,
            n_per_block=256,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=7,
            n_xdl_per_wave=8,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=2,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v3",
        ),
        ck_gemm_f16_rcr_compute_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=128,
            n_per_block=128,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=2,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v3",
        ),
        ck_gemm_f16_rcr_compute_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=128,
            n_per_block=128,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=2,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v4",
        ),
        ck_gemm_f16_rcr_compute_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=128,
            n_per_block=128,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=2,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v5",
        ),
        ck_gemm_f16_rcr_compute_friendly(
            gemm_specialization="GemmSpecialization::Default",
            m_per_block=128,
            n_per_block=128,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=2,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v3",
        ),
        ck_gemm_f16_rcr_compute_friendly(
            gemm_specialization="GemmSpecialization::Default",
            m_per_block=128,
            n_per_block=128,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=2,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v4",
        ),
        ck_gemm_f16_rcr_compute_friendly(
            gemm_specialization="GemmSpecialization::Default",
            m_per_block=128,
            n_per_block=128,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=2,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            block_gemm_pipeline_scheduler="BlockGemmPipelineScheduler::Intrawave",
            block_gemm_pipeline_version="BlockGemmPipelineVersion::v5",
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::Default",
            m_per_block=16,
            n_per_block=32,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=1,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                16,
                1,
                8,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=4,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=16,
            n_per_block=32,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=1,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                16,
                1,
                8,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=4,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=16,
            n_per_block=64,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=1,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=2,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                16,
                1,
                8,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=32,
            n_per_block=64,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=1,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                16,
                1,
                8,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=32,
            n_per_block=128,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=1,
            n_xdl_per_wave=2,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                16,
                1,
                8,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::Default",
            m_per_block=32,
            n_per_block=16,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=1,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                32,
                1,
                4,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=4,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=32,
            n_per_block=16,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=1,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                32,
                1,
                4,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=4,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=64,
            n_per_block=16,
            m_per_xdl=16,
            n_per_xdl=16,
            m_xdl_per_wave=2,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=2,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                64,
                1,
                2,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=64,
            n_per_block=32,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=1,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=1,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                32,
                1,
                4,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
        ),
        ck_gemm_f16_rcr_memory_friendly(
            gemm_specialization="GemmSpecialization::MNKPadding",
            m_per_block=128,
            n_per_block=32,
            m_per_xdl=32,
            n_per_xdl=32,
            m_xdl_per_wave=2,
            n_xdl_per_wave=1,
            c_shuffle_m_xdl_per_wave_per_shuffle=2,
            c_shuffle_n_xdl_per_wave_per_shuffle=1,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                32,
                1,
                4,
            ),
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=8,
        ),
        ck_gemm_f16_rcr_latency_friendly(
            m_per_block=16,
            n_per_block=32,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                16,
                1,
                8,
            ),
        ),
        ck_gemm_f16_rcr_latency_friendly(
            m_per_block=32,
            n_per_block=16,
            c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block=(
                1,
                32,
                1,
                4,
            ),
        ),
    ]
