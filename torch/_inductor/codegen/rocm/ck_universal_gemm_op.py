from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass
class CKGemmOperation:
    """
    A python dataclass storing the template parameters of a CK Universal Gemm template instance
    """

    a_layout: str
    b_layout: str
    c_layout: str

    a_element_dtype: str
    b_element_dtype: str
    c_element_dtype: str

    acc_dtype: str
    c_shuffle_dtype: str

    a_elementwise_op: str
    b_elementwise_op: str
    c_elementwise_op: str

    gemm_specialization: str

    block_size: int

    m_per_block: int
    n_per_block: int
    k_per_block: int

    a_k1: int
    b_k1: int

    m_per_xdl: int
    n_per_xdl: int

    m_xdl_per_wave: int
    n_xdl_per_wave: int

    a_block_transfer_thread_cluster_lengths_ak0_m_ak1: Tuple[int, int, int]
    a_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    a_block_transfer_src_access_order: Tuple[int, int, int]
    a_block_transfer_src_vector_dim: int
    a_block_transfer_src_scalar_per_vector: int
    a_block_transfer_dst_scalar_per_vector_ak1: int
    a_block_lds_extra_m: bool

    b_block_transfer_thread_cluster_lengths_bk0_n_bk1: Tuple[int, int, int]
    b_block_transfer_thread_cluster_arrange_order: Tuple[int, int, int]
    b_block_transfer_src_access_order: Tuple[int, int, int]

    b_block_transfer_src_vector_dim: int
    b_block_transfer_src_scalar_per_vector: int
    b_block_transfer_dst_scalar_per_vector_bk1: int
    b_block_lds_extra_n: bool

    c_shuffle_m_xdl_per_wave_per_shuffle: int
    c_shuffle_n_xdl_per_wave_per_shuffle: int

    c_shuffle_block_transfer_cluster_lengths_m_block_m_per_block_n_block_n_per_block: (
        Tuple[int, int, int, int]
    )
    c_shuffle_block_transfer_scalar_per_vector_n_per_block: int

    block_gemm_pipeline_scheduler: str
    block_gemm_pipeline_version: Optional[str]

    a_compute_dtype: Optional[str]
    b_compute_dtype: Optional[str]

    def name(self):
        # cpp alias for template instance
        return f"ck_devicegemm_xdl_shuffle_v3_{self.key_name()}"

    def key_name(self):
        # TBD; must be unique per instance. Intended to use as dict key
        return "_".join(
            [
                "K"
                + field_name.replace("_", "").lower()
                + "V"
                + (
                    "x".join(map(str, iter(field_value)))
                    if isinstance(field_value, tuple)
                    else str(field_value).replace(":", "")
                )
                for field_name, field_value in self.dict_items()
            ]
        )

    def dict_items(self):
        return asdict(self).items()
