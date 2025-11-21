# mypy: ignore-errors
from ..utils import try_import_cutlass


# copied / modified from original at
# https://github.com/NVIDIA/cutlass/blob/8783c41851cd3582490e04e69e0cd756a8c1db7f/tools/library/scripts/gemm_operation.py#L658

if try_import_cutlass():
    import enum

    from cutlass_library.gemm_operation import *  # noqa: F401, F403
    from cutlass_library.library import *  # noqa: F401, F403

    _LOGGER = logging.getLogger(__name__)

    class EmitGemmUniversal3xInstanceWithEVT:
        """Responsible for emitting a CUTLASS 3.x template definition"""

        def __init__(self, operation_suffix="", evt_name=None):
            self.operation_suffix = operation_suffix
            self.includes = [
                "cutlass/cutlass.h",
                "cutlass/gemm/gemm.h",
                "cutlass/numeric_types.h",
                "cutlass/gemm/kernel/gemm_universal.hpp",
                "cutlass/gemm/collective/collective_builder.hpp",
                "cutlass/epilogue/collective/collective_builder.hpp",
            ]
            self.builtin_epilogue_functor_template = """${epilogue_functor}<
            ${element_d},
            ${element_epilogue},
            ${element_c},
            ${element_epilogue}
            >"""

            self.evt_name = evt_name
            self.gemm_template = """
using ${operation_name}_epilogue =
typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_epi},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<${cluster_shape_m}, ${cluster_shape_n}, ${cluster_shape_k}>,
    ${epi_tile_mn},
    ${element_accumulator}, ${element_epilogue},
    ${element_c}, ${layout_c}, ${align_c},
    ${element_d}, ${layout_d}, ${align_d},
    ${epilogue_schedule},
    ${epilogue_functor}
>::CollectiveOp;

${mixed_dtype_prepare_code}

using ${operation_name}_mainloop =
typename cutlass::gemm::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_main},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator},
    cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
    cute::Shape<${cluster_shape_m}, ${cluster_shape_n}, ${cluster_shape_k}>,
    ${stages},
    ${kernel_schedule}
>::CollectiveOp;

// Gemm operator ${operation_name}
using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<
    ${problem_shape},
    ${operation_name}_mainloop,
    ${operation_name}_epilogue,
    ${tile_scheduler}>;

// Define named type
struct ${operation_name} :
public ${operation_name}_base { };

        """

        #
        def instance_template(self):
            return """
${compile_guard_start}
{
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
    manifest.append(
    new ${gemm_kind}<GemmKernel>("${operation_name}"));
}
${compile_guard_end}
        """

        def emit_block_scale_epilogue_functor(self, operation):
            block_scaled_template = """
            ${epilogue_functor}<
                ${epi_vs},
                ${element_d},
                ${element_accumulator},
                ${element_sfd},
                ${layout_sfd},
                ${element_c},
                ${element_scalar}
            >
            """
            block_scaled_values = {
                "epi_vs": str(operation.ScaleFactorVectorSize),
                "element_d": str(DataTypeTag[operation.D.element]),
                "element_sfd": str(DataTypeTag[operation.ScaleFactorD.element]),
                "layout_sfd": LayoutTag[operation.ScaleFactorD.layout],
                "epilogue_functor": EpilogueFunctor3xTag[
                    EpilogueFunctor3x.LinearCombinationBlockScaleFactor
                ],
                "element_accumulator": str(DataTypeTag[operation.accumulator_type()]),
                "element_scalar": str(DataTypeTag[operation.accumulator_type()]),
                "element_c": str(DataTypeTag[operation.C.element]),
            }
            return SubstituteTemplate(block_scaled_template, block_scaled_values)

        @staticmethod
        def pointerize_if_grouped(operation, layout):
            return layout if not is_grouped(operation.gemm_kind) else layout + "* "

        @staticmethod
        def problem_shape(operation):
            gemm_shape_type = "cute::Shape<int,int,int,int>"
            grouped_gemm_shape_type = "cute::Shape<int,int,int>"
            grouped_gemm_shape_type = (
                "cutlass::gemm::GroupProblemShape<" + grouped_gemm_shape_type + ">"
            )

            return (
                gemm_shape_type
                if not is_grouped(operation.gemm_kind)
                else grouped_gemm_shape_type
            )

        def emit(self, operation):
            """Given a gem operation, emits a template definition of the operation"""

            opcode_class_main = operation.tile_description.math_instruction.opcode_class
            opcode_class_epi = opcode_class_main

            tile_shape = operation.tile_description.tile_shape
            instruction_shape = (
                operation.tile_description.math_instruction.instruction_shape
            )
            cluster_m = operation.tile_description.cluster_shape[0]
            cluster_n = operation.tile_description.cluster_shape[1]

            tile_shape_m, tile_shape_n, tile_shape_k = tile_shape

            # account for static/dynamic cluster shapes
            cta_m = tile_shape[0] // cluster_m if cluster_m > 0 else tile_shape[0]
            cta_n = tile_shape[1] // cluster_n if cluster_n > 0 else tile_shape[1]

            # Shape passed to epilogue builder
            is_sm100_kernel = operation.arch == 100
            if is_sm100_kernel:
                cta_m_per_mma_instruction = (
                    2 if "2sm" in operation.procedural_name() else 1
                )
                if cluster_m <= 0:
                    cta_m = cta_m // cta_m_per_mma_instruction

                if opcode_class_main in [
                    OpcodeClass.TensorOp,
                    OpcodeClass.BlockScaledTensorOp,
                ]:
                    tile_shape_m = instruction_shape[0]
                    tile_shape_n = instruction_shape[1]

            # stage count set to zero indicates builder automatic stage selection
            if operation.tile_description.stages > 0:
                stage_count_string = f"cutlass::gemm::collective::StageCount<\
{str(operation.tile_description.stages)}>"
            else:
                stage_count_string = (
                    f"cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(\
sizeof(typename {str(operation.procedural_name())}_epilogue::SharedStorage))>"
                )

            epi_tile_mn = "cutlass::epilogue::collective::EpilogueTileAuto"

            (
                instance_layout_A,
                instance_layout_B,
                instance_layout_C,
                instance_layout_D,
            ) = (
                operation.A.layout,
                operation.B.layout,
                operation.C.layout,
                operation.D.layout,
            )

            # 3.0 profiler integration only supports trivial epilogues for now
            epilogue_vector_length = 1

            # Support built-in epilogue functors or user-defined functions
            if isinstance(operation.epilogue_functor, enum.Enum):
                values = {
                    "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
                    "epilogue_functor": EpilogueFunctor3xTag[
                        operation.epilogue_functor
                    ],
                }
                epilogue_functor = SubstituteTemplate(
                    self.builtin_epilogue_functor_template, values
                )

                if (
                    is_block_scaled(operation.gemm_kind)
                    and operation.ScaleFactorD.element != DataType.void
                ):
                    epilogue_functor = self.emit_block_scale_epilogue_functor(operation)
            else:
                epilogue_functor = self.epilogue_functor.emit_declaration()

            if (
                is_block_scaled(operation.gemm_kind)
                and operation.ScaleFactorD.element != DataType.void
            ):
                epilogue_functor = self.emit_block_scale_epilogue_functor(operation)

            #
            # Cutlass3x complex kernels' ElementA(B) is a tuple in collective mainloop builder,
            # e.g. cute::tuple<Element, Transform>, Transform : cute::identity / cute::conjugate.
            element_a = (
                DataTypeTag[operation.A.element]
                if not operation.is_complex()
                else f"cute::tuple<{str(DataTypeTag[operation.A.element])},\
{str(ComplexTransformTag3x[operation.A.complex_transform])}>"
            )
            element_b = (
                DataTypeTag[operation.B.element]
                if not operation.is_complex()
                else f"cute::tuple<{str(DataTypeTag[operation.B.element])},\
{str(ComplexTransformTag3x[operation.B.complex_transform])}>"
            )
            epilogue_schedule_type = EpilogueScheduleTag[operation.epilogue_schedule]

            if opcode_class_main == OpcodeClass.BlockScaledTensorOp:
                is_no_smem_epilogue = operation.epilogue_schedule in [
                    EpilogueScheduleType.NoSmemWarpSpecialized1Sm,
                    EpilogueScheduleType.NoSmemWarpSpecialized2Sm,
                ]
                grouped = is_grouped(operation.gemm_kind)
                if cta_n == 256 and operation.kernel_schedule == to_grouped_schedule(
                    KernelScheduleType.Nvf4TmaWarpSpecialized1SmSm100, grouped
                ):
                    epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
                    if not is_no_smem_epilogue:
                        epilogue_schedule_type = EpilogueScheduleTag[
                            to_grouped_schedule(
                                EpilogueScheduleType.TmaWarpSpecialized1Sm, grouped
                            )
                        ]
                if cta_n == 256 and operation.kernel_schedule == to_grouped_schedule(
                    KernelScheduleType.Nvf4TmaWarpSpecialized2SmSm100, grouped
                ):
                    epi_tile_mn = "cute::Shape<cute::_128,cute::_64>"
                    if not is_no_smem_epilogue:
                        epilogue_schedule_type = EpilogueScheduleTag[
                            to_grouped_schedule(
                                EpilogueScheduleType.TmaWarpSpecialized2Sm, grouped
                            )
                        ]
                element_a = f"cute::tuple<{str(element_a)},{str(DataTypeTag[operation.ScaleFactorA])}>"
                element_b = f"cute::tuple<{str(element_b)},{str(DataTypeTag[operation.ScaleFactorB])}>"

            operation_name_str = operation.procedural_name()
            layout_a_str = LayoutTag[instance_layout_A]
            layout_b_str = LayoutTag[instance_layout_B]
            mixed_dtype_prepare_code = ""
            if operation.mixed_input_mode is not None:
                A_dtype = operation.A.element
                B_dtype = operation.B.element
                A_dtype_bits = DataTypeSize[A_dtype]
                B_dtype_bits = DataTypeSize[B_dtype]
                is_A_dtype_narrow = A_dtype_bits < B_dtype_bits
                if is_A_dtype_narrow:
                    narrow_dtype, wide_dtype = (A_dtype, B_dtype)
                    narrow_dtype_bits, wide_dtype_bits = (A_dtype_bits, B_dtype_bits)
                else:
                    narrow_dtype, wide_dtype = (B_dtype, A_dtype)
                    narrow_dtype_bits, wide_dtype_bits = (B_dtype_bits, A_dtype_bits)

                narrow_tag = DataTypeTag[narrow_dtype]
                wide_tag = DataTypeTag[wide_dtype]
                scale_tag = DataTypeTag[wide_dtype]
                zero_tag = DataTypeTag[wide_dtype]

                do_shuffle = False
                value_shuffle_str = ""
                if narrow_dtype_bits == 4 and wide_dtype_bits == 16:
                    value_shuffle_str = "cute::Layout<cute::Shape<cute::_2,cute::_4>, \
cute::Stride<cute::_4,cute::_1>>"
                    do_shuffle = True
                if narrow_dtype_bits == 8 and wide_dtype_bits == 16:
                    value_shuffle_str = "cute::Layout<cute::Shape<cute::_2,cute::_2>, \
cute::Stride<cute::_2,cute::_1>>"
                    do_shuffle = True
                do_shuffle = operation.mixed_input_shuffle and do_shuffle

                if do_shuffle:
                    if is_A_dtype_narrow:
                        stride_narrow_str = (
                            f"cutlass::detail::TagToStrideA_t<{layout_a_str}>"
                        )
                        layout_a_str = f"{operation_name_str}_LayoutNarrowReordered"
                    else:
                        stride_narrow_str = (
                            f"cutlass::detail::TagToStrideB_t<{layout_b_str}>"
                        )
                        layout_b_str = f"{operation_name_str}_LayoutNarrowReordered"
                    # The {operation_name_str}_ prefixs in mixed_dtype_prepare_code and
                    # layout_{a, b}_str are to prevent errors in Windows platform unity build
                    mixed_dtype_prepare_code = f"""
            using {operation_name_str}_StrideNarrow = {stride_narrow_str};
            using {operation_name_str}_ValueShuffle = {value_shuffle_str};
            static constexpr int {operation_name_str}_NumShuffleAtoms = 1;
            using {operation_name_str}_MmaAtomShape = \
cute::Layout<cute::Shape<cute::_1, cute::Int<{operation_name_str}_NumShuffleAtoms>>>;
            using {operation_name_str}_LayoutAtomQuant = \
decltype(cutlass::compute_memory_reordering_atom<{wide_tag}, {operation_name_str}_MmaAtomShape, \
{operation_name_str}_ValueShuffle>());
            using {operation_name_str}_LayoutNarrowReordered = \
decltype(cute::tile_to_shape({operation_name_str}_LayoutAtomQuant{{}}, \
cute::Layout<cute::Shape<int,int,int>, {operation_name_str}_StrideNarrow>{{}}));
                    """

                mixed_input_modes_to_element = {
                    MixedInputMode.ConvertOnly: narrow_tag,
                    MixedInputMode.ScaleOnly: f"cute::tuple<{narrow_tag}, {scale_tag}>",
                    MixedInputMode.ScaleWithZeroPoint: f"cute::tuple<{narrow_tag}, {scale_tag}, {zero_tag}>",
                }
                narrow_element = mixed_input_modes_to_element.get(
                    operation.mixed_input_mode, narrow_tag
                )

                if narrow_dtype == DataType.s4 and (
                    wide_dtype == DataType.e4m3 or wide_dtype == DataType.e5m2
                ):
                    narrow_element = (
                        f"cute::tuple<{narrow_tag}, cutlass::Array<{scale_tag}, 8>>"
                    )

                if is_A_dtype_narrow:
                    element_a = narrow_element
                else:
                    element_b = narrow_element

            if self.evt_name:
                epilogue_functor = self.evt_name

            values = {
                "operation_name": operation_name_str,
                "operation_suffix": self.operation_suffix,
                "problem_shape": self.problem_shape(operation),
                "element_a": element_a,
                "layout_a": self.pointerize_if_grouped(operation, layout_a_str),
                "element_b": element_b,
                "layout_b": self.pointerize_if_grouped(operation, layout_b_str),
                "element_c": DataTypeTag[operation.C.element],
                "layout_c": self.pointerize_if_grouped(
                    operation, LayoutTag[instance_layout_C]
                ),
                "element_d": DataTypeTag[operation.D.element],
                "layout_d": self.pointerize_if_grouped(
                    operation, LayoutTag[instance_layout_D]
                ),
                "element_accumulator": DataTypeTag[operation.accumulator_type()],
                "opcode_class_main": OpcodeClassTag[opcode_class_main],
                "opcode_class_epi": OpcodeClassTag[opcode_class_epi],
                "arch": f"cutlass::arch::Sm{operation.arch}",
                "tile_shape_m": str(tile_shape_m),
                "tile_shape_n": str(tile_shape_n),
                "tile_shape_k": str(tile_shape_k),
                "cluster_shape_m": "cute::_"
                + str(operation.tile_description.cluster_shape[0])
                if operation.tile_description.cluster_shape[0] > 0
                else "int",
                "cluster_shape_n": "cute::_"
                + str(operation.tile_description.cluster_shape[1])
                if operation.tile_description.cluster_shape[1] > 0
                else "int",
                "cluster_shape_k": "cute::_"
                + str(operation.tile_description.cluster_shape[2])
                if operation.tile_description.cluster_shape[2] > 0
                else "int",
                "instruction_shape_m": str(instruction_shape[0]),
                "instruction_shape_n": str(instruction_shape[1]),
                "instruction_shape_k": str(instruction_shape[2]),
                "kernel_schedule": str(KernelScheduleTag[operation.kernel_schedule]),
                "epilogue_schedule": str(epilogue_schedule_type),
                "epi_tile_mn": epi_tile_mn,
                "epilogue_functor": epilogue_functor,
                "stages": stage_count_string,
                "align_a": str(operation.A.alignment),
                "align_b": str(operation.B.alignment),
                "align_c": str(operation.C.alignment),
                "align_d": str(operation.D.alignment),
                "transform_a": ComplexTransformTag[operation.A.complex_transform],
                "transform_b": ComplexTransformTag[operation.B.complex_transform],
                "math_operation": MathOperationTag[
                    operation.tile_description.math_instruction.math_operation
                ],
                "epilogue_vector_length": str(epilogue_vector_length),
                "element_epilogue": str(DataTypeTag[operation.element_epilogue]),
                "tile_scheduler": str(TileSchedulerTag[operation.tile_scheduler]),
                "mixed_dtype_prepare_code": mixed_dtype_prepare_code,
            }

            return SubstituteTemplate(self.gemm_template, values)
