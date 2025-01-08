# mypy: allow-untyped-defs
from ..cutlass_utils import try_import_cutlass


if try_import_cutlass():
    import enum

    from cutlass_library.gemm_operation import *  # noqa: F401, F403
    from cutlass_library.library import *  # noqa: F401, F403

    # copied / modified from original at
    # https://github.com/NVIDIA/cutlass/blob/8783c41851cd3582490e04e69e0cd756a8c1db7f/tools/library/scripts/gemm_operation.py#L658
    # to support EVT similar to
    # https://github.com/NVIDIA/cutlass/blob/8783c41851cd3582490e04e69e0cd756a8c1db7f/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu#L315C69-L315C69  # noqa: B950
    class EmitGemmUniversal3xInstanceWithEVT:
        """Responsible for emitting a CUTLASS 3.x template definition"""

        def __init__(self, operation_suffix="") -> None:
            self.operation_suffix = operation_suffix
            self.includes = [
                "cutlass/cutlass.h",
                "cutlass/gemm/gemm.h",
                "cutlass/numeric_types.h",
                "cutlass/gemm/kernel/gemm_universal.hpp",
                "cutlass/gemm/collective/collective_builder.hpp",
                "cutlass/epilogue/collective/collective_builder.hpp",
            ]
            self.builtin_epilogue_functor_template = """
            ${epilogue_functor}<
              ${element_c},
              ${epilogue_vector_length},
              ${element_accumulator},
              ${element_epilogue}
            >
        """
            self.gemm_template = """
        using EpilogueScheduleType = ${epilogue_schedule};
        static_assert(cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecialized> ||
                 cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecializedCooperative>,
                "Epilogue visitor trees are currently only supported by the TMA warp-specialized epilogue");
        static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
        using ElementAcc = ${element_accumulator};
        using ElementD = ${element_d};
        ${epilogue_functor};
        using ${operation_name}_epilogue =
          typename cutlass::epilogue::collective::CollectiveBuilder<
            ${arch}, ${opcode_class},
            cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
            cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ${element_accumulator}, ${element_epilogue},
            ${element_c}, ${layout_c}, ${align_c},
            ${element_d}, ${layout_d}, ${align_d},
            EpilogueScheduleType,
            ${operation_name}_epilogue_functor
          >::CollectiveOp;

        using ${operation_name}_mainloop =
          typename cutlass::gemm::collective::CollectiveBuilder<
            ${arch}, ${opcode_class},
            ${element_a}, ${layout_a}, ${align_a},
            ${element_b}, ${layout_b}, ${align_b},
            ${element_accumulator},
            cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,
            cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,
            ${stages},
          ${kernel_schedule}
          >::CollectiveOp;

        // Gemm operator ${operation_name}
        using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<
            cute::Shape<int,int,int,int>,
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
          using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
          manifest.append(
            new ${gemm_kind}<GemmKernel>("${operation_name}"));
        ${compile_guard_end}
        """

        #
        def emit(self, operation):
            tile_shape = operation.tile_description.tile_shape
            warp_count = operation.tile_description.warp_count
            # stage count set to zero indicates builder automatic stage selection
            if operation.tile_description.stages > 0:
                stage_count_string = f"cutlass::gemm::collective::StageCount<{str(operation.tile_description.stages)}>"
            else:
                stage_count_string = f"cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename {str(operation.procedural_name())}_epilogue::SharedStorage)>"  # noqa: B950
            warp_shape = [tile_shape[idx] // warp_count[idx] for idx in range(3)]

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
                    "epilogue_vector_length": str(epilogue_vector_length),
                    "element_epilogue": str(DataTypeTag[operation.element_epilogue]),  # type: ignore[name-defined]
                    "epilogue_functor": EpilogueFunctorTag[operation.epilogue_functor],  # type: ignore[name-defined]
                }
                epilogue_functor = SubstituteTemplate(  # type: ignore[name-defined]
                    self.builtin_epilogue_functor_template, values
                )

            elif callable(operation.epilogue_functor):
                epilogue_functor = operation.epilogue_functor(
                    operation.procedural_name() + "_epilogue_functor"
                )
            else:
                epilogue_functor = str(operation.epilogue_functor)
            #

            values = {
                "operation_name": operation.procedural_name(),
                "operation_suffix": self.operation_suffix,
                "element_a": DataTypeTag[operation.A.element],  # type: ignore[name-defined]
                "layout_a": LayoutTag[instance_layout_A],  # type: ignore[name-defined]
                "element_b": DataTypeTag[operation.B.element],  # type: ignore[name-defined]
                "layout_b": LayoutTag[instance_layout_B],  # type: ignore[name-defined]
                "element_c": DataTypeTag[operation.C.element],  # type: ignore[name-defined]
                "layout_c": LayoutTag[instance_layout_C],  # type: ignore[name-defined]
                "element_d": DataTypeTag[operation.D.element],  # type: ignore[name-defined]
                "layout_d": LayoutTag[instance_layout_D],  # type: ignore[name-defined]
                "element_accumulator": DataTypeTag[operation.accumulator_type()],  # type: ignore[name-defined]
                "opcode_class": OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],  # type: ignore[name-defined] # noqa: B950
                "arch": "cutlass::arch::Sm%d" % operation.arch,
                "tile_shape_m": str(operation.tile_description.tile_shape[0]),
                "tile_shape_n": str(operation.tile_description.tile_shape[1]),
                "tile_shape_k": str(operation.tile_description.tile_shape[2]),
                "cluster_m": str(operation.tile_description.cluster_shape[0]),
                "cluster_n": str(operation.tile_description.cluster_shape[1]),
                "cluster_k": str(operation.tile_description.cluster_shape[2]),
                "warp_shape_m": str(warp_shape[0]),
                "warp_shape_n": str(warp_shape[1]),
                "warp_shape_k": str(warp_shape[2]),
                "instruction_shape_m": str(
                    operation.tile_description.math_instruction.instruction_shape[0]
                ),
                "instruction_shape_n": str(
                    operation.tile_description.math_instruction.instruction_shape[1]
                ),
                "instruction_shape_k": str(
                    operation.tile_description.math_instruction.instruction_shape[2]
                ),
                "kernel_schedule": str(KernelScheduleTag[operation.kernel_schedule]),  # type: ignore[name-defined]
                "epilogue_schedule": str(EpilogueScheduleTag[operation.epilogue_schedule]),  # type: ignore[name-defined]
                "epilogue_functor": epilogue_functor,
                "stages": stage_count_string,
                "align_a": str(operation.A.alignment),
                "align_b": str(operation.B.alignment),
                "align_c": str(operation.C.alignment),
                "align_d": str(operation.C.alignment),
                "transform_a": ComplexTransformTag[operation.A.complex_transform],  # type: ignore[name-defined]
                "transform_b": ComplexTransformTag[operation.B.complex_transform],  # type: ignore[name-defined]
                "math_operation": MathOperationTag[  # type: ignore[name-defined]
                    operation.tile_description.math_instruction.math_operation
                ],
                "epilogue_vector_length": str(epilogue_vector_length),
                "element_epilogue": str(DataTypeTag[operation.element_epilogue]),  # type: ignore[name-defined]
                "tile_scheduler": str(TileSchedulerTag[operation.tile_scheduler]),  # type: ignore[name-defined]
            }

            return SubstituteTemplate(self.gemm_template, values)  # type: ignore[name-defined]
