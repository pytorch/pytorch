# Patch to cutlass_library.generator

from cutlass_library.generator import *
def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    tile_schedulers=[TileSchedulerType.Persistent]):

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedules:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, complex_transforms, schedules, tile_schedulers)
  for layout, tile_description, data_type, complex_transform, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    if tile_description.threadblock_shape[0] < 64:
      continue
    if tile_description.threadblock_shape[0]<128 and kernel_schedule == KernelScheduleType.TmaWarpSpecializedCooperative:
      continue # TmaWarpSpecializedCooperative does not support M<128
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1], complex_transform[0])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1], complex_transform[1])

    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    extra_args = {}
    gemm_kind = GemmKind.Universal3x
    element_compute = data_type.get("epi_type", data_type["acc_type"])

    operation = GemmOperation(
        gemm_kind, tile_description.minimum_compute_capability,
        tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
        kernel_schedule, epilogue_schedule, tile_scheduler, extra_args)

    manifest.append(operation)
    operations.append(operation)

  return operations


def GenerateSM90_TensorOp_16b_WGMMA_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [32, 32, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 64, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 32, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [32, 64, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 64, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.bf16, DataType.bf16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    tile_descriptions_small = [
      # Not compatible with TmaWarpSpecializedCooperative
      TileDescription(
        [math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2] * 2],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1, 1, 1]),
      TileDescription(
        [math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2] * 4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1, 1, 1]),
      TileDescription(
        [math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2] * 2],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2, 1, 1]),
      TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
       0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
       0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    tile_descriptions_medium = [
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    tile_descriptions_large = [
      TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0]*4, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,2,1]),
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1]*2, math_inst.instruction_shape[2]*4],
        0, [4, 2, 1], math_inst, min_cc, max_cc, [2,1,1]),
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1]*2, math_inst.instruction_shape[2]*4],
        0, [4, 2, 1], math_inst, min_cc, max_cc, [1,2,1]),
    ]
    tile_descriptions = tile_descriptions_small + tile_descriptions_medium + tile_descriptions_large

    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format.
    for layout in layouts:
      if data_type["c_type"] in [DataType.s32, DataType.f32]:
        layout[2][1] = 4
      elif data_type["c_type"] in [DataType.f16, DataType.bf16]:
        layout[2][1] = 8

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      schedules = [
        [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized],
        [KernelScheduleType.TmaWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
      ]
      stream_k_schedules = [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]]
    else:
      schedules = [
        [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
        [KernelScheduleType.TmaWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
        # TmaWarpSpecializedCooperative and TmaWarpSpecializedPingpong require CUDA version >= 12.1 for optimal performance.
      ]
      stream_k_schedules = []

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      # Add stream-K variants
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])

    # persistent kernels with TMA epilogues
    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      # not enough smem for 256x128 f32 out with C allocation
      if data_type["d_type"] == DataType.f32:
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_medium, data_type,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions_medium, data_type,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])
      else:
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])

      # Emit instance without C allocation + load
      data_type["c_type"] = DataType.void
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
         [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
        [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
        tile_schedulers=[TileSchedulerType.StreamK])

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format.
      for layout in layouts:
        if data_type_mixed["c_type"] in [DataType.s32, DataType.f32]:
          layout[2][1] = 4
        elif data_type_mixed["c_type"] in [DataType.f16, DataType.bf16]:
          layout[2][1] = 8

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])

      # persistent kernels with TMA epilogues
      if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])

        # Emit instance without C allocation+load
        data_type_mixed["c_type"] = DataType.void
        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
           [KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

        CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed,
          [[KernelScheduleType.TmaWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
          tile_schedulers=[TileSchedulerType.StreamK])

#
def GenerateSM90_TensorOp_16b_WGMMA_alignx_gemm(manifest, cuda_version):
  if not CudaToolkitVersionSatisfies(cuda_version, 12, 0):
    return

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 4], [LayoutType.RowMajor,    4], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 2], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 2], [LayoutType.RowMajor,    2], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [32, 64, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 32, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 64, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f16,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
    MathInstruction(
      [64, 128, 16],
      DataType.bf16, DataType.bf16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  min_cc = 90
  max_cc = 90

  for math_inst in math_instructions:
    tile_descriptions_small = [
      TileDescription(
        [math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2] * 2],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1, 1, 1]),
      TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
         0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]
    tile_descriptions_medium = [
      TileDescription([math_inst.instruction_shape[0]*2, math_inst.instruction_shape[1], math_inst.instruction_shape[2]*4],
        0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
       TileDescription([math_inst.instruction_shape[0], math_inst.instruction_shape[1]*2, math_inst.instruction_shape[2]*4],
         0, [4, 1, 1], math_inst, min_cc, max_cc, [1,1,1]),
    ]
    tile_descriptions = tile_descriptions_small + tile_descriptions_medium

    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format.
    for layout in layouts:
      if data_type["c_type"] in [DataType.s32, DataType.f32]:
        layout[2][1] = 4
      elif data_type["c_type"] in [DataType.f16, DataType.bf16]:
        layout[2][1] = 8

    schedules = [
      # [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
      [KernelScheduleType.CpAsyncWarpSpecialized, EpilogueScheduleType.NoSmemWarpSpecialized]
    ]
    stream_k_schedules = []

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      schedules += [
        [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized],
        # [KernelScheduleType.CpAsyncWarpSpecializedPingpong, EpilogueScheduleType.NoSmemWarpSpecialized]
      ]
      stream_k_schedules += [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.NoSmemWarpSpecialized]]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
      # Add stream-K variants
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])

    # persistent kernels with TMA epilogues
    # if CudaToolkitVersionSatisfies(cuda_version, 12, 1):
    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
    #      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
    #     tile_schedulers=[TileSchedulerType.StreamK])

    #   # Emit instance without C allocation + load
    #   data_type["c_type"] = DataType.void
    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedPingpong,    EpilogueScheduleType.TmaWarpSpecialized],
    #      [KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]])

    #   CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type,
    #     [[KernelScheduleType.CpAsyncWarpSpecializedCooperative, EpilogueScheduleType.TmaWarpSpecializedCooperative]],
    #     tile_schedulers=[TileSchedulerType.StreamK])

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format.
      for layout in layouts:
        if data_type_mixed["c_type"] in [DataType.s32, DataType.f32]:
          layout[2][1] = 4
        elif data_type_mixed["c_type"] in [DataType.f16, DataType.bf16]:
          layout[2][1] = 8

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])
