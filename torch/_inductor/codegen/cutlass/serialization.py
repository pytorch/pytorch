# mypy: allow-untyped-defs
import functools
import json
from enum import Enum
from typing import Any, Optional

from torch._inductor.codegen.cutlass.utils import try_import_cutlass


class CUTLASSOperationSerializer:
    """Serializes and deserializes CUTLASS GEMM operations to/from JSON.

    Handles GemmOperation objects and their nested components (TileDescription, TensorDescription).
    """

    # not used, but keeping in case we want to generalize the serializer
    _SUPPORTED_CLASSES: list[str] = [
        "GemmOperation",
        "GemmKind",
        "TileDescription",
        "TensorDescription",
        "DataType",
        "EpilogueFunctor",
        "EpilogueFunctor3x",
        "SwizzlingFunctor",
        "KernelScheduleType",
        "EpilogueScheduleType",
        "TileSchedulerType",
    ]

    @classmethod
    def serialize(cls, operation: "GemmOperation") -> str:  # type: ignore[name-defined]  # noqa: F821
        """Serialize a GEMM operation to JSON string.

        Args:
            operation: GemmOperation object

        Returns:
            str: JSON string representation of the operation
        """
        assert operation.__class__.__qualname__ == "GemmOperation", (
            "Only GemmOperation objects are supported via the main API"
        )
        return json.dumps(cls._gemm_operation_to_json(operation))

    @classmethod
    def deserialize(cls, json_str: str) -> "GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """Deserialize JSON string to a GEMM operation.

        Args:
            json_str: JSON string of a GEMM operation

        Returns:
            GemmOperation: Reconstructed operation
        """
        json_dict = json.loads(json_str)
        return cls._json_to_gemm_operation(json_dict)

    @classmethod
    def _gemm_operation_to_json(cls, operation: "GemmOperation") -> dict[str, Any]:  # type: ignore[name-defined]  # noqa: F821
        """Convert GemmOperation to JSON-serializable dict.

        Args:
            operation: GemmOperation object

        Returns:
            dict: Dictionary representation
        """
        from cutlass_library.library import TensorDescription

        # Create the main dictionary with required and optional parameters
        result = {
            # Required parameters
            "gemm_kind": cls._enum_to_json(operation.gemm_kind),
            "arch": operation.arch,
            "tile_description": cls._tile_description_to_json(
                operation.tile_description
            ),
            "A": cls._tensor_description_to_json(operation.A),
            "B": cls._tensor_description_to_json(operation.B),
            "C": cls._tensor_description_to_json(operation.C),
            "element_epilogue": cls._enum_to_json(operation.element_epilogue),
            # Optional parameters
            "epilogue_functor": cls._enum_to_json(operation.epilogue_functor),
            "swizzling_functor": cls._enum_to_json(operation.swizzling_functor),
            "D": cls._tensor_description_to_json(operation.D) if operation.D else None,
            "kernel_schedule": cls._enum_to_json(operation.kernel_schedule),
            "epilogue_schedule": cls._enum_to_json(operation.epilogue_schedule),
            "tile_scheduler": cls._enum_to_json(operation.tile_scheduler),
        }

        # Process optional attributes
        optional_attrs = [
            "mixed_input_mode",
            "mixed_input_shuffle",
            "ScaleFactorA",
            "ScaleFactorB",
            "ScaleFactorD",
            "ScaleFactorMVecSize",
            "ScaleFactorNVecSize",
            "ScaleFactorKVecSize",
            "ScaleFactorVectorSize",
            "is_3x",
        ]

        for attr in optional_attrs:
            if not hasattr(operation, attr):
                continue

            value = getattr(operation, attr)

            if isinstance(value, TensorDescription):
                result[attr] = cls._tensor_description_to_json(value)
            elif isinstance(value, Enum):
                result[attr] = cls._enum_to_json(value)
            else:
                result[attr] = value

        return result

    @classmethod
    def _json_to_gemm_operation(cls, json_dict: dict[str, Any]) -> "GemmOperation":  # type: ignore[name-defined]  # noqa: F821
        """Convert JSON dict to GemmOperation object.

        Args:
            json_dict: Dictionary representation

        Returns:
            GemmOperation: Reconstructed object
        """
        from cutlass_library import DataType
        from cutlass_library.gemm_operation import GemmKind, GemmOperation
        from cutlass_library.library import (
            EpilogueFunctor,
            EpilogueFunctor3x,
            EpilogueScheduleType,
            KernelScheduleType,
            MixedInputMode,
            SwizzlingFunctor,
            TileSchedulerType,
        )

        # Extract constructor parameters from the JSON dictionary
        gemm_kind = cls._json_to_enum(json_dict["gemm_kind"], GemmKind)
        arch = json_dict["arch"]
        tile_description = cls._json_to_tile_description(json_dict["tile_description"])
        A = cls._json_to_tensor_description(json_dict.get("A"), "A")
        B = cls._json_to_tensor_description(json_dict.get("B"), "B")
        C = cls._json_to_tensor_description(json_dict.get("C"), "C")
        element_epilogue = cls._json_to_enum(json_dict["element_epilogue"], DataType)

        # Get optional parameters with defaults
        epilogue_functor = cls._json_to_enum(
            json_dict.get("epilogue_functor"),
            EpilogueFunctor3x if json_dict.get("is_3x") else EpilogueFunctor,
        )
        swizzling_functor = cls._json_to_enum(
            json_dict.get("swizzling_functor"), SwizzlingFunctor
        )
        D = cls._json_to_tensor_description(json_dict.get("D"), "D")
        kernel_schedule = cls._json_to_enum(
            json_dict.get("kernel_schedule"), KernelScheduleType
        )
        epilogue_schedule = cls._json_to_enum(
            json_dict.get("epilogue_schedule"), EpilogueScheduleType
        )
        tile_scheduler = cls._json_to_enum(
            json_dict.get("tile_scheduler"), TileSchedulerType
        )

        mixed_input_mode = cls._json_to_enum(
            json_dict.get("mixed_input_mode"), MixedInputMode
        )
        mixed_input_shuffle = json_dict.get("mixed_input_shuffle", False)

        # Scale factors
        ScaleFactorA = cls._json_to_enum(json_dict.get("ScaleFactorA"), DataType)
        ScaleFactorB = cls._json_to_enum(json_dict.get("ScaleFactorB"), DataType)

        ScaleFactorD = None
        if "ScaleFactorD" in json_dict and "ScaleFactorVectorSize" in json_dict:
            ScaleFactorD = {
                "tensor": cls._json_to_tensor_description(
                    json_dict.get("ScaleFactorD"), "ScaleFactorD"
                ),
                "vector_size": json_dict.get("ScaleFactorVectorSize"),
            }

        ScaleFactorMVecSize = json_dict.get("ScaleFactorMVecSize")
        ScaleFactorNVecSize = json_dict.get("ScaleFactorNVecSize")
        ScaleFactorKVecSize = json_dict.get("ScaleFactorKVecSize")

        # Create the GemmOperation with the extracted parameters
        operation = GemmOperation(
            gemm_kind=gemm_kind,
            arch=arch,
            tile_description=tile_description,
            A=A,
            B=B,
            C=C,
            element_epilogue=element_epilogue,
            epilogue_functor=epilogue_functor,
            swizzling_functor=swizzling_functor,
            D=D,
            kernel_schedule=kernel_schedule,
            epilogue_schedule=epilogue_schedule,
            tile_scheduler=tile_scheduler,
            mixed_input_mode=mixed_input_mode,
            mixed_input_shuffle=mixed_input_shuffle,
            ScaleFactorA=ScaleFactorA,
            ScaleFactorB=ScaleFactorB,
            ScaleFactorD=ScaleFactorD,
            ScaleFactorMVecSize=ScaleFactorMVecSize,
            ScaleFactorNVecSize=ScaleFactorNVecSize,
            ScaleFactorKVecSize=ScaleFactorKVecSize,
        )

        return operation

    @classmethod
    @functools.lru_cache(None)
    def _tile_description_to_json(cls, tile_desc: "TileDescription") -> str:  # type: ignore[name-defined]  # noqa: F821
        """
        Convert TileDescription to JSON string.

        Args:
            tile_desc: TileDescription object

        Returns:
            str: JSON string representation
        """

        # Create the main dictionary with field names matching TileDescription constructor parameters
        result = {
            "threadblock_shape": tile_desc.threadblock_shape,
            "stages": tile_desc.stages,
            "warp_count": tile_desc.warp_count,
            "math_instruction": cls._math_instruction_to_json(
                tile_desc.math_instruction
            ),
            "min_compute": tile_desc.minimum_compute_capability,  # Store as min_compute for constructor
            "max_compute": tile_desc.maximum_compute_capability,  # Store as max_compute for constructor
            "cluster_shape": tile_desc.cluster_shape,
            "explicit_vector_sizes": tile_desc.explicit_vector_sizes,
        }

        # Add tile_shape if it exists and differs from threadblock_shape
        if (
            hasattr(tile_desc, "tile_shape")
            and tile_desc.tile_shape != tile_desc.threadblock_shape
        ):
            result["tile_shape"] = tile_desc.tile_shape

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_tile_description(
        cls, json_dict: Optional[str]
    ) -> Optional["TileDescription"]:  # type: ignore[name-defined]  # noqa: F821
        """
        Convert JSON dict to TileDescription object.

        Args:
            json_dict: Dictionary representation

        Returns:
            TileDescription: Reconstructed object
        """
        if json_dict is None:
            return None

        tile_dict = json.loads(json_dict)

        from cutlass_library.library import TileDescription

        math_instruction = cls._json_to_math_instruction(tile_dict["math_instruction"])

        # Get compute capability values, checking both naming conventions
        min_compute = tile_dict.get(
            "min_compute", tile_dict.get("minimum_compute_capability")
        )
        max_compute = tile_dict.get(
            "max_compute", tile_dict.get("maximum_compute_capability")
        )

        # Get cluster shape with default value
        cluster_shape = tile_dict.get("cluster_shape", [1, 1, 1])

        # Create the TileDescription object
        tile_desc = TileDescription(
            threadblock_shape=tile_dict["threadblock_shape"],
            stages=tile_dict["stages"],
            warp_count=tile_dict["warp_count"],
            math_instruction=math_instruction,
            min_compute=min_compute,
            max_compute=max_compute,
            cluster_shape=cluster_shape,
            explicit_vector_sizes=tile_dict.get("explicit_vector_sizes"),
        )

        # Set tile_shape if it exists and differs from threadblock_shape
        if (
            "tile_shape" in tile_dict
            and tile_dict["tile_shape"] != tile_dict["threadblock_shape"]
        ):
            tile_desc.tile_shape = tile_dict["tile_shape"]

        return tile_desc

    @classmethod
    @functools.lru_cache(None)
    def _math_instruction_to_json(
        cls,
        math_instruction: Optional["MathInstruction"],  # type: ignore[name-defined]  # noqa: F821
    ) -> Optional[str]:
        """Convert MathInstruction to JSON string.

        Args:
            math_instruction: MathInstruction object

        Returns:
            Optional[str]: JSON string representation or None
        """
        if math_instruction is None:
            return None

        result = {
            "instruction_shape": math_instruction.instruction_shape,
            "element_a": cls._enum_to_json(math_instruction.element_a),
            "element_b": cls._enum_to_json(math_instruction.element_b),
            "element_accumulator": cls._enum_to_json(
                math_instruction.element_accumulator
            ),
            "opcode_class": cls._enum_to_json(math_instruction.opcode_class),
            "math_operation": cls._enum_to_json(math_instruction.math_operation),
            "element_scale_factor": cls._enum_to_json(
                math_instruction.element_scale_factor
            ),
        }

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_math_instruction(
        cls, json_dict: Optional[str]
    ) -> Optional["MathInstruction"]:  # type: ignore[name-defined]  # noqa: F821
        """Convert JSON string to MathInstruction object.

        Args:
            json_dict: JSON string representation

        Returns:
            Optional[MathInstruction]: Reconstructed object or None
        """
        if json_dict is None:
            return None

        from cutlass_library import DataType
        from cutlass_library.library import MathInstruction, MathOperation, OpcodeClass

        mi_dict = json.loads(json_dict)

        # Convert string enum names back to enum values
        element_a = cls._json_to_enum(mi_dict["element_a"], DataType)
        element_b = cls._json_to_enum(mi_dict["element_b"], DataType)
        element_acc = cls._json_to_enum(mi_dict["element_accumulator"], DataType)

        # Get the opcode_class enum
        opcode_class = cls._json_to_enum(mi_dict["opcode_class"], OpcodeClass)

        # Get the math_operation enum
        math_op = cls._json_to_enum(mi_dict["math_operation"], MathOperation)

        # Create the MathInstruction object
        math_instruction_obj = MathInstruction(
            instruction_shape=mi_dict["instruction_shape"],
            element_a=element_a,
            element_b=element_b,
            element_accumulator=element_acc,
            opcode_class=opcode_class,
            math_operation=math_op,
        )

        # Add element_scale_factor if it exists
        if (
            "element_scale_factor" in mi_dict
            and mi_dict["element_scale_factor"] is not None
        ):
            math_instruction_obj.element_scale_factor = cls._json_to_enum(
                mi_dict["element_scale_factor"], DataType
            )

        return math_instruction_obj

    @classmethod
    @functools.lru_cache(None)
    def _tensor_description_to_json(
        cls,
        tensor_desc: Optional["TensorDescription"],  # type: ignore[name-defined]  # noqa: F821
    ) -> Optional[str]:
        """Convert TensorDescription to JSON string.

        Args:
            tensor_desc: TensorDescription object

        Returns:
            Optional[str]: JSON string representation or None
        """
        if tensor_desc is None:
            return None

        result = {
            "element": cls._enum_to_json(tensor_desc.element),
            "layout": cls._enum_to_json(tensor_desc.layout),
            "alignment": tensor_desc.alignment,
            "complex_transform": cls._enum_to_json(tensor_desc.complex_transform),
        }

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_tensor_description(
        cls,
        json_dict: Optional[str],
        tensor_name: Optional[str] = None,
    ) -> Optional["TensorDescription"]:  # type: ignore[name-defined]  # noqa: F821
        """Convert JSON string to TensorDescription object.

        Args:
            json_dict: JSON string representation
            tensor_name: Name of the tensor to avoid cache in the same op

        Returns:
            Optional[TensorDescription]: Reconstructed object or None
        """
        if json_dict is None:
            return None

        tensor_dict = json.loads(json_dict)

        from cutlass_library import DataType
        from cutlass_library.library import (
            ComplexTransform,
            LayoutType,
            TensorDescription,
        )

        element = cls._json_to_enum(tensor_dict["element"], DataType)
        layout = cls._json_to_enum(tensor_dict["layout"], LayoutType)
        alignment = tensor_dict["alignment"]
        complex_transform = cls._json_to_enum(
            tensor_dict["complex_transform"], ComplexTransform
        )

        return TensorDescription(element, layout, alignment, complex_transform)

    @classmethod
    @functools.lru_cache(None)
    def _enum_to_json(cls, enum_value: Optional[Enum]) -> Optional[str]:
        """Convert enum value to JSON string.

        Args:
            enum_value: Enum value

        Returns:
            Optional[str]: JSON string representation or None
        """
        if enum_value is None:
            return None

        result = {
            "type": enum_value.__class__.__name__,
            "name": enum_value.name,
        }

        return json.dumps(result)

    @classmethod
    @functools.lru_cache(None)
    def _json_to_enum(cls, json_dict: Optional[str], enum_class: Any) -> Optional[Enum]:
        """Convert JSON string to enum value.

        Format: {name: "EnumName", value: 1}

        Args:
            json_dict: JSON string representation
            enum_class: Target enum class

        Returns:
            Optional[Enum]: Reconstructed enum value or None
        """
        if json_dict is None:
            return None

        enum_dict = json.loads(json_dict)

        return enum_class[enum_dict["name"]]


@functools.lru_cache(1)
def get_cutlass_operation_serializer() -> Optional[CUTLASSOperationSerializer]:
    if not try_import_cutlass():
        return None
    return CUTLASSOperationSerializer()
