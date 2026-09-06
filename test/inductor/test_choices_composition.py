# Owner(s): ["module: inductor"]

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import patch

from torch._inductor import config
from torch._inductor.choices import (
    create_inductor_choices,
    InductorChoices,
    register_inductor_choices,
    registered_inductor_choices,
    unregister_inductor_choices,
)
from torch._inductor.virtualized import _choices, threadlocal, V
from torch.testing._internal.common_utils import run_tests, TestCase


_UNSET = object()


class AlphaChoices(InductorChoices):
    suffix = "_alpha"

    def __init__(self, threshold: int = 7) -> None:
        super().__init__()
        self.result_suffix = f"{self.suffix}:{threshold}"

    def uuid(self) -> str:
        return f"alpha:{self.result_suffix}"

    def customize_fused_kernel_name(self, fused_name: str, src_code: str) -> str:
        return f"{fused_name}{self.result_suffix}"


class _BetaHeuristics:
    def get_conv_configs(self) -> str:
        return "beta"


class BetaChoices(InductorChoices):
    def __init__(self) -> None:
        super().__init__()
        self.heuristics = _BetaHeuristics()

    def uuid(self) -> str:
        return "beta"

    def get_config_heuristics(self, device_type: str | None = "cuda") -> Any:
        return self.heuristics


class SecondNameChoices(InductorChoices):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def uuid(self) -> str:
        return "second"

    def customize_fused_kernel_name(self, fused_name: str, src_code: str) -> str:
        self.calls += 1
        return f"{fused_name}_second"


class UUIDChoices(InductorChoices):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def uuid(self) -> str:
        return self.value


class NotChoices:
    def uuid(self) -> str:
        return "not-choices"


class PlainChoices(InductorChoices):
    pass


class StaticFalseChoices(InductorChoices):
    def uuid(self) -> str:
        return "static-false"

    @staticmethod
    def can_fuse(
        scheduler: Any,
        node1: Any,
        node2: Any,
        shared_data_score: int,
    ) -> bool:
        return False


class StaticTrueChoices(InductorChoices):
    def uuid(self) -> str:
        return "static-true"

    @staticmethod
    def can_fuse(
        scheduler: Any,
        node1: Any,
        node2: Any,
        shared_data_score: int,
    ) -> bool:
        return True


class ChoicesCompositionTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._prior_factory = config.inductor_choices_class
        self._prior_handler = getattr(threadlocal, _choices._key, _UNSET)
        self._prior_registrations = registered_inductor_choices()
        config.inductor_choices_class = None
        for key, _ in self._prior_registrations:
            unregister_inductor_choices(key)
        if hasattr(threadlocal, _choices._key):
            delattr(threadlocal, _choices._key)

    def tearDown(self) -> None:
        config.inductor_choices_class = None
        for key, _ in registered_inductor_choices():
            unregister_inductor_choices(key)
        for key, factory in self._prior_registrations:
            register_inductor_choices(key, factory)
        config.inductor_choices_class = self._prior_factory
        if hasattr(threadlocal, _choices._key):
            delattr(threadlocal, _choices._key)
        if self._prior_handler is not _UNSET:
            setattr(threadlocal, _choices._key, self._prior_handler)
        super().tearDown()

    def current_choices(self) -> InductorChoices:
        return create_inductor_choices(config.inductor_choices_class)

    def test_no_contributors_returns_base(self) -> None:
        self.assertIs(type(self.current_choices()), InductorChoices)

    def test_existing_config_factory_is_returned_untouched(self) -> None:
        part = AlphaChoices(threshold=3)
        with config.patch(inductor_choices_class=lambda: part):
            self.assertIs(self.current_choices(), part)

    def test_direct_virtualized_handler_is_unchanged(self) -> None:
        direct = AlphaChoices(threshold=3)
        with V.set_choices_handler(direct):
            self.assertIs(V.choices, direct)

    def test_registered_choices_compose_in_order(self) -> None:
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=99))
        register_inductor_choices("beta", BetaChoices)

        handler = V.choices
        self.assertEqual(
            handler.customize_fused_kernel_name("kernel", ""), "kernel_alpha:99"
        )
        self.assertEqual(handler.get_conv_configs(), "beta")
        expected_uuid = (
            "composed_inductor_choices",
            ("alpha:_alpha:99", "beta"),
        )
        self.assertEqual(handler.uuid(), expected_uuid)
        self.assertEqual(
            config.save_config_portable(ignore_private_configs=False)[
                "inductor_choices_class"
            ],
            expected_uuid,
        )

    def test_config_factory_precedes_registered_choices(self) -> None:
        second = SecondNameChoices()
        register_inductor_choices("second", lambda: second)

        with config.patch(inductor_choices_class=AlphaChoices):
            handler = self.current_choices()
            with self.assertLogs("torch._inductor.choices", level="WARNING"):
                self.assertEqual(
                    handler.customize_fused_kernel_name("kernel", ""),
                    "kernel_alpha:7",
                )

        self.assertEqual(second.calls, 0)

    def test_conflicting_override_warns_and_first_wins(self) -> None:
        second = SecondNameChoices()
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("second", lambda: second)
        handler = V.choices

        with self.assertLogs("torch._inductor.choices", level="WARNING") as logs:
            self.assertEqual(
                handler.customize_fused_kernel_name("kernel", ""),
                "kernel_alpha:7",
            )
        self.assertEqual(second.calls, 0)
        self.assertIn("customize_fused_kernel_name", "\n".join(logs.output))

    def test_staticmethod_conflict_warns_once(self) -> None:
        register_inductor_choices("false", StaticFalseChoices)
        register_inductor_choices("true", StaticTrueChoices)
        handler = V.choices

        with self.assertLogs("torch._inductor.choices", level="WARNING") as logs:
            self.assertFalse(handler.can_fuse(None, None, None, 0))
            self.assertFalse(handler.can_fuse(None, None, None, 0))

        self.assertEqual(len(logs.output), 1)
        self.assertIn("StaticFalseChoices", logs.output[0])
        self.assertIn("StaticTrueChoices", logs.output[0])

    def test_dispatch_target_is_cached(self) -> None:
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("beta", BetaChoices)
        handler = V.choices

        self.assertIs(
            handler.customize_fused_kernel_name,
            handler.customize_fused_kernel_name,
        )

    def test_runtime_instance_override_is_detected(self) -> None:
        choice = UUIDChoices("runtime")
        choice.__dict__["customize_fused_kernel_name"] = (
            lambda fused_name, src_code: f"{fused_name}_runtime"
        )

        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("runtime", lambda: choice)
        handler = V.choices

        self.assertEqual(
            handler.customize_fused_kernel_name("kernel", ""),
            "kernel_runtime",
        )

    def test_instance_patch_takes_precedence(self) -> None:
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("beta", BetaChoices)
        handler = V.choices

        with patch.object(
            handler,
            "customize_fused_kernel_name",
            lambda fused_name, src_code: f"{fused_name}_patched",
        ):
            self.assertEqual(
                handler.customize_fused_kernel_name("kernel", ""),
                "kernel_patched",
            )

        self.assertEqual(
            handler.customize_fused_kernel_name("kernel", ""),
            "kernel_alpha:7",
        )

    def test_fresh_thread_uses_registry_and_matching_cache_key(self) -> None:
        register_inductor_choices("alpha", AlphaChoices)
        register_inductor_choices("beta", BetaChoices)

        def get_runtime_and_cache_uuid() -> tuple[Any, Any]:
            return (
                V.choices.uuid(),
                config.save_config_portable(ignore_private_configs=False)[
                    "inductor_choices_class"
                ],
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            runtime_uuid, cache_uuid = executor.submit(
                get_runtime_and_cache_uuid
            ).result()

        self.assertEqual(runtime_uuid, cache_uuid)
        self.assertEqual(
            runtime_uuid,
            ("composed_inductor_choices", ("alpha:_alpha:7", "beta")),
        )

    def test_inherited_default_is_used_when_nobody_overrides(self) -> None:
        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("uuid", lambda: UUIDChoices("uuid-only"))
        self.assertEqual(
            V.choices.customize_fused_kernel_name("kernel", ""),
            "kernel",
        )

    def test_composed_uuid_is_structured_and_ordered(self) -> None:
        register_inductor_choices("first", lambda: UUIDChoices("a+b"))
        register_inductor_choices("second", lambda: UUIDChoices("c"))
        forward = V.choices.uuid()
        unregister_inductor_choices("first")
        unregister_inductor_choices("second")
        register_inductor_choices("second", lambda: UUIDChoices("c"))
        register_inductor_choices("first", lambda: UUIDChoices("a+b"))
        reverse = V.choices.uuid()

        self.assertEqual(
            forward,
            (
                "composed_inductor_choices",
                ("a+b", "c"),
            ),
        )
        self.assertNotEqual(forward, reverse)

    def test_composed_uuid_has_no_delimiter_collisions(self) -> None:
        register_inductor_choices("first", lambda: UUIDChoices("a+b"))
        register_inductor_choices("second", lambda: UUIDChoices("c"))
        ab_c = V.choices.uuid()
        unregister_inductor_choices("first")
        unregister_inductor_choices("second")
        register_inductor_choices("first", lambda: UUIDChoices("a"))
        register_inductor_choices("second", lambda: UUIDChoices("b+c"))
        a_bc = V.choices.uuid()
        self.assertNotEqual(ab_c, a_bc)

    def test_composed_uuid_tracks_contributor_state(self) -> None:
        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=1))
        first = V.choices.uuid()
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=2))
        second = V.choices.uuid()
        self.assertNotEqual(first, second)

    def test_non_choices_contributor_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must return InductorChoices"):
            register_inductor_choices("invalid", lambda: _as_choices(NotChoices()))
        self.assertEqual(registered_inductor_choices(), ())

    def test_missing_uuid_is_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not implement uuid"):
            register_inductor_choices("plain", PlainChoices)
        self.assertEqual(registered_inductor_choices(), ())

    def test_registration_replaces_in_place(self) -> None:
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=1))
        register_inductor_choices("beta", BetaChoices)
        register_inductor_choices("alpha", lambda: AlphaChoices(threshold=2))

        self.assertEqual(
            tuple(key for key, _ in registered_inductor_choices()),
            ("alpha", "beta"),
        )
        self.assertEqual(
            V.choices.uuid(),
            ("composed_inductor_choices", ("alpha:_alpha:2", "beta")),
        )


def _as_choices(obj: Any) -> InductorChoices:
    return obj


if __name__ == "__main__":
    run_tests()
