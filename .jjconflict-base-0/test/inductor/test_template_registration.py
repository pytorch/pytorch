# Owner(s): ["module: inductor"]
import torch._inductor.select_algorithm as select_algorithm
from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
from torch._inductor.select_algorithm import ExternKernelChoice, TritonTemplate
from torch._inductor.test_case import run_tests, TestCase


# A module that registers Inductor templates / extern kernels can be initialized
# more than once in a single process (e.g. a double-import path). Registration
# must tolerate re-registration under an existing name when it is equivalent, but
# still reject a genuine name collision. These are pure-Python registry checks
# (no GPU / compilation) so they live in their own file rather than the GPU-gated
# select-algorithm test file. See https://github.com/pytorch/pytorch/issues/186220.
class TestRegistrationIdempotency(TestCase):
    @staticmethod
    def _grid(*args, **kwargs):
        return (1, 1, 1)

    def test_triton_template_reregistration_same_source(self):
        name = "test_idempotent_triton_template"
        self.addCleanup(TritonTemplate.all_templates.pop, name, None)
        source = "placeholder template source"
        TritonTemplate(name=name, grid=self._grid, source=source)
        # Re-registering the same source under the same name must not raise.
        second = TritonTemplate(name=name, grid=self._grid, source=source)
        self.assertIs(TritonTemplate.all_templates[name], second)

    def test_triton_template_reregistration_different_source_raises(self):
        name = "test_conflicting_triton_template"
        self.addCleanup(TritonTemplate.all_templates.pop, name, None)
        TritonTemplate(name=name, grid=self._grid, source="source A")
        with self.assertRaisesRegex(AssertionError, "duplicate template name"):
            TritonTemplate(name=name, grid=self._grid, source="source B")

    def test_cutedsl_template_reregistration_same_source(self):
        name = "test_idempotent_cutedsl_template"
        self.addCleanup(CuteDSLTemplate.all_templates.pop, name, None)
        source = "placeholder cutedsl source"
        CuteDSLTemplate(name=name, source=source)
        second = CuteDSLTemplate(name=name, source=source)
        self.assertIs(CuteDSLTemplate.all_templates[name], second)

    def test_cutedsl_template_reregistration_different_source_raises(self):
        name = "test_conflicting_cutedsl_template"
        self.addCleanup(CuteDSLTemplate.all_templates.pop, name, None)
        CuteDSLTemplate(name=name, source="source A")
        with self.assertRaisesRegex(AssertionError, "duplicate template name"):
            CuteDSLTemplate(name=name, source="source B")

    def test_extern_kernel_reregistration_same_callable(self):
        name = "test_idempotent_extern_kernel"
        self.addCleanup(ExternKernelChoice._registry.pop, name, None)
        self.addCleanup(lambda: delattr(select_algorithm.extern_kernels, name))

        def kernel(*args, **kwargs):
            return None

        ExternKernelChoice(kernel, name=name)
        # Re-registering the same callable under the same name must not raise, and
        # the kernel must still resolve through the namespace afterwards.
        second = ExternKernelChoice(kernel, name=name)
        self.assertIs(ExternKernelChoice.lookup(name), second)
        self.assertIs(second.to_callable(), kernel)

    def test_extern_kernel_reregistration_different_callable_raises(self):
        name = "test_conflicting_extern_kernel"
        self.addCleanup(ExternKernelChoice._registry.pop, name, None)
        self.addCleanup(lambda: delattr(select_algorithm.extern_kernels, name))

        def kernel_a(*args, **kwargs):
            return None

        def kernel_b(*args, **kwargs):
            return None

        ExternKernelChoice(kernel_a, name=name)
        with self.assertRaisesRegex(AssertionError, "duplicate extern kernel"):
            ExternKernelChoice(kernel_b, name=name)


if __name__ == "__main__":
    run_tests()
