import unittest
import torch
from torch._inductor.codecache import FxGraphHashDetails
from torch.fx import GraphModule
from torch._inductor.compile_fx import _CompileFxKwargs


class TestSDPACacheKey(unittest.TestCase):
    def test_sdpa_flags_in_key_components(self):
        # Сохраняем исходные значения флагов
        original_flash = torch.backends.cuda.flash_sdp_enabled()
        original_cudnn = torch.backends.cuda.cudnn_sdp_enabled()
        original_math = torch.backends.cuda.math_sdp_enabled()
        original_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()

        try:
            # Устанавливаем разные комбинации флагов
            torch.backends.cuda.flash_sdp_enabled = lambda: True
            torch.backends.cuda.cudnn_sdp_enabled = lambda: False
            torch.backends.cuda.math_sdp_enabled = lambda: False
            torch.backends.cuda.mem_efficient_sdp_enabled = lambda: False

            # Создаём фиктивные объекты для хэширования
            gm = GraphModule(torch.nn.Linear(4,4), torch.fx.Graph())
            example_inputs = [torch.randn(4,4)]
            fx_kwargs: _CompileFxKwargs = {}
            inputs_to_check = []

            # Получаем детали хэша
            details1 = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
            key1 = details1.__dict__.copy()
            # Меняем флаги
            torch.backends.cuda.flash_sdp_enabled = lambda: False
            torch.backends.cuda.cudnn_sdp_enabled = lambda: True
            details2 = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
            key2 = details2.__dict__.copy()
            # Хэши должны отличаться
            self.assertNotEqual(key1.get("flash_sdp_enabled"), key2.get("flash_sdp_enabled"))
            self.assertNotEqual(key1.get("cudnn_sdp_enabled"), key2.get("cudnn_sdp_enabled"))
        finally:
            # Восстанавливаем оригинальные значения
            torch.backends.cuda.flash_sdp_enabled = lambda: original_flash
            torch.backends.cuda.cudnn_sdp_enabled = lambda: original_cudnn
            torch.backends.cuda.math_sdp_enabled = lambda: original_math
            torch.backends.cuda.mem_efficient_sdp_enabled = lambda: original_mem_efficient

if __name__ == "__main__":
    unittest.main()
