import torch
from torch import nn
# import rocmKernels

def benchmark_torch_function(iters: int, function, *args) -> float:
    function(*args)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        function(*args)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters

def custom_ln(input, normalized_shape, weight, bias, eps):
    # norm = normalized_shape[0]
    # for x in normalized_shape[1:]:
    #     norm = norm * x
    # print(input.shape, norm)
    # ln_input = input.reshape(-1, input.shape[-1])
    # # ln_input = input.view(-1, norm).contiguous()
    # output = torch.empty_like(ln_input)
    # rocmKernels.layernorm2d_fwd(
    #     output,
    #     ln_input,
    #     weight,
    #     bias,
    #     eps
    # )
    # print("diff:",output.shape, output_torch.shape, torch.sum(output),torch.sum(output_torch), torch.sum(torch.abs(output - output_torch)))

    output_torch =  torch.nn.functional.layer_norm(
                input=input,
                normalized_shape=normalized_shape,
                weight=weight,
                bias=bias,
                eps=eps,
            )
    # print("sum", torch.sum(output_torch))    
    return output_torch
class ExportedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._attr_0 = nn.Parameter(torch.randn(4608, 3594, dtype=torch.float16))
        self._attr_1 = nn.Parameter(torch.randn(4608, dtype=torch.float16))
        self._attr_2 = nn.Parameter(torch.randn(1, 200, 64, dtype=torch.float16))
        self._attr_3 = nn.Parameter(torch.randn(1, 200, 64, dtype=torch.float16))
        self._attr_4 = nn.Parameter(torch.randn(1, 200, 64, dtype=torch.float16))
        self._attr_5 = nn.Parameter(torch.randn(1, 200, 64, dtype=torch.float16))
        self._attr_6 = nn.Parameter(torch.randn(1, 200, 64, dtype=torch.float16))
        self._attr_7 = nn.Parameter(torch.randn(1, 200, 64, dtype=torch.float16))
        self._attr_8 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_9 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_10 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_11 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_12 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_13 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_14 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_15 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_16 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_17 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_18 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_19 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_20 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_21 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_22 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_23 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_24 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_25 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_26 = nn.Parameter(torch.randn(160, 32, dtype=torch.float16))
        self._attr_27 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_28 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_29 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_30 = nn.Parameter(torch.randn(160, 192, dtype=torch.float16))
        self._attr_31 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_32 = nn.Parameter(torch.randn(160, 192, dtype=torch.float16))
        self._attr_33 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_34 = nn.Parameter(torch.randn(160, 32, dtype=torch.float16))
        self._attr_35 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_36 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_37 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_38 = nn.Parameter(torch.randn(160, 240, dtype=torch.float16))
        self._attr_39 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_40 = nn.Parameter(torch.randn(160, 128, dtype=torch.float16))
        self._attr_41 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_42 = nn.Parameter(torch.randn(160, 120, dtype=torch.float16))
        self._attr_43 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_44 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_45 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_46 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_47 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_48 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_49 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_50 = nn.Parameter(torch.randn(160, 64, dtype=torch.float16))
        self._attr_51 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_52 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_53 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_54 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_55 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_56 = nn.Parameter(torch.randn(160, 64, dtype=torch.float16))
        self._attr_57 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_58 = nn.Parameter(torch.randn(160, 72, dtype=torch.float16))
        self._attr_59 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_60 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_61 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_62 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_63 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_64 = nn.Parameter(torch.randn(160, 64, dtype=torch.float16))
        self._attr_65 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_66 = nn.Parameter(torch.randn(160, 64, dtype=torch.float16))
        self._attr_67 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_68 = nn.Parameter(torch.randn(160, 72, dtype=torch.float16))
        self._attr_69 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_70 = nn.Parameter(torch.randn(160, 64, dtype=torch.float16))
        self._attr_71 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_72 = nn.Parameter(torch.randn(160, 64, dtype=torch.float16))
        self._attr_73 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_74 = nn.Parameter(torch.randn(160, 96, dtype=torch.float16))
        self._attr_75 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_76 = nn.Parameter(torch.randn(160, 144, dtype=torch.float16))
        self._attr_77 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_78 = nn.Parameter(torch.randn(160, 144, dtype=torch.float16))
        self._attr_79 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_80 = nn.Parameter(torch.randn(160, 72, dtype=torch.float16))
        self._attr_81 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_82 = nn.Parameter(torch.randn(160, 64, dtype=torch.float16))
        self._attr_83 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_84 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_85 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_86 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_87 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_88 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_89 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_90 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_91 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_92 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_93 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_94 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_95 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_96 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_97 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_98 = nn.Parameter(torch.randn(39008, 512, dtype=torch.float16))
        self._attr_99 = nn.Parameter(torch.randn(39008, dtype=torch.float16))
        self._attr_100 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_101 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_102 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_103 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_104 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_105 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_106 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_107 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_108 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_109 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_110 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_111 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_112 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_113 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_114 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_115 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_116 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_117 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_118 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_119 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_120 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_121 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_122 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_123 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_124 = nn.Parameter(torch.randn(64, 64, dtype=torch.float16))
        self._attr_125 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_126 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_127 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_128 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_129 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_130 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_131 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_132 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_133 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_134 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_135 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_136 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_137 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_138 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_139 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_140 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_141 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_142 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_143 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_144 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_145 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_146 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_147 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_148 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_149 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_150 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_151 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_152 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_153 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_154 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_155 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_156 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_157 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_158 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_159 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_160 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_161 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_162 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_163 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_164 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_165 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_166 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_167 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_168 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_169 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_170 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_171 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_172 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_173 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_174 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_175 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_176 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_177 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_178 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_179 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_180 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_181 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_182 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_183 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_184 = nn.Parameter(torch.randn(3594, 512, dtype=torch.float16))
        self._attr_185 = nn.Parameter(torch.randn(3594, dtype=torch.float16))
        self._attr_186 = nn.Parameter(torch.randn(2048, 3594, dtype=torch.float16))
        self._attr_187 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_188 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_189 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_190 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_191 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_192 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_193 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_194 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_195 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_196 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_197 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_198 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_199 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_200 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_201 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_202 = nn.Parameter(torch.randn(256, 64, dtype=torch.float16))
        self._attr_203 = nn.Parameter(torch.randn(256, dtype=torch.float16))
        self._attr_204 = nn.Parameter(torch.randn(256, 64, dtype=torch.float16))
        self._attr_205 = nn.Parameter(torch.randn(256, dtype=torch.float16))
        self._attr_206 = nn.Parameter(torch.randn(256, 64, dtype=torch.float16))
        self._attr_207 = nn.Parameter(torch.randn(256, dtype=torch.float16))
        self._attr_208 = nn.Parameter(torch.randn(256, 64, dtype=torch.float16))
        self._attr_209 = nn.Parameter(torch.randn(256, dtype=torch.float16))
        self._attr_210 = nn.Parameter(torch.randn(256, 64, dtype=torch.float16))
        self._attr_211 = nn.Parameter(torch.randn(256, dtype=torch.float16))
        self._attr_212 = nn.Parameter(torch.randn(256, 64, dtype=torch.float16))
        self._attr_213 = nn.Parameter(torch.randn(256, dtype=torch.float16))
        self._attr_214 = nn.Parameter(torch.randn(480, 2048, dtype=torch.float16))
        self._attr_215 = nn.Parameter(torch.randn(480, dtype=torch.float16))
        self._attr_216 = nn.Parameter(torch.randn(64, 256, dtype=torch.float16))
        self._attr_217 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_218 = nn.Parameter(torch.randn(64, 256, dtype=torch.float16))
        self._attr_219 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_220 = nn.Parameter(torch.randn(64, 256, dtype=torch.float16))
        self._attr_221 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_222 = nn.Parameter(torch.randn(64, 256, dtype=torch.float16))
        self._attr_223 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_224 = nn.Parameter(torch.randn(64, 256, dtype=torch.float16))
        self._attr_225 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_226 = nn.Parameter(torch.randn(64, 256, dtype=torch.float16))
        self._attr_227 = nn.Parameter(torch.randn(64, dtype=torch.float16))
        self._attr_228 = nn.Parameter(torch.randn(144, 64, 160, dtype=torch.float16))
        self._attr_229 = nn.Parameter(torch.randn(144, 1, 160, dtype=torch.float16))
        self._attr_230 = nn.Parameter(torch.randn(296, 1219, dtype=torch.float16))
        self._attr_231 = nn.Parameter(torch.randn(296, dtype=torch.float16))
        self._attr_232 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_233 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_234 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_235 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_236 = nn.Parameter(torch.randn(39008, 512, dtype=torch.float16))
        self._attr_237 = nn.Parameter(torch.randn(39008, dtype=torch.float16))
        self._attr_238 = nn.Parameter(torch.randn(2048, 39008, dtype=torch.float16))
        self._attr_239 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_240 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_241 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_242 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_243 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_244 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_245 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_246 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_247 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_248 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_249 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_250 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_251 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_252 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_253 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_254 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_255 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_256 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_257 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_258 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_259 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_260 = nn.Parameter(torch.randn(5120, 4096, dtype=torch.float16))
        self._attr_261 = nn.Parameter(torch.randn(5120, dtype=torch.float16))
        self._attr_262 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_263 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_264 = nn.Parameter(torch.randn(128, 88, dtype=torch.float16))
        self._attr_265 = nn.Parameter(torch.randn(128, dtype=torch.float16))
        self._attr_266 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_267 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_268 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_269 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_270 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_271 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_272 = nn.Parameter(torch.randn(2048, 2816, dtype=torch.float16))
        self._attr_273 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_274 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_275 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_276 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_277 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_278 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_279 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_280 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_281 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_282 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_283 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_284 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_285 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_286 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_287 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_288 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_289 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_290 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_291 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_292 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_293 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_294 = nn.Parameter(torch.randn(5120, 4096, dtype=torch.float16))
        self._attr_295 = nn.Parameter(torch.randn(5120, dtype=torch.float16))
        self._attr_296 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_297 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_298 = nn.Parameter(torch.randn(128, 88, dtype=torch.float16))
        self._attr_299 = nn.Parameter(torch.randn(128, dtype=torch.float16))
        self._attr_300 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_301 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_302 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_303 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_304 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_305 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_306 = nn.Parameter(torch.randn(2048, 2816, dtype=torch.float16))
        self._attr_307 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_308 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_309 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_310 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_311 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_312 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_313 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_314 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_315 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_316 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_317 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_318 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_319 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_320 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_321 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_322 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_323 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_324 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_325 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_326 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_327 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_328 = nn.Parameter(torch.randn(5120, 4096, dtype=torch.float16))
        self._attr_329 = nn.Parameter(torch.randn(5120, dtype=torch.float16))
        self._attr_330 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_331 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_332 = nn.Parameter(torch.randn(128, 88, dtype=torch.float16))
        self._attr_333 = nn.Parameter(torch.randn(128, dtype=torch.float16))
        self._attr_334 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_335 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_336 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_337 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_338 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_339 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_340 = nn.Parameter(torch.randn(2048, 2816, dtype=torch.float16))
        self._attr_341 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_342 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_343 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_344 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_345 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_346 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_347 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_348 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_349 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_350 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_351 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_352 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_353 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_354 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_355 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_356 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_357 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_358 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_359 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_360 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_361 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_362 = nn.Parameter(torch.randn(5120, 4096, dtype=torch.float16))
        self._attr_363 = nn.Parameter(torch.randn(5120, dtype=torch.float16))
        self._attr_364 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_365 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_366 = nn.Parameter(torch.randn(128, 88, dtype=torch.float16))
        self._attr_367 = nn.Parameter(torch.randn(128, dtype=torch.float16))
        self._attr_368 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_369 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_370 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_371 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_372 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_373 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_374 = nn.Parameter(torch.randn(2048, 2816, dtype=torch.float16))
        self._attr_375 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_376 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_377 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_378 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_379 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_380 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_381 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_382 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_383 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_384 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_385 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_386 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_387 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_388 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_389 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_390 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_391 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_392 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_393 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_394 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_395 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_396 = nn.Parameter(torch.randn(5120, 4096, dtype=torch.float16))
        self._attr_397 = nn.Parameter(torch.randn(5120, dtype=torch.float16))
        self._attr_398 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_399 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_400 = nn.Parameter(torch.randn(128, 88, dtype=torch.float16))
        self._attr_401 = nn.Parameter(torch.randn(128, dtype=torch.float16))
        self._attr_402 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_403 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_404 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_405 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_406 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_407 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_408 = nn.Parameter(torch.randn(2048, 2816, dtype=torch.float16))
        self._attr_409 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_410 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_411 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_412 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_413 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_414 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_415 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_416 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_417 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_418 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_419 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_420 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_421 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_422 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_423 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_424 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_425 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_426 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_427 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_428 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_429 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_430 = nn.Parameter(torch.randn(5120, 4096, dtype=torch.float16))
        self._attr_431 = nn.Parameter(torch.randn(5120, dtype=torch.float16))
        self._attr_432 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_433 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_434 = nn.Parameter(torch.randn(128, 88, dtype=torch.float16))
        self._attr_435 = nn.Parameter(torch.randn(128, dtype=torch.float16))
        self._attr_436 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_437 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_438 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_439 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_440 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_441 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_442 = nn.Parameter(torch.randn(2048, 2816, dtype=torch.float16))
        self._attr_443 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_444 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_445 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_446 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_447 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_448 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_449 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_450 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_451 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_452 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_453 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_454 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_455 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_456 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_457 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_458 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_459 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_460 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_461 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_462 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_463 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_464 = nn.Parameter(torch.randn(5120, 4096, dtype=torch.float16))
        self._attr_465 = nn.Parameter(torch.randn(5120, dtype=torch.float16))
        self._attr_466 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_467 = nn.Parameter(torch.randn(160, dtype=torch.float16))
        self._attr_468 = nn.Parameter(torch.randn(32, 88, dtype=torch.float16))
        self._attr_469 = nn.Parameter(torch.randn(32, dtype=torch.float16))
        self._attr_470 = nn.Parameter(torch.randn(512, 5120, dtype=torch.float16))
        self._attr_471 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_472 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_473 = nn.Parameter(torch.randn(512, dtype=torch.float16))
        self._attr_474 = nn.Parameter(torch.randn(2816, 512, dtype=torch.float16))
        self._attr_475 = nn.Parameter(torch.randn(2816, dtype=torch.float16))
        self._attr_476 = nn.Parameter(torch.randn(2048, 2816, dtype=torch.float16))
        self._attr_477 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_478 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_479 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_480 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_481 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_482 = nn.Parameter(torch.randn(1024, 2048, dtype=torch.float16))
        self._attr_483 = nn.Parameter(torch.randn(1024, dtype=torch.float16))
        self._attr_484 = nn.Parameter(torch.randn(2048, 1024, dtype=torch.float16))
        self._attr_485 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_486 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_487 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_488 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_489 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_490 = nn.Parameter(torch.randn(2048, 4096, dtype=torch.float16))
        self._attr_491 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_492 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_493 = nn.Parameter(torch.randn(2048, dtype=torch.float16))
        self._attr_494 = nn.Parameter(torch.randn(4096, 2048, dtype=torch.float16))
        self._attr_495 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_496 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_497 = nn.Parameter(torch.randn(4096, dtype=torch.float16))
        self._attr_498 = nn.Parameter(torch.randn(128, 4096, dtype=torch.float16))
        self._attr_499 = nn.Parameter(torch.randn(128, dtype=torch.float16))
        self._attr_500 = nn.Parameter(torch.randn(1, 128, dtype=torch.float16))
        self._attr_501 = nn.Parameter(torch.randn(1, dtype=torch.float16))

    def forward(
        self,
        getitem,
        getitem_3228,
        getitem_3226,
        getitem_3225,
        getitem_3227,
        getitem_3224,
        getitem_3223,
        getitem_3222,
        getitem_3221,
        getitem_3220,
        getitem_3219,
        repeat,
        repeat_1,
        repeat_2,
        repeat_3,
        repeat_4,
        repeat_5,
    ):
        tanh = torch.tanh(input=getitem)
        getitem = None
        tanh_68 = torch.tanh(input=getitem_3228)
        getitem_3228 = None
        clamp = torch.clamp(input=getitem_3226, min=-1000.1, max=1000.1)
        getitem_3226 = None
        nan_to_num = torch.nan_to_num(
            input=clamp, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp = None
        getitem_3683 = tanh[:, 0:6240]
        getitem_3684 = tanh[:, 6240:7040]
        getitem_3685 = tanh[:, 7040:8000]
        getitem_3686 = tanh[:, 8000:8640]
        getitem_3687 = tanh[:, 8640:48160]
        getitem_3688 = tanh[:, 48160:48640]
        getitem_3689 = tanh[:, 48640:48800]
        getitem_3690 = tanh[:, 48800:48960]
        getitem_3691 = tanh[:, 48960:49120]
        getitem_3692 = tanh[:, 49120:49280]
        getitem_3693 = tanh[:, 49280:49440]
        getitem_3694 = tanh[:, 49440:49600]
        getitem_3695 = tanh[:, 49600:49760]
        tanh = None
        getitem_3696 = tanh_68[:, 0:1120]
        getitem_3697 = tanh_68[:, 1120:1760]
        getitem_3698 = tanh_68[:, 1760:3520]
        getitem_3699 = tanh_68[:, 3520:4960]
        getitem_3700 = tanh_68[:, 4960:9760]
        getitem_3701 = tanh_68[:, 9760:116000]
        getitem_3702 = tanh_68[:, 116000:116160]
        getitem_3703 = tanh_68[:, 116160:116320]
        getitem_3704 = tanh_68[:, 116320:116480]
        getitem_3705 = tanh_68[:, 116480:116640]
        getitem_3706 = tanh_68[:, 116640:116800]
        getitem_3707 = tanh_68[:, 116800:116960]
        getitem_3708 = tanh_68[:, 116960:117120]
        tanh_68 = None
        getitem_3710 = nan_to_num[:, 0:32]
        getitem_3711 = nan_to_num[:, 32:128]
        getitem_3712 = nan_to_num[:, 128:320]
        getitem_3713 = nan_to_num[:, 320:512]
        getitem_3714 = nan_to_num[:, 512:544]
        getitem_3715 = nan_to_num[:, 544:640]
        getitem_3716 = nan_to_num[:, 640:880]
        nan_to_num = None
        getitem_3717 = getitem_3225[:, 0:128]
        getitem_3718 = getitem_3225[:, 128:248]
        getitem_3719 = getitem_3225[:, 248:344]
        getitem_3720 = getitem_3225[:, 344:440]
        getitem_3721 = getitem_3225[:, 440:536]
        getitem_3722 = getitem_3225[:, 536:600]
        getitem_3723 = getitem_3225[:, 600:696]
        getitem_3724 = getitem_3225[:, 696:792]
        getitem_3725 = getitem_3225[:, 792:856]
        getitem_3726 = getitem_3225[:, 856:928]
        getitem_3727 = getitem_3225[:, 928:1024]
        getitem_3728 = getitem_3225[:, 1024:1120]
        getitem_3729 = getitem_3225[:, 1120:1184]
        getitem_3730 = getitem_3225[:, 1184:1248]
        getitem_3731 = getitem_3225[:, 1248:1320]
        getitem_3732 = getitem_3225[:, 1320:1384]
        getitem_3733 = getitem_3225[:, 1384:1448]
        getitem_3734 = getitem_3225[:, 1448:1544]
        getitem_3735 = getitem_3225[:, 1544:1688]
        getitem_3736 = getitem_3225[:, 1688:1832]
        getitem_3737 = getitem_3225[:, 1832:1904]
        getitem_3738 = getitem_3225[:, 1904:1968]
        getitem_3225 = None
        _holder__attr_0 = self._attr_0
        _holder__attr_1 = self._attr_1
        linear = torch.nn.functional.linear(
            input=getitem_3227, weight=_holder__attr_0, bias=_holder__attr_1
        )
        _holder__attr_0 = _holder__attr_1 = None
        getitem_4137 = linear[:, 0:512]
        getitem_4138 = linear[:, 512:1024]
        getitem_4139 = linear[:, 1024:1536]
        getitem_4140 = linear[:, 1536:2048]
        getitem_4141 = linear[:, 2048:2560]
        getitem_4142 = linear[:, 2560:3072]
        getitem_4143 = linear[:, 3072:3584]
        getitem_4144 = linear[:, 3584:4096]
        getitem_4145 = linear[:, 4096:4608]
        linear = None
        _holder__attr_2 = self._attr_2
        add_42 = torch.add(input=getitem_3224, other=_holder__attr_2)
        getitem_3224 = _holder__attr_2 = None
        _holder__attr_3 = self._attr_3
        add_43 = torch.add(input=getitem_3223, other=_holder__attr_3)
        getitem_3223 = _holder__attr_3 = None
        _holder__attr_4 = self._attr_4
        add_44 = torch.add(input=getitem_3222, other=_holder__attr_4)
        getitem_3222 = _holder__attr_4 = None
        _holder__attr_5 = self._attr_5
        add_45 = torch.add(input=getitem_3221, other=_holder__attr_5)
        getitem_3221 = _holder__attr_5 = None
        _holder__attr_6 = self._attr_6
        add_46 = torch.add(input=getitem_3220, other=_holder__attr_6)
        getitem_3220 = _holder__attr_6 = None
        _holder__attr_7 = self._attr_7
        add_47 = torch.add(input=getitem_3219, other=_holder__attr_7)
        getitem_3219 = _holder__attr_7 = None
        clamp_37 = torch.clamp(input=getitem_3717, min=-1000.1, max=1000.1)
        getitem_3717 = None
        clamp_38 = torch.clamp(input=getitem_3718, min=-1000.1, max=1000.1)
        getitem_3718 = None
        clamp_39 = torch.clamp(input=getitem_3719, min=-1000.1, max=1000.1)
        getitem_3719 = None
        clamp_40 = torch.clamp(input=getitem_3720, min=-1000.1, max=1000.1)
        getitem_3720 = None
        clamp_41 = torch.clamp(input=getitem_3721, min=-1000.1, max=1000.1)
        getitem_3721 = None
        clamp_42 = torch.clamp(input=getitem_3722, min=-1000.1, max=1000.1)
        getitem_3722 = None
        clamp_43 = torch.clamp(input=getitem_3723, min=-1000.1, max=1000.1)
        getitem_3723 = None
        clamp_44 = torch.clamp(input=getitem_3724, min=-1000.1, max=1000.1)
        getitem_3724 = None
        clamp_45 = torch.clamp(input=getitem_3725, min=-1000.1, max=1000.1)
        getitem_3725 = None
        clamp_46 = torch.clamp(input=getitem_3726, min=-1000.1, max=1000.1)
        getitem_3726 = None
        clamp_47 = torch.clamp(input=getitem_3727, min=-1000.1, max=1000.1)
        getitem_3727 = None
        clamp_48 = torch.clamp(input=getitem_3728, min=-1000.1, max=1000.1)
        getitem_3728 = None
        clamp_49 = torch.clamp(input=getitem_3729, min=-1000.1, max=1000.1)
        getitem_3729 = None
        clamp_50 = torch.clamp(input=getitem_3730, min=-1000.1, max=1000.1)
        getitem_3730 = None
        clamp_51 = torch.clamp(input=getitem_3731, min=-100.1, max=100.1)
        getitem_3731 = None
        clamp_52 = torch.clamp(input=getitem_3732, min=-1000.1, max=1000.1)
        getitem_3732 = None
        clamp_53 = torch.clamp(input=getitem_3733, min=-1000.1, max=1000.1)
        getitem_3733 = None
        clamp_54 = torch.clamp(input=getitem_3734, min=-1000.1, max=1000.1)
        getitem_3734 = None
        clamp_55 = torch.clamp(input=getitem_3735, min=-1000.1, max=1000.1)
        getitem_3735 = None
        clamp_56 = torch.clamp(input=getitem_3736, min=-1000.1, max=1000.1)
        getitem_3736 = None
        clamp_57 = torch.clamp(input=getitem_3737, min=-1000.1, max=1000.1)
        getitem_3737 = None
        clamp_58 = torch.clamp(input=getitem_3738, min=-1000.1, max=1000.1)
        getitem_3738 = None
        size_72 = getitem_4137.size()
        sigmoid_51 = torch.sigmoid(input=getitem_4138)
        sigmoid_52 = torch.sigmoid(input=getitem_4139)
        sigmoid_53 = torch.sigmoid(input=getitem_4140)
        sigmoid_54 = torch.sigmoid(input=getitem_4141)
        sigmoid_55 = torch.sigmoid(input=getitem_4142)
        sigmoid_56 = torch.sigmoid(input=getitem_4143)
        sigmoid_57 = torch.sigmoid(input=getitem_4144)
        sigmoid_58 = torch.sigmoid(input=getitem_4145)
        _holder__attr_8 = self._attr_8
        _holder__attr_9 = self._attr_9
        layer_norm_135 = custom_ln(
            input=add_42,
            normalized_shape=(64,),
            weight=_holder__attr_8,
            bias=_holder__attr_9,
            eps=1e-05,
        )
        add_42 = _holder__attr_8 = _holder__attr_9 = None
        _holder__attr_10 = self._attr_10
        _holder__attr_11 = self._attr_11
        layer_norm_136 = custom_ln(
            input=add_43,
            normalized_shape=(64,),
            weight=_holder__attr_10,
            bias=_holder__attr_11,
            eps=1e-05,
        )
        add_43 = _holder__attr_10 = _holder__attr_11 = None
        _holder__attr_12 = self._attr_12
        _holder__attr_13 = self._attr_13
        layer_norm_137 = custom_ln(
            input=add_44,
            normalized_shape=(64,),
            weight=_holder__attr_12,
            bias=_holder__attr_13,
            eps=1e-05,
        )
        add_44 = _holder__attr_12 = _holder__attr_13 = None
        _holder__attr_14 = self._attr_14
        _holder__attr_15 = self._attr_15
        layer_norm_138 = custom_ln(
            input=add_45,
            normalized_shape=(64,),
            weight=_holder__attr_14,
            bias=_holder__attr_15,
            eps=1e-05,
        )
        add_45 = _holder__attr_14 = _holder__attr_15 = None
        _holder__attr_16 = self._attr_16
        _holder__attr_17 = self._attr_17
        layer_norm_139 = custom_ln(
            input=add_46,
            normalized_shape=(64,),
            weight=_holder__attr_16,
            bias=_holder__attr_17,
            eps=1e-05,
        )
        add_46 = _holder__attr_16 = _holder__attr_17 = None
        _holder__attr_18 = self._attr_18
        _holder__attr_19 = self._attr_19
        layer_norm_140 = custom_ln(
            input=add_47,
            normalized_shape=(64,),
            weight=_holder__attr_18,
            bias=_holder__attr_19,
            eps=1e-05,
        )
        add_47 = _holder__attr_18 = _holder__attr_19 = None
        nan_to_num_37 = torch.nan_to_num(
            input=clamp_37, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_37 = None
        nan_to_num_38 = torch.nan_to_num(
            input=clamp_38, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_38 = None
        nan_to_num_39 = torch.nan_to_num(
            input=clamp_39, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_39 = None
        nan_to_num_40 = torch.nan_to_num(
            input=clamp_40, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_40 = None
        nan_to_num_41 = torch.nan_to_num(
            input=clamp_41, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_41 = None
        nan_to_num_42 = torch.nan_to_num(
            input=clamp_42, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_42 = None
        nan_to_num_43 = torch.nan_to_num(
            input=clamp_43, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_43 = None
        nan_to_num_44 = torch.nan_to_num(
            input=clamp_44, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_44 = None
        nan_to_num_45 = torch.nan_to_num(
            input=clamp_45, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_45 = None
        nan_to_num_46 = torch.nan_to_num(
            input=clamp_46, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_46 = None
        nan_to_num_47 = torch.nan_to_num(
            input=clamp_47, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_47 = None
        nan_to_num_48 = torch.nan_to_num(
            input=clamp_48, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_48 = None
        nan_to_num_49 = torch.nan_to_num(
            input=clamp_49, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_49 = None
        nan_to_num_50 = torch.nan_to_num(
            input=clamp_50, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_50 = None
        nan_to_num_51 = torch.nan_to_num(
            input=clamp_51, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_51 = None
        nan_to_num_52 = torch.nan_to_num(
            input=clamp_52, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_52 = None
        nan_to_num_53 = torch.nan_to_num(
            input=clamp_53, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_53 = None
        nan_to_num_54 = torch.nan_to_num(
            input=clamp_54, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_54 = None
        nan_to_num_55 = torch.nan_to_num(
            input=clamp_55, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_55 = None
        nan_to_num_56 = torch.nan_to_num(
            input=clamp_56, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_56 = None
        nan_to_num_57 = torch.nan_to_num(
            input=clamp_57, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_57 = None
        nan_to_num_58 = torch.nan_to_num(
            input=clamp_58, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        clamp_58 = None
        getitem_3709 = size_72[1:]
        size_72 = None
        mul_59 = torch.mul(input=getitem_4138, other=sigmoid_51)
        getitem_4138 = sigmoid_51 = None
        mul_60 = torch.mul(input=getitem_4139, other=sigmoid_52)
        getitem_4139 = sigmoid_52 = None
        mul_61 = torch.mul(input=getitem_4140, other=sigmoid_53)
        getitem_4140 = sigmoid_53 = None
        mul_62 = torch.mul(input=getitem_4141, other=sigmoid_54)
        getitem_4141 = sigmoid_54 = None
        mul_63 = torch.mul(input=getitem_4142, other=sigmoid_55)
        getitem_4142 = sigmoid_55 = None
        mul_64 = torch.mul(input=getitem_4143, other=sigmoid_56)
        getitem_4143 = sigmoid_56 = None
        mul_65 = torch.mul(input=getitem_4144, other=sigmoid_57)
        getitem_4144 = sigmoid_57 = None
        mul_66 = torch.mul(input=getitem_4145, other=sigmoid_58)
        getitem_4145 = sigmoid_58 = None
        _holder__attr_20 = self._attr_20
        linear_208 = torch.nn.functional.linear(
            input=layer_norm_135, weight=_holder__attr_20, bias=None
        )
        _holder__attr_20 = None
        size_109 = layer_norm_135.size()
        _holder__attr_21 = self._attr_21
        linear_209 = torch.nn.functional.linear(
            input=layer_norm_136, weight=_holder__attr_21, bias=None
        )
        _holder__attr_21 = None
        size_110 = layer_norm_136.size()
        _holder__attr_22 = self._attr_22
        linear_210 = torch.nn.functional.linear(
            input=layer_norm_137, weight=_holder__attr_22, bias=None
        )
        _holder__attr_22 = None
        size_111 = layer_norm_137.size()
        _holder__attr_23 = self._attr_23
        linear_211 = torch.nn.functional.linear(
            input=layer_norm_138, weight=_holder__attr_23, bias=None
        )
        _holder__attr_23 = None
        size_112 = layer_norm_138.size()
        _holder__attr_24 = self._attr_24
        linear_212 = torch.nn.functional.linear(
            input=layer_norm_139, weight=_holder__attr_24, bias=None
        )
        _holder__attr_24 = None
        size_113 = layer_norm_139.size()
        _holder__attr_25 = self._attr_25
        linear_213 = torch.nn.functional.linear(
            input=layer_norm_140, weight=_holder__attr_25, bias=None
        )
        _holder__attr_25 = None
        size_114 = layer_norm_140.size()
        _holder__attr_26 = self._attr_26
        _holder__attr_27 = self._attr_27
        linear_176 = torch.nn.functional.linear(
            input=getitem_3710, weight=_holder__attr_26, bias=_holder__attr_27
        )
        getitem_3710 = _holder__attr_26 = _holder__attr_27 = None
        _holder__attr_28 = self._attr_28
        _holder__attr_29 = self._attr_29
        linear_177 = torch.nn.functional.linear(
            input=getitem_3711, weight=_holder__attr_28, bias=_holder__attr_29
        )
        getitem_3711 = _holder__attr_28 = _holder__attr_29 = None
        _holder__attr_30 = self._attr_30
        _holder__attr_31 = self._attr_31
        linear_178 = torch.nn.functional.linear(
            input=getitem_3712, weight=_holder__attr_30, bias=_holder__attr_31
        )
        getitem_3712 = _holder__attr_30 = _holder__attr_31 = None
        _holder__attr_32 = self._attr_32
        _holder__attr_33 = self._attr_33
        linear_179 = torch.nn.functional.linear(
            input=getitem_3713, weight=_holder__attr_32, bias=_holder__attr_33
        )
        getitem_3713 = _holder__attr_32 = _holder__attr_33 = None
        _holder__attr_34 = self._attr_34
        _holder__attr_35 = self._attr_35
        linear_180 = torch.nn.functional.linear(
            input=getitem_3714, weight=_holder__attr_34, bias=_holder__attr_35
        )
        getitem_3714 = _holder__attr_34 = _holder__attr_35 = None
        _holder__attr_36 = self._attr_36
        _holder__attr_37 = self._attr_37
        linear_181 = torch.nn.functional.linear(
            input=getitem_3715, weight=_holder__attr_36, bias=_holder__attr_37
        )
        getitem_3715 = _holder__attr_36 = _holder__attr_37 = None
        _holder__attr_38 = self._attr_38
        _holder__attr_39 = self._attr_39
        linear_182 = torch.nn.functional.linear(
            input=getitem_3716, weight=_holder__attr_38, bias=_holder__attr_39
        )
        getitem_3716 = _holder__attr_38 = _holder__attr_39 = None
        _holder__attr_40 = self._attr_40
        _holder__attr_41 = self._attr_41
        linear_183 = torch.nn.functional.linear(
            input=nan_to_num_37, weight=_holder__attr_40, bias=_holder__attr_41
        )
        nan_to_num_37 = _holder__attr_40 = _holder__attr_41 = None
        _holder__attr_42 = self._attr_42
        _holder__attr_43 = self._attr_43
        linear_184 = torch.nn.functional.linear(
            input=nan_to_num_38, weight=_holder__attr_42, bias=_holder__attr_43
        )
        nan_to_num_38 = _holder__attr_42 = _holder__attr_43 = None
        _holder__attr_44 = self._attr_44
        _holder__attr_45 = self._attr_45
        linear_185 = torch.nn.functional.linear(
            input=nan_to_num_39, weight=_holder__attr_44, bias=_holder__attr_45
        )
        nan_to_num_39 = _holder__attr_44 = _holder__attr_45 = None
        _holder__attr_46 = self._attr_46
        _holder__attr_47 = self._attr_47
        linear_186 = torch.nn.functional.linear(
            input=nan_to_num_40, weight=_holder__attr_46, bias=_holder__attr_47
        )
        nan_to_num_40 = _holder__attr_46 = _holder__attr_47 = None
        _holder__attr_48 = self._attr_48
        _holder__attr_49 = self._attr_49
        linear_187 = torch.nn.functional.linear(
            input=nan_to_num_41, weight=_holder__attr_48, bias=_holder__attr_49
        )
        nan_to_num_41 = _holder__attr_48 = _holder__attr_49 = None
        _holder__attr_50 = self._attr_50
        _holder__attr_51 = self._attr_51
        linear_188 = torch.nn.functional.linear(
            input=nan_to_num_42, weight=_holder__attr_50, bias=_holder__attr_51
        )
        nan_to_num_42 = _holder__attr_50 = _holder__attr_51 = None
        _holder__attr_52 = self._attr_52
        _holder__attr_53 = self._attr_53
        linear_189 = torch.nn.functional.linear(
            input=nan_to_num_43, weight=_holder__attr_52, bias=_holder__attr_53
        )
        nan_to_num_43 = _holder__attr_52 = _holder__attr_53 = None
        _holder__attr_54 = self._attr_54
        _holder__attr_55 = self._attr_55
        linear_190 = torch.nn.functional.linear(
            input=nan_to_num_44, weight=_holder__attr_54, bias=_holder__attr_55
        )
        nan_to_num_44 = _holder__attr_54 = _holder__attr_55 = None
        _holder__attr_56 = self._attr_56
        _holder__attr_57 = self._attr_57
        linear_191 = torch.nn.functional.linear(
            input=nan_to_num_45, weight=_holder__attr_56, bias=_holder__attr_57
        )
        nan_to_num_45 = _holder__attr_56 = _holder__attr_57 = None
        _holder__attr_58 = self._attr_58
        _holder__attr_59 = self._attr_59
        linear_192 = torch.nn.functional.linear(
            input=nan_to_num_46, weight=_holder__attr_58, bias=_holder__attr_59
        )
        nan_to_num_46 = _holder__attr_58 = _holder__attr_59 = None
        _holder__attr_60 = self._attr_60
        _holder__attr_61 = self._attr_61
        linear_193 = torch.nn.functional.linear(
            input=nan_to_num_47, weight=_holder__attr_60, bias=_holder__attr_61
        )
        nan_to_num_47 = _holder__attr_60 = _holder__attr_61 = None
        _holder__attr_62 = self._attr_62
        _holder__attr_63 = self._attr_63
        linear_194 = torch.nn.functional.linear(
            input=nan_to_num_48, weight=_holder__attr_62, bias=_holder__attr_63
        )
        nan_to_num_48 = _holder__attr_62 = _holder__attr_63 = None
        _holder__attr_64 = self._attr_64
        _holder__attr_65 = self._attr_65
        linear_195 = torch.nn.functional.linear(
            input=nan_to_num_49, weight=_holder__attr_64, bias=_holder__attr_65
        )
        nan_to_num_49 = _holder__attr_64 = _holder__attr_65 = None
        _holder__attr_66 = self._attr_66
        _holder__attr_67 = self._attr_67
        linear_196 = torch.nn.functional.linear(
            input=nan_to_num_50, weight=_holder__attr_66, bias=_holder__attr_67
        )
        nan_to_num_50 = _holder__attr_66 = _holder__attr_67 = None
        _holder__attr_68 = self._attr_68
        _holder__attr_69 = self._attr_69
        linear_197 = torch.nn.functional.linear(
            input=nan_to_num_51, weight=_holder__attr_68, bias=_holder__attr_69
        )
        nan_to_num_51 = _holder__attr_68 = _holder__attr_69 = None
        _holder__attr_70 = self._attr_70
        _holder__attr_71 = self._attr_71
        linear_198 = torch.nn.functional.linear(
            input=nan_to_num_52, weight=_holder__attr_70, bias=_holder__attr_71
        )
        nan_to_num_52 = _holder__attr_70 = _holder__attr_71 = None
        _holder__attr_72 = self._attr_72
        _holder__attr_73 = self._attr_73
        linear_199 = torch.nn.functional.linear(
            input=nan_to_num_53, weight=_holder__attr_72, bias=_holder__attr_73
        )
        nan_to_num_53 = _holder__attr_72 = _holder__attr_73 = None
        _holder__attr_74 = self._attr_74
        _holder__attr_75 = self._attr_75
        linear_200 = torch.nn.functional.linear(
            input=nan_to_num_54, weight=_holder__attr_74, bias=_holder__attr_75
        )
        nan_to_num_54 = _holder__attr_74 = _holder__attr_75 = None
        _holder__attr_76 = self._attr_76
        _holder__attr_77 = self._attr_77
        linear_201 = torch.nn.functional.linear(
            input=nan_to_num_55, weight=_holder__attr_76, bias=_holder__attr_77
        )
        nan_to_num_55 = _holder__attr_76 = _holder__attr_77 = None
        _holder__attr_78 = self._attr_78
        _holder__attr_79 = self._attr_79
        linear_202 = torch.nn.functional.linear(
            input=nan_to_num_56, weight=_holder__attr_78, bias=_holder__attr_79
        )
        nan_to_num_56 = _holder__attr_78 = _holder__attr_79 = None
        _holder__attr_80 = self._attr_80
        _holder__attr_81 = self._attr_81
        linear_203 = torch.nn.functional.linear(
            input=nan_to_num_57, weight=_holder__attr_80, bias=_holder__attr_81
        )
        nan_to_num_57 = _holder__attr_80 = _holder__attr_81 = None
        _holder__attr_82 = self._attr_82
        _holder__attr_83 = self._attr_83
        linear_204 = torch.nn.functional.linear(
            input=nan_to_num_58, weight=_holder__attr_82, bias=_holder__attr_83
        )
        nan_to_num_58 = _holder__attr_82 = _holder__attr_83 = None
        _holder__attr_84 = self._attr_84
        _holder__attr_85 = self._attr_85
        layer_norm_141 = custom_ln(
            input=repeat,
            normalized_shape=(64,),
            weight=_holder__attr_84,
            bias=_holder__attr_85,
            eps=1e-05,
        )
        _holder__attr_84 = _holder__attr_85 = None
        _holder__attr_86 = self._attr_86
        _holder__attr_87 = self._attr_87
        layer_norm_142 = custom_ln(
            input=repeat_1,
            normalized_shape=(64,),
            weight=_holder__attr_86,
            bias=_holder__attr_87,
            eps=1e-05,
        )
        _holder__attr_86 = _holder__attr_87 = None
        _holder__attr_88 = self._attr_88
        _holder__attr_89 = self._attr_89
        layer_norm_143 = custom_ln(
            input=repeat_2,
            normalized_shape=(64,),
            weight=_holder__attr_88,
            bias=_holder__attr_89,
            eps=1e-05,
        )
        _holder__attr_88 = _holder__attr_89 = None
        _holder__attr_90 = self._attr_90
        _holder__attr_91 = self._attr_91
        layer_norm_144 = custom_ln(
            input=repeat_3,
            normalized_shape=(64,),
            weight=_holder__attr_90,
            bias=_holder__attr_91,
            eps=1e-05,
        )
        _holder__attr_90 = _holder__attr_91 = None
        _holder__attr_92 = self._attr_92
        _holder__attr_93 = self._attr_93
        layer_norm_145 = custom_ln(
            input=repeat_4,
            normalized_shape=(64,),
            weight=_holder__attr_92,
            bias=_holder__attr_93,
            eps=1e-05,
        )
        _holder__attr_92 = _holder__attr_93 = None
        _holder__attr_94 = self._attr_94
        _holder__attr_95 = self._attr_95
        layer_norm_146 = custom_ln(
            input=repeat_5,
            normalized_shape=(64,),
            weight=_holder__attr_94,
            bias=_holder__attr_95,
            eps=1e-05,
        )
        _holder__attr_94 = _holder__attr_95 = None
        _holder__attr_96 = self._attr_96
        _holder__attr_97 = self._attr_97
        layer_norm_104 = custom_ln(
            input=getitem_4137,
            normalized_shape=getitem_3709,
            weight=_holder__attr_96,
            bias=_holder__attr_97,
            eps=1e-05,
        )
        getitem_3709 = _holder__attr_96 = _holder__attr_97 = None
        _holder__attr_98 = self._attr_98
        _holder__attr_99 = self._attr_99
        linear_166 = torch.nn.functional.linear(
            input=mul_59, weight=_holder__attr_98, bias=_holder__attr_99
        )
        mul_59 = _holder__attr_98 = _holder__attr_99 = None
        _holder__attr_100 = self._attr_100
        _holder__attr_101 = self._attr_101
        linear_167 = torch.nn.functional.linear(
            input=mul_60, weight=_holder__attr_100, bias=_holder__attr_101
        )
        mul_60 = _holder__attr_100 = _holder__attr_101 = None
        _holder__attr_102 = self._attr_102
        _holder__attr_103 = self._attr_103
        linear_168 = torch.nn.functional.linear(
            input=mul_61, weight=_holder__attr_102, bias=_holder__attr_103
        )
        mul_61 = _holder__attr_102 = _holder__attr_103 = None
        _holder__attr_104 = self._attr_104
        _holder__attr_105 = self._attr_105
        linear_169 = torch.nn.functional.linear(
            input=mul_62, weight=_holder__attr_104, bias=_holder__attr_105
        )
        mul_62 = _holder__attr_104 = _holder__attr_105 = None
        _holder__attr_106 = self._attr_106
        _holder__attr_107 = self._attr_107
        linear_170 = torch.nn.functional.linear(
            input=mul_63, weight=_holder__attr_106, bias=_holder__attr_107
        )
        mul_63 = _holder__attr_106 = _holder__attr_107 = None
        _holder__attr_108 = self._attr_108
        _holder__attr_109 = self._attr_109
        linear_171 = torch.nn.functional.linear(
            input=mul_64, weight=_holder__attr_108, bias=_holder__attr_109
        )
        mul_64 = _holder__attr_108 = _holder__attr_109 = None
        _holder__attr_110 = self._attr_110
        _holder__attr_111 = self._attr_111
        linear_172 = torch.nn.functional.linear(
            input=mul_65, weight=_holder__attr_110, bias=_holder__attr_111
        )
        mul_65 = _holder__attr_110 = _holder__attr_111 = None
        _holder__attr_112 = self._attr_112
        _holder__attr_113 = self._attr_113
        linear_173 = torch.nn.functional.linear(
            input=mul_66, weight=_holder__attr_112, bias=_holder__attr_113
        )
        mul_66 = _holder__attr_112 = _holder__attr_113 = None
        getitem_3775 = size_109[1]
        size_109 = None
        getitem_3776 = size_110[1]
        size_110 = None
        getitem_3777 = size_111[1]
        size_111 = None
        getitem_3778 = size_112[1]
        size_112 = None
        getitem_3779 = size_113[1]
        size_113 = None
        getitem_3780 = size_114[1]
        size_114 = None
        size_74 = linear_176.size()
        size_75 = linear_177.size()
        size_76 = linear_178.size()
        size_77 = linear_179.size()
        size_78 = linear_180.size()
        size_79 = linear_181.size()
        size_80 = linear_182.size()
        size_81 = linear_183.size()
        size_82 = linear_184.size()
        size_83 = linear_185.size()
        size_84 = linear_186.size()
        size_85 = linear_187.size()
        size_86 = linear_188.size()
        size_87 = linear_189.size()
        size_88 = linear_190.size()
        size_89 = linear_191.size()
        size_90 = linear_192.size()
        size_91 = linear_193.size()
        size_92 = linear_194.size()
        size_93 = linear_195.size()
        size_94 = linear_196.size()
        size_95 = linear_197.size()
        size_96 = linear_198.size()
        size_97 = linear_199.size()
        size_98 = linear_200.size()
        size_99 = linear_201.size()
        size_100 = linear_202.size()
        size_101 = linear_203.size()
        size_102 = linear_204.size()
        _holder__attr_114 = self._attr_114
        _holder__attr_115 = self._attr_115
        linear_214 = torch.nn.functional.linear(
            input=layer_norm_141, weight=_holder__attr_114, bias=_holder__attr_115
        )
        _holder__attr_114 = _holder__attr_115 = None
        size_115 = layer_norm_141.size()
        layer_norm_141 = None
        _holder__attr_116 = self._attr_116
        _holder__attr_117 = self._attr_117
        linear_215 = torch.nn.functional.linear(
            input=layer_norm_142, weight=_holder__attr_116, bias=_holder__attr_117
        )
        _holder__attr_116 = _holder__attr_117 = None
        size_116 = layer_norm_142.size()
        layer_norm_142 = None
        _holder__attr_118 = self._attr_118
        _holder__attr_119 = self._attr_119
        linear_216 = torch.nn.functional.linear(
            input=layer_norm_143, weight=_holder__attr_118, bias=_holder__attr_119
        )
        _holder__attr_118 = _holder__attr_119 = None
        size_117 = layer_norm_143.size()
        layer_norm_143 = None
        _holder__attr_120 = self._attr_120
        _holder__attr_121 = self._attr_121
        linear_217 = torch.nn.functional.linear(
            input=layer_norm_144, weight=_holder__attr_120, bias=_holder__attr_121
        )
        _holder__attr_120 = _holder__attr_121 = None
        size_118 = layer_norm_144.size()
        layer_norm_144 = None
        _holder__attr_122 = self._attr_122
        _holder__attr_123 = self._attr_123
        linear_218 = torch.nn.functional.linear(
            input=layer_norm_145, weight=_holder__attr_122, bias=_holder__attr_123
        )
        _holder__attr_122 = _holder__attr_123 = None
        size_119 = layer_norm_145.size()
        layer_norm_145 = None
        _holder__attr_124 = self._attr_124
        _holder__attr_125 = self._attr_125
        linear_219 = torch.nn.functional.linear(
            input=layer_norm_146, weight=_holder__attr_124, bias=_holder__attr_125
        )
        _holder__attr_124 = _holder__attr_125 = None
        size_120 = layer_norm_146.size()
        layer_norm_146 = None
        sigmoid_59 = torch.sigmoid(input=layer_norm_104)
        layer_norm_104 = None
        sigmoid_60 = torch.sigmoid(input=linear_166)
        linear_166 = None
        sigmoid_61 = torch.sigmoid(input=linear_167)
        linear_167 = None
        sigmoid_62 = torch.sigmoid(input=linear_168)
        linear_168 = None
        sigmoid_63 = torch.sigmoid(input=linear_169)
        linear_169 = None
        sigmoid_64 = torch.sigmoid(input=linear_170)
        linear_170 = None
        sigmoid_65 = torch.sigmoid(input=linear_171)
        linear_171 = None
        sigmoid_66 = torch.sigmoid(input=linear_172)
        linear_172 = None
        sigmoid_67 = torch.sigmoid(input=linear_173)
        linear_173 = None
        getitem_3740 = size_74[1:]
        size_74 = None
        getitem_3741 = size_75[1:]
        size_75 = None
        getitem_3742 = size_76[1:]
        size_76 = None
        getitem_3743 = size_77[1:]
        size_77 = None
        getitem_3744 = size_78[1:]
        size_78 = None
        getitem_3745 = size_79[1:]
        size_79 = None
        getitem_3746 = size_80[1:]
        size_80 = None
        getitem_3747 = size_81[1:]
        size_81 = None
        getitem_3748 = size_82[1:]
        size_82 = None
        getitem_3749 = size_83[1:]
        size_83 = None
        getitem_3750 = size_84[1:]
        size_84 = None
        getitem_3751 = size_85[1:]
        size_85 = None
        getitem_3752 = size_86[1:]
        size_86 = None
        getitem_3753 = size_87[1:]
        size_87 = None
        getitem_3754 = size_88[1:]
        size_88 = None
        getitem_3755 = size_89[1:]
        size_89 = None
        getitem_3756 = size_90[1:]
        size_90 = None
        getitem_3757 = size_91[1:]
        size_91 = None
        getitem_3758 = size_92[1:]
        size_92 = None
        getitem_3759 = size_93[1:]
        size_93 = None
        getitem_3760 = size_94[1:]
        size_94 = None
        getitem_3761 = size_95[1:]
        size_95 = None
        getitem_3762 = size_96[1:]
        size_96 = None
        getitem_3763 = size_97[1:]
        size_97 = None
        getitem_3764 = size_98[1:]
        size_98 = None
        getitem_3765 = size_99[1:]
        size_99 = None
        getitem_3766 = size_100[1:]
        size_100 = None
        getitem_3767 = size_101[1:]
        size_101 = None
        getitem_3768 = size_102[1:]
        size_102 = None
        getitem_3782 = size_115[1]
        getitem_3783 = size_115[2]
        size_115 = None
        getitem_3785 = size_116[1]
        getitem_3786 = size_116[2]
        size_116 = None
        getitem_3788 = size_117[1]
        getitem_3789 = size_117[2]
        size_117 = None
        getitem_3791 = size_118[1]
        getitem_3792 = size_118[2]
        size_118 = None
        getitem_3794 = size_119[1]
        getitem_3795 = size_119[2]
        size_119 = None
        getitem_3797 = size_120[1]
        getitem_3798 = size_120[2]
        size_120 = None
        mul_67 = torch.mul(input=getitem_4137, other=sigmoid_59)
        getitem_4137 = sigmoid_59 = None
        _holder__attr_126 = self._attr_126
        _holder__attr_127 = self._attr_127
        layer_norm_106 = custom_ln(
            input=linear_176,
            normalized_shape=getitem_3740,
            weight=_holder__attr_126,
            bias=_holder__attr_127,
            eps=0.0001,
        )
        linear_176 = getitem_3740 = _holder__attr_126 = _holder__attr_127 = None
        _holder__attr_128 = self._attr_128
        _holder__attr_129 = self._attr_129
        layer_norm_107 = custom_ln(
            input=linear_177,
            normalized_shape=getitem_3741,
            weight=_holder__attr_128,
            bias=_holder__attr_129,
            eps=0.0001,
        )
        linear_177 = getitem_3741 = _holder__attr_128 = _holder__attr_129 = None
        _holder__attr_130 = self._attr_130
        _holder__attr_131 = self._attr_131
        layer_norm_108 = custom_ln(
            input=linear_178,
            normalized_shape=getitem_3742,
            weight=_holder__attr_130,
            bias=_holder__attr_131,
            eps=0.0001,
        )
        linear_178 = getitem_3742 = _holder__attr_130 = _holder__attr_131 = None
        _holder__attr_132 = self._attr_132
        _holder__attr_133 = self._attr_133
        layer_norm_109 = custom_ln(
            input=linear_179,
            normalized_shape=getitem_3743,
            weight=_holder__attr_132,
            bias=_holder__attr_133,
            eps=0.0001,
        )
        linear_179 = getitem_3743 = _holder__attr_132 = _holder__attr_133 = None
        _holder__attr_134 = self._attr_134
        _holder__attr_135 = self._attr_135
        layer_norm_110 = custom_ln(
            input=linear_180,
            normalized_shape=getitem_3744,
            weight=_holder__attr_134,
            bias=_holder__attr_135,
            eps=0.0001,
        )
        linear_180 = getitem_3744 = _holder__attr_134 = _holder__attr_135 = None
        _holder__attr_136 = self._attr_136
        _holder__attr_137 = self._attr_137
        layer_norm_111 = custom_ln(
            input=linear_181,
            normalized_shape=getitem_3745,
            weight=_holder__attr_136,
            bias=_holder__attr_137,
            eps=0.0001,
        )
        linear_181 = getitem_3745 = _holder__attr_136 = _holder__attr_137 = None
        _holder__attr_138 = self._attr_138
        _holder__attr_139 = self._attr_139
        layer_norm_112 = custom_ln(
            input=linear_182,
            normalized_shape=getitem_3746,
            weight=_holder__attr_138,
            bias=_holder__attr_139,
            eps=0.0001,
        )
        linear_182 = getitem_3746 = _holder__attr_138 = _holder__attr_139 = None
        _holder__attr_140 = self._attr_140
        _holder__attr_141 = self._attr_141
        layer_norm_113 = custom_ln(
            input=linear_183,
            normalized_shape=getitem_3747,
            weight=_holder__attr_140,
            bias=_holder__attr_141,
            eps=0.0001,
        )
        linear_183 = getitem_3747 = _holder__attr_140 = _holder__attr_141 = None
        _holder__attr_142 = self._attr_142
        _holder__attr_143 = self._attr_143
        layer_norm_114 = custom_ln(
            input=linear_184,
            normalized_shape=getitem_3748,
            weight=_holder__attr_142,
            bias=_holder__attr_143,
            eps=0.0001,
        )
        linear_184 = getitem_3748 = _holder__attr_142 = _holder__attr_143 = None
        _holder__attr_144 = self._attr_144
        _holder__attr_145 = self._attr_145
        layer_norm_115 = custom_ln(
            input=linear_185,
            normalized_shape=getitem_3749,
            weight=_holder__attr_144,
            bias=_holder__attr_145,
            eps=0.0001,
        )
        linear_185 = getitem_3749 = _holder__attr_144 = _holder__attr_145 = None
        _holder__attr_146 = self._attr_146
        _holder__attr_147 = self._attr_147
        layer_norm_116 = custom_ln(
            input=linear_186,
            normalized_shape=getitem_3750,
            weight=_holder__attr_146,
            bias=_holder__attr_147,
            eps=0.0001,
        )
        linear_186 = getitem_3750 = _holder__attr_146 = _holder__attr_147 = None
        _holder__attr_148 = self._attr_148
        _holder__attr_149 = self._attr_149
        layer_norm_117 = custom_ln(
            input=linear_187,
            normalized_shape=getitem_3751,
            weight=_holder__attr_148,
            bias=_holder__attr_149,
            eps=0.0001,
        )
        linear_187 = getitem_3751 = _holder__attr_148 = _holder__attr_149 = None
        _holder__attr_150 = self._attr_150
        _holder__attr_151 = self._attr_151
        layer_norm_118 = custom_ln(
            input=linear_188,
            normalized_shape=getitem_3752,
            weight=_holder__attr_150,
            bias=_holder__attr_151,
            eps=0.0001,
        )
        linear_188 = getitem_3752 = _holder__attr_150 = _holder__attr_151 = None
        _holder__attr_152 = self._attr_152
        _holder__attr_153 = self._attr_153
        layer_norm_119 = custom_ln(
            input=linear_189,
            normalized_shape=getitem_3753,
            weight=_holder__attr_152,
            bias=_holder__attr_153,
            eps=0.0001,
        )
        linear_189 = getitem_3753 = _holder__attr_152 = _holder__attr_153 = None
        _holder__attr_154 = self._attr_154
        _holder__attr_155 = self._attr_155
        layer_norm_120 = custom_ln(
            input=linear_190,
            normalized_shape=getitem_3754,
            weight=_holder__attr_154,
            bias=_holder__attr_155,
            eps=0.0001,
        )
        linear_190 = getitem_3754 = _holder__attr_154 = _holder__attr_155 = None
        _holder__attr_156 = self._attr_156
        _holder__attr_157 = self._attr_157
        layer_norm_121 = custom_ln(
            input=linear_191,
            normalized_shape=getitem_3755,
            weight=_holder__attr_156,
            bias=_holder__attr_157,
            eps=0.0001,
        )
        linear_191 = getitem_3755 = _holder__attr_156 = _holder__attr_157 = None
        _holder__attr_158 = self._attr_158
        _holder__attr_159 = self._attr_159
        layer_norm_122 = custom_ln(
            input=linear_192,
            normalized_shape=getitem_3756,
            weight=_holder__attr_158,
            bias=_holder__attr_159,
            eps=0.0001,
        )
        linear_192 = getitem_3756 = _holder__attr_158 = _holder__attr_159 = None
        _holder__attr_160 = self._attr_160
        _holder__attr_161 = self._attr_161
        layer_norm_123 = custom_ln(
            input=linear_193,
            normalized_shape=getitem_3757,
            weight=_holder__attr_160,
            bias=_holder__attr_161,
            eps=0.0001,
        )
        linear_193 = getitem_3757 = _holder__attr_160 = _holder__attr_161 = None
        _holder__attr_162 = self._attr_162
        _holder__attr_163 = self._attr_163
        layer_norm_124 = custom_ln(
            input=linear_194,
            normalized_shape=getitem_3758,
            weight=_holder__attr_162,
            bias=_holder__attr_163,
            eps=0.0001,
        )
        linear_194 = getitem_3758 = _holder__attr_162 = _holder__attr_163 = None
        _holder__attr_164 = self._attr_164
        _holder__attr_165 = self._attr_165
        layer_norm_125 = custom_ln(
            input=linear_195,
            normalized_shape=getitem_3759,
            weight=_holder__attr_164,
            bias=_holder__attr_165,
            eps=0.0001,
        )
        linear_195 = getitem_3759 = _holder__attr_164 = _holder__attr_165 = None
        _holder__attr_166 = self._attr_166
        _holder__attr_167 = self._attr_167
        layer_norm_126 = custom_ln(
            input=linear_196,
            normalized_shape=getitem_3760,
            weight=_holder__attr_166,
            bias=_holder__attr_167,
            eps=0.0001,
        )
        linear_196 = getitem_3760 = _holder__attr_166 = _holder__attr_167 = None
        _holder__attr_168 = self._attr_168
        _holder__attr_169 = self._attr_169
        layer_norm_127 = custom_ln(
            input=linear_197,
            normalized_shape=getitem_3761,
            weight=_holder__attr_168,
            bias=_holder__attr_169,
            eps=0.0001,
        )
        linear_197 = getitem_3761 = _holder__attr_168 = _holder__attr_169 = None
        _holder__attr_170 = self._attr_170
        _holder__attr_171 = self._attr_171
        layer_norm_128 = custom_ln(
            input=linear_198,
            normalized_shape=getitem_3762,
            weight=_holder__attr_170,
            bias=_holder__attr_171,
            eps=0.0001,
        )
        linear_198 = getitem_3762 = _holder__attr_170 = _holder__attr_171 = None
        _holder__attr_172 = self._attr_172
        _holder__attr_173 = self._attr_173
        layer_norm_129 = custom_ln(
            input=linear_199,
            normalized_shape=getitem_3763,
            weight=_holder__attr_172,
            bias=_holder__attr_173,
            eps=0.0001,
        )
        linear_199 = getitem_3763 = _holder__attr_172 = _holder__attr_173 = None
        _holder__attr_174 = self._attr_174
        _holder__attr_175 = self._attr_175
        layer_norm_130 = custom_ln(
            input=linear_200,
            normalized_shape=getitem_3764,
            weight=_holder__attr_174,
            bias=_holder__attr_175,
            eps=0.0001,
        )
        linear_200 = getitem_3764 = _holder__attr_174 = _holder__attr_175 = None
        _holder__attr_176 = self._attr_176
        _holder__attr_177 = self._attr_177
        layer_norm_131 = custom_ln(
            input=linear_201,
            normalized_shape=getitem_3765,
            weight=_holder__attr_176,
            bias=_holder__attr_177,
            eps=0.0001,
        )
        linear_201 = getitem_3765 = _holder__attr_176 = _holder__attr_177 = None
        _holder__attr_178 = self._attr_178
        _holder__attr_179 = self._attr_179
        layer_norm_132 = custom_ln(
            input=linear_202,
            normalized_shape=getitem_3766,
            weight=_holder__attr_178,
            bias=_holder__attr_179,
            eps=0.0001,
        )
        linear_202 = getitem_3766 = _holder__attr_178 = _holder__attr_179 = None
        _holder__attr_180 = self._attr_180
        _holder__attr_181 = self._attr_181
        layer_norm_133 = custom_ln(
            input=linear_203,
            normalized_shape=getitem_3767,
            weight=_holder__attr_180,
            bias=_holder__attr_181,
            eps=0.0001,
        )
        linear_203 = getitem_3767 = _holder__attr_180 = _holder__attr_181 = None
        _holder__attr_182 = self._attr_182
        _holder__attr_183 = self._attr_183
        layer_norm_134 = custom_ln(
            input=linear_204,
            normalized_shape=getitem_3768,
            weight=_holder__attr_182,
            bias=_holder__attr_183,
            eps=0.0001,
        )
        linear_204 = getitem_3768 = _holder__attr_182 = _holder__attr_183 = None
        size = linear_208.size()
        getitem_4151 = size[0]
        size = None
        reshape_343 = torch.reshape(
            input=linear_208,
            shape=(getitem_4151, getitem_3775, 1, 64),
        )
        linear_208 = getitem_4151 = None
        size_144 = layer_norm_135.size()
        getitem_4152 = size_144[0]
        size_144 = None
        reshape_344 = torch.reshape(
            input=layer_norm_135,
            shape=(getitem_4152, getitem_3775, 1, 64),
        )
        layer_norm_135 = getitem_4152 = getitem_3775 = None
        size_145 = linear_214.size()
        getitem_4153 = size_145[0]
        size_145 = None
        reshape_345 = torch.reshape(
            input=linear_214,
            shape=(getitem_4153, getitem_3782, 1, 64),
        )
        linear_214 = getitem_4153 = None
        size_146 = linear_209.size()
        getitem_4154 = size_146[0]
        size_146 = None
        reshape_346 = torch.reshape(
            input=linear_209,
            shape=(getitem_4154, getitem_3776, 1, 64),
        )
        linear_209 = getitem_4154 = None
        size_147 = layer_norm_136.size()
        getitem_4155 = size_147[0]
        size_147 = None
        reshape_347 = torch.reshape(
            input=layer_norm_136,
            shape=(getitem_4155, getitem_3776, 1, 64),
        )
        layer_norm_136 = getitem_4155 = getitem_3776 = None
        size_148 = linear_215.size()
        getitem_4156 = size_148[0]
        size_148 = None
        reshape_348 = torch.reshape(
            input=linear_215,
            shape=(getitem_4156, getitem_3785, 1, 64),
        )
        linear_215 = getitem_4156 = None
        size_149 = linear_210.size()
        getitem_4157 = size_149[0]
        size_149 = None
        reshape_349 = torch.reshape(
            input=linear_210,
            shape=(getitem_4157, getitem_3777, 1, 64),
        )
        linear_210 = getitem_4157 = None
        size_150 = layer_norm_137.size()
        getitem_4158 = size_150[0]
        size_150 = None
        reshape_350 = torch.reshape(
            input=layer_norm_137,
            shape=(getitem_4158, getitem_3777, 1, 64),
        )
        layer_norm_137 = getitem_4158 = getitem_3777 = None
        size_151 = linear_216.size()
        getitem_4159 = size_151[0]
        size_151 = None
        reshape_351 = torch.reshape(
            input=linear_216,
            shape=(getitem_4159, getitem_3788, 1, 64),
        )
        linear_216 = getitem_4159 = None
        size_152 = linear_211.size()
        getitem_4160 = size_152[0]
        size_152 = None
        reshape_352 = torch.reshape(
            input=linear_211,
            shape=(getitem_4160, getitem_3778, 1, 64),
        )
        linear_211 = getitem_4160 = None
        size_153 = layer_norm_138.size()
        getitem_4161 = size_153[0]
        size_153 = None
        reshape_353 = torch.reshape(
            input=layer_norm_138,
            shape=(getitem_4161, getitem_3778, 1, 64),
        )
        layer_norm_138 = getitem_4161 = getitem_3778 = None
        size_154 = linear_217.size()
        getitem_4162 = size_154[0]
        size_154 = None
        reshape_354 = torch.reshape(
            input=linear_217,
            shape=(getitem_4162, getitem_3791, 1, 64),
        )
        linear_217 = getitem_4162 = None
        size_155 = linear_212.size()
        getitem_4163 = size_155[0]
        size_155 = None
        reshape_355 = torch.reshape(
            input=linear_212,
            shape=(getitem_4163, getitem_3779, 1, 64),
        )
        linear_212 = getitem_4163 = None
        size_156 = layer_norm_139.size()
        getitem_4164 = size_156[0]
        size_156 = None
        reshape_356 = torch.reshape(
            input=layer_norm_139,
            shape=(getitem_4164, getitem_3779, 1, 64),
        )
        layer_norm_139 = getitem_4164 = getitem_3779 = None
        size_157 = linear_218.size()
        getitem_4165 = size_157[0]
        size_157 = None
        reshape_357 = torch.reshape(
            input=linear_218,
            shape=(getitem_4165, getitem_3794, 1, 64),
        )
        linear_218 = getitem_4165 = None
        size_158 = linear_213.size()
        getitem_4166 = size_158[0]
        size_158 = None
        reshape_358 = torch.reshape(
            input=linear_213,
            shape=(getitem_4166, getitem_3780, 1, 64),
        )
        linear_213 = getitem_4166 = None
        size_159 = layer_norm_140.size()
        getitem_4167 = size_159[0]
        size_159 = None
        reshape_359 = torch.reshape(
            input=layer_norm_140,
            shape=(getitem_4167, getitem_3780, 1, 64),
        )
        layer_norm_140 = getitem_4167 = getitem_3780 = None
        size_160 = linear_219.size()
        getitem_4168 = size_160[0]
        size_160 = None
        reshape_360 = torch.reshape(
            input=linear_219,
            shape=(getitem_4168, getitem_3797, 1, 64),
        )
        linear_219 = getitem_4168 = None
        _holder__attr_184 = self._attr_184
        _holder__attr_185 = self._attr_185
        linear_174 = torch.nn.functional.linear(
            input=mul_67, weight=_holder__attr_184, bias=_holder__attr_185
        )
        mul_67 = _holder__attr_184 = _holder__attr_185 = None
        permute_50 = reshape_343.permute([0, 2, 1, 3])
        reshape_343 = None
        permute_51 = reshape_344.permute([0, 2, 1, 3])
        reshape_344 = None
        permute_52 = reshape_345.permute([0, 2, 1, 3])
        reshape_345 = None
        permute_53 = reshape_346.permute([0, 2, 1, 3])
        reshape_346 = None
        permute_54 = reshape_347.permute([0, 2, 1, 3])
        reshape_347 = None
        permute_55 = reshape_348.permute([0, 2, 1, 3])
        reshape_348 = None
        permute_56 = reshape_349.permute([0, 2, 1, 3])
        reshape_349 = None
        permute_57 = reshape_350.permute([0, 2, 1, 3])
        reshape_350 = None
        permute_58 = reshape_351.permute([0, 2, 1, 3])
        reshape_351 = None
        permute_59 = reshape_352.permute([0, 2, 1, 3])
        reshape_352 = None
        permute_60 = reshape_353.permute([0, 2, 1, 3])
        reshape_353 = None
        permute_61 = reshape_354.permute([0, 2, 1, 3])
        reshape_354 = None
        permute_62 = reshape_355.permute([0, 2, 1, 3])
        reshape_355 = None
        permute_63 = reshape_356.permute([0, 2, 1, 3])
        reshape_356 = None
        permute_64 = reshape_357.permute([0, 2, 1, 3])
        reshape_357 = None
        permute_65 = reshape_358.permute([0, 2, 1, 3])
        reshape_358 = None
        permute_66 = reshape_359.permute([0, 2, 1, 3])
        reshape_359 = None
        permute_67 = reshape_360.permute([0, 2, 1, 3])
        reshape_360 = None
        sigmoid_68 = torch.sigmoid(input=linear_174)
        linear_174 = None
        scaled_dot_product_attention = torch._C._nn.scaled_dot_product_attention(
            permute_52,
            permute_50,
            permute_51,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
        permute_52 = permute_50 = permute_51 = None
        scaled_dot_product_attention_1 = torch._C._nn.scaled_dot_product_attention(
            permute_55,
            permute_53,
            permute_54,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
        permute_55 = permute_53 = permute_54 = None
        scaled_dot_product_attention_2 = torch._C._nn.scaled_dot_product_attention(
            permute_58,
            permute_56,
            permute_57,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
        permute_58 = permute_56 = permute_57 = None
        scaled_dot_product_attention_3 = torch._C._nn.scaled_dot_product_attention(
            permute_61,
            permute_59,
            permute_60,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
        permute_61 = permute_59 = permute_60 = None
        scaled_dot_product_attention_4 = torch._C._nn.scaled_dot_product_attention(
            permute_64,
            permute_62,
            permute_63,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
        permute_64 = permute_62 = permute_63 = None
        scaled_dot_product_attention_5 = torch._C._nn.scaled_dot_product_attention(
            permute_67,
            permute_65,
            permute_66,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )
        permute_67 = permute_65 = permute_66 = None
        mul_68 = torch.mul(input=getitem_3227, other=sigmoid_68)
        getitem_3227 = sigmoid_68 = None
        permute_68 = scaled_dot_product_attention.permute([0, 2, 1, 3])
        scaled_dot_product_attention = None
        permute_69 = scaled_dot_product_attention_1.permute([0, 2, 1, 3])
        scaled_dot_product_attention_1 = None
        permute_70 = scaled_dot_product_attention_2.permute([0, 2, 1, 3])
        scaled_dot_product_attention_2 = None
        permute_71 = scaled_dot_product_attention_3.permute([0, 2, 1, 3])
        scaled_dot_product_attention_3 = None
        permute_72 = scaled_dot_product_attention_4.permute([0, 2, 1, 3])
        scaled_dot_product_attention_4 = None
        permute_73 = scaled_dot_product_attention_5.permute([0, 2, 1, 3])
        scaled_dot_product_attention_5 = None
        _holder__attr_186 = self._attr_186
        _holder__attr_187 = self._attr_187
        linear_175 = torch.nn.functional.linear(
            input=mul_68, weight=_holder__attr_186, bias=_holder__attr_187
        )
        mul_68 = _holder__attr_186 = _holder__attr_187 = None

        contiguous_15 = permute_68.contiguous()
        permute_68 = None
        contiguous_16 = permute_69.contiguous()
        permute_69 = None
        contiguous_17 = permute_70.contiguous()
        permute_70 = None
        contiguous_18 = permute_71.contiguous()
        permute_71 = None
        contiguous_19 = permute_72.contiguous()
        permute_72 = None
        contiguous_20 = permute_73.contiguous()
        permute_73 = None
        size_73 = linear_175.size()
        size_161 = contiguous_15.size()
        getitem_4169 = size_161[0]
        size_161 = None
        reshape_361 = torch.reshape(
            input=contiguous_15,
            shape=(getitem_4169, getitem_3782, getitem_3783),
        )
        contiguous_15 = getitem_4169 = getitem_3782 = getitem_3783 = None
        size_162 = contiguous_16.size()
        getitem_4170 = size_162[0]
        size_162 = None
        reshape_362 = torch.reshape(
            input=contiguous_16,
            shape=(getitem_4170, getitem_3785, getitem_3786),
        )
        contiguous_16 = getitem_4170 = getitem_3785 = getitem_3786 = None
        size_163 = contiguous_17.size()
        getitem_4171 = size_163[0]
        size_163 = None
        reshape_363 = torch.reshape(
            input=contiguous_17,
            shape=(getitem_4171, getitem_3788, getitem_3789),
        )
        contiguous_17 = getitem_4171 = getitem_3788 = getitem_3789 = None
        size_164 = contiguous_18.size()
        getitem_4172 = size_164[0]
        size_164 = None
        reshape_364 = torch.reshape(
            input=contiguous_18,
            shape=(getitem_4172, getitem_3791, getitem_3792),
        )
        contiguous_18 = getitem_4172 = getitem_3791 = getitem_3792 = None
        size_165 = contiguous_19.size()
        getitem_4173 = size_165[0]
        size_165 = None
        reshape_365 = torch.reshape(
            input=contiguous_19,
            shape=(getitem_4173, getitem_3794, getitem_3795),
        )
        contiguous_19 = getitem_4173 = getitem_3794 = getitem_3795 = None
        size_166 = contiguous_20.size()
        getitem_4174 = size_166[0]
        size_166 = None
        reshape_366 = torch.reshape(
            input=contiguous_20,
            shape=(getitem_4174, getitem_3797, getitem_3798),
        )
        contiguous_20 = getitem_4174 = getitem_3797 = getitem_3798 = None
        getitem_3739 = size_73[1:]
        size_73 = None
        add_48 = torch.add(input=reshape_361, other=repeat)
        reshape_361 = repeat = None
        add_49 = torch.add(input=reshape_362, other=repeat_1)
        reshape_362 = repeat_1 = None
        add_50 = torch.add(input=reshape_363, other=repeat_2)
        reshape_363 = repeat_2 = None
        add_51 = torch.add(input=reshape_364, other=repeat_3)
        reshape_364 = repeat_3 = None
        add_52 = torch.add(input=reshape_365, other=repeat_4)
        reshape_365 = repeat_4 = None
        add_53 = torch.add(input=reshape_366, other=repeat_5)
        reshape_366 = repeat_5 = None
        _holder__attr_188 = self._attr_188
        _holder__attr_189 = self._attr_189
        layer_norm_105 = custom_ln(
            input=linear_175,
            normalized_shape=getitem_3739,
            weight=_holder__attr_188,
            bias=_holder__attr_189,
            eps=1e-05,
        )
        getitem_3739 = _holder__attr_188 = _holder__attr_189 = None
        _holder__attr_190 = self._attr_190
        _holder__attr_191 = self._attr_191
        layer_norm_147 = custom_ln(
            input=add_48,
            normalized_shape=(64,),
            weight=_holder__attr_190,
            bias=_holder__attr_191,
            eps=1e-05,
        )
        _holder__attr_190 = _holder__attr_191 = None
        _holder__attr_192 = self._attr_192
        _holder__attr_193 = self._attr_193
        layer_norm_148 = custom_ln(
            input=add_49,
            normalized_shape=(64,),
            weight=_holder__attr_192,
            bias=_holder__attr_193,
            eps=1e-05,
        )
        _holder__attr_192 = _holder__attr_193 = None
        _holder__attr_194 = self._attr_194
        _holder__attr_195 = self._attr_195
        layer_norm_149 = custom_ln(
            input=add_50,
            normalized_shape=(64,),
            weight=_holder__attr_194,
            bias=_holder__attr_195,
            eps=1e-05,
        )
        _holder__attr_194 = _holder__attr_195 = None
        _holder__attr_196 = self._attr_196
        _holder__attr_197 = self._attr_197
        layer_norm_150 = custom_ln(
            input=add_51,
            normalized_shape=(64,),
            weight=_holder__attr_196,
            bias=_holder__attr_197,
            eps=1e-05,
        )
        _holder__attr_196 = _holder__attr_197 = None
        _holder__attr_198 = self._attr_198
        _holder__attr_199 = self._attr_199
        layer_norm_151 = custom_ln(
            input=add_52,
            normalized_shape=(64,),
            weight=_holder__attr_198,
            bias=_holder__attr_199,
            eps=1e-05,
        )
        _holder__attr_198 = _holder__attr_199 = None
        _holder__attr_200 = self._attr_200
        _holder__attr_201 = self._attr_201
        layer_norm_152 = custom_ln(
            input=add_53,
            normalized_shape=(64,),
            weight=_holder__attr_200,
            bias=_holder__attr_201,
            eps=1e-05,
        )
        _holder__attr_200 = _holder__attr_201 = None
        sigmoid_69 = torch.sigmoid(input=layer_norm_105)
        layer_norm_105 = None
        _holder__attr_202 = self._attr_202
        _holder__attr_203 = self._attr_203
        linear_220 = torch.nn.functional.linear(
            input=layer_norm_147, weight=_holder__attr_202, bias=_holder__attr_203
        )
        layer_norm_147 = _holder__attr_202 = _holder__attr_203 = None
        _holder__attr_204 = self._attr_204
        _holder__attr_205 = self._attr_205
        linear_221 = torch.nn.functional.linear(
            input=layer_norm_148, weight=_holder__attr_204, bias=_holder__attr_205
        )
        layer_norm_148 = _holder__attr_204 = _holder__attr_205 = None
        _holder__attr_206 = self._attr_206
        _holder__attr_207 = self._attr_207
        linear_222 = torch.nn.functional.linear(
            input=layer_norm_149, weight=_holder__attr_206, bias=_holder__attr_207
        )
        layer_norm_149 = _holder__attr_206 = _holder__attr_207 = None
        _holder__attr_208 = self._attr_208
        _holder__attr_209 = self._attr_209
        linear_223 = torch.nn.functional.linear(
            input=layer_norm_150, weight=_holder__attr_208, bias=_holder__attr_209
        )
        layer_norm_150 = _holder__attr_208 = _holder__attr_209 = None
        _holder__attr_210 = self._attr_210
        _holder__attr_211 = self._attr_211
        linear_224 = torch.nn.functional.linear(
            input=layer_norm_151, weight=_holder__attr_210, bias=_holder__attr_211
        )
        layer_norm_151 = _holder__attr_210 = _holder__attr_211 = None
        _holder__attr_212 = self._attr_212
        _holder__attr_213 = self._attr_213
        linear_225 = torch.nn.functional.linear(
            input=layer_norm_152, weight=_holder__attr_212, bias=_holder__attr_213
        )
        layer_norm_152 = _holder__attr_212 = _holder__attr_213 = None
        mul_69 = torch.mul(input=linear_175, other=sigmoid_69)
        linear_175 = sigmoid_69 = None
        gelu_6 = torch.nn.functional.gelu(input=linear_220, approximate="none")
        linear_220 = None
        gelu_7 = torch.nn.functional.gelu(input=linear_221, approximate="none")
        linear_221 = None
        gelu_8 = torch.nn.functional.gelu(input=linear_222, approximate="none")
        linear_222 = None
        gelu_9 = torch.nn.functional.gelu(input=linear_223, approximate="none")
        linear_223 = None
        gelu_10 = torch.nn.functional.gelu(input=linear_224, approximate="none")
        linear_224 = None
        gelu_11 = torch.nn.functional.gelu(input=linear_225, approximate="none")
        linear_225 = None
        _holder__attr_214 = self._attr_214
        _holder__attr_215 = self._attr_215
        linear_314 = torch.nn.functional.linear(
            input=mul_69, weight=_holder__attr_214, bias=_holder__attr_215
        )
        mul_69 = _holder__attr_214 = _holder__attr_215 = None
        _holder__attr_216 = self._attr_216
        _holder__attr_217 = self._attr_217
        linear_226 = torch.nn.functional.linear(
            input=gelu_6, weight=_holder__attr_216, bias=_holder__attr_217
        )
        gelu_6 = _holder__attr_216 = _holder__attr_217 = None
        _holder__attr_218 = self._attr_218
        _holder__attr_219 = self._attr_219
        linear_227 = torch.nn.functional.linear(
            input=gelu_7, weight=_holder__attr_218, bias=_holder__attr_219
        )
        gelu_7 = _holder__attr_218 = _holder__attr_219 = None
        _holder__attr_220 = self._attr_220
        _holder__attr_221 = self._attr_221
        linear_228 = torch.nn.functional.linear(
            input=gelu_8, weight=_holder__attr_220, bias=_holder__attr_221
        )
        gelu_8 = _holder__attr_220 = _holder__attr_221 = None
        _holder__attr_222 = self._attr_222
        _holder__attr_223 = self._attr_223
        linear_229 = torch.nn.functional.linear(
            input=gelu_9, weight=_holder__attr_222, bias=_holder__attr_223
        )
        gelu_9 = _holder__attr_222 = _holder__attr_223 = None
        _holder__attr_224 = self._attr_224
        _holder__attr_225 = self._attr_225
        linear_230 = torch.nn.functional.linear(
            input=gelu_10, weight=_holder__attr_224, bias=_holder__attr_225
        )
        gelu_10 = _holder__attr_224 = _holder__attr_225 = None
        _holder__attr_226 = self._attr_226
        _holder__attr_227 = self._attr_227
        linear_231 = torch.nn.functional.linear(
            input=gelu_11, weight=_holder__attr_226, bias=_holder__attr_227
        )
        gelu_11 = _holder__attr_226 = _holder__attr_227 = None
        add_54 = torch.add(input=add_48, other=linear_226)
        add_48 = linear_226 = None
        add_55 = torch.add(input=add_49, other=linear_227)
        add_49 = linear_227 = None
        add_56 = torch.add(input=add_50, other=linear_228)
        add_50 = linear_228 = None
        add_57 = torch.add(input=add_51, other=linear_229)
        add_51 = linear_229 = None
        add_58 = torch.add(input=add_52, other=linear_230)
        add_52 = linear_230 = None
        add_59 = torch.add(input=add_53, other=linear_231)
        add_53 = linear_231 = None
        permute = add_54.permute([1, 0, 2])
        add_54 = None
        permute_100 = add_55.permute([1, 0, 2])
        add_55 = None
        permute_101 = add_56.permute([1, 0, 2])
        add_56 = None
        permute_102 = add_57.permute([1, 0, 2])
        add_57 = None
        permute_103 = add_58.permute([1, 0, 2])
        add_58 = None
        permute_104 = add_59.permute([1, 0, 2])
        add_59 = None
        cat_24 = torch.cat(
            tensors=(
                permute_104,
                permute,
                permute_101,
                permute_100,
                permute_102,
                permute_103,
            ),
            dim=0,
        )
        permute_104 = permute = permute_101 = permute_100 = permute_102 = (
            permute_103
        ) = None
        nan_to_num_59 = torch.nan_to_num(
            input=cat_24, nan=0.0, posinf=65504.0, neginf=-65504.0
        )
        cat_24 = None
        clamp_59 = torch.clamp(input=nan_to_num_59, min=-100.1, max=100.1)
        nan_to_num_59 = None
        _holder__attr_228 = self._attr_228
        matmul_17 = torch.matmul(input=clamp_59, other=_holder__attr_228)
        clamp_59 = _holder__attr_228 = None
        _holder__attr_229 = self._attr_229
        add_60 = torch.add(input=_holder__attr_229, other=matmul_17)
        _holder__attr_229 = matmul_17 = None
        permute_105 = add_60.permute([1, 0, 2])
        add_60 = None
        reshape = torch.reshape(
            input=permute_105,
            shape=(-1, 23040),
        )
        permute_105 = None
        cat_25 = torch.cat(
            tensors=(
                getitem_3708,
                getitem_3683,
                getitem_3707,
                getitem_3691,
                getitem_3706,
                getitem_3690,
                getitem_3705,
                getitem_3684,
                getitem_3696,
                getitem_3695,
                getitem_3704,
                getitem_3685,
                getitem_3697,
                getitem_3694,
                getitem_3698,
                getitem_3693,
                getitem_3703,
                getitem_3692,
                getitem_3699,
                getitem_3689,
                getitem_3702,
                getitem_3686,
                getitem_3700,
                getitem_3687,
                getitem_3701,
                getitem_3688,
                linear_314,
                layer_norm_120,
                layer_norm_128,
                layer_norm_124,
                layer_norm_117,
                layer_norm_111,
                layer_norm_109,
                layer_norm_116,
                layer_norm_132,
                layer_norm_126,
                layer_norm_127,
                layer_norm_107,
                layer_norm_113,
                layer_norm_110,
                layer_norm_133,
                layer_norm_118,
                layer_norm_112,
                layer_norm_131,
                layer_norm_122,
                layer_norm_125,
                layer_norm_121,
                layer_norm_119,
                layer_norm_130,
                layer_norm_108,
                layer_norm_129,
                layer_norm_106,
                layer_norm_114,
                layer_norm_134,
                layer_norm_123,
                layer_norm_115,
                reshape,
            ),
            dim=1,
        )
        getitem_3708 = getitem_3683 = getitem_3707 = getitem_3691 = getitem_3706 = (
            getitem_3690
        ) = getitem_3705 = getitem_3684 = getitem_3696 = getitem_3695 = getitem_3704 = (
            getitem_3685
        ) = getitem_3697 = getitem_3694 = getitem_3698 = getitem_3693 = getitem_3703 = (
            getitem_3692
        ) = getitem_3699 = getitem_3689 = getitem_3702 = getitem_3686 = getitem_3700 = (
            getitem_3687
        ) = getitem_3701 = getitem_3688 = linear_314 = layer_norm_120 = (
            layer_norm_128
        ) = layer_norm_124 = layer_norm_117 = layer_norm_111 = layer_norm_109 = (
            layer_norm_116
        ) = layer_norm_132 = layer_norm_126 = layer_norm_127 = layer_norm_107 = (
            layer_norm_113
        ) = layer_norm_110 = layer_norm_133 = layer_norm_118 = layer_norm_112 = (
            layer_norm_131
        ) = layer_norm_122 = layer_norm_125 = layer_norm_121 = layer_norm_119 = (
            layer_norm_130
        ) = layer_norm_108 = layer_norm_129 = layer_norm_106 = layer_norm_114 = (
            layer_norm_134
        ) = layer_norm_123 = layer_norm_115 = reshape = None
        reshape_655 = torch.reshape(
            input=cat_25,
            shape=[-1, 1219, 160],
        )
        cat_25 = None
        permute_74 = reshape_655.permute([0, 2, 1])
        contiguous_21 = permute_74.contiguous()
        _holder__attr_230 = self._attr_230
        _holder__attr_231 = self._attr_231
        linear_315 = torch.nn.functional.linear(
            input=contiguous_21, weight=_holder__attr_230, bias=_holder__attr_231
        )
        contiguous_21 = _holder__attr_230 = _holder__attr_231 = None
        getitem_4149 = linear_315[:, :, 0:232]
        getitem_4150 = linear_315[:, :, 232:296]
        linear_315 = None
        permute_77 = getitem_4149.permute([0, 2, 1])
        getitem_4149 = None
        permute_78 = getitem_4150.permute([0, 2, 1])
        getitem_4150 = None
        split_7 = torch.split(
            permute_77,
            split_size_or_sections=[24, 24, 24, 24, 24, 24, 24, 32, 32],
            dim=1,
        )
        permute_77 = None
        getitem_4087 = split_7[-2]
        getitem_4088 = split_7[-1]
        getitem_4089 = split_7[0]
        getitem_4090 = split_7[1]
        getitem_4091 = split_7[2]
        getitem_4092 = split_7[3]
        getitem_4093 = split_7[4]
        getitem_4094 = split_7[5]
        getitem_4095 = split_7[6]
        split_7 = None
        reshape_656 = torch.reshape(
            input=getitem_4087,
            shape=[-1, 5120],
        )
        getitem_4087 = None
        reshape_657 = torch.reshape(
            input=getitem_4088,
            shape=(-1, 5120),
        )
        getitem_4088 = None
        _holder__attr_232 = self._attr_232
        _holder__attr_233 = self._attr_233
        linear_234 = torch.nn.functional.linear(
            input=reshape_657, weight=_holder__attr_232, bias=_holder__attr_233
        )
        reshape_657 = _holder__attr_232 = _holder__attr_233 = None
        _holder__attr_234 = self._attr_234
        _holder__attr_235 = self._attr_235
        layer_norm_153 = custom_ln(
            input=linear_234,
            normalized_shape=(512,),
            weight=_holder__attr_234,
            bias=_holder__attr_235,
            eps=1e-05,
        )
        _holder__attr_234 = _holder__attr_235 = None
        sigmoid_70 = torch.sigmoid(input=layer_norm_153)
        layer_norm_153 = None
        mul_70 = torch.mul(input=linear_234, other=sigmoid_70)
        linear_234 = sigmoid_70 = None
        _holder__attr_236 = self._attr_236
        _holder__attr_237 = self._attr_237
        linear_235 = torch.nn.functional.linear(
            input=mul_70, weight=_holder__attr_236, bias=_holder__attr_237
        )
        mul_70 = _holder__attr_236 = _holder__attr_237 = None
        reshape_658 = torch.reshape(
            input=linear_235,
            shape=(-1, 1219, 32),
        )
        linear_235 = None
        matmul_18 = torch.matmul(input=permute_74, other=reshape_658)
        permute_74 = reshape_658 = None
        matmul_19 = torch.matmul(input=reshape_655, other=matmul_18)
        reshape_655 = matmul_18 = None
        flatten_8 = torch.flatten(input=matmul_19, start_dim=-2, end_dim=-1)
        matmul_19 = None
        mul_71 = torch.mul(input=sigmoid_60, other=flatten_8)
        sigmoid_60 = flatten_8 = None
        tanh_60 = torch.tanh(input=mul_71)
        mul_71 = None
        _holder__attr_238 = self._attr_238
        _holder__attr_239 = self._attr_239
        linear_236 = torch.nn.functional.linear(
            input=tanh_60, weight=_holder__attr_238, bias=_holder__attr_239
        )
        tanh_60 = _holder__attr_238 = _holder__attr_239 = None
        size_121 = linear_236.size()
        getitem_4096 = size_121[1:]
        size_121 = None
        _holder__attr_240 = self._attr_240
        _holder__attr_241 = self._attr_241
        layer_norm_154 = custom_ln(
            input=linear_236,
            normalized_shape=getitem_4096,
            weight=_holder__attr_240,
            bias=_holder__attr_241,
            eps=1e-05,
        )
        linear_236 = getitem_4096 = _holder__attr_240 = _holder__attr_241 = None
        size_122 = layer_norm_154.size()
        getitem_4097 = size_122[1:]
        size_122 = None
        _holder__attr_242 = self._attr_242
        _holder__attr_243 = self._attr_243
        layer_norm_155 = custom_ln(
            input=layer_norm_154,
            normalized_shape=getitem_4097,
            weight=_holder__attr_242,
            bias=_holder__attr_243,
            eps=1e-05,
        )
        layer_norm_154 = getitem_4097 = _holder__attr_242 = _holder__attr_243 = None
        _holder__attr_244 = self._attr_244
        _holder__attr_245 = self._attr_245
        linear_237 = torch.nn.functional.linear(
            input=layer_norm_155, weight=_holder__attr_244, bias=_holder__attr_245
        )
        _holder__attr_244 = _holder__attr_245 = None
        _holder__attr_246 = self._attr_246
        _holder__attr_247 = self._attr_247
        linear_238 = torch.nn.functional.linear(
            input=linear_237, weight=_holder__attr_246, bias=_holder__attr_247
        )
        linear_237 = _holder__attr_246 = _holder__attr_247 = None
        mul_72 = torch.mul(input=layer_norm_155, other=linear_238)
        linear_238 = None
        add_61 = torch.add(input=layer_norm_155, other=mul_72)
        layer_norm_155 = mul_72 = None
        _holder__attr_248 = self._attr_248
        _holder__attr_249 = self._attr_249
        linear_239 = torch.nn.functional.linear(
            input=add_61, weight=_holder__attr_248, bias=_holder__attr_249
        )
        add_61 = _holder__attr_248 = _holder__attr_249 = None
        _holder__attr_250 = self._attr_250
        _holder__attr_251 = self._attr_251
        layer_norm_156 = custom_ln(
            input=linear_239,
            normalized_shape=(4096,),
            weight=_holder__attr_250,
            bias=_holder__attr_251,
            eps=1e-05,
        )
        _holder__attr_250 = _holder__attr_251 = None
        sigmoid_71 = torch.sigmoid(input=layer_norm_156)
        layer_norm_156 = None
        mul_73 = torch.mul(input=linear_239, other=sigmoid_71)
        linear_239 = sigmoid_71 = None
        _holder__attr_252 = self._attr_252
        _holder__attr_253 = self._attr_253
        linear_240 = torch.nn.functional.linear(
            input=mul_73, weight=_holder__attr_252, bias=_holder__attr_253
        )
        _holder__attr_252 = _holder__attr_253 = None
        _holder__attr_254 = self._attr_254
        _holder__attr_255 = self._attr_255
        layer_norm_157 = custom_ln(
            input=linear_240,
            normalized_shape=(2048,),
            weight=_holder__attr_254,
            bias=_holder__attr_255,
            eps=1e-05,
        )
        _holder__attr_254 = _holder__attr_255 = None
        sigmoid_72 = torch.sigmoid(input=layer_norm_157)
        layer_norm_157 = None
        mul_74 = torch.mul(input=linear_240, other=sigmoid_72)
        linear_240 = sigmoid_72 = None
        _holder__attr_256 = self._attr_256
        _holder__attr_257 = self._attr_257
        linear_241 = torch.nn.functional.linear(
            input=mul_74, weight=_holder__attr_256, bias=_holder__attr_257
        )
        mul_74 = _holder__attr_256 = _holder__attr_257 = None
        add_62 = torch.add(input=mul_73, other=linear_241)
        mul_73 = linear_241 = None
        _holder__attr_258 = self._attr_258
        _holder__attr_259 = self._attr_259
        layer_norm_158 = custom_ln(
            input=add_62,
            normalized_shape=(4096,),
            weight=_holder__attr_258,
            bias=_holder__attr_259,
            eps=1e-05,
        )
        _holder__attr_258 = _holder__attr_259 = None
        sigmoid_73 = torch.sigmoid(input=layer_norm_158)
        layer_norm_158 = None
        mul_75 = torch.mul(input=add_62, other=sigmoid_73)
        add_62 = sigmoid_73 = None
        _holder__attr_260 = self._attr_260
        _holder__attr_261 = self._attr_261
        linear_242 = torch.nn.functional.linear(
            input=mul_75, weight=_holder__attr_260, bias=_holder__attr_261
        )
        mul_75 = _holder__attr_260 = _holder__attr_261 = None
        cat_27 = torch.cat(tensors=[reshape_656, linear_242], dim=1)
        reshape_656 = linear_242 = None
        reshape_659 = torch.reshape(
            input=cat_27,
            shape=[-1, 64, 160],
        )
        cat_27 = None
        add_63 = torch.add(input=permute_78, other=reshape_659)
        permute_78 = reshape_659 = None
        size_123 = add_63.size()
        getitem_4098 = size_123[2:]
        size_123 = None
        _holder__attr_262 = self._attr_262
        _holder__attr_263 = self._attr_263
        layer_norm_159 = custom_ln(
            input=add_63,
            normalized_shape=getitem_4098,
            weight=_holder__attr_262,
            bias=_holder__attr_263,
            eps=1e-05,
        )
        add_63 = getitem_4098 = _holder__attr_262 = _holder__attr_263 = None
        cat_28 = torch.cat(tensors=[layer_norm_159, getitem_4089], dim=1)
        layer_norm_159 = getitem_4089 = None
        ss = ''
        ss += "cat_28 Shape:" + str(cat_28.shape)
        ss += "Stride:" + str(cat_28.stride())
        permute_79 = cat_28.permute([0, 2, 1])
        ss += "permute_79 Shape:" + str(permute_79.shape)
        ss += "Stride:" + str(permute_79.stride())
        contiguous_23 = permute_79.contiguous()
        ss += "contiguous_23 Shape:" + str(contiguous_23.shape)
        ss += "Stride:" + str(contiguous_23.stride())
        print(ss)
        ss = ''
        _holder__attr_264 = self._attr_264
        _holder__attr_265 = self._attr_265
        linear_243 = torch.nn.functional.linear(
            input=contiguous_23, weight=_holder__attr_264, bias=_holder__attr_265
        )
        contiguous_23 = _holder__attr_264 = _holder__attr_265 = None
        permute_81 = linear_243.permute([0, 2, 1])
        linear_243 = None
        split_8 = torch.split(permute_81, split_size_or_sections=[32, 32, 64], dim=1)
        permute_81 = None
        getitem_4099 = split_8[0]
        getitem_4100 = split_8[1]
        getitem_4101 = split_8[2]
        split_8 = None
        reshape_660 = torch.reshape(
            input=getitem_4099,
            shape=[-1, 5120],
        )
        getitem_4099 = None
        reshape_661 = torch.reshape(
            input=getitem_4100,
            shape=(-1, 5120),
        )
        getitem_4100 = None
        _holder__attr_266 = self._attr_266
        _holder__attr_267 = self._attr_267
        linear_244 = torch.nn.functional.linear(
            input=reshape_661, weight=_holder__attr_266, bias=_holder__attr_267
        )
        reshape_661 = _holder__attr_266 = _holder__attr_267 = None
        _holder__attr_268 = self._attr_268
        _holder__attr_269 = self._attr_269
        layer_norm_160 = custom_ln(
            input=linear_244,
            normalized_shape=(512,),
            weight=_holder__attr_268,
            bias=_holder__attr_269,
            eps=1e-05,
        )
        _holder__attr_268 = _holder__attr_269 = None
        sigmoid_74 = torch.sigmoid(input=layer_norm_160)
        layer_norm_160 = None
        mul_76 = torch.mul(input=linear_244, other=sigmoid_74)
        linear_244 = sigmoid_74 = None
        _holder__attr_270 = self._attr_270
        _holder__attr_271 = self._attr_271
        linear_245 = torch.nn.functional.linear(
            input=mul_76, weight=_holder__attr_270, bias=_holder__attr_271
        )
        mul_76 = _holder__attr_270 = _holder__attr_271 = None
        reshape_662 = torch.reshape(
            input=linear_245,
            shape=(-1, 88, 32),
        )
        linear_245 = None
        matmul_20 = torch.matmul(input=permute_79, other=reshape_662)
        permute_79 = reshape_662 = None
        matmul_21 = torch.matmul(input=cat_28, other=matmul_20)
        cat_28 = matmul_20 = None
        flatten_9 = torch.flatten(input=matmul_21, start_dim=-2, end_dim=-1)
        matmul_21 = None
        mul_77 = torch.mul(input=sigmoid_61, other=flatten_9)
        sigmoid_61 = flatten_9 = None
        tanh_61 = torch.tanh(input=mul_77)
        mul_77 = None
        _holder__attr_272 = self._attr_272
        _holder__attr_273 = self._attr_273
        linear_246 = torch.nn.functional.linear(
            input=tanh_61, weight=_holder__attr_272, bias=_holder__attr_273
        )
        tanh_61 = _holder__attr_272 = _holder__attr_273 = None
        size_124 = linear_246.size()
        getitem_4102 = size_124[1:]
        size_124 = None
        _holder__attr_274 = self._attr_274
        _holder__attr_275 = self._attr_275
        layer_norm_161 = custom_ln(
            input=linear_246,
            normalized_shape=getitem_4102,
            weight=_holder__attr_274,
            bias=_holder__attr_275,
            eps=1e-05,
        )
        linear_246 = getitem_4102 = _holder__attr_274 = _holder__attr_275 = None
        size_125 = layer_norm_161.size()
        getitem_4103 = size_125[1:]
        size_125 = None
        _holder__attr_276 = self._attr_276
        _holder__attr_277 = self._attr_277
        layer_norm_162 = custom_ln(
            input=layer_norm_161,
            normalized_shape=getitem_4103,
            weight=_holder__attr_276,
            bias=_holder__attr_277,
            eps=1e-05,
        )
        layer_norm_161 = getitem_4103 = _holder__attr_276 = _holder__attr_277 = None
        _holder__attr_278 = self._attr_278
        _holder__attr_279 = self._attr_279
        linear_247 = torch.nn.functional.linear(
            input=layer_norm_162, weight=_holder__attr_278, bias=_holder__attr_279
        )
        _holder__attr_278 = _holder__attr_279 = None
        _holder__attr_280 = self._attr_280
        _holder__attr_281 = self._attr_281
        linear_248 = torch.nn.functional.linear(
            input=linear_247, weight=_holder__attr_280, bias=_holder__attr_281
        )
        linear_247 = _holder__attr_280 = _holder__attr_281 = None
        mul_78 = torch.mul(input=layer_norm_162, other=linear_248)
        linear_248 = None
        add_64 = torch.add(input=layer_norm_162, other=mul_78)
        layer_norm_162 = mul_78 = None
        _holder__attr_282 = self._attr_282
        _holder__attr_283 = self._attr_283
        linear_249 = torch.nn.functional.linear(
            input=add_64, weight=_holder__attr_282, bias=_holder__attr_283
        )
        add_64 = _holder__attr_282 = _holder__attr_283 = None
        _holder__attr_284 = self._attr_284
        _holder__attr_285 = self._attr_285
        layer_norm_163 = custom_ln(
            input=linear_249,
            normalized_shape=(4096,),
            weight=_holder__attr_284,
            bias=_holder__attr_285,
            eps=1e-05,
        )
        _holder__attr_284 = _holder__attr_285 = None
        sigmoid_75 = torch.sigmoid(input=layer_norm_163)
        layer_norm_163 = None
        mul_79 = torch.mul(input=linear_249, other=sigmoid_75)
        linear_249 = sigmoid_75 = None
        _holder__attr_286 = self._attr_286
        _holder__attr_287 = self._attr_287
        linear_250 = torch.nn.functional.linear(
            input=mul_79, weight=_holder__attr_286, bias=_holder__attr_287
        )
        _holder__attr_286 = _holder__attr_287 = None
        _holder__attr_288 = self._attr_288
        _holder__attr_289 = self._attr_289
        layer_norm_164 = custom_ln(
            input=linear_250,
            normalized_shape=(2048,),
            weight=_holder__attr_288,
            bias=_holder__attr_289,
            eps=1e-05,
        )
        _holder__attr_288 = _holder__attr_289 = None
        sigmoid_76 = torch.sigmoid(input=layer_norm_164)
        layer_norm_164 = None
        mul_80 = torch.mul(input=linear_250, other=sigmoid_76)
        linear_250 = sigmoid_76 = None
        _holder__attr_290 = self._attr_290
        _holder__attr_291 = self._attr_291
        linear_251 = torch.nn.functional.linear(
            input=mul_80, weight=_holder__attr_290, bias=_holder__attr_291
        )
        mul_80 = _holder__attr_290 = _holder__attr_291 = None
        add_65 = torch.add(input=mul_79, other=linear_251)
        mul_79 = linear_251 = None
        _holder__attr_292 = self._attr_292
        _holder__attr_293 = self._attr_293
        layer_norm_165 = custom_ln(
            input=add_65,
            normalized_shape=(4096,),
            weight=_holder__attr_292,
            bias=_holder__attr_293,
            eps=1e-05,
        )
        _holder__attr_292 = _holder__attr_293 = None
        sigmoid_77 = torch.sigmoid(input=layer_norm_165)
        layer_norm_165 = None
        mul_81 = torch.mul(input=add_65, other=sigmoid_77)
        add_65 = sigmoid_77 = None
        _holder__attr_294 = self._attr_294
        _holder__attr_295 = self._attr_295
        linear_252 = torch.nn.functional.linear(
            input=mul_81, weight=_holder__attr_294, bias=_holder__attr_295
        )
        mul_81 = _holder__attr_294 = _holder__attr_295 = None
        cat_30 = torch.cat(tensors=[reshape_660, linear_252], dim=1)
        reshape_660 = linear_252 = None
        reshape_663 = torch.reshape(
            input=cat_30,
            shape=[-1, 64, 160],
        )
        cat_30 = None
        add_66 = torch.add(input=getitem_4101, other=reshape_663)
        getitem_4101 = reshape_663 = None
        size_126 = add_66.size()
        getitem_4104 = size_126[2:]
        size_126 = None
        _holder__attr_296 = self._attr_296
        _holder__attr_297 = self._attr_297
        layer_norm_166 = custom_ln(
            input=add_66,
            normalized_shape=getitem_4104,
            weight=_holder__attr_296,
            bias=_holder__attr_297,
            eps=1e-05,
        )
        add_66 = getitem_4104 = _holder__attr_296 = _holder__attr_297 = None
        cat_31 = torch.cat(tensors=[layer_norm_166, getitem_4090], dim=1)
        layer_norm_166 = getitem_4090 = None
        permute_82 = cat_31.permute([0, 2, 1])
        contiguous_24 = permute_82.contiguous()
        _holder__attr_298 = self._attr_298
        _holder__attr_299 = self._attr_299
        linear_253 = torch.nn.functional.linear(
            input=contiguous_24, weight=_holder__attr_298, bias=_holder__attr_299
        )
        contiguous_24 = _holder__attr_298 = _holder__attr_299 = None
        permute_84 = linear_253.permute([0, 2, 1])
        linear_253 = None
        split_9 = torch.split(permute_84, split_size_or_sections=[32, 32, 64], dim=1)
        permute_84 = None
        getitem_4105 = split_9[0]
        getitem_4106 = split_9[1]
        getitem_4107 = split_9[2]
        split_9 = None
        reshape_664 = torch.reshape(
            input=getitem_4105,
            shape=[-1, 5120],
        )
        getitem_4105 = None
        reshape_665 = torch.reshape(
            input=getitem_4106,
            shape=(-1, 5120),
        )
        getitem_4106 = None
        _holder__attr_300 = self._attr_300
        _holder__attr_301 = self._attr_301
        linear_254 = torch.nn.functional.linear(
            input=reshape_665, weight=_holder__attr_300, bias=_holder__attr_301
        )
        reshape_665 = _holder__attr_300 = _holder__attr_301 = None
        _holder__attr_302 = self._attr_302
        _holder__attr_303 = self._attr_303
        layer_norm_167 = custom_ln(
            input=linear_254,
            normalized_shape=(512,),
            weight=_holder__attr_302,
            bias=_holder__attr_303,
            eps=1e-05,
        )
        _holder__attr_302 = _holder__attr_303 = None
        sigmoid_78 = torch.sigmoid(input=layer_norm_167)
        layer_norm_167 = None
        mul_82 = torch.mul(input=linear_254, other=sigmoid_78)
        linear_254 = sigmoid_78 = None
        _holder__attr_304 = self._attr_304
        _holder__attr_305 = self._attr_305
        linear_255 = torch.nn.functional.linear(
            input=mul_82, weight=_holder__attr_304, bias=_holder__attr_305
        )
        mul_82 = _holder__attr_304 = _holder__attr_305 = None
        reshape_666 = torch.reshape(
            input=linear_255,
            shape=(-1, 88, 32),
        )
        linear_255 = None
        matmul_22 = torch.matmul(input=permute_82, other=reshape_666)
        permute_82 = reshape_666 = None
        matmul_23 = torch.matmul(input=cat_31, other=matmul_22)
        cat_31 = matmul_22 = None
        flatten_10 = torch.flatten(input=matmul_23, start_dim=-2, end_dim=-1)
        matmul_23 = None
        mul_83 = torch.mul(input=sigmoid_62, other=flatten_10)
        sigmoid_62 = flatten_10 = None
        tanh_62 = torch.tanh(input=mul_83)
        mul_83 = None
        _holder__attr_306 = self._attr_306
        _holder__attr_307 = self._attr_307
        linear_256 = torch.nn.functional.linear(
            input=tanh_62, weight=_holder__attr_306, bias=_holder__attr_307
        )
        tanh_62 = _holder__attr_306 = _holder__attr_307 = None
        size_127 = linear_256.size()
        getitem_4108 = size_127[1:]
        size_127 = None
        _holder__attr_308 = self._attr_308
        _holder__attr_309 = self._attr_309
        layer_norm_168 = custom_ln(
            input=linear_256,
            normalized_shape=getitem_4108,
            weight=_holder__attr_308,
            bias=_holder__attr_309,
            eps=1e-05,
        )
        linear_256 = getitem_4108 = _holder__attr_308 = _holder__attr_309 = None
        size_128 = layer_norm_168.size()
        getitem_4109 = size_128[1:]
        size_128 = None
        _holder__attr_310 = self._attr_310
        _holder__attr_311 = self._attr_311
        layer_norm_169 = custom_ln(
            input=layer_norm_168,
            normalized_shape=getitem_4109,
            weight=_holder__attr_310,
            bias=_holder__attr_311,
            eps=1e-05,
        )
        layer_norm_168 = getitem_4109 = _holder__attr_310 = _holder__attr_311 = None
        _holder__attr_312 = self._attr_312
        _holder__attr_313 = self._attr_313
        linear_257 = torch.nn.functional.linear(
            input=layer_norm_169, weight=_holder__attr_312, bias=_holder__attr_313
        )
        _holder__attr_312 = _holder__attr_313 = None
        _holder__attr_314 = self._attr_314
        _holder__attr_315 = self._attr_315
        linear_258 = torch.nn.functional.linear(
            input=linear_257, weight=_holder__attr_314, bias=_holder__attr_315
        )
        linear_257 = _holder__attr_314 = _holder__attr_315 = None
        mul_84 = torch.mul(input=layer_norm_169, other=linear_258)
        linear_258 = None
        add_67 = torch.add(input=layer_norm_169, other=mul_84)
        layer_norm_169 = mul_84 = None
        _holder__attr_316 = self._attr_316
        _holder__attr_317 = self._attr_317
        linear_259 = torch.nn.functional.linear(
            input=add_67, weight=_holder__attr_316, bias=_holder__attr_317
        )
        add_67 = _holder__attr_316 = _holder__attr_317 = None
        _holder__attr_318 = self._attr_318
        _holder__attr_319 = self._attr_319
        layer_norm_170 = custom_ln(
            input=linear_259,
            normalized_shape=(4096,),
            weight=_holder__attr_318,
            bias=_holder__attr_319,
            eps=1e-05,
        )
        _holder__attr_318 = _holder__attr_319 = None
        sigmoid_79 = torch.sigmoid(input=layer_norm_170)
        layer_norm_170 = None
        mul_85 = torch.mul(input=linear_259, other=sigmoid_79)
        linear_259 = sigmoid_79 = None
        _holder__attr_320 = self._attr_320
        _holder__attr_321 = self._attr_321
        linear_260 = torch.nn.functional.linear(
            input=mul_85, weight=_holder__attr_320, bias=_holder__attr_321
        )
        _holder__attr_320 = _holder__attr_321 = None
        _holder__attr_322 = self._attr_322
        _holder__attr_323 = self._attr_323
        layer_norm_171 = custom_ln(
            input=linear_260,
            normalized_shape=(2048,),
            weight=_holder__attr_322,
            bias=_holder__attr_323,
            eps=1e-05,
        )
        _holder__attr_322 = _holder__attr_323 = None
        sigmoid_80 = torch.sigmoid(input=layer_norm_171)
        layer_norm_171 = None
        mul_86 = torch.mul(input=linear_260, other=sigmoid_80)
        linear_260 = sigmoid_80 = None
        _holder__attr_324 = self._attr_324
        _holder__attr_325 = self._attr_325
        linear_261 = torch.nn.functional.linear(
            input=mul_86, weight=_holder__attr_324, bias=_holder__attr_325
        )
        mul_86 = _holder__attr_324 = _holder__attr_325 = None
        add_68 = torch.add(input=mul_85, other=linear_261)
        mul_85 = linear_261 = None
        _holder__attr_326 = self._attr_326
        _holder__attr_327 = self._attr_327
        layer_norm_172 = custom_ln(
            input=add_68,
            normalized_shape=(4096,),
            weight=_holder__attr_326,
            bias=_holder__attr_327,
            eps=1e-05,
        )
        _holder__attr_326 = _holder__attr_327 = None
        sigmoid_81 = torch.sigmoid(input=layer_norm_172)
        layer_norm_172 = None
        mul_87 = torch.mul(input=add_68, other=sigmoid_81)
        add_68 = sigmoid_81 = None
        _holder__attr_328 = self._attr_328
        _holder__attr_329 = self._attr_329
        linear_262 = torch.nn.functional.linear(
            input=mul_87, weight=_holder__attr_328, bias=_holder__attr_329
        )
        mul_87 = _holder__attr_328 = _holder__attr_329 = None
        cat_33 = torch.cat(tensors=[reshape_664, linear_262], dim=1)
        reshape_664 = linear_262 = None
        reshape_667 = torch.reshape(
            input=cat_33,
            shape=[-1, 64, 160],
        )
        cat_33 = None
        add_69 = torch.add(input=getitem_4107, other=reshape_667)
        getitem_4107 = reshape_667 = None
        size_129 = add_69.size()
        getitem_4110 = size_129[2:]
        size_129 = None
        _holder__attr_330 = self._attr_330
        _holder__attr_331 = self._attr_331
        layer_norm_173 = custom_ln(
            input=add_69,
            normalized_shape=getitem_4110,
            weight=_holder__attr_330,
            bias=_holder__attr_331,
            eps=1e-05,
        )
        add_69 = getitem_4110 = _holder__attr_330 = _holder__attr_331 = None
        cat_34 = torch.cat(tensors=[layer_norm_173, getitem_4091], dim=1)
        layer_norm_173 = getitem_4091 = None
        permute_85 = cat_34.permute([0, 2, 1])
        contiguous_25 = permute_85.contiguous()
        _holder__attr_332 = self._attr_332
        _holder__attr_333 = self._attr_333
        linear_263 = torch.nn.functional.linear(
            input=contiguous_25, weight=_holder__attr_332, bias=_holder__attr_333
        )
        contiguous_25 = _holder__attr_332 = _holder__attr_333 = None
        permute_87 = linear_263.permute([0, 2, 1])
        linear_263 = None
        split_10 = torch.split(permute_87, split_size_or_sections=[32, 32, 64], dim=1)
        permute_87 = None
        getitem_4111 = split_10[0]
        getitem_4112 = split_10[1]
        getitem_4113 = split_10[2]
        split_10 = None
        reshape_668 = torch.reshape(
            input=getitem_4111,
            shape=[-1, 5120],
        )
        getitem_4111 = None
        reshape_669 = torch.reshape(
            input=getitem_4112,
            shape=(-1, 5120),
        )
        getitem_4112 = None
        _holder__attr_334 = self._attr_334
        _holder__attr_335 = self._attr_335
        linear_264 = torch.nn.functional.linear(
            input=reshape_669, weight=_holder__attr_334, bias=_holder__attr_335
        )
        reshape_669 = _holder__attr_334 = _holder__attr_335 = None
        _holder__attr_336 = self._attr_336
        _holder__attr_337 = self._attr_337
        layer_norm_174 = custom_ln(
            input=linear_264,
            normalized_shape=(512,),
            weight=_holder__attr_336,
            bias=_holder__attr_337,
            eps=1e-05,
        )
        _holder__attr_336 = _holder__attr_337 = None
        sigmoid_82 = torch.sigmoid(input=layer_norm_174)
        layer_norm_174 = None
        mul_88 = torch.mul(input=linear_264, other=sigmoid_82)
        linear_264 = sigmoid_82 = None
        _holder__attr_338 = self._attr_338
        _holder__attr_339 = self._attr_339
        linear_265 = torch.nn.functional.linear(
            input=mul_88, weight=_holder__attr_338, bias=_holder__attr_339
        )
        mul_88 = _holder__attr_338 = _holder__attr_339 = None
        reshape_670 = torch.reshape(
            input=linear_265,
            shape=(-1, 88, 32),
        )
        linear_265 = None
        matmul_24 = torch.matmul(input=permute_85, other=reshape_670)
        permute_85 = reshape_670 = None
        matmul_25 = torch.matmul(input=cat_34, other=matmul_24)
        cat_34 = matmul_24 = None
        flatten_11 = torch.flatten(input=matmul_25, start_dim=-2, end_dim=-1)
        matmul_25 = None
        mul_89 = torch.mul(input=sigmoid_63, other=flatten_11)
        sigmoid_63 = flatten_11 = None
        tanh_63 = torch.tanh(input=mul_89)
        mul_89 = None
        _holder__attr_340 = self._attr_340
        _holder__attr_341 = self._attr_341
        linear_266 = torch.nn.functional.linear(
            input=tanh_63, weight=_holder__attr_340, bias=_holder__attr_341
        )
        tanh_63 = _holder__attr_340 = _holder__attr_341 = None
        size_130 = linear_266.size()
        getitem_4114 = size_130[1:]
        size_130 = None
        _holder__attr_342 = self._attr_342
        _holder__attr_343 = self._attr_343
        layer_norm_175 = custom_ln(
            input=linear_266,
            normalized_shape=getitem_4114,
            weight=_holder__attr_342,
            bias=_holder__attr_343,
            eps=1e-05,
        )
        linear_266 = getitem_4114 = _holder__attr_342 = _holder__attr_343 = None
        size_131 = layer_norm_175.size()
        getitem_4115 = size_131[1:]
        size_131 = None
        _holder__attr_344 = self._attr_344
        _holder__attr_345 = self._attr_345
        layer_norm_176 = custom_ln(
            input=layer_norm_175,
            normalized_shape=getitem_4115,
            weight=_holder__attr_344,
            bias=_holder__attr_345,
            eps=1e-05,
        )
        layer_norm_175 = getitem_4115 = _holder__attr_344 = _holder__attr_345 = None
        _holder__attr_346 = self._attr_346
        _holder__attr_347 = self._attr_347
        linear_267 = torch.nn.functional.linear(
            input=layer_norm_176, weight=_holder__attr_346, bias=_holder__attr_347
        )
        _holder__attr_346 = _holder__attr_347 = None
        _holder__attr_348 = self._attr_348
        _holder__attr_349 = self._attr_349
        linear_268 = torch.nn.functional.linear(
            input=linear_267, weight=_holder__attr_348, bias=_holder__attr_349
        )
        linear_267 = _holder__attr_348 = _holder__attr_349 = None
        mul_90 = torch.mul(input=layer_norm_176, other=linear_268)
        linear_268 = None
        add_70 = torch.add(input=layer_norm_176, other=mul_90)
        layer_norm_176 = mul_90 = None
        _holder__attr_350 = self._attr_350
        _holder__attr_351 = self._attr_351
        linear_269 = torch.nn.functional.linear(
            input=add_70, weight=_holder__attr_350, bias=_holder__attr_351
        )
        add_70 = _holder__attr_350 = _holder__attr_351 = None
        _holder__attr_352 = self._attr_352
        _holder__attr_353 = self._attr_353
        layer_norm_177 = custom_ln(
            input=linear_269,
            normalized_shape=(4096,),
            weight=_holder__attr_352,
            bias=_holder__attr_353,
            eps=1e-05,
        )
        _holder__attr_352 = _holder__attr_353 = None
        sigmoid_83 = torch.sigmoid(input=layer_norm_177)
        layer_norm_177 = None
        mul_91 = torch.mul(input=linear_269, other=sigmoid_83)
        linear_269 = sigmoid_83 = None
        _holder__attr_354 = self._attr_354
        _holder__attr_355 = self._attr_355
        linear_270 = torch.nn.functional.linear(
            input=mul_91, weight=_holder__attr_354, bias=_holder__attr_355
        )
        _holder__attr_354 = _holder__attr_355 = None
        _holder__attr_356 = self._attr_356
        _holder__attr_357 = self._attr_357
        layer_norm_178 = custom_ln(
            input=linear_270,
            normalized_shape=(2048,),
            weight=_holder__attr_356,
            bias=_holder__attr_357,
            eps=1e-05,
        )
        _holder__attr_356 = _holder__attr_357 = None
        sigmoid_84 = torch.sigmoid(input=layer_norm_178)
        layer_norm_178 = None
        mul_92 = torch.mul(input=linear_270, other=sigmoid_84)
        linear_270 = sigmoid_84 = None
        _holder__attr_358 = self._attr_358
        _holder__attr_359 = self._attr_359
        linear_271 = torch.nn.functional.linear(
            input=mul_92, weight=_holder__attr_358, bias=_holder__attr_359
        )
        mul_92 = _holder__attr_358 = _holder__attr_359 = None
        add_71 = torch.add(input=mul_91, other=linear_271)
        mul_91 = linear_271 = None
        _holder__attr_360 = self._attr_360
        _holder__attr_361 = self._attr_361
        layer_norm_179 = custom_ln(
            input=add_71,
            normalized_shape=(4096,),
            weight=_holder__attr_360,
            bias=_holder__attr_361,
            eps=1e-05,
        )
        _holder__attr_360 = _holder__attr_361 = None
        sigmoid_85 = torch.sigmoid(input=layer_norm_179)
        layer_norm_179 = None
        mul_93 = torch.mul(input=add_71, other=sigmoid_85)
        add_71 = sigmoid_85 = None
        _holder__attr_362 = self._attr_362
        _holder__attr_363 = self._attr_363
        linear_272 = torch.nn.functional.linear(
            input=mul_93, weight=_holder__attr_362, bias=_holder__attr_363
        )
        mul_93 = _holder__attr_362 = _holder__attr_363 = None
        cat_36 = torch.cat(tensors=[reshape_668, linear_272], dim=1)
        reshape_668 = linear_272 = None
        reshape_671 = torch.reshape(
            input=cat_36,
            shape=[-1, 64, 160],
        )
        cat_36 = None
        add_72 = torch.add(input=getitem_4113, other=reshape_671)
        getitem_4113 = reshape_671 = None
        size_132 = add_72.size()
        getitem_4116 = size_132[2:]
        size_132 = None
        _holder__attr_364 = self._attr_364
        _holder__attr_365 = self._attr_365
        layer_norm_180 = custom_ln(
            input=add_72,
            normalized_shape=getitem_4116,
            weight=_holder__attr_364,
            bias=_holder__attr_365,
            eps=1e-05,
        )
        add_72 = getitem_4116 = _holder__attr_364 = _holder__attr_365 = None
        cat_37 = torch.cat(tensors=[layer_norm_180, getitem_4092], dim=1)
        layer_norm_180 = getitem_4092 = None
        permute_88 = cat_37.permute([0, 2, 1])
        contiguous_26 = permute_88.contiguous()
        _holder__attr_366 = self._attr_366
        _holder__attr_367 = self._attr_367
        linear_273 = torch.nn.functional.linear(
            input=contiguous_26, weight=_holder__attr_366, bias=_holder__attr_367
        )
        contiguous_26 = _holder__attr_366 = _holder__attr_367 = None
        permute_90 = linear_273.permute([0, 2, 1])
        linear_273 = None
        split_11 = torch.split(permute_90, split_size_or_sections=[32, 32, 64], dim=1)
        permute_90 = None
        getitem_4117 = split_11[0]
        getitem_4118 = split_11[1]
        getitem_4119 = split_11[2]
        split_11 = None
        reshape_672 = torch.reshape(
            input=getitem_4117,
            shape=[-1, 5120],
        )
        getitem_4117 = None
        reshape_673 = torch.reshape(
            input=getitem_4118,
            shape=(-1, 5120),
        )
        getitem_4118 = None
        _holder__attr_368 = self._attr_368
        _holder__attr_369 = self._attr_369
        linear_274 = torch.nn.functional.linear(
            input=reshape_673, weight=_holder__attr_368, bias=_holder__attr_369
        )
        reshape_673 = _holder__attr_368 = _holder__attr_369 = None
        _holder__attr_370 = self._attr_370
        _holder__attr_371 = self._attr_371
        layer_norm_181 = custom_ln(
            input=linear_274,
            normalized_shape=(512,),
            weight=_holder__attr_370,
            bias=_holder__attr_371,
            eps=1e-05,
        )
        _holder__attr_370 = _holder__attr_371 = None
        sigmoid_86 = torch.sigmoid(input=layer_norm_181)
        layer_norm_181 = None
        mul_94 = torch.mul(input=linear_274, other=sigmoid_86)
        linear_274 = sigmoid_86 = None
        _holder__attr_372 = self._attr_372
        _holder__attr_373 = self._attr_373
        linear_275 = torch.nn.functional.linear(
            input=mul_94, weight=_holder__attr_372, bias=_holder__attr_373
        )
        mul_94 = _holder__attr_372 = _holder__attr_373 = None
        reshape_674 = torch.reshape(
            input=linear_275,
            shape=(-1, 88, 32),
        )
        linear_275 = None
        matmul_26 = torch.matmul(input=permute_88, other=reshape_674)
        permute_88 = reshape_674 = None
        matmul_27 = torch.matmul(input=cat_37, other=matmul_26)
        cat_37 = matmul_26 = None
        flatten_12 = torch.flatten(input=matmul_27, start_dim=-2, end_dim=-1)
        matmul_27 = None
        mul_95 = torch.mul(input=sigmoid_64, other=flatten_12)
        sigmoid_64 = flatten_12 = None
        tanh_64 = torch.tanh(input=mul_95)
        mul_95 = None
        _holder__attr_374 = self._attr_374
        _holder__attr_375 = self._attr_375
        linear_276 = torch.nn.functional.linear(
            input=tanh_64, weight=_holder__attr_374, bias=_holder__attr_375
        )
        tanh_64 = _holder__attr_374 = _holder__attr_375 = None
        size_133 = linear_276.size()
        getitem_4120 = size_133[1:]
        size_133 = None
        _holder__attr_376 = self._attr_376
        _holder__attr_377 = self._attr_377
        layer_norm_182 = custom_ln(
            input=linear_276,
            normalized_shape=getitem_4120,
            weight=_holder__attr_376,
            bias=_holder__attr_377,
            eps=1e-05,
        )
        linear_276 = getitem_4120 = _holder__attr_376 = _holder__attr_377 = None
        size_134 = layer_norm_182.size()
        getitem_4121 = size_134[1:]
        size_134 = None
        _holder__attr_378 = self._attr_378
        _holder__attr_379 = self._attr_379
        layer_norm_183 = custom_ln(
            input=layer_norm_182,
            normalized_shape=getitem_4121,
            weight=_holder__attr_378,
            bias=_holder__attr_379,
            eps=1e-05,
        )
        layer_norm_182 = getitem_4121 = _holder__attr_378 = _holder__attr_379 = None
        _holder__attr_380 = self._attr_380
        _holder__attr_381 = self._attr_381
        linear_277 = torch.nn.functional.linear(
            input=layer_norm_183, weight=_holder__attr_380, bias=_holder__attr_381
        )
        _holder__attr_380 = _holder__attr_381 = None
        _holder__attr_382 = self._attr_382
        _holder__attr_383 = self._attr_383
        linear_278 = torch.nn.functional.linear(
            input=linear_277, weight=_holder__attr_382, bias=_holder__attr_383
        )
        linear_277 = _holder__attr_382 = _holder__attr_383 = None
        mul_96 = torch.mul(input=layer_norm_183, other=linear_278)
        linear_278 = None
        add_73 = torch.add(input=layer_norm_183, other=mul_96)
        layer_norm_183 = mul_96 = None
        _holder__attr_384 = self._attr_384
        _holder__attr_385 = self._attr_385
        linear_279 = torch.nn.functional.linear(
            input=add_73, weight=_holder__attr_384, bias=_holder__attr_385
        )
        add_73 = _holder__attr_384 = _holder__attr_385 = None
        _holder__attr_386 = self._attr_386
        _holder__attr_387 = self._attr_387
        layer_norm_184 = custom_ln(
            input=linear_279,
            normalized_shape=(4096,),
            weight=_holder__attr_386,
            bias=_holder__attr_387,
            eps=1e-05,
        )
        _holder__attr_386 = _holder__attr_387 = None
        sigmoid_87 = torch.sigmoid(input=layer_norm_184)
        layer_norm_184 = None
        mul_97 = torch.mul(input=linear_279, other=sigmoid_87)
        linear_279 = sigmoid_87 = None
        _holder__attr_388 = self._attr_388
        _holder__attr_389 = self._attr_389
        linear_280 = torch.nn.functional.linear(
            input=mul_97, weight=_holder__attr_388, bias=_holder__attr_389
        )
        _holder__attr_388 = _holder__attr_389 = None
        _holder__attr_390 = self._attr_390
        _holder__attr_391 = self._attr_391
        layer_norm_185 = custom_ln(
            input=linear_280,
            normalized_shape=(2048,),
            weight=_holder__attr_390,
            bias=_holder__attr_391,
            eps=1e-05,
        )
        _holder__attr_390 = _holder__attr_391 = None
        sigmoid_88 = torch.sigmoid(input=layer_norm_185)
        layer_norm_185 = None
        mul_98 = torch.mul(input=linear_280, other=sigmoid_88)
        linear_280 = sigmoid_88 = None
        _holder__attr_392 = self._attr_392
        _holder__attr_393 = self._attr_393
        linear_281 = torch.nn.functional.linear(
            input=mul_98, weight=_holder__attr_392, bias=_holder__attr_393
        )
        mul_98 = _holder__attr_392 = _holder__attr_393 = None
        add_74 = torch.add(input=mul_97, other=linear_281)
        mul_97 = linear_281 = None
        _holder__attr_394 = self._attr_394
        _holder__attr_395 = self._attr_395
        layer_norm_186 = custom_ln(
            input=add_74,
            normalized_shape=(4096,),
            weight=_holder__attr_394,
            bias=_holder__attr_395,
            eps=1e-05,
        )
        _holder__attr_394 = _holder__attr_395 = None
        sigmoid_89 = torch.sigmoid(input=layer_norm_186)
        layer_norm_186 = None
        mul_99 = torch.mul(input=add_74, other=sigmoid_89)
        add_74 = sigmoid_89 = None
        _holder__attr_396 = self._attr_396
        _holder__attr_397 = self._attr_397
        linear_282 = torch.nn.functional.linear(
            input=mul_99, weight=_holder__attr_396, bias=_holder__attr_397
        )
        mul_99 = _holder__attr_396 = _holder__attr_397 = None
        cat_39 = torch.cat(tensors=[reshape_672, linear_282], dim=1)
        reshape_672 = linear_282 = None
        reshape_675 = torch.reshape(
            input=cat_39,
            shape=[-1, 64, 160],
        )
        cat_39 = None
        add_75 = torch.add(input=getitem_4119, other=reshape_675)
        getitem_4119 = reshape_675 = None
        size_135 = add_75.size()
        getitem_4122 = size_135[2:]
        size_135 = None
        _holder__attr_398 = self._attr_398
        _holder__attr_399 = self._attr_399
        layer_norm_187 = custom_ln(
            input=add_75,
            normalized_shape=getitem_4122,
            weight=_holder__attr_398,
            bias=_holder__attr_399,
            eps=1e-05,
        )
        add_75 = getitem_4122 = _holder__attr_398 = _holder__attr_399 = None
        cat_40 = torch.cat(tensors=[layer_norm_187, getitem_4093], dim=1)
        layer_norm_187 = getitem_4093 = None
        permute_91 = cat_40.permute([0, 2, 1])
        contiguous_27 = permute_91.contiguous()
        _holder__attr_400 = self._attr_400
        _holder__attr_401 = self._attr_401
        linear_283 = torch.nn.functional.linear(
            input=contiguous_27, weight=_holder__attr_400, bias=_holder__attr_401
        )
        contiguous_27 = _holder__attr_400 = _holder__attr_401 = None
        permute_93 = linear_283.permute([0, 2, 1])
        linear_283 = None
        split_12 = torch.split(permute_93, split_size_or_sections=[32, 32, 64], dim=1)
        permute_93 = None
        getitem_4123 = split_12[0]
        getitem_4124 = split_12[1]
        getitem_4125 = split_12[2]
        split_12 = None
        reshape_676 = torch.reshape(
            input=getitem_4123,
            shape=[-1, 5120],
        )
        getitem_4123 = None
        reshape_677 = torch.reshape(
            input=getitem_4124,
            shape=(-1, 5120),
        )
        getitem_4124 = None
        _holder__attr_402 = self._attr_402
        _holder__attr_403 = self._attr_403
        linear_284 = torch.nn.functional.linear(
            input=reshape_677, weight=_holder__attr_402, bias=_holder__attr_403
        )
        reshape_677 = _holder__attr_402 = _holder__attr_403 = None
        _holder__attr_404 = self._attr_404
        _holder__attr_405 = self._attr_405
        layer_norm_188 = custom_ln(
            input=linear_284,
            normalized_shape=(512,),
            weight=_holder__attr_404,
            bias=_holder__attr_405,
            eps=1e-05,
        )
        _holder__attr_404 = _holder__attr_405 = None
        sigmoid_90 = torch.sigmoid(input=layer_norm_188)
        layer_norm_188 = None
        mul_100 = torch.mul(input=linear_284, other=sigmoid_90)
        linear_284 = sigmoid_90 = None
        _holder__attr_406 = self._attr_406
        _holder__attr_407 = self._attr_407
        linear_285 = torch.nn.functional.linear(
            input=mul_100, weight=_holder__attr_406, bias=_holder__attr_407
        )
        mul_100 = _holder__attr_406 = _holder__attr_407 = None
        reshape_678 = torch.reshape(
            input=linear_285,
            shape=(-1, 88, 32),
        )
        linear_285 = None
        matmul_28 = torch.matmul(input=permute_91, other=reshape_678)
        permute_91 = reshape_678 = None
        matmul_29 = torch.matmul(input=cat_40, other=matmul_28)
        cat_40 = matmul_28 = None
        flatten_13 = torch.flatten(input=matmul_29, start_dim=-2, end_dim=-1)
        matmul_29 = None
        mul_101 = torch.mul(input=sigmoid_65, other=flatten_13)
        sigmoid_65 = flatten_13 = None
        tanh_65 = torch.tanh(input=mul_101)
        mul_101 = None
        _holder__attr_408 = self._attr_408
        _holder__attr_409 = self._attr_409
        linear_286 = torch.nn.functional.linear(
            input=tanh_65, weight=_holder__attr_408, bias=_holder__attr_409
        )
        tanh_65 = _holder__attr_408 = _holder__attr_409 = None
        size_136 = linear_286.size()
        getitem_4126 = size_136[1:]
        size_136 = None
        _holder__attr_410 = self._attr_410
        _holder__attr_411 = self._attr_411
        layer_norm_189 = custom_ln(
            input=linear_286,
            normalized_shape=getitem_4126,
            weight=_holder__attr_410,
            bias=_holder__attr_411,
            eps=1e-05,
        )
        linear_286 = getitem_4126 = _holder__attr_410 = _holder__attr_411 = None
        size_137 = layer_norm_189.size()
        getitem_4127 = size_137[1:]
        size_137 = None
        _holder__attr_412 = self._attr_412
        _holder__attr_413 = self._attr_413
        layer_norm_190 = custom_ln(
            input=layer_norm_189,
            normalized_shape=getitem_4127,
            weight=_holder__attr_412,
            bias=_holder__attr_413,
            eps=1e-05,
        )
        layer_norm_189 = getitem_4127 = _holder__attr_412 = _holder__attr_413 = None
        _holder__attr_414 = self._attr_414
        _holder__attr_415 = self._attr_415
        linear_287 = torch.nn.functional.linear(
            input=layer_norm_190, weight=_holder__attr_414, bias=_holder__attr_415
        )
        _holder__attr_414 = _holder__attr_415 = None
        _holder__attr_416 = self._attr_416
        _holder__attr_417 = self._attr_417
        linear_288 = torch.nn.functional.linear(
            input=linear_287, weight=_holder__attr_416, bias=_holder__attr_417
        )
        linear_287 = _holder__attr_416 = _holder__attr_417 = None
        mul_102 = torch.mul(input=layer_norm_190, other=linear_288)
        linear_288 = None
        add_76 = torch.add(input=layer_norm_190, other=mul_102)
        layer_norm_190 = mul_102 = None
        _holder__attr_418 = self._attr_418
        _holder__attr_419 = self._attr_419
        linear_289 = torch.nn.functional.linear(
            input=add_76, weight=_holder__attr_418, bias=_holder__attr_419
        )
        add_76 = _holder__attr_418 = _holder__attr_419 = None
        _holder__attr_420 = self._attr_420
        _holder__attr_421 = self._attr_421
        layer_norm_191 = custom_ln(
            input=linear_289,
            normalized_shape=(4096,),
            weight=_holder__attr_420,
            bias=_holder__attr_421,
            eps=1e-05,
        )
        _holder__attr_420 = _holder__attr_421 = None
        sigmoid_91 = torch.sigmoid(input=layer_norm_191)
        layer_norm_191 = None
        mul_103 = torch.mul(input=linear_289, other=sigmoid_91)
        linear_289 = sigmoid_91 = None
        _holder__attr_422 = self._attr_422
        _holder__attr_423 = self._attr_423
        linear_290 = torch.nn.functional.linear(
            input=mul_103, weight=_holder__attr_422, bias=_holder__attr_423
        )
        _holder__attr_422 = _holder__attr_423 = None
        _holder__attr_424 = self._attr_424
        _holder__attr_425 = self._attr_425
        layer_norm_192 = custom_ln(
            input=linear_290,
            normalized_shape=(2048,),
            weight=_holder__attr_424,
            bias=_holder__attr_425,
            eps=1e-05,
        )
        _holder__attr_424 = _holder__attr_425 = None
        sigmoid_92 = torch.sigmoid(input=layer_norm_192)
        layer_norm_192 = None
        mul_104 = torch.mul(input=linear_290, other=sigmoid_92)
        linear_290 = sigmoid_92 = None
        _holder__attr_426 = self._attr_426
        _holder__attr_427 = self._attr_427
        linear_291 = torch.nn.functional.linear(
            input=mul_104, weight=_holder__attr_426, bias=_holder__attr_427
        )
        mul_104 = _holder__attr_426 = _holder__attr_427 = None
        add_77 = torch.add(input=mul_103, other=linear_291)
        mul_103 = linear_291 = None
        _holder__attr_428 = self._attr_428
        _holder__attr_429 = self._attr_429
        layer_norm_193 = custom_ln(
            input=add_77,
            normalized_shape=(4096,),
            weight=_holder__attr_428,
            bias=_holder__attr_429,
            eps=1e-05,
        )
        _holder__attr_428 = _holder__attr_429 = None
        sigmoid_93 = torch.sigmoid(input=layer_norm_193)
        layer_norm_193 = None
        mul_105 = torch.mul(input=add_77, other=sigmoid_93)
        add_77 = sigmoid_93 = None
        _holder__attr_430 = self._attr_430
        _holder__attr_431 = self._attr_431
        linear_292 = torch.nn.functional.linear(
            input=mul_105, weight=_holder__attr_430, bias=_holder__attr_431
        )
        mul_105 = _holder__attr_430 = _holder__attr_431 = None
        cat_42 = torch.cat(tensors=[reshape_676, linear_292], dim=1)
        reshape_676 = linear_292 = None
        reshape_679 = torch.reshape(
            input=cat_42,
            shape=[-1, 64, 160],
        )
        cat_42 = None
        add_78 = torch.add(input=getitem_4125, other=reshape_679)
        getitem_4125 = reshape_679 = None
        size_138 = add_78.size()
        getitem_4128 = size_138[2:]
        size_138 = None
        _holder__attr_432 = self._attr_432
        _holder__attr_433 = self._attr_433
        layer_norm_194 = custom_ln(
            input=add_78,
            normalized_shape=getitem_4128,
            weight=_holder__attr_432,
            bias=_holder__attr_433,
            eps=1e-05,
        )
        add_78 = getitem_4128 = _holder__attr_432 = _holder__attr_433 = None
        cat_43 = torch.cat(tensors=[layer_norm_194, getitem_4094], dim=1)
        layer_norm_194 = getitem_4094 = None
        permute_94 = cat_43.permute([0, 2, 1])
        contiguous_28 = permute_94.contiguous()
        _holder__attr_434 = self._attr_434
        _holder__attr_435 = self._attr_435
        linear_293 = torch.nn.functional.linear(
            input=contiguous_28, weight=_holder__attr_434, bias=_holder__attr_435
        )
        contiguous_28 = _holder__attr_434 = _holder__attr_435 = None
        permute_96 = linear_293.permute([0, 2, 1])
        linear_293 = None
        split_13 = torch.split(permute_96, split_size_or_sections=[32, 32, 64], dim=1)
        permute_96 = None
        getitem_4129 = split_13[0]
        getitem_4130 = split_13[1]
        getitem_4131 = split_13[2]
        split_13 = None
        reshape_680 = torch.reshape(
            input=getitem_4129,
            shape=[-1, 5120],
        )
        getitem_4129 = None
        reshape_681 = torch.reshape(
            input=getitem_4130,
            shape=(-1, 5120),
        )
        getitem_4130 = None
        _holder__attr_436 = self._attr_436
        _holder__attr_437 = self._attr_437
        linear_294 = torch.nn.functional.linear(
            input=reshape_681, weight=_holder__attr_436, bias=_holder__attr_437
        )
        reshape_681 = _holder__attr_436 = _holder__attr_437 = None
        _holder__attr_438 = self._attr_438
        _holder__attr_439 = self._attr_439
        layer_norm_195 = custom_ln(
            input=linear_294,
            normalized_shape=(512,),
            weight=_holder__attr_438,
            bias=_holder__attr_439,
            eps=1e-05,
        )
        _holder__attr_438 = _holder__attr_439 = None
        sigmoid_94 = torch.sigmoid(input=layer_norm_195)
        layer_norm_195 = None
        mul_106 = torch.mul(input=linear_294, other=sigmoid_94)
        linear_294 = sigmoid_94 = None
        _holder__attr_440 = self._attr_440
        _holder__attr_441 = self._attr_441
        linear_295 = torch.nn.functional.linear(
            input=mul_106, weight=_holder__attr_440, bias=_holder__attr_441
        )
        mul_106 = _holder__attr_440 = _holder__attr_441 = None
        reshape_682 = torch.reshape(
            input=linear_295,
            shape=(-1, 88, 32),
        )
        linear_295 = None
        matmul_30 = torch.matmul(input=permute_94, other=reshape_682)
        permute_94 = reshape_682 = None
        matmul_31 = torch.matmul(input=cat_43, other=matmul_30)
        cat_43 = matmul_30 = None
        flatten_14 = torch.flatten(input=matmul_31, start_dim=-2, end_dim=-1)
        matmul_31 = None
        mul_107 = torch.mul(input=sigmoid_66, other=flatten_14)
        sigmoid_66 = flatten_14 = None
        tanh_66 = torch.tanh(input=mul_107)
        mul_107 = None
        _holder__attr_442 = self._attr_442
        _holder__attr_443 = self._attr_443
        linear_296 = torch.nn.functional.linear(
            input=tanh_66, weight=_holder__attr_442, bias=_holder__attr_443
        )
        tanh_66 = _holder__attr_442 = _holder__attr_443 = None
        size_139 = linear_296.size()
        getitem_4132 = size_139[1:]
        size_139 = None
        _holder__attr_444 = self._attr_444
        _holder__attr_445 = self._attr_445
        layer_norm_196 = custom_ln(
            input=linear_296,
            normalized_shape=getitem_4132,
            weight=_holder__attr_444,
            bias=_holder__attr_445,
            eps=1e-05,
        )
        linear_296 = getitem_4132 = _holder__attr_444 = _holder__attr_445 = None
        size_140 = layer_norm_196.size()
        getitem_4133 = size_140[1:]
        size_140 = None
        _holder__attr_446 = self._attr_446
        _holder__attr_447 = self._attr_447
        layer_norm_197 = custom_ln(
            input=layer_norm_196,
            normalized_shape=getitem_4133,
            weight=_holder__attr_446,
            bias=_holder__attr_447,
            eps=1e-05,
        )
        layer_norm_196 = getitem_4133 = _holder__attr_446 = _holder__attr_447 = None
        _holder__attr_448 = self._attr_448
        _holder__attr_449 = self._attr_449
        linear_297 = torch.nn.functional.linear(
            input=layer_norm_197, weight=_holder__attr_448, bias=_holder__attr_449
        )
        _holder__attr_448 = _holder__attr_449 = None
        _holder__attr_450 = self._attr_450
        _holder__attr_451 = self._attr_451
        linear_298 = torch.nn.functional.linear(
            input=linear_297, weight=_holder__attr_450, bias=_holder__attr_451
        )
        linear_297 = _holder__attr_450 = _holder__attr_451 = None
        mul_108 = torch.mul(input=layer_norm_197, other=linear_298)
        linear_298 = None
        add_79 = torch.add(input=layer_norm_197, other=mul_108)
        layer_norm_197 = mul_108 = None
        _holder__attr_452 = self._attr_452
        _holder__attr_453 = self._attr_453
        linear_299 = torch.nn.functional.linear(
            input=add_79, weight=_holder__attr_452, bias=_holder__attr_453
        )
        add_79 = _holder__attr_452 = _holder__attr_453 = None
        _holder__attr_454 = self._attr_454
        _holder__attr_455 = self._attr_455
        layer_norm_198 = custom_ln(
            input=linear_299,
            normalized_shape=(4096,),
            weight=_holder__attr_454,
            bias=_holder__attr_455,
            eps=1e-05,
        )
        _holder__attr_454 = _holder__attr_455 = None
        sigmoid_95 = torch.sigmoid(input=layer_norm_198)
        layer_norm_198 = None
        mul_109 = torch.mul(input=linear_299, other=sigmoid_95)
        linear_299 = sigmoid_95 = None
        _holder__attr_456 = self._attr_456
        _holder__attr_457 = self._attr_457
        linear_300 = torch.nn.functional.linear(
            input=mul_109, weight=_holder__attr_456, bias=_holder__attr_457
        )
        _holder__attr_456 = _holder__attr_457 = None
        _holder__attr_458 = self._attr_458
        _holder__attr_459 = self._attr_459
        layer_norm_199 = custom_ln(
            input=linear_300,
            normalized_shape=(2048,),
            weight=_holder__attr_458,
            bias=_holder__attr_459,
            eps=1e-05,
        )
        _holder__attr_458 = _holder__attr_459 = None
        sigmoid_96 = torch.sigmoid(input=layer_norm_199)
        layer_norm_199 = None
        mul_110 = torch.mul(input=linear_300, other=sigmoid_96)
        linear_300 = sigmoid_96 = None
        _holder__attr_460 = self._attr_460
        _holder__attr_461 = self._attr_461
        linear_301 = torch.nn.functional.linear(
            input=mul_110, weight=_holder__attr_460, bias=_holder__attr_461
        )
        mul_110 = _holder__attr_460 = _holder__attr_461 = None
        add_80 = torch.add(input=mul_109, other=linear_301)
        mul_109 = linear_301 = None
        _holder__attr_462 = self._attr_462
        _holder__attr_463 = self._attr_463
        layer_norm_200 = custom_ln(
            input=add_80,
            normalized_shape=(4096,),
            weight=_holder__attr_462,
            bias=_holder__attr_463,
            eps=1e-05,
        )
        _holder__attr_462 = _holder__attr_463 = None
        sigmoid_97 = torch.sigmoid(input=layer_norm_200)
        layer_norm_200 = None
        mul_111 = torch.mul(input=add_80, other=sigmoid_97)
        add_80 = sigmoid_97 = None
        _holder__attr_464 = self._attr_464
        _holder__attr_465 = self._attr_465
        linear_302 = torch.nn.functional.linear(
            input=mul_111, weight=_holder__attr_464, bias=_holder__attr_465
        )
        mul_111 = _holder__attr_464 = _holder__attr_465 = None
        cat_45 = torch.cat(tensors=[reshape_680, linear_302], dim=1)
        reshape_680 = linear_302 = None
        reshape_683 = torch.reshape(
            input=cat_45,
            shape=[-1, 64, 160],
        )
        cat_45 = None
        add_81 = torch.add(input=getitem_4131, other=reshape_683)
        getitem_4131 = reshape_683 = None
        size_141 = add_81.size()
        getitem_4134 = size_141[2:]
        size_141 = None
        _holder__attr_466 = self._attr_466
        _holder__attr_467 = self._attr_467
        layer_norm_201 = custom_ln(
            input=add_81,
            normalized_shape=getitem_4134,
            weight=_holder__attr_466,
            bias=_holder__attr_467,
            eps=1e-05,
        )
        add_81 = getitem_4134 = _holder__attr_466 = _holder__attr_467 = None
        cat_46 = torch.cat(tensors=[layer_norm_201, getitem_4095], dim=1)
        layer_norm_201 = getitem_4095 = None
        permute_97 = cat_46.permute([0, 2, 1])
        contiguous_29 = permute_97.contiguous()
        _holder__attr_468 = self._attr_468
        _holder__attr_469 = self._attr_469
        linear_303 = torch.nn.functional.linear(
            input=contiguous_29, weight=_holder__attr_468, bias=_holder__attr_469
        )
        contiguous_29 = _holder__attr_468 = _holder__attr_469 = None
        permute_99 = linear_303.permute([0, 2, 1])
        linear_303 = None
        reshape_684 = torch.reshape(
            input=permute_99,
            shape=(-1, 5120),
        )
        permute_99 = None
        _holder__attr_470 = self._attr_470
        _holder__attr_471 = self._attr_471
        linear_304 = torch.nn.functional.linear(
            input=reshape_684, weight=_holder__attr_470, bias=_holder__attr_471
        )
        reshape_684 = _holder__attr_470 = _holder__attr_471 = None
        _holder__attr_472 = self._attr_472
        _holder__attr_473 = self._attr_473
        layer_norm_202 = custom_ln(
            input=linear_304,
            normalized_shape=(512,),
            weight=_holder__attr_472,
            bias=_holder__attr_473,
            eps=1e-05,
        )
        _holder__attr_472 = _holder__attr_473 = None
        sigmoid_98 = torch.sigmoid(input=layer_norm_202)
        layer_norm_202 = None
        mul_112 = torch.mul(input=linear_304, other=sigmoid_98)
        linear_304 = sigmoid_98 = None
        _holder__attr_474 = self._attr_474
        _holder__attr_475 = self._attr_475
        linear_305 = torch.nn.functional.linear(
            input=mul_112, weight=_holder__attr_474, bias=_holder__attr_475
        )
        mul_112 = _holder__attr_474 = _holder__attr_475 = None
        reshape_685 = torch.reshape(
            input=linear_305,
            shape=(-1, 88, 32),
        )
        linear_305 = None
        matmul_32 = torch.matmul(input=permute_97, other=reshape_685)
        permute_97 = reshape_685 = None
        matmul_33 = torch.matmul(input=cat_46, other=matmul_32)
        cat_46 = matmul_32 = None
        flatten_15 = torch.flatten(input=matmul_33, start_dim=-2, end_dim=-1)
        matmul_33 = None
        mul_113 = torch.mul(input=sigmoid_67, other=flatten_15)
        sigmoid_67 = flatten_15 = None
        tanh_67 = torch.tanh(input=mul_113)
        mul_113 = None
        _holder__attr_476 = self._attr_476
        _holder__attr_477 = self._attr_477
        linear_306 = torch.nn.functional.linear(
            input=tanh_67, weight=_holder__attr_476, bias=_holder__attr_477
        )
        tanh_67 = _holder__attr_476 = _holder__attr_477 = None
        size_142 = linear_306.size()
        getitem_4135 = size_142[1:]
        size_142 = None
        _holder__attr_478 = self._attr_478
        _holder__attr_479 = self._attr_479
        layer_norm_203 = custom_ln(
            input=linear_306,
            normalized_shape=getitem_4135,
            weight=_holder__attr_478,
            bias=_holder__attr_479,
            eps=1e-05,
        )
        linear_306 = getitem_4135 = _holder__attr_478 = _holder__attr_479 = None
        size_143 = layer_norm_203.size()
        getitem_4136 = size_143[1:]
        size_143 = None
        _holder__attr_480 = self._attr_480
        _holder__attr_481 = self._attr_481
        layer_norm_204 = custom_ln(
            input=layer_norm_203,
            normalized_shape=getitem_4136,
            weight=_holder__attr_480,
            bias=_holder__attr_481,
            eps=1e-05,
        )
        layer_norm_203 = getitem_4136 = _holder__attr_480 = _holder__attr_481 = None
        _holder__attr_482 = self._attr_482
        _holder__attr_483 = self._attr_483
        linear_307 = torch.nn.functional.linear(
            input=layer_norm_204, weight=_holder__attr_482, bias=_holder__attr_483
        )
        _holder__attr_482 = _holder__attr_483 = None
        _holder__attr_484 = self._attr_484
        _holder__attr_485 = self._attr_485
        linear_308 = torch.nn.functional.linear(
            input=linear_307, weight=_holder__attr_484, bias=_holder__attr_485
        )
        linear_307 = _holder__attr_484 = _holder__attr_485 = None
        mul_114 = torch.mul(input=layer_norm_204, other=linear_308)
        linear_308 = None
        add_82 = torch.add(input=layer_norm_204, other=mul_114)
        layer_norm_204 = mul_114 = None
        _holder__attr_486 = self._attr_486
        _holder__attr_487 = self._attr_487
        linear_309 = torch.nn.functional.linear(
            input=add_82, weight=_holder__attr_486, bias=_holder__attr_487
        )
        add_82 = _holder__attr_486 = _holder__attr_487 = None
        _holder__attr_488 = self._attr_488
        _holder__attr_489 = self._attr_489
        layer_norm_205 = custom_ln(
            input=linear_309,
            normalized_shape=(4096,),
            weight=_holder__attr_488,
            bias=_holder__attr_489,
            eps=1e-05,
        )
        _holder__attr_488 = _holder__attr_489 = None
        sigmoid_99 = torch.sigmoid(input=layer_norm_205)
        layer_norm_205 = None
        mul_115 = torch.mul(input=linear_309, other=sigmoid_99)
        linear_309 = sigmoid_99 = None
        _holder__attr_490 = self._attr_490
        _holder__attr_491 = self._attr_491
        linear_310 = torch.nn.functional.linear(
            input=mul_115, weight=_holder__attr_490, bias=_holder__attr_491
        )
        _holder__attr_490 = _holder__attr_491 = None
        _holder__attr_492 = self._attr_492
        _holder__attr_493 = self._attr_493
        layer_norm_206 = custom_ln(
            input=linear_310,
            normalized_shape=(2048,),
            weight=_holder__attr_492,
            bias=_holder__attr_493,
            eps=1e-05,
        )
        _holder__attr_492 = _holder__attr_493 = None
        sigmoid_100 = torch.sigmoid(input=layer_norm_206)
        layer_norm_206 = None
        mul_116 = torch.mul(input=linear_310, other=sigmoid_100)
        linear_310 = sigmoid_100 = None
        _holder__attr_494 = self._attr_494
        _holder__attr_495 = self._attr_495
        linear_311 = torch.nn.functional.linear(
            input=mul_116, weight=_holder__attr_494, bias=_holder__attr_495
        )
        mul_116 = _holder__attr_494 = _holder__attr_495 = None
        add_83 = torch.add(input=mul_115, other=linear_311)
        mul_115 = linear_311 = None
        _holder__attr_496 = self._attr_496
        _holder__attr_497 = self._attr_497
        layer_norm_207 = custom_ln(
            input=add_83,
            normalized_shape=(4096,),
            weight=_holder__attr_496,
            bias=_holder__attr_497,
            eps=1e-05,
        )
        _holder__attr_496 = _holder__attr_497 = None
        sigmoid_101 = torch.sigmoid(input=layer_norm_207)
        layer_norm_207 = None
        mul_117 = torch.mul(input=add_83, other=sigmoid_101)
        add_83 = sigmoid_101 = None
        _holder__attr_498 = self._attr_498
        _holder__attr_499 = self._attr_499
        linear_312 = torch.nn.functional.linear(
            input=mul_117, weight=_holder__attr_498, bias=_holder__attr_499
        )
        mul_117 = _holder__attr_498 = _holder__attr_499 = None
        relu_1 = torch.nn.functional.relu(input=linear_312, inplace=False)
        linear_312 = None
        _holder__attr_500 = self._attr_500
        _holder__attr_501 = self._attr_501
        linear_313 = torch.nn.functional.linear(
            input=relu_1, weight=_holder__attr_500, bias=_holder__attr_501
        )
        relu_1 = _holder__attr_500 = _holder__attr_501 = None
        return linear_313


def main():
    BS = 4096
    module = ExportedModule().half().cuda().eval()
    inputs = [
        torch.rand([4096, 49760], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 117120], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 880], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 1968], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 3594], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 200, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 200, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 200, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 200, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 200, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 200, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 24, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 24, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 24, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 24, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 24, 64], dtype=torch.float16, device="cuda"),
        torch.rand([4096, 24, 64], dtype=torch.float16, device="cuda"),
    ]
    dense_over_arch_flops = 1420.28 * 1e6
    t = benchmark_torch_function(
        # 100,
        1,
        lambda: module(*inputs),
    )
    print(
        f"Module (Eager), BS: {BS}, Time per iter: {t * 1.0e3:.2f}ms, QPS: {BS / t:.2f}, TFLOP/s: {BS * dense_over_arch_flops / t / 1.0e12:.2f},"
    )


if __name__ == '__main__':
    main()
    
