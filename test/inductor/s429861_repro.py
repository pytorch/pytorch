# flake8: noqa
# ruff: noqa: F841
import torch


inf = float("inf")


def forward(
    self,
    arg0_1: "f32[][]cuda:0",
    arg1_1: "f32[50][1]cuda:0",
    arg2_1: "f32[23][1]cuda:0",
    arg3_1: "f32[38][1]cuda:0",
    arg4_1: "f32[5][1]cuda:0",
    arg5_1: "f32[100][1]cuda:0",
    arg6_1: "f32[50][1]cuda:0",
    arg7_1: "f32[77][1]cuda:0",
    arg8_1: "f32[100][1]cuda:0",
    arg9_1: "f32[100][1]cuda:0",
    arg10_1: "f32[96][1]cuda:0",
    arg11_1: "f32[78][1]cuda:0",
    arg12_1: "f32[100][1]cuda:0",
    arg13_1: "f32[100][1]cuda:0",
    arg14_1: "f32[97][1]cuda:0",
    arg15_1: "f32[819, 732][732, 1]cuda:0",
    arg16_1: "f32[204][1]cuda:0",
    arg17_1: "f32[64][1]cuda:0",
    arg18_1: "f32[204][1]cuda:0",
    arg19_1: "f32[64, 204][204, 1]cuda:0",
    arg20_1: "f32[204][1]cuda:0",
    arg21_1: "f32[204, 160][160, 1]cuda:0",
    arg22_1: "f32[204][1]cuda:0",
    arg23_1: "f32[64][1]cuda:0",
    arg24_1: "f32[204][1]cuda:0",
    arg25_1: "f32[64, 204][204, 1]cuda:0",
    arg26_1: "f32[204][1]cuda:0",
    arg27_1: "f32[204][1]cuda:0",
    arg28_1: "f32[64][1]cuda:0",
    arg29_1: "f32[204][1]cuda:0",
    arg30_1: "f32[64, 204][204, 1]cuda:0",
    arg31_1: "f32[204][1]cuda:0",
    arg32_1: "f32[204, 72][72, 1]cuda:0",
    arg33_1: "f32[204][1]cuda:0",
    arg34_1: "f32[64][1]cuda:0",
    arg35_1: "f32[64, 204][204, 1]cuda:0",
    arg36_1: "f32[768, 2675][2675, 1]cuda:0",
    arg37_1: "f32[768, 2048][2048, 1]cuda:0",
    arg38_1: "f32[768][1]cuda:0",
    arg39_1: "f32[4096][1]cuda:0",
    arg40_1: "f32[4096, 256][256, 1]cuda:0",
    arg41_1: "f32[64][1]cuda:0",
    arg42_1: "f32[2675][1]cuda:0",
    arg43_1: "f32[1536, 4096][4096, 1]cuda:0",
    arg44_1: "f32[4096][1]cuda:0",
    arg45_1: "f32[1840][1]cuda:0",
    arg46_1: "f32[2048, 2675][2675, 1]cuda:0",
    arg47_1: "f32[2048][1]cuda:0",
    arg48_1: "f32[2048][1]cuda:0",
    arg49_1: "f32[768][1]cuda:0",
    arg50_1: "f32[256][1]cuda:0",
    arg51_1: "f32[768, 2048][2048, 1]cuda:0",
    arg52_1: "f32[4096][1]cuda:0",
    arg53_1: "f32[104][1]cuda:0",
    arg54_1: "f32[768][1]cuda:0",
    arg55_1: "f32[1024][1]cuda:0",
    arg56_1: "f32[2048][1]cuda:0",
    arg57_1: "f32[768, 2675][2675, 1]cuda:0",
    arg58_1: "f32[2675][1]cuda:0",
    arg59_1: "f32[256][1]cuda:0",
    arg60_1: "f32[768][1]cuda:0",
    arg61_1: "f32[256, 768][768, 1]cuda:0",
    arg62_1: "f32[64][1]cuda:0",
    arg63_1: "f32[1536][1]cuda:0",
    arg64_1: "f32[2048][1]cuda:0",
    arg65_1: "f32[3360][1]cuda:0",
    arg66_1: "f32[768][1]cuda:0",
    arg67_1: "f32[768, 2048][2048, 1]cuda:0",
    arg68_1: "f32[256][1]cuda:0",
    arg69_1: "f32[104, 256][256, 1]cuda:0",
    arg70_1: "f32[2675][1]cuda:0",
    arg71_1: "f32[768][1]cuda:0",
    arg72_1: "f32[2048][1]cuda:0",
    arg73_1: "f32[1024][1]cuda:0",
    arg74_1: "f32[64, 612][612, 1]cuda:0",
    arg75_1: "f32[128][1]cuda:0",
    arg76_1: "f32[308, 256][256, 1]cuda:0",
    arg77_1: "f32[1][1]cuda:0",
    arg78_1: "f32[512][1]cuda:0",
    arg79_1: "f32[512][1]cuda:0",
    arg80_1: "f32[50][1]cuda:0",
    arg81_1: "f32[23][1]cuda:0",
    arg82_1: "f32[38][1]cuda:0",
    arg83_1: "f32[5][1]cuda:0",
    arg84_1: "f32[100][1]cuda:0",
    arg85_1: "f32[50][1]cuda:0",
    arg86_1: "f32[77][1]cuda:0",
    arg87_1: "f32[100][1]cuda:0",
    arg88_1: "f32[100][1]cuda:0",
    arg89_1: "f32[96][1]cuda:0",
    arg90_1: "f32[78][1]cuda:0",
    arg91_1: "f32[100][1]cuda:0",
    arg92_1: "f32[100][1]cuda:0",
    arg93_1: "f32[97][1]cuda:0",
    arg94_1: "f32[819, 732][732, 1]cuda:0",
    arg95_1: "f32[204][1]cuda:0",
    arg96_1: "f32[64][1]cuda:0",
    arg97_1: "f32[204][1]cuda:0",
    arg98_1: "f32[64, 204][204, 1]cuda:0",
    arg99_1: "f32[204][1]cuda:0",
    arg100_1: "f32[204, 160][160, 1]cuda:0",
    arg101_1: "f32[204][1]cuda:0",
    arg102_1: "f32[64][1]cuda:0",
    arg103_1: "f32[204][1]cuda:0",
    arg104_1: "f32[64, 204][204, 1]cuda:0",
    arg105_1: "f32[204][1]cuda:0",
    arg106_1: "f32[204][1]cuda:0",
    arg107_1: "f32[64][1]cuda:0",
    arg108_1: "f32[204][1]cuda:0",
    arg109_1: "f32[64, 204][204, 1]cuda:0",
    arg110_1: "f32[204][1]cuda:0",
    arg111_1: "f32[204, 72][72, 1]cuda:0",
    arg112_1: "f32[204][1]cuda:0",
    arg113_1: "f32[64][1]cuda:0",
    arg114_1: "f32[64, 204][204, 1]cuda:0",
    arg115_1: "f32[768, 2675][2675, 1]cuda:0",
    arg116_1: "f32[768, 2048][2048, 1]cuda:0",
    arg117_1: "f32[768][1]cuda:0",
    arg118_1: "f32[4096][1]cuda:0",
    arg119_1: "f32[4096, 256][256, 1]cuda:0",
    arg120_1: "f32[64][1]cuda:0",
    arg121_1: "f32[2675][1]cuda:0",
    arg122_1: "f32[1536, 4096][22320, 1]cuda:0",
    arg123_1: "f32[4096][1]cuda:0",
    arg124_1: "f32[1840][1]cuda:0",
    arg125_1: "f32[2048, 2675][2675, 1]cuda:0",
    arg126_1: "f32[2048][1]cuda:0",
    arg127_1: "f32[2048][1]cuda:0",
    arg128_1: "f32[768][1]cuda:0",
    arg129_1: "f32[256][1]cuda:0",
    arg130_1: "f32[768, 2048][2048, 1]cuda:0",
    arg131_1: "f32[4096][1]cuda:0",
    arg132_1: "f32[104][1]cuda:0",
    arg133_1: "f32[768][1]cuda:0",
    arg134_1: "f32[1024][1]cuda:0",
    arg135_1: "f32[2048][1]cuda:0",
    arg136_1: "f32[768, 2675][2675, 1]cuda:0",
    arg137_1: "f32[2675][1]cuda:0",
    arg138_1: "f32[256][1]cuda:0",
    arg139_1: "f32[768][1]cuda:0",
    arg140_1: "f32[256, 768][768, 1]cuda:0",
    arg141_1: "f32[64][1]cuda:0",
    arg142_1: "f32[1536][1]cuda:0",
    arg143_1: "f32[2048][1]cuda:0",
    arg144_1: "f32[3360][1]cuda:0",
    arg145_1: "f32[768][1]cuda:0",
    arg146_1: "f32[768, 2048][2048, 1]cuda:0",
    arg147_1: "f32[256][1]cuda:0",
    arg148_1: "f32[104, 256][256, 1]cuda:0",
    arg149_1: "f32[2675][1]cuda:0",
    arg150_1: "f32[768][1]cuda:0",
    arg151_1: "f32[2048][1]cuda:0",
    arg152_1: "f32[1024][1]cuda:0",
    arg153_1: "f32[64, 612][612, 1]cuda:0",
    arg154_1: "f32[128][1]cuda:0",
    arg155_1: "f32[308, 256][256, 1]cuda:0",
    arg156_1: "f32[1][1]cuda:0",
    arg157_1: "f32[512][1]cuda:0",
    arg158_1: "f32[512][1]cuda:0",
):
    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:328 in torch_dynamo_resume_in__per_group_step_impl_at_316, code: -lr,
    neg: "f32[][]cuda:0" = torch.ops.aten.neg.default(arg0_1)
    arg0_1 = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:231 in _compute_clippy_shrinkage, code: masked_blocked_nom = torch._foreach_mul(
    _foreach_mul = torch.ops.aten._foreach_mul.Tensor(
        [
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            arg5_1,
            arg6_1,
            arg7_1,
            arg8_1,
            arg9_1,
            arg10_1,
            arg11_1,
            arg12_1,
            arg13_1,
            arg14_1,
            arg15_1,
            arg16_1,
            arg17_1,
            arg18_1,
            arg19_1,
            arg20_1,
            arg21_1,
            arg22_1,
            arg23_1,
            arg24_1,
            arg25_1,
            arg26_1,
            arg27_1,
            arg28_1,
            arg29_1,
            arg30_1,
            arg31_1,
            arg32_1,
            arg33_1,
            arg34_1,
            arg35_1,
            arg36_1,
            arg37_1,
            arg38_1,
            arg39_1,
            arg40_1,
            arg41_1,
            arg42_1,
            arg43_1,
            arg44_1,
            arg45_1,
            arg46_1,
            arg47_1,
            arg48_1,
            arg49_1,
            arg50_1,
            arg51_1,
            arg52_1,
            arg53_1,
            arg54_1,
            arg55_1,
            arg56_1,
            arg57_1,
            arg58_1,
            arg59_1,
            arg60_1,
            arg61_1,
            arg62_1,
            arg63_1,
            arg64_1,
            arg65_1,
            arg66_1,
            arg67_1,
            arg68_1,
            arg69_1,
            arg70_1,
            arg71_1,
            arg72_1,
            arg73_1,
            arg74_1,
            arg75_1,
            arg76_1,
            arg77_1,
            arg78_1,
            arg79_1,
        ],
        neg,
    )
    getitem: "f32[50][1]cuda:0" = _foreach_mul[0]
    getitem_1: "f32[23][1]cuda:0" = _foreach_mul[1]
    getitem_2: "f32[38][1]cuda:0" = _foreach_mul[2]
    getitem_3: "f32[5][1]cuda:0" = _foreach_mul[3]
    getitem_4: "f32[100][1]cuda:0" = _foreach_mul[4]
    getitem_5: "f32[50][1]cuda:0" = _foreach_mul[5]
    getitem_6: "f32[77][1]cuda:0" = _foreach_mul[6]
    getitem_7: "f32[100][1]cuda:0" = _foreach_mul[7]
    getitem_8: "f32[100][1]cuda:0" = _foreach_mul[8]
    getitem_9: "f32[96][1]cuda:0" = _foreach_mul[9]
    getitem_10: "f32[78][1]cuda:0" = _foreach_mul[10]
    getitem_11: "f32[100][1]cuda:0" = _foreach_mul[11]
    getitem_12: "f32[100][1]cuda:0" = _foreach_mul[12]
    getitem_13: "f32[97][1]cuda:0" = _foreach_mul[13]
    getitem_14: "f32[819, 732][732, 1]cuda:0" = _foreach_mul[14]
    getitem_15: "f32[204][1]cuda:0" = _foreach_mul[15]
    getitem_16: "f32[64][1]cuda:0" = _foreach_mul[16]
    getitem_17: "f32[204][1]cuda:0" = _foreach_mul[17]
    getitem_18: "f32[64, 204][204, 1]cuda:0" = _foreach_mul[18]
    getitem_19: "f32[204][1]cuda:0" = _foreach_mul[19]
    getitem_20: "f32[204, 160][160, 1]cuda:0" = _foreach_mul[20]
    getitem_21: "f32[204][1]cuda:0" = _foreach_mul[21]
    getitem_22: "f32[64][1]cuda:0" = _foreach_mul[22]
    getitem_23: "f32[204][1]cuda:0" = _foreach_mul[23]
    getitem_24: "f32[64, 204][204, 1]cuda:0" = _foreach_mul[24]
    getitem_25: "f32[204][1]cuda:0" = _foreach_mul[25]
    getitem_26: "f32[204][1]cuda:0" = _foreach_mul[26]
    getitem_27: "f32[64][1]cuda:0" = _foreach_mul[27]
    getitem_28: "f32[204][1]cuda:0" = _foreach_mul[28]
    getitem_29: "f32[64, 204][204, 1]cuda:0" = _foreach_mul[29]
    getitem_30: "f32[204][1]cuda:0" = _foreach_mul[30]
    getitem_31: "f32[204, 72][72, 1]cuda:0" = _foreach_mul[31]
    getitem_32: "f32[204][1]cuda:0" = _foreach_mul[32]
    getitem_33: "f32[64][1]cuda:0" = _foreach_mul[33]
    getitem_34: "f32[64, 204][204, 1]cuda:0" = _foreach_mul[34]
    getitem_35: "f32[768, 2675][2675, 1]cuda:0" = _foreach_mul[35]
    getitem_36: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul[36]
    getitem_37: "f32[768][1]cuda:0" = _foreach_mul[37]
    getitem_38: "f32[4096][1]cuda:0" = _foreach_mul[38]
    getitem_39: "f32[4096, 256][256, 1]cuda:0" = _foreach_mul[39]
    getitem_40: "f32[64][1]cuda:0" = _foreach_mul[40]
    getitem_41: "f32[2675][1]cuda:0" = _foreach_mul[41]
    getitem_42: "f32[1536, 4096][4096, 1]cuda:0" = _foreach_mul[42]
    getitem_43: "f32[4096][1]cuda:0" = _foreach_mul[43]
    getitem_44: "f32[1840][1]cuda:0" = _foreach_mul[44]
    getitem_45: "f32[2048, 2675][2675, 1]cuda:0" = _foreach_mul[45]
    getitem_46: "f32[2048][1]cuda:0" = _foreach_mul[46]
    getitem_47: "f32[2048][1]cuda:0" = _foreach_mul[47]
    getitem_48: "f32[768][1]cuda:0" = _foreach_mul[48]
    getitem_49: "f32[256][1]cuda:0" = _foreach_mul[49]
    getitem_50: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul[50]
    getitem_51: "f32[4096][1]cuda:0" = _foreach_mul[51]
    getitem_52: "f32[104][1]cuda:0" = _foreach_mul[52]
    getitem_53: "f32[768][1]cuda:0" = _foreach_mul[53]
    getitem_54: "f32[1024][1]cuda:0" = _foreach_mul[54]
    getitem_55: "f32[2048][1]cuda:0" = _foreach_mul[55]
    getitem_56: "f32[768, 2675][2675, 1]cuda:0" = _foreach_mul[56]
    getitem_57: "f32[2675][1]cuda:0" = _foreach_mul[57]
    getitem_58: "f32[256][1]cuda:0" = _foreach_mul[58]
    getitem_59: "f32[768][1]cuda:0" = _foreach_mul[59]
    getitem_60: "f32[256, 768][768, 1]cuda:0" = _foreach_mul[60]
    getitem_61: "f32[64][1]cuda:0" = _foreach_mul[61]
    getitem_62: "f32[1536][1]cuda:0" = _foreach_mul[62]
    getitem_63: "f32[2048][1]cuda:0" = _foreach_mul[63]
    getitem_64: "f32[3360][1]cuda:0" = _foreach_mul[64]
    getitem_65: "f32[768][1]cuda:0" = _foreach_mul[65]
    getitem_66: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul[66]
    getitem_67: "f32[256][1]cuda:0" = _foreach_mul[67]
    getitem_68: "f32[104, 256][256, 1]cuda:0" = _foreach_mul[68]
    getitem_69: "f32[2675][1]cuda:0" = _foreach_mul[69]
    getitem_70: "f32[768][1]cuda:0" = _foreach_mul[70]
    getitem_71: "f32[2048][1]cuda:0" = _foreach_mul[71]
    getitem_72: "f32[1024][1]cuda:0" = _foreach_mul[72]
    getitem_73: "f32[64, 612][612, 1]cuda:0" = _foreach_mul[73]
    getitem_74: "f32[128][1]cuda:0" = _foreach_mul[74]
    getitem_75: "f32[308, 256][256, 1]cuda:0" = _foreach_mul[75]
    getitem_76: "f32[1][1]cuda:0" = _foreach_mul[76]
    getitem_77: "f32[512][1]cuda:0" = _foreach_mul[77]
    getitem_78: "f32[512][1]cuda:0" = _foreach_mul[78]
    _foreach_mul = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:234 in _compute_clippy_shrinkage, code: masked_blocked_denom = torch._foreach_abs(masked_blocked_params)
    _foreach_abs = torch.ops.aten._foreach_abs.default(
        [
            arg80_1,
            arg81_1,
            arg82_1,
            arg83_1,
            arg84_1,
            arg85_1,
            arg86_1,
            arg87_1,
            arg88_1,
            arg89_1,
            arg90_1,
            arg91_1,
            arg92_1,
            arg93_1,
            arg94_1,
            arg95_1,
            arg96_1,
            arg97_1,
            arg98_1,
            arg99_1,
            arg100_1,
            arg101_1,
            arg102_1,
            arg103_1,
            arg104_1,
            arg105_1,
            arg106_1,
            arg107_1,
            arg108_1,
            arg109_1,
            arg110_1,
            arg111_1,
            arg112_1,
            arg113_1,
            arg114_1,
            arg115_1,
            arg116_1,
            arg117_1,
            arg118_1,
            arg119_1,
            arg120_1,
            arg121_1,
            arg122_1,
            arg123_1,
            arg124_1,
            arg125_1,
            arg126_1,
            arg127_1,
            arg128_1,
            arg129_1,
            arg130_1,
            arg131_1,
            arg132_1,
            arg133_1,
            arg134_1,
            arg135_1,
            arg136_1,
            arg137_1,
            arg138_1,
            arg139_1,
            arg140_1,
            arg141_1,
            arg142_1,
            arg143_1,
            arg144_1,
            arg145_1,
            arg146_1,
            arg147_1,
            arg148_1,
            arg149_1,
            arg150_1,
            arg151_1,
            arg152_1,
            arg153_1,
            arg154_1,
            arg155_1,
            arg156_1,
            arg157_1,
            arg158_1,
        ]
    )
    getitem_79: "f32[50][1]cuda:0" = _foreach_abs[0]
    getitem_80: "f32[23][1]cuda:0" = _foreach_abs[1]
    getitem_81: "f32[38][1]cuda:0" = _foreach_abs[2]
    getitem_82: "f32[5][1]cuda:0" = _foreach_abs[3]
    getitem_83: "f32[100][1]cuda:0" = _foreach_abs[4]
    getitem_84: "f32[50][1]cuda:0" = _foreach_abs[5]
    getitem_85: "f32[77][1]cuda:0" = _foreach_abs[6]
    getitem_86: "f32[100][1]cuda:0" = _foreach_abs[7]
    getitem_87: "f32[100][1]cuda:0" = _foreach_abs[8]
    getitem_88: "f32[96][1]cuda:0" = _foreach_abs[9]
    getitem_89: "f32[78][1]cuda:0" = _foreach_abs[10]
    getitem_90: "f32[100][1]cuda:0" = _foreach_abs[11]
    getitem_91: "f32[100][1]cuda:0" = _foreach_abs[12]
    getitem_92: "f32[97][1]cuda:0" = _foreach_abs[13]
    getitem_93: "f32[819, 732][732, 1]cuda:0" = _foreach_abs[14]
    getitem_94: "f32[204][1]cuda:0" = _foreach_abs[15]
    getitem_95: "f32[64][1]cuda:0" = _foreach_abs[16]
    getitem_96: "f32[204][1]cuda:0" = _foreach_abs[17]
    getitem_97: "f32[64, 204][204, 1]cuda:0" = _foreach_abs[18]
    getitem_98: "f32[204][1]cuda:0" = _foreach_abs[19]
    getitem_99: "f32[204, 160][160, 1]cuda:0" = _foreach_abs[20]
    getitem_100: "f32[204][1]cuda:0" = _foreach_abs[21]
    getitem_101: "f32[64][1]cuda:0" = _foreach_abs[22]
    getitem_102: "f32[204][1]cuda:0" = _foreach_abs[23]
    getitem_103: "f32[64, 204][204, 1]cuda:0" = _foreach_abs[24]
    getitem_104: "f32[204][1]cuda:0" = _foreach_abs[25]
    getitem_105: "f32[204][1]cuda:0" = _foreach_abs[26]
    getitem_106: "f32[64][1]cuda:0" = _foreach_abs[27]
    getitem_107: "f32[204][1]cuda:0" = _foreach_abs[28]
    getitem_108: "f32[64, 204][204, 1]cuda:0" = _foreach_abs[29]
    getitem_109: "f32[204][1]cuda:0" = _foreach_abs[30]
    getitem_110: "f32[204, 72][72, 1]cuda:0" = _foreach_abs[31]
    getitem_111: "f32[204][1]cuda:0" = _foreach_abs[32]
    getitem_112: "f32[64][1]cuda:0" = _foreach_abs[33]
    getitem_113: "f32[64, 204][204, 1]cuda:0" = _foreach_abs[34]
    getitem_114: "f32[768, 2675][2675, 1]cuda:0" = _foreach_abs[35]
    getitem_115: "f32[768, 2048][2048, 1]cuda:0" = _foreach_abs[36]
    getitem_116: "f32[768][1]cuda:0" = _foreach_abs[37]
    getitem_117: "f32[4096][1]cuda:0" = _foreach_abs[38]
    getitem_118: "f32[4096, 256][256, 1]cuda:0" = _foreach_abs[39]
    getitem_119: "f32[64][1]cuda:0" = _foreach_abs[40]
    getitem_120: "f32[2675][1]cuda:0" = _foreach_abs[41]
    getitem_121: "f32[1536, 4096][4096, 1]cuda:0" = _foreach_abs[42]
    getitem_122: "f32[4096][1]cuda:0" = _foreach_abs[43]
    getitem_123: "f32[1840][1]cuda:0" = _foreach_abs[44]
    getitem_124: "f32[2048, 2675][2675, 1]cuda:0" = _foreach_abs[45]
    getitem_125: "f32[2048][1]cuda:0" = _foreach_abs[46]
    getitem_126: "f32[2048][1]cuda:0" = _foreach_abs[47]
    getitem_127: "f32[768][1]cuda:0" = _foreach_abs[48]
    getitem_128: "f32[256][1]cuda:0" = _foreach_abs[49]
    getitem_129: "f32[768, 2048][2048, 1]cuda:0" = _foreach_abs[50]
    getitem_130: "f32[4096][1]cuda:0" = _foreach_abs[51]
    getitem_131: "f32[104][1]cuda:0" = _foreach_abs[52]
    getitem_132: "f32[768][1]cuda:0" = _foreach_abs[53]
    getitem_133: "f32[1024][1]cuda:0" = _foreach_abs[54]
    getitem_134: "f32[2048][1]cuda:0" = _foreach_abs[55]
    getitem_135: "f32[768, 2675][2675, 1]cuda:0" = _foreach_abs[56]
    getitem_136: "f32[2675][1]cuda:0" = _foreach_abs[57]
    getitem_137: "f32[256][1]cuda:0" = _foreach_abs[58]
    getitem_138: "f32[768][1]cuda:0" = _foreach_abs[59]
    getitem_139: "f32[256, 768][768, 1]cuda:0" = _foreach_abs[60]
    getitem_140: "f32[64][1]cuda:0" = _foreach_abs[61]
    getitem_141: "f32[1536][1]cuda:0" = _foreach_abs[62]
    getitem_142: "f32[2048][1]cuda:0" = _foreach_abs[63]
    getitem_143: "f32[3360][1]cuda:0" = _foreach_abs[64]
    getitem_144: "f32[768][1]cuda:0" = _foreach_abs[65]
    getitem_145: "f32[768, 2048][2048, 1]cuda:0" = _foreach_abs[66]
    getitem_146: "f32[256][1]cuda:0" = _foreach_abs[67]
    getitem_147: "f32[104, 256][256, 1]cuda:0" = _foreach_abs[68]
    getitem_148: "f32[2675][1]cuda:0" = _foreach_abs[69]
    getitem_149: "f32[768][1]cuda:0" = _foreach_abs[70]
    getitem_150: "f32[2048][1]cuda:0" = _foreach_abs[71]
    getitem_151: "f32[1024][1]cuda:0" = _foreach_abs[72]
    getitem_152: "f32[64, 612][612, 1]cuda:0" = _foreach_abs[73]
    getitem_153: "f32[128][1]cuda:0" = _foreach_abs[74]
    getitem_154: "f32[308, 256][256, 1]cuda:0" = _foreach_abs[75]
    getitem_155: "f32[1][1]cuda:0" = _foreach_abs[76]
    getitem_156: "f32[512][1]cuda:0" = _foreach_abs[77]
    getitem_157: "f32[512][1]cuda:0" = _foreach_abs[78]
    _foreach_abs = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:235 in _compute_clippy_shrinkage, code: torch._foreach_mul_(masked_blocked_denom, self._gamma1)
    _foreach_mul_1 = torch.ops.aten._foreach_mul.Scalar(
        [
            getitem_79,
            getitem_80,
            getitem_81,
            getitem_82,
            getitem_83,
            getitem_84,
            getitem_85,
            getitem_86,
            getitem_87,
            getitem_88,
            getitem_89,
            getitem_90,
            getitem_91,
            getitem_92,
            getitem_93,
            getitem_94,
            getitem_95,
            getitem_96,
            getitem_97,
            getitem_98,
            getitem_99,
            getitem_100,
            getitem_101,
            getitem_102,
            getitem_103,
            getitem_104,
            getitem_105,
            getitem_106,
            getitem_107,
            getitem_108,
            getitem_109,
            getitem_110,
            getitem_111,
            getitem_112,
            getitem_113,
            getitem_114,
            getitem_115,
            getitem_116,
            getitem_117,
            getitem_118,
            getitem_119,
            getitem_120,
            getitem_121,
            getitem_122,
            getitem_123,
            getitem_124,
            getitem_125,
            getitem_126,
            getitem_127,
            getitem_128,
            getitem_129,
            getitem_130,
            getitem_131,
            getitem_132,
            getitem_133,
            getitem_134,
            getitem_135,
            getitem_136,
            getitem_137,
            getitem_138,
            getitem_139,
            getitem_140,
            getitem_141,
            getitem_142,
            getitem_143,
            getitem_144,
            getitem_145,
            getitem_146,
            getitem_147,
            getitem_148,
            getitem_149,
            getitem_150,
            getitem_151,
            getitem_152,
            getitem_153,
            getitem_154,
            getitem_155,
            getitem_156,
            getitem_157,
        ],
        0.5,
    )
    getitem_79 = (
        getitem_80
    ) = (
        getitem_81
    ) = (
        getitem_82
    ) = (
        getitem_83
    ) = (
        getitem_84
    ) = (
        getitem_85
    ) = (
        getitem_86
    ) = (
        getitem_87
    ) = (
        getitem_88
    ) = (
        getitem_89
    ) = (
        getitem_90
    ) = (
        getitem_91
    ) = (
        getitem_92
    ) = (
        getitem_93
    ) = (
        getitem_94
    ) = (
        getitem_95
    ) = (
        getitem_96
    ) = (
        getitem_97
    ) = (
        getitem_98
    ) = (
        getitem_99
    ) = (
        getitem_100
    ) = (
        getitem_101
    ) = (
        getitem_102
    ) = (
        getitem_103
    ) = (
        getitem_104
    ) = (
        getitem_105
    ) = (
        getitem_106
    ) = (
        getitem_107
    ) = (
        getitem_108
    ) = (
        getitem_109
    ) = (
        getitem_110
    ) = (
        getitem_111
    ) = (
        getitem_112
    ) = (
        getitem_113
    ) = (
        getitem_114
    ) = (
        getitem_115
    ) = (
        getitem_116
    ) = (
        getitem_117
    ) = (
        getitem_118
    ) = (
        getitem_119
    ) = (
        getitem_120
    ) = (
        getitem_121
    ) = (
        getitem_122
    ) = (
        getitem_123
    ) = (
        getitem_124
    ) = (
        getitem_125
    ) = (
        getitem_126
    ) = (
        getitem_127
    ) = (
        getitem_128
    ) = (
        getitem_129
    ) = (
        getitem_130
    ) = (
        getitem_131
    ) = (
        getitem_132
    ) = (
        getitem_133
    ) = (
        getitem_134
    ) = (
        getitem_135
    ) = (
        getitem_136
    ) = (
        getitem_137
    ) = (
        getitem_138
    ) = (
        getitem_139
    ) = (
        getitem_140
    ) = (
        getitem_141
    ) = (
        getitem_142
    ) = (
        getitem_143
    ) = (
        getitem_144
    ) = (
        getitem_145
    ) = (
        getitem_146
    ) = (
        getitem_147
    ) = (
        getitem_148
    ) = (
        getitem_149
    ) = (
        getitem_150
    ) = (
        getitem_151
    ) = (
        getitem_152
    ) = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = None
    getitem_158: "f32[50][1]cuda:0" = _foreach_mul_1[0]
    getitem_159: "f32[23][1]cuda:0" = _foreach_mul_1[1]
    getitem_160: "f32[38][1]cuda:0" = _foreach_mul_1[2]
    getitem_161: "f32[5][1]cuda:0" = _foreach_mul_1[3]
    getitem_162: "f32[100][1]cuda:0" = _foreach_mul_1[4]
    getitem_163: "f32[50][1]cuda:0" = _foreach_mul_1[5]
    getitem_164: "f32[77][1]cuda:0" = _foreach_mul_1[6]
    getitem_165: "f32[100][1]cuda:0" = _foreach_mul_1[7]
    getitem_166: "f32[100][1]cuda:0" = _foreach_mul_1[8]
    getitem_167: "f32[96][1]cuda:0" = _foreach_mul_1[9]
    getitem_168: "f32[78][1]cuda:0" = _foreach_mul_1[10]
    getitem_169: "f32[100][1]cuda:0" = _foreach_mul_1[11]
    getitem_170: "f32[100][1]cuda:0" = _foreach_mul_1[12]
    getitem_171: "f32[97][1]cuda:0" = _foreach_mul_1[13]
    getitem_172: "f32[819, 732][732, 1]cuda:0" = _foreach_mul_1[14]
    getitem_173: "f32[204][1]cuda:0" = _foreach_mul_1[15]
    getitem_174: "f32[64][1]cuda:0" = _foreach_mul_1[16]
    getitem_175: "f32[204][1]cuda:0" = _foreach_mul_1[17]
    getitem_176: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_1[18]
    getitem_177: "f32[204][1]cuda:0" = _foreach_mul_1[19]
    getitem_178: "f32[204, 160][160, 1]cuda:0" = _foreach_mul_1[20]
    getitem_179: "f32[204][1]cuda:0" = _foreach_mul_1[21]
    getitem_180: "f32[64][1]cuda:0" = _foreach_mul_1[22]
    getitem_181: "f32[204][1]cuda:0" = _foreach_mul_1[23]
    getitem_182: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_1[24]
    getitem_183: "f32[204][1]cuda:0" = _foreach_mul_1[25]
    getitem_184: "f32[204][1]cuda:0" = _foreach_mul_1[26]
    getitem_185: "f32[64][1]cuda:0" = _foreach_mul_1[27]
    getitem_186: "f32[204][1]cuda:0" = _foreach_mul_1[28]
    getitem_187: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_1[29]
    getitem_188: "f32[204][1]cuda:0" = _foreach_mul_1[30]
    getitem_189: "f32[204, 72][72, 1]cuda:0" = _foreach_mul_1[31]
    getitem_190: "f32[204][1]cuda:0" = _foreach_mul_1[32]
    getitem_191: "f32[64][1]cuda:0" = _foreach_mul_1[33]
    getitem_192: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_1[34]
    getitem_193: "f32[768, 2675][2675, 1]cuda:0" = _foreach_mul_1[35]
    getitem_194: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul_1[36]
    getitem_195: "f32[768][1]cuda:0" = _foreach_mul_1[37]
    getitem_196: "f32[4096][1]cuda:0" = _foreach_mul_1[38]
    getitem_197: "f32[4096, 256][256, 1]cuda:0" = _foreach_mul_1[39]
    getitem_198: "f32[64][1]cuda:0" = _foreach_mul_1[40]
    getitem_199: "f32[2675][1]cuda:0" = _foreach_mul_1[41]
    getitem_200: "f32[1536, 4096][4096, 1]cuda:0" = _foreach_mul_1[42]
    getitem_201: "f32[4096][1]cuda:0" = _foreach_mul_1[43]
    getitem_202: "f32[1840][1]cuda:0" = _foreach_mul_1[44]
    getitem_203: "f32[2048, 2675][2675, 1]cuda:0" = _foreach_mul_1[45]
    getitem_204: "f32[2048][1]cuda:0" = _foreach_mul_1[46]
    getitem_205: "f32[2048][1]cuda:0" = _foreach_mul_1[47]
    getitem_206: "f32[768][1]cuda:0" = _foreach_mul_1[48]
    getitem_207: "f32[256][1]cuda:0" = _foreach_mul_1[49]
    getitem_208: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul_1[50]
    getitem_209: "f32[4096][1]cuda:0" = _foreach_mul_1[51]
    getitem_210: "f32[104][1]cuda:0" = _foreach_mul_1[52]
    getitem_211: "f32[768][1]cuda:0" = _foreach_mul_1[53]
    getitem_212: "f32[1024][1]cuda:0" = _foreach_mul_1[54]
    getitem_213: "f32[2048][1]cuda:0" = _foreach_mul_1[55]
    getitem_214: "f32[768, 2675][2675, 1]cuda:0" = _foreach_mul_1[56]
    getitem_215: "f32[2675][1]cuda:0" = _foreach_mul_1[57]
    getitem_216: "f32[256][1]cuda:0" = _foreach_mul_1[58]
    getitem_217: "f32[768][1]cuda:0" = _foreach_mul_1[59]
    getitem_218: "f32[256, 768][768, 1]cuda:0" = _foreach_mul_1[60]
    getitem_219: "f32[64][1]cuda:0" = _foreach_mul_1[61]
    getitem_220: "f32[1536][1]cuda:0" = _foreach_mul_1[62]
    getitem_221: "f32[2048][1]cuda:0" = _foreach_mul_1[63]
    getitem_222: "f32[3360][1]cuda:0" = _foreach_mul_1[64]
    getitem_223: "f32[768][1]cuda:0" = _foreach_mul_1[65]
    getitem_224: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul_1[66]
    getitem_225: "f32[256][1]cuda:0" = _foreach_mul_1[67]
    getitem_226: "f32[104, 256][256, 1]cuda:0" = _foreach_mul_1[68]
    getitem_227: "f32[2675][1]cuda:0" = _foreach_mul_1[69]
    getitem_228: "f32[768][1]cuda:0" = _foreach_mul_1[70]
    getitem_229: "f32[2048][1]cuda:0" = _foreach_mul_1[71]
    getitem_230: "f32[1024][1]cuda:0" = _foreach_mul_1[72]
    getitem_231: "f32[64, 612][612, 1]cuda:0" = _foreach_mul_1[73]
    getitem_232: "f32[128][1]cuda:0" = _foreach_mul_1[74]
    getitem_233: "f32[308, 256][256, 1]cuda:0" = _foreach_mul_1[75]
    getitem_234: "f32[1][1]cuda:0" = _foreach_mul_1[76]
    getitem_235: "f32[512][1]cuda:0" = _foreach_mul_1[77]
    getitem_236: "f32[512][1]cuda:0" = _foreach_mul_1[78]
    _foreach_mul_1 = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:236 in _compute_clippy_shrinkage, code: torch._foreach_add_(masked_blocked_denom, self._gamma2)
    _foreach_add = torch.ops.aten._foreach_add.Scalar(
        [
            getitem_158,
            getitem_159,
            getitem_160,
            getitem_161,
            getitem_162,
            getitem_163,
            getitem_164,
            getitem_165,
            getitem_166,
            getitem_167,
            getitem_168,
            getitem_169,
            getitem_170,
            getitem_171,
            getitem_172,
            getitem_173,
            getitem_174,
            getitem_175,
            getitem_176,
            getitem_177,
            getitem_178,
            getitem_179,
            getitem_180,
            getitem_181,
            getitem_182,
            getitem_183,
            getitem_184,
            getitem_185,
            getitem_186,
            getitem_187,
            getitem_188,
            getitem_189,
            getitem_190,
            getitem_191,
            getitem_192,
            getitem_193,
            getitem_194,
            getitem_195,
            getitem_196,
            getitem_197,
            getitem_198,
            getitem_199,
            getitem_200,
            getitem_201,
            getitem_202,
            getitem_203,
            getitem_204,
            getitem_205,
            getitem_206,
            getitem_207,
            getitem_208,
            getitem_209,
            getitem_210,
            getitem_211,
            getitem_212,
            getitem_213,
            getitem_214,
            getitem_215,
            getitem_216,
            getitem_217,
            getitem_218,
            getitem_219,
            getitem_220,
            getitem_221,
            getitem_222,
            getitem_223,
            getitem_224,
            getitem_225,
            getitem_226,
            getitem_227,
            getitem_228,
            getitem_229,
            getitem_230,
            getitem_231,
            getitem_232,
            getitem_233,
            getitem_234,
            getitem_235,
            getitem_236,
        ],
        0.01,
    )
    getitem_158 = (
        getitem_159
    ) = (
        getitem_160
    ) = (
        getitem_161
    ) = (
        getitem_162
    ) = (
        getitem_163
    ) = (
        getitem_164
    ) = (
        getitem_165
    ) = (
        getitem_166
    ) = (
        getitem_167
    ) = (
        getitem_168
    ) = (
        getitem_169
    ) = (
        getitem_170
    ) = (
        getitem_171
    ) = (
        getitem_172
    ) = (
        getitem_173
    ) = (
        getitem_174
    ) = (
        getitem_175
    ) = (
        getitem_176
    ) = (
        getitem_177
    ) = (
        getitem_178
    ) = (
        getitem_179
    ) = (
        getitem_180
    ) = (
        getitem_181
    ) = (
        getitem_182
    ) = (
        getitem_183
    ) = (
        getitem_184
    ) = (
        getitem_185
    ) = (
        getitem_186
    ) = (
        getitem_187
    ) = (
        getitem_188
    ) = (
        getitem_189
    ) = (
        getitem_190
    ) = (
        getitem_191
    ) = (
        getitem_192
    ) = (
        getitem_193
    ) = (
        getitem_194
    ) = (
        getitem_195
    ) = (
        getitem_196
    ) = (
        getitem_197
    ) = (
        getitem_198
    ) = (
        getitem_199
    ) = (
        getitem_200
    ) = (
        getitem_201
    ) = (
        getitem_202
    ) = (
        getitem_203
    ) = (
        getitem_204
    ) = (
        getitem_205
    ) = (
        getitem_206
    ) = (
        getitem_207
    ) = (
        getitem_208
    ) = (
        getitem_209
    ) = (
        getitem_210
    ) = (
        getitem_211
    ) = (
        getitem_212
    ) = (
        getitem_213
    ) = (
        getitem_214
    ) = (
        getitem_215
    ) = (
        getitem_216
    ) = (
        getitem_217
    ) = (
        getitem_218
    ) = (
        getitem_219
    ) = (
        getitem_220
    ) = (
        getitem_221
    ) = (
        getitem_222
    ) = (
        getitem_223
    ) = (
        getitem_224
    ) = (
        getitem_225
    ) = (
        getitem_226
    ) = (
        getitem_227
    ) = (
        getitem_228
    ) = (
        getitem_229
    ) = (
        getitem_230
    ) = (
        getitem_231
    ) = getitem_232 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = None
    getitem_237: "f32[50][1]cuda:0" = _foreach_add[0]
    getitem_238: "f32[23][1]cuda:0" = _foreach_add[1]
    getitem_239: "f32[38][1]cuda:0" = _foreach_add[2]
    getitem_240: "f32[5][1]cuda:0" = _foreach_add[3]
    getitem_241: "f32[100][1]cuda:0" = _foreach_add[4]
    getitem_242: "f32[50][1]cuda:0" = _foreach_add[5]
    getitem_243: "f32[77][1]cuda:0" = _foreach_add[6]
    getitem_244: "f32[100][1]cuda:0" = _foreach_add[7]
    getitem_245: "f32[100][1]cuda:0" = _foreach_add[8]
    getitem_246: "f32[96][1]cuda:0" = _foreach_add[9]
    getitem_247: "f32[78][1]cuda:0" = _foreach_add[10]
    getitem_248: "f32[100][1]cuda:0" = _foreach_add[11]
    getitem_249: "f32[100][1]cuda:0" = _foreach_add[12]
    getitem_250: "f32[97][1]cuda:0" = _foreach_add[13]
    getitem_251: "f32[819, 732][732, 1]cuda:0" = _foreach_add[14]
    getitem_252: "f32[204][1]cuda:0" = _foreach_add[15]
    getitem_253: "f32[64][1]cuda:0" = _foreach_add[16]
    getitem_254: "f32[204][1]cuda:0" = _foreach_add[17]
    getitem_255: "f32[64, 204][204, 1]cuda:0" = _foreach_add[18]
    getitem_256: "f32[204][1]cuda:0" = _foreach_add[19]
    getitem_257: "f32[204, 160][160, 1]cuda:0" = _foreach_add[20]
    getitem_258: "f32[204][1]cuda:0" = _foreach_add[21]
    getitem_259: "f32[64][1]cuda:0" = _foreach_add[22]
    getitem_260: "f32[204][1]cuda:0" = _foreach_add[23]
    getitem_261: "f32[64, 204][204, 1]cuda:0" = _foreach_add[24]
    getitem_262: "f32[204][1]cuda:0" = _foreach_add[25]
    getitem_263: "f32[204][1]cuda:0" = _foreach_add[26]
    getitem_264: "f32[64][1]cuda:0" = _foreach_add[27]
    getitem_265: "f32[204][1]cuda:0" = _foreach_add[28]
    getitem_266: "f32[64, 204][204, 1]cuda:0" = _foreach_add[29]
    getitem_267: "f32[204][1]cuda:0" = _foreach_add[30]
    getitem_268: "f32[204, 72][72, 1]cuda:0" = _foreach_add[31]
    getitem_269: "f32[204][1]cuda:0" = _foreach_add[32]
    getitem_270: "f32[64][1]cuda:0" = _foreach_add[33]
    getitem_271: "f32[64, 204][204, 1]cuda:0" = _foreach_add[34]
    getitem_272: "f32[768, 2675][2675, 1]cuda:0" = _foreach_add[35]
    getitem_273: "f32[768, 2048][2048, 1]cuda:0" = _foreach_add[36]
    getitem_274: "f32[768][1]cuda:0" = _foreach_add[37]
    getitem_275: "f32[4096][1]cuda:0" = _foreach_add[38]
    getitem_276: "f32[4096, 256][256, 1]cuda:0" = _foreach_add[39]
    getitem_277: "f32[64][1]cuda:0" = _foreach_add[40]
    getitem_278: "f32[2675][1]cuda:0" = _foreach_add[41]
    getitem_279: "f32[1536, 4096][4096, 1]cuda:0" = _foreach_add[42]
    getitem_280: "f32[4096][1]cuda:0" = _foreach_add[43]
    getitem_281: "f32[1840][1]cuda:0" = _foreach_add[44]
    getitem_282: "f32[2048, 2675][2675, 1]cuda:0" = _foreach_add[45]
    getitem_283: "f32[2048][1]cuda:0" = _foreach_add[46]
    getitem_284: "f32[2048][1]cuda:0" = _foreach_add[47]
    getitem_285: "f32[768][1]cuda:0" = _foreach_add[48]
    getitem_286: "f32[256][1]cuda:0" = _foreach_add[49]
    getitem_287: "f32[768, 2048][2048, 1]cuda:0" = _foreach_add[50]
    getitem_288: "f32[4096][1]cuda:0" = _foreach_add[51]
    getitem_289: "f32[104][1]cuda:0" = _foreach_add[52]
    getitem_290: "f32[768][1]cuda:0" = _foreach_add[53]
    getitem_291: "f32[1024][1]cuda:0" = _foreach_add[54]
    getitem_292: "f32[2048][1]cuda:0" = _foreach_add[55]
    getitem_293: "f32[768, 2675][2675, 1]cuda:0" = _foreach_add[56]
    getitem_294: "f32[2675][1]cuda:0" = _foreach_add[57]
    getitem_295: "f32[256][1]cuda:0" = _foreach_add[58]
    getitem_296: "f32[768][1]cuda:0" = _foreach_add[59]
    getitem_297: "f32[256, 768][768, 1]cuda:0" = _foreach_add[60]
    getitem_298: "f32[64][1]cuda:0" = _foreach_add[61]
    getitem_299: "f32[1536][1]cuda:0" = _foreach_add[62]
    getitem_300: "f32[2048][1]cuda:0" = _foreach_add[63]
    getitem_301: "f32[3360][1]cuda:0" = _foreach_add[64]
    getitem_302: "f32[768][1]cuda:0" = _foreach_add[65]
    getitem_303: "f32[768, 2048][2048, 1]cuda:0" = _foreach_add[66]
    getitem_304: "f32[256][1]cuda:0" = _foreach_add[67]
    getitem_305: "f32[104, 256][256, 1]cuda:0" = _foreach_add[68]
    getitem_306: "f32[2675][1]cuda:0" = _foreach_add[69]
    getitem_307: "f32[768][1]cuda:0" = _foreach_add[70]
    getitem_308: "f32[2048][1]cuda:0" = _foreach_add[71]
    getitem_309: "f32[1024][1]cuda:0" = _foreach_add[72]
    getitem_310: "f32[64, 612][612, 1]cuda:0" = _foreach_add[73]
    getitem_311: "f32[128][1]cuda:0" = _foreach_add[74]
    getitem_312: "f32[308, 256][256, 1]cuda:0" = _foreach_add[75]
    getitem_313: "f32[1][1]cuda:0" = _foreach_add[76]
    getitem_314: "f32[512][1]cuda:0" = _foreach_add[77]
    getitem_315: "f32[512][1]cuda:0" = _foreach_add[78]
    _foreach_add = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:237 in _compute_clippy_shrinkage, code: torch._foreach_div_(masked_blocked_nom, masked_blocked_denom)
    _foreach_div = torch.ops.aten._foreach_div.List(
        [
            getitem,
            getitem_1,
            getitem_2,
            getitem_3,
            getitem_4,
            getitem_5,
            getitem_6,
            getitem_7,
            getitem_8,
            getitem_9,
            getitem_10,
            getitem_11,
            getitem_12,
            getitem_13,
            getitem_14,
            getitem_15,
            getitem_16,
            getitem_17,
            getitem_18,
            getitem_19,
            getitem_20,
            getitem_21,
            getitem_22,
            getitem_23,
            getitem_24,
            getitem_25,
            getitem_26,
            getitem_27,
            getitem_28,
            getitem_29,
            getitem_30,
            getitem_31,
            getitem_32,
            getitem_33,
            getitem_34,
            getitem_35,
            getitem_36,
            getitem_37,
            getitem_38,
            getitem_39,
            getitem_40,
            getitem_41,
            getitem_42,
            getitem_43,
            getitem_44,
            getitem_45,
            getitem_46,
            getitem_47,
            getitem_48,
            getitem_49,
            getitem_50,
            getitem_51,
            getitem_52,
            getitem_53,
            getitem_54,
            getitem_55,
            getitem_56,
            getitem_57,
            getitem_58,
            getitem_59,
            getitem_60,
            getitem_61,
            getitem_62,
            getitem_63,
            getitem_64,
            getitem_65,
            getitem_66,
            getitem_67,
            getitem_68,
            getitem_69,
            getitem_70,
            getitem_71,
            getitem_72,
            getitem_73,
            getitem_74,
            getitem_75,
            getitem_76,
            getitem_77,
            getitem_78,
        ],
        [
            getitem_237,
            getitem_238,
            getitem_239,
            getitem_240,
            getitem_241,
            getitem_242,
            getitem_243,
            getitem_244,
            getitem_245,
            getitem_246,
            getitem_247,
            getitem_248,
            getitem_249,
            getitem_250,
            getitem_251,
            getitem_252,
            getitem_253,
            getitem_254,
            getitem_255,
            getitem_256,
            getitem_257,
            getitem_258,
            getitem_259,
            getitem_260,
            getitem_261,
            getitem_262,
            getitem_263,
            getitem_264,
            getitem_265,
            getitem_266,
            getitem_267,
            getitem_268,
            getitem_269,
            getitem_270,
            getitem_271,
            getitem_272,
            getitem_273,
            getitem_274,
            getitem_275,
            getitem_276,
            getitem_277,
            getitem_278,
            getitem_279,
            getitem_280,
            getitem_281,
            getitem_282,
            getitem_283,
            getitem_284,
            getitem_285,
            getitem_286,
            getitem_287,
            getitem_288,
            getitem_289,
            getitem_290,
            getitem_291,
            getitem_292,
            getitem_293,
            getitem_294,
            getitem_295,
            getitem_296,
            getitem_297,
            getitem_298,
            getitem_299,
            getitem_300,
            getitem_301,
            getitem_302,
            getitem_303,
            getitem_304,
            getitem_305,
            getitem_306,
            getitem_307,
            getitem_308,
            getitem_309,
            getitem_310,
            getitem_311,
            getitem_312,
            getitem_313,
            getitem_314,
            getitem_315,
        ],
    )
    getitem = (
        getitem_1
    ) = (
        getitem_2
    ) = (
        getitem_3
    ) = (
        getitem_4
    ) = (
        getitem_5
    ) = (
        getitem_6
    ) = (
        getitem_7
    ) = (
        getitem_8
    ) = (
        getitem_9
    ) = (
        getitem_10
    ) = (
        getitem_11
    ) = (
        getitem_12
    ) = (
        getitem_13
    ) = (
        getitem_14
    ) = (
        getitem_15
    ) = (
        getitem_16
    ) = (
        getitem_17
    ) = (
        getitem_18
    ) = (
        getitem_19
    ) = (
        getitem_20
    ) = (
        getitem_21
    ) = (
        getitem_22
    ) = (
        getitem_23
    ) = (
        getitem_24
    ) = (
        getitem_25
    ) = (
        getitem_26
    ) = (
        getitem_27
    ) = (
        getitem_28
    ) = (
        getitem_29
    ) = (
        getitem_30
    ) = (
        getitem_31
    ) = (
        getitem_32
    ) = (
        getitem_33
    ) = (
        getitem_34
    ) = (
        getitem_35
    ) = (
        getitem_36
    ) = (
        getitem_37
    ) = (
        getitem_38
    ) = (
        getitem_39
    ) = (
        getitem_40
    ) = (
        getitem_41
    ) = (
        getitem_42
    ) = (
        getitem_43
    ) = (
        getitem_44
    ) = (
        getitem_45
    ) = (
        getitem_46
    ) = (
        getitem_47
    ) = (
        getitem_48
    ) = (
        getitem_49
    ) = (
        getitem_50
    ) = (
        getitem_51
    ) = (
        getitem_52
    ) = (
        getitem_53
    ) = (
        getitem_54
    ) = (
        getitem_55
    ) = (
        getitem_56
    ) = (
        getitem_57
    ) = (
        getitem_58
    ) = (
        getitem_59
    ) = (
        getitem_60
    ) = (
        getitem_61
    ) = (
        getitem_62
    ) = (
        getitem_63
    ) = (
        getitem_64
    ) = (
        getitem_65
    ) = (
        getitem_66
    ) = (
        getitem_67
    ) = (
        getitem_68
    ) = (
        getitem_69
    ) = (
        getitem_70
    ) = (
        getitem_71
    ) = (
        getitem_72
    ) = (
        getitem_73
    ) = (
        getitem_74
    ) = (
        getitem_75
    ) = (
        getitem_76
    ) = (
        getitem_77
    ) = (
        getitem_78
    ) = (
        getitem_237
    ) = (
        getitem_238
    ) = (
        getitem_239
    ) = (
        getitem_240
    ) = (
        getitem_241
    ) = (
        getitem_242
    ) = (
        getitem_243
    ) = (
        getitem_244
    ) = (
        getitem_245
    ) = (
        getitem_246
    ) = (
        getitem_247
    ) = (
        getitem_248
    ) = (
        getitem_249
    ) = (
        getitem_250
    ) = (
        getitem_251
    ) = (
        getitem_252
    ) = (
        getitem_253
    ) = (
        getitem_254
    ) = (
        getitem_255
    ) = (
        getitem_256
    ) = (
        getitem_257
    ) = (
        getitem_258
    ) = (
        getitem_259
    ) = (
        getitem_260
    ) = (
        getitem_261
    ) = (
        getitem_262
    ) = (
        getitem_263
    ) = (
        getitem_264
    ) = (
        getitem_265
    ) = (
        getitem_266
    ) = (
        getitem_267
    ) = (
        getitem_268
    ) = (
        getitem_269
    ) = (
        getitem_270
    ) = (
        getitem_271
    ) = (
        getitem_272
    ) = (
        getitem_273
    ) = (
        getitem_274
    ) = (
        getitem_275
    ) = (
        getitem_276
    ) = (
        getitem_277
    ) = (
        getitem_278
    ) = (
        getitem_279
    ) = (
        getitem_280
    ) = (
        getitem_281
    ) = (
        getitem_282
    ) = (
        getitem_283
    ) = (
        getitem_284
    ) = (
        getitem_285
    ) = (
        getitem_286
    ) = (
        getitem_287
    ) = (
        getitem_288
    ) = (
        getitem_289
    ) = (
        getitem_290
    ) = (
        getitem_291
    ) = (
        getitem_292
    ) = (
        getitem_293
    ) = (
        getitem_294
    ) = (
        getitem_295
    ) = (
        getitem_296
    ) = (
        getitem_297
    ) = (
        getitem_298
    ) = (
        getitem_299
    ) = (
        getitem_300
    ) = (
        getitem_301
    ) = (
        getitem_302
    ) = (
        getitem_303
    ) = (
        getitem_304
    ) = (
        getitem_305
    ) = (
        getitem_306
    ) = (
        getitem_307
    ) = (
        getitem_308
    ) = (
        getitem_309
    ) = (
        getitem_310
    ) = getitem_311 = getitem_312 = getitem_313 = getitem_314 = getitem_315 = None
    getitem_316: "f32[50][1]cuda:0" = _foreach_div[0]
    getitem_317: "f32[23][1]cuda:0" = _foreach_div[1]
    getitem_318: "f32[38][1]cuda:0" = _foreach_div[2]
    getitem_319: "f32[5][1]cuda:0" = _foreach_div[3]
    getitem_320: "f32[100][1]cuda:0" = _foreach_div[4]
    getitem_321: "f32[50][1]cuda:0" = _foreach_div[5]
    getitem_322: "f32[77][1]cuda:0" = _foreach_div[6]
    getitem_323: "f32[100][1]cuda:0" = _foreach_div[7]
    getitem_324: "f32[100][1]cuda:0" = _foreach_div[8]
    getitem_325: "f32[96][1]cuda:0" = _foreach_div[9]
    getitem_326: "f32[78][1]cuda:0" = _foreach_div[10]
    getitem_327: "f32[100][1]cuda:0" = _foreach_div[11]
    getitem_328: "f32[100][1]cuda:0" = _foreach_div[12]
    getitem_329: "f32[97][1]cuda:0" = _foreach_div[13]
    getitem_330: "f32[819, 732][732, 1]cuda:0" = _foreach_div[14]
    getitem_331: "f32[204][1]cuda:0" = _foreach_div[15]
    getitem_332: "f32[64][1]cuda:0" = _foreach_div[16]
    getitem_333: "f32[204][1]cuda:0" = _foreach_div[17]
    getitem_334: "f32[64, 204][204, 1]cuda:0" = _foreach_div[18]
    getitem_335: "f32[204][1]cuda:0" = _foreach_div[19]
    getitem_336: "f32[204, 160][160, 1]cuda:0" = _foreach_div[20]
    getitem_337: "f32[204][1]cuda:0" = _foreach_div[21]
    getitem_338: "f32[64][1]cuda:0" = _foreach_div[22]
    getitem_339: "f32[204][1]cuda:0" = _foreach_div[23]
    getitem_340: "f32[64, 204][204, 1]cuda:0" = _foreach_div[24]
    getitem_341: "f32[204][1]cuda:0" = _foreach_div[25]
    getitem_342: "f32[204][1]cuda:0" = _foreach_div[26]
    getitem_343: "f32[64][1]cuda:0" = _foreach_div[27]
    getitem_344: "f32[204][1]cuda:0" = _foreach_div[28]
    getitem_345: "f32[64, 204][204, 1]cuda:0" = _foreach_div[29]
    getitem_346: "f32[204][1]cuda:0" = _foreach_div[30]
    getitem_347: "f32[204, 72][72, 1]cuda:0" = _foreach_div[31]
    getitem_348: "f32[204][1]cuda:0" = _foreach_div[32]
    getitem_349: "f32[64][1]cuda:0" = _foreach_div[33]
    getitem_350: "f32[64, 204][204, 1]cuda:0" = _foreach_div[34]
    getitem_351: "f32[768, 2675][2675, 1]cuda:0" = _foreach_div[35]
    getitem_352: "f32[768, 2048][2048, 1]cuda:0" = _foreach_div[36]
    getitem_353: "f32[768][1]cuda:0" = _foreach_div[37]
    getitem_354: "f32[4096][1]cuda:0" = _foreach_div[38]
    getitem_355: "f32[4096, 256][256, 1]cuda:0" = _foreach_div[39]
    getitem_356: "f32[64][1]cuda:0" = _foreach_div[40]
    getitem_357: "f32[2675][1]cuda:0" = _foreach_div[41]
    getitem_358: "f32[1536, 4096][4096, 1]cuda:0" = _foreach_div[42]
    getitem_359: "f32[4096][1]cuda:0" = _foreach_div[43]
    getitem_360: "f32[1840][1]cuda:0" = _foreach_div[44]
    getitem_361: "f32[2048, 2675][2675, 1]cuda:0" = _foreach_div[45]
    getitem_362: "f32[2048][1]cuda:0" = _foreach_div[46]
    getitem_363: "f32[2048][1]cuda:0" = _foreach_div[47]
    getitem_364: "f32[768][1]cuda:0" = _foreach_div[48]
    getitem_365: "f32[256][1]cuda:0" = _foreach_div[49]
    getitem_366: "f32[768, 2048][2048, 1]cuda:0" = _foreach_div[50]
    getitem_367: "f32[4096][1]cuda:0" = _foreach_div[51]
    getitem_368: "f32[104][1]cuda:0" = _foreach_div[52]
    getitem_369: "f32[768][1]cuda:0" = _foreach_div[53]
    getitem_370: "f32[1024][1]cuda:0" = _foreach_div[54]
    getitem_371: "f32[2048][1]cuda:0" = _foreach_div[55]
    getitem_372: "f32[768, 2675][2675, 1]cuda:0" = _foreach_div[56]
    getitem_373: "f32[2675][1]cuda:0" = _foreach_div[57]
    getitem_374: "f32[256][1]cuda:0" = _foreach_div[58]
    getitem_375: "f32[768][1]cuda:0" = _foreach_div[59]
    getitem_376: "f32[256, 768][768, 1]cuda:0" = _foreach_div[60]
    getitem_377: "f32[64][1]cuda:0" = _foreach_div[61]
    getitem_378: "f32[1536][1]cuda:0" = _foreach_div[62]
    getitem_379: "f32[2048][1]cuda:0" = _foreach_div[63]
    getitem_380: "f32[3360][1]cuda:0" = _foreach_div[64]
    getitem_381: "f32[768][1]cuda:0" = _foreach_div[65]
    getitem_382: "f32[768, 2048][2048, 1]cuda:0" = _foreach_div[66]
    getitem_383: "f32[256][1]cuda:0" = _foreach_div[67]
    getitem_384: "f32[104, 256][256, 1]cuda:0" = _foreach_div[68]
    getitem_385: "f32[2675][1]cuda:0" = _foreach_div[69]
    getitem_386: "f32[768][1]cuda:0" = _foreach_div[70]
    getitem_387: "f32[2048][1]cuda:0" = _foreach_div[71]
    getitem_388: "f32[1024][1]cuda:0" = _foreach_div[72]
    getitem_389: "f32[64, 612][612, 1]cuda:0" = _foreach_div[73]
    getitem_390: "f32[128][1]cuda:0" = _foreach_div[74]
    getitem_391: "f32[308, 256][256, 1]cuda:0" = _foreach_div[75]
    getitem_392: "f32[1][1]cuda:0" = _foreach_div[76]
    getitem_393: "f32[512][1]cuda:0" = _foreach_div[77]
    getitem_394: "f32[512][1]cuda:0" = _foreach_div[78]
    _foreach_div = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:238 in _compute_clippy_shrinkage, code: masked_blocked_shrinkage = torch._foreach_norm(masked_blocked_nom, float("inf"))
    _foreach_norm = torch.ops.aten._foreach_norm.Scalar(
        [
            getitem_316,
            getitem_317,
            getitem_318,
            getitem_319,
            getitem_320,
            getitem_321,
            getitem_322,
            getitem_323,
            getitem_324,
            getitem_325,
            getitem_326,
            getitem_327,
            getitem_328,
            getitem_329,
            getitem_330,
            getitem_331,
            getitem_332,
            getitem_333,
            getitem_334,
            getitem_335,
            getitem_336,
            getitem_337,
            getitem_338,
            getitem_339,
            getitem_340,
            getitem_341,
            getitem_342,
            getitem_343,
            getitem_344,
            getitem_345,
            getitem_346,
            getitem_347,
            getitem_348,
            getitem_349,
            getitem_350,
            getitem_351,
            getitem_352,
            getitem_353,
            getitem_354,
            getitem_355,
            getitem_356,
            getitem_357,
            getitem_358,
            getitem_359,
            getitem_360,
            getitem_361,
            getitem_362,
            getitem_363,
            getitem_364,
            getitem_365,
            getitem_366,
            getitem_367,
            getitem_368,
            getitem_369,
            getitem_370,
            getitem_371,
            getitem_372,
            getitem_373,
            getitem_374,
            getitem_375,
            getitem_376,
            getitem_377,
            getitem_378,
            getitem_379,
            getitem_380,
            getitem_381,
            getitem_382,
            getitem_383,
            getitem_384,
            getitem_385,
            getitem_386,
            getitem_387,
            getitem_388,
            getitem_389,
            getitem_390,
            getitem_391,
            getitem_392,
            getitem_393,
            getitem_394,
        ],
        inf,
    )
    getitem_316 = (
        getitem_317
    ) = (
        getitem_318
    ) = (
        getitem_319
    ) = (
        getitem_320
    ) = (
        getitem_321
    ) = (
        getitem_322
    ) = (
        getitem_323
    ) = (
        getitem_324
    ) = (
        getitem_325
    ) = (
        getitem_326
    ) = (
        getitem_327
    ) = (
        getitem_328
    ) = (
        getitem_329
    ) = (
        getitem_330
    ) = (
        getitem_331
    ) = (
        getitem_332
    ) = (
        getitem_333
    ) = (
        getitem_334
    ) = (
        getitem_335
    ) = (
        getitem_336
    ) = (
        getitem_337
    ) = (
        getitem_338
    ) = (
        getitem_339
    ) = (
        getitem_340
    ) = (
        getitem_341
    ) = (
        getitem_342
    ) = (
        getitem_343
    ) = (
        getitem_344
    ) = (
        getitem_345
    ) = (
        getitem_346
    ) = (
        getitem_347
    ) = (
        getitem_348
    ) = (
        getitem_349
    ) = (
        getitem_350
    ) = (
        getitem_351
    ) = (
        getitem_352
    ) = (
        getitem_353
    ) = (
        getitem_354
    ) = (
        getitem_355
    ) = (
        getitem_356
    ) = (
        getitem_357
    ) = (
        getitem_358
    ) = (
        getitem_359
    ) = (
        getitem_360
    ) = (
        getitem_361
    ) = (
        getitem_362
    ) = (
        getitem_363
    ) = (
        getitem_364
    ) = (
        getitem_365
    ) = (
        getitem_366
    ) = (
        getitem_367
    ) = (
        getitem_368
    ) = (
        getitem_369
    ) = (
        getitem_370
    ) = (
        getitem_371
    ) = (
        getitem_372
    ) = (
        getitem_373
    ) = (
        getitem_374
    ) = (
        getitem_375
    ) = (
        getitem_376
    ) = (
        getitem_377
    ) = (
        getitem_378
    ) = (
        getitem_379
    ) = (
        getitem_380
    ) = (
        getitem_381
    ) = (
        getitem_382
    ) = (
        getitem_383
    ) = (
        getitem_384
    ) = (
        getitem_385
    ) = (
        getitem_386
    ) = (
        getitem_387
    ) = (
        getitem_388
    ) = (
        getitem_389
    ) = getitem_390 = getitem_391 = getitem_392 = getitem_393 = getitem_394 = None
    getitem_395: "f32[][]cuda:0" = _foreach_norm[0]
    getitem_396: "f32[][]cuda:0" = _foreach_norm[1]
    getitem_397: "f32[][]cuda:0" = _foreach_norm[2]
    getitem_398: "f32[][]cuda:0" = _foreach_norm[3]
    getitem_399: "f32[][]cuda:0" = _foreach_norm[4]
    getitem_400: "f32[][]cuda:0" = _foreach_norm[5]
    getitem_401: "f32[][]cuda:0" = _foreach_norm[6]
    getitem_402: "f32[][]cuda:0" = _foreach_norm[7]
    getitem_403: "f32[][]cuda:0" = _foreach_norm[8]
    getitem_404: "f32[][]cuda:0" = _foreach_norm[9]
    getitem_405: "f32[][]cuda:0" = _foreach_norm[10]
    getitem_406: "f32[][]cuda:0" = _foreach_norm[11]
    getitem_407: "f32[][]cuda:0" = _foreach_norm[12]
    getitem_408: "f32[][]cuda:0" = _foreach_norm[13]
    getitem_409: "f32[][]cuda:0" = _foreach_norm[14]
    getitem_410: "f32[][]cuda:0" = _foreach_norm[15]
    getitem_411: "f32[][]cuda:0" = _foreach_norm[16]
    getitem_412: "f32[][]cuda:0" = _foreach_norm[17]
    getitem_413: "f32[][]cuda:0" = _foreach_norm[18]
    getitem_414: "f32[][]cuda:0" = _foreach_norm[19]
    getitem_415: "f32[][]cuda:0" = _foreach_norm[20]
    getitem_416: "f32[][]cuda:0" = _foreach_norm[21]
    getitem_417: "f32[][]cuda:0" = _foreach_norm[22]
    getitem_418: "f32[][]cuda:0" = _foreach_norm[23]
    getitem_419: "f32[][]cuda:0" = _foreach_norm[24]
    getitem_420: "f32[][]cuda:0" = _foreach_norm[25]
    getitem_421: "f32[][]cuda:0" = _foreach_norm[26]
    getitem_422: "f32[][]cuda:0" = _foreach_norm[27]
    getitem_423: "f32[][]cuda:0" = _foreach_norm[28]
    getitem_424: "f32[][]cuda:0" = _foreach_norm[29]
    getitem_425: "f32[][]cuda:0" = _foreach_norm[30]
    getitem_426: "f32[][]cuda:0" = _foreach_norm[31]
    getitem_427: "f32[][]cuda:0" = _foreach_norm[32]
    getitem_428: "f32[][]cuda:0" = _foreach_norm[33]
    getitem_429: "f32[][]cuda:0" = _foreach_norm[34]
    getitem_430: "f32[][]cuda:0" = _foreach_norm[35]
    getitem_431: "f32[][]cuda:0" = _foreach_norm[36]
    getitem_432: "f32[][]cuda:0" = _foreach_norm[37]
    getitem_433: "f32[][]cuda:0" = _foreach_norm[38]
    getitem_434: "f32[][]cuda:0" = _foreach_norm[39]
    getitem_435: "f32[][]cuda:0" = _foreach_norm[40]
    getitem_436: "f32[][]cuda:0" = _foreach_norm[41]
    getitem_437: "f32[][]cuda:0" = _foreach_norm[42]
    getitem_438: "f32[][]cuda:0" = _foreach_norm[43]
    getitem_439: "f32[][]cuda:0" = _foreach_norm[44]
    getitem_440: "f32[][]cuda:0" = _foreach_norm[45]
    getitem_441: "f32[][]cuda:0" = _foreach_norm[46]
    getitem_442: "f32[][]cuda:0" = _foreach_norm[47]
    getitem_443: "f32[][]cuda:0" = _foreach_norm[48]
    getitem_444: "f32[][]cuda:0" = _foreach_norm[49]
    getitem_445: "f32[][]cuda:0" = _foreach_norm[50]
    getitem_446: "f32[][]cuda:0" = _foreach_norm[51]
    getitem_447: "f32[][]cuda:0" = _foreach_norm[52]
    getitem_448: "f32[][]cuda:0" = _foreach_norm[53]
    getitem_449: "f32[][]cuda:0" = _foreach_norm[54]
    getitem_450: "f32[][]cuda:0" = _foreach_norm[55]
    getitem_451: "f32[][]cuda:0" = _foreach_norm[56]
    getitem_452: "f32[][]cuda:0" = _foreach_norm[57]
    getitem_453: "f32[][]cuda:0" = _foreach_norm[58]
    getitem_454: "f32[][]cuda:0" = _foreach_norm[59]
    getitem_455: "f32[][]cuda:0" = _foreach_norm[60]
    getitem_456: "f32[][]cuda:0" = _foreach_norm[61]
    getitem_457: "f32[][]cuda:0" = _foreach_norm[62]
    getitem_458: "f32[][]cuda:0" = _foreach_norm[63]
    getitem_459: "f32[][]cuda:0" = _foreach_norm[64]
    getitem_460: "f32[][]cuda:0" = _foreach_norm[65]
    getitem_461: "f32[][]cuda:0" = _foreach_norm[66]
    getitem_462: "f32[][]cuda:0" = _foreach_norm[67]
    getitem_463: "f32[][]cuda:0" = _foreach_norm[68]
    getitem_464: "f32[][]cuda:0" = _foreach_norm[69]
    getitem_465: "f32[][]cuda:0" = _foreach_norm[70]
    getitem_466: "f32[][]cuda:0" = _foreach_norm[71]
    getitem_467: "f32[][]cuda:0" = _foreach_norm[72]
    getitem_468: "f32[][]cuda:0" = _foreach_norm[73]
    getitem_469: "f32[][]cuda:0" = _foreach_norm[74]
    getitem_470: "f32[][]cuda:0" = _foreach_norm[75]
    getitem_471: "f32[][]cuda:0" = _foreach_norm[76]
    getitem_472: "f32[][]cuda:0" = _foreach_norm[77]
    getitem_473: "f32[][]cuda:0" = _foreach_norm[78]
    _foreach_norm = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:239 in _compute_clippy_shrinkage, code: torch._foreach_maximum_(masked_blocked_shrinkage, 1.0)
    _foreach_maximum = torch.ops.aten._foreach_maximum.Scalar(
        [
            getitem_395,
            getitem_396,
            getitem_397,
            getitem_398,
            getitem_399,
            getitem_400,
            getitem_401,
            getitem_402,
            getitem_403,
            getitem_404,
            getitem_405,
            getitem_406,
            getitem_407,
            getitem_408,
            getitem_409,
            getitem_410,
            getitem_411,
            getitem_412,
            getitem_413,
            getitem_414,
            getitem_415,
            getitem_416,
            getitem_417,
            getitem_418,
            getitem_419,
            getitem_420,
            getitem_421,
            getitem_422,
            getitem_423,
            getitem_424,
            getitem_425,
            getitem_426,
            getitem_427,
            getitem_428,
            getitem_429,
            getitem_430,
            getitem_431,
            getitem_432,
            getitem_433,
            getitem_434,
            getitem_435,
            getitem_436,
            getitem_437,
            getitem_438,
            getitem_439,
            getitem_440,
            getitem_441,
            getitem_442,
            getitem_443,
            getitem_444,
            getitem_445,
            getitem_446,
            getitem_447,
            getitem_448,
            getitem_449,
            getitem_450,
            getitem_451,
            getitem_452,
            getitem_453,
            getitem_454,
            getitem_455,
            getitem_456,
            getitem_457,
            getitem_458,
            getitem_459,
            getitem_460,
            getitem_461,
            getitem_462,
            getitem_463,
            getitem_464,
            getitem_465,
            getitem_466,
            getitem_467,
            getitem_468,
            getitem_469,
            getitem_470,
            getitem_471,
            getitem_472,
            getitem_473,
        ],
        1.0,
    )
    getitem_395 = (
        getitem_396
    ) = (
        getitem_397
    ) = (
        getitem_398
    ) = (
        getitem_399
    ) = (
        getitem_400
    ) = (
        getitem_401
    ) = (
        getitem_402
    ) = (
        getitem_403
    ) = (
        getitem_404
    ) = (
        getitem_405
    ) = (
        getitem_406
    ) = (
        getitem_407
    ) = (
        getitem_408
    ) = (
        getitem_409
    ) = (
        getitem_410
    ) = (
        getitem_411
    ) = (
        getitem_412
    ) = (
        getitem_413
    ) = (
        getitem_414
    ) = (
        getitem_415
    ) = (
        getitem_416
    ) = (
        getitem_417
    ) = (
        getitem_418
    ) = (
        getitem_419
    ) = (
        getitem_420
    ) = (
        getitem_421
    ) = (
        getitem_422
    ) = (
        getitem_423
    ) = (
        getitem_424
    ) = (
        getitem_425
    ) = (
        getitem_426
    ) = (
        getitem_427
    ) = (
        getitem_428
    ) = (
        getitem_429
    ) = (
        getitem_430
    ) = (
        getitem_431
    ) = (
        getitem_432
    ) = (
        getitem_433
    ) = (
        getitem_434
    ) = (
        getitem_435
    ) = (
        getitem_436
    ) = (
        getitem_437
    ) = (
        getitem_438
    ) = (
        getitem_439
    ) = (
        getitem_440
    ) = (
        getitem_441
    ) = (
        getitem_442
    ) = (
        getitem_443
    ) = (
        getitem_444
    ) = (
        getitem_445
    ) = (
        getitem_446
    ) = (
        getitem_447
    ) = (
        getitem_448
    ) = (
        getitem_449
    ) = (
        getitem_450
    ) = (
        getitem_451
    ) = (
        getitem_452
    ) = (
        getitem_453
    ) = (
        getitem_454
    ) = (
        getitem_455
    ) = (
        getitem_456
    ) = (
        getitem_457
    ) = (
        getitem_458
    ) = (
        getitem_459
    ) = (
        getitem_460
    ) = (
        getitem_461
    ) = (
        getitem_462
    ) = (
        getitem_463
    ) = (
        getitem_464
    ) = (
        getitem_465
    ) = (
        getitem_466
    ) = (
        getitem_467
    ) = (
        getitem_468
    ) = getitem_469 = getitem_470 = getitem_471 = getitem_472 = getitem_473 = None
    getitem_474: "f32[][]cuda:0" = _foreach_maximum[0]
    getitem_475: "f32[][]cuda:0" = _foreach_maximum[1]
    getitem_476: "f32[][]cuda:0" = _foreach_maximum[2]
    getitem_477: "f32[][]cuda:0" = _foreach_maximum[3]
    getitem_478: "f32[][]cuda:0" = _foreach_maximum[4]
    getitem_479: "f32[][]cuda:0" = _foreach_maximum[5]
    getitem_480: "f32[][]cuda:0" = _foreach_maximum[6]
    getitem_481: "f32[][]cuda:0" = _foreach_maximum[7]
    getitem_482: "f32[][]cuda:0" = _foreach_maximum[8]
    getitem_483: "f32[][]cuda:0" = _foreach_maximum[9]
    getitem_484: "f32[][]cuda:0" = _foreach_maximum[10]
    getitem_485: "f32[][]cuda:0" = _foreach_maximum[11]
    getitem_486: "f32[][]cuda:0" = _foreach_maximum[12]
    getitem_487: "f32[][]cuda:0" = _foreach_maximum[13]
    getitem_488: "f32[][]cuda:0" = _foreach_maximum[14]
    getitem_489: "f32[][]cuda:0" = _foreach_maximum[15]
    getitem_490: "f32[][]cuda:0" = _foreach_maximum[16]
    getitem_491: "f32[][]cuda:0" = _foreach_maximum[17]
    getitem_492: "f32[][]cuda:0" = _foreach_maximum[18]
    getitem_493: "f32[][]cuda:0" = _foreach_maximum[19]
    getitem_494: "f32[][]cuda:0" = _foreach_maximum[20]
    getitem_495: "f32[][]cuda:0" = _foreach_maximum[21]
    getitem_496: "f32[][]cuda:0" = _foreach_maximum[22]
    getitem_497: "f32[][]cuda:0" = _foreach_maximum[23]
    getitem_498: "f32[][]cuda:0" = _foreach_maximum[24]
    getitem_499: "f32[][]cuda:0" = _foreach_maximum[25]
    getitem_500: "f32[][]cuda:0" = _foreach_maximum[26]
    getitem_501: "f32[][]cuda:0" = _foreach_maximum[27]
    getitem_502: "f32[][]cuda:0" = _foreach_maximum[28]
    getitem_503: "f32[][]cuda:0" = _foreach_maximum[29]
    getitem_504: "f32[][]cuda:0" = _foreach_maximum[30]
    getitem_505: "f32[][]cuda:0" = _foreach_maximum[31]
    getitem_506: "f32[][]cuda:0" = _foreach_maximum[32]
    getitem_507: "f32[][]cuda:0" = _foreach_maximum[33]
    getitem_508: "f32[][]cuda:0" = _foreach_maximum[34]
    getitem_509: "f32[][]cuda:0" = _foreach_maximum[35]
    getitem_510: "f32[][]cuda:0" = _foreach_maximum[36]
    getitem_511: "f32[][]cuda:0" = _foreach_maximum[37]
    getitem_512: "f32[][]cuda:0" = _foreach_maximum[38]
    getitem_513: "f32[][]cuda:0" = _foreach_maximum[39]
    getitem_514: "f32[][]cuda:0" = _foreach_maximum[40]
    getitem_515: "f32[][]cuda:0" = _foreach_maximum[41]
    getitem_516: "f32[][]cuda:0" = _foreach_maximum[42]
    getitem_517: "f32[][]cuda:0" = _foreach_maximum[43]
    getitem_518: "f32[][]cuda:0" = _foreach_maximum[44]
    getitem_519: "f32[][]cuda:0" = _foreach_maximum[45]
    getitem_520: "f32[][]cuda:0" = _foreach_maximum[46]
    getitem_521: "f32[][]cuda:0" = _foreach_maximum[47]
    getitem_522: "f32[][]cuda:0" = _foreach_maximum[48]
    getitem_523: "f32[][]cuda:0" = _foreach_maximum[49]
    getitem_524: "f32[][]cuda:0" = _foreach_maximum[50]
    getitem_525: "f32[][]cuda:0" = _foreach_maximum[51]
    getitem_526: "f32[][]cuda:0" = _foreach_maximum[52]
    getitem_527: "f32[][]cuda:0" = _foreach_maximum[53]
    getitem_528: "f32[][]cuda:0" = _foreach_maximum[54]
    getitem_529: "f32[][]cuda:0" = _foreach_maximum[55]
    getitem_530: "f32[][]cuda:0" = _foreach_maximum[56]
    getitem_531: "f32[][]cuda:0" = _foreach_maximum[57]
    getitem_532: "f32[][]cuda:0" = _foreach_maximum[58]
    getitem_533: "f32[][]cuda:0" = _foreach_maximum[59]
    getitem_534: "f32[][]cuda:0" = _foreach_maximum[60]
    getitem_535: "f32[][]cuda:0" = _foreach_maximum[61]
    getitem_536: "f32[][]cuda:0" = _foreach_maximum[62]
    getitem_537: "f32[][]cuda:0" = _foreach_maximum[63]
    getitem_538: "f32[][]cuda:0" = _foreach_maximum[64]
    getitem_539: "f32[][]cuda:0" = _foreach_maximum[65]
    getitem_540: "f32[][]cuda:0" = _foreach_maximum[66]
    getitem_541: "f32[][]cuda:0" = _foreach_maximum[67]
    getitem_542: "f32[][]cuda:0" = _foreach_maximum[68]
    getitem_543: "f32[][]cuda:0" = _foreach_maximum[69]
    getitem_544: "f32[][]cuda:0" = _foreach_maximum[70]
    getitem_545: "f32[][]cuda:0" = _foreach_maximum[71]
    getitem_546: "f32[][]cuda:0" = _foreach_maximum[72]
    getitem_547: "f32[][]cuda:0" = _foreach_maximum[73]
    getitem_548: "f32[][]cuda:0" = _foreach_maximum[74]
    getitem_549: "f32[][]cuda:0" = _foreach_maximum[75]
    getitem_550: "f32[][]cuda:0" = _foreach_maximum[76]
    getitem_551: "f32[][]cuda:0" = _foreach_maximum[77]
    getitem_552: "f32[][]cuda:0" = _foreach_maximum[78]
    _foreach_maximum = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:242 in _compute_clippy_shrinkage, code: (alphas).repeat(len(masked_blocked_params)),
    repeat: "f32[79][1]cuda:0" = torch.ops.aten.repeat.default(neg, [79])
    neg = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:241 in _compute_clippy_shrinkage, code: minus_lrs = torch.split(
    split = torch.ops.aten.split.Tensor(repeat, 1)
    getitem_553: "f32[1][1]cuda:0" = split[0]
    getitem_554: "f32[1][1]cuda:0" = split[1]
    getitem_555: "f32[1][1]cuda:0" = split[2]
    getitem_556: "f32[1][1]cuda:0" = split[3]
    getitem_557: "f32[1][1]cuda:0" = split[4]
    getitem_558: "f32[1][1]cuda:0" = split[5]
    getitem_559: "f32[1][1]cuda:0" = split[6]
    getitem_560: "f32[1][1]cuda:0" = split[7]
    getitem_561: "f32[1][1]cuda:0" = split[8]
    getitem_562: "f32[1][1]cuda:0" = split[9]
    getitem_563: "f32[1][1]cuda:0" = split[10]
    getitem_564: "f32[1][1]cuda:0" = split[11]
    getitem_565: "f32[1][1]cuda:0" = split[12]
    getitem_566: "f32[1][1]cuda:0" = split[13]
    getitem_567: "f32[1][1]cuda:0" = split[14]
    getitem_568: "f32[1][1]cuda:0" = split[15]
    getitem_569: "f32[1][1]cuda:0" = split[16]
    getitem_570: "f32[1][1]cuda:0" = split[17]
    getitem_571: "f32[1][1]cuda:0" = split[18]
    getitem_572: "f32[1][1]cuda:0" = split[19]
    getitem_573: "f32[1][1]cuda:0" = split[20]
    getitem_574: "f32[1][1]cuda:0" = split[21]
    getitem_575: "f32[1][1]cuda:0" = split[22]
    getitem_576: "f32[1][1]cuda:0" = split[23]
    getitem_577: "f32[1][1]cuda:0" = split[24]
    getitem_578: "f32[1][1]cuda:0" = split[25]
    getitem_579: "f32[1][1]cuda:0" = split[26]
    getitem_580: "f32[1][1]cuda:0" = split[27]
    getitem_581: "f32[1][1]cuda:0" = split[28]
    getitem_582: "f32[1][1]cuda:0" = split[29]
    getitem_583: "f32[1][1]cuda:0" = split[30]
    getitem_584: "f32[1][1]cuda:0" = split[31]
    getitem_585: "f32[1][1]cuda:0" = split[32]
    getitem_586: "f32[1][1]cuda:0" = split[33]
    getitem_587: "f32[1][1]cuda:0" = split[34]
    getitem_588: "f32[1][1]cuda:0" = split[35]
    getitem_589: "f32[1][1]cuda:0" = split[36]
    getitem_590: "f32[1][1]cuda:0" = split[37]
    getitem_591: "f32[1][1]cuda:0" = split[38]
    getitem_592: "f32[1][1]cuda:0" = split[39]
    getitem_593: "f32[1][1]cuda:0" = split[40]
    getitem_594: "f32[1][1]cuda:0" = split[41]
    getitem_595: "f32[1][1]cuda:0" = split[42]
    getitem_596: "f32[1][1]cuda:0" = split[43]
    getitem_597: "f32[1][1]cuda:0" = split[44]
    getitem_598: "f32[1][1]cuda:0" = split[45]
    getitem_599: "f32[1][1]cuda:0" = split[46]
    getitem_600: "f32[1][1]cuda:0" = split[47]
    getitem_601: "f32[1][1]cuda:0" = split[48]
    getitem_602: "f32[1][1]cuda:0" = split[49]
    getitem_603: "f32[1][1]cuda:0" = split[50]
    getitem_604: "f32[1][1]cuda:0" = split[51]
    getitem_605: "f32[1][1]cuda:0" = split[52]
    getitem_606: "f32[1][1]cuda:0" = split[53]
    getitem_607: "f32[1][1]cuda:0" = split[54]
    getitem_608: "f32[1][1]cuda:0" = split[55]
    getitem_609: "f32[1][1]cuda:0" = split[56]
    getitem_610: "f32[1][1]cuda:0" = split[57]
    getitem_611: "f32[1][1]cuda:0" = split[58]
    getitem_612: "f32[1][1]cuda:0" = split[59]
    getitem_613: "f32[1][1]cuda:0" = split[60]
    getitem_614: "f32[1][1]cuda:0" = split[61]
    getitem_615: "f32[1][1]cuda:0" = split[62]
    getitem_616: "f32[1][1]cuda:0" = split[63]
    getitem_617: "f32[1][1]cuda:0" = split[64]
    getitem_618: "f32[1][1]cuda:0" = split[65]
    getitem_619: "f32[1][1]cuda:0" = split[66]
    getitem_620: "f32[1][1]cuda:0" = split[67]
    getitem_621: "f32[1][1]cuda:0" = split[68]
    getitem_622: "f32[1][1]cuda:0" = split[69]
    getitem_623: "f32[1][1]cuda:0" = split[70]
    getitem_624: "f32[1][1]cuda:0" = split[71]
    getitem_625: "f32[1][1]cuda:0" = split[72]
    getitem_626: "f32[1][1]cuda:0" = split[73]
    getitem_627: "f32[1][1]cuda:0" = split[74]
    getitem_628: "f32[1][1]cuda:0" = split[75]
    getitem_629: "f32[1][1]cuda:0" = split[76]
    getitem_630: "f32[1][1]cuda:0" = split[77]
    getitem_631: "f32[1][1]cuda:0" = split[78]
    split = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:245 in _compute_clippy_shrinkage, code: torch._foreach_div_(minus_lrs, masked_blocked_shrinkage)
    _foreach_div_1 = torch.ops.aten._foreach_div.List(
        [
            getitem_553,
            getitem_554,
            getitem_555,
            getitem_556,
            getitem_557,
            getitem_558,
            getitem_559,
            getitem_560,
            getitem_561,
            getitem_562,
            getitem_563,
            getitem_564,
            getitem_565,
            getitem_566,
            getitem_567,
            getitem_568,
            getitem_569,
            getitem_570,
            getitem_571,
            getitem_572,
            getitem_573,
            getitem_574,
            getitem_575,
            getitem_576,
            getitem_577,
            getitem_578,
            getitem_579,
            getitem_580,
            getitem_581,
            getitem_582,
            getitem_583,
            getitem_584,
            getitem_585,
            getitem_586,
            getitem_587,
            getitem_588,
            getitem_589,
            getitem_590,
            getitem_591,
            getitem_592,
            getitem_593,
            getitem_594,
            getitem_595,
            getitem_596,
            getitem_597,
            getitem_598,
            getitem_599,
            getitem_600,
            getitem_601,
            getitem_602,
            getitem_603,
            getitem_604,
            getitem_605,
            getitem_606,
            getitem_607,
            getitem_608,
            getitem_609,
            getitem_610,
            getitem_611,
            getitem_612,
            getitem_613,
            getitem_614,
            getitem_615,
            getitem_616,
            getitem_617,
            getitem_618,
            getitem_619,
            getitem_620,
            getitem_621,
            getitem_622,
            getitem_623,
            getitem_624,
            getitem_625,
            getitem_626,
            getitem_627,
            getitem_628,
            getitem_629,
            getitem_630,
            getitem_631,
        ],
        [
            getitem_474,
            getitem_475,
            getitem_476,
            getitem_477,
            getitem_478,
            getitem_479,
            getitem_480,
            getitem_481,
            getitem_482,
            getitem_483,
            getitem_484,
            getitem_485,
            getitem_486,
            getitem_487,
            getitem_488,
            getitem_489,
            getitem_490,
            getitem_491,
            getitem_492,
            getitem_493,
            getitem_494,
            getitem_495,
            getitem_496,
            getitem_497,
            getitem_498,
            getitem_499,
            getitem_500,
            getitem_501,
            getitem_502,
            getitem_503,
            getitem_504,
            getitem_505,
            getitem_506,
            getitem_507,
            getitem_508,
            getitem_509,
            getitem_510,
            getitem_511,
            getitem_512,
            getitem_513,
            getitem_514,
            getitem_515,
            getitem_516,
            getitem_517,
            getitem_518,
            getitem_519,
            getitem_520,
            getitem_521,
            getitem_522,
            getitem_523,
            getitem_524,
            getitem_525,
            getitem_526,
            getitem_527,
            getitem_528,
            getitem_529,
            getitem_530,
            getitem_531,
            getitem_532,
            getitem_533,
            getitem_534,
            getitem_535,
            getitem_536,
            getitem_537,
            getitem_538,
            getitem_539,
            getitem_540,
            getitem_541,
            getitem_542,
            getitem_543,
            getitem_544,
            getitem_545,
            getitem_546,
            getitem_547,
            getitem_548,
            getitem_549,
            getitem_550,
            getitem_551,
            getitem_552,
        ],
    )
    getitem_553 = (
        getitem_554
    ) = (
        getitem_555
    ) = (
        getitem_556
    ) = (
        getitem_557
    ) = (
        getitem_558
    ) = (
        getitem_559
    ) = (
        getitem_560
    ) = (
        getitem_561
    ) = (
        getitem_562
    ) = (
        getitem_563
    ) = (
        getitem_564
    ) = (
        getitem_565
    ) = (
        getitem_566
    ) = (
        getitem_567
    ) = (
        getitem_568
    ) = (
        getitem_569
    ) = (
        getitem_570
    ) = (
        getitem_571
    ) = (
        getitem_572
    ) = (
        getitem_573
    ) = (
        getitem_574
    ) = (
        getitem_575
    ) = (
        getitem_576
    ) = (
        getitem_577
    ) = (
        getitem_578
    ) = (
        getitem_579
    ) = (
        getitem_580
    ) = (
        getitem_581
    ) = (
        getitem_582
    ) = (
        getitem_583
    ) = (
        getitem_584
    ) = (
        getitem_585
    ) = (
        getitem_586
    ) = (
        getitem_587
    ) = (
        getitem_588
    ) = (
        getitem_589
    ) = (
        getitem_590
    ) = (
        getitem_591
    ) = (
        getitem_592
    ) = (
        getitem_593
    ) = (
        getitem_594
    ) = (
        getitem_595
    ) = (
        getitem_596
    ) = (
        getitem_597
    ) = (
        getitem_598
    ) = (
        getitem_599
    ) = (
        getitem_600
    ) = (
        getitem_601
    ) = (
        getitem_602
    ) = (
        getitem_603
    ) = (
        getitem_604
    ) = (
        getitem_605
    ) = (
        getitem_606
    ) = (
        getitem_607
    ) = (
        getitem_608
    ) = (
        getitem_609
    ) = (
        getitem_610
    ) = (
        getitem_611
    ) = (
        getitem_612
    ) = (
        getitem_613
    ) = (
        getitem_614
    ) = (
        getitem_615
    ) = (
        getitem_616
    ) = (
        getitem_617
    ) = (
        getitem_618
    ) = (
        getitem_619
    ) = (
        getitem_620
    ) = (
        getitem_621
    ) = (
        getitem_622
    ) = (
        getitem_623
    ) = (
        getitem_624
    ) = (
        getitem_625
    ) = (
        getitem_626
    ) = (
        getitem_627
    ) = (
        getitem_628
    ) = (
        getitem_629
    ) = (
        getitem_630
    ) = (
        getitem_631
    ) = (
        getitem_474
    ) = (
        getitem_475
    ) = (
        getitem_476
    ) = (
        getitem_477
    ) = (
        getitem_478
    ) = (
        getitem_479
    ) = (
        getitem_480
    ) = (
        getitem_481
    ) = (
        getitem_482
    ) = (
        getitem_483
    ) = (
        getitem_484
    ) = (
        getitem_485
    ) = (
        getitem_486
    ) = (
        getitem_487
    ) = (
        getitem_488
    ) = (
        getitem_489
    ) = (
        getitem_490
    ) = (
        getitem_491
    ) = (
        getitem_492
    ) = (
        getitem_493
    ) = (
        getitem_494
    ) = (
        getitem_495
    ) = (
        getitem_496
    ) = (
        getitem_497
    ) = (
        getitem_498
    ) = (
        getitem_499
    ) = (
        getitem_500
    ) = (
        getitem_501
    ) = (
        getitem_502
    ) = (
        getitem_503
    ) = (
        getitem_504
    ) = (
        getitem_505
    ) = (
        getitem_506
    ) = (
        getitem_507
    ) = (
        getitem_508
    ) = (
        getitem_509
    ) = (
        getitem_510
    ) = (
        getitem_511
    ) = (
        getitem_512
    ) = (
        getitem_513
    ) = (
        getitem_514
    ) = (
        getitem_515
    ) = (
        getitem_516
    ) = (
        getitem_517
    ) = (
        getitem_518
    ) = (
        getitem_519
    ) = (
        getitem_520
    ) = (
        getitem_521
    ) = (
        getitem_522
    ) = (
        getitem_523
    ) = (
        getitem_524
    ) = (
        getitem_525
    ) = (
        getitem_526
    ) = (
        getitem_527
    ) = (
        getitem_528
    ) = (
        getitem_529
    ) = (
        getitem_530
    ) = (
        getitem_531
    ) = (
        getitem_532
    ) = (
        getitem_533
    ) = (
        getitem_534
    ) = (
        getitem_535
    ) = (
        getitem_536
    ) = (
        getitem_537
    ) = (
        getitem_538
    ) = (
        getitem_539
    ) = (
        getitem_540
    ) = (
        getitem_541
    ) = (
        getitem_542
    ) = (
        getitem_543
    ) = (
        getitem_544
    ) = (
        getitem_545
    ) = (
        getitem_546
    ) = (
        getitem_547
    ) = getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = None
    getitem_632: "f32[1][1]cuda:0" = _foreach_div_1[0]
    getitem_633: "f32[1][1]cuda:0" = _foreach_div_1[1]
    getitem_634: "f32[1][1]cuda:0" = _foreach_div_1[2]
    getitem_635: "f32[1][1]cuda:0" = _foreach_div_1[3]
    getitem_636: "f32[1][1]cuda:0" = _foreach_div_1[4]
    getitem_637: "f32[1][1]cuda:0" = _foreach_div_1[5]
    getitem_638: "f32[1][1]cuda:0" = _foreach_div_1[6]
    getitem_639: "f32[1][1]cuda:0" = _foreach_div_1[7]
    getitem_640: "f32[1][1]cuda:0" = _foreach_div_1[8]
    getitem_641: "f32[1][1]cuda:0" = _foreach_div_1[9]
    getitem_642: "f32[1][1]cuda:0" = _foreach_div_1[10]
    getitem_643: "f32[1][1]cuda:0" = _foreach_div_1[11]
    getitem_644: "f32[1][1]cuda:0" = _foreach_div_1[12]
    getitem_645: "f32[1][1]cuda:0" = _foreach_div_1[13]
    getitem_646: "f32[1][1]cuda:0" = _foreach_div_1[14]
    getitem_647: "f32[1][1]cuda:0" = _foreach_div_1[15]
    getitem_648: "f32[1][1]cuda:0" = _foreach_div_1[16]
    getitem_649: "f32[1][1]cuda:0" = _foreach_div_1[17]
    getitem_650: "f32[1][1]cuda:0" = _foreach_div_1[18]
    getitem_651: "f32[1][1]cuda:0" = _foreach_div_1[19]
    getitem_652: "f32[1][1]cuda:0" = _foreach_div_1[20]
    getitem_653: "f32[1][1]cuda:0" = _foreach_div_1[21]
    getitem_654: "f32[1][1]cuda:0" = _foreach_div_1[22]
    getitem_655: "f32[1][1]cuda:0" = _foreach_div_1[23]
    getitem_656: "f32[1][1]cuda:0" = _foreach_div_1[24]
    getitem_657: "f32[1][1]cuda:0" = _foreach_div_1[25]
    getitem_658: "f32[1][1]cuda:0" = _foreach_div_1[26]
    getitem_659: "f32[1][1]cuda:0" = _foreach_div_1[27]
    getitem_660: "f32[1][1]cuda:0" = _foreach_div_1[28]
    getitem_661: "f32[1][1]cuda:0" = _foreach_div_1[29]
    getitem_662: "f32[1][1]cuda:0" = _foreach_div_1[30]
    getitem_663: "f32[1][1]cuda:0" = _foreach_div_1[31]
    getitem_664: "f32[1][1]cuda:0" = _foreach_div_1[32]
    getitem_665: "f32[1][1]cuda:0" = _foreach_div_1[33]
    getitem_666: "f32[1][1]cuda:0" = _foreach_div_1[34]
    getitem_667: "f32[1][1]cuda:0" = _foreach_div_1[35]
    getitem_668: "f32[1][1]cuda:0" = _foreach_div_1[36]
    getitem_669: "f32[1][1]cuda:0" = _foreach_div_1[37]
    getitem_670: "f32[1][1]cuda:0" = _foreach_div_1[38]
    getitem_671: "f32[1][1]cuda:0" = _foreach_div_1[39]
    getitem_672: "f32[1][1]cuda:0" = _foreach_div_1[40]
    getitem_673: "f32[1][1]cuda:0" = _foreach_div_1[41]
    getitem_674: "f32[1][1]cuda:0" = _foreach_div_1[42]
    getitem_675: "f32[1][1]cuda:0" = _foreach_div_1[43]
    getitem_676: "f32[1][1]cuda:0" = _foreach_div_1[44]
    getitem_677: "f32[1][1]cuda:0" = _foreach_div_1[45]
    getitem_678: "f32[1][1]cuda:0" = _foreach_div_1[46]
    getitem_679: "f32[1][1]cuda:0" = _foreach_div_1[47]
    getitem_680: "f32[1][1]cuda:0" = _foreach_div_1[48]
    getitem_681: "f32[1][1]cuda:0" = _foreach_div_1[49]
    getitem_682: "f32[1][1]cuda:0" = _foreach_div_1[50]
    getitem_683: "f32[1][1]cuda:0" = _foreach_div_1[51]
    getitem_684: "f32[1][1]cuda:0" = _foreach_div_1[52]
    getitem_685: "f32[1][1]cuda:0" = _foreach_div_1[53]
    getitem_686: "f32[1][1]cuda:0" = _foreach_div_1[54]
    getitem_687: "f32[1][1]cuda:0" = _foreach_div_1[55]
    getitem_688: "f32[1][1]cuda:0" = _foreach_div_1[56]
    getitem_689: "f32[1][1]cuda:0" = _foreach_div_1[57]
    getitem_690: "f32[1][1]cuda:0" = _foreach_div_1[58]
    getitem_691: "f32[1][1]cuda:0" = _foreach_div_1[59]
    getitem_692: "f32[1][1]cuda:0" = _foreach_div_1[60]
    getitem_693: "f32[1][1]cuda:0" = _foreach_div_1[61]
    getitem_694: "f32[1][1]cuda:0" = _foreach_div_1[62]
    getitem_695: "f32[1][1]cuda:0" = _foreach_div_1[63]
    getitem_696: "f32[1][1]cuda:0" = _foreach_div_1[64]
    getitem_697: "f32[1][1]cuda:0" = _foreach_div_1[65]
    getitem_698: "f32[1][1]cuda:0" = _foreach_div_1[66]
    getitem_699: "f32[1][1]cuda:0" = _foreach_div_1[67]
    getitem_700: "f32[1][1]cuda:0" = _foreach_div_1[68]
    getitem_701: "f32[1][1]cuda:0" = _foreach_div_1[69]
    getitem_702: "f32[1][1]cuda:0" = _foreach_div_1[70]
    getitem_703: "f32[1][1]cuda:0" = _foreach_div_1[71]
    getitem_704: "f32[1][1]cuda:0" = _foreach_div_1[72]
    getitem_705: "f32[1][1]cuda:0" = _foreach_div_1[73]
    getitem_706: "f32[1][1]cuda:0" = _foreach_div_1[74]
    getitem_707: "f32[1][1]cuda:0" = _foreach_div_1[75]
    getitem_708: "f32[1][1]cuda:0" = _foreach_div_1[76]
    getitem_709: "f32[1][1]cuda:0" = _foreach_div_1[77]
    getitem_710: "f32[1][1]cuda:0" = _foreach_div_1[78]
    _foreach_div_1 = None
    slice_scatter: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        repeat, getitem_632, 0, 0, 1
    )
    repeat = getitem_632 = None
    slice_scatter_1: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter, getitem_633, 0, 1, 2
    )
    slice_scatter = getitem_633 = None
    slice_scatter_2: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_1, getitem_634, 0, 2, 3
    )
    slice_scatter_1 = getitem_634 = None
    slice_scatter_3: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_2, getitem_635, 0, 3, 4
    )
    slice_scatter_2 = getitem_635 = None
    slice_scatter_4: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_3, getitem_636, 0, 4, 5
    )
    slice_scatter_3 = getitem_636 = None
    slice_scatter_5: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_4, getitem_637, 0, 5, 6
    )
    slice_scatter_4 = getitem_637 = None
    slice_scatter_6: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_5, getitem_638, 0, 6, 7
    )
    slice_scatter_5 = getitem_638 = None
    slice_scatter_7: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_6, getitem_639, 0, 7, 8
    )
    slice_scatter_6 = getitem_639 = None
    slice_scatter_8: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_7, getitem_640, 0, 8, 9
    )
    slice_scatter_7 = getitem_640 = None
    slice_scatter_9: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_8, getitem_641, 0, 9, 10
    )
    slice_scatter_8 = getitem_641 = None
    slice_scatter_10: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_9, getitem_642, 0, 10, 11
    )
    slice_scatter_9 = getitem_642 = None
    slice_scatter_11: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_10, getitem_643, 0, 11, 12
    )
    slice_scatter_10 = getitem_643 = None
    slice_scatter_12: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_11, getitem_644, 0, 12, 13
    )
    slice_scatter_11 = getitem_644 = None
    slice_scatter_13: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_12, getitem_645, 0, 13, 14
    )
    slice_scatter_12 = getitem_645 = None
    slice_scatter_14: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_13, getitem_646, 0, 14, 15
    )
    slice_scatter_13 = getitem_646 = None
    slice_scatter_15: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_14, getitem_647, 0, 15, 16
    )
    slice_scatter_14 = getitem_647 = None
    slice_scatter_16: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_15, getitem_648, 0, 16, 17
    )
    slice_scatter_15 = getitem_648 = None
    slice_scatter_17: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_16, getitem_649, 0, 17, 18
    )
    slice_scatter_16 = getitem_649 = None
    slice_scatter_18: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_17, getitem_650, 0, 18, 19
    )
    slice_scatter_17 = getitem_650 = None
    slice_scatter_19: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_18, getitem_651, 0, 19, 20
    )
    slice_scatter_18 = getitem_651 = None
    slice_scatter_20: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_19, getitem_652, 0, 20, 21
    )
    slice_scatter_19 = getitem_652 = None
    slice_scatter_21: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_20, getitem_653, 0, 21, 22
    )
    slice_scatter_20 = getitem_653 = None
    slice_scatter_22: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_21, getitem_654, 0, 22, 23
    )
    slice_scatter_21 = getitem_654 = None
    slice_scatter_23: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_22, getitem_655, 0, 23, 24
    )
    slice_scatter_22 = getitem_655 = None
    slice_scatter_24: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_23, getitem_656, 0, 24, 25
    )
    slice_scatter_23 = getitem_656 = None
    slice_scatter_25: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_24, getitem_657, 0, 25, 26
    )
    slice_scatter_24 = getitem_657 = None
    slice_scatter_26: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_25, getitem_658, 0, 26, 27
    )
    slice_scatter_25 = getitem_658 = None
    slice_scatter_27: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_26, getitem_659, 0, 27, 28
    )
    slice_scatter_26 = getitem_659 = None
    slice_scatter_28: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_27, getitem_660, 0, 28, 29
    )
    slice_scatter_27 = getitem_660 = None
    slice_scatter_29: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_28, getitem_661, 0, 29, 30
    )
    slice_scatter_28 = getitem_661 = None
    slice_scatter_30: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_29, getitem_662, 0, 30, 31
    )
    slice_scatter_29 = getitem_662 = None
    slice_scatter_31: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_30, getitem_663, 0, 31, 32
    )
    slice_scatter_30 = getitem_663 = None
    slice_scatter_32: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_31, getitem_664, 0, 32, 33
    )
    slice_scatter_31 = getitem_664 = None
    slice_scatter_33: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_32, getitem_665, 0, 33, 34
    )
    slice_scatter_32 = getitem_665 = None
    slice_scatter_34: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_33, getitem_666, 0, 34, 35
    )
    slice_scatter_33 = getitem_666 = None
    slice_scatter_35: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_34, getitem_667, 0, 35, 36
    )
    slice_scatter_34 = getitem_667 = None
    slice_scatter_36: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_35, getitem_668, 0, 36, 37
    )
    slice_scatter_35 = getitem_668 = None
    slice_scatter_37: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_36, getitem_669, 0, 37, 38
    )
    slice_scatter_36 = getitem_669 = None
    slice_scatter_38: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_37, getitem_670, 0, 38, 39
    )
    slice_scatter_37 = getitem_670 = None
    slice_scatter_39: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_38, getitem_671, 0, 39, 40
    )
    slice_scatter_38 = getitem_671 = None
    slice_scatter_40: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_39, getitem_672, 0, 40, 41
    )
    slice_scatter_39 = getitem_672 = None
    slice_scatter_41: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_40, getitem_673, 0, 41, 42
    )
    slice_scatter_40 = getitem_673 = None
    slice_scatter_42: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_41, getitem_674, 0, 42, 43
    )
    slice_scatter_41 = getitem_674 = None
    slice_scatter_43: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_42, getitem_675, 0, 43, 44
    )
    slice_scatter_42 = getitem_675 = None
    slice_scatter_44: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_43, getitem_676, 0, 44, 45
    )
    slice_scatter_43 = getitem_676 = None
    slice_scatter_45: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_44, getitem_677, 0, 45, 46
    )
    slice_scatter_44 = getitem_677 = None
    slice_scatter_46: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_45, getitem_678, 0, 46, 47
    )
    slice_scatter_45 = getitem_678 = None
    slice_scatter_47: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_46, getitem_679, 0, 47, 48
    )
    slice_scatter_46 = getitem_679 = None
    slice_scatter_48: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_47, getitem_680, 0, 48, 49
    )
    slice_scatter_47 = getitem_680 = None
    slice_scatter_49: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_48, getitem_681, 0, 49, 50
    )
    slice_scatter_48 = getitem_681 = None
    slice_scatter_50: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_49, getitem_682, 0, 50, 51
    )
    slice_scatter_49 = getitem_682 = None
    slice_scatter_51: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_50, getitem_683, 0, 51, 52
    )
    slice_scatter_50 = getitem_683 = None
    slice_scatter_52: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_51, getitem_684, 0, 52, 53
    )
    slice_scatter_51 = getitem_684 = None
    slice_scatter_53: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_52, getitem_685, 0, 53, 54
    )
    slice_scatter_52 = getitem_685 = None
    slice_scatter_54: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_53, getitem_686, 0, 54, 55
    )
    slice_scatter_53 = getitem_686 = None
    slice_scatter_55: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_54, getitem_687, 0, 55, 56
    )
    slice_scatter_54 = getitem_687 = None
    slice_scatter_56: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_55, getitem_688, 0, 56, 57
    )
    slice_scatter_55 = getitem_688 = None
    slice_scatter_57: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_56, getitem_689, 0, 57, 58
    )
    slice_scatter_56 = getitem_689 = None
    slice_scatter_58: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_57, getitem_690, 0, 58, 59
    )
    slice_scatter_57 = getitem_690 = None
    slice_scatter_59: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_58, getitem_691, 0, 59, 60
    )
    slice_scatter_58 = getitem_691 = None
    slice_scatter_60: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_59, getitem_692, 0, 60, 61
    )
    slice_scatter_59 = getitem_692 = None
    slice_scatter_61: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_60, getitem_693, 0, 61, 62
    )
    slice_scatter_60 = getitem_693 = None
    slice_scatter_62: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_61, getitem_694, 0, 62, 63
    )
    slice_scatter_61 = getitem_694 = None
    slice_scatter_63: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_62, getitem_695, 0, 63, 64
    )
    slice_scatter_62 = getitem_695 = None
    slice_scatter_64: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_63, getitem_696, 0, 64, 65
    )
    slice_scatter_63 = getitem_696 = None
    slice_scatter_65: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_64, getitem_697, 0, 65, 66
    )
    slice_scatter_64 = getitem_697 = None
    slice_scatter_66: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_65, getitem_698, 0, 66, 67
    )
    slice_scatter_65 = getitem_698 = None
    slice_scatter_67: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_66, getitem_699, 0, 67, 68
    )
    slice_scatter_66 = getitem_699 = None
    slice_scatter_68: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_67, getitem_700, 0, 68, 69
    )
    slice_scatter_67 = getitem_700 = None
    slice_scatter_69: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_68, getitem_701, 0, 69, 70
    )
    slice_scatter_68 = getitem_701 = None
    slice_scatter_70: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_69, getitem_702, 0, 70, 71
    )
    slice_scatter_69 = getitem_702 = None
    slice_scatter_71: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_70, getitem_703, 0, 71, 72
    )
    slice_scatter_70 = getitem_703 = None
    slice_scatter_72: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_71, getitem_704, 0, 72, 73
    )
    slice_scatter_71 = getitem_704 = None
    slice_scatter_73: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_72, getitem_705, 0, 73, 74
    )
    slice_scatter_72 = getitem_705 = None
    slice_scatter_74: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_73, getitem_706, 0, 74, 75
    )
    slice_scatter_73 = getitem_706 = None
    slice_scatter_75: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_74, getitem_707, 0, 75, 76
    )
    slice_scatter_74 = getitem_707 = None
    slice_scatter_76: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_75, getitem_708, 0, 76, 77
    )
    slice_scatter_75 = getitem_708 = None
    slice_scatter_77: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_76, getitem_709, 0, 77, 78
    )
    slice_scatter_76 = getitem_709 = None
    slice_scatter_78: "f32[79][1]cuda:0" = torch.ops.aten.slice_scatter.default(
        slice_scatter_77, getitem_710, 0, 78, 79
    )
    slice_scatter_77 = getitem_710 = None
    split_1 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_711: "f32[1][1]cuda:0" = split_1[0]
    split_1 = None
    split_2 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_791: "f32[1][1]cuda:0" = split_2[1]
    split_2 = None
    split_3 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_871: "f32[1][1]cuda:0" = split_3[2]
    split_3 = None
    split_4 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_951: "f32[1][1]cuda:0" = split_4[3]
    split_4 = None
    split_5 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1031: "f32[1][1]cuda:0" = split_5[4]
    split_5 = None
    split_6 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1111: "f32[1][1]cuda:0" = split_6[5]
    split_6 = None
    split_7 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1191: "f32[1][1]cuda:0" = split_7[6]
    split_7 = None
    split_8 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1271: "f32[1][1]cuda:0" = split_8[7]
    split_8 = None
    split_9 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1351: "f32[1][1]cuda:0" = split_9[8]
    split_9 = None
    split_10 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1431: "f32[1][1]cuda:0" = split_10[9]
    split_10 = None
    split_11 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1511: "f32[1][1]cuda:0" = split_11[10]
    split_11 = None
    split_12 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1591: "f32[1][1]cuda:0" = split_12[11]
    split_12 = None
    split_13 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1671: "f32[1][1]cuda:0" = split_13[12]
    split_13 = None
    split_14 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1751: "f32[1][1]cuda:0" = split_14[13]
    split_14 = None
    split_15 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1831: "f32[1][1]cuda:0" = split_15[14]
    split_15 = None
    split_16 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1911: "f32[1][1]cuda:0" = split_16[15]
    split_16 = None
    split_17 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_1991: "f32[1][1]cuda:0" = split_17[16]
    split_17 = None
    split_18 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2071: "f32[1][1]cuda:0" = split_18[17]
    split_18 = None
    split_19 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2151: "f32[1][1]cuda:0" = split_19[18]
    split_19 = None
    split_20 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2231: "f32[1][1]cuda:0" = split_20[19]
    split_20 = None
    split_21 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2311: "f32[1][1]cuda:0" = split_21[20]
    split_21 = None
    split_22 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2391: "f32[1][1]cuda:0" = split_22[21]
    split_22 = None
    split_23 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2471: "f32[1][1]cuda:0" = split_23[22]
    split_23 = None
    split_24 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2551: "f32[1][1]cuda:0" = split_24[23]
    split_24 = None
    split_25 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2631: "f32[1][1]cuda:0" = split_25[24]
    split_25 = None
    split_26 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2711: "f32[1][1]cuda:0" = split_26[25]
    split_26 = None
    split_27 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2791: "f32[1][1]cuda:0" = split_27[26]
    split_27 = None
    split_28 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2871: "f32[1][1]cuda:0" = split_28[27]
    split_28 = None
    split_29 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_2951: "f32[1][1]cuda:0" = split_29[28]
    split_29 = None
    split_30 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3031: "f32[1][1]cuda:0" = split_30[29]
    split_30 = None
    split_31 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3111: "f32[1][1]cuda:0" = split_31[30]
    split_31 = None
    split_32 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3191: "f32[1][1]cuda:0" = split_32[31]
    split_32 = None
    split_33 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3271: "f32[1][1]cuda:0" = split_33[32]
    split_33 = None
    split_34 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3351: "f32[1][1]cuda:0" = split_34[33]
    split_34 = None
    split_35 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3431: "f32[1][1]cuda:0" = split_35[34]
    split_35 = None
    split_36 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3511: "f32[1][1]cuda:0" = split_36[35]
    split_36 = None
    split_37 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3591: "f32[1][1]cuda:0" = split_37[36]
    split_37 = None
    split_38 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3671: "f32[1][1]cuda:0" = split_38[37]
    split_38 = None
    split_39 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3751: "f32[1][1]cuda:0" = split_39[38]
    split_39 = None
    split_40 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3831: "f32[1][1]cuda:0" = split_40[39]
    split_40 = None
    split_41 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3911: "f32[1][1]cuda:0" = split_41[40]
    split_41 = None
    split_42 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_3991: "f32[1][1]cuda:0" = split_42[41]
    split_42 = None
    split_43 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4071: "f32[1][1]cuda:0" = split_43[42]
    split_43 = None
    split_44 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4151: "f32[1][1]cuda:0" = split_44[43]
    split_44 = None
    split_45 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4231: "f32[1][1]cuda:0" = split_45[44]
    split_45 = None
    split_46 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4311: "f32[1][1]cuda:0" = split_46[45]
    split_46 = None
    split_47 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4391: "f32[1][1]cuda:0" = split_47[46]
    split_47 = None
    split_48 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4471: "f32[1][1]cuda:0" = split_48[47]
    split_48 = None
    split_49 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4551: "f32[1][1]cuda:0" = split_49[48]
    split_49 = None
    split_50 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4631: "f32[1][1]cuda:0" = split_50[49]
    split_50 = None
    split_51 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4711: "f32[1][1]cuda:0" = split_51[50]
    split_51 = None
    split_52 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4791: "f32[1][1]cuda:0" = split_52[51]
    split_52 = None
    split_53 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4871: "f32[1][1]cuda:0" = split_53[52]
    split_53 = None
    split_54 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_4951: "f32[1][1]cuda:0" = split_54[53]
    split_54 = None
    split_55 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5031: "f32[1][1]cuda:0" = split_55[54]
    split_55 = None
    split_56 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5111: "f32[1][1]cuda:0" = split_56[55]
    split_56 = None
    split_57 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5191: "f32[1][1]cuda:0" = split_57[56]
    split_57 = None
    split_58 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5271: "f32[1][1]cuda:0" = split_58[57]
    split_58 = None
    split_59 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5351: "f32[1][1]cuda:0" = split_59[58]
    split_59 = None
    split_60 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5431: "f32[1][1]cuda:0" = split_60[59]
    split_60 = None
    split_61 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5511: "f32[1][1]cuda:0" = split_61[60]
    split_61 = None
    split_62 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5591: "f32[1][1]cuda:0" = split_62[61]
    split_62 = None
    split_63 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5671: "f32[1][1]cuda:0" = split_63[62]
    split_63 = None
    split_64 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5751: "f32[1][1]cuda:0" = split_64[63]
    split_64 = None
    split_65 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5831: "f32[1][1]cuda:0" = split_65[64]
    split_65 = None
    split_66 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5911: "f32[1][1]cuda:0" = split_66[65]
    split_66 = None
    split_67 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_5991: "f32[1][1]cuda:0" = split_67[66]
    split_67 = None
    split_68 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6071: "f32[1][1]cuda:0" = split_68[67]
    split_68 = None
    split_69 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6151: "f32[1][1]cuda:0" = split_69[68]
    split_69 = None
    split_70 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6231: "f32[1][1]cuda:0" = split_70[69]
    split_70 = None
    split_71 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6311: "f32[1][1]cuda:0" = split_71[70]
    split_71 = None
    split_72 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6391: "f32[1][1]cuda:0" = split_72[71]
    split_72 = None
    split_73 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6471: "f32[1][1]cuda:0" = split_73[72]
    split_73 = None
    split_74 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6551: "f32[1][1]cuda:0" = split_74[73]
    split_74 = None
    split_75 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6631: "f32[1][1]cuda:0" = split_75[74]
    split_75 = None
    split_76 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6711: "f32[1][1]cuda:0" = split_76[75]
    split_76 = None
    split_77 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6791: "f32[1][1]cuda:0" = split_77[76]
    split_77 = None
    split_78 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    getitem_6871: "f32[1][1]cuda:0" = split_78[77]
    split_78 = None
    split_79 = torch.ops.aten.split.Tensor(slice_scatter_78, 1)
    slice_scatter_78 = None
    getitem_6951: "f32[1][1]cuda:0" = split_79[78]
    split_79 = None

    # File: <torch_package_0>.hpc/optimizers/distributed_shampoo/prod/distributed_shampoo.py:824 in _apply_decoupled_weight_decay, code: torch._foreach_add_(
    _foreach_add_1 = torch.ops.aten._foreach_add.List(
        [
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            arg5_1,
            arg6_1,
            arg7_1,
            arg8_1,
            arg9_1,
            arg10_1,
            arg11_1,
            arg12_1,
            arg13_1,
            arg14_1,
            arg15_1,
            arg16_1,
            arg17_1,
            arg18_1,
            arg19_1,
            arg20_1,
            arg21_1,
            arg22_1,
            arg23_1,
            arg24_1,
            arg25_1,
            arg26_1,
            arg27_1,
            arg28_1,
            arg29_1,
            arg30_1,
            arg31_1,
            arg32_1,
            arg33_1,
            arg34_1,
            arg35_1,
            arg36_1,
            arg37_1,
            arg38_1,
            arg39_1,
            arg40_1,
            arg41_1,
            arg42_1,
            arg43_1,
            arg44_1,
            arg45_1,
            arg46_1,
            arg47_1,
            arg48_1,
            arg49_1,
            arg50_1,
            arg51_1,
            arg52_1,
            arg53_1,
            arg54_1,
            arg55_1,
            arg56_1,
            arg57_1,
            arg58_1,
            arg59_1,
            arg60_1,
            arg61_1,
            arg62_1,
            arg63_1,
            arg64_1,
            arg65_1,
            arg66_1,
            arg67_1,
            arg68_1,
            arg69_1,
            arg70_1,
            arg71_1,
            arg72_1,
            arg73_1,
            arg74_1,
            arg75_1,
            arg76_1,
            arg77_1,
            arg78_1,
            arg79_1,
        ],
        [
            arg80_1,
            arg81_1,
            arg82_1,
            arg83_1,
            arg84_1,
            arg85_1,
            arg86_1,
            arg87_1,
            arg88_1,
            arg89_1,
            arg90_1,
            arg91_1,
            arg92_1,
            arg93_1,
            arg94_1,
            arg95_1,
            arg96_1,
            arg97_1,
            arg98_1,
            arg99_1,
            arg100_1,
            arg101_1,
            arg102_1,
            arg103_1,
            arg104_1,
            arg105_1,
            arg106_1,
            arg107_1,
            arg108_1,
            arg109_1,
            arg110_1,
            arg111_1,
            arg112_1,
            arg113_1,
            arg114_1,
            arg115_1,
            arg116_1,
            arg117_1,
            arg118_1,
            arg119_1,
            arg120_1,
            arg121_1,
            arg122_1,
            arg123_1,
            arg124_1,
            arg125_1,
            arg126_1,
            arg127_1,
            arg128_1,
            arg129_1,
            arg130_1,
            arg131_1,
            arg132_1,
            arg133_1,
            arg134_1,
            arg135_1,
            arg136_1,
            arg137_1,
            arg138_1,
            arg139_1,
            arg140_1,
            arg141_1,
            arg142_1,
            arg143_1,
            arg144_1,
            arg145_1,
            arg146_1,
            arg147_1,
            arg148_1,
            arg149_1,
            arg150_1,
            arg151_1,
            arg152_1,
            arg153_1,
            arg154_1,
            arg155_1,
            arg156_1,
            arg157_1,
            arg158_1,
        ],
        alpha=1e-05,
    )
    arg80_1 = (
        arg81_1
    ) = (
        arg82_1
    ) = (
        arg83_1
    ) = (
        arg84_1
    ) = (
        arg85_1
    ) = (
        arg86_1
    ) = (
        arg87_1
    ) = (
        arg88_1
    ) = (
        arg89_1
    ) = (
        arg90_1
    ) = (
        arg91_1
    ) = (
        arg92_1
    ) = (
        arg93_1
    ) = (
        arg94_1
    ) = (
        arg95_1
    ) = (
        arg96_1
    ) = (
        arg97_1
    ) = (
        arg98_1
    ) = (
        arg99_1
    ) = (
        arg100_1
    ) = (
        arg101_1
    ) = (
        arg102_1
    ) = (
        arg103_1
    ) = (
        arg104_1
    ) = (
        arg105_1
    ) = (
        arg106_1
    ) = (
        arg107_1
    ) = (
        arg108_1
    ) = (
        arg109_1
    ) = (
        arg110_1
    ) = (
        arg111_1
    ) = (
        arg112_1
    ) = (
        arg113_1
    ) = (
        arg114_1
    ) = (
        arg115_1
    ) = (
        arg116_1
    ) = (
        arg117_1
    ) = (
        arg118_1
    ) = (
        arg119_1
    ) = (
        arg120_1
    ) = (
        arg121_1
    ) = (
        arg122_1
    ) = (
        arg123_1
    ) = (
        arg124_1
    ) = (
        arg125_1
    ) = (
        arg126_1
    ) = (
        arg127_1
    ) = (
        arg128_1
    ) = (
        arg129_1
    ) = (
        arg130_1
    ) = (
        arg131_1
    ) = (
        arg132_1
    ) = (
        arg133_1
    ) = (
        arg134_1
    ) = (
        arg135_1
    ) = (
        arg136_1
    ) = (
        arg137_1
    ) = (
        arg138_1
    ) = (
        arg139_1
    ) = (
        arg140_1
    ) = (
        arg141_1
    ) = (
        arg142_1
    ) = (
        arg143_1
    ) = (
        arg144_1
    ) = (
        arg145_1
    ) = (
        arg146_1
    ) = (
        arg147_1
    ) = (
        arg148_1
    ) = (
        arg149_1
    ) = (
        arg150_1
    ) = (
        arg151_1
    ) = (
        arg152_1
    ) = arg153_1 = arg154_1 = arg155_1 = arg156_1 = arg157_1 = arg158_1 = None
    getitem_6952: "f32[50][1]cuda:0" = _foreach_add_1[0]
    getitem_6953: "f32[23][1]cuda:0" = _foreach_add_1[1]
    getitem_6954: "f32[38][1]cuda:0" = _foreach_add_1[2]
    getitem_6955: "f32[5][1]cuda:0" = _foreach_add_1[3]
    getitem_6956: "f32[100][1]cuda:0" = _foreach_add_1[4]
    getitem_6957: "f32[50][1]cuda:0" = _foreach_add_1[5]
    getitem_6958: "f32[77][1]cuda:0" = _foreach_add_1[6]
    getitem_6959: "f32[100][1]cuda:0" = _foreach_add_1[7]
    getitem_6960: "f32[100][1]cuda:0" = _foreach_add_1[8]
    getitem_6961: "f32[96][1]cuda:0" = _foreach_add_1[9]
    getitem_6962: "f32[78][1]cuda:0" = _foreach_add_1[10]
    getitem_6963: "f32[100][1]cuda:0" = _foreach_add_1[11]
    getitem_6964: "f32[100][1]cuda:0" = _foreach_add_1[12]
    getitem_6965: "f32[97][1]cuda:0" = _foreach_add_1[13]
    getitem_6966: "f32[819, 732][732, 1]cuda:0" = _foreach_add_1[14]
    getitem_6967: "f32[204][1]cuda:0" = _foreach_add_1[15]
    getitem_6968: "f32[64][1]cuda:0" = _foreach_add_1[16]
    getitem_6969: "f32[204][1]cuda:0" = _foreach_add_1[17]
    getitem_6970: "f32[64, 204][204, 1]cuda:0" = _foreach_add_1[18]
    getitem_6971: "f32[204][1]cuda:0" = _foreach_add_1[19]
    getitem_6972: "f32[204, 160][160, 1]cuda:0" = _foreach_add_1[20]
    getitem_6973: "f32[204][1]cuda:0" = _foreach_add_1[21]
    getitem_6974: "f32[64][1]cuda:0" = _foreach_add_1[22]
    getitem_6975: "f32[204][1]cuda:0" = _foreach_add_1[23]
    getitem_6976: "f32[64, 204][204, 1]cuda:0" = _foreach_add_1[24]
    getitem_6977: "f32[204][1]cuda:0" = _foreach_add_1[25]
    getitem_6978: "f32[204][1]cuda:0" = _foreach_add_1[26]
    getitem_6979: "f32[64][1]cuda:0" = _foreach_add_1[27]
    getitem_6980: "f32[204][1]cuda:0" = _foreach_add_1[28]
    getitem_6981: "f32[64, 204][204, 1]cuda:0" = _foreach_add_1[29]
    getitem_6982: "f32[204][1]cuda:0" = _foreach_add_1[30]
    getitem_6983: "f32[204, 72][72, 1]cuda:0" = _foreach_add_1[31]
    getitem_6984: "f32[204][1]cuda:0" = _foreach_add_1[32]
    getitem_6985: "f32[64][1]cuda:0" = _foreach_add_1[33]
    getitem_6986: "f32[64, 204][204, 1]cuda:0" = _foreach_add_1[34]
    getitem_6987: "f32[768, 2675][2675, 1]cuda:0" = _foreach_add_1[35]
    getitem_6988: "f32[768, 2048][2048, 1]cuda:0" = _foreach_add_1[36]
    getitem_6989: "f32[768][1]cuda:0" = _foreach_add_1[37]
    getitem_6990: "f32[4096][1]cuda:0" = _foreach_add_1[38]
    getitem_6991: "f32[4096, 256][256, 1]cuda:0" = _foreach_add_1[39]
    getitem_6992: "f32[64][1]cuda:0" = _foreach_add_1[40]
    getitem_6993: "f32[2675][1]cuda:0" = _foreach_add_1[41]
    getitem_6994: "f32[1536, 4096][4096, 1]cuda:0" = _foreach_add_1[42]
    getitem_6995: "f32[4096][1]cuda:0" = _foreach_add_1[43]
    getitem_6996: "f32[1840][1]cuda:0" = _foreach_add_1[44]
    getitem_6997: "f32[2048, 2675][2675, 1]cuda:0" = _foreach_add_1[45]
    getitem_6998: "f32[2048][1]cuda:0" = _foreach_add_1[46]
    getitem_6999: "f32[2048][1]cuda:0" = _foreach_add_1[47]
    getitem_7000: "f32[768][1]cuda:0" = _foreach_add_1[48]
    getitem_7001: "f32[256][1]cuda:0" = _foreach_add_1[49]
    getitem_7002: "f32[768, 2048][2048, 1]cuda:0" = _foreach_add_1[50]
    getitem_7003: "f32[4096][1]cuda:0" = _foreach_add_1[51]
    getitem_7004: "f32[104][1]cuda:0" = _foreach_add_1[52]
    getitem_7005: "f32[768][1]cuda:0" = _foreach_add_1[53]
    getitem_7006: "f32[1024][1]cuda:0" = _foreach_add_1[54]
    getitem_7007: "f32[2048][1]cuda:0" = _foreach_add_1[55]
    getitem_7008: "f32[768, 2675][2675, 1]cuda:0" = _foreach_add_1[56]
    getitem_7009: "f32[2675][1]cuda:0" = _foreach_add_1[57]
    getitem_7010: "f32[256][1]cuda:0" = _foreach_add_1[58]
    getitem_7011: "f32[768][1]cuda:0" = _foreach_add_1[59]
    getitem_7012: "f32[256, 768][768, 1]cuda:0" = _foreach_add_1[60]
    getitem_7013: "f32[64][1]cuda:0" = _foreach_add_1[61]
    getitem_7014: "f32[1536][1]cuda:0" = _foreach_add_1[62]
    getitem_7015: "f32[2048][1]cuda:0" = _foreach_add_1[63]
    getitem_7016: "f32[3360][1]cuda:0" = _foreach_add_1[64]
    getitem_7017: "f32[768][1]cuda:0" = _foreach_add_1[65]
    getitem_7018: "f32[768, 2048][2048, 1]cuda:0" = _foreach_add_1[66]
    getitem_7019: "f32[256][1]cuda:0" = _foreach_add_1[67]
    getitem_7020: "f32[104, 256][256, 1]cuda:0" = _foreach_add_1[68]
    getitem_7021: "f32[2675][1]cuda:0" = _foreach_add_1[69]
    getitem_7022: "f32[768][1]cuda:0" = _foreach_add_1[70]
    getitem_7023: "f32[2048][1]cuda:0" = _foreach_add_1[71]
    getitem_7024: "f32[1024][1]cuda:0" = _foreach_add_1[72]
    getitem_7025: "f32[64, 612][612, 1]cuda:0" = _foreach_add_1[73]
    getitem_7026: "f32[128][1]cuda:0" = _foreach_add_1[74]
    getitem_7027: "f32[308, 256][256, 1]cuda:0" = _foreach_add_1[75]
    getitem_7028: "f32[1][1]cuda:0" = _foreach_add_1[76]
    getitem_7029: "f32[512][1]cuda:0" = _foreach_add_1[77]
    getitem_7030: "f32[512][1]cuda:0" = _foreach_add_1[78]
    _foreach_add_1 = None

    # File: <torch_package_0>.caffe2/torch/fb/optim/shampoo_wrapper.py:356 in torch_dynamo_resume_in__per_group_step_impl_at_316, code: torch._foreach_mul_(masked_blocked_search_directions, adjusted_lr)  # pyre-ignore [6]
    _foreach_mul_2 = torch.ops.aten._foreach_mul.List(
        [
            getitem_6952,
            getitem_6953,
            getitem_6954,
            getitem_6955,
            getitem_6956,
            getitem_6957,
            getitem_6958,
            getitem_6959,
            getitem_6960,
            getitem_6961,
            getitem_6962,
            getitem_6963,
            getitem_6964,
            getitem_6965,
            getitem_6966,
            getitem_6967,
            getitem_6968,
            getitem_6969,
            getitem_6970,
            getitem_6971,
            getitem_6972,
            getitem_6973,
            getitem_6974,
            getitem_6975,
            getitem_6976,
            getitem_6977,
            getitem_6978,
            getitem_6979,
            getitem_6980,
            getitem_6981,
            getitem_6982,
            getitem_6983,
            getitem_6984,
            getitem_6985,
            getitem_6986,
            getitem_6987,
            getitem_6988,
            getitem_6989,
            getitem_6990,
            getitem_6991,
            getitem_6992,
            getitem_6993,
            getitem_6994,
            getitem_6995,
            getitem_6996,
            getitem_6997,
            getitem_6998,
            getitem_6999,
            getitem_7000,
            getitem_7001,
            getitem_7002,
            getitem_7003,
            getitem_7004,
            getitem_7005,
            getitem_7006,
            getitem_7007,
            getitem_7008,
            getitem_7009,
            getitem_7010,
            getitem_7011,
            getitem_7012,
            getitem_7013,
            getitem_7014,
            getitem_7015,
            getitem_7016,
            getitem_7017,
            getitem_7018,
            getitem_7019,
            getitem_7020,
            getitem_7021,
            getitem_7022,
            getitem_7023,
            getitem_7024,
            getitem_7025,
            getitem_7026,
            getitem_7027,
            getitem_7028,
            getitem_7029,
            getitem_7030,
        ],
        [
            getitem_711,
            getitem_791,
            getitem_871,
            getitem_951,
            getitem_1031,
            getitem_1111,
            getitem_1191,
            getitem_1271,
            getitem_1351,
            getitem_1431,
            getitem_1511,
            getitem_1591,
            getitem_1671,
            getitem_1751,
            getitem_1831,
            getitem_1911,
            getitem_1991,
            getitem_2071,
            getitem_2151,
            getitem_2231,
            getitem_2311,
            getitem_2391,
            getitem_2471,
            getitem_2551,
            getitem_2631,
            getitem_2711,
            getitem_2791,
            getitem_2871,
            getitem_2951,
            getitem_3031,
            getitem_3111,
            getitem_3191,
            getitem_3271,
            getitem_3351,
            getitem_3431,
            getitem_3511,
            getitem_3591,
            getitem_3671,
            getitem_3751,
            getitem_3831,
            getitem_3911,
            getitem_3991,
            getitem_4071,
            getitem_4151,
            getitem_4231,
            getitem_4311,
            getitem_4391,
            getitem_4471,
            getitem_4551,
            getitem_4631,
            getitem_4711,
            getitem_4791,
            getitem_4871,
            getitem_4951,
            getitem_5031,
            getitem_5111,
            getitem_5191,
            getitem_5271,
            getitem_5351,
            getitem_5431,
            getitem_5511,
            getitem_5591,
            getitem_5671,
            getitem_5751,
            getitem_5831,
            getitem_5911,
            getitem_5991,
            getitem_6071,
            getitem_6151,
            getitem_6231,
            getitem_6311,
            getitem_6391,
            getitem_6471,
            getitem_6551,
            getitem_6631,
            getitem_6711,
            getitem_6791,
            getitem_6871,
            getitem_6951,
        ],
    )
    getitem_6952 = (
        getitem_6953
    ) = (
        getitem_6954
    ) = (
        getitem_6955
    ) = (
        getitem_6956
    ) = (
        getitem_6957
    ) = (
        getitem_6958
    ) = (
        getitem_6959
    ) = (
        getitem_6960
    ) = (
        getitem_6961
    ) = (
        getitem_6962
    ) = (
        getitem_6963
    ) = (
        getitem_6964
    ) = (
        getitem_6965
    ) = (
        getitem_6966
    ) = (
        getitem_6967
    ) = (
        getitem_6968
    ) = (
        getitem_6969
    ) = (
        getitem_6970
    ) = (
        getitem_6971
    ) = (
        getitem_6972
    ) = (
        getitem_6973
    ) = (
        getitem_6974
    ) = (
        getitem_6975
    ) = (
        getitem_6976
    ) = (
        getitem_6977
    ) = (
        getitem_6978
    ) = (
        getitem_6979
    ) = (
        getitem_6980
    ) = (
        getitem_6981
    ) = (
        getitem_6982
    ) = (
        getitem_6983
    ) = (
        getitem_6984
    ) = (
        getitem_6985
    ) = (
        getitem_6986
    ) = (
        getitem_6987
    ) = (
        getitem_6988
    ) = (
        getitem_6989
    ) = (
        getitem_6990
    ) = (
        getitem_6991
    ) = (
        getitem_6992
    ) = (
        getitem_6993
    ) = (
        getitem_6994
    ) = (
        getitem_6995
    ) = (
        getitem_6996
    ) = (
        getitem_6997
    ) = (
        getitem_6998
    ) = (
        getitem_6999
    ) = (
        getitem_7000
    ) = (
        getitem_7001
    ) = (
        getitem_7002
    ) = (
        getitem_7003
    ) = (
        getitem_7004
    ) = (
        getitem_7005
    ) = (
        getitem_7006
    ) = (
        getitem_7007
    ) = (
        getitem_7008
    ) = (
        getitem_7009
    ) = (
        getitem_7010
    ) = (
        getitem_7011
    ) = (
        getitem_7012
    ) = (
        getitem_7013
    ) = (
        getitem_7014
    ) = (
        getitem_7015
    ) = (
        getitem_7016
    ) = (
        getitem_7017
    ) = (
        getitem_7018
    ) = (
        getitem_7019
    ) = (
        getitem_7020
    ) = (
        getitem_7021
    ) = (
        getitem_7022
    ) = (
        getitem_7023
    ) = (
        getitem_7024
    ) = (
        getitem_7025
    ) = (
        getitem_7026
    ) = (
        getitem_7027
    ) = (
        getitem_7028
    ) = (
        getitem_7029
    ) = (
        getitem_7030
    ) = (
        getitem_711
    ) = (
        getitem_791
    ) = (
        getitem_871
    ) = (
        getitem_951
    ) = (
        getitem_1031
    ) = (
        getitem_1111
    ) = (
        getitem_1191
    ) = (
        getitem_1271
    ) = (
        getitem_1351
    ) = (
        getitem_1431
    ) = (
        getitem_1511
    ) = (
        getitem_1591
    ) = (
        getitem_1671
    ) = (
        getitem_1751
    ) = (
        getitem_1831
    ) = (
        getitem_1911
    ) = (
        getitem_1991
    ) = (
        getitem_2071
    ) = (
        getitem_2151
    ) = (
        getitem_2231
    ) = (
        getitem_2311
    ) = (
        getitem_2391
    ) = (
        getitem_2471
    ) = (
        getitem_2551
    ) = (
        getitem_2631
    ) = (
        getitem_2711
    ) = (
        getitem_2791
    ) = (
        getitem_2871
    ) = (
        getitem_2951
    ) = (
        getitem_3031
    ) = (
        getitem_3111
    ) = (
        getitem_3191
    ) = (
        getitem_3271
    ) = (
        getitem_3351
    ) = (
        getitem_3431
    ) = (
        getitem_3511
    ) = (
        getitem_3591
    ) = (
        getitem_3671
    ) = (
        getitem_3751
    ) = (
        getitem_3831
    ) = (
        getitem_3911
    ) = (
        getitem_3991
    ) = (
        getitem_4071
    ) = (
        getitem_4151
    ) = (
        getitem_4231
    ) = (
        getitem_4311
    ) = (
        getitem_4391
    ) = (
        getitem_4471
    ) = (
        getitem_4551
    ) = (
        getitem_4631
    ) = (
        getitem_4711
    ) = (
        getitem_4791
    ) = (
        getitem_4871
    ) = (
        getitem_4951
    ) = (
        getitem_5031
    ) = (
        getitem_5111
    ) = (
        getitem_5191
    ) = (
        getitem_5271
    ) = (
        getitem_5351
    ) = (
        getitem_5431
    ) = (
        getitem_5511
    ) = (
        getitem_5591
    ) = (
        getitem_5671
    ) = (
        getitem_5751
    ) = (
        getitem_5831
    ) = (
        getitem_5911
    ) = (
        getitem_5991
    ) = (
        getitem_6071
    ) = (
        getitem_6151
    ) = (
        getitem_6231
    ) = (
        getitem_6311
    ) = (
        getitem_6391
    ) = (
        getitem_6471
    ) = (
        getitem_6551
    ) = getitem_6631 = getitem_6711 = getitem_6791 = getitem_6871 = getitem_6951 = None
    getitem_7031: "f32[50][1]cuda:0" = _foreach_mul_2[0]
    getitem_7032: "f32[23][1]cuda:0" = _foreach_mul_2[1]
    getitem_7033: "f32[38][1]cuda:0" = _foreach_mul_2[2]
    getitem_7034: "f32[5][1]cuda:0" = _foreach_mul_2[3]
    getitem_7035: "f32[100][1]cuda:0" = _foreach_mul_2[4]
    getitem_7036: "f32[50][1]cuda:0" = _foreach_mul_2[5]
    getitem_7037: "f32[77][1]cuda:0" = _foreach_mul_2[6]
    getitem_7038: "f32[100][1]cuda:0" = _foreach_mul_2[7]
    getitem_7039: "f32[100][1]cuda:0" = _foreach_mul_2[8]
    getitem_7040: "f32[96][1]cuda:0" = _foreach_mul_2[9]
    getitem_7041: "f32[78][1]cuda:0" = _foreach_mul_2[10]
    getitem_7042: "f32[100][1]cuda:0" = _foreach_mul_2[11]
    getitem_7043: "f32[100][1]cuda:0" = _foreach_mul_2[12]
    getitem_7044: "f32[97][1]cuda:0" = _foreach_mul_2[13]
    getitem_7045: "f32[819, 732][732, 1]cuda:0" = _foreach_mul_2[14]
    getitem_7046: "f32[204][1]cuda:0" = _foreach_mul_2[15]
    getitem_7047: "f32[64][1]cuda:0" = _foreach_mul_2[16]
    getitem_7048: "f32[204][1]cuda:0" = _foreach_mul_2[17]
    getitem_7049: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_2[18]
    getitem_7050: "f32[204][1]cuda:0" = _foreach_mul_2[19]
    getitem_7051: "f32[204, 160][160, 1]cuda:0" = _foreach_mul_2[20]
    getitem_7052: "f32[204][1]cuda:0" = _foreach_mul_2[21]
    getitem_7053: "f32[64][1]cuda:0" = _foreach_mul_2[22]
    getitem_7054: "f32[204][1]cuda:0" = _foreach_mul_2[23]
    getitem_7055: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_2[24]
    getitem_7056: "f32[204][1]cuda:0" = _foreach_mul_2[25]
    getitem_7057: "f32[204][1]cuda:0" = _foreach_mul_2[26]
    getitem_7058: "f32[64][1]cuda:0" = _foreach_mul_2[27]
    getitem_7059: "f32[204][1]cuda:0" = _foreach_mul_2[28]
    getitem_7060: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_2[29]
    getitem_7061: "f32[204][1]cuda:0" = _foreach_mul_2[30]
    getitem_7062: "f32[204, 72][72, 1]cuda:0" = _foreach_mul_2[31]
    getitem_7063: "f32[204][1]cuda:0" = _foreach_mul_2[32]
    getitem_7064: "f32[64][1]cuda:0" = _foreach_mul_2[33]
    getitem_7065: "f32[64, 204][204, 1]cuda:0" = _foreach_mul_2[34]
    getitem_7066: "f32[768, 2675][2675, 1]cuda:0" = _foreach_mul_2[35]
    getitem_7067: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul_2[36]
    getitem_7068: "f32[768][1]cuda:0" = _foreach_mul_2[37]
    getitem_7069: "f32[4096][1]cuda:0" = _foreach_mul_2[38]
    getitem_7070: "f32[4096, 256][256, 1]cuda:0" = _foreach_mul_2[39]
    getitem_7071: "f32[64][1]cuda:0" = _foreach_mul_2[40]
    getitem_7072: "f32[2675][1]cuda:0" = _foreach_mul_2[41]
    getitem_7073: "f32[1536, 4096][4096, 1]cuda:0" = _foreach_mul_2[42]
    getitem_7074: "f32[4096][1]cuda:0" = _foreach_mul_2[43]
    getitem_7075: "f32[1840][1]cuda:0" = _foreach_mul_2[44]
    getitem_7076: "f32[2048, 2675][2675, 1]cuda:0" = _foreach_mul_2[45]
    getitem_7077: "f32[2048][1]cuda:0" = _foreach_mul_2[46]
    getitem_7078: "f32[2048][1]cuda:0" = _foreach_mul_2[47]
    getitem_7079: "f32[768][1]cuda:0" = _foreach_mul_2[48]
    getitem_7080: "f32[256][1]cuda:0" = _foreach_mul_2[49]
    getitem_7081: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul_2[50]
    getitem_7082: "f32[4096][1]cuda:0" = _foreach_mul_2[51]
    getitem_7083: "f32[104][1]cuda:0" = _foreach_mul_2[52]
    getitem_7084: "f32[768][1]cuda:0" = _foreach_mul_2[53]
    getitem_7085: "f32[1024][1]cuda:0" = _foreach_mul_2[54]
    getitem_7086: "f32[2048][1]cuda:0" = _foreach_mul_2[55]
    getitem_7087: "f32[768, 2675][2675, 1]cuda:0" = _foreach_mul_2[56]
    getitem_7088: "f32[2675][1]cuda:0" = _foreach_mul_2[57]
    getitem_7089: "f32[256][1]cuda:0" = _foreach_mul_2[58]
    getitem_7090: "f32[768][1]cuda:0" = _foreach_mul_2[59]
    getitem_7091: "f32[256, 768][768, 1]cuda:0" = _foreach_mul_2[60]
    getitem_7092: "f32[64][1]cuda:0" = _foreach_mul_2[61]
    getitem_7093: "f32[1536][1]cuda:0" = _foreach_mul_2[62]
    getitem_7094: "f32[2048][1]cuda:0" = _foreach_mul_2[63]
    getitem_7095: "f32[3360][1]cuda:0" = _foreach_mul_2[64]
    getitem_7096: "f32[768][1]cuda:0" = _foreach_mul_2[65]
    getitem_7097: "f32[768, 2048][2048, 1]cuda:0" = _foreach_mul_2[66]
    getitem_7098: "f32[256][1]cuda:0" = _foreach_mul_2[67]
    getitem_7099: "f32[104, 256][256, 1]cuda:0" = _foreach_mul_2[68]
    getitem_7100: "f32[2675][1]cuda:0" = _foreach_mul_2[69]
    getitem_7101: "f32[768][1]cuda:0" = _foreach_mul_2[70]
    getitem_7102: "f32[2048][1]cuda:0" = _foreach_mul_2[71]
    getitem_7103: "f32[1024][1]cuda:0" = _foreach_mul_2[72]
    getitem_7104: "f32[64, 612][612, 1]cuda:0" = _foreach_mul_2[73]
    getitem_7105: "f32[128][1]cuda:0" = _foreach_mul_2[74]
    getitem_7106: "f32[308, 256][256, 1]cuda:0" = _foreach_mul_2[75]
    getitem_7107: "f32[1][1]cuda:0" = _foreach_mul_2[76]
    getitem_7108: "f32[512][1]cuda:0" = _foreach_mul_2[77]
    getitem_7109: "f32[512][1]cuda:0" = _foreach_mul_2[78]
    _foreach_mul_2 = None
    copy_: "f32[50][1]cuda:0" = torch.ops.aten.copy_.default(arg1_1, getitem_7031)
    arg1_1 = getitem_7031 = None  #
    copy__1: "f32[23][1]cuda:0" = torch.ops.aten.copy_.default(arg2_1, getitem_7032)
    arg2_1 = getitem_7032 = None  #
    copy__2: "f32[38][1]cuda:0" = torch.ops.aten.copy_.default(arg3_1, getitem_7033)
    arg3_1 = getitem_7033 = None  #
    copy__3: "f32[5][1]cuda:0" = torch.ops.aten.copy_.default(arg4_1, getitem_7034)
    arg4_1 = getitem_7034 = None  #
    copy__4: "f32[100][1]cuda:0" = torch.ops.aten.copy_.default(arg5_1, getitem_7035)
    arg5_1 = getitem_7035 = None  #
    copy__5: "f32[50][1]cuda:0" = torch.ops.aten.copy_.default(arg6_1, getitem_7036)
    arg6_1 = getitem_7036 = None  #
    copy__6: "f32[77][1]cuda:0" = torch.ops.aten.copy_.default(arg7_1, getitem_7037)
    arg7_1 = getitem_7037 = None  #
    copy__7: "f32[100][1]cuda:0" = torch.ops.aten.copy_.default(arg8_1, getitem_7038)
    arg8_1 = getitem_7038 = None  #
    copy__8: "f32[100][1]cuda:0" = torch.ops.aten.copy_.default(arg9_1, getitem_7039)
    arg9_1 = getitem_7039 = None  #
    copy__9: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(arg10_1, getitem_7040)
    arg10_1 = getitem_7040 = None  #
    copy__10: "f32[78][1]cuda:0" = torch.ops.aten.copy_.default(arg11_1, getitem_7041)
    arg11_1 = getitem_7041 = None
    copy__11: "f32[100][1]cuda:0" = torch.ops.aten.copy_.default(arg12_1, getitem_7042)
    arg12_1 = getitem_7042 = None
    copy__12: "f32[100][1]cuda:0" = torch.ops.aten.copy_.default(arg13_1, getitem_7043)
    arg13_1 = getitem_7043 = None
    copy__13: "f32[97][1]cuda:0" = torch.ops.aten.copy_.default(arg14_1, getitem_7044)
    arg14_1 = getitem_7044 = None
    copy__14: "f32[819, 732][732, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg15_1, getitem_7045
    )
    arg15_1 = getitem_7045 = None
    copy__15: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg16_1, getitem_7046)
    arg16_1 = getitem_7046 = None
    copy__16: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(arg17_1, getitem_7047)
    arg17_1 = getitem_7047 = None
    copy__17: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg18_1, getitem_7048)
    arg18_1 = getitem_7048 = None
    copy__18: "f32[64, 204][204, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg19_1, getitem_7049
    )
    arg19_1 = getitem_7049 = None
    copy__19: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg20_1, getitem_7050)
    arg20_1 = getitem_7050 = None
    copy__20: "f32[204, 160][160, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg21_1, getitem_7051
    )
    arg21_1 = getitem_7051 = None
    copy__21: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg22_1, getitem_7052)
    arg22_1 = getitem_7052 = None
    copy__23: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg24_1, getitem_7054)
    arg24_1 = getitem_7054 = None
    copy__24: "f32[64, 204][204, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg25_1, getitem_7055
    )
    arg25_1 = getitem_7055 = None
    copy__25: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg26_1, getitem_7056)
    arg26_1 = getitem_7056 = None
    copy__26: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg27_1, getitem_7057)
    arg27_1 = getitem_7057 = None
    copy__27: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(arg28_1, getitem_7058)
    arg28_1 = getitem_7058 = None
    copy__28: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg29_1, getitem_7059)
    arg29_1 = getitem_7059 = None
    copy__29: "f32[64, 204][204, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg30_1, getitem_7060
    )
    arg30_1 = getitem_7060 = None
    copy__30: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg31_1, getitem_7061)
    arg31_1 = getitem_7061 = None
    copy__31: "f32[204, 72][72, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg32_1, getitem_7062
    )
    arg32_1 = getitem_7062 = None
    copy__32: "f32[204][1]cuda:0" = torch.ops.aten.copy_.default(arg33_1, getitem_7063)
    arg33_1 = getitem_7063 = None
    copy__33: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(arg34_1, getitem_7064)
    arg34_1 = getitem_7064 = None
    copy__34: "f32[64, 204][204, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg35_1, getitem_7065
    )
    arg35_1 = getitem_7065 = None
    copy__35: "f32[768, 2675][2675, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg36_1, getitem_7066
    )
    arg36_1 = getitem_7066 = None
    copy__36: "f32[768, 2048][2048, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg37_1, getitem_7067
    )
    arg37_1 = getitem_7067 = None
    copy__37: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(arg38_1, getitem_7068)
    arg38_1 = getitem_7068 = None
    copy__38: "f32[4096][1]cuda:0" = torch.ops.aten.copy_.default(arg39_1, getitem_7069)
    arg39_1 = getitem_7069 = None
    copy__39: "f32[4096, 256][256, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg40_1, getitem_7070
    )
    arg40_1 = getitem_7070 = None
    copy__40: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(arg41_1, getitem_7071)
    arg41_1 = getitem_7071 = None
    copy__41: "f32[2675][1]cuda:0" = torch.ops.aten.copy_.default(arg42_1, getitem_7072)
    arg42_1 = getitem_7072 = None
    copy__42: "f32[1536, 4096][4096, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg43_1, getitem_7073
    )
    arg43_1 = getitem_7073 = None
    copy__43: "f32[4096][1]cuda:0" = torch.ops.aten.copy_.default(arg44_1, getitem_7074)
    arg44_1 = getitem_7074 = None
    copy__44: "f32[1840][1]cuda:0" = torch.ops.aten.copy_.default(arg45_1, getitem_7075)
    arg45_1 = getitem_7075 = None
    copy__45: "f32[2048, 2675][2675, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg46_1, getitem_7076
    )
    arg46_1 = getitem_7076 = None
    copy__46: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(arg47_1, getitem_7077)
    arg47_1 = getitem_7077 = None
    copy__47: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(arg48_1, getitem_7078)
    arg48_1 = getitem_7078 = None
    copy__48: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(arg49_1, getitem_7079)
    arg49_1 = getitem_7079 = None
    copy__50: "f32[768, 2048][2048, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg51_1, getitem_7081
    )
    arg51_1 = getitem_7081 = None
    copy__51: "f32[4096][1]cuda:0" = torch.ops.aten.copy_.default(arg52_1, getitem_7082)
    arg52_1 = getitem_7082 = None
    copy__52: "f32[104][1]cuda:0" = torch.ops.aten.copy_.default(arg53_1, getitem_7083)
    arg53_1 = getitem_7083 = None
    copy__53: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(arg54_1, getitem_7084)
    arg54_1 = getitem_7084 = None
    copy__54: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(arg55_1, getitem_7085)
    arg55_1 = getitem_7085 = None
    copy__55: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(arg56_1, getitem_7086)
    arg56_1 = getitem_7086 = None
    copy__56: "f32[768, 2675][2675, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg57_1, getitem_7087
    )
    arg57_1 = getitem_7087 = None
    copy__57: "f32[2675][1]cuda:0" = torch.ops.aten.copy_.default(arg58_1, getitem_7088)
    arg58_1 = getitem_7088 = None
    copy__58: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(arg59_1, getitem_7089)
    arg59_1 = getitem_7089 = None
    copy__59: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(arg60_1, getitem_7090)
    arg60_1 = getitem_7090 = None
    copy__60: "f32[256, 768][768, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg61_1, getitem_7091
    )
    arg61_1 = getitem_7091 = None
    copy__61: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(arg62_1, getitem_7092)
    arg62_1 = getitem_7092 = None
    copy__62: "f32[1536][1]cuda:0" = torch.ops.aten.copy_.default(arg63_1, getitem_7093)
    arg63_1 = getitem_7093 = None
    copy__63: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(arg64_1, getitem_7094)
    arg64_1 = getitem_7094 = None
    copy__64: "f32[3360][1]cuda:0" = torch.ops.aten.copy_.default(arg65_1, getitem_7095)
    arg65_1 = getitem_7095 = None
    copy__65: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(arg66_1, getitem_7096)
    arg66_1 = getitem_7096 = None
    copy__66: "f32[768, 2048][2048, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg67_1, getitem_7097
    )
    arg67_1 = getitem_7097 = None
    copy__67: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(arg68_1, getitem_7098)
    arg68_1 = getitem_7098 = None
    copy__68: "f32[104, 256][256, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg69_1, getitem_7099
    )
    arg69_1 = getitem_7099 = None
    copy__69: "f32[2675][1]cuda:0" = torch.ops.aten.copy_.default(arg70_1, getitem_7100)
    arg70_1 = getitem_7100 = None
    copy__70: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(arg71_1, getitem_7101)
    arg71_1 = getitem_7101 = None
    copy__71: "f32[2048][1]cuda:0" = torch.ops.aten.copy_.default(arg72_1, getitem_7102)
    arg72_1 = getitem_7102 = None
    copy__72: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(arg73_1, getitem_7103)
    arg73_1 = getitem_7103 = None
    copy__73: "f32[64, 612][612, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg74_1, getitem_7104
    )
    arg74_1 = getitem_7104 = None
    copy__74: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(arg75_1, getitem_7105)
    arg75_1 = getitem_7105 = None
    copy__75: "f32[308, 256][256, 1]cuda:0" = torch.ops.aten.copy_.default(
        arg76_1, getitem_7106
    )
    arg76_1 = getitem_7106 = None
    copy__76: "f32[1][1]cuda:0" = torch.ops.aten.copy_.default(arg77_1, getitem_7107)
    arg77_1 = getitem_7107 = None
    copy__77: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(arg78_1, getitem_7108)
    arg78_1 = getitem_7108 = None
    copy__78: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(arg79_1, getitem_7109)
    arg79_1 = getitem_7109 = None
    return ()
