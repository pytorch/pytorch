import torch
import triton
import torch._dynamo
import torch._dynamo.config
import torch._inductor.config as config
from triton.ops.matmul import matmul
from benchmark_helper import time_with_torch_timer
config.max_autotune_gemm = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
# The flag below controls whether to allow GROUP_M to be 4 for inductor GEMMs.
# config.matmul_allow_group_m_of_4 = True



@torch._dynamo.optimize("inductor", nopython=True)
def inductor_aten_mm(a, b):
    return torch._int_mm(a, b)


@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_mm(a, b):
    return torch._int_mm(a, b)

@torch._dynamo.optimize("inductor", nopython=True)
def inductor_triton_bp_mm(a, b):
    return torch._int_mm(a, b)



def torch_mm(a, b):
    return torch._int_mm(a, b)


def triton_mm(a, b):
    return matmul(a, b)


def test_total_time(shapes):
    print("shape; torch int_mm; triton matmul; inductor aten int_mm; inductor triton int_mm; inductor triton bp int_mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randint(-128, 127, a_shape, device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 127, b_shape, device="cuda", dtype=a.dtype).t().contiguous().t()

        config.max_autotune_gemm_backends = "aten"
        inductor_aten_mm(a, b)

        config.max_autotune_gemm_backends = "triton"
        inductor_triton_mm(a, b)

        config.use_block_pointer_mm_kernel = True
        inductor_triton_bp_mm(a,b)


        torch_ms = time_with_torch_timer(torch_mm, (a, b)).mean * 1000

        triton_ms = time_with_torch_timer(triton_mm, (a, b)).mean * 1000

        config.max_autotune_gemm_backends = "aten"
        ind_aten_ms = time_with_torch_timer(inductor_aten_mm, (a, b)).mean * 1000

        config.max_autotune_gemm_backends = "triton"
        ind_triton_ms = time_with_torch_timer(inductor_triton_mm, (a, b)).mean * 1000
        ind_triton_bp_ms = time_with_torch_timer(inductor_triton_bp_mm, (a, b)).mean * 1000


        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, ind_triton_bp_ms, sep="; ")

        torch._dynamo.reset()


def test_GPU_time(shapes):
    print("shape; torch int_mm; triton matmul; inductor aten int_mm; inductor triton int_mm; inductor triton bp int_mm")
    for i in range(len(shapes)):
        a_shape, b_shape = shapes[i]
        print(a_shape, "x", b_shape, end="; ")
        a = torch.randint(-128, 127, a_shape, device="cuda", dtype=torch.int8)
        b = torch.randint(-128, 127, b_shape, device="cuda", dtype=a.dtype).t().contiguous().t()

        config.max_autotune_gemm_backends = "aten"
        inductor_aten_mm(a, b)

        config.max_autotune_gemm_backends = "triton"
        inductor_triton_mm(a, b)

        config.use_block_pointer_mm_kernel = True
        inductor_triton_bp_mm(a,b)

        torch_ms = triton.testing.do_bench(lambda: torch_mm(a, b))
        triton_ms = triton.testing.do_bench(lambda: triton_mm(a, b))
        ind_aten_ms = triton.testing.do_bench(lambda: inductor_aten_mm(a, b))
        ind_triton_ms = triton.testing.do_bench(lambda: inductor_triton_mm(a, b))
        ind_triton_bp_ms = triton.testing.do_bench(lambda: inductor_triton_bp_mm(a, b))
        print(torch_ms, triton_ms, ind_aten_ms, ind_triton_ms, ind_triton_bp_ms, sep="; ")

        torch._dynamo.reset()


if __name__ == "__main__":
    shapes = [
        # alexnet
        ([128, 9216], [9216, 4096]),
        ([128, 4096], [4096, 4096]),
        ([128, 4096], [4096, 1000]),
        # BERT
        ([2048, 768], [768, 768]),
        ([2048, 768], [768, 3072]),
        ([2048, 3072], [3072, 768]),
        # hf_GPT2
        ([1024, 768], [768, 768]),
        ([1024, 768], [768, 3072]),
        ([1024, 3072], [3072, 768]),
        ([1024, 768], [768, 2304]),
        # SAM vit_h
        # ([78400, 3840], [3840, 1280]),
        # ([78400, 1280], [1280, 1280]),
        # ([65536, 5120], [5120, 1280]),
        # ([65536, 1280], [1280, 5120]),
    ]
    print("test total time")
    test_total_time(shapes)

    print("test GPU time")
    test_GPU_time(shapes)


# Results Preview on AWS AI cluster
"""
---------------------------------------WITHOUT GROUP_M=4---------------------------------------
test total time
shape; torch int_mm; triton matmul; inductor aten int_mm; inductor triton int_mm; inductor triton bp int_mm
[128, 9216] x [9216, 4096]; 0.05799745209515095; 0.10517491959035397; 0.0681717786937952; 0.06909355521202087; 0.0709458440542221
[128, 4096] x [4096, 4096]; 0.024167336523532867; 0.11976310051977634; 0.07393747568130493; 0.07287905551493168; 0.07042675279080868
[128, 4096] x [4096, 1000]; 0.03626302815973759; 0.11042582802474499; 0.06695923395454884; 0.0704268366098404; 0.07709362544119358
[2048, 768] x [768, 768]; 0.01828146167099476; 0.10658193379640579; 0.06808807142078876; 0.07399755530059338; 0.07611512206494808
[2048, 768] x [768, 3072]; 0.04024887457489967; 0.10827269405126572; 0.06870915181934834; 0.07038645446300507; 0.07122677750885487
[2048, 3072] x [3072, 768]; 0.04303659312427044; 0.10412078350782394; 0.06950575858354568; 0.07113117724657059; 0.06890214048326015
[1024, 768] x [768, 768]; 0.015347497537732124; 0.10368253104388714; 0.06642700172960758; 0.06732809357345104; 0.07504894398152828
[1024, 768] x [768, 3072]; 0.02212977036833763; 0.10721346363425255; 0.06806271150708199; 0.07025539875030518; 0.07209897041320801
[1024, 3072] x [3072, 768]; 0.028439955785870552; 0.10040698572993279; 0.07356751710176468; 0.07597001269459724; 0.06497411988675594
[1024, 768] x [768, 2304]; 0.021420521661639214; 0.10798651725053787; 0.07265874184668064; 0.07583596743643284; 0.0757780484855175
test GPU time
shape; torch int_mm; triton matmul; inductor aten int_mm; inductor triton int_mm; inductor triton bp int_mm
[128, 9216] x [9216, 4096]; 0.06937263906002045; 0.06606791168451309; 0.06941090524196625; 0.0749414786696434; 0.07445883005857468
[128, 4096] x [4096, 4096]; 0.03367209434509277; 0.03306420147418976; 0.033618275076150894; 0.03770475834608078; 0.03811996057629585
[128, 4096] x [4096, 1000]; 0.04053916409611702; 0.026428181678056717; 0.04048563912510872; 0.035524092614650726; 0.03511857986450195
[2048, 768] x [768, 768]; 0.021091658622026443; 0.014952811412513256; 0.02114434540271759; 0.016352148726582527; 0.01636400632560253
[2048, 768] x [768, 3072]; 0.044250234961509705; 0.034535523504018784; 0.04477497190237045; 0.0407957062125206; 0.0403938889503479
[2048, 3072] x [3072, 768]; 0.0443657785654068; 0.030912216752767563; 0.043943509459495544; 0.03152229264378548; 0.03151807561516762
[1024, 768] x [768, 768]; 0.015046144835650921; 0.012049268931150436; 0.015472489409148693; 0.014219170436263084; 0.014599963091313839
[1024, 768] x [768, 3072]; 0.025993037968873978; 0.02061585523188114; 0.025979794561862946; 0.024023013189435005; 0.023613713681697845
[1024, 3072] x [3072, 768]; 0.032189272344112396; 0.02463752031326294; 0.0325237400829792; 0.029383298009634018; 0.029373636469244957
[1024, 768] x [768, 2304]; 0.0256328321993351; 0.019821636378765106; 0.025220103561878204; 0.0207295510917902; 0.020740577951073647

---------------------------------------ALLOW GROUP_M=4---------------------------------------
test total time
shape; torch int_mm; triton matmul; inductor aten int_mm; inductor triton int_mm; inductor triton bp int_mm
[128, 9216] x [9216, 4096]; 0.058004120364785194; 0.10535435751080513; 0.073293661698699; 0.07340429350733757; 0.07662827149033546
[128, 4096] x [4096, 4096]; 0.024163806810975075; 0.10901077650487423; 0.07408992387354374; 0.0722301471978426; 0.07132194936275482
[128, 4096] x [4096, 1000]; 0.036254385486245155; 0.10616344399750233; 0.07640384137630463; 0.07024327293038368; 0.0717272236943245
[2048, 768] x [768, 768]; 0.018268311396241188; 0.10573056526482105; 0.07481606677174568; 0.07651427760720253; 0.07615952752530575
[2048, 768] x [768, 3072]; 0.04023437388241291; 0.11132276616990566; 0.07324442267417908; 0.07273463532328606; 0.06787577643990517
[2048, 3072] x [3072, 768]; 0.043066078796982765; 0.10560395196080208; 0.07386437617242336; 0.07078452967107296; 0.06927150301635265
[1024, 768] x [768, 768]; 0.015496015548706056; 0.10603046976029873; 0.07354759611189365; 0.0758734904229641; 0.07564103230834007
[1024, 768] x [768, 3072]; 0.022138087078928947; 0.10170646943151951; 0.06307680159807205; 0.06898500956594944; 0.06927525624632835
[1024, 3072] x [3072, 768]; 0.028426432982087135; 0.09982505813241005; 0.07412933744490147; 0.068456856533885; 0.06740028038620949
[1024, 768] x [768, 2304]; 0.021387580782175064; 0.1016910932958126; 0.07339306175708771; 0.0769086554646492; 0.07137617096304893
test GPU time
shape; torch int_mm; triton matmul; inductor aten int_mm; inductor triton int_mm; inductor triton bp int_mm
[128, 9216] x [9216, 4096]; 0.06933990865945816; 0.06610661745071411; 0.06979862600564957; 0.07433804869651794; 0.07468699663877487
[128, 4096] x [4096, 4096]; 0.03411557525396347; 0.033387377858161926; 0.03366631641983986; 0.03769776597619057; 0.037724412977695465
[128, 4096] x [4096, 1000]; 0.04040428251028061; 0.025955677032470703; 0.04039134085178375; 0.03547864779829979; 0.03545495495200157
[2048, 768] x [768, 768]; 0.02109251357614994; 0.014950398355722427; 0.02105111815035343; 0.01599065214395523; 0.016331763938069344
[2048, 768] x [768, 3072]; 0.044213179498910904; 0.0345146544277668; 0.04417350888252258; 0.040452226996421814; 0.04010145738720894
[2048, 3072] x [3072, 768]; 0.04449857398867607; 0.03184685856103897; 0.04453713446855545; 0.03114512376487255; 0.031160898506641388
[1024, 768] x [768, 768]; 0.015048310160636902; 0.01198173500597477; 0.015472636558115482; 0.014197333715856075; 0.014555573463439941
[1024, 768] x [768, 3072]; 0.02595936693251133; 0.020467696711421013; 0.02595866098999977; 0.02351984567940235; 0.023510659113526344
[1024, 3072] x [3072, 768]; 0.032588981091976166; 0.02414623275399208; 0.03259038180112839; 0.029331060126423836; 0.02935468591749668
[1024, 768] x [768, 2304]; 0.025275489315390587; 0.019762994721531868; 0.025669628754258156; 0.021082209423184395; 0.021062878891825676
"""
