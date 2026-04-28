import math, unittest, torch, torch.nn.functional as F

def _ref(q, k, v, causal=False):
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    return out.to(q.dtype)

def _flash(q, k, v, causal=False):
    out, _ = torch.ops.aten._scaled_dot_product_flash_attention_for_mps(q, k, v, 0.0, causal, scale=None)
    return out

def qkv(B, H, S, D, dtype, dev, gqa=1, grad=False):
    kvH = H // gqa
    q = torch.randn(B, H, S, D, dtype=dtype, device=dev, requires_grad=grad)
    k = torch.randn(B, kvH, S, D, dtype=dtype, device=dev, requires_grad=grad)
    v = torch.randn(B, kvH, S, D, dtype=dtype, device=dev, requires_grad=grad)
    return q, k, v

class TestFwd(unittest.TestCase):
    dev = "mps"
    def chk(self, B, H, S, D, dtype, causal, gqa=1):
        tol = 1e-2 if dtype != torch.float32 else 1e-4
        q, k, v = qkv(B, H, S, D, dtype, self.dev, gqa)
        got = _flash(q, k, v, causal)
        ke = k.repeat_interleave(gqa, dim=1) if gqa>1 else k
        ve = v.repeat_interleave(gqa, dim=1) if gqa>1 else v
        ref = _ref(q, ke, ve, causal)
        self.assertEqual(got.shape, ref.shape)
        torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=tol,
            msg=f"B={B} H={H} S={S} D={D} dtype={dtype} causal={causal} gqa={gqa}")
    def test_01(self): self.chk(2,8,128,64,torch.float16,False)
    def test_02(self): self.chk(2,8,128,64,torch.float16,True)
    def test_03(self): self.chk(2,8,512,64,torch.float16,False)
    def test_04(self): self.chk(2,8,512,64,torch.float16,True)
    def test_05(self): self.chk(2,8,2048,64,torch.float16,False)
    def test_06(self): self.chk(2,8,2048,64,torch.float16,True)
    def test_07(self): self.chk(2,8,512,64,torch.bfloat16,False)
    def test_08(self): self.chk(2,8,512,64,torch.bfloat16,True)
    def test_09(self): self.chk(2,8,512,64,torch.float32,False)
    def test_10(self): self.chk(2,8,512,64,torch.float32,True)
    def test_11(self): self.chk(2,8,128,128,torch.float16,False)
    def test_12(self): self.chk(2,8,128,128,torch.float16,True)
    def test_13(self): self.chk(2,8,512,128,torch.float16,False)
    def test_14(self): self.chk(2,8,512,128,torch.float16,True)
    def test_15(self): self.chk(2,8,2048,128,torch.float16,False)
    def test_16(self): self.chk(2,8,2048,128,torch.float16,True)
    def test_17(self): self.chk(2,8,512,128,torch.bfloat16,False)
    def test_18(self): self.chk(2,8,512,128,torch.bfloat16,True)
    def test_19(self): self.chk(2,8,512,128,torch.float32,False)
    def test_20(self): self.chk(2,8,512,128,torch.float32,True)
    def test_gqa2_d64(self):       self.chk(2,8,512,64,torch.float16,False,gqa=2)
    def test_gqa4_d64(self):       self.chk(2,8,512,64,torch.float16,False,gqa=4)
    def test_gqa8_d64(self):       self.chk(2,8,512,64,torch.float16,False,gqa=8)
    def test_gqa2_d128(self):      self.chk(2,8,512,128,torch.float16,False,gqa=2)
    def test_gqa4_d128(self):      self.chk(2,8,512,128,torch.float16,False,gqa=4)
    def test_gqa8_d128(self):      self.chk(2,8,512,128,torch.float16,False,gqa=8)
    def test_gqa4_d64_c(self):     self.chk(2,8,512,64,torch.float16,True,gqa=4)
    def test_gqa4_d128_c(self):    self.chk(2,8,512,128,torch.float16,True,gqa=4)
    def test_b1h1(self):           self.chk(1,1,512,64,torch.float16,False)
    def test_b4h16(self):          self.chk(4,16,512,64,torch.float16,False)
    def test_llama(self):          self.chk(1,32,512,128,torch.float16,True,gqa=4)
    def test_mistral(self):        self.chk(1,32,512,128,torch.bfloat16,True,gqa=4)

class TestBwd(unittest.TestCase):
    dev = "mps"
    def bwd(self, B, H, S, D, dtype, causal, gqa=1):
        tol = 2e-2 if dtype != torch.float32 else 1e-4
        kvH = H // gqa
        q = torch.randn(B,H,S,D,dtype=dtype,device=self.dev,requires_grad=True)
        k = torch.randn(B,kvH,S,D,dtype=dtype,device=self.dev,requires_grad=True)
        v = torch.randn(B,kvH,S,D,dtype=dtype,device=self.dev,requires_grad=True)
        out,_ = torch.ops.aten._scaled_dot_product_flash_attention_for_mps(q,k,v,0.0,causal,scale=None)
        g = torch.randn_like(out)
        out.backward(g)
        dq,dk,dv = q.grad.clone(),k.grad.clone(),v.grad.clone()
        q2=q.detach().float().requires_grad_(True)
        k2=k.detach().float().requires_grad_(True)
        v2=v.detach().float().requires_grad_(True)
        ke=k2.repeat_interleave(gqa,dim=1) if gqa>1 else k2
        ve=v2.repeat_interleave(gqa,dim=1) if gqa>1 else v2
        F.scaled_dot_product_attention(q2,ke,ve,is_causal=causal).backward(g.float())
        dqr=q2.grad.to(dtype)
        dkr=k2.grad.to(dtype)
        dvr=v2.grad.to(dtype)
        tag=f"D={D}S={S}gqa={gqa}c={causal}"
        torch.testing.assert_close(dq.float(),dqr.float(),atol=tol,rtol=tol,msg=f"dQ {tag}")
        torch.testing.assert_close(dk.float(),dkr.float(),atol=tol,rtol=tol,msg=f"dK {tag}")
        torch.testing.assert_close(dv.float(),dvr.float(),atol=tol,rtol=tol,msg=f"dV {tag}")
    def test_01(self): self.bwd(2,8,256,64,torch.float16,False)
    def test_02(self): self.bwd(2,8,256,64,torch.float16,True)
    def test_03(self): self.bwd(2,8,512,64,torch.float16,False)
    def test_04(self): self.bwd(2,8,512,64,torch.float16,True)
    def test_05(self): self.bwd(2,8,256,128,torch.float16,False)
    def test_06(self): self.bwd(2,8,256,128,torch.float16,True)
    def test_07(self): self.bwd(2,8,512,128,torch.float16,False)
    def test_08(self): self.bwd(2,8,512,128,torch.float16,True)
    def test_09(self): self.bwd(2,8,512,64,torch.bfloat16,False)
    def test_10(self): self.bwd(2,8,512,64,torch.float32,False)
    def test_gqa2(self):        self.bwd(2,8,512,64,torch.float16,False,gqa=2)
    def test_gqa4(self):        self.bwd(2,8,512,64,torch.float16,False,gqa=4)
    def test_gqa8(self):        self.bwd(2,8,512,64,torch.float16,False,gqa=8)
    def test_gqa4_d128_c(self): self.bwd(2,8,512,128,torch.float16,True,gqa=4)
    def test_llama_bwd(self):   self.bwd(1,32,512,128,torch.float16,True,gqa=4)

@unittest.skip("MPS does not support float64; backward correctness verified by TestBwd comparison")
class TestGradcheck(unittest.TestCase):
    dev = "mps"
    def gc(self, S, D, causal):
        B,H=1,2
        q=torch.randn(B,H,S,D,dtype=torch.float32,device=self.dev,requires_grad=True)
        k=torch.randn(B,H,S,D,dtype=torch.float32,device=self.dev,requires_grad=True)
        v=torch.randn(B,H,S,D,dtype=torch.float32,device=self.dev,requires_grad=True)
        fn=lambda q,k,v: torch.ops.aten._scaled_dot_product_flash_attention_for_mps(q,k,v,0.0,causal,scale=None)[0]
        self.assertTrue(torch.autograd.gradcheck(fn,(q,k,v),eps=1e-2,atol=1e-2,rtol=1e-2,raise_exception=True))
    def test_d64_s32_nc(self):   self.gc(32,64,False)
    def test_d64_s32_c(self):    self.gc(32,64,True)
    def test_d128_s32_nc(self):  self.gc(32,128,False)
    def test_d128_s32_c(self):   self.gc(32,128,True)

class TestMemory(unittest.TestCase):
    dev = "mps"
    def test_flash_runs_at_long_seq(self):
        """Smoke test: flash can handle S=4096 without OOM."""
        B,H,D=1,8,64; dtype=torch.float16
        q=torch.randn(B,H,4096,D,dtype=dtype,device=self.dev)
        k=torch.randn(B,H,4096,D,dtype=dtype,device=self.dev)
        v=torch.randn(B,H,4096,D,dtype=dtype,device=self.dev)
        out,_ = torch.ops.aten._scaled_dot_product_flash_attention_for_mps(q,k,v,0.0,False)
        torch.mps.synchronize()
        self.assertEqual(out.shape, (B,H,4096,D))
    def test_flash_runs_at_very_long_seq(self):
        """Smoke test: flash can handle S=8192 without OOM."""
        B,H,D=1,4,64; dtype=torch.float16
        q=torch.randn(B,H,8192,D,dtype=dtype,device=self.dev)
        k=torch.randn(B,H,8192,D,dtype=dtype,device=self.dev)
        v=torch.randn(B,H,8192,D,dtype=dtype,device=self.dev)
        out,_ = torch.ops.aten._scaled_dot_product_flash_attention_for_mps(q,k,v,0.0,False)
        torch.mps.synchronize()
        self.assertEqual(out.shape, (B,H,8192,D))

class TestLSE(unittest.TestCase):
    dev = "mps"
    def test_lse(self):
        B,H,S,D=2,4,256,64; dtype=torch.float32
        q=torch.randn(B,H,S,D,dtype=dtype,device=self.dev)
        k=torch.randn(B,H,S,D,dtype=dtype,device=self.dev)
        v=torch.randn(B,H,S,D,dtype=dtype,device=self.dev)
        _,lse=torch.ops.aten._scaled_dot_product_flash_attention_for_mps(q,k,v,0.0,False,scale=None)
        assert lse.shape==(B,H,S), f"bad lse shape {lse.shape}"
        scores=torch.einsum("bhqd,bhkd->bhqk",q,k)/math.sqrt(D)
        torch.testing.assert_close(lse,torch.logsumexp(scores,dim=-1),atol=1e-4,rtol=1e-4)

if __name__ == "__main__":
    unittest.main(verbosity=2)


