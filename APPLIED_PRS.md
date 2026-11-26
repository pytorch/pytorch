# PRs Applied to `daisyden/any_gate`

Base: `upstream/viable/strict` @ `1e562eecd7cc4b320e029dc11df99981233c7aca` (2026-11-12).
Total PRs applied: **75** (13 merged to `pytorch/pytorch` main, 62 not merged).

## Merged to `pytorch/pytorch` main (13)

| # | PR | Author | Title |
|---|------|--------|-------|
| 1 | [#181559](https://github.com/pytorch/pytorch/pull/181559) | LuFinch | [XPU] Enable _flash_attention_forward/_flash_attention_backward |
| 2 | [#182906](https://github.com/pytorch/pytorch/pull/182906) | madhumitha0102 | Port 7 tests under test/distributed folder to XPU |
| 3 | [#188963](https://github.com/pytorch/pytorch/pull/188963) | daisyden | [Test][XPU] Enable XPU testing for test classes with allow_xpu=True |
| 4 | [#189335](https://github.com/pytorch/pytorch/pull/189335) | libohao1201 | [Test] Using TEST_MULTIACCELERATOR for multi-gpu checking |
| 5 | [#189337](https://github.com/pytorch/pytorch/pull/189337) | libohao1201 | [Test] Port 3 distributed/_shard tests to Intel GPU |
| 6 | [#196443](https://github.com/pytorch/pytorch/pull/196443) | pbielak | [3/N] Refactor `TestAOTAutogradDevice` and enable for XPU |
| 7 | [#198474](https://github.com/pytorch/pytorch/pull/198474) | daisyden | [Test][XPU]Enabel XPU on TestNNDeviceType and skip known issues |
| 8 | [#202750](https://github.com/pytorch/pytorch/pull/202750) | pbielak | [Test][XPU] Enable more `nn/test_convolution.py` tests on XPU |
| 9 | [#202760](https://github.com/pytorch/pytorch/pull/202760) | pbielak | [Test][XPU] Enable more `test_vmap.py` tests on XPU |
| 10 | [#202763](https://github.com/pytorch/pytorch/pull/202763) | pbielak | [Test][XPU] Enable more `test_foreach.py` cases on XPU |
| 11 | [#203236](https://github.com/pytorch/pytorch/pull/203236) | pbielak | [Test][XPU] Enable more `test_dataloader.py` tests on XPU |
| 12 | [#203731](https://github.com/pytorch/pytorch/pull/203731) | guangyey | Generalize CudaIPCTypes to be device-agnostic |
| 13 | [#204255](https://github.com/pytorch/pytorch/pull/204255) | BBBela | [XPU] Use a fake device guard under `FakeTensorMode` when no XPU device is available |

Note: #189335, #189337, and #203236 each landed, were reverted, and re-landed a second time via a follow-up PR (hence #189335/#189337 still show GitHub PR state `OPEN`, even though their content is in `main`).

## Not merged (62)

| # | PR | Author | Title |
|---|------|--------|-------|
| 1 | [#176420](https://github.com/pytorch/pytorch/pull/176420) | wpietka | [XPU] Ensure dense (canonical) strides for convolution inputs |
| 2 | [#182147](https://github.com/pytorch/pytorch/pull/182147) | tszulist-hbn | Support XPU and other accelerators in RNG state handling |
| 3 | [#183389](https://github.com/pytorch/pytorch/pull/183389) | madhumitha0102 | Port 7 tests under test/distributed folder to XPU |
| 4 | [#183620](https://github.com/pytorch/pytorch/pull/183620) | madhumitha0102 | [Test][XPU]Port 2 tests under test/distributed folder to XPU |
| 5 | [#189678](https://github.com/pytorch/pytorch/pull/189678) | libohao1201 | [Test] Enable xpu for test_schema_check |
| 6 | [#190773](https://github.com/pytorch/pytorch/pull/190773) | DamJanusz | [XPU] Describe conv buffers to oneDNN by their strides |
| 7 | [#191729](https://github.com/pytorch/pytorch/pull/191729) | libohao1201 | [Test][Refactor] Decouple custom_ops device tests from CUDA to be accelerator-agnostic |
| 8 | [#192493](https://github.com/pytorch/pytorch/pull/192493) | libohao1201 | [Test][XPU] Enable XPU for TestSortAndSelectDevice |
| 9 | [#193116](https://github.com/pytorch/pytorch/pull/193116) | libohao1201 | [Test] Enable XPU for test_decomp cross-reference tests |
| 10 | [#195806](https://github.com/pytorch/pytorch/pull/195806) | zxd1997066 | [Test][XPU][Refactor] Refactor test decoupling and device-agnostic in test/dynamo/test_misc.py |
| 11 | [#195828](https://github.com/pytorch/pytorch/pull/195828) | aeytkn | Generalize accelerator stream dispatch in from_dlpack |
| 12 | [#195997](https://github.com/pytorch/pytorch/pull/195997) | zxd1997066 | [Test][XPU][Refactor] Refactor test decoupling in test_dynamo_decompositions.py / test_wrap_inductor_compiled_regions.py |
| 13 | [#196910](https://github.com/pytorch/pytorch/pull/196910) | zxd1997066 | [Test][XPU]Port test/test_nestedtensor.py to XPU |
| 14 | [#197247](https://github.com/pytorch/pytorch/pull/197247) | gplutop7 | [4/N][Test][XPU] linalg: enable TestLinalgDevice on XPU |
| 15 | [#197782](https://github.com/pytorch/pytorch/pull/197782) | daisyden | [Test][XPU] enable TestUnaryUfuncs on xpu |
| 16 | [#198167](https://github.com/pytorch/pytorch/pull/198167) | zxd1997066 | [Test][XPU] Device-agnostic and enable for XPU in test_proxy_tensor.py, test_getitem.py, test_spectral_ops.py |
| 17 | [#199131](https://github.com/pytorch/pytorch/pull/199131) | tszulist-hbn | [XPU] Allow math SDPA fallback in test_dummy_mha_with_nt on XPU |
| 18 | [#199499](https://github.com/pytorch/pytorch/pull/199499) | tszulist-hbn | [FlopCounter] Count SDPA dispatched to the overrideable fused attention op |
| 19 | [#199768](https://github.com/pytorch/pytorch/pull/199768) | tszulist-hbn | [XPU] Exclude xpu from the undersized-src branch of test_masked_scatter |
| 20 | [#200063](https://github.com/pytorch/pytorch/pull/200063) | pbielak | [Test] Fix accelerator device never being resolved in test_expanded_weights.py |
| 21 | [#200965](https://github.com/pytorch/pytorch/pull/200965) | tszulist-hbn | [profiler][xpu] Emit stream on XPU kernel trace events |
| 22 | [#201547](https://github.com/pytorch/pytorch/pull/201547) | jkosnox | [inductor][xpu] Support align_random_eager dropout on XPU |
| 23 | [#201643](https://github.com/pytorch/pytorch/pull/201643) | pponikox | [XPU] Skip unsupported complex scatter_reduce cases on XPU in test_torch.py |
| 24 | [#201752](https://github.com/pytorch/pytorch/pull/201752) | lchen2331 | [XPU] Add _scaled_addmm support via oneDNN post-ops |
| 25 | [#201815](https://github.com/pytorch/pytorch/pull/201815) | daisyden | [Test][XPU] Enable some test classes to XPU |
| 26 | [#202524](https://github.com/pytorch/pytorch/pull/202524) | shangerxin | [Test][XPU] Enable XPU support in dynamo and testing test suites |
| 27 | [#202885](https://github.com/pytorch/pytorch/pull/202885) | daisyden | [Test][XPU] Enable XPU on test_opsinfo_schema and test_nested_graph_breaks_wraped |
| 28 | [#203021](https://github.com/pytorch/pytorch/pull/203021) | daisyden | [Test][XPU] Remove skips for fixed issue on XPU backend |
| 29 | [#203036](https://github.com/pytorch/pytorch/pull/203036) | tszulist-hbn | [XPU] Remove stale skipIfXpu from test_leaf_function_closure_constants_without_grad |
| 30 | [#203340](https://github.com/pytorch/pytorch/pull/203340) | zxd1997066 | [Test][XPU] Port more cases to XPU in test_pooling.py, test_user_streams.py, test_tp_richcompare.py |
| 31 | [#203354](https://github.com/pytorch/pytorch/pull/203354) | daisyden | [Test][XPU] Enable missing test cases for XPU platform |
| 32 | [#203372](https://github.com/pytorch/pytorch/pull/203372) | daisyden | [Test][XPU] Port 2 test cases in TestTorchCUDA to TestTorchDeviceType |
| 33 | [#203373](https://github.com/pytorch/pytorch/pull/203373) | AKloniecki | Enable more binary ufuncs tests on XPU |
| 34 | [#203522](https://github.com/pytorch/pytorch/pull/203522) | chunhuanMeng | Report num_sync_all_streams/num_device_alloc/num_device_free on XPU |
| 35 | [#203577](https://github.com/pytorch/pytorch/pull/203577) | dominikjastrzebskix | [Test][XPU] Refactor and enable XPU for TestLibtorchAgnostic |
| 36 | [#203750](https://github.com/pytorch/pytorch/pull/203750) | daisyden | [Test][XPU] Enable missing XPU inductor cases |
| 37 | [#203752](https://github.com/pytorch/pytorch/pull/203752) | daisyden | [Test][XPU] Enable missing XPU inductor cases |
| 38 | [#203794](https://github.com/pytorch/pytorch/pull/203794) | libohao1201 | [XPU] Enable SchemaCheck tests |
| 39 | [#203798](https://github.com/pytorch/pytorch/pull/203798) | dominikjastrzebskix | [XPU][UT] Remove skip for test_shapes from TestBmmOuterProductDevice |
| 40 | [#203962](https://github.com/pytorch/pytorch/pull/203962) | xuhancn | [XPU] Make the test memory-leak check work on any accelerator, not just CUDA |
| 41 | [#204050](https://github.com/pytorch/pytorch/pull/204050) | dominikjastrzebskix | [Test][XPU] Enable XPU for TestTensorPreparation |
| 42 | [#204057](https://github.com/pytorch/pytorch/pull/204057) | dominikjastrzebskix | [Test][XPU] Enable XPU for TestPrecompileNumerics |
| 43 | [#204204](https://github.com/pytorch/pytorch/pull/204204) | dominikjastrzebskix | [Test][XPU] Enable test_stream_creation_callback for XPU |
| 44 | [#204206](https://github.com/pytorch/pytorch/pull/204206) | jkosnox | [PATCH][XPU] Remove expectedFailureXPU from test_coalesce_wide_dense_dims |
| 45 | [#204211](https://github.com/pytorch/pytorch/pull/204211) | gplutop7 | [Test][XPU] Move test_reused_subgraph_square_backward to TestInvokeSubgraphCompileDevice |
| 46 | [#204216](https://github.com/pytorch/pytorch/pull/204216) | gplutop7 | [Test][XPU] Re-enable test_return_none in test_invoke_subgraph.py |
| 47 | [#204217](https://github.com/pytorch/pytorch/pull/204217) | daisyden | [Test][XPU] Enable some missing test on XPU |
| 48 | [#204232](https://github.com/pytorch/pytorch/pull/204232) | daisyden | [Test][XPU] Enable XPU test on some test cases to improve Intel GPU test coverage |
| 49 | [#204275](https://github.com/pytorch/pytorch/pull/204275) | gplutop7 | [Test][XPU] Enable test_bucketization_int32_overflow on accelerators |
| 50 | [#204340](https://github.com/pytorch/pytorch/pull/204340) | madhumitha0102 | [xpu][test] Port 10 distributed tests cases on Intel GPUs |
| 51 | [#204399](https://github.com/pytorch/pytorch/pull/204399) | libohao1201 | [XPU] Enable FP8 scaled matmul tests |
| 52 | [#204411](https://github.com/pytorch/pytorch/pull/204411) | daisyden | [Test][XPU] Enable XPU test by mirroring dtypesIfCUDA to dtypesIfXPU |
| 53 | [#204608](https://github.com/pytorch/pytorch/pull/204608) | tszulist-hbn | [XPU] Run the remaining move_to_device_pass tests on all accelerators |
| 54 | [#204787](https://github.com/pytorch/pytorch/pull/204787) | kdrozd-dev | [XPU][Profiler][UT]Enable TestProfilerEventsParityDevice and TestProfilerDeviceStopped |
| 55 | [#204811](https://github.com/pytorch/pytorch/pull/204811) | gplutop7 | [Test][XPU] Enable test_median_dim_index_is_first_occurrence on XPU |
| 56 | [#205100](https://github.com/pytorch/pytorch/pull/205100) | zxd1997066 | [Test][XPU] Device-agnostic and enable for XPU in test_ctx_manager.py and test_tensor_creation_ops.py |
| 57 | [#205294](https://github.com/pytorch/pytorch/pull/205294) | gplutop7 | [Test][XPU] Make test_sdpa device-agnostic in test_invoke_subgraph |
| 58 | [#205672](https://github.com/pytorch/pytorch/pull/205672) | jkosnox | Don't request a SYCL build for the DLPack exchange-API test extensions |
| 59 | [#205854](https://github.com/pytorch/pytorch/pull/205854) | pbielak | [Test][XPU] Enable more test_scatter_gather_ops tests for XPU |
| 60 | [#205862](https://github.com/pytorch/pytorch/pull/205862) | pbielak | [XPU] Allow multinomial with replacement past 2^24 categories on XPU |
| 61 | [#206179](https://github.com/pytorch/pytorch/pull/206179) | zxd1997066 | [Test][XPU] Device-agnostic and enable for XPU in test_torchinductor.py and test_autograd.py |
| 62 | [#206449](https://github.com/pytorch/pytorch/pull/206449) | tszulist-hbn | [xpu] Enable linear_cross_entropy fp16 test_redispatch on XPU |

## Author distribution (all 75 PRs)

| Author | PR count | % of 75 |
|---|---|---|
| **daisyden** | 13 | 17.3% |
| **libohao1201** | 8 | 10.7% |
| **pbielak** | 8 | 10.7% |
| **tszulist-hbn** | 8 | 10.7% |
| **zxd1997066** | 7 | 9.3% |
| **gplutop7** | 6 | 8.0% |
| **dominikjastrzebskix** | 5 | 6.7% |
| **madhumitha0102** | 4 | 5.3% |
| jkosnox | 3 | 4.0% |
| AKloniecki | 1 | 1.3% |
| BBBela | 1 | 1.3% |
| DamJanusz | 1 | 1.3% |
| LuFinch | 1 | 1.3% |
| aeytkn | 1 | 1.3% |
| chunhuanMeng | 1 | 1.3% |
| guangyey | 1 | 1.3% |
| kdrozd-dev | 1 | 1.3% |
| lchen2331 | 1 | 1.3% |
| pponikox | 1 | 1.3% |
| shangerxin | 1 | 1.3% |
| wpietka | 1 | 1.3% |
| xuhancn | 1 | 1.3% |

**21 distinct authors** across 75 PRs. Top 7 authors (daisyden, libohao1201, pbielak, tszulist-hbn, zxd1997066, gplutop7, dominikjastrzebskix) account for **55 of 75 (73.3%)**; the remaining 14 authors each contribute 1-3 PRs.

### Author distribution — merged only (13 PRs)

| Author | PR count | % of 13 |
|---|---|---|
| **pbielak** | 5 | 38.5% |
| **daisyden** | 2 | 15.4% |
| **libohao1201** | 2 | 15.4% |
| BBBela | 1 | 7.7% |
| LuFinch | 1 | 7.7% |
| guangyey | 1 | 7.7% |
| madhumitha0102 | 1 | 7.7% |

### Author distribution — not merged (62 PRs)

| Author | PR count | % of 62 |
|---|---|---|
| **daisyden** | 11 | 17.7% |
| **tszulist-hbn** | 8 | 12.9% |
| **zxd1997066** | 7 | 11.3% |
| **gplutop7** | 6 | 9.7% |
| **libohao1201** | 6 | 9.7% |
| **dominikjastrzebskix** | 5 | 8.1% |
| jkosnox | 3 | 4.8% |
| madhumitha0102 | 3 | 4.8% |
| pbielak | 3 | 4.8% |
| AKloniecki | 1 | 1.6% |
| DamJanusz | 1 | 1.6% |
| aeytkn | 1 | 1.6% |
| chunhuanMeng | 1 | 1.6% |
| kdrozd-dev | 1 | 1.6% |
| lchen2331 | 1 | 1.6% |
| pponikox | 1 | 1.6% |
| shangerxin | 1 | 1.6% |
| wpietka | 1 | 1.6% |
| xuhancn | 1 | 1.6% |
