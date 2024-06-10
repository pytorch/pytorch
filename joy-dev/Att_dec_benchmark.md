# Xformers Flash Decoding Benchmark
| Only supports dypte=fp16, bf16,  No fp32
B: batch size  
Mq: query length (Q or M)  = 1 
Mkv: kv length (N)   = 256-132k 
Hq: Query heads = 16
Hkv: KV heads = 1 or 2 (8-16 query heads sharing one kv head. )
K: head dim (D)  = 128
```
[---------------------------------------------- attn_decodingfw ----------------------------------------------]                                                                                                                                                                                                                               
                                                                                       |  pytorch  |  optimized
1 threads: ----------------------------------------------------------------------------------------------------
      B=256 Mq=1 Mkv=256 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=35651584     |   1170.1  |     15.5  
      B=256 Mq=1 Mkv=256 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=35651584    |   1173.2  |     16.5  
      B=256 Mq=1 Mkv=256 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=35651584     |   1513.6  |           
      B=256 Mq=1 Mkv=256 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=69206016     |   1948.1  |     23.2  
      B=256 Mq=1 Mkv=256 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=69206016    |   1950.0  |     22.6  
      B=256 Mq=1 Mkv=256 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=69206016     |   2582.8  |           
      B=128 Mq=1 Mkv=512 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=34603008     |   1137.7  |     19.4  
      B=128 Mq=1 Mkv=512 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=34603008    |   1134.4  |     20.0  
      B=128 Mq=1 Mkv=512 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=34603008     |   1523.9  |           
      B=128 Mq=1 Mkv=512 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=68157440     |   1951.7  |     23.6  
      B=128 Mq=1 Mkv=512 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=68157440    |   1950.5  |     23.4  
      B=128 Mq=1 Mkv=512 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=68157440     |   2593.3  |           
      B=64 Mq=1 Mkv=1024 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=34078720     |   1162.2  |     19.2  
      B=64 Mq=1 Mkv=1024 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=34078720    |   1168.0  |     19.3  
      B=64 Mq=1 Mkv=1024 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=34078720     |   1632.1  |           
      B=64 Mq=1 Mkv=1024 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=67633152     |   1976.7  |     23.0  
      B=64 Mq=1 Mkv=1024 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=67633152    |   1982.2  |     23.0  
      B=64 Mq=1 Mkv=1024 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=67633152     |   2606.0  |           
      B=32 Mq=1 Mkv=2048 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=33816576     |   1318.1  |     19.0  
      B=32 Mq=1 Mkv=2048 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=33816576    |   1318.9  |     20.9  
      B=32 Mq=1 Mkv=2048 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=33816576     |   1656.6  |           
      B=32 Mq=1 Mkv=2048 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=67371008     |   1991.1  |     26.5  
      B=32 Mq=1 Mkv=2048 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=67371008    |   1993.3  |     34.0  
      B=32 Mq=1 Mkv=2048 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=67371008     |   2609.8  |           
      B=16 Mq=1 Mkv=4096 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=33685504     |   1348.2  |     20.9  
      B=16 Mq=1 Mkv=4096 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=33685504    |   1336.7  |     20.6  
      B=16 Mq=1 Mkv=4096 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=33685504     |   1667.6  |           
      B=16 Mq=1 Mkv=4096 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=67239936     |   2004.5  |     32.9  
      B=16 Mq=1 Mkv=4096 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=67239936    |   1993.2  |     29.8  
      B=16 Mq=1 Mkv=4096 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=67239936     |   2587.9  |           
      B=8 Mq=1 Mkv=8192 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=33619968      |   1340.7  |     20.4  
      B=8 Mq=1 Mkv=8192 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=33619968     |   1338.2  |     20.7  
      B=8 Mq=1 Mkv=8192 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=33619968      |   1661.3  |           
      B=8 Mq=1 Mkv=8192 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=67174400      |   1979.2  |     27.6  
      B=8 Mq=1 Mkv=8192 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=67174400     |   1979.5  |     35.8  
      B=8 Mq=1 Mkv=8192 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=67174400      |   2601.1  |           
      B=4 Mq=1 Mkv=16384 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=33587200     |   1335.6  |     17.2  
      B=4 Mq=1 Mkv=16384 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=33587200    |   1342.5  |     19.2  
      B=4 Mq=1 Mkv=16384 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=33587200     |   1670.4  |           
      B=4 Mq=1 Mkv=16384 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=67141632     |   1986.6  |     20.3  
      B=4 Mq=1 Mkv=16384 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=67141632    |   1993.3  |     26.1  
      B=4 Mq=1 Mkv=16384 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=67141632     |   2655.3  |           
      B=2 Mq=1 Mkv=32768 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=33570816     |   1366.4  |     19.1  
      B=2 Mq=1 Mkv=32768 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=33570816    |   1370.3  |     23.2  
      B=2 Mq=1 Mkv=32768 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=33570816     |   1711.1  |           
      B=2 Mq=1 Mkv=32768 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=67125248     |   2024.4  |     20.3  
      B=2 Mq=1 Mkv=32768 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=67125248    |   2028.6  |     34.2  
      B=2 Mq=1 Mkv=32768 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=67125248     |   2612.3  |           
      B=1 Mq=1 Mkv=65536 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=33562624     |    173.5  |     19.5  
      B=1 Mq=1 Mkv=65536 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=33562624    |    180.1  |     27.2  
      B=1 Mq=1 Mkv=65536 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=33562624     |    383.6  |           
      B=1 Mq=1 Mkv=65536 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=67117056     |    824.3  |     19.6  
      B=1 Mq=1 Mkv=65536 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=67117056    |    822.7  |     28.7  
      B=1 Mq=1 Mkv=65536 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=67117056     |   1287.1  |           
      B=1 Mq=1 Mkv=131072 Hq=16 Hkv=1 K=128 dtype=torch.float16 TotalBytes=67117056    |    331.1  |     52.0  
      B=1 Mq=1 Mkv=131072 Hq=16 Hkv=1 K=128 dtype=torch.bfloat16 TotalBytes=67117056   |    346.6  |     79.7  
      B=1 Mq=1 Mkv=131072 Hq=16 Hkv=1 K=128 dtype=torch.float32 TotalBytes=67117056    |    752.0  |           
      B=1 Mq=1 Mkv=131072 Hq=16 Hkv=2 K=128 dtype=torch.float16 TotalBytes=134225920   |   1618.7  |     67.1  
      B=1 Mq=1 Mkv=131072 Hq=16 Hkv=2 K=128 dtype=torch.bfloat16 TotalBytes=134225920  |   1602.1  |     86.7  
      B=1 Mq=1 Mkv=131072 Hq=16 Hkv=2 K=128 dtype=torch.float32 TotalBytes=134225920   |   2609.8  |           

Times are in microseconds (us).        
```


# FlexDecoding Performance Matrix 
