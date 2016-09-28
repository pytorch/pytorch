 # torch.Tensor
## apply_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    apply_                    |         no  | All CPU Types |

**No Arguments**

**Returns        : nothing**

## element_size 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    element_size              |        yes  | All Types (CPU and CUDA) |

**No Arguments**

**Returns        : nothing**

## map_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    map_                      |         no  | All CPU Types |

**No Arguments**

**Returns        : nothing**

## map2_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    map2_                     |         no  | All CPU Types |

**No Arguments**

**Returns        : nothing**

## dim 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    dim                       |        yes  | All Types (CPU and CUDA) |

**No Arguments**

**Returns        : nothing**

## new 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    new                       |         no  | IS_CUDA |

**No Arguments**

**Returns        : nothing**

## nelement 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    nelement                  |        yes  | All Types (CPU and CUDA) |

**No Arguments**

**Returns        : nothing**

## select 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    select                    |        yes  | All Types (CPU and CUDA) |

**No Arguments**

**Returns        : nothing**

## storage 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    storage                   |         no  | All Types (CPU and CUDA) |

**No Arguments**

**Returns        : nothing**

## numpy 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    numpy                     |         no  | Byte // Short // Int // Long // Float // Double |

**No Arguments**

**Returns        : nothing**

## cat 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cat                       |        yes  | Cuda_Float // All CPU Types |

**No Arguments**

**Returns        : nothing**

## abs 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    abs  //  abs_             |        yes  | Float // Double // Long // Int // Cuda_Float // Cuda_Half // Cuda_Double // Cuda_Int // Cuda_Long |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## acos 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    acos  //  acos_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## add 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    add  //  add_             |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## addbmm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    addbmm  //  addbmm_       |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            beta    |          real   |            1 |
|            self    |        Tensor   |        [required] |
|           alpha    |          real   |            1 |
|          batch1    |        Tensor   |        [required] |
|          batch2    |        Tensor   |        [required] |

**Returns        : argument 0**

## addcdiv 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    addcdiv  //  addcdiv_     |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |            1 |
|         tensor1    |        Tensor   |        [required] |
|         tensor2    |        Tensor   |        [required] |

**Returns        : argument 0**

## addcmul 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    addcmul  //  addcmul_     |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |            1 |
|         tensor1    |        Tensor   |        [required] |
|         tensor2    |        Tensor   |        [required] |

**Returns        : argument 0**

## addmm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    addmm  //  addmm_         |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            beta    |          real   |            1 |
|            self    |        Tensor   |        [required] |
|           alpha    |          real   |            1 |
|            mat1    |        Tensor   |        [required] |
|            mat2    |        Tensor   |        [required] |

**Returns        : argument 0**

## addmv 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    addmv  //  addmv_         |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            beta    |          real   |            1 |
|            self    |        Tensor   |        [required] |
|           alpha    |          real   |            1 |
|             mat    |        Tensor   |        [required] |
|             vec    |        Tensor   |        [required] |

**Returns        : argument 0**

## addr 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    addr  //  addr_           |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            beta    |          real   |            1 |
|            self    |        Tensor   |        [required] |
|           alpha    |          real   |            1 |
|            vec1    |        Tensor   |        [required] |
|            vec2    |        Tensor   |        [required] |

**Returns        : argument 0**

## all 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    all                       |         no  | Byte |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : bool**

## any 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    any                       |         no  | Byte |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : bool**

## asin 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    asin  //  asin_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## atan 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    atan  //  atan_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## atan2 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    atan2  //  atan2_         |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           other    |        Tensor   |        [required] |

**Returns        : argument 0**

## baddbmm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    baddbmm  //  baddbmm_     |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            beta    |          real   |            1 |
|            self    |        Tensor   |        [required] |
|           alpha    |          real   |            1 |
|          batch1    |        Tensor   |        [required] |
|          batch2    |        Tensor   |        [required] |

**Returns        : argument 0**

## bernoulli_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    bernoulli_                |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|               p    |        double   |          0.5 |

**Returns        : self**

## bmm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    bmm                       |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|      AS_REAL(0)    |      CONSTANT   |        [required] |
|               0    |        Tensor   |        [required] |
|      AS_REAL(1)    |      CONSTANT   |        [required] |
|            mat1    |        Tensor   |        [required] |
|            mat2    |        Tensor   |        [required] |

**Returns        : argument 0**

## cauchy_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cauchy_                   |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|        location    |          real   |            0 |
|           scale    |          real   |            1 |

**Returns        : self**

## ceil 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    ceil  //  ceil_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## cinv 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cinv  //  cinv_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## clamp 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    clamp  //  clamp_         |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|             min    |          real   |        [required] |
|             max    |          real   |        [required] |

**Returns        : argument 0**

## clone 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    clone                     |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : Tensor**

## cmax 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cmax  //  cmax_           |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           other    |        Tensor   |        [required] |

**Returns        : argument 0**

## cmin 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cmin  //  cmin_           |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           other    |        Tensor   |        [required] |

**Returns        : argument 0**

## contiguous 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    contiguous                |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : Tensor**

## cos 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cos  //  cos_             |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## cosh 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cosh  //  cosh_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## cross 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cross                     |         no  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           other    |        Tensor   |        [required] |
|             dim    |          long   |           -1 |

**Returns        : argument 0**

## cumprod 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cumprod                   |         no  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |

**Returns        : argument 0**

## cumsum 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    cumsum                    |         no  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |

**Returns        : argument 0**

## data_ptr 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    data_ptr                  |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : void***

## diag 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    diag                      |        yes  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|        diagonal    |          long   |            0 |

**Returns        : argument 0**

## dist 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    dist                      |        yes  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|           other    |        Tensor   |        [required] |
|               p    |          real   |            2 |

**Returns        : nothing**

## div 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    div  //  div_             |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## dot 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    dot                       |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|          tensor    |        Tensor   |        [required] |

**Returns        : accreal**

## eig 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    eig                       |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|    eigenvectors    |          bool   |            N |

**Returns        : argument 0,1**

## eq 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    eq  //  eq_               |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |    ByteTensor   |        [optional] |
|          tensor    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## equal 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    equal                     |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|           other    |        Tensor   |        [required] |

**Returns        : bool**

## exp 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    exp  //  exp_             |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## exponential_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    exponential_              |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|           lambd    |          real   |            1 |

**Returns        : self**

## eye 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    eye                       |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|               n    |          long   |        [required] |
|               1    |          long   |        [required] |

**Returns        : argument 0**

## fill_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    fill_                     |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : self**

## floor 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    floor  //  floor_         |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## fmod 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    fmod  //  fmod_           |        yes  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## frac 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    frac  //  frac_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## free 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    free                      |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : self**

## gather 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    gather                    |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |
|           index    |    LongTensor   |        [required] |

**Returns        : argument 0**

## ge 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    ge  //  ge_               |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |    ByteTensor   |        [optional] |
|          tensor    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## gels 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    gels                      |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               A    |        Tensor   |        [required] |

**Returns        : argument 0,1**

## geometric_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    geometric_                |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|               p    |        double   |        [required] |

**Returns        : self**

## geqrf 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    geqrf                     |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0,1**

## ger 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    ger                       |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|      AS_REAL(0)    |      CONSTANT   |        [required] |
|               0    |        Tensor   |        [required] |
|      AS_REAL(1)    |      CONSTANT   |        [required] |
|            vec1    |        Tensor   |        [required] |
|            vec2    |        Tensor   |        [required] |

**Returns        : argument 0**

## gesv 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    gesv                      |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|        solution    |        Tensor   |        [optional] |
|              lu    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               A    |        Tensor   |        [required] |

**Returns        : argument 0,1**

## get_device 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    get_device                |         no  | IS_CUDA |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : long**

## gt 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    gt  //  gt_               |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |    ByteTensor   |        [optional] |
|          tensor    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## histc 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    histc                     |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|             100    |      CONSTANT   |        [required] |
|               0    |      CONSTANT   |        [required] |
|               0    |      CONSTANT   |        [required] |

**Returns        : argument 0**

## index_add_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    index_add_                |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |
|           index    |    LongTensor   |        [required] |
|          source    |        Tensor   |        [required] |

**Returns        : argument 0**

## index_copy_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    index_copy_               |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |
|           index    |    LongTensor   |        [required] |
|          source    |        Tensor   |        [required] |

**Returns        : argument 0**

## index_fill_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    index_fill_               |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |
|           index    |    LongTensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## index_select 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    index_select              |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |
|           index    |    LongTensor   |        [required] |

**Returns        : argument 0**

## inverse 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    inverse                   |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          output    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## is_contiguous 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    is_contiguous             |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : bool**

## is_same_size 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    is_same_size              |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|           other    |        Tensor   |        [required] |

**Returns        : bool**

## is_set_to 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    is_set_to                 |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|          tensor    |        Tensor   |        [required] |

**Returns        : bool**

## is_size 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    is_size                   |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|            size    |   LongStorage   |        [required] |

**Returns        : bool**

## kthvalue 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    kthvalue                  |        yes  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          values    |        Tensor   |        [optional] |
|         indices    |    LongTensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               k    |          long   |        [required] |
|      __last_dim    |      CONSTANT   |        [required] |

**Returns        : argument 0,1**

## le 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    le  //  le_               |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |    ByteTensor   |        [optional] |
|          tensor    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## lerp 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    lerp  //  lerp_           |        yes  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|             end    |        Tensor   |        [required] |
|          weight    |          real   |        [required] |

**Returns        : argument 0**

## linspace 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    linspace                  |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|           start    |          real   |        [required] |
|             end    |          real   |        [required] |
|           steps    |          long   |          100 |

**Returns        : argument 0**

## log 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    log  //  log_             |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## log1p 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    log1p  //  log1p_         |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## log_normal_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    log_normal_               |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|        location    |          real   |            1 |
|           scale    |          real   |            2 |

**Returns        : self**

## logspace 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    logspace                  |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|           start    |          real   |        [required] |
|             end    |          real   |        [required] |
|           steps    |          long   |          100 |

**Returns        : argument 0**

## lt 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    lt  //  lt_               |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |    ByteTensor   |        [optional] |
|          tensor    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## masked_copy_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    masked_copy_              |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|            mask    |    ByteTensor   |        [required] |
|          source    |        Tensor   |        [required] |

**Returns        : self**

## masked_fill_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    masked_fill_              |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|            mask    |    ByteTensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : self**

## masked_select 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    masked_select             |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|            mask    |    ByteTensor   |        [required] |

**Returns        : argument 0**

## max 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    max                       |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## mean 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    mean                      |        yes  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## median 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    median                    |        yes  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          values    |        Tensor   |        [optional] |
|         indices    |    LongTensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|      __last_dim    |      CONSTANT   |        [required] |

**Returns        : argument 0,1**

## min 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    min                       |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## mm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    mm                        |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|      AS_REAL(0)    |      CONSTANT   |        [required] |
|               0    |        Tensor   |        [required] |
|      AS_REAL(1)    |      CONSTANT   |        [required] |
|            mat1    |        Tensor   |        [required] |
|            mat2    |        Tensor   |        [required] |

**Returns        : argument 0**

## mode 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    mode                      |        yes  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          values    |        Tensor   |        [optional] |
|         indices    |    LongTensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|      __last_dim    |      CONSTANT   |        [required] |

**Returns        : argument 0,1**

## mul 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    mul  //  mul_             |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## multinomial 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    multinomial               |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|     num_samples    |          long   |        [required] |
|     replacement    |          bool   |        false |

**Returns        : argument 0**

## mv 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    mv                        |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|      AS_REAL(0)    |      CONSTANT   |        [required] |
|               0    |        Tensor   |        [required] |
|      AS_REAL(1)    |      CONSTANT   |        [required] |
|             mat    |        Tensor   |        [required] |
|             vec    |        Tensor   |        [required] |

**Returns        : argument 0**

## ndimension 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    ndimension                |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : long**

## narrow 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    narrow                    |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|       dimension    |          long   |        [required] |
|           start    |          long   |        [required] |
|          length    |          long   |        [required] |

**Returns        : argument 0**

## ne 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    ne  //  ne_               |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |    ByteTensor   |        [optional] |
|          tensor    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## neg 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    neg  //  neg_             |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## nonzero 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    nonzero                   |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |    LongTensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## norm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    norm                      |        yes  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|               p    |          real   |            2 |

**Returns        : nothing**

## normal_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    normal_                   |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|            mean    |          real   |            0 |
|             var    |          real   |            1 |

**Returns        : self**

## numel 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    numel                     |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : long**

## ones 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    ones  //  ones_           |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|       long_args    |   LongStorage   |        [required] |

**Returns        : argument 0**

## orgqr 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    orgqr                     |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|          input2    |        Tensor   |        [required] |

**Returns        : argument 0,1**

## ormqr 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    ormqr                     |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|          input2    |        Tensor   |        [required] |
|          input3    |        Tensor   |        [required] |
|            left    |          bool   |            L |
|       transpose    |          bool   |            N |

**Returns        : argument 0,1**

## potrf 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    potrf                     |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          output    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## potri 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    potri                     |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          output    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## potrs 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    potrs                     |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|          input2    |        Tensor   |        [required] |

**Returns        : argument 0**

## pow 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    pow  //  pow_             |        yes  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|        exponent    |          real   |        [required] |

**Returns        : argument 0**

## prod 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    prod                      |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## pstrf 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    pstrf                     |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |  THIntTensor*   |        [optional] |
|            self    |        Tensor   |        [required] |
|           upper    |          bool   |            U |
|             tol    |          real   |           -1 |

**Returns        : argument 0,1**

## qr 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    qr                        |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0,1**

## rand 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    rand                      |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|       long_args    |   LongStorage   |        [required] |

**Returns        : argument 0**

## randn 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    randn                     |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|       long_args    |   LongStorage   |        [required] |

**Returns        : argument 0**

## random_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    random_                   |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : self**

## randperm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    randperm                  |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|               n    |          long   |        [required] |

**Returns        : argument 0**

## range 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    range                     |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            xmin    |       accreal   |        [required] |
|            xmax    |       accreal   |        [required] |
|            step    |       accreal   |            1 |

**Returns        : argument 0**

## remainder 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    remainder  //  remainder_ |        yes  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## renorm 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    renorm  //  renorm_       |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               p    |          real   |        [required] |
|             dim    |          long   |        [required] |
|         maxnorm    |          real   |        [required] |

**Returns        : nothing**

## resize_as_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    resize_as_                |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|        template    |        Tensor   |        [required] |

**Returns        : self**

## resize_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    resize_                   |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|       long_args    |   LongStorage   |        [required] |
|            NULL    |      CONSTANT   |        [required] |

**Returns        : self**

## retain 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    retain                    |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : self**

## round 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    round  //  round_         |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## rsqrt 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    rsqrt  //  rsqrt_         |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## scatter_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    scatter_                  |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |
|           index    |    LongTensor   |        [required] |
|             src    |        Tensor   |        [required] |

**Returns        : argument 0**

## set_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    set_                      |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|          source    |        Tensor   |        [required] |

**Returns        : argument 0**

## sigmoid 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sigmoid  //  sigmoid_     |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## sign 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sign  //  sign_           |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## sin 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sin  //  sin_             |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## sinh 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sinh  //  sinh_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## size 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    size                      |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |

**Returns        : nothing**

## sort 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sort                      |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          values    |        Tensor   |        [optional] |
|         indices    |    LongTensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|      __last_dim    |      CONSTANT   |        [required] |
|           false    |      CONSTANT   |        [required] |

**Returns        : argument 0,1**

## sqrt 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sqrt  //  sqrt_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## squeeze 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    squeeze  //  squeeze_     |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## std 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    std                       |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## storage_offset 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    storage_offset            |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : long**

## stride 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    stride                    |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|             dim    |          long   |        [required] |

**Returns        : nothing**

## sub 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sub  //  sub_             |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|           value    |          real   |        [required] |

**Returns        : argument 0**

## sum 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    sum                       |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## svd 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    svd                       |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |        Tensor   |        [optional] |
|            res3    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|            some    |          bool   |            S |

**Returns        : argument 0,1,2**

## symeig 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    symeig                    |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|    eigenvectors    |          bool   |            N |
|           upper    |          bool   |            U |

**Returns        : argument 0,1**

## t 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    t  //  t_                 |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|               0    |      CONSTANT   |        [required] |
|               1    |      CONSTANT   |        [required] |

**Returns        : Tensor**

## tan 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    tan  //  tan_             |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## tanh 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    tanh  //  tanh_           |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## topk 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    topk                      |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          values    |        Tensor   |        [optional] |
|         indices    |    LongTensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               k    |          long   |        [required] |
|      __last_dim    |      CONSTANT   |        [required] |
|           false    |      CONSTANT   |        [required] |
|           false    |      CONSTANT   |        [required] |

**Returns        : argument 0,1**

## trace 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    trace                     |         no  | All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : accreal**

## transpose 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    transpose  //  transpose_ |        yes  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|            dim0    |          long   |        [required] |
|            dim1    |          long   |        [required] |

**Returns        : Tensor**

## tril 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    tril  //  tril_           |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               k    |          long   |            0 |

**Returns        : argument 0**

## triu 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    triu  //  triu_           |        yes  | Cuda_Float // All CPU Types |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|     destination    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               k    |          long   |            0 |

**Returns        : argument 0**

## trtrs 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    trtrs                     |         no  | Float // Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            res1    |        Tensor   |        [optional] |
|            res2    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|               A    |        Tensor   |        [required] |
|           upper    |          bool   |            U |
|       transpose    |          bool   |            N |
|   unitriangular    |          bool   |            N |

**Returns        : argument 0,1**

## trunc 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    trunc  //  trunc_         |        yes  | Float // Double // Cuda_Float // Cuda_Half // Cuda_Double |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |

**Returns        : argument 0**

## unfold 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    unfold                    |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|            self    |        Tensor   |        [required] |
|       dimension    |          long   |        [required] |
|            size    |          long   |        [required] |
|            step    |          long   |        [required] |

**Returns        : argument 0**

## uniform_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    uniform_                  |         no  | Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |
|            from    |          real   |            0 |
|              to    |          real   |            1 |

**Returns        : self**

## var 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    var                       |         no  | Float // Double // Cuda_Float |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : nothing**

## zero_ 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    zero_                     |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|            self    |        Tensor   |        [required] |

**Returns        : self**

## zeros 

|    Name                      |    Autograd |    defined if                |
| ---------------------------- | ----------- | ---------------------------- |
|    zeros  //  zeros_         |         no  | All Types (CPU and CUDA) |

**Arguments**

|    Name            |    Type         |    Default         |
| ------------------ | --------------- | ------------------ |
|          result    |        Tensor   |        [optional] |
|       long_args    |   LongStorage   |        [required] |

**Returns        : argument 0**

