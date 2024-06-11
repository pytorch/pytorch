# Flash decoding pseudo code

## Partial Rowmax: 
**Requires:** Matrices $\mathbf{Q} \in \R^{Q\times d}$; $\mathbf{K}, \mathbf{V} \in \R^{N\times d}$ in HBM, block size $B_c$ (parallelize against $T_c = N/B_c$ blocks for each head per query)   
divide $\mathbf{K}$, $\mathbf{V}$ into $\mathbf{K_1}, \mathbf{K_2}, ... , \mathbf{K_{T_c}}$, and $\mathbf{V_1}, \mathbf{V_2}, ... , \mathbf{V_{T_c}}$ of size $B_r \times d$ each  
**Returns**: output:$\mathbf{O_{agg}} \in \R^{Q\times d}$, logsumexp: $\mathbf{L_{agg}} \in \R^Q$
> **for** each block in range($T_c$):  // unrolled, parallel on each block  
> $\quad$ $i = \text{bid}$       
> $\quad$ load $\mathbf{Q}$ (boardcast) and $\mathbf{K_i}$ from HBM to cache.   
> $\quad$ $\mathbf{S_i} = \mathbf{Q}\mathbf{K_i^T}$  ($\mathbf{S_i} \in \R^{Q\times B_c}$)  
> $\quad$ $\mathbf{\tilde{S_i}} = \text{scoremod}(\mathbf{S_i})$ // Apply costumized score func  
> $\quad$ $\mathbf{R_i} = \text{rowmax}(\mathbf{\tilde{S_i}})$ ($\mathbf{R_i} \in \R^{Q}$, rowmax)   
> $\quad$ $\mathbf{P_i}$ = exp($\mathbf{\tilde{S_i}}$ - $\mathbf{R_i}$), $\mathbf{O_i} = \mathbf{P_i}\mathbf{V_i}$ ($\mathbf{O_i} \in \R^{Q\times d}$)   
> $\quad$ $\mathbf{L_i}$ = rowsum($\mathbf{P_i}$)    
> $\quad$ Write $\mathbf{R_i} \in \R^Q$ to HMB, Write $\mathbf{O_i}$, $\mathbf{L_i}$ to on-chip cache   
> **end for**    
> _syncblocks()_    
> // Agregate R    
> Load $\mathbf{R} \in \R^{T_c\times Q}$ from HBM     
> $\mathbf{R_{max}} = \text{max}(\mathbf{R_i}) \in \R^Q$   
> write $\mathbf{R_{max}} \in \R^{Q}$ to HBM     
> // Rebase rowmax     
> _syncblocks()_    
> **for** each block in range($T_c$): // unrolled, parallel on each block   
> $\quad$ $i = \text{bid}$    
> $\quad$ load $\mathbf{R_{max}} \in \R^Q$ from HBM to on-chip cache (boardcast)   
> $\quad$ $\mathbf{\tilde{O_i}} = \mathbf{O_i}(\mathbf{R_i}-\mathbf{R_{max}}) \in \R^{Q \times d}$    
> $\quad$ $\mathbf{\tilde{L_i}} = \mathbf{L_i}(\mathbf{R_i}-\mathbf{R_{max}}) \in \R^{Q}$    
> $\quad$ write $\mathbf{\tilde{O_i}} \in \R^{Q\times d}$, $\mathbf{\tilde{L_i}} \in \R^{Q}$ to HBM    
> **end for**   
> _syncblocks()_   
> // Agregate O, L    
> Load $\mathbf{\tilde{O}} \in \R^{T_c\times Q \times d}$, $\mathbf{\tilde{L}} \in \R^{T_c \times Q}$ from HBM    
> $\mathbf{L_{agg}} = colsum(\mathbf{\tilde{L}}) \quad \R^{T_c \times Q}  \rightarrow \R^{Q}$     
> $\mathbf{O_{agg}} = colsum(\mathbf{\tilde{O}})  / \mathbf{L_{agg}} \quad \R^{T_c \times Q \times d}  \rightarrow \R^{Q \times d}$    
> write $\mathbf{O_{agg}} \in \R^{Q \times d}$, $\mathbf{L_{agg}} \in \R^Q$ to HBM     
> Returns $\mathbf{O_{agg}}$, $\mathbf{L_{agg}}$

### Complexity Summary
 - 3 block syncs
 - Computational Complexity: $2QNd + QN + Qd + d$
    - $\mathbf{S_i}$: $Q\times N \times d$
    - $\text{exp}(\mathbf{\tilde{S_i}})$: $Q \times N$
    - $\mathbf{O_i}$: $Q \times N \times d$
    - Rebase rowmax: $Q \times d + d$
 - Memory loading complexity: 
    - Assumptions : local shared mem per block is large enough to hold $\mathbf{Q} \in \R^{Q\times d}$ + $\mathbf{K_i} \in \R^{B_c\times d}$ + $\mathbf{{S_i}} \in \R^{Q\times B_c}$  
    local shared mem per block is large enough to hold $\mathbf{{O_i}} \in \R^{Q\times d}$ + $\mathbf{{L_i}} \in \R^{Q}$    
    gmem $Qd + 2Nd$; L2 $(log(T_c) + 1)Qd + 2QT_c$
        - Construct $\mathbf{O_i}$ and $\mathbf{L_i}$: load K, Q & V once: $Qd + 2Nd$. broadcast Q to each block: $Qd$
        - Reduction for $\mathbf{R_{max}}$: load R once:$T_c \times Q$ // This can happen in L2 cache
        - Rebase rowmax: broadcast $\mathbf{R_{max}}$: $QT_c$ // This can happen in L2 cache
        - Reduction for $\mathbf{L}$ and $\mathbf{O}$: $log(T_c) \times (Qd + Q)$ // This can happen in L2 cache
    -  Assumptions: local shared mem per block is large enough to hold $\mathbf{Q} \in \R^{Q\times d}$, but not $\mathbf{{S_i}} \in \R^{Q\times B_c}$ + $\mathbf{K_i} \in \R^{B_c\times d}$, not even in L2   
    local shared mem per block is large enough to hold $\mathbf{{O_i}} \in \R^{Q\times d}$ + $\mathbf{{L_i}} \in \R^{Q}$    
    gmem $Qd + 4Nd$; L2 $(log(T_c) + 1)Qd + 2QT_c$
        - Constuct $\mathbf{S_i}$: load K, Q & V once: $Qd + 2Nd$. broadcast Q to each block: $Qd$. 
        - Construct $\mathbf{R_i}$: load $\mathbf{S_i}$: $Nd$
        - Construct $\mathbf{O_i}$ and $\mathbf{L_i}$: load $\mathbf{S_i}$: $Nd$
        - Reduction for $\mathbf{R_{max}}$: load R once:$T_c \times Q$ // This can happen in L2 cache
        - Rebase rowmax: broadcast $\mathbf{R_{max}}$: $QT_c$ // This can happen in L2 cache
        - Reduction for $\mathbf{L}$ and $\mathbf{O}$: $log(T_c) \times (Qd + Q)$ // This can happen in L2 cache


## Flash Decoding (Alg 2): Global Rowmax
**Requires:** Matrices $\mathbf{Q} \in \R^{Q\times d}$; $\mathbf{K}, \mathbf{V} \in \R^{N\times d}$ in HBM, block size $B_c$ (parallelize against $T_c = N/B_c$ blocks for each head per query)   
divide $\mathbf{K}$, $\mathbf{V}$ into $\mathbf{K_1}, \mathbf{K_2}, ... , \mathbf{K_{T_c}}$, and $\mathbf{V_1}, \mathbf{V_2}, ... , \mathbf{V_{T_c}}$ of size $B_r \times d$ each  
**Returns**: output:$\mathbf{O_{agg}} \in \R^{Q\times d}$, logsumexp: $\mathbf{L_{agg}} \in \R^Q$
> **for** each block in range($T_c$):  // unrolled, parallel on each block  
> $\quad$ $i = \text{bid}$       
> $\quad$ load $\mathbf{Q}$ (boardcast) and $\mathbf{K_i}$ from HBM to cache.   
> $\quad$ $\mathbf{S_i} = \mathbf{Q}\mathbf{K_i^T}$  ($\mathbf{S_i} \in \R^{Q\times B_c}$)  
> $\quad$ $\mathbf{\tilde{S_i}} = \text{scoremod}(\mathbf{S_i})$ // Apply costumized score func  
> $\quad$ $\mathbf{R_i} = \text{rowmax}(\mathbf{\tilde{S_i}})$ ($\mathbf{R_i} \in \R^{Q}$, rowmax)   
> $\quad$ Write $\mathbf{R_i} \in \R^Q$ to HMB, Write $\mathbf{S_i}$ to on-chip cache   
> **end for**    
> _syncblocks()_    
> // Agregate R    
> Load $\mathbf{R} \in \R^{T_c\times Q}$ from HBM     
> $\mathbf{R_{max}} = \text{max}(\mathbf{R_i}) \in \R^Q$   
> write $\mathbf{R_{max}} \in \R^{Q}$ to HBM     
> // Rebase rowmax     
> _syncblocks()_    
> **for** each block in range($T_c$): // unrolled, parallel on each block   
> $\quad$ $i = \text{bid}$    
> $\quad$ load $\mathbf{R_{max}} \in \R^Q$ from HBM to on-chip cache (boardcast)   
> $\quad$ $\mathbf{P_i}$ = exp($\mathbf{\tilde{S_i}}$ - $\mathbf{R_{max}}$), $\mathbf{O_i} = \mathbf{P_i}\mathbf{V_i}$ ($\mathbf{O_i} \in \R^{Q\times d}$)   
> $\quad$ $\mathbf{L_i}$ = rowsum($\mathbf{P_i}$)     
> $\quad$ write $\mathbf{O_i} \in \R^{Q\times d}$, $\mathbf{L_i} \in \R^{Q}$ to HBM    
> **end for**   
> _syncblocks()_   
> // Agregate O, L    
> Load $\mathbf{O} \in \R^{T_c\times Q \times d}$, $\mathbf{L} \in \R^{T_c \times Q}$ from HBM    
> $\mathbf{L_{agg}} = colsum(\mathbf{L}) \quad \R^{T_c \times Q}  \rightarrow \R^{Q}$     
> $\mathbf{O_{agg}} = colsum(\mathbf{O})  / \mathbf{L_{agg}} \quad \R^{T_c \times Q \times d}  \rightarrow \R^{Q \times d}$    
> write $\mathbf{O_{agg}} \in \R^{Q \times d}$, $\mathbf{L_{agg}} \in \R^Q$ to HBM     
> Returns $\mathbf{O_{agg}}$, $\mathbf{L_{agg}}$


### Complexity Summary
> [!NOTE]
> Same number of syncblocks.     
> Computational Complexity: $2QNd + QN + Qd + d$ vs $2QNd + QN$    
> Global Memory Complexit: same regardless whether $\mathbf{S_i} = Q \times T_c$ fits in local mem. 

 - 3 block syncs _**(same as Alg1)**_
 - Computational Complexity: $2QNd + QN$ **_reduce Qd + d_**
    - $\mathbf{S_i}$: $Q\times N \times d$
    - $\text{exp}(\mathbf{\tilde{S_i}})$: $Q \times N$
    - $\mathbf{O_i}$: $Q \times N \times d$
 - Memory loading complexity: 
    - Assumptions: local shared mem per block is large enough to hold $\mathbf{Q} \in \R^{Q\times d}$ + $\mathbf{K_i} \in \R^{B_c\times d}$ + $\mathbf{{S_i}} \in \R^{Q\times B_c}$  
    local shared mem per block is large enough to hold $\mathbf{{O_i}} \in \R^{Q\times d}$ + $\mathbf{{L_i}} \in \R^{Q}$     
    gmem $Qd + 2Nd$; L2 $(log(T_c) + 1)(Qd +Q) + QT_c$ **_(same as Alg1)_**
        - Construct $\mathbf{R}$ and $\mathbf{S_i}$: load K V & Q, boardcast Q: $Qd + 2Nd$. broadcast Q: $Qd$
        - Reduction for $\mathbf{R_{max}}$: load R once:$Q \times T_{c}$ // This can happen in L2 cache
        - Construct $\mathbf{O_i}$ and $\mathbf{L_i}$: broadcast $\mathbf{R_{max}}$ once: $Q\times T_c$ // This can happen in L2
        - Reduction for $\mathbf{L}$ and $\mathbf{O}$: $log(T_c) \times (Qd + Q)$ // This can happen in L2 cache
    -  Assumptions: local shared mem per block is large enough to hold $\mathbf{Q} \in \R^{Q\times d}$, but not $\mathbf{{S_i}} \in \R^{Q\times B_c}$ + $\mathbf{K_i} \in \R^{B_c\times d}$, not even in L2   
    local shared mem per block is large enough to hold $\mathbf{{O_i}} \in \R^{Q\times d}$ + $\mathbf{{L_i}} \in \R^{Q}$    
    gmem $Qd + 4Nd$; L2 $(log(T_c) + 1)Qd + 2QT_c$ **_(same as Alg1)_**
        - Constuct $\mathbf{S_i}$: load K, Q & V once: $Qd + 2Nd$. broadcast Q to each block: $Qd$. 
        - Construct $\mathbf{R_i}$: load $\mathbf{S_i}$: $Nd$
        - Reduction for $\mathbf{R_{max}}$: load R once:$T_c \times Q$ // This can happen in L2 cache
        - Construct $\mathbf{O_i}$ and $\mathbf{L_i}$: load $\mathbf{S_i}$: $Nd$
        - Rebase rowmax: broadcast $\mathbf{R_{max}}$: $QT_c$ // This can happen in L2 cache
        - Reduction for $\mathbf{L}$ and $\mathbf{O}$: $log(T_c) \times (Qd + Q)$ // This can happen in L2 cache
