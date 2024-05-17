Flash decoding pseudo code

**Requires:** Matrices Q, K, V $\in \R^{N\times d} in HBM, block size $B_c$ (parallelize against $T_c = N/B_c$ CTAs for each head per query)   
divide K, V into $K_1, K_2, ... , K_{T_c}$, and $V_1, V_2, ... , V_{T_c}$ of size $B_r \times d$ each  
**Returns**: output:$ O \in \R^{N\times d}$, logsumexp: $L \in \R^N$
> **for** each block in range($T_c$):  // unrolled, parallel on each block  
> $\quad$ i = bid   
> $\quad$ load Q and $K_i$ from HBM to cache.   
> $\quad$ $S_i = Q@K_i$  ($S_i \in \R^{N\times N}$)  
> $\quad$ $S_i = score\_comp(S_i)$  
> $\quad$ $R_i = rowmax(S_i)$ ($\R_i \in \R^{N}$, rowmax)   
> $\quad$ $P_i$ = exp($S_i$ - $R_i$), $O_i = P_iV_i$ ($O_i \in \R^{N\times d}$)   
> $\quad$ $L_i$ = rowsum($P_i$)    
> $\quad$ Write $R_i$ to HMB, Write $O_i$, $L_i$ to local cache   
> **end for**    
> syncblocks()    
> // Agregate R
> Load $R \in \R^{T_c\times N}$ from HBM   
> $R_{max} = colmax(R) \in \R^N$ 
> write $R_{max}$ to HBM     
> // Rebase rowmax     
> **for** each block in range($T_c$): // unrolled, parallel on each block   
> $\quad$ i = bid    
> $\quad$ load $R_{max} \in \R^N $ from HBM to on-chip cache    
> $\quad$ $O_i = O_i*(R_i-R_{max}) \in \R^{N \times d}$    
> $\quad$ $L_i = L_i*(R_i-R_{max}) \in \R^{N}$    
> $\quad$ write $O_i$, $L_i$ to HBM    
> **end for**   
> syncblocks()   
> // Agregate O, L    
> Load $O \in \R^{T_c\times N \times d}$, $L \in \R^{T_c \times N}$ from HBM 
> $L_{agg} = colsum(L) \quad \R^{T_c \times N}  \rightarrow \R^{N}$     
> $O_{agg} = colsum(O) \quad \R^{T_c \times N \times d}  \rightarrow \R^{N \times d}$    
> $O_{agg} = O_{agg} / L_{agg} \in \R^{N\times d}$     
> write $O_{agg}$, $L_{agg}$ to HBM     
> Returns $O_{agg}$, $L_{agg}$
