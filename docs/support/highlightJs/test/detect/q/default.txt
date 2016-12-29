select time, price by date,stock from quote where price=(max;price)fby stock
data:raze value flip trade
select vwap:size wavg price by 5 xbar time.minute from aapl where date within (.z.d-10;.z.d)
f1:{[x;y;z] show (x;y+z);sum 1 2 3}
.z.pc:{[handle] show -3!(`long$.z.p;"Closed";handle)}
// random normal distribution, e.g. nor 10
nor:{$[x=2*n:x div 2;raze sqrt[-2*log n?1f]*/:(sin;cos)@\:(2*pi)*n?1f;-1_.z.s 1+x]}

mode:{where g=max g:count each group x}		// mode function