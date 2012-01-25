-- 
-- rudimentary histogram diplay on the command line.
--
-- Author: Marco Scoffier
-- Date  : 
-- Mod   : Oct 21, 2011
--  + made 80 columns default
--  + save index of max bin in h.max not pointer to bin
--
function torch.histc__tostring(h, barHeight)
   barHeight = barHeight or 10
   local lastm = h[h.max].nb
   local incr  = lastm/(barHeight+1)
   local m     = lastm - incr
   local tl    = torch.Tensor(#h):fill(0)
   local toph  = '|'
   local topm  = ':'
   local topl  = '.'
   local bar   = '|'
   local blank = ' '
   local yaxis = '--------:' 
   local str = 'nsamples:'
   str = str .. 
      string.format('  min:(bin:%d/#%d/cntr:%2.2f)  max:(bin:%d/#%d/cntr:%2.2f)\n',
                    h.min,h[h.min].nb,h[h.min].val, 
                    h.max,h[h.max].nb,h[h.max].val)
   
   str = str .. yaxis
   for j = 1,#h do 
      str = str .. '-'
   end
   str = str .. '\n'

   for i = 1,barHeight do
      -- y axis
      if i%1==0 then
         str = str .. string.format('%1.2e:',m)
      end
      for j = 1,#h do
         if tl[j] == 1 then
            str = str .. bar
         elseif h[j].nb < m then
            str = str .. blank
         else
            -- in the bracket
            tl[j] = 1
            -- find 1/3rds
            local p = (lastm - h[j].nb) / incr
            if p > 0.66 then
               str = str .. toph
            elseif p > 0.33 then
               str = str .. topm
            else
               str = str .. topl
            end
         end
      end
      str = str .. '\n'
      lastm = m 
      m     = m - incr
   end
   -- x axis
   str = str .. yaxis 
   for j = 1,#h do
      if ((j - 2) % 6 == 0)then
         str = str .. '^'
      else
         str = str .. '-'
      end
   end
   str = str .. '\ncenters '
   for j = 1,#h do
      if ((j - 2) % 6 == 0)then
         if h[j].val < 0 then
            str = str .. '-'
         else
            str = str .. '+'
         end
         str = str .. string.format('%1.2f ',math.abs(h[j].val))
      end
   end
   return str
end

-- a simple function that computes the histogram of a tensor
function torch.histc(...)
   -- get args
   local args = {...}
   local tensor = args[1] or error('usage: torch.histc (tensor [, nBins] [, min] [, max]')
   local bins = args[2] or 80 - 8
   local min = args[3] or tensor:min()
   local max = args[4] or tensor:max()
   local raw = args[5] or false

   -- compute histogram
   local hist = torch.zeros(bins)
   local ten = torch.Tensor(tensor:nElement()):copy(tensor)
   ten:add(-min):div(max-min):mul(bins - 1e-6):floor():add(1)
   ten.torch._histc(ten, hist, bins)

   -- return raw histogram (no extra info)
   if raw then return hist end

   -- cleanup hist
   local cleanhist = {}
   cleanhist.raw = hist
   local _,mx = torch.max(cleanhist.raw)
   local _,mn = torch.min(cleanhist.raw)
   cleanhist.bins = bins
   cleanhist.binwidth = (max-min)/bins
   for i = 1,bins do
      cleanhist[i] = {}
      cleanhist[i].val = min + (i-0.5)*cleanhist.binwidth
      cleanhist[i].nb = hist[i]
   end
   cleanhist.max = mx[1]
   cleanhist.min = mn[1]

   -- print function
   setmetatable(cleanhist, {__tostring=torch.histc__tostring})
   return cleanhist
end

