-- Test torch sort, show it suffers from the problems of quicksort
-- i.e. complexity O(N^2) in worst-case of sorted list
require 'gnuplot'

function testSort(output, descending)
    descending = descending or false
    local pow10 = torch.linspace(1,5,10)
    local bench_rnd = torch.zeros(pow10:numel())
    local bench_srt = torch.zeros(pow10:numel())
    local nrep = 3

    local function time_sort(x)
        local start = os.clock()
        torch.sort(x,descending)
        return (os.clock()-start)
    end

    for j = 1,nrep do
        for i = 1,pow10:numel() do

            local this_time
            local n = 10^pow10[i]

            -- on random
            this_time = time_sort(torch.rand(n))
            print('RND j:', j, 'for 10^', pow10[i], ' time: ', this_time)
            bench_rnd[i] = bench_rnd[i] + this_time/nrep
            collectgarbage()

            -- on random
            this_time = time_sort(torch.linspace(0,1,n))
            print('SRT j:', j, 'for 10^', pow10[i], ' time: ', this_time)
            bench_srt[i] = bench_srt[i] + this_time/nrep
            collectgarbage()

        end
        io.flush()
    end
    gnuplot.plot({'Random', pow10, bench_rnd},
                 {'Sorted', pow10, bench_srt})
    gnuplot.xlabel('Log10(N)')
    gnuplot.ylabel('Time (s)')
    gnuplot.figprint(output)
    print('RND:', bench_rnd)
    print('SRT:', bench_srt)
end

testSort('timeSortAscending.png', false) -- Ascending
testSort('timeSortDescending.png', true) -- Descending
