set pagination off
set logging overwrite on
set logging file fma.txt
set logging on
set $counter = 0
r
while($counter < 15000)
	echo "===GDB BEGIN==="
	where
	echo "===GDB END==="
	set $counter = $counter + 1
	c
end

