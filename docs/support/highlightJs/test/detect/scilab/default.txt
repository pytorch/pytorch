// A comment
function I = foo(dims, varargin)
  d=[1; matrix(dims(1:$-1),-1,1)]
  for i=1:size(varargin)
    if varargin(i)==[] then
       I=[],
       return;
    end
  end
endfunction

b = cos(a) + cosh(a);
bar_matrix = [ "Hello", "world" ];
foo_matrix = [1, 2, 3; 4, 5, 6];
