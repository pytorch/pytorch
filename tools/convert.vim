"Slightly adjust indentation
%s/^   /        /g

" # -> len
%s/#\(\S*\) /len(\1)/g

" for loops
%s/for\( \)\{-\}\(\S*\)\( \)\{-\}=\( \)\{-\}\(\S*\),\( \)\{-\}\(\S*\)\( \)\{-\}do/for \2 in range(\5, \7+1)/g

" Change comments
%s/--\[\[/"""/g
%s/]]/"""/g
%s/--/#/g

" Add spacing between commas
%s/\(\S\),\(\S\)/\1, \2/g

%s/local //g
%s/ then/:/g
%s/ do/:/g
%s/end//g
%s/elseif/elif/g
%s/else/else:/g
%s/true/True/g
%s/false/False/g
%s/\~=/!=/g
%s/math\.min/min/g
%s/math\.max/max/g
%s/math\.abs/abs/g


%s/__init/__init__/g

" Rewrite function declarations
%s/function \w*:\(\w*\)/    def \1/g
%s/def \(.*\)$/def \1:/g

" class declaration
%s/\(\w*\), parent = torch\.class.*$/import torch\rfrom torch.legacy import nn\r\rclass \1(nn.Module):/g

%s/input\.THNN/self._backend/g
%s/\(self\.backend\w*$\)/\1\r        self._backend.library_state,/g
%s/def \(\w*\)(/def \1(self, /g

%s/__init__(self)/__init__()/g

%s/:\(\S\)/.\1/g

%s/\.cdata()//g
%s/THNN\.optionalTensor(\(.*\))/\1/g

%s/parent\./super(##, self)./g
