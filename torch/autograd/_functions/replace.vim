%s/self/ctx/g
%s/\s\+def forward/    @staticmethod\r    def forward/g
%s/\s\+def backward/    @staticmethod\r    @once_differentiable\r    def backward/g
