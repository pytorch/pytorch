# LaTeX macros used in the document
latex_macros = r"""
    \def \i                {\mathrm{i}}
    \def \e              #1{\mathrm{e}^{#1}}
    \def \w                {\omega}
    \def \wc               {\frac{\w}{c}}
    \def \vec            #1{\mathbf{#1}}
    \def \x                {\vec{x}}
    \def \xs               {\x_\text{s}}
    \def \xref             {\x_\text{ref}}
    \def \k                {\vec{k}}
    \def \n                {\vec{n}}
    \def \d                {\operatorname{d}\!}
    \def \dirac          #1{\operatorname{\delta}\left(#1\right)}
    \def \scalarprod   #1#2{\left\langle#1,#2\right\rangle}
    \def \Hankel     #1#2#3{\mathop{{}H_{#2}^{(#1)}}\!\left(#3\right)}
    \def \hankel     #1#2#3{\mathop{{}h_{#2}^{(#1)}}\!\left(#3\right)}
"""
