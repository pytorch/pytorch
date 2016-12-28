## Master [UNRELEASED]

New languages:

- *Golo* by [Philippe Charrière][]
- *GAMS* by [Stefan Bechert][]
- *IRPF90* by [Anthony Scemama][]

Notable fixes and improvements to existing languages:

- JavaScript highlighting no longer fails with ES6 default parameters
- Added keywords `async` and `await` to Python

[Philippe Charrière]: https://github.com/k33g
[Stefan Bechert]: https://github.com/b-pos465
[Anthony Scemama]: https://github.com/scemama

## Version 8.7

New languages:

- *Zephir* by [Oleg Efimov][]
- *Elm* by [Janis Voigtländer][]
- *XQuery* by [Dirk Kirsten][]
- *Mojolicious* by [Dotan Dimet][]
- *AutoIt* by Manh Tuan from [J2TeaM][]
- *Toml* (ini extension) by [Guillaume Gomez][]

New Styles:

- *Hopscotch* by [Jan T. Sott][]
- *Grayscale* by [MY Sun][]

Notable fixes and improvements to existing languages:

- Fix encoding of images when copied over in certain builds
- Fix incorrect highlighting of the word "bug" in comments
- Treat decorators different from matrix multiplication in Python
- Fix traits inheritance highlighting in Rust
- Fix incorrect document
- Oracle keywords added to SQL language definition by [Vadimtro][]
- Postgres keywords added to SQL language definition by [Benjamin Auder][]
- Fix registers in x86asm being highlighted as a hex number
- Fix highlighting for numbers with a leading decimal point
- Correctly highlight numbers and strings inside of C/C++ macros
- C/C++ functions now support pointer, reference, and move returns

[Oleg Efimov]: https://github.com/Sannis
[Guillaume Gomez]: https://github.com/GuillaumeGomez
[Janis Voigtländer]: https://github.com/jvoigtlaender
[Jan T. Sott]: https://github.com/idleberg
[Dirk Kirsten]: https://github.com/dirkk
[MY Sun]: https://github.com/simonmysun
[Vadimtro]: https://github.com/Vadimtro
[Benjamin Auder]: https://github.com/ghost
[Dotan Dimet]: https://github.com/dotandimet
[J2TeaM]: https://github.com/J2TeaM

## Version 8.6

New languages:

- *C/AL* by [Kenneth Fuglsang][]
- *DNS zone file* by [Tim Schumacher][]
- *Ceylon* by [Lucas Werkmeister][]
- *OpenSCAD* by [Dan Panzarella][]
- *Inform7* by [Bruno Dias][]
- *armasm* by [Dan Panzarella][]
- *TP* by [Jay Strybis][]

New Styles:

- *Atelier Cave*, *Atelier Estuary*,
  *Atelier Plateau* and *Atelier Savanna* by [Bram de Haan][]
- *GitHub Gist* by [Louis Barranqueiro][]

Notable fixes and improvements to existing languages:

- Multi-line raw strings from C++11 are now supported
- Fix class names with dashes in HAML
- The `async` keyword from ES6/7 is now supported
- TypeScript functions handle type and parameter complexity better
- We unified phpdoc/javadoc/yardoc etc modes across all languages
- CSS .class selectors relevance was dropped to prevent wrong language detection
- Images is now included to CDN build
- Release process is now automated

[Bram de Haan]: https://github.com/atelierbram
[Kenneth Fuglsang]: https://github.com/kfuglsang
[Louis Barranqueiro]: https://github.com/LouisBarranqueiro
[Tim Schumacher]: https://github.com/enko
[Lucas Werkmeister]: https://github.com/lucaswerkmeister
[Dan Panzarella]: https://github.com/pzl
[Bruno Dias]: https://github.com/sequitur
[Jay Strybis]: https://github.com/unreal

## Version 8.5

New languages:

- *pf.conf* by [Peter Piwowarski][]
- *Julia* by [Kenta Sato][]
- *Prolog* by [Raivo Laanemets][]
- *Docker* by [Alexis Hénaut][]
- *Fortran* by [Anthony Scemama][] and [Thomas Applencourt][]
- *Kotlin* by [Sergey Mashkov][]

New Styles:

- *Agate* by [Taufik Nurrohman][]
- *Darkula* by [Jet Brains][]
- *Atelier Sulphurpool* by [Bram de Haan][]
- *Android Studio* by [Pedro Oliveira][]

Notable fixes and improvements to existing languages:

- ES6 features in JavaScript are better supported now by [Gu Yiling][].
- Swift now recognizes body-less method definitions.
- Single expression functions `def foo, do: ... ` now work in Elixir.
- More uniform detection of built-in classes in Objective C.
- Fixes for number literals and processor directives in Rust.
- HTML `<script>` tag now allows any language, not just JavaScript.
- Multi-line comments are supported now in MatLab.

[Taufik Nurrohman]: https://github.com/tovic
[Jet Brains]: https://www.jetbrains.com/
[Peter Piwowarski]: https://github.com/oldlaptop
[Kenta Sato]: https://github.com/bicycle1885
[Bram de Haan]: https://github.com/atelierbram
[Raivo Laanemets]: https://github.com/rla
[Alexis Hénaut]: https://github.com/AlexisNo
[Anthony Scemama]: https://github.com/scemama
[Pedro Oliveira]: https://github.com/kanytu
[Gu Yiling]: https://github.com/Justineo
[Sergey Mashkov]: https://github.com/cy6erGn0m
[Thomas Applencourt]: https://github.com/TApplencourt

## Version 8.4

We've got the new [demo page][]! The obvious new feature is the new look, but
apart from that it's got smarter: by presenting languages in groups it avoids
running 10000 highlighting attempts after first load which was slowing it down
and giving bad overall impression. It is now also being generated from test
code snippets so the authors of new languages don't have to update both tests
and the demo page with the same thing.

Other notable changes:

- The `template_comment` class is gone in favor of the more general `comment`.
- Number parsing unified and improved across languages.
- C++, Java and C# now use unified grammar to highlight titles in
  function/method definitions.
- The browser build is now usable as an AMD module, there's no separate build
  target for that anymore.
- OCaml has got a [comprehensive overhaul][ocaml] by [Mickaël Delahaye][].
- Clojure's data structures and literals are now highlighted outside of lists
  and we can now highlight Clojure's REPL sessions.

New languages:

- *AspectJ* by [Hakan Özler][]
- *STEP Part 21* by [Adam Joseph Cook][]
- *SML* derived by [Edwin Dalorzo][] from OCaml definition
- *Mercury* by [mucaho][]
- *Smali* by [Dennis Titze][]
- *Verilog* by [Jon Evans][]
- *Stata* by [Brian Quistorff][]

[Hakan Özler]: https://github.com/ozlerhakan
[Adam Joseph Cook]: https://github.com/adamjcook
[demo page]: https://highlightjs.org/static/demo/
[Ivan Sagalaev]: https://github.com/isagalaev
[Edwin Dalorzo]: https://github.com/edalorzo
[mucaho]: https://github.com/mucaho
[Dennis Titze]: https://github.com/titze
[Jon Evans]: https://github.com/craftyjon
[Brian Quistorff]: https://github.com/bquistorff
[ocaml]: https://github.com/isagalaev/highlight.js/pull/608#issue-46190207
[Mickaël Delahaye]: https://github.com/polazarus


## Version 8.3

We streamlined our tool chain, it is now based entirely on node.js instead of
being a mix of node.js, Python and Java. The build script options and arguments
remained the same, and we've noted all the changes in the [documentation][b].
Apart from reducing complexity, the new build script is also faster from not
having to start Java machine repeatedly. The credits for the work go to [Jeremy
Hull][].

Some notable fixes:

- PHP and JavaScript mixed in HTML now live happily with each other.
- JavaScript regexes now understand ES6 flags "u" and "y".
- `throw` keyword is no longer detected as a method name in Java.
- Fixed parsing of numbers and symbols in Clojure thanks to [input from Ivan
  Kleshnin][ik].

New languages in this release:

- *Less* by [Max Mikhailov][]
- *Stylus* by [Bryant Williams][]
- *Tcl* by [Radek Liska][]
- *Puppet* by [Jose Molina Colmenero][]
- *Processing* by [Erik Paluka][]
- *Twig* templates by [Luke Holder][]
- *PowerShell* by [David Mohundro][], based on [the work of Nicholas Blumhardt][ps]
- *XL* by [Christophe de Dinechin][]
- *LiveScript* by [Taneli Vatanen][] and [Jen Evers-Corvina][]
- *ERB* (Ruby in HTML) by [Lucas Mazza][]
- *Roboconf* by [Vincent Zurczak][]

[b]: http://highlightjs.readthedocs.org/en/latest/building-testing.html
[Jeremy Hull]: https://github.com/sourrust
[ik]: https://twitter.com/IvanKleshnin/status/514041599484231680
[Max Mikhailov]: https://github.com/seven-phases-max
[Bryant Williams]: https://github.com/scien
[Radek Liska]: https://github.com/Nindaleth
[Jose Molina Colmenero]: https://github.com/Moliholy
[Erik Paluka]: https://github.com/paluka
[Luke Holder]: https://github.com/lukeholder
[David Mohundro]: https://github.com/drmohundro
[ps]: https://github.com/OctopusDeploy/Library/blob/master/app/shared/presentation/highlighting/powershell.js
[Christophe de Dinechin]: https://github.com/c3d
[Taneli Vatanen]: https://github.com/Daiz-
[Jen Evers-Corvina]: https://github.com/sevvie
[Lucas Mazza]: https://github.com/lucasmazza
[Vincent Zurczak]: https://github.com/vincent-zurczak


## Version 8.2

We've finally got [real tests][test] and [continuous testing on Travis][ci]
thanks to [Jeremy Hull][] and [Chris Eidhof][]. The tests designed to cover
everything: language detection, correct parsing of individual language features
and various special cases. This is a very important change that gives us
confidence in extending language definitions and refactoring library core.

We're going to redesign the old [demo/test suite][demo] into an interactive
demo web app. If you're confident front-end developer or designer and want to
help us with it, drop a comment into [the issue][#542] on GitHub.

[test]: https://github.com/isagalaev/highlight.js/tree/master/test
[demo]: https://highlightjs.org/static/test.html
[#542]: https://github.com/isagalaev/highlight.js/issues/542
[ci]: https://travis-ci.org/isagalaev/highlight.js
[Jeremy Hull]: https://github.com/sourrust
[Chris Eidhof]: https://github.com/chriseidhof

As usually there's a handful of new languages in this release:

- *Groovy* by [Guillaume Laforge][]
- *Dart* by [Maxim Dikun][]
- *Dust* by [Michael Allen][]
- *Scheme* by [JP Verkamp][]
- *G-Code* by [Adam Joseph Cook][]
- *Q* from Kx Systems by [Sergey Vidyuk][]

[Guillaume Laforge]: https://github.com/glaforge
[Maxim Dikun]: https://github.com/dikmax
[Michael Allen]: https://github.com/bfui
[JP Verkamp]: https://github.com/jpverkamp
[Adam Joseph Cook]: https://github.com/adamjcook
[Sergey Vidyuk]: https://github.com/sv

Other improvements:

- [Erik Osheim][] heavily reworked Scala definitions making it richer.
- [Lucas Mazza][] fixed Ruby hashes highlighting
- Lisp variants (Lisp, Clojure and Scheme) are unified in regard to naming
  the first symbol in parentheses: it's "keyword" in general case and also
  "built_in" for built-in functions in Clojure and Scheme.

[Erik Osheim]: https://github.com/non
[Lucas Mazza]: https://github.com/lucasmazza

## Version 8.1

New languages:

- *Gherkin* by [Sam Pikesley][]
- *Elixir* by [Josh Adams][]
- *NSIS* by [Jan T. Sott][]
- *VIM script* by [Jun Yang][]
- *Protocol Buffers* by [Dan Tao][]
- *Nix* by [Domen Kožar][]
- *x86asm* by [innocenat][]
- *Cap’n Proto* and *Thrift* by [Oleg Efimov][]
- *Monkey* by [Arthur Bikmullin][]
- *TypeScript* by [Panu Horsmalahti][]
- *Nimrod* by [Flaviu Tamas][]
- *Gradle* by [Damian Mee][]
- *Haxe* by [Christopher Kaster][]
- *Swift* by [Chris Eidhof][] and [Nate Cook][]

New styles:

- *Kimbie*, light and dark variants by [Jan T. Sott][]
- *Color brewer* by [Fabrício Tavares de Oliveira][]
- *Codepen.io embed* by [Justin Perry][]
- *Hybrid* by [Nic West][]

[Sam Pikesley]: https://github.com/pikesley
[Sindre Sorhus]: https://github.com/sindresorhus
[Josh Adams]: https://github.com/knewter
[Jan T. Sott]: https://github.com/idleberg
[Jun Yang]: https://github.com/harttle
[Dan Tao]: https://github.com/dtao
[Domen Kožar]: https://github.com/iElectric
[innocenat]: https://github.com/innocenat
[Oleg Efimov]: https://github.com/Sannis
[Arthur Bikmullin]: https://github.com/devolonter
[Panu Horsmalahti]: https://github.com/panuhorsmalahti
[Flaviu Tamas]: https://github.com/flaviut
[Damian Mee]: https://github.com/chester1000
[Christopher Kaster]: http://christopher.kaster.ws
[Fabrício Tavares de Oliveira]: https://github.com/fabriciotav
[Justin Perry]: https://github.com/ourmaninamsterdam
[Nic West]: https://github.com/nicwest
[Chris Eidhof]: https://github.com/chriseidhof
[Nate Cook]: https://github.com/natecook1000

Other improvements:

- The README is heavily reworked and brought up to date by [Jeremy Hull][].
- Added [`listLanguages()`][ll] method in the API.
- Improved C/C++/C# detection.
- Added a bunch of new language aliases, documented the existing ones. Thanks to
  [Sindre Sorhus][] for background research.
- Added phrasal English words to boost relevance in comments.
- Many improvements to SQL definition made by [Heiko August][],
  [Nikolay Lisienko][] and [Travis Odom][].
- The shorter `lang-` prefix for language names in HTML classes supported
  alongside `language-`. Thanks to [Jeff Escalante][].
- Ruby's got support for interactive console sessions. Thanks to
  [Pascal Hurni][].
- Added built-in functions for R language. Thanks to [Artem A. Klevtsov][].
- Rust's got definition for lifetime parameters and improved string syntax.
  Thanks to [Roman Shmatov][].
- Various improvements to Objective-C definition by [Matt Diephouse][].
- Fixed highlighting of generics in Java.

[ll]: http://highlightjs.readthedocs.org/en/latest/api.html#listlanguages
[Sindre Sorhus]: https://github.com/sindresorhus
[Heiko August]: https://github.com/auge8472
[Nikolay Lisienko]: https://github.com/neor-ru
[Travis Odom]: https://github.com/Burstaholic
[Jeff Escalante]: https://github.com/jenius
[Pascal Hurni]: https://github.com/phurni
[Jiyin Yiyong]: https://github.com/jiyinyiyong
[Artem A. Klevtsov]: https://github.com/unikum
[Roman Shmatov]: https://github.com/shmatov
[Jeremy Hull]: https://github.com/sourrust
[Matt Diephouse]: https://github.com/mdiep


## Version 8.0

This new major release is quite a big overhaul bringing both new features and
some backwards incompatible changes. However, chances are that the majority of
users won't be affected by the latter: the basic scenario described in the
README is left intact.

Here's what did change in an incompatible way:

- We're now prefixing all classes located in [CSS classes reference][cr] with
  `hljs-`, by default, because some class names would collide with other
  people's stylesheets. If you were using an older version, you might still want
  the previous behavior, but still want to upgrade. To suppress this new
  behavior, you would initialize like so:

  ```html
  <script type="text/javascript">
    hljs.configure({classPrefix: ''});
    hljs.initHighlightingOnLoad();
  </script>
  ```

- `tabReplace` and `useBR` that were used in different places are also unified
  into the global options object and are to be set using `configure(options)`.
  This function is documented in our [API docs][]. Also note that these
  parameters are gone from `highlightBlock` and `fixMarkup` which are now also
  rely on `configure`.

- We removed public-facing (though undocumented) object `hljs.LANGUAGES` which
  was used to register languages with the library in favor of two new methods:
  `registerLanguage` and `getLanguage`. Both are documented in our [API docs][].

- Result returned from `highlight` and `highlightAuto` no longer contains two
  separate attributes contributing to relevance score, `relevance` and
  `keyword_count`. They are now unified in `relevance`.

Another technically compatible change that nonetheless might need attention:

- The structure of the NPM package was refactored, so if you had installed it
  locally, you'll have to update your paths. The usual `require('highlight.js')`
  works as before. This is contributed by [Dmitry Smolin][].

New features:

- Languages now can be recognized by multiple names like "js" for JavaScript or
  "html" for, well, HTML (which earlier insisted on calling it "xml"). These
  aliases can be specified in the class attribute of the code container in your
  HTML as well as in various API calls. For now there are only a few very common
  aliases but we'll expand it in the future. All of them are listed in the
  [class reference][cr].

- Language detection can now be restricted to a subset of languages relevant in
  a given context — a web page or even a single highlighting call. This is
  especially useful for node.js build that includes all the known languages.
  Another example is a StackOverflow-style site where users specify languages
  as tags rather than in the markdown-formatted code snippets. This is
  documented in the [API reference][] (see methods `highlightAuto` and
  `configure`).

- Language definition syntax streamlined with [variants][] and
  [beginKeywords][].

New languages and styles:

- *Oxygene* by [Carlo Kok][]
- *Mathematica* by [Daniel Kvasnička][]
- *Autohotkey* by [Seongwon Lee][]
- *Atelier* family of styles in 10 variants by [Bram de Haan][]
- *Paraíso* styles by [Jan T. Sott][]

Miscellaneous improvements:

- Highlighting `=>` prompts in Clojure.
- [Jeremy Hull][] fixed a lot of styles for consistency.
- Finally, highlighting PHP and HTML [mixed in peculiar ways][php-html].
- Objective C and C# now properly highlight titles in method definition.
- Big overhaul of relevance counting for a number of languages. Please do report
  bugs about mis-detection of non-trivial code snippets!

[API reference]: http://highlightjs.readthedocs.org/en/latest/api.html

[cr]: http://highlightjs.readthedocs.org/en/latest/css-classes-reference.html
[api docs]: http://highlightjs.readthedocs.org/en/latest/api.html
[variants]: https://groups.google.com/d/topic/highlightjs/VoGC9-1p5vk/discussion
[beginKeywords]: https://github.com/isagalaev/highlight.js/commit/6c7fdea002eb3949577a85b3f7930137c7c3038d
[php-html]: https://twitter.com/highlightjs/status/408890903017689088

[Carlo Kok]: https://github.com/carlokok
[Bram de Haan]: https://github.com/atelierbram
[Daniel Kvasnička]: https://github.com/dkvasnicka
[Dmitry Smolin]: https://github.com/dimsmol
[Jeremy Hull]: https://github.com/sourrust
[Seongwon Lee]: https://github.com/dlimpid
[Jan T. Sott]: https://github.com/idleberg


## Version 7.5

A catch-up release dealing with some of the accumulated contributions. This one
is probably will be the last before the 8.0 which will be slightly backwards
incompatible regarding some advanced use-cases.

One outstanding change in this version is the addition of 6 languages to the
[hosted script][d]: Markdown, ObjectiveC, CoffeeScript, Apache, Nginx and
Makefile. It now weighs about 6K more but we're going to keep it under 30K.

New languages:

- OCaml by [Mehdi Dogguy][mehdid] and [Nicolas Braud-Santoni][nbraud]
- [LiveCode Server][lcs] by [Ralf Bitter][revig]
- Scilab by [Sylvestre Ledru][sylvestre]
- basic support for Makefile by [Ivan Sagalaev][isagalaev]

Improvements:

- Ruby's got support for characters like `?A`, `?1`, `?\012` etc. and `%r{..}`
  regexps.
- Clojure now allows a function call in the beginning of s-expressions
  `(($filter "myCount") (arr 1 2 3 4 5))`.
- Haskell's got new keywords and now recognizes more things like pragmas,
  preprocessors, modules, containers, FFIs etc. Thanks to [Zena Treep][treep]
  for the implementation and to [Jeremy Hull][sourrust] for guiding it.
- Miscellaneous fixes in PHP, Brainfuck, SCSS, Asciidoc, CMake, Python and F#.

[mehdid]: https://github.com/mehdid
[nbraud]: https://github.com/nbraud
[revig]: https://github.com/revig
[lcs]: http://livecode.com/developers/guides/server/
[sylvestre]: https://github.com/sylvestre
[isagalaev]: https://github.com/isagalaev
[treep]: https://github.com/treep
[sourrust]: https://github.com/sourrust
[d]: http://highlightjs.org/download/


## New core developers

The latest long period of almost complete inactivity in the project coincided
with growing interest to it led to a decision that now seems completely obvious:
we need more core developers.

So without further ado let me welcome to the core team two long-time
contributors: [Jeremy Hull][] and [Oleg
Efimov][].

Hope now we'll be able to work through stuff faster!

P.S. The historical commit is [here][1] for the record.

[Jeremy Hull]: https://github.com/sourrust
[Oleg Efimov]: https://github.com/sannis
[1]: https://github.com/isagalaev/highlight.js/commit/f3056941bda56d2b72276b97bc0dd5f230f2473f


## Version 7.4

This long overdue version is a snapshot of the current source tree with all the
changes that happened during the past year. Sorry for taking so long!

Along with the changes in code highlight.js has finally got its new home at
<http://highlightjs.org/>, moving from its cradle on Software Maniacs which it
outgrew a long time ago. Be sure to report any bugs about the site to
<mailto:info@highlightjs.org>.

On to what's new…

New languages:

- Handlebars templates by [Robin Ward][]
- Oracle Rules Language by [Jason Jacobson][]
- F# by [Joans Follesø][]
- AsciiDoc and Haml by [Dan Allen][]
- Lasso by [Eric Knibbe][]
- SCSS by [Kurt Emch][]
- VB.NET by [Poren Chiang][]
- Mizar by [Kelley van Evert][]

[Robin Ward]: https://github.com/eviltrout
[Jason Jacobson]: https://github.com/jayce7
[Joans Follesø]: https://github.com/follesoe
[Dan Allen]: https://github.com/mojavelinux
[Eric Knibbe]: https://github.com/EricFromCanada
[Kurt Emch]: https://github.com/kemch
[Poren Chiang]: https://github.com/rschiang
[Kelley van Evert]: https://github.com/kelleyvanevert

New style themes:

- Monokai Sublime by [noformnocontent][]
- Railscasts by [Damien White][]
- Obsidian by [Alexander Marenin][]
- Docco by [Simon Madine][]
- Mono Blue by [Ivan Sagalaev][] (uses a single color hue for everything)
- Foundation by [Dan Allen][]

[noformnocontent]: http://nn.mit-license.org/
[Damien White]: https://github.com/visoft
[Alexander Marenin]: https://github.com/ioncreature
[Simon Madine]: https://github.com/thingsinjars
[Ivan Sagalaev]: https://github.com/isagalaev

Other notable changes:

- Corrected many corner cases in CSS.
- Dropped Python 2 version of the build tool.
- Implemented building for the AMD format.
- Updated Rust keywords (thanks to [Dmitry Medvinsky][]).
- Literal regexes can now be used in language definitions.
- CoffeeScript highlighting is now significantly more robust and rich due to
  input from [Cédric Néhémie][].

[Dmitry Medvinsky]: https://github.com/dmedvinsky
[Cédric Néhémie]: https://github.com/abe33


## Version 7.3

- Since this version highlight.js no longer works in IE version 8 and older.
  It's made it possible to reduce the library size and dramatically improve code
  readability and made it easier to maintain. Time to go forward!

- New languages: AppleScript (by [Nathan Grigg][ng] and [Dr. Drang][dd]) and
  Brainfuck (by [Evgeny Stepanischev][bolk]).

- Improvements to existing languages:

    - interpreter prompt in Python (`>>>` and `...`)
    - @-properties and classes in CoffeeScript
    - E4X in JavaScript (by [Oleg Efimov][oe])
    - new keywords in Perl (by [Kirk Kimmel][kk])
    - big Ruby syntax update (by [Vasily Polovnyov][vast])
    - small fixes in Bash

- Also Oleg Efimov did a great job of moving all the docs for language and style
  developers and contributors from the old wiki under the source code in the
  "docs" directory. Now these docs are nicely presented at
  <http://highlightjs.readthedocs.org/>.

[ng]: https://github.com/nathan11g
[dd]: https://github.com/drdrang
[bolk]: https://github.com/bolknote
[oe]: https://github.com/Sannis
[kk]: https://github.com/kimmel
[vast]: https://github.com/vast


## Version 7.2

A regular bug-fix release without any significant new features. Enjoy!


## Version 7.1

A Summer crop:

- [Marc Fornos][mf] made the definition for Clojure along with the matching
  style Rainbow (which, of course, works for other languages too).
- CoffeeScript support continues to improve getting support for regular
  expressions.
- Yoshihide Jimbo ported to highlight.js [five Tomorrow styles][tm] from the
  [project by Chris Kempson][tm0].
- Thanks to [Casey Duncun][cd] the library can now be built in the popular
  [AMD format][amd].
- And last but not least, we've got a fair number of correctness and consistency
  fixes, including a pretty significant refactoring of Ruby.

[mf]: https://github.com/mfornos
[tm]: http://jmblog.github.com/color-themes-for-highlightjs/
[tm0]: https://github.com/ChrisKempson/Tomorrow-Theme
[cd]: https://github.com/caseman
[amd]: http://requirejs.org/docs/whyamd.html


## Version 7.0

The reason for the new major version update is a global change of keyword syntax
which resulted in the library getting smaller once again. For example, the
hosted build is 2K less than at the previous version while supporting two new
languages.

Notable changes:

- The library now works not only in a browser but also with [node.js][]. It is
  installable with `npm install highlight.js`. [API][] docs are available on our
  wiki.

- The new unique feature (apparently) among syntax highlighters is highlighting
  *HTTP* headers and an arbitrary language in the request body. The most useful
  languages here are *XML* and *JSON* both of which highlight.js does support.
  Here's [the detailed post][p] about the feature.

- Two new style themes: a dark "south" *[Pojoaque][]* by Jason Tate and an
  emulation of*XCode* IDE by [Angel Olloqui][ao].

- Three new languages: *D* by [Aleksandar Ružičić][ar], *R* by [Joe Cheng][jc]
  and *GLSL* by [Sergey Tikhomirov][st].

- *Nginx* syntax has become a million times smaller and more universal thanks to
  remaking it in a more generic manner that doesn't require listing all the
  directives in the known universe.

- Function titles are now highlighted in *PHP*.

- *Haskell* and *VHDL* were significantly reworked to be more rich and correct
  by their respective maintainers [Jeremy Hull][sr] and [Igor Kalnitsky][ik].

And last but not least, many bugs have been fixed around correctness and
language detection.

Overall highlight.js currently supports 51 languages and 20 style themes.

[node.js]: http://nodejs.org/
[api]: http://softwaremaniacs.org/wiki/doku.php/highlight.js:api
[p]: http://softwaremaniacs.org/blog/2012/05/10/http-and-json-in-highlight-js/en/
[pojoaque]: http://web-cms-designs.com/ftopict-10-pojoaque-style-for-highlight-js-code-highlighter.html
[ao]: https://github.com/angelolloqui
[ar]: https://github.com/raleksandar
[jc]: https://github.com/jcheng5
[st]: https://github.com/tikhomirov
[sr]: https://github.com/sourrust
[ik]: https://github.com/ikalnitsky


## Version 6.2

A lot of things happened in highlight.js since the last version! We've got nine
new contributors, the discussion group came alive, and the main branch on GitHub
now counts more than 350 followers. Here are most significant results coming
from all this activity:

- 5 (five!) new languages: Rust, ActionScript, CoffeeScript, MatLab and
  experimental support for markdown. Thanks go to [Andrey Vlasovskikh][av],
  [Alexander Myadzel][am], [Dmytrii Nagirniak][dn], [Oleg Efimov][oe], [Denis
  Bardadym][db] and [John Crepezzi][jc].

- 2 new style themes: Monokai by [Luigi Maselli][lm] and stylistic imitation of
  another well-known highlighter Google Code Prettify by [Aahan Krish][ak].

- A vast number of [correctness fixes and code refactorings][log], mostly made
  by [Oleg Efimov][oe] and [Evgeny Stepanischev][es].

[av]: https://github.com/vlasovskikh
[am]: https://github.com/myadzel
[dn]: https://github.com/dnagir
[oe]: https://github.com/Sannis
[db]: https://github.com/btd
[jc]: https://github.com/seejohnrun
[lm]: http://grigio.org/
[ak]: https://github.com/geekpanth3r
[es]: https://github.com/bolknote
[log]: https://github.com/isagalaev/highlight.js/commits/


## Version 6.1 — Solarized

[Jeremy Hull][jh] has implemented my dream feature — a port of [Solarized][]
style theme famous for being based on the intricate color theory to achieve
correct contrast and color perception. It is now available for highlight.js in
both variants — light and dark.

This version also adds a new original style Arta. Its author pumbur maintains a
[heavily modified fork of highlight.js][pb] on GitHub.

[jh]: https://github.com/sourrust
[solarized]: http://ethanschoonover.com/solarized
[pb]: https://github.com/pumbur/highlight.js


## Version 6.0

New major version of the highlighter has been built on a significantly
refactored syntax. Due to this it's even smaller than the previous one while
supporting more languages!

New languages are:

- Haskell by [Jeremy Hull][sourrust]
- Erlang in two varieties — module and REPL — made collectively by [Nikolay
  Zakharov][desh], [Dmitry Kovega][arhibot] and [Sergey Ignatov][ignatov]
- Objective C by [Valerii Hiora][vhbit]
- Vala by [Antono Vasiljev][antono]
- Go by [Stephan Kountso][steplg]

[sourrust]: https://github.com/sourrust
[desh]: http://desh.su/
[arhibot]: https://github.com/arhibot
[ignatov]: https://github.com/ignatov
[vhbit]: https://github.com/vhbit
[antono]: https://github.com/antono
[steplg]: https://github.com/steplg

Also this version is marginally faster and fixes a number of small long-standing
bugs.

Developer overview of the new language syntax is available in a [blog post about
recent beta release][beta].

[beta]: http://softwaremaniacs.org/blog/2011/04/25/highlight-js-60-beta/en/

P.S. New version is not yet available on a Yandex CDN, so for now you have to
download [your own copy][d].

[d]: /soft/highlight/en/download/


## Version 5.14

Fixed bugs in HTML/XML detection and relevance introduced in previous
refactoring.

Also test.html now shows the second best result of language detection by
relevance.


## Version 5.13

Past weekend began with a couple of simple additions for existing languages but
ended up in a big code refactoring bringing along nice improvements for language
developers.

### For users

- Description of C++ has got new keywords from the upcoming [C++ 0x][] standard.
- Description of HTML has got new tags from [HTML 5][].
- CSS-styles have been unified to use consistent padding and also have lost
  pop-outs with names of detected languages.
- [Igor Kalnitsky][ik] has sent two new language descriptions: CMake & VHDL.

This makes total number of languages supported by highlight.js to reach 35.

Bug fixes:

- Custom classes on `<pre>` tags are not being overridden anymore
- More correct highlighting of code blocks inside non-`<pre>` containers:
  highlighter now doesn't insist on replacing them with its own container and
  just replaces the contents.
- Small fixes in browser compatibility and heuristics.

[c++ 0x]: http://ru.wikipedia.org/wiki/C%2B%2B0x
[html 5]: http://en.wikipedia.org/wiki/HTML5
[ik]: http://kalnitsky.org.ua/

### For developers

The most significant change is the ability to include language submodes right
under `contains` instead of defining explicit named submodes in the main array:

    contains: [
      'string',
      'number',
      {begin: '\\n', end: hljs.IMMEDIATE_RE}
    ]

This is useful for auxiliary modes needed only in one place to define parsing.
Note that such modes often don't have `className` and hence won't generate a
separate `<span>` in the resulting markup. This is similar in effect to
`noMarkup: true`. All existing languages have been refactored accordingly.

Test file test.html has at last become a real test. Now it not only puts the
detected language name under the code snippet but also tests if it matches the
expected one. Test summary is displayed right above all language snippets.


## CDN

Fine people at [Yandex][] agreed to host highlight.js on their big fast servers.
[Link up][l]!

[yandex]: http://yandex.com/
[l]: http://softwaremaniacs.org/soft/highlight/en/download/


## Version 5.10 — "Paris".

Though I'm on a vacation in Paris, I decided to release a new version with a
couple of small fixes:

- Tomas Vitvar discovered that TAB replacement doesn't always work when used
  with custom markup in code
- SQL parsing is even more rigid now and doesn't step over SmallTalk in tests


## Version 5.9

A long-awaited version is finally released.

New languages:

- Andrew Fedorov made a definition for Lua
- a long-time highlight.js contributor [Peter Leonov][pl] made a definition for
  Nginx config
- [Vladimir Moskva][vm] made a definition for TeX

[pl]: http://kung-fu-tzu.ru/
[vm]: http://fulc.ru/

Fixes for existing languages:

- [Loren Segal][ls] reworked the Ruby definition and added highlighting for
  [YARD][] inline documentation
- the definition of SQL has become more solid and now it shouldn't be overly
  greedy when it comes to language detection

[ls]: http://gnuu.org/
[yard]: http://yardoc.org/

The highlighter has become more usable as a library allowing to do highlighting
from initialization code of JS frameworks and in ajax methods (see.
readme.eng.txt).

Also this version drops support for the [WordPress][wp] plugin. Everyone is
welcome to [pick up its maintenance][p] if needed.

[wp]: http://wordpress.org/
[p]: http://bazaar.launchpad.net/~isagalaev/+junk/highlight/annotate/342/src/wp_highlight.js.php


## Version 5.8

- Jan Berkel has contributed a definition for Scala. +1 to hotness!
- All CSS-styles are rewritten to work only inside `<pre>` tags to avoid
  conflicts with host site styles.


## Version 5.7.

Fixed escaping of quotes in VBScript strings.


## Version 5.5

This version brings a small change: now .ini-files allow digits, underscores and
square brackets in key names.


## Version 5.4

Fixed small but upsetting bug in the packer which caused incorrect highlighting
of explicitly specified languages. Thanks to Andrew Fedorov for precise
diagnostics!


## Version 5.3

The version to fulfil old promises.

The most significant change is that highlight.js now preserves custom user
markup in code along with its own highlighting markup. This means that now it's
possible to use, say, links in code. Thanks to [Vladimir Dolzhenko][vd] for the
[initial proposal][1] and for making a proof-of-concept patch.

Also in this version:

- [Vasily Polovnyov][vp] has sent a GitHub-like style and has implemented
  support for CSS @-rules and Ruby symbols.
- Yura Zaripov has sent two styles: Brown Paper and School Book.
- Oleg Volchkov has sent a definition for [Parser 3][p3].

[1]: http://softwaremaniacs.org/forum/highlightjs/6612/
[p3]: http://www.parser.ru/
[vp]: http://vasily.polovnyov.ru/
[vd]: http://dolzhenko.blogspot.com/


## Version 5.2

- at last it's possible to replace indentation TABs with something sensible
  (e.g. 2 or 4 spaces)
- new keywords and built-ins for 1C by Sergey Baranov
- a couple of small fixes to Apache highlighting


## Version 5.1

This is one of those nice version consisting entirely of new and shiny
contributions!

- [Vladimir Ermakov][vooon] created highlighting for AVR Assembler
- [Ruslan Keba][rukeba] created highlighting for Apache config file. Also his
  original visual style for it is now available for all highlight.js languages
  under the name "Magula".
- [Shuen-Huei Guan][drake] (aka Drake) sent new keywords for RenderMan
  languages. Also thanks go to [Konstantin Evdokimenko][ke] for his advice on
  the matter.

[vooon]: http://vehq.ru/about/
[rukeba]: http://rukeba.com/
[drake]: http://drakeguan.org/
[ke]: http://k-evdokimenko.moikrug.ru/


## Version 5.0

The main change in the new major version of highlight.js is a mechanism for
packing several languages along with the library itself into a single compressed
file. Now sites using several languages will load considerably faster because
the library won't dynamically include additional files while loading.

Also this version fixes a long-standing bug with Javascript highlighting that
couldn't distinguish between regular expressions and division operations.

And as usually there were a couple of minor correctness fixes.

Great thanks to all contributors! Keep using highlight.js.


## Version 4.3

This version comes with two contributions from [Jason Diamond][jd]:

- language definition for C# (yes! it was a long-missed thing!)
- Visual Studio-like highlighting style

Plus there are a couple of minor bug fixes for parsing HTML and XML attributes.

[jd]: http://jason.diamond.name/weblog/


## Version 4.2

The biggest news is highlighting for Lisp, courtesy of Vasily Polovnyov. It's
somewhat experimental meaning that for highlighting "keywords" it doesn't use
any pre-defined set of a Lisp dialect. Instead it tries to highlight first word
in parentheses wherever it makes sense. I'd like to ask people programming in
Lisp to confirm if it's a good idea and send feedback to [the forum][f].

Other changes:

- Smalltalk was excluded from DEFAULT_LANGUAGES to save traffic
- [Vladimir Epifanov][voldmar] has implemented javascript style switcher for
  test.html
- comments now allowed inside Ruby function definition
- [MEL][] language from [Shuen-Huei Guan][drake]
- whitespace now allowed between `<pre>` and `<code>`
- better auto-detection of C++ and PHP
- HTML allows embedded VBScript (`<% .. %>`)

[f]: http://softwaremaniacs.org/forum/highlightjs/
[voldmar]: http://voldmar.ya.ru/
[mel]: http://en.wikipedia.org/wiki/Maya_Embedded_Language
[drake]: http://drakeguan.org/


## Version 4.1

Languages:

- Bash from Vah
- DOS bat-files from Alexander Makarov (Sam)
- Diff files from Vasily Polovnyov
- Ini files from myself though initial idea was from Sam

Styles:

- Zenburn from Vladimir Epifanov, this is an imitation of a
  [well-known theme for Vim][zenburn].
- Ascetic from myself, as a realization of ideals of non-flashy highlighting:
  just one color in only three gradations :-)

In other news. [One small bug][bug] was fixed, built-in keywords were added for
Python and C++ which improved auto-detection for the latter (it was shame that
[my wife's blog][alenacpp] had issues with it from time to time). And lastly
thanks go to Sam for getting rid of my stylistic comments in code that were
getting in the way of [JSMin][].

[zenburn]: http://en.wikipedia.org/wiki/Zenburn
[alenacpp]: http://alenacpp.blogspot.com/
[bug]: http://softwaremaniacs.org/forum/viewtopic.php?id=1823
[jsmin]: http://code.google.com/p/jsmin-php/


## Version 4.0

New major version is a result of vast refactoring and of many contributions.

Visible new features:

- Highlighting of embedded languages. Currently is implemented highlighting of
  Javascript and CSS inside HTML.
- Bundled 5 ready-made style themes!

Invisible new features:

- Highlight.js no longer pollutes global namespace. Only one object and one
  function for backward compatibility.
- Performance is further increased by about 15%.

Changing of a major version number caused by a new format of language definition
files. If you use some third-party language files they should be updated.


## Version 3.5

A very nice version in my opinion fixing a number of small bugs and slightly
increased speed in a couple of corner cases. Thanks to everybody who reports
bugs in he [forum][f] and by email!

There is also a new language — XML. A custom XML formerly was detected as HTML
and didn't highlight custom tags. In this version I tried to make custom XML to
be detected and highlighted by its own rules. Which by the way include such
things as CDATA sections and processing instructions (`<? ... ?>`).

[f]: http://softwaremaniacs.org/forum/viewforum.php?id=6


## Version 3.3

[Vladimir Gubarkov][xonix] has provided an interesting and useful addition.
File export.html contains a little program that shows and allows to copy and
paste an HTML code generated by the highlighter for any code snippet. This can
be useful in situations when one can't use the script itself on a site.


[xonix]: http://xonixx.blogspot.com/


## Version 3.2 consists completely of contributions:

- Vladimir Gubarkov has described SmallTalk
- Yuri Ivanov has described 1C
- Peter Leonov has packaged the highlighter as a Firefox extension
- Vladimir Ermakov has compiled a mod for phpBB

Many thanks to you all!


## Version 3.1

Three new languages are available: Django templates, SQL and Axapta. The latter
two are sent by [Dmitri Roudakov][1]. However I've almost entirely rewrote an
SQL definition but I'd never started it be it from the ground up :-)

The engine itself has got a long awaited feature of grouping keywords
("keyword", "built-in function", "literal"). No more hacks!

[1]: http://roudakov.ru/


## Version 3.0

It is major mainly because now highlight.js has grown large and has become
modular. Now when you pass it a list of languages to highlight it will
dynamically load into a browser only those languages.

Also:

- Konstantin Evdokimenko of [RibKit][] project has created a highlighting for
  RenderMan Shading Language and RenderMan Interface Bytestream. Yay for more
  languages!
- Heuristics for C++ and HTML got better.
- I've implemented (at last) a correct handling of backslash escapes in C-like
  languages.

There is also a small backwards incompatible change in the new version. The
function initHighlighting that was used to initialize highlighting instead of
initHighlightingOnLoad a long time ago no longer works. If you by chance still
use it — replace it with the new one.

[RibKit]: http://ribkit.sourceforge.net/


## Version 2.9

Highlight.js is a parser, not just a couple of regular expressions. That said
I'm glad to announce that in the new version 2.9 has support for:

- in-string substitutions for Ruby -- `#{...}`
- strings from from numeric symbol codes (like #XX) for Delphi


## Version 2.8

A maintenance release with more tuned heuristics. Fully backwards compatible.


## Version 2.7

- Nikita Ledyaev presents highlighting for VBScript, yay!
- A couple of bugs with escaping in strings were fixed thanks to Mickle
- Ongoing tuning of heuristics

Fixed bugs were rather unpleasant so I encourage everyone to upgrade!


## Version 2.4

- Peter Leonov provides another improved highlighting for Perl
- Javascript gets a new kind of keywords — "literals". These are the words
  "true", "false" and "null"

Also highlight.js homepage now lists sites that use the library. Feel free to
add your site by [dropping me a message][mail] until I find the time to build a
submit form.

[mail]: mailto:Maniac@SoftwareManiacs.Org


## Version 2.3

This version fixes IE breakage in previous version. My apologies to all who have
already downloaded that one!


## Version 2.2

- added highlighting for Javascript
- at last fixed parsing of Delphi's escaped apostrophes in strings
- in Ruby fixed highlighting of keywords 'def' and 'class', same for 'sub' in
  Perl


## Version 2.0

- Ruby support by [Anton Kovalyov][ak]
- speed increased by orders of magnitude due to new way of parsing
- this same way allows now correct highlighting of keywords in some tricky
  places (like keyword "End" at the end of Delphi classes)

[ak]: http://anton.kovalyov.net/


## Version 1.0

Version 1.0 of javascript syntax highlighter is released!

It's the first version available with English description. Feel free to post
your comments and question to [highlight.js forum][forum]. And don't be afraid
if you find there some fancy Cyrillic letters -- it's for Russian users too :-)

[forum]: http://softwaremaniacs.org/forum/viewforum.php?id=6
