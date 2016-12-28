CSS classes reference
=====================

This is a full list of available classes corresponding to languages'
syntactic structures. The parentheses after language name contain identifiers
used as class names in ``<code>`` element.


Gams ("gms", "gams")
--------------------

* ``section``:          section
* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string


Python ("python", "py", "gyp")
------------------------------

* ``keyword``:          keyword
* ``built_in``:         built-in objects (None, False, True and Ellipsis)
* ``number``:           number
* ``string``:           string (of any type)
* ``comment``:          comment
* ``decorator``:        @-decorator for functions
* ``function``:         function header "def some_name(...):"
* ``class``:            class header "class SomeName(...):"
* ``title``:            name of a function or a class inside a header
* ``params``:           everything inside parentheses in a function's or class' header

Python profiler results ("profile")
-----------------------------------

* ``number``:           number
* ``string``:           string
* ``built_in``:         built-in function entry
* ``filename``:         filename in an entry
* ``summary``:          profiling summary
* ``header``:           header of table of results
* ``keyword``:          column header
* ``function``:         function name in an entry (including parentheses)
* ``title``:            actual name of a function in an entry (excluding parentheses)
* ``prompt``:           interpreter prompt (>>> or ...)

Ruby ("ruby", "rb", "gemspec", "podspec", "thor", "irb")
--------------------------------------------------------

* ``keyword``:          keyword
* ``string``:           string
* ``subst``:            in-string substitution (#{...})
* ``comment``:          comment
* ``doctag``:           YARD doctag
* ``function``:         function header "def some_name(...):"
* ``class``:            class header "class SomeName(...):"
* ``title``:            name of a function or a class inside a header
* ``parent``:           name of a parent class
* ``symbol``:           symbol
* ``input``:            complete input line (interpreter)
* ``output``:           complete output line  (interpreter)
* ``prompt``:           interpreter prompt (>>)
* ``status``:           interpreter response (=>)

Haml ("haml")
-------------

* ``tag``:              any tag starting with "%"
* ``title``:            tag's name
* ``attribute``:        tag's attribute
* ``keyword``:          tag's attribute that is a keyword
* ``string``:           attribute's value that is a string
* ``value``:            attribute's value, shorthand id or class for tag
* ``comment``:          comment
* ``doctype``:          !!! declaration
* ``bullet``:           line defined by variable

Perl ("perl", "pl")
-------------------

* ``keyword``:          keyword
* ``comment``:          comment
* ``number``:           number
* ``string``:           string
* ``regexp``:           regular expression
* ``sub``:              subroutine header (from "sub" till "{")
* ``variable``:         variable starting with "$", "%", "@"
* ``operator``:         operator
* ``pod``:              plain old doc

PHP ("php", "php3", "php4", "php5", "php6")
-------------------------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string (of any type)
* ``comment``:          comment
* ``doctag``:           phpdoc params in comments
* ``variable``:         variable starting with "$"
* ``preprocessor``:     preprocessor marks: "<?php" and "?>"
* ``class``:            class header
* ``function``:         header of a function
* ``title``:            name of a function inside a header
* ``params``:           parentheses and everything inside them in a function's header

Scala ("scala")
---------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``doctag``:           @-tag in javadoc comment
* ``annotation``:       annotation
* ``class``:            class header
* ``title``:            class name inside a header
* ``params``:           everything in parentheses inside a class header
* ``inheritance``:      keywords "extends" and "with" inside class header

Groovy ("groovy")
-----------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string, map string keys and named argument labels
* ``regex``:            regular expression
* ``comment``:          comment
* ``doctag``:           @-tag in javadoc comment
* ``annotation``:       annotation
* ``class``:            class header
* ``title``:            class name inside a header
* ``label``:            label
* ``shebang``:          Groovy shell script header

Go ("go", "golang")
-------------------

* ``comment``:          comment
* ``string``:           string constant
* ``number``:           number
* ``keyword``:          language keywords
* ``constant``:         true false nil iota
* ``typename``:         built-in plain types (int, string etc.)
* ``built_in``:         built-in functions

Golo ("golo", "gololang")
-------------------

* ``keyword``:          language keywords
* ``literal``:          true false null
* ``typename``:         Golo type objects (DynamicObject, struct, ...)
* ``annotation``:       decorator

Gradle ("gradle")
-----------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string and character
* ``comment``:          comment
* ``regexp``:           regular expression


HTML, XML ("xml", "html", "xhtml", "rss", "atom", "xsl", "plist")
-----------------------------------------------------------------

* ``tag``:              any tag from "<" till ">"
* ``attribute``:        tag's attribute with or without value
* ``value``:            attribute's value
* ``comment``:          comment
* ``pi``:               processing instruction (<? ... ?>)
* ``doctype``:          <!DOCTYPE ... > declaration
* ``cdata``:            CDATA section

Lasso ("lasso", "ls", "lassoscript")
------------------------------------

* ``preprocessor``:     delimiters and interpreter flags
* ``shebang``:          Lasso 9 shell script header
* ``comment``:          single- or multi-line comment
* ``keyword``:          keyword
* ``literal``:          keyword representing a value
* ``built_in``:         built-in types and variables
* ``number``:           number
* ``string``:           string
* ``variable``:         variable reference starting with "#" or "$"
* ``tag``:              tag literal
* ``attribute``:        named or rest parameter in method signature
* ``subst``:            unary/binary/ternary operator symbols
* ``class``:            type, trait, or method header
* ``title``:            name following "define" inside a header

CSS ("css")
-----------

* ``tag``:              tag in selectors
* ``id``:               #some_name in selectors
* ``class``:            .some_name in selectors
* ``at_rule``:          @-rule till first "{" or ";"
* ``keyword``:          name of @-rule after @ sign
* ``attr_selector``:    attribute selector (square brackets in a[href^=http://])
* ``pseudo``:           pseudo classes and elements (:after, ::after etc.)
* ``comment``:          comment
* ``rules``:            everything from "{" till "}"
* ``rule``:             rule itself — everything inside "{" and "}"
* ``attribute``:        property name inside a rule
* ``value``:            property value inside a rule, from ":" till ";" or till the end of rule block
* ``number``:           number within a value
* ``string``:           string within a value
* ``hexcolor``:         hex color (#FFFFFF) within a value
* ``function``:         CSS function within a value
* ``important``:        "!important" symbol

SCSS ("scss")
-------------

* ``tag``:              tag in selectors
* ``id``:               #some_name in selectors
* ``class``:            .some_name in selectors
* ``at_rule``:          @-rule till first "{" or ";"
* ``attr_selector``:    attribute selector (square brackets in a[href^=http://])
* ``pseudo``:           pseudo classes and elements (:after, ::after etc.)
* ``comment``:          comment
* ``rules``:            everything from "{" till "}"
* ``attribute``:        property name inside a rule
* ``value``:            property value inside a rule, from ":" till ";" or till the end of rule block
* ``number``:           number within a value
* ``string``:           string within a value
* ``hexcolor``:         hex color (#FFFFFF) within a value
* ``function``:         CSS function within a value
* ``important``:        "!important" symbol
* ``variable``:         variable starting with "$"
* ``preprocessor``:     keywords after @

Less ("less")
-------------

* ``comment``:          comment
* ``number``:           number
* ``string``:           string
* ``attribute``:        property name
* ``variable``:         @var, @@var or @{var}
* ``keyword``:          Less keywords (when, extend etc.)
* ``function``:         Less and CSS functions (rgba, unit etc.)
* ``tag``:              tag
* ``id``:               #id
* ``class``:            .class
* ``at_rule``:          at-rule keyword (@media, @keyframes etc.)
* ``attr_selector``:    attribute selector (e.g. [href^=http://])
* ``pseudo``:           pseudo classes and elements (:hover, ::before etc.)
* ``hexcolor``:         hex color (#FFF)
* ``built_in``:         inline javascript (or whatever host language) string

Stylus ("stylus", "styl")
-------------------------

* ``at_rule``:          @-rule till first "{" or ";"
* ``attribute``:        property name inside a rule
* ``class``:            .some_name in selectors
* ``comment``:          comment
* ``function``:         Stylus function
* ``hexcolor``:         hex color (#FFFFFF) within a value
* ``id``:               #some_name in selectors
* ``number``:           number within a value
* ``pseudo``:           pseudo classes and elements (:after, ::after etc.)
* ``string``:           string within a value
* ``tag``:              tag in selectors
* ``variable``:         variable starting with "$"

Markdown ("markdown", "md", "mkdown", "mkd")
--------------------------------------------

* ``header``:            header
* ``bullet``:            list bullet
* ``emphasis``:          emphasis
* ``strong``:            strong emphasis
* ``blockquote``:        blockquote
* ``code``:              code
* ``horizontal_rule``:   horizontal rule
* ``link_label``:        link label
* ``link_url``:          link url
* ``link_reference``:    link reference

AsciiDoc ("asciidoc", "adoc")
-----------------------------

* ``header``:            heading
* ``bullet``:            list or labeled bullet
* ``emphasis``:          emphasis
* ``strong``:            strong emphasis
* ``blockquote``:        blockquote
* ``code``:              inline or block code
* ``horizontal_rule``:   horizontal rule
* ``link_label``:        link or image label
* ``link_url``:          link or image url
* ``comment``:           comment
* ``attribute``:         document attribute, block attributes
* ``label``:             admonition label

Django ("django", "jinja")
--------------------------

* ``keyword``:          HTML tag in HTML, default tags and default filters in templates
* ``tag``:              any tag from "<" till ">"
* ``comment``:          template comment, both {# .. #} and {% comment %}
* ``doctype``:          <!DOCTYPE ... > declaration
* ``attribute``:        tag's attribute with or without value
* ``value``:            attribute's value
* ``template_tag``:     template tag {% .. %}
* ``variable``:         template variable {{ .. }}
* ``filter``:           filter from "|" till the next filter or the end of tag
* ``argument``:         filter argument


Twig ("twig", "craftcms")
-------------------------

* ``keyword``:          HTML tag in HTML, default tags and default filters in templates
* ``tag``:              any tag from "<" till ">"
* ``comment``:          template comment {# .. #}
* ``doctype``:          <!DOCTYPE ... > declaration
* ``attribute``:        tag's attribute with or withou value
* ``value``:            attribute's value
* ``template_tag``:     template tag {% .. %}
* ``variable``:         template variable {{ .. }}
* ``filter``:           filter from "|" till the next filter or the end of tag
* ``argument``:         filter argument


Handlebars ("handlebars", "hbs", "html.hbs", "html.handlebars")
---------------------------------------------------------------

* ``expression``:       expression to be evaluated
* ``variable``:         variable
* ``begin-block``:      the beginning of a block
* ``end-block``:        the ending of a block
* ``string``:           string

Dust ("dust", "dst")
--------------------

* ``expression``:       expression to be evaluated
* ``variable``:         variable
* ``begin-block``:      the beginning of a block
* ``end-block``:        the ending of a block
* ``string``:           string

JSON ("json")
-------------

* ``number``:           number
* ``literal``:          "true", "false" and "null"
* ``string``:           string value
* ``attribute``:        name of an object property
* ``value``:            value of an object property

Mathematica ("mathematica", "mma")
----------------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``comment``:          comment
* ``string``:           string
* ``list``:             a list { .. } - the basic Mma structure

JavaScript ("javascript", "js")
-------------------------------

* ``keyword``:          keyword
* ``comment``:          comment
* ``number``:           number
* ``literal``:          special literal: "true", "false" and "null"
* ``built_in``:         built-in objects and functions ("window", "console", "require", etc...)
* ``string``:           string
* ``regexp``:           regular expression
* ``function``:         header of a function
* ``title``:            name of a function inside a header
* ``params``:           parentheses and everything inside them in a function's header
* ``pi``:               'use strict' processing instruction

TypeScript ("typescript", "ts")
-------------------------------

* ``keyword``:          keyword
* ``comment``:          comment
* ``number``:           number
* ``literal``:          special literal: "true", "false" and "null"
* ``built_in``:         built-in objects and functions ("window", "console", "require", etc...)
* ``string``:           string
* ``regexp``:           regular expression
* ``function``:         header of a function
* ``title``:            name of a function inside a header
* ``params``:           parentheses and everything inside them in a function's header
* ``pi``:               'use strict' processing instruction

CoffeeScript ("coffeescript", "coffee", "cson", "iced")
-------------------------------------------------------

* ``keyword``:          keyword
* ``comment``:          comment
* ``number``:           number
* ``literal``:          special literal: "true", "false" and "null"
* ``built_in``:         built-in objects and functions ("window", "console", "require", etc...)
* ``string``:           string
* ``subst``:            #{ ... } interpolation in double-quoted strings
* ``regexp``:           regular expression
* ``function``:         header of a function
* ``class``:            header of a class
* ``title``:            name of a function variable inside a header
* ``params``:           parentheses and everything inside them in a function's header
* ``property``:         @-property within class and functions

Dart ("dart")
-------------

* ``keyword``:          keyword
* ``literal``:          keyword that can be uses as identifier but have special meaning in some cases
* ``built_in``:         some of basic built in classes and function
* ``number``:           number
* ``string``:           string
* ``subst``:            in-string substitution (${...})
* ``comment``:          commment
* ``annotation``:       annotation
* ``class``:            class header from "class" till "{"
* ``title``:            class name

LiveScript ("livescript", "ls")
-------------------------------

* ``keyword``:          keyword
* ``comment``:          comment
* ``number``:           number
* ``literal``:          special literal: "true", "false" and "null"
* ``built_in``:         built-in objects and functions ("window", "console", "require", etc...)
* ``string``:           string
* ``subst``:            #{ ... } interpolation in double-quoted strings
* ``regexp``:           regular expression
* ``function``:         header of a function
* ``class``:            header of a class
* ``title``:            name of a function variable inside a header
* ``params``:           parentheses and everything inside them in a function's header
* ``property``:         @-property within class and functions

ActionScript ("actionscript", "as")
-----------------------------------

* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          keywords
* ``literal``:          literal
* ``reserved``:         reserved keyword
* ``title``:            name of declaration (package, class or function)
* ``preprocessor``:     preprocessor directive (import, include)
* ``type``:             type of returned value (for functions)
* ``package``:          package (named or not)
* ``class``:            class/interface
* ``function``:         function
* ``param``:            params of function
* ``rest_arg``:         rest argument of function

Haxe ("haxe", "hx")
-------------------

* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          keywords
* ``literal``:          literal
* ``reserved``:         reserved keyword
* ``title``:            name of declaration (package, class or function)
* ``preprocessor``:     preprocessor directive (if, else, elseif, error)
* ``type``:             type of returned value (for functions)
* ``package``:          package (named or not)
* ``class``:            class/interface
* ``function``:         function
* ``param``:            params of function
* ``rest_arg``:         rest argument of function

VBScript ("vbscript", "vbs")
----------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``built_in``:         built-in function

VB.Net ("vbnet", "vb")
----------------------

* ``keyword``:          keyword
* ``built_in``:         built-in types
* ``literal``:          "true", "false" and "nothing"
* ``string``:           string
* ``comment``:          comment
* ``xmlDocTag``:        xmldoc tag ("'''", "<!--", "-->", "<..>")
* ``preprocessor``:     preprocessor directive

Protocol Buffers ("protobuf")
-----------------------------

* ``keyword``:          keyword
* ``built_in``:         built-in types (e.g. `int64`, `string`)
* ``string``:           string
* ``number``:           number
* ``literal``:          "true" and "false"
* ``comment``:          comment
* ``class``:            message, service or enum definition header
* ``title``:            message, service or enum identifier
* ``function``:         RPC call identifier

Cap’n Proto ("capnproto", "capnp")
----------------------------------

* ``shebang``:          message identifier
* ``keyword``:          keyword
* ``built_in``:         built-in types (e.g. `Int64`, `Text`)
* ``string``:           string
* ``number``:           number or field number (e.g. @N)
* ``literal``:          "true" and "false"
* ``comment``:          comment
* ``class``:            message, interface or enum definition header
* ``title``:            message, interface or enum identifier

Thrift ("thrift")
-----------------

* ``keyword``:          keyword
* ``built_in``:         built-in types (e.g. `byte`, `i32`)
* ``string``:           string
* ``number``:           number
* ``literal``:          "true" and "false"
* ``comment``:          comment
* ``class``:            struct, enum, service or exception definition header
* ``title``:            struct, enum, service or exception identifier

HTTP ("http", "https")
----------------------

* ``request``:          first line of a request
* ``status``:           first line of a response
* ``attribute``:        header name
* ``string``:           header value or query string in a request line
* ``number``:           status code

Lua ("lua")
-----------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``built_in``:         built-in operator
* ``function``:         header of a function
* ``title``:            name of a function inside a header
* ``params``:           everything inside parentheses in a function's header
* ``long_brackets``:    multiline string in [=[ .. ]=]

Delphi ("delphi")
-----------------

* ``keyword``:          keyword
* ``comment``:          comment (of any type)
* ``number``:           number
* ``string``:           string
* ``function``:         header of a function, procedure, constructor and destructor
* ``title``:            name of a function, procedure, constructor or destructor inside a header
* ``params``:           everything inside parentheses in a function's header
* ``class``:            class' body from "= class" till "end;"

Oxygene ("oxygene")
-------------------

* ``keyword``:          keyword
* ``comment``:          comment (of any type)
* ``string``:           string/char
* ``function``:         method, destructor, procedure or function
* ``title``:            name of a function (inside function)
* ``params``:           everything inside parentheses in a function's header
* ``number``:           number
* ``class``:            class' body from "= class" till "end;"

Java ("java", "jsp")
--------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``annotaion``:        annotation
* ``class``:            class header from "class" till "{"
* ``function``:         method header
* ``title``:            class or method name
* ``params``:           everything in parentheses inside a class header
* ``inheritance``:      keywords "extends" and "implements" inside class header

Processing ("processing")
-------------------------

* ``constant``:         Processing constants
* ``variable``:         Processing special variables
* ``keyword``:          Variable types
* ``function``:         Processing setup and draw functions
* ``built_in``:         Processing built in functions

AspectJ ("aspectj")
-------------------

* ``comment``:          comment
* ``doctag``:           @-tag in javadoc comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          keyword
* ``annotation``:       annotation
* ``function``:         method and intertype method header
* ``aspect``:           aspect header from "aspect" till "{"
* ``params``:           everything in parentheses inside an aspect header
* ``inheritance``:      keywords "extends" and "implements" inside an aspect header
* ``title``:            aspect, (intertype) method name or pointcut name inside an aspect header

Fortran ("fortran", "f90", "f95")
---------------------------------

* ``comment``:          comment
* ``function``:         name of a function or a subroutine 
* ``keyword``:          language keywords (function, if) 
* ``number``:           number
* ``string``:           string constant (single or double quote)

IRPF90 ("irpf90")
-----------------

* ``comment``:          comment
* ``function``:         name of a function or a subroutine
* ``keyword``:          language keywords (function, if) 
* ``number``:           number
* ``string``:           string constant (single or double quote)

C++ ("cpp", "c", "cc", "h", "c++", "h++", "hpp")
------------------------------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string and character
* ``comment``:          comment
* ``preprocessor``:     preprocessor directive

Objective C ("objectivec", "mm", "objc", "obj-c")
-------------------------------------------------

* ``keyword``:          keyword
* ``built_in``:         Cocoa/Cocoa Touch constants and classes
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``preprocessor``:     preprocessor directive
* ``class``:            interface/implementation, protocol and forward class declaration
* ``title``:            title (id) of interface, implementation, protocol, class
* ``variable``:         properties and struct accessors

Vala ("vala")
-------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``class``:            class definitions
* ``title``:            in class definition
* ``constant``:         ALL_UPPER_CASE

C# ("cs", "csharp")
-------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``xmlDocTag``:        xmldoc tag ("///", "<!--", "-->", "<..>")
* ``class``:            class header from "class" till "{"
* ``function``:         method header
* ``title``:            title of namespace or class

F# ("fsharp", "fs")
-------------------

* ``keywords``:         keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``class``:            any custom F# type
* ``title``:            the name of a custom F# type
* ``annotation``:       any attribute

OCaml ("ocaml", "ml")
---------------------

* ``keywords``:         keyword
* ``literal``:          true false etc.
* ``number``:           number
* ``string``:           string
* ``char``:             character
* ``comment``:          comment
* ``built_in``:         built-in type (int, list etc.)
* ``type``:             variant constructor, module name
* ``tag``:              polymorphic variant tag
* ``symbol``:           type variable

D ("d")
-------

* ``comment``:          comment
* ``string``:           string constant
* ``number``:           number
* ``keyword``:          language keywords (including @attributes)
* ``constant``:         true false null
* ``built_in``:         built-in plain types (int, string etc.)

RenderMan RSL ("rsl")
---------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string (including @"..")
* ``comment``:          comment
* ``preprocessor``:     preprocessor directive
* ``shader``:           shader keywords
* ``shading``:          shading keywords
* ``built_in``:         built-in function

RenderMan RIB ("rib")
---------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``commands``:         command

Maya Embedded Language ("mel")
------------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``variable``:         variable

SQL ("sql")
-----------

* ``keyword``:          keyword (mostly SQL'92, SQL'99 and T-SQL)
* ``literal``:          special literal: "true" and "false"
* ``built_in``:         built-in type name
* ``number``:           number
* ``string``:           string (of any type: "..", '..', \`..\`)
* ``comment``:          comment

Smalltalk ("smalltalk", "st")
-----------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``symbol``:           symbol
* ``array``:            array
* ``class``:            name of a class
* ``char``:             char
* ``localvars``:        block of local variables

Lisp ("lisp")
-------------

* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``variable``:         variable
* ``literal``:          b, t and nil
* ``list``:             non-quoted list
* ``keyword``:          first symbol in a non-quoted list
* ``body``:             remainder of the non-quoted list
* ``quoted``:           quoted list, both "(quote .. )" and "'(..)"

Clojure ("clojure", "clj")
--------------------------

* ``comment``:          comments and hints
* ``string``:           string
* ``number``:           number
* ``collection``:       collections
* ``attribute``:        :keyword
* ``list``:             non-quoted list
* ``keyword``:          first symbol in a list
* ``built_in``:         built-in function name as the first symbol in a list
* ``prompt``:           REPL prompt

Scheme ("scheme")
-----------------

* ``shebang``:          script interpreter header
* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``regexp``:           regexp
* ``variable``:         single-quote 'identifier
* ``list``:             non-quoted list
* ``keyword``:          first symbol in a list
* ``built_in``:         built-in function name as the first symbol in a list
* ``literal``:          #t, #f, #\...\

Ini ("ini")
-----------

* ``title``:            title of a section
* ``value``:            value of a setting of any type
* ``string``:           string
* ``number``:           number
* ``keyword``:          boolean value keyword

Apache ("apache", "apacheconf")
-------------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``comment``:          comment
* ``literal``:          On and Off
* ``sqbracket``:        variables in rewrites "%{..}"
* ``cbracket``:         options in rewrites "[..]"
* ``tag``:              begin and end of a configuration section

Nginx ("nginx", "nginxconf")
----------------------------

* ``title``:            directive title
* ``string``:           string
* ``number``:           number
* ``comment``:          comment
* ``built_in``:         built-in constant
* ``variable``:         $-variable
* ``regexp``:           regexp

Diff ("diff", "patch")
----------------------

* ``header``:           file header
* ``chunk``:            chunk header within a file
* ``addition``:         added lines
* ``deletion``:         deleted lines
* ``change``:           changed lines

DOS ("dos", "bat", "cmd")
-------------------------

* ``keyword``:          keyword
* ``flow``:             batch control keyword
* ``stream``:           DOS special files ("con", "prn", ...)
* ``winutils``:         some commands (see dos.js specifically)
* ``envvar``:           environment variables

PowerShell ("powershell", "ps")
-------------------------------

* ``keyword``:          keyword
* ``string``:           string
* ``number``:           number
* ``comment``:          comment
* ``literal``:          special literal: "true" and "false"
* ``variable``:         variable

Bash ("bash", "sh", "zsh")
--------------------------

* ``keyword``:          keyword
* ``string``:           string
* ``number``:           number
* ``comment``:          comment
* ``literal``:          special literal: "true" and "false"
* ``variable``:         variable
* ``shebang``:          script interpreter header

Makefile ("makefile", "mk", "mak")
----------------------------------

* ``keyword``:          keyword ".PHONY" within the phony line
* ``string``:           string
* ``comment``:          comment
* ``variable``:         $(..) variable
* ``title``:            target title
* ``constant``:         constant within the initial definition

CMake ("cmake", "cmake.in")
---------------------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``envvar``:           $-variable
* ``operator``:         operator (LESS, STREQUAL, MATCHES, etc)

Nix ("nix")
-----------

* ``keyword``:          keyword
* ``built_in``:         built-in constant
* ``number``:           number
* ``string``:           single and double quotes
* ``subst``:            antiquote ${}
* ``comment``:          comment
* ``variable``:         function parameter name

NSIS ("nsis")
-------------

* ``symbol``:           directory constants
* ``number``:           number
* ``constant``:         definitions, language-strings, compiler commands
* ``variable``:         $-variable
* ``string``:           string
* ``comment``:          comment
* ``params``:           parameters
* ``keyword``:          keywords
* ``literal``:          keyword options

Axapta ("axapta")
-----------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``class``:            class header from "class" till "{"
* ``title``:            class name inside a header
* ``params``:           everything in parentheses inside a class header
* ``preprocessor``:     preprocessor directive

Oracle Rules Language ("ruleslanguage")
---------------------------------------

* ``comment``:          comment
* ``string``:           string constant
* ``number``:           number
* ``keyword``:          language keywords
* ``built_in``:         built-in functions
* ``array``:            array stem

1C ("1c")
---------

* ``keyword``:          keyword
* ``number``:           number
* ``date``:             date
* ``string``:           string
* ``comment``:          comment
* ``function``:         header of function or procedure
* ``title``:            function name inside a header
* ``params``:           everything in parentheses inside a function header
* ``preprocessor``:     preprocessor directive

x86 Assembly ("x86asm")
-----------------------

* ``keyword``:          instruction mnemonic
* ``literal``:          register name
* ``pseudo``:           assembler's pseudo instruction
* ``preprocessor``:     macro
* ``built_in``:         assembler's keyword
* ``comment``:          comment
* ``number``:           number
* ``string``:           string
* ``label``:            jump label
* ``argument``:         macro's argument

AVR assembler ("avrasm")
------------------------

* ``keyword``:          keyword
* ``built_in``:         pre-defined register
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``label``:            label
* ``preprocessor``:     preprocessor directive
* ``localvars``:        substitution in .macro

VHDL ("vhdl")
-------------

* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``literal``:          signal logical value
* ``typename``:         typename
* ``attribute``:        signal attribute

Parser3 ("parser3")
-------------------

* ``keyword``:          keyword
* ``number``:           number
* ``comment``:          comment
* ``variable``:         variable starting with "$"
* ``preprocessor``:     preprocessor directive
* ``title``:            user-defined name starting with "@"

LiveCode Server ("livecodeserver")
----------------------------------

* ``variable``:         variable starting with "g", "t", "p", "s", "$_"
* ``string``:           string
* ``comment``:          comment
* ``number``:           number
* ``title``:            name of a command or a function
* ``keyword``:          keyword
* ``constant``:         constant
* ``operator``:         operator
* ``built_in``:         built_in functions and commands
* ``function``:         header of a function
* ``command``:          header of a command
* ``preprocessor``:     preprocessor marks: "<?", "<?rev", "<?lc", "<?livecode" and "?>"

TeX ("tex")
-----------

* ``comment``:          comment
* ``number``:           number
* ``command``:          command
* ``parameter``:        parameter
* ``formula``:          formula
* ``special``:          special symbol

Haskell ("haskell", "hs")
-------------------------

* ``comment``:          comment
* ``pragma``:           GHC pragma
* ``preprocessor``:     CPP preprocessor directive
* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``title``:            function or variable name
* ``type``:             value, type or type class constructor name (i.e. capitalized)
* ``container``:        (..., ...) or {...; ...} list in declaration or record
* ``module``:           module declaration
* ``import``:           import declaration
* ``class``:            type class or instance declaration
* ``typedef``:          type declaration (type, newtype, data)
* ``default``:          default declaration
* ``infix``:            infix declaration
* ``foreign``:          FFI declaration
* ``shebang``:          shebang line

Elm ("elm")
-------------------------

* ``comment``:          comment
* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``title``:            function or variable name
* ``type``:             value or type constructor name (i.e. capitalized)
* ``container``:        (..., ...) or {...; ...} list in declaration or record
* ``module``:           module declaration
* ``import``:           import declaration
* ``typedef``:          type declaration (type, type alias)
* ``infix``:            infix declaration
* ``foreign``:          javascript interop declaration

Erlang ("erlang", "erl")
------------------------

* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          keyword
* ``record_name``:      record access (#record_name)
* ``title``:            name of declaration function
* ``variable``:         variable (starts with capital letter or with _)
* ``pp``:.keywords      module's attribute (-attribute)
* ``function_name``:    atom or atom:atom in case of function call

Elixir ("elixir")
-----------------

*  ``keyword``:         keyword
*  ``string``:          string
*  ``subst``:           in-string substitution (#{...})
*  ``comment``:         comment
*  ``function``:        function header "def some_name(...):"
*  ``class``:           defmodule and defrecord headers
*  ``title``:           name of a function or a module inside a header
*  ``symbol``:          atom
*  ``constant``:        name of a module
*  ``number``:          number
*  ``variable``:        variable
*  ``regexp``:          regexp

Rust ("rust", "rs")
-------------------

* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          keyword
* ``title``:            name of declaration
* ``preprocessor``:     preprocessor directive

Matlab ("matlab")
-----------------

* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          keyword
* ``title``:            function name
* ``function``:         function
* ``param``:            params of function
* ``matrix``:           matrix in [ .. ]
* ``cell``:             cell in { .. }

Scilab ("scilab", "sci")
------------------------

* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          keyword
* ``title``:            function name
* ``function``:         function
* ``param``:            params of function
* ``matrix``:           matrix in [ .. ]

R ("r")
-------

* ``comment``:          comment
* ``string``:           string constant
* ``number``:           number
* ``keyword``:          language keywords (function, if) plus "structural" functions (attach, require, setClass)
* ``literal``:          special literal: TRUE, FALSE, NULL, NA, etc.

OpenGL Shading Language ("glsl")
--------------------------------

* ``comment``:          comment
* ``number``:           number
* ``preprocessor``:     preprocessor directive
* ``keyword``:          keyword
* ``built_in``:         GLSL built-in functions and variables
* ``literal``:          true false

AppleScript ("applescript", "osascript")
----------------------------------------

* ``keyword``:          keyword
* ``command``:          core AppleScript command
* ``constant``:         AppleScript built in constant
* ``type``:             AppleScript variable type (integer, etc.)
* ``property``:         Applescript built in property (length, etc.)
* ``number``:           number
* ``string``:           string
* ``comment``:          comment
* ``title``:            name of a handler

Vim Script ("vim")
------------------

* ``keyword``:          keyword
* ``built_in``:         built-in functions
* ``string``:           string, comment
* ``number``:           number
* ``function``:         function header "function Foo(...)"
* ``title``:            name of a function
* ``params``:           everything inside parentheses in a function's header
* ``variable``:         vim variables with different visibilities "g:foo, b:bar"

Brainfuck ("brainfuck", "bf")
-----------------------------

* ``title``:            Brainfuck while loop command
* ``literal``:          Brainfuck inc and dec commands
* ``comment``:          comment
* ``string``:           Brainfuck input and output commands

Mizar ("mizar")
---------------

* ``keyword``:          keyword
* ``comment``:          comment

AutoHotkey ("autohotkey")
-------------------------

* ``keyword``:          keyword
* ``literal``:          A (active window), true, false, NOT, AND, OR
* ``built_in``:         built-in variables
* ``string``:           string
* ``comment``:          comment
* ``number``:           number
* ``var_expand``:       variable expansion (enclosed in percent sign)
* ``label``:            label, hotkey label, hotstring label

Monkey ("monkey")
-----------------

* ``keyword``:          keyword
* ``built_in``:         built-in functions, variables and types of variables
* ``literal``:          True, False, Null, And, Or, Shl, Shr, Mod
* ``string``:           string
* ``comment``:          comment
* ``number``:           number
* ``function``:         header of a function, method and constructor
* ``class``:            class header
* ``title``:            name of an alias, class, interface, function or method inside a header
* ``variable``:         self and super keywords
* ``preprocessor``:     import and preprocessor
* ``pi``:               Strict directive

FIX ("fix")
-----------

* ``attribute``:        attribute name
* ``string``:           attribute value

Gherkin ("gherkin")
-------------------

* ``keyword``:          keyword
* ``number``:           number
* ``comment``:          comment
* ``string``:           string

TP ("tp")
---------

* ``keyword``:          keyword
* ``constant``:         ON, OFF, max_speed, LPOS, JPOS, ENABLE, DISABLE, START, STOP, RESET
* ``number``:           number
* ``comment``:          comment
* ``string``:           string
* ``data``:             numeric registers, positions, position registers, etc.
* ``io``:               inputs and outputs
* ``label``:            data and io labels
* ``variable``:         system variables
* ``units``:            units (e.g. mm/sec, sec, deg)

Nimrod ("nimrod", "nim")
------------------------

* ``decorator``         pragma
* ``string``            string literal
* ``type``              variable type
* ``number``            numeric literal
* ``comment``           comment

Swift ("swift")
---------------

* ``keyword``:          keyword
* ``comment``:          comment
* ``number``:           number
* ``string``:           string
* ``literal``:          special literal: "true", "false" and "nil"
* ``built_in``:         built-in Swift functions
* ``func``:             header of a function
* ``class``:            class, protocol, enum, struct, or extension declaration
* ``title``:            name of a function or class (or protocol, etc)
* ``generics``:         generic type of a function
* ``params``:           parameters of a function
* ``type``:             a type
* ``preprocessor``:     @attributes

G-Code ("gcode", "nc")
----------------------

* ``keyword``:          G words, looping constructs and conditional operators
* ``comment``:          comment
* ``number``:           number
* ``built_in``:         trigonometric and mathematical functions
* ``title``:            M words and variable registers
* ``preprocessor``:     program number and ending character
* ``label``:            block number

Q ("k", "kdb")
--------------

* ``comment``:          comment
* ``string``:           string constant
* ``number``:           number
* ``keyword``:          language keywords
* ``constant``:         0/1b
* ``typename``:         built-in plain types (int, symbol etc.)
* ``built_in``:         built-in function

Tcl ("tcl", "tk")
-----------------

* ``keyword``:          keyword
* ``comment``:          comment
* ``symbol``:           function (proc)
* ``variable``:         variable
* ``string``:           string
* ``number``:           number

Puppet ("puppet", "pp")
-----------------------

* ``comment``:          comment
* ``string``:           string
* ``number``:           number
* ``keyword``:          classes and types
* ``constant``:         dependencies

Stata ("stata")
---------------

* ``keyword``:          commands and control flow
* ``label``:            macros (locals and globals)
* ``string``:           string
* ``comment``:          comment
* ``literal``:          built-in functions

XL ("xl", "tao")
----------------

* ``keyword``:          keywords defined in the default syntax file
* ``literal``:          names entered in the compiler (true, false, nil)
* ``type``:             basic types (integer, real, text, name, etc)
* ``built_in``:         built-in functions (sin, exp, mod, etc)
* ``module``:           names of frequently used Tao modules
* ``id``:               names of frequently used Tao functions
* ``constant``:         all-uppercase names such as HELLO
* ``variable``:         Mixed-case names such as Hello (style convention)
* ``id``:               Lower-case names such as hello
* ``string``:           Text between single or double quote, long text << >>
* ``number``:           Number values
* ``function``:         Function or variable definition
* ``import``:           Import clause

Roboconf ("graph", "instances")
-------------------------------

* ``keyword``:          keyword
* ``string``:           names of imported variables
* ``comment``:          comment
* ``facet``:            a **facet** section
* ``component``:        a **component** section
* ``instance-of``:      an **instance** section

STEP Part 21 ("p21", "step", "stp")
-----------------------------------

* ``preprocessor``:     delimiters
* ``comment``:          single- or multi-line comment
* ``keyword``:          keyword
* ``number``:           number
* ``string``:           string
* ``label``:            variable reference starting with "#"

Mercury ("mercury")
-------------------

* ``keyword``:          keyword
* ``pragma``:           compiler directive
* ``preprocessor``:     foreign language interface
* ``built_in``:         control flow, logical, implication, head-body conjunction, purity
* ``number``:           number, numcode of character
* ``comment``:          comment
* ``label``:            TODO label inside comment
* ``string``:           string
* ``constant``:         string format

Smali ("smali")
---------------

* ``string``:           string
* ``comment``:          comment
* ``keyword``:          smali keywords
* ``instruction``:      instruction
* ``class``:            classtypes
* ``function``:         function (call or signature)
* ``variable``:         variable or parameter

Verilog ("verilog", "v")
------------------------

* ``keyword``:          keyword, operator
* ``comment``:          comment
* ``typename``:         types of data, register, and net
* ``number``:           number literals (including X and Z)
* ``value``:            parameters passed to instances

Dockerfile ("dockerfile", "docker")
-----------------------------------

* ``keyword``:          instruction keyword
* ``comment``:          comment
* ``number``:           number
* ``string``:           string

PF ("pf", "pf.conf")
--------------------

* ``built_in``:         top level action, e.g. block/match/pass
* ``keyword``:          some parameter/modifier to an action (in, on, nat-to, most reserved words)
* ``literal``:          words representing special values, e.g. all, egress
* ``comment``:          comment
* ``number``:           number
* ``string``:           string
* ``variable``:         used for both macros and tables

XQuery ("xpath", "xq")
----------------------

* ``keyword``:          instruction keyword
* ``literal``:          words representing special values, e.g. all, egress
* ``comment``:          comment
* ``number``:           number
* ``string``:           string
* ``variable``:         variable
* ``decorator``:        annotations
* ``function``:         function

C/AL ("cal")
------------

* ``keyword``:          keyword
* ``comment``:          comment (of any type)
* ``number``:           number
* ``string``:           string
* ``date``:             date, time, or datetime
* ``function``:         header of a procedure
* ``title``:            name of an object or procedure inside a header
* ``params``:           everything inside parentheses in a function's header
* ``class``:            objects body
* ``variable``:         reference to variables

Inform7 ("I7")
--------------

* ``string``:           string
* ``comment``:          comment
* ``title``:            a section header or table header
* ``subst``:            a substitution inside a string
* ``kind``:             a built-in kind (thing, room, person, etc), for relevance
* ``characteristic``:   a commonly-used characteristic (open, closed, scenery, etc), for relevance
* ``verb``:             a commonly-used verb (is, understand), for relevance.
* ``misc_keyword``:     a word with specific I7 meaning (kind, rule), for relevance.

Prolog ("prolog")
-----------------

* ``atom``:             non-quoted atoms and functor names
* ``string``:           quoted atoms, strings, character code list literals, character code literals
* ``number``:           numbers
* ``variable``:         variables
* ``comment``:          comments

DNS Zone file ("dns", "zone", "bind")
-------------------------------------

* ``keyword``:          DNS resource records as defined in various RFCs
* ``operator``:         operator
* ``number``:           IPv4 and IPv6 addresses
* ``comment``:          comments

Ceylon ("ceylon")
-----------------

* ``keyword``:          keyword
* ``annotation``:       language annotation or compiler annotation
* ``string``:           string literal, part of string template, character literal
* ``number``:           number
* ``comment``:          comment

OpenSCAD ("openscad", "scad")
-----------------------------

* ``built_in``:          built-in functions (cube, sphere, translate, ...)
* ``comment``:           comments
* ``function``:          function or module definition
* ``keyword``:           keywords
* ``literal``:           words representing values (e.g. false, undef, PI)
* ``number``:            numbers
* ``params``:            parameters in function or module header or call
* ``preprocessor``:      file includes (i.e. include, use)
* ``string``:            quoted strings
* ``title``:             names of function or module in a header

ARM assembler ("armasm", "arm")
-------------------------------

* ``keyword``:          keyword (instruction mnemonics)
* ``literal``:          pre-defined register
* ``number``:           number
* ``built_in``:         constants (true, false)
* ``string``:           string
* ``comment``:          comment
* ``label``:            label
* ``preprocessor``:     preprocessor directive
* ``title``:            symbol versions

AutoIt ("autoit")
-------------------------

* ``keyword``:          keyword
* ``literal``:          True, False, And, Null, Not, Or
* ``built_in``:         built-in functions and UDF
* ``constant``:     	constant, macros
* ``variable``:         variables
* ``string``:           string
* ``comment``:          comment
* ``number``:           number
* ``preprocessor``:     AutoIt3Wrapper directives section