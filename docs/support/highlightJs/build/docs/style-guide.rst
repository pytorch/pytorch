Style guide
===========

Overview
--------

Highlighted code is a single ``<code>`` element containing ``<span>``'s with pre-defined classes.
Its appearance is controlled by a normal CSS file that can use all the features of CSS.

There are however a couple of specific requirements for highlight.js styles:

* Simplicity
* Consistency
* Portability


Simplicity
----------

Overall style of highlight.js is minimalist, usable and generally not very bright.
It's discouraged to use individual styling for small elements such as parentheses, commas, quotes etc.
The ultimate goal of styling is to help people to actually read the code.


Consistency
-----------

Highlight.js uses consistent class names across all the supported languages:
strings are always "string", comments are always "comment", numbers are always "number".
This allows to define a style that will work for all languages, not just one.
This means that language names in style definition should be avoided:

::

  /* wrong! */
  .html .tag {
    font-weight: bold;
  }
  
  /* right, works for tags in HTML and XML */ 
  .tag {
    font-weight: bold;
  }

There are also unique syntax elements that languages don't share with each other:
"attr_selector" in CSS, "doctype" in XML etc.
The best way to style them is by "packing" them into a group of selectors for a single CSS group of rules:

::

  .javadoc,
  .decorator,
  .filter .argument,
  .localvars,
  .array,
  .attr_selector,
  .pi,
  .doctype {
    color: #88F;
  }

This pattern helps keeping the number of different style rules to a minimum.
It's also easier to maintain: when a programmer adds definition of a new language it's easier
to stack its unique elements to existing groups instead of trying to define its style from scratch
(something that programmers are known to be not very good at).

The only case where you *should* include a language class name is when its commonly named element should not adhere to general rules.
For example a class ``title`` serves in most languages as a title of a function or class definition and has a unique color.
If we want it in, say, HTML to be black as the general text we should specify a language for this rule:

::

  .html .tag. .title {
    color: black;
  }


Portability
-----------

CSS for syntax highlighting should not interfere with the main site styling.
To achieve this all style rules are defined within ``pre`` element:

::

  pre .string {
    color: red;
  }
  
  pre .number {
    color: green;
  }


CSS file header
---------------

A good idea is to include a comment at the top of your contributed CSS file that properly attributes your work:

::

  /*
  
  Mean-and-clean style (c) John Smith <email@domain.com>
  
  */


Contributing
------------

Follow the :doc:`style contributor checklist </style-contribution>`.
