About open source development
-----------------------------

If this is your first time contributing to an open source project, some
aspects of the development process may seem unusual to you.

-  **There is no way to “claim” issues.** People often want to “claim”
   an issue when they decide to work on it, to ensure that there isn't
   wasted work when someone else ends up working on it. This doesn't
   really work too well in open source, since someone may decide to work
   on something, and end up not having time to do it. Feel free to give
   information in an advisory fashion, but at the end of the day, we
   will take running code and rough consensus.
-  **There is a high bar for new functionality that is added.** Unlike
   in a corporate environment, where the person who wrote code
   implicitly “owns” it and can be expected to take care of it in the
   beginning of its lifetime, once a pull request is merged into an open
   source project, it immediately becomes the collective responsibility
   of all maintainers on the project. When we merge code, we are saying
   that we, the maintainers, are able to review subsequent changes and
   make a bugfix to the code. This naturally leads to a higher standard
   of contribution.

Common Mistakes To Avoid
------------------------

-  **Did you add tests?** (Or if the change is hard to test, did you
   describe how you tested your change?)

   -  We have a few motivations for why we ask for tests:

      1. to help us tell if we break it later
      2. to help us tell if the patch is correct in the first place
         (yes, we did review it, but as Knuth says, “beware of the
         following code, for I have not run it, merely proven it
         correct”)

   -  When is it OK not to add a test? Sometimes a change can't be
      conveniently tested, or the change is so obviously correct (and
      unlikely to be broken) that it's OK not to test it. On the
      contrary, if a change is seems likely (or is known to be likely)
      to be accidentally broken, it's important to put in the time to
      work out a testing strategy.

-  **Is your PR too long?**

   -  It's easier for us to review and merge small PRs. Difficulty of
      reviewing a PR scales nonlinearly with its size.
   -  When is it OK to submit a large PR? It helps a lot if there was a
      corresponding design discussion in an issue, with sign off from
      the people who are going to review your diff. We can also help
      give advice about how to split up a large change into individually
      shippable parts. Similarly, it helps if there is a complete
      description of the contents of the PR: it's easier to review code
      if we know what's inside!

-  **Comments for subtle things?** In cases where behavior of your code
   is nuanced, please include extra comments and documentation to allow
   us to better understand the intention of your code.
-  **Did you add a hack?** Sometimes a hack is the right answer. But
   usually we will have to discuss it.
-  **Do you want to touch a very core component?** In order to prevent
   major regressions, pull requests that touch core components receive
   extra scrutiny. Make sure you've discussed your changes with the team
   before undertaking major changes.
-  **Want to add a new feature?** If you want to add new features,
   comment your intention on the related issue. Our team tries to
   comment on and provide feedback to the community. It's better to have
   an open discussion with the team and the rest of the community prior
   to building new features. This helps us stay aware of what you're
   working on and increases the chance that it'll be merged.
-  **Did you touch unrelated code to the PR?** To aid in code review,
   please only include files in your pull request that are directly
   related to your changes.


