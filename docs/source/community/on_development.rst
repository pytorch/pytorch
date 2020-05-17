The PyTorch Contribution Process
--------------------------------

The PyTorch organization is governed by :doc:`PyTorch
Governance <governance>`.

The PyTorch development process involves a healthy amount of open
discussions between the core development team and the community.

PyTorch operates similar to most open source projects on GitHub.
However, if you've never contributed to an open source project before,
here is the basic process.

-  **Figure out what you're going to work on.** The majority of open
   source contributions come from people scratching their own itches.
   However, if you don't know what you want to work on, or are just
   looking to get more acquainted with the project, here are some tips
   for how to find appropriate tasks:

   -  Look through the `issue
      tracker <https://github.com/pytorch/pytorch/issues/>`__ and see if
      there are any issues you know how to fix. Issues that are
      confirmed by other contributors tend to be better to investigate.
      We also maintain some labels for issues which are likely to be
      good for new people, e.g., **bootcamp** and **1hr**, although
      these labels are less well maintained.
   -  Join us on Slack and let us know you're interested in getting to
      know PyTorch. We're very happy to help out researchers and
      partners get up to speed with the codebase.

-  **Figure out the scope of your change and reach out for design
   comments on a GitHub issue if it's large.** The majority of pull
   requests are small; in that case, no need to let us know about what
   you want to do, just get cracking. But if the change is going to be
   large, it's usually a good idea to get some design comments about it
   first.

   -  If you don't know how big a change is going to be, we can help you
      figure it out! Just post about it on issues or Slack.
   -  Some feature additions are very standardized; for example, lots of
      people add new operators or optimizers to PyTorch. Design
      discussion in these cases boils down mostly to, “Do we want this
      operator/optimizer?” Giving evidence for its utility, e.g., usage
      in peer reviewed papers, or existence in other frameworks, helps a
      bit when making this case.

      - **Adding operators / algorithms from recently-released research**
        is generally not accepted, unless there is overwhelming evidence that
        this newly published work has ground-breaking results and will eventually
        become a standard in the field. If you are not sure where your method falls,
        open an issue first before implementing a PR.

   -  Core changes and refactors can be quite difficult to coordinate,
      as the pace of development on PyTorch master is quite fast.
      Definitely reach out about fundamental or cross-cutting changes;
      we can often give guidance about how to stage such changes into
      more easily reviewable pieces.

-  **Code it out!**

   -  See the technical guide for advice for working with PyTorch in a
      technical form.

-  **Open a pull request.**

   -  If you are not ready for the pull request to be reviewed, tag it
      with [WIP]. We will ignore it when doing review passes. If you are
      working on a complex change, it's good to start things off as WIP,
      because you will need to spend time looking at CI results to see
      if things worked out or not.
   -  Find an appropriate reviewer for your change. We have some folks
      who regularly go through the PR queue and try to review
      everything, but if you happen to know who the maintainer for a
      given subsystem affected by your patch is, feel free to include
      them directly on the pull request. You can learn more about this
      structure at PyTorch Subsystem Ownership.

-  **Iterate on the pull request until it's accepted!**

   -  We'll try our best to minimize the number of review roundtrips and
      block PRs only when there are major issues. For the most common
      issues in pull requests, take a look at `Common Mistakes <#common-mistakes-to-avoid>`__.
   -  Once a pull request is accepted and CI is passing, there is
      nothing else you need to do; we will merge the PR for you.


