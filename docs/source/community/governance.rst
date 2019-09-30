PyTorch Governance
==========================

Governance Philosophy and Guiding Tenets
-----------------------------------------

PyTorch adopts a governance structure with a small set of maintainers
driving the overall project direction with a strong bias towards
PyTorch's design philosophy where design and code contributions are
valued. Beyond the core maintainers, there is also a slightly broader
set of core developers that have the ability to directly merge pull
requests and own various parts of the core code base.

Beyond the maintainers and core devs, the community is encouraged to
contribute, file issues, make proposals, review pull requests and be
present in the community. Given contributions and willingness to
invest, anyone can be provided write access or ownership of parts of
the codebase.

Based on this governance structure, the project has the following core
operating tenets by which decisions are made and overall culture is
derived:

1. **Code contributions** matter much more than corporate sponsorship
   and independent developers are highly valued.
2. **Project influence** is gained through contributions (whether PRs,
   forum answers, code reviews or otherwise)

Key people and their functions
------------------------------

Project Maintainers
~~~~~~~~~~~~~~~~~~~

Project maintainers provide leadership and direction for the PyTorch
project. Specifics include:

-  Articulate a cohesive long-term vision for the project
-  Possess a deep understanding of the PyTorch code base
-  Negotiate and resolve contentious issues in ways acceptable to all
   parties involved

PyTorch Maintainers:

-  Adam Paszke (`apaszke <https://github.com/apaszke>`__)
-  Soumith Chintala (`soumith <https://github.com/soumith>`__)
-  Edward Yang (`ezyang <https://github.com/ezyang>`__)
-  Greg Chanan (`gchanan <https://github.com/gchanan>`__)
-  Dmytro Dzhulgakov (`dzhulgakov <https://github.com/dzhulgakov>`__)
-  (sunsetting) Sam Gross (`colesbury <https://github.com/colesbury>`__)

Core Developers
~~~~~~~~~~~~~~~

The PyTorch project is developed by a team of core developers. You can
find the list of core developers at `PyTorch Governance \| Persons of
Interest </docs/stable/community/persons_of_interest.html>`__.

While membership is determined by presence in the "PyTorch core" team in
the "PyTorch"
`organization <https://github.com/orgs/pytorch/teams/facebook>`__ on
GitHub, contribution takes many forms:

-  committing changes to the repository;
-  reviewing pull requests by others;
-  triaging bug reports on the issue tracker;
-  discussing topics on official PyTorch communication channels.

Moderators
~~~~~~~~~~

There is a group of people, some of which are not core developers,
responsible for ensuring that discussions on official communication
channels adhere to the Code of Conduct. They take action in view of
violations and help to support a healthy community. You can find the
list of moderators `here <https://discuss.pytorch.org/about>`__.

Decision Making
---------------

Uncontroversial Changes
~~~~~~~~~~~~~~~~~~~~~~~

Primary work happens through bug tracker issues and pull requests on
GitHub. Core developers should avoid pushing their changes directly to
the PyTorch repository, instead relying on pull requests. Approving a
pull request by a core developer allows it to be merged without further
process. Core Developers and Project Maintainers ultimately approve
these changes.

Notifying relevant experts about a bug tracker issue or a pull request
is important. Reviews from experts in the given interest area are
strongly preferred, especially on pull request approvals. Failure to do
so might end up with the change being reverted by the relevant expert.

Controversial decision process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Substantial changes in a given interest area require a GitHub issue to
be opened for discussion. This includes:

-  Any semantic or syntactic change to the framework.
-  Backwards-incompatible changes to the Python or Cpp API.
-  Additions to the core framework, including substantial new
   functionality within an existing library.
-  Removing core features

Project Maintainers ultimately approve these changes.

FAQ
---

**Q: What if I would like to own (or partly own) a part of the project
such as a domain api (i.e. Torch Vision)?** This is absolutely possible.
The first step is to start contributing to the existing project area and
contributing to its health and success. In addition to this, you can
make a proposal through a GitHub issue for new functionality or changes
to improve the project area.

**Q: What if I am a company looking to use PyTorch internally for
development, can I be granted or purchase a board seat to drive the
project direction?** No, the PyTorch project is strictly driven by the
maintainer-driven project philosophy and does not have a board or
vehicle to take financial contributions relating to gaining influence
over technical direction.

**Q: Does the PyTorch project support grants or ways to support
independent developers using or contributing to the project?** No, not
at this point. We are however looking at ways to better support the
community of independent developers around PyTorch. If you have
suggestions or inputs, please reach out on the PyTorch forums to
discuss.

**Q: How do I contribute code to the project?** If the change is
relatively minor, a pull request on GitHub can be opened up immediately
for review and merge by the project committers. For larger changes,
please open an issue to make a proposal to discuss prior. Please also
see the `PyTorch Contributor
Guide </docs/stable/community/contribution_guide.html>`__ for contribution
guidelines.

**Q: Can I become a committer on the project?** Unfortunately, the
current commit process to PyTorch involves an interaction with Facebook
infrastructure that can only be triggered by Facebook employees. We are
however looking at ways to expand the committer base to individuals
outside of Facebook and will provide an update when the tooling exists
to allow this.

**Q: What if i would like to deliver a PyTorch tutorial at a conference
or otherwise? Do I need to be 'officially' a committer to do this?** No,
we encourage community members to showcase their work wherever and
whenever they can. Please reach out to
`pytorch-marketing@fb.com <http://mailto:pytorch-marketing@fb.com/>`__
for marketing support.
