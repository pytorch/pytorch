PyTorch Governance | Mechanics
==============================

Summary
-------

PyTorch adopts a technical governance structure that is hierarchical.

* A community of **contributors** who file issues, make pull requests,
  and contribute to the project.
* A small set of **module maintainers** drive each module of the PyTorch
  project.
* They are overseen by **core maintainers**, who drive the
  overall project direction.
* The core maintainers have a **lead core maintainer**
  who is the catch-all decision maker.

All maintainers are expected to have a strong bias towards
PyTorch’s design philosophy.

Beyond the maintainers, the community is encouraged to contribute,
file issues, make proposals, review pull requests and be present
in the community. Given contributions and willingness to invest,
anyone can be accepted as a maintainer and provided write access
or ownership of parts of the codebase.

Technical governance is strictly separated from business governance.
Separating technical from business governance ensures that there is
no way for any person or company to “buy their way into” the
technical guidance of the project. Additionally, membership in
the technical governance process is for **individuals**, not companies.
That is, there are no seats reserved for specific companies, and
membership is associated with the person rather than the company
employing that person.

Module Maintainers
------------------

Modules are defined as GitHub repositories within the PyTorch org,
or as directories within the core repository
`pytorch/pytorch <https://github.com/pytorch/pytorch>`__.
Each module will have its own maintainer group. Maintainer
groups are responsible for reviewing and approving commits,
improving design, and changing the scope of the module.
Each maintainer group may adopt its own rules and procedures
for making decisions (majority vote being default). Module
maintainers have the right to dispute decisions made by other
module maintainers -- especially if it affects them. When
disputes are made, the module maintainer group should
provide a reasonable and public explanation of the dispute,
the relevant arguments, and the resolution. In the exceptional
cases where module maintainers cannot come to a conclusion
themselves, they will escalate to core maintainers for review.
The escalations are resolved by the core maintainers in
accordance with their rules and procedures.

Each maintainer group should publish publicly available
communication for their module (a vision, rough roadmap,
design docs, any disputes and dispute resolutions) so that
contributors and other interested parties understand the
future direction of the project and can participate in discussion.

Responsibilities of the maintainer includes:

* Triaging high priority issues of the module
* Triaging and reviewing and landing high priority pull requests of the module
* Supporting public documentation related to the module
* Running public developer meetings

Core Maintainers
----------------

The core maintainers are expected to have a deep understanding
of the PyTorch code base and design philosophies. Their responsibilities
include:

* Articulating a cohesive long-term vision for the project
* Negotiating and resolving contentious issues in ways
  acceptable to all parties involved
* Receiving broad requests for changes from stakeholders of
  PyTorch and evaluating / accepting them (small module-level
  requests are handled by module maintainers)

The core maintainers as a group have the power to veto any
decision made at a Module maintainer level. The core
maintainers have power to resolve disputes as they see fit.
The core maintainers should publicly articulate their
decision-making, and give a clear reasoning for their
decisions, vetoes and dispute resolution.

The core maintainers are admins of the PyTorch GitHub Org
and are listed in `Maintainers <https://pytorch.org/docs/stable/community/persons_of_interest.html>`__.

Lead Core Maintainer (BDFL)
---------------------------

There may be decisions in which the core maintainers cannot
come to a consensus. To make such difficult decisions, the
core maintainers have an assigned and publicly declared Lead
Core Maintainer amongst them, also commonly known in open-source
governance models as a BDFL.

The Lead Core Maintainer should publicly articulate their
decision-making, and give a clear reasoning for their
decisions. The Lead Core Maintainer is also responsible for
confirming or removing core maintainers.

Nominating, Confirming and Removing Maintainers
-----------------------------------------------

The Principles
~~~~~~~~~~~~~~

* Membership in module maintainer groups is given to **individuals**
  on **merit basis** after they demonstrated strong expertise of the
  component through contributions, reviews and discussions and are
  aligned with how the component fits in overall PyTorch direction.
* For membership in the maintainer group the individual has to
  demonstrate strong and continued alignment with the overall
  PyTorch principles.
* No term limits for module maintainers or core maintainers
* Light criteria of moving module maintenance to ‘emeritus’
  status if they don’t actively participate over long periods
  of time. Each module maintainer group may define the inactive
  period that’s appropriate for that module.
* The membership is for an individual, not a company.

The Process for Nomination
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Each module has its own process. Please contact module maintainers for more information.
  However, if there is no process identified, you can file a request to the core
  maintainers by submitting `this form <https://share.hsforms.com/1fh3SpHFMR2ihEBQ2orgN8A4tvhy>`__.
  Core maintainers are meeting every three months.
* If you are submitting a request to the core maintainers, the information in your request
  must include the following items:

  * The nominees depth and breadth of code, review and design
    contributions on the module
  * Testimonials (positive and negative) of the nominee’s interactions
    with the maintainers, users, and the community
  * General testimonials of support from the maintainers

* The core maintainers then evaluate all information and make
  a final decision to Confirm or Decline the nomination. The
  decision of the core maintainers has to be articulated well
  and would be public.

The Process for Removal
~~~~~~~~~~~~~~~~~~~~~~~

* Similar to the process for nomination, anyone in the community
  can nominate a person to be removed from a Module maintainer
  position or a Core maintainer position.
* A person can also self-nominate to be removed
* The core maintainers (excluding persons with conflict of
  interest) will request or put together more information around
  the following:

  * Their activity (or lack of) on the project
  * Their changing thinking of the space, which results in
    conflict with the overall direction of the project
  * Other information that makes them unfit to be a maintainer,
    such as Code of Conduct issues, their activity outside the
    scope of the project that conflicts with the project’s values
  * **Conflicts of interest**: filial or romantic relationships

* The core maintainers then evaluate all information and make
  a final decision to Confirm or Decline the removal. The decision
  of the core maintainers has to be articulated well and would be
  public.

Nominating Core Maintainers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Any core or module maintainer can nominate someone to become a
  core maintainer
* The lead maintainer (BDFL) is responsible for evaluating the
  nomination.
* The lead maintainer requests or puts together more information
  around the strength of the candidate to be a core maintainer:

  * Letters of support from other core and module maintainers
  * General letters of support from stakeholders within the PyTorch
    community
  * Any new relevant information that is befitting for the candidacy

* The lead maintainer evaluates all information and makes a final
  decision to Confirm or Decline the nomination, with a clear public
  articulation of their reasoning behind the decision.

Removing the Lead Core Maintainer and Nominating a New Lead Core Maintainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A super-majority of core maintainers (75%) can choose to
  remove the Lead Core Maintainer
* After a removal of the Lead Core Maintainer or in unforeseen
  circumstances (such as permanent unavailability of the Lead Core
  Maintainer), the core maintainers follow a Ranked-Choice voting
  method to elect a new Lead Core Maintainer.

Add, Remove, and Re-Scope Modules and Projects
----------------------------------------------

The core maintainers together are responsible for taking
decisions on adding, removing and re-scoping new modules
in the PyTorch org, either as new repositories in the
PyTorch GitHub org, or as folders in the
`pytorch/pytorch <https://github.com/pytorch/pytorch>`__
repository.

They invite proposals from members in the community
(including themselves) for such changes.
The proposals are open-ended, but should have some basic
ground-work to make a convincing case to make change. The
following is an example approach to this process:

#. Interview researchers / stakeholders, talk to community, gather issues;
#. Read papers, attend conferences, build example pipelines based on experience;
#. Create a state of the world - make sure this change is necessary,
   for example adding a new project or module is worth the maintenance
   cost; or removing a project or module will not remove too much value
   from PyTorch;
#. Create a proposal; the proposal covers the maintainership, development
   and community plan once the proposal is approved.

The core maintainers take final decisions on the proposal, articulating
the reasoning behind the decision publicly.


Decision Making
---------------

Uncontroversial Changes
~~~~~~~~~~~~~~~~~~~~~~~

Primary work happens through issues and pull requests on
GitHub. Maintainers should avoid pushing their changes directly to
the PyTorch repository, instead relying on pull requests. Approving a
pull request by a core or module maintainer allows it to be merged
without further process. Core and module maintainers, as listed on
the `Maintainers <https://pytorch.org/docs/stable/community/persons_of_interest.html>`__
page and within `CODEOWNERS <https://github.com/pytorch/pytorch/blob/master/CODEOWNERS>`__
ultimately approve these changes.

Notifying relevant experts about an issue or a pull request
is important. Reviews from experts in the given interest area are
strongly preferred, especially on pull request approvals. Failure to do
so might end up with the change being reverted by the relevant expert.

Controversial Decision Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Substantial changes in a given interest area require a GitHub issue to
be opened for discussion. This includes:

-  Any semantic or syntactic change to the PyTorch framework or library.
-  Backwards-incompatible changes to the Python or C++ API.
-  Additions to the core framework or library, including substantial new
   functionality within an existing library.
-  Removal of core features or platform support

Core and module maintainers ultimately approve these changes.

General Project Policies
~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch has been established as PyTorch a Series of LF Projects, LLC.
Policies applicable to PyTorch and participants in PyTorch, including
guidelines on the usage of trademarks, are located at https://www.lfprojects.org/policies/.

PyTorch participants acknowledge that the copyright in all new contributions
will be retained by the copyright holder as independent works of authorship
and that no contributor or copyright holder will be required to assign copyrights
to the project. Except as described below, all code contributions to the project
must be made using the 3-Clause-BSD License available here:
https://opensource.org/licenses/BSD-3-Clause (the “Project License”).
All outbound code will be made available under the Project License.
The Maintainers may approve the use of an alternative open license or
licenses for inbound or outbound contributions on an exception basis.

FAQ
---

**Q: What if I would like to own (or partly own) a part of the project
such as a feature area or domain library, for example** `Linear Algebra <https://github.com/pytorch/pytorch/tree/master/torch/linalg>`__
**or** `Torch Vision <https://github.com/pytorch/vision>`__ **?**
This is absolutely possible.
The first step is to start contributing to the existing project area and
supporting its health and success. In addition to this, you can
make a proposal through a GitHub issue for new functionality or changes
to improve the project area.

**Q: What if I am a company looking to use PyTorch internally for
development, can I be granted or purchase a board seat to drive the
project direction?** No, the PyTorch project is strictly driven by the
a maintainer project philosophy and clearly separates technical
governance from business governance. However, if you want to be
involved in sponsorship and support, you can become involved in the
PyTorch Foundation (PTF) and sponsorship through this. You can also
have individual engineers look to become maintainers, but this is
not guaranteed and is merit-based.

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
Wiki <https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions>`__ for contribution
for a walkthrough.

**Q: Can I become a committer on the project?** Unfortunately, the
current commit process to PyTorch involves an interaction with Facebook
infrastructure that can only be triggered by Facebook employees. We are
however looking at ways to expand the committer base to individuals
outside of Facebook and will provide an update when the tooling exists
to allow this.

**Q: What if I would like to deliver a PyTorch tutorial at a conference
or otherwise? Do I need to be 'officially' a committer to do this?** No,
we encourage community members to showcase their work wherever and
whenever they can. Please reach out to
`marketing@pytorch.org <mailto:marketing@pytorch.org>`__
for marketing support.
