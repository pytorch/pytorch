The components are:

``<major>.<minor>``

  The VS major and minor version numbers.
  These are the same as the release version numbers.

``<date>``

  A build date in the format ``MMMDD``, where ``MMM`` is a month index
  since an epoch used by Microsoft, and ``DD`` is a day in that month.

``<build>``

  A build index on the day represented by ``<date>``.

The build number is reported by ``vswhere`` as ``installationVersion``.
For example, VS 16.11.10 has build number ``16.11.32126.315``.
