import distutils.command.bdist_rpm as orig

from ..warnings import SetuptoolsDeprecationWarning


class bdist_rpm(orig.bdist_rpm):
    """
    Override the default bdist_rpm behavior to do the following:

    1. Run egg_info to ensure the name and version are properly calculated.
    2. Always run 'install' using --single-version-externally-managed to
       disable eggs in RPM distributions.
    """

    def run(self):
        SetuptoolsDeprecationWarning.emit(
            "Deprecated command",
            """
            bdist_rpm is deprecated and will be removed in a future version.
            Use bdist_wheel (wheel packages) instead.
            """,
            see_url="https://github.com/pypa/setuptools/issues/1988",
            due_date=(2023, 10, 30),  # Deprecation introduced in 22 Oct 2021.
        )

        # ensure distro name is up-to-date
        self.run_command('egg_info')

        orig.bdist_rpm.run(self)

    def _make_spec_file(self):
        spec = orig.bdist_rpm._make_spec_file(self)
        return [
            line.replace(
                "setup.py install ",
                "setup.py install --single-version-externally-managed ",
            ).replace("%setup", "%setup -n %{name}-%{unmangled_version}")
            for line in spec
        ]
