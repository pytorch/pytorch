VS_DEPLOYMENT_CONTENT
---------------------

.. versionadded:: 3.1

Mark a source file as content for deployment with a Windows Phone or
Windows Store application when built with a
:ref:`Visual Studio Generators`.
The value must evaluate to either ``1`` or ``0`` and may use
:manual:`generator expressions <cmake-generator-expressions(7)>`
to make the choice based on the build configuration.
The ``.vcxproj`` file entry for the source file will be
marked either ``DeploymentContent`` or ``ExcludedFromBuild``
for values ``1`` and ``0``, respectively.
