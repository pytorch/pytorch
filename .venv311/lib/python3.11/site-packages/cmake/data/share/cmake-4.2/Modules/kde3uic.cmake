# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# used internally by KDE3Macros.cmake
# neundorf@kde.org


execute_process(COMMAND ${KDE_UIC_EXECUTABLE}
   -L ${KDE_UIC_PLUGIN_DIR} -nounload -tr tr2i18n
   -impl ${KDE_UIC_H_FILE}
   ${KDE_UIC_FILE}
   OUTPUT_VARIABLE _uic_CONTENTS
   ERROR_QUIET
  )

string(REGEX REPLACE "tr2i18n\\(\"\"\\)" "QString::null" _uic_CONTENTS "${_uic_CONTENTS}" )
string(REGEX REPLACE "tr2i18n\\(\"\", \"\"\\)" "QString::null" _uic_CONTENTS "${_uic_CONTENTS}" )

file(WRITE ${KDE_UIC_CPP_FILE} "#include <kdialog.h>\n#include <klocale.h>\n\n")
file(APPEND ${KDE_UIC_CPP_FILE} "${_uic_CONTENTS}")
