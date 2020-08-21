#include <ATen/core/operator_name.h>

static_assert(c10::OperatorNameView("foo", "bar").name == "foo", "");
static_assert(c10::OperatorNameView("foo", "bar").overload_name == "bar", "");
static_assert(c10::OperatorNameView::parse("foo.bar").name == "foo", "");
static_assert(c10::OperatorNameView::parse("foo.bar").overload_name == "bar", "");
