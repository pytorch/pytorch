#pragma once

namespace torch { namespace jit {
struct Graph;
struct ArgumentSpec;
void PropagateInputShapes(Graph & graph, const ArgumentSpec & spec);

}}
