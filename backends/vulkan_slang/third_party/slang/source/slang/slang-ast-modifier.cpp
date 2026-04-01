// slang-ast-modifier.cpp
#include "slang-ast-modifier.h"

#include "slang-ast-expr.h"

namespace Slang
{
const OrderedDictionary<Type*, SubtypeWitness*>& DifferentiableAttribute::
    getMapTypeToIDifferentiableWitness()
{
    for (Index i = m_mapToIDifferentiableWitness.getCount();
         i < m_typeToIDifferentiableWitnessMappings.getCount();
         i++)
        m_mapToIDifferentiableWitness.add(
            m_typeToIDifferentiableWitnessMappings[i].key,
            m_typeToIDifferentiableWitnessMappings[i].value);
    return m_mapToIDifferentiableWitness;
}

void printDiagnosticArg(StringBuilder& sb, Modifier* modifier)
{
    if (!modifier)
        return;
    if (modifier->keywordName && modifier->keywordName->text.getLength())
        sb << modifier->keywordName->text;
    if (auto hlslSemantic = as<HLSLSemantic>(modifier))
        sb << hlslSemantic->name.getContent();
    return;
}

} // namespace Slang
