#pragma once

// ${generated_comment}
${ts_lowering_sysinc}
${ts_lowering_inc}

namespace ${backend_namespace} {

TSOpVector LowerToTS(const ir::Node* node) {
    switch (node->op().op){
${lowering_dispatches}
    }
}

${lowering_definitions}

} // namespace ${backend_namespace}

