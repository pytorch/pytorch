#include "Printer.h"

std::ostream& Fuser::operator<<(std::ostream& os, const Expr& e) {
    if(!e.defined())
        return os;

    PrintVisitor printer(os);
    e.accept(&printer);
    return os;
}