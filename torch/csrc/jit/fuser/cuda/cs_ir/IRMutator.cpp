#include "IRMutator.h"
#include "IR.h"
#include "Printer.h"

namespace Fuser{

IRMutator::IRMutator() {
}

IRMutator::~IRMutator() {
}

Expr IRMutator::mutate(const Expr &e) {
    return e.defined() ? e.get()->mutate_expr(this) : Expr();
}

Expr IRMutator::visit(const IntImm *op) {return op;}
Expr IRMutator::visit(const Variable *op) {return op;}

namespace {
template<typename T>
Expr mutate_binary_operator(IRMutator *mutator, const T *op) {
    Expr a = mutator->mutate(op->a);
    Expr b = mutator->mutate(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
        return op;
    }
    return T::make(std::move(a), std::move(b));
    
}
}  // namespace

Expr IRMutator::visit(const Add *op) {
    return mutate_binary_operator(this, op);
}
Expr IRMutator::visit(const Sub *op) {
    return mutate_binary_operator(this, op);
}
Expr IRMutator::visit(const Mul *op) {
    return mutate_binary_operator(this, op);
}
Expr IRMutator::visit(const Div *op) {
    return mutate_binary_operator(this, op);
}
Expr IRMutator::visit(const Mod *op) {
    return mutate_binary_operator(this, op);
}
Expr IRMutator::visit(const Set *op) {
    return mutate_binary_operator(this, op);
}

Expr IRMutator::visit(const LT *op) {
    return mutate_binary_operator(this, op);
}

Expr IRMutator::visit(const Tensor *op){

    bool modded = false;
    std::vector<Expr> shapes;
    std::vector<Expr> strides;

    for(const auto &shape: op->shapes){
        auto new_shape = mutate(shape);
        shapes.push_back(new_shape);
        if(!shape.same_as(new_shape))
            modded = true;
    }

    for(const auto &stride: op->strides){
        auto new_stride = mutate(stride);
        strides.push_back(new_stride);
        if(!stride.same_as(new_stride))
            modded = true;
    }

    if(modded)
        return Tensor::make(op->ndims, shapes, strides, op->name.c_str());
    return op;
}

Expr IRMutator::visit(const TensorAccessor *op){
    auto tensor = mutate(op->tensor);
    bool mutated = !tensor.same_as(op->tensor);
    std::vector<Expr> indexers;
    for(const auto &ind: op->indexers){
        auto new_ind = mutate(ind);
        indexers.push_back(new_ind);
        mutated = mutated | !ind.same_as(new_ind);
    }
    if(mutated)
        return TensorAccessor::make(tensor, indexers);
    return op;
}

Expr IRMutator::visit(const For *op){
    Expr min = mutate(op->min);
    Expr extent = mutate(op->extent);
    Expr loop_var = mutate(op->loop_var);
    Expr body = mutate(op->body);
    if(min.same_as(op->loop_var)
        && extent.same_as(op->extent)
        && loop_var.same_as(op->loop_var)
        && body.same_as(op->body))
        return op;
    return For::make(min, extent, loop_var, body);
}

Expr IRMutator::visit(const If *op){
    Expr pred = mutate(op->pred);
    Expr body = mutate(op->body);
    if(pred.same_as(op->pred)
        && body.same_as(op->body))
        return op;
    return If::make(pred, body);
}

Expr IRMutator::visit(const Attr *op){
    Expr body = mutate(op->body);
    Expr value = mutate(op->value);
    
    if( body.same_as(op->body)
     && value.same_as(op->value))
        return op;
    return Attr::make(op->attr_type, body, value);
}

Expr IRMutator::visit(const Thread *op){
    return op;
}

Expr IRMutator::visit(const Block *op){
    std::vector<Expr> exprs;
    bool mutated = false;
    for(const auto& expr : op->exprs){
        auto new_expr = mutate(expr);
        exprs.push_back(new_expr);
        if(!expr.same_as(new_expr))
            mutated = true;
    }
    if(mutated)
        return Block::make(exprs);
    return op;
    
}

}
