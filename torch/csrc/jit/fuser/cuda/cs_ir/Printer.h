#pragma once
#include <ostream>
#include "IRVisitor.h"
#include "IR.h"

namespace Fuser
{

std::ostream &operator<<(std::ostream &os, const Expr &e);

template <typename T>
std::ostream &operator<<(std::ostream &os, const ExprNode<T> *s)
{
    return os << Expr(s);
}

class PrintVisitor : public IRVisitor
{
protected:
    std::ostream &os;

    void visit(Expr op)
    {
        IRVisitor::visit((const Expr *)&op);
    }

    int indent_count;

    void indent()
    {
        for (int i = 0; i < indent_count; i++)
            os << "  ";
    }

public:
    PrintVisitor(std::ostream &os) : os(os), indent_count(0) {}

    void visit(const Variable *op)
    {
        os << op->name;
    }

    void visit(const IntImm *val)
    {
        os << val->value;
    };

#define BINARY_PRINT(TYPE, STRING)  \
    void visit(const TYPE *op)      \
    {                               \
        os << "( ";                 \
        visit(op->a);               \
        os << " " << STRING << " "; \
        visit(op->b);               \
        os << " )";                 \
    }

    BINARY_PRINT(Add, "+")
    BINARY_PRINT(Sub, "-")
    BINARY_PRINT(Mul, "*")
    BINARY_PRINT(Div, "/")
    BINARY_PRINT(Mod, "%")
    BINARY_PRINT(LT, "<")

    void visit(const Set *op)
    {
        indent();
        visit(op->a);
        os << " = ";
        visit(op->b);
        os << ";\n";
    }

    void visit(const Tensor *op)
    {
        os << "{ " << op->name << " [";
        for (const auto shape : op->shapes)
        {
            visit(shape);
            if (!shape.same_as(*(op->shapes.end() - 1)))
                os << ", ";
        }
        os << "] (";

        for (const auto stride : op->strides)
        {
            visit(stride);
            if (!stride.same_as(*(op->strides.end() - 1)))
                os << ", ";
        }
        os << ") } ";
    }

    void visit(const TensorAccessor *op)
    {
        const Tensor *tensor = op->tensor.as<Tensor>();
        os << tensor->name << "(";

        for (const auto ind : op->indexers)
        {
            visit(ind);

            if (!ind.same_as(*(op->indexers.end() - 1)))
                os << ", ";
        }
        os << ") ";
    }

    void visit(const For *op)
    {
        indent();
        os << "For (";
        visit(op->loop_var);
        os << " in ";
        visit(op->min);
        os << ":";
        visit(op->extent);
        os << "){\n";
        indent_count++;
        visit(op->body);
        indent_count--;
        indent();
        os << "}\n";
    }

    void visit(const If *op)
    {
        indent();
        os << "if (";
        visit(op->pred);
        os << "){\n";
        indent_count++;
        visit(op->body);
        indent_count--;
        indent();
        os << "}\n";
    }

    void visit(const Attr *op)
    {

        switch (op->attr_type)
        {
        case (Attr::ATTR_TYPE::ThreadBinding):
            os << "//Attr thread ";
            visit(op->value);
            os << " bound to " << op->body.as<For>()->extent;
            break;
        case (Attr::ATTR_TYPE::Null):
            os << "Unknown attribute: ";
            break;
        }
        os << "\n";
        visit(op->body);
    }

    void visit(const Thread *op)
    {
        switch (op->thread_type)
        {
        case (Thread::THREAD_TYPE::TIDx):
            os << "threadIdx.x";
            break;
        case (Thread::THREAD_TYPE::TIDy):
            os << "threadIdx.y";
            break;
        case (Thread::THREAD_TYPE::TIDz):
            os << "threadIdx.z";
            break;
        case (Thread::THREAD_TYPE::BIDx):
            os << "blockIdx.x";
            break;
        case (Thread::THREAD_TYPE::BIDy):
            os << "blockIdx.y";
            break;
        case (Thread::THREAD_TYPE::BIDz):
            os << "blockIdx.z";
            break;
        }
    }

    void visit(const Block* op){
        for(const auto& expr: op->exprs){
            expr.accept(this);
        }

    }
};

} // namespace Fuser