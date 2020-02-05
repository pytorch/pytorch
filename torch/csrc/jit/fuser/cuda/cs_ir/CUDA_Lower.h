#pragma once
#include <ostream>
#include <map>
#include <sstream>
#include <string>
#include "IRVisitor.h"
#include "IR.h"

#include "Printer.h"

namespace Fuser
{


class CUDALower : public PrintVisitor
{

    std::map<const Variable*, std::string> var_lookup;
    std::map<const Thread::THREAD_TYPE, Expr> thread_lookup;
    friend std::ostream& lower(std::ostream &os, Expr container, std::vector<Expr> tensor_list, std::string kernel_name);

public:
    
    CUDALower(std::ostream &os) : PrintVisitor(os){}

    void visit(const Variable* var){
        if(var_lookup.find(var) != var_lookup.end())
            os << var_lookup[var];
        else
            os << var->name;
    }

    void visit(const TensorAccessor *op)
    {
        const Tensor *tensor = op->tensor.as<Tensor>();
        os << tensor->name << ".data[";

        for(size_t i = 0; i<op->indexers.size(); ++i){
            PrintVisitor::visit(op->indexers[i]);
            os<<" * ";
            PrintVisitor::visit(op->tensor.as<Tensor>()->strides[i]);
            if(i != op->indexers.size()-1)
                os<<" + ";
        }
        os << "]";
    }

    void visit(const For *op)
    {
        if(op->loop_var.as<Thread>()){
            PrintVisitor::visit(op->body);
            return;
        }
        indent();
        os << "For(size_t ";
        PrintVisitor::visit(op->loop_var);
        os << " = ";
        PrintVisitor::visit(op->min);
        PrintVisitor::visit(op->extent);
        os << " ; ++";
        PrintVisitor::visit(op->loop_var);
        os << "){\n";
        indent_count++;
        PrintVisitor::visit(op->body);
        indent_count--;
        indent();
        os << "}\n";
    }

    void visit(const Attr *op){
        if(op->attr_type == Attr::ATTR_TYPE::ThreadBinding)
            if(thread_lookup.find(op->value.as<Thread>()->thread_type) == thread_lookup.end())
                thread_lookup[op->value.as<Thread>()->thread_type] = op->body.as<For>()->extent;
        PrintVisitor::visit(op->body);
    }
    
    static std::ostream& lower(std::ostream &os, Expr container, std::vector<Expr> tensor_list, std::string kernel_name){
        std::map<const Variable*, std::string> var_lookup;
        for(const auto& T : tensor_list){
            const auto& tensor = T.as<Tensor>();
            assert(tensor);
            for(int i=0; i<tensor->ndims; i++){
                const auto& size = tensor->shapes[i].as<Variable>();
                assert(size);
                var_lookup[size] = tensor->name + ".shapes[" + std::to_string(i) + "]";
                const auto& stride = tensor->strides[i].as<Variable>();
                assert(stride);
                var_lookup[stride] = tensor->name + ".strides[" + std::to_string(i) + "]";
            }
        }

        //os<<"\n__global__\n"<<"template<typename T>\nvoid "<<kernel_name<<"( ";
        os<<"\ntemplate<typename T>\n__global__\nvoid "<<kernel_name<<"( ";
        for(const auto& T : tensor_list){
            const auto& tensor = T.as<Tensor>();
            assert(tensor);
            os<<"T "<<tensor->name;
            if(!T.same_as(tensor_list[tensor_list.size()-1]))
                os<<", ";

        }
        os<<" ){\n";
        CUDALower lower(os);
        lower.var_lookup = var_lookup;
        lower.indent_count++;
        container.accept(&lower);
        os<<"}"<<std::endl;
        std::vector<Thread::THREAD_TYPE>types {
            Thread::THREAD_TYPE::BIDx,
            Thread::THREAD_TYPE::BIDy,
            Thread::THREAD_TYPE::BIDz,
            Thread::THREAD_TYPE::TIDx,
            Thread::THREAD_TYPE::TIDy,
            Thread::THREAD_TYPE::TIDz};
        std::vector<Expr> bindings;
        for(const auto& type : types)
            if(lower.thread_lookup.find(type) != lower.thread_lookup.end())
                bindings.push_back(lower.thread_lookup[type]);
            else
                bindings.push_back(IntImm::make(1));

/*
        os << "Thread binding: dim3(";
        bindings[0].accept(&lower);
        os<<", ";
        bindings[1].accept(&lower);
        os<<", ";
        bindings[2].accept(&lower);
        os<<") dim3(";
        bindings[3].accept(&lower);
        os<<", ";
        bindings[4].accept(&lower);
        os<<", ";
        bindings[5].accept(&lower);
        os<<")";
*/

        return os;
        //std::cout<<lower.thread_lookup[]<<std::endl;
    }

};

} // namespace Fuser
