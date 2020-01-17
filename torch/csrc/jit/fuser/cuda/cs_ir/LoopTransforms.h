#pragma once
#include "IR.h"
#include "IRVisitor.h"
#include "IRMutator.h"
#include "Printer.h"
#include <unordered_map>
#include <set>
#include <algorithm>

namespace Fuser
{

class LoopTranslate : public IRMutator{

    std::vector<Expr> loop_vars;
    

public:
    Expr visit(const Tensor *op){
        //Only working for pointwise operations where all tensors have same number of dims
        //and same sizes on those dims (not strides)
        assert(op->ndims == loop_vars.size());
        return TensorAccessor::make(op, loop_vars);       
    }
    
    static Expr translate(const Tensor* domain, const Block* exprs){
        
        const char* loop_names = "ijklmnop";
        std::vector<Expr> loops;
        std::vector<Expr> loop_vars;
        //Create loops, with unset bodies, we will use these to set tensor 
        assert(domain->shapes.size()<std::string(loop_names).length()); //better add more loop variables!
        for(int i=domain->shapes.size()-1; i>=0; i--){
            std::string loop_name{loop_names[i]};
            Expr loop_var = Variable::make(loop_name.c_str());
            loop_vars.push_back(loop_var);
            
            if(i == domain->shapes.size()-1)
                loops.push_back(For::make(IntImm::make(0), domain->shapes[i], loop_var, exprs));
            else
                loops.push_back(For::make(IntImm::make(0), domain->shapes[i], loop_var, loops[loops.size()-1]));
        }

        Expr loop_nest = loops[loops.size()-1];

        std::reverse(loop_vars.begin(), loop_vars.end());

        LoopTranslate translater;
        translater.loop_vars = loop_vars;
        return translater.mutate(loop_nest);
    }

};

class ValidateLoopsFusible : public IRVisitor
{
bool outer_found = false;
public:
    bool valid = false;
    Expr outer;
    Expr inner;

    void visit(const For *op)
    {
        if (Expr(op).same_as(outer)){
            outer_found = true;
        //outer and inner need to be exclusively next to eachother, nested.
        }else if (Expr(op).same_as(inner) && outer_found){
            valid = true;
        }else{
            outer_found = false;
        }
        IRVisitor::visit( (const Expr*) &(op->body));
    }

    //First bool is if they can be fused
    //Second bool is if outer and inner are swapped (can still be fused)
    static std::pair<bool, bool> check(Expr container, Expr outer_loop, Expr inner_loop)
    {
        ValidateLoopsFusible checker;
        checker.outer = outer_loop;
        checker.inner = inner_loop;

        container.accept(&checker);

        if (checker.valid)
            return std::pair<bool, bool>(checker.valid, false);

        checker.outer = inner_loop;
        checker.inner = outer_loop;
        container.accept(&checker);

        if (checker.valid)
            return std::pair<bool, bool>(checker.valid, true);

        return std::pair<bool, bool>(false, false);
    }
};

class LoopFuser : public IRMutator
{
std::unordered_map<Expr, Expr, ExprHash> replace;
std::set<Expr> remove;
Expr outer, inner;
friend Expr fuse(Expr container, Expr outer_loop, Expr inner_loop);
public:

    Expr visit(const For* op){
        {
            auto itr = replace.find(op);
            if(itr!=replace.end()){
                return IRMutator::mutate(itr->second);
            }
        }
        {
            auto itr = remove.find(op);
            if(itr!=remove.end()){
                return IRMutator::mutate(op->body);
            }
        }        
        return IRMutator::visit(op);
    }

    Expr visit(const Variable* op){
        auto itr = replace.find(op);
        if(itr!=replace.end()){
            return IRMutator::mutate(itr->second);
        }
        return IRMutator::visit(op);
    }

    //for(i=0;i<I;i++)
    //  a[i]
    //  for(j=0;j<J;j++)
    //      b[i, j]

    //for(k=0; k<I*J; k++)
    //a[k/J]
    //b[k/J, k%J]
    static Expr fuse(Expr container, Expr outer_loop, Expr inner_loop){
        auto can_be_fused = ValidateLoopsFusible::check(container, outer_loop, inner_loop);
        if(!can_be_fused.first){
            std::cerr<<"Could not fuse loops. Loops must be adjacent for fusion."<<std::endl;
            return Expr();
        }

        LoopFuser fuser;
        if(!can_be_fused.second){
            fuser.outer = outer_loop;
            fuser.inner = inner_loop;
        }else{
            fuser.outer = inner_loop;
            fuser.inner = outer_loop;
        }
        auto ol = outer_loop.as<For>();
        auto il = inner_loop.as<For>();
        if(ol->min.as<IntImm>()->value != 0
        || il->min.as<IntImm>()->value != 0){
            std::cerr<<"Could not fuse loops. Loop fusion requires loops starting at 0."<<std::endl;
            return Expr();
        }
        std::string new_name = ol->loop_var.as<Variable>()->name + "." + il->loop_var.as<Variable>()->name + ".fused";
        Expr new_loop_var = Variable::make(new_name.c_str());
        Expr fused_loop = For::make(
            IntImm::make(0),
            Mul::make(ol->extent, il->extent),
            new_loop_var,
            ol->body);
        fuser.replace.emplace(ol, fused_loop);
        fuser.replace.emplace(ol->loop_var, Div::make(new_loop_var, il->extent));
        fuser.replace.emplace(il->loop_var, Mod::make(new_loop_var, il->extent));
        fuser.remove.emplace(inner_loop);
        return fuser.mutate(container);
    }

};


//for(i=0;i<I;i++)
//  a[i]

//for(j=0; j<I/S; j++)
    //for(k=0; k<S ; k++)
        //if(j*S+k<I)
            //a[j*S+k]
class LoopSplitter : public IRMutator
{
friend Expr split(Expr container, Expr loop, Expr split_factor);
Expr loop;
Expr split_factor;
bool loop_found = false;
Expr ol, il;
public:
    Expr split(){
        auto l = loop.as<For>();
        auto lv = l->loop_var.as<Variable>();
        Expr lov = Variable::make( (lv->name+".outer").c_str());
        Expr liv = Variable::make( (lv->name+".inner").c_str());
        //if(lov * split_factor + liv < loop->extent){}
        Expr pred = If::make(LT::make(Add::make( Mul::make(lov, split_factor), liv ), l->extent), l->body);
        il = For::make(IntImm::make(0), split_factor, liv, pred);
        Expr extent = Add::make(l->extent, split_factor);
        extent = Sub::make(extent, IntImm::make(1));
        extent = Div::make(extent, split_factor);
        ol = For::make(IntImm::make(0), extent, lov, il);
        return ol;
    }

    Expr visit(const For* op){
        if(loop.same_as(op)){
            loop_found=true;
            return IRMutator::visit(split().as<For>());
        }
        return IRMutator::visit(op);
    }

    Expr visit(const Variable* op){
        if(loop.as<For>()->loop_var.same_as(op)){
            return Add::make( Mul::make(ol.as<For>()->loop_var , split_factor), il.as<For>()->loop_var );
        }
        return IRMutator::visit(op);
    }

    static Expr split(Expr container, Expr loop, Expr split_factor){

        LoopSplitter splitter;
        splitter.loop = loop;
        splitter.split_factor = split_factor;
        return splitter.mutate(container);
    }

};

//To-do need to check that thread is not already bound inside, or above loop context
class LoopBinder : public IRMutator
{
friend Expr bind(Expr container, Expr loop, Expr bind);

Expr loop;
Expr thread;
bool bound = false;
bool in_loop_scope = false;
public:

    Expr visit(const For* op){
        if(loop.same_as(op) && !bound){
            bound = true;

            in_loop_scope = true;
            Expr body = IRMutator::mutate(op->body);
            in_loop_scope = false;

            return IRMutator::mutate(
                Attr::make(Attr::ATTR_TYPE::ThreadBinding, 
                For::make(loop.as<For>()->min, loop.as<For>()->extent, thread, body), thread)
            );
        }
        return IRMutator::visit(op);
    }

    Expr visit(const Variable* var){
        if(loop.as<For>()->loop_var.same_as(var)){
            return thread;
        }
        return var;

    }


    static Expr bind(Expr container, Expr loop, Expr thread){
        assert(loop.as<For>());
        auto t = thread.as<Thread>();
        assert(loop.as<For>()->min.as<IntImm>());
        if(t->thread_type == Thread::THREAD_TYPE::TIDx
        || t->thread_type == Thread::THREAD_TYPE::TIDy
        || t->thread_type == Thread::THREAD_TYPE::TIDz
        )
            assert(loop.as<For>()->extent.as<IntImm>());
        assert(loop.as<For>()->min.as<IntImm>()->value==0);
        assert(thread.as<Thread>());
        LoopBinder binder;
        
        binder.loop = loop;
        binder.thread = thread;
        
        return binder.mutate(container);
    }

};

} // namespace Fuser
