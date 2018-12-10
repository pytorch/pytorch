/*
 *Copyright (c) 2018 Intel Corporation.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */

#ifndef _COMPUTATION_WEB_HPP_
#define _COMPUTATION_WEB_HPP_

#include <vector>
#include <list>
#include <sched.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <assert.h>
#include "utils.hpp"

namespace ideep {
namespace utils {

class computation_web {
public:
  computation_web() = default;

  template<typename param_t> class node;
  template<typename param_t> class _node;

public:
#if _IDEEP4PY_WEB_OPT_ == true
  template<typename param_t>
  class parameter {
  public:
    parameter() :
        materialized_(std::make_shared<bool>(true)), creator_(nullptr),
        opts_(std::make_shared<std::vector<param_t>>(std::vector<param_t> {})) {}
    parameter(const parameter& t) :
        materialized_(t.get_materialized()), creator_(t.creator()),
        opts_(t.opts()) {}

    using cn_t = typename utils::computation_web::node<param_t>::cn_t;

    static void computation_param_materialize(const param_t& t) {
      if (t.is_materialized())
        return;
      computation_web::template executor<param_t>::trigger_evaluation(t);
      return;
    }

  public:
    inline void unmark_materialized() { *materialized_.get() = false; }
    inline void mark_materialized() { *materialized_.get() = true; }
    inline bool is_materialized() const { return *materialized_.get(); }
    inline std::shared_ptr<bool> get_materialized() const { return materialized_; }

    inline bool computation_param_is_same(const parameter& t) const {
      return t.get_materialized().get() == materialized_.get() &&
          t.creator().get() == creator_.get() ? true : false;
    }

  public:
    inline cn_t creator() const { return creator_; }
    inline void set_creator(cn_t cn) { creator_ = cn; }
    inline void reset_creator() { creator_.reset(); }

    inline std::shared_ptr<std::vector<param_t>> opts() const { return opts_; }
    inline void set_opts(param_t& t) { opts_->push_back(t); }
    inline bool has_opts() const { /*printf("has_opts opts @ 0x%llx\n", (unsigned long long)opts_.get());*/ return opts_.get() != nullptr && opts_->size() != 0; }

    virtual bool computation_param_own_of_memory() const { return false; }

  private:
    // share materialized status among tensors
    std::shared_ptr<bool> materialized_;
    cn_t creator_;
    std::shared_ptr<std::vector<param_t>> opts_;
  };
#else
  template<typename param_t>
  class parameter {
  public:
    parameter() {}

    using cn_t = typename utils::computation_web::node<param_t>::cn_t;

    static void computation_param_materialize(const param_t& t) { return; }

  public:
    inline void unmark_materialized() { return; }
    inline void mark_materialized() { return; }
    inline bool is_materialized() const { return false; }
    inline std::shared_ptr<bool> get_materialized() const { return std::make_shared<bool>(false); }

    inline bool computation_param_is_same(const parameter& t) const {
      return false;
    }

  public:
    inline cn_t creator() const { return nullptr; }
    inline void set_creator(cn_t cn) { return; }
    inline void reset_creator() { return; }

    inline std::shared_ptr<std::vector<param_t>> opts() const { return nullptr; }
    inline void set_opts(param_t& t) { return; }
    inline bool has_opts() const { return false; }

    bool computation_param_own_of_memory() const { return false; }
  };
#endif

  template<typename param_t>
  class node {
  public:
    node() = default;

    typedef enum {
      CN_FUSION_CONV = 0,
      CN_FUSION_RELU,
      CN_FUSION_SUM,
      CN_FUSION_BN,
      CN_FUSION_NA,
    } fusion_type_t;

    typedef enum {
      CN_PROP_FORWARD = 0,
      CN_PROP_BACKWARD,
      CN_PROP_NA,
    } prop_kind_t;

    typedef struct _fusion_attr_ {
      fusion_type_t ftype;
      std::vector<float> fattrs;
      std::vector<param_t> deps;
    } fusion_attr_t;

    typedef typename std::shared_ptr<_node<param_t>> cn_t;

  public:
    virtual void fire_computation_node(
        std::vector<param_t>& deps, std::vector<param_t>& tars) {}

    virtual cn_t fuse_if_necessary(
        std::shared_ptr<node<param_t>>, fusion_attr_t&, param_t&) {
      return nullptr;
    }
  };

  template<typename param_t>
  class _node {
  public:
    _node() : successor_(nullptr) {}

    typedef typename std::shared_ptr<_node<param_t>> cn_t;
    using fusion_attr_t = typename node<param_t>::fusion_attr_t;
    using fusion_type_t = typename node<param_t>::fusion_type_t;
    using prop_kind_t = typename node<param_t>::prop_kind_t;

  public:
    cn_t successor(void) { return successor_; }
    void set_successor(cn_t n) { successor_ = n; }
    void reset_successor() { successor_.reset(); }


    virtual std::vector<param_t>& deps() {
      static std::vector<param_t> dummy;
      return dummy;
    }

    virtual std::vector<param_t>& tars() {
      static std::vector<param_t> dummy;
      return dummy;
    }

    virtual void fire() {}
    virtual void clear() {}

    virtual cn_t fuse(cn_t cur) {
      return nullptr;
    }

    virtual void reset_creator() {}

    virtual prop_kind_t prop_kind() {
      return prop_kind_t::CN_PROP_NA;
    }

    virtual fusion_attr_t& fusion_attr() {
      static fusion_attr_t dummy{fusion_type_t::CN_FUSION_NA, {}};
      return dummy;
    }

    virtual void set_scattered() {}
    virtual bool scattered() { return true; }

  private:
    cn_t successor_;
  };

  template<typename comp_inst_t, typename param_t>
  class computation_node : public _node<param_t> {
  public:
    using cn_t = typename utils::computation_web::node<param_t>::cn_t;
    using fusion_attr_t = typename node<param_t>::fusion_attr_t;
    using fusion_type_t = typename node<param_t>::fusion_type_t;
    using prop_kind_t = typename node<param_t>::prop_kind_t;

    class computation_param {
    public:
      computation_param() = default;

      std::vector<param_t>& deps() { return deps_; }
      std::vector<param_t>& tars() { return tars_; }

      void clear() { deps_.clear(); tars_.clear(); }

    private:
      std::vector<param_t> deps_;
      std::vector<param_t> tars_;
    };

  public:
    computation_node(comp_inst_t& inst, prop_kind_t pkind,
        fusion_attr_t fattr = { fusion_type_t::CN_FUSION_NA, {}, {} }) :
        comp_(std::make_shared<comp_inst_t>(inst)),
        params_(std::make_shared<computation_param>(computation_param())),
        fattr_(std::make_shared<fusion_attr_t>(fattr)),
        pkind_(pkind), scattered_(false) {}

    computation_node(std::shared_ptr<node<param_t>>& inst_ptr, prop_kind_t pkind,
        fusion_attr_t fattr = { fusion_type_t::CN_FUSION_NA, {}, {} }) :
        comp_(inst_ptr),
        params_(std::make_shared<computation_param>(computation_param())),
        fattr_(std::make_shared<fusion_attr_t>(fattr)),
        pkind_(pkind), scattered_(false) {}

    ~computation_node() {
    }

    template<typename ...params_t>
    static std::shared_ptr<computation_node<comp_inst_t, param_t>>
    create(comp_inst_t& comp_inst, prop_kind_t pkind, params_t&... comp_tars) {
      fusion_attr_t fattr = { fusion_type_t::CN_FUSION_NA, {}, {} };
      auto cn = std::make_shared<computation_node<comp_inst_t, param_t>>(
          computation_node<comp_inst_t, param_t>(comp_inst, pkind, fattr));
      auto success = cn->bind(cn, comp_tars...);
      assert(success);
      return cn;
    }

    template<typename ...params_t>
    static std::shared_ptr<computation_node<comp_inst_t, param_t>>
    create(comp_inst_t& comp_inst, prop_kind_t pkind,
        fusion_attr_t fattr, params_t&... comp_tars) {
      auto cn = std::make_shared<computation_node<comp_inst_t, param_t>>(
          computation_node<comp_inst_t, param_t>(comp_inst, pkind, fattr));
      auto success = cn->bind(cn, comp_tars...);
      assert(success);
      return cn;
    }

    template<typename ...params_t>
    static std::shared_ptr<computation_node<comp_inst_t, param_t>>
    create(std::shared_ptr<node<param_t>> comp_ptr,
        prop_kind_t pkind, params_t&... comp_tars) {
      fusion_attr_t fattr = { fusion_type_t::CN_FUSION_NA, {}, {} };
      auto cn = std::make_shared<computation_node<comp_inst_t, param_t>>(
          computation_node<comp_inst_t, param_t>(comp_ptr, pkind, fattr));
      auto success = cn->bind(cn, comp_tars...);
      assert(success);
      return cn;
    }

    template<typename ...params_t>
    static std::shared_ptr<computation_node<comp_inst_t, param_t>>
    create(std::shared_ptr<node<param_t>> comp_ptr, prop_kind_t pkind,
        fusion_attr_t fattr, params_t&... comp_tars) {
      auto cn = std::make_shared<computation_node<comp_inst_t, param_t>>(
          computation_node<comp_inst_t, param_t>(comp_ptr, pkind, fattr));
      auto success = cn->bind(cn, comp_tars...);
      assert(success);
      return cn;
    }

  public:
    bool build_deps(const param_t& dep) {
      if (!check_or_clear(dep)) { return false; }
      deps().push_back(dep);
      return true;
    }

    bool build_deps(const std::vector<param_t>& _deps) {
      for (auto dep : _deps) {
        if (!build_deps(dep))
          return false;
      }
      return true;
    }

    template<typename ...params_t>
    bool build_deps(const param_t& dep, const params_t&... _deps) {
      if (!check_or_clear(dep)) { return false; }
      deps().push_back(dep);
      return build_deps(_deps...);
    }

    bool bind(std::shared_ptr<computation_node<
        comp_inst_t, param_t>> cn, param_t& tar) {
      if (!check_or_clear(tar)) { return false; }
      tar.unmark_materialized();
      tar.set_creator(cn);
      tars().push_back(tar);
      if (tar.has_extra()) {
        tar.get_extra()->unmark_materialized();
        tar.get_extra()->set_creator(cn);
        tars().push_back(*tar.get_extra());
      }
      return true;
    }

    template<typename ...params_t>
    bool bind(std::shared_ptr<computation_node<
        comp_inst_t, param_t>> cn, param_t& tar, params_t&... _tars) {
      if (!check_or_clear(tar)) { return false; }
      tar.unmark_materialized();
      tar.set_creator(cn);
      tars().push_back(tar);
      if (tar.has_extra()) {
        tar.get_extra()->unmark_materialized();
        tar.get_extra()->set_creator(cn);
        tars().push_back(*tar.get_extra());
      }
      return bind(cn, _tars...);
    }

    bool check_or_clear(const param_t& t) {
      auto workable = t.computation_param_own_of_memory();
      if (!workable) {
        for (auto tar: tars()) {
          tar.mark_materialized();
          tar.reset_creator();
        }
        clear();
      }
      return workable;
    }

    std::vector<param_t>& deps() { return params_->deps(); }
    std::vector<param_t>& tars() { return params_->tars(); }

    fusion_attr_t& fusion_attr() { return *fattr_.get(); }

  public:
    static void
    enqueue(std::shared_ptr<computation_node<comp_inst_t, param_t>> cn) {
      computation_web::template executor<param_t>::lazy_evaluate(cn);
      cn->unset_scattered();
    }

    void fire() {
      comp_->fire_computation_node(deps(), tars());
      for (auto tar : tars())
        tar.mark_materialized();
    }

    cn_t fuse(cn_t cur) {
      return comp_->fuse_if_necessary(
          comp_, cur->fusion_attr(), cur->tars()[0]);
    }

    void clear() { params_->clear(); params_.reset(); comp_.reset(); }

    void reset_creator() {
      std::vector<param_t> new_tars;
      for (auto tar : tars()) {
        tar.reset_creator();
        new_tars.push_back(tar);
      }
      tars().clear();
      for (auto tar : new_tars)
        tars().push_back(tar);
    }

    prop_kind_t prop_kind() { return pkind_; }

    void set_scattered() { scattered_ = true; }
    void unset_scattered() { scattered_ = false; }
    bool scattered() { return scattered_; }

  private:
    std::shared_ptr<node<param_t>> comp_;
    std::shared_ptr<computation_param> params_;
    std::shared_ptr<fusion_attr_t> fattr_;
    prop_kind_t pkind_;
    bool scattered_;
  };

  template<typename param_t>
  class executor {
  public:
    using cn_t = typename utils::computation_web::node<param_t>::cn_t;
    using prop_kind_t = typename node<param_t>::prop_kind_t;

    static void lazy_evaluate(cn_t n) {
      dag_build<param_t>::build_dag(n);
      return;
    }

    static void trigger_evaluation(const param_t& t) {
      auto d = dag_build<param_t>::fetch_dag(t);
      if (d.get() == nullptr)
        return;

      prop_kind_t pre_pkind;
      if (prop_kind_change(d->prop_kind(), pre_pkind)) {
        // keep previous prop kind
        do {
          auto residue = dag_build<param_t>::fetch_dag(pre_pkind);
          if (residue.get() == nullptr)
            break;
          parameter<param_t>::computation_param_materialize(
             residue->target_tensors().at(0));
        } while (true);

        // set current prop kind
        prop_kind_set(d->prop_kind());
      }

      // Execute dependent dag before d
      auto d_deps = d->get_deps();
      for (auto dd : d_deps)
        parameter<param_t>::computation_param_materialize(dd);

      // Try optimize dag, then run it
      // dag_optimizer<param_t>::optimize(d);
      d->execute();

      dag_build<param_t>::remove_dag(d);
      return;
    }

    static prop_kind_t& prop_kind() {
      static prop_kind_t cur_pkind = prop_kind_t::CN_PROP_NA;
      return cur_pkind;
    }

    static void prop_kind_set(prop_kind_t pkind) {
      prop_kind_t& cur_pkind = prop_kind();
      cur_pkind = pkind;
    }

    static bool prop_kind_change(prop_kind_t pkind, prop_kind_t& _pre_pkind) {
      static bool stat_init = false;
      prop_kind_t& pre_pkind = prop_kind();
      if (stat_init == false) {
        if (pkind != prop_kind_t::CN_PROP_NA) {
          pre_pkind = pkind;
          stat_init = true;
        }
      } else {
        if (pkind != prop_kind_t::CN_PROP_NA && pre_pkind != pkind) {
          _pre_pkind = pre_pkind;
          return true;
        }
      }
      return false;
    }
  };

  template<typename param_t>
  class dag {
  public:
    using cn_t = typename utils::computation_web::node<param_t>::cn_t;
    using prop_kind_t = typename node<param_t>::prop_kind_t;

    dag() : head_(nullptr), tail_(nullptr) {}

    inline bool is_associated(const param_t& t) {
      auto cur = head_;
      do {
        if (cur.get() == t.creator().get())
          return true;
        cur = cur->successor();
      } while (cur.get() != nullptr);
      return false;
    }

    std::vector<param_t> target_tensors() {
      return tail_->tars();
    }

    void build(cn_t& n) {
      if (head_.get() == nullptr) {
        head_ = tail_ = n;
      } else {
        tail_->set_successor(n);
        tail_ = n;
      }
      return;
    }

    // hidden tensor => target tensor
    std::shared_ptr<dag<param_t>> rebuild(const param_t& t) {
      auto predecessor = tensor_to_cnode(t);
      assert(predecessor.get() != nullptr);

      auto successor = predecessor->successor();
      // set tail and successor for origin dag
      set_tail(predecessor);
      predecessor->reset_successor();

      auto d = std::make_shared<dag<param_t>>(dag());
      d->set_head(successor);
      auto suc = successor;
      while (suc->successor().get() != nullptr)
        suc = suc->successor();
      d->set_tail(suc);
      return d;
    }

    inline cn_t tensor_to_cnode(const param_t& t) {
      return t.creator();
    }

    void execute() {
      for (auto cn = head_; cn.get() != nullptr; cn = cn->successor()) {
        for (auto dep : cn->deps()) {
          if (dep.creator().get() != nullptr && dep.creator()->scattered()) {
            dep.creator()->fire();
            dep.creator()->clear();
          }
        }
        cn->fire(); cn->clear();
      }
      return;
    }

    prop_kind_t prop_kind() const {
      return head_->prop_kind();
    }

    std::vector<param_t>& get_deps() {
      return head_->deps();
    }

    cn_t get_head() {return head_;}
    cn_t get_tail() {return tail_;}
    void set_head(cn_t& n) { head_ = n; }
    void set_tail(cn_t& n) { tail_ = n; }

  private:
    cn_t head_;
    cn_t tail_;
  };

  template<typename param_t>
  class dag_build {
  public:
    using cn_t = typename utils::computation_web::node<param_t>::cn_t;
    using prop_kind_t = typename node<param_t>::prop_kind_t;
    typedef typename std::shared_ptr<dag<param_t>> dag_t;

    static void build_dag(cn_t& n) {
      auto deps = n->deps();
      std::vector<dag_t> related_dags;
      for (auto d : dags()) {
        for (auto i : deps) {
          if (d->is_associated(i)) {
            bool included = false;
            for (auto rd : related_dags)
              if (d.get() == rd.get())
                included = true;
            if (included == false)
              related_dags.push_back(d);
          }
        }
      }

      if (related_dags.size() == 1) {
        auto d = related_dags[0];
        std::vector<cn_t> related_cns;
        for (auto i : deps) {
          auto cn = d->tensor_to_cnode(i);
          if (cn.get() != nullptr) {
            bool included = false;
            for (auto _cn : related_cns)
              if (cn.get() == _cn.get())
                included = true;
            if (included == false)
              related_cns.push_back(cn);
          }
        }

        // case-1 (expand): node added in the end of only related dag
        if (related_cns.size() == 1 &&
            related_cns[0]->successor().get() == nullptr) {
          d->build(n);
          return;
        // case-2 (expand): node associated to middle node and the ending node of the dag
        } else if (related_cns.size() > 1) {
          for (auto rc : related_cns) {
            if (rc->successor().get() == nullptr) {
              d->build(n);
              return;
            }
          }
        }
      }

      // case-3 (branch): node added in the middle of only related dag
      // case-4 (branch): more than one dag related to the added node
      // case-0 (new): no related dag
      auto new_dag = std::make_shared<dag<param_t>>(dag<param_t>());
      new_dag->build(n);
      dags().push_back(new_dag);
      return;
    }

    static void trim_dag(dag_t& dag, cn_t pre_opt_cn,
        cn_t opt_cn, cn_t pre_cn, cn_t cn) {
      // re-connect origin cn with opt cn
      if (pre_cn.get() == dag->get_head().get())
        dag->set_head(opt_cn);
      else
        pre_opt_cn->set_successor(opt_cn);

      if (cn.get() != dag->get_tail().get())
        opt_cn->set_successor(cn->successor());
      else
        dag->set_tail(opt_cn);

      // cut down pre_cn with cn
      pre_cn->reset_successor();
      pre_cn->reset_creator();
      pre_cn->set_scattered();
      cn->reset_successor();
      cn->reset_creator();
      cn->clear();
    }

    static dag_t fetch_dag(const param_t& t) {
      for (auto d : dags()) {
        // target tensor
        auto tts = d->target_tensors();
        for (auto tt : tts)
          if (tt == t or t.computation_param_is_same(tt))
            return d;

        // hidden tensor
        if (d->is_associated(t)) {

          // rebuild
          auto new_dag = d->rebuild(t);
          dags().push_back(new_dag);

          return d;
        }
      }
      return nullptr;
    }

    static dag_t fetch_dag(prop_kind_t pkind) {
      for (auto d : dags())
        if (d->prop_kind() == pkind)
          return d;
      return nullptr;
    }

    static void remove_dag(dag_t& d) {
      auto& _dags = dags();
      for (auto iter = _dags.begin(); iter != _dags.end(); iter++) {
        dag_t& _d = *iter;
        if (_d.get() == d.get()) {
          _dags.erase(iter);
          break;
        }
      }
    }

  private:
    static std::vector<dag_t>& dags() {
      static std::vector<dag_t> dags_;
      return dags_;
    }
  };

  template<typename param_t>
  class dag_optimizer {
  public:
    typedef typename std::shared_ptr<dag<param_t>> dag_t;
    typedef typename std::shared_ptr<_node<param_t>> cn_t;

    static void optimize(dag_t& dag) {
      cn_t pre = dag->get_head();
      cn_t cur = pre->successor();
      cn_t pre_opt = pre;

      for (; cur.get(); pre = cur, cur = cur->successor()) {
        auto opt_cn = pre->fuse(cur);
        if (opt_cn.get()) {
          dag_build<param_t>::trim_dag(dag, pre_opt, opt_cn, pre, cur);
          // reset pre and cur position
          pre = pre_opt;
          cur = opt_cn;
        }
        pre_opt = pre;
      }
    }
  };
};

}
}
#endif
