[1mdiff --cc torch/csrc/jit/import_source.cpp[m
[1mindex d1cac26507,80f6cee414..0000000000[m
[1m--- a/torch/csrc/jit/import_source.cpp[m
[1m+++ b/torch/csrc/jit/import_source.cpp[m
[36m@@@ -202,10 -202,12 +201,10 @@@[m [mstruct SourceImporter [m
          }[m
  [m
          auto class_type =[m
[31m-             ClassType::create(c10::QualifiedName(qualified_classname), cu);[m
[31m-         owner.register_class(class_type);[m
[32m+             ClassType::create(c10::QualifiedName(qualified_classname), owner);[m
[32m+         owner->register_class(class_type);[m
          const auto self = SimpleSelf(class_type);[m
[31m-         cu->define(qualified_classname, definitions, resolvers, &self);[m
[31m -        owner->define(names, definitions, resolvers, &self);[m
[32m++        owner->define(qualified_classname, definitions, resolvers, &self);[m
        } else if (parsed_treeref->kind() == TK_NAMED_TUPLE_DEF) {[m
          auto named_tuple_def = NamedTupleDef(parsed_treeref);[m
  [m
[36m@@@ -256,8 -258,11 +254,8 @@@[m
        auto def = Def(p_.parseFunction(/*is_method=*/bool(self)));[m
        definitions.emplace_back(def);[m
        resolvers.emplace_back(resolver_);[m
[31m -      auto name = prefix ? QualifiedName(*prefix, def.name().name())[m
[31m -                         : QualifiedName(def.name().name());[m
[31m -      names.push_back(std::move(name));[m
      }[m
[31m-     cu.define(prefix, definitions, resolvers, self);[m
[31m -    cu_->define(names, definitions, resolvers, self);[m
[32m++    cu_->define(prefix, definitions, resolvers, self);[m
    }[m
  [m
    size_t parseVersionNumber() {[m
[1mdiff --git a/aten/src/ATen/core/jit_type.h b/aten/src/ATen/core/jit_type.h[m
[1mindex 596e924b40..6d4f33b5ea 100644[m
[1m--- a/aten/src/ATen/core/jit_type.h[m
[1m+++ b/aten/src/ATen/core/jit_type.h[m
[36m@@ -1375,10 +1375,6 @@[m [mstruct CAFFE2_API ClassType : public NamedType {[m
     return qualname();[m
   }[m
 [m
[31m-  void registerMethod(Function* method) {[m
[31m-    methods_.push_back(method);[m
[31m-  }[m
[31m-[m
   TypePtr getAttribute(const std::string& name) const {[m
     AT_ASSERT(attributeNames_.size() == attributeTypes_.size());[m
     size_t pos = 0;[m
[36m@@ -1409,6 +1405,9 @@[m [mstruct CAFFE2_API ClassType : public NamedType {[m
 [m
   Function* getMethod(const std::string& name) const;[m
   const std::vector<Function*>& methods() const;[m
[32m+[m[32m  void addMethod(Function* method) {[m
[32m+[m[32m    methods_.push_back(method);[m
[32m+[m[32m  }[m
 [m
   std::weak_ptr<CompilationUnit> compilation_unit();[m
   std::weak_ptr<const CompilationUnit> compilation_unit() const;[m
