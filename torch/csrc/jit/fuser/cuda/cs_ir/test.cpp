#include "test.h"

#include <iostream>
#include <string>
#include <memory>

#include "IR.h"
#include "Visitors.h"
#include "Printer.h"
#include "Mutators.h"
#include "LoopTransforms.h"
#include "CUDA_Lower.h"

namespace Fuser {

Expr get_tensor(int ndims = 0, const char* name = ""){

  std::vector<Expr> size;
  std::vector<Expr> stride;
  
  for(int i=0; i<ndims; i++){
    size.push_back(Variable::make(
      ("size"+std::to_string(i)).c_str()
    ));

    stride.push_back(Variable::make(
      ("stride"+std::to_string(i)).c_str()
    ));
  }

  return Tensor::make(ndims, size, stride, name);

}

std::string saxpy_codegen(std::string name){
  std::cout<<"Start"<<std::endl;

  //A print visitor
  auto A = get_tensor(3, "A");
  auto B = get_tensor(3, "B");
  auto C = get_tensor(3, "C");

  Expr my_add = Add::make(A, B);
  Expr result = Set::make(C, my_add);
  //std::cout<<"Printing a Stmt:\n"<<result<<std::endl;

  LoopTranslate loop_nest_writer;
  Expr block = Block::make({result});
  Expr loop_nest = LoopTranslate::translate(C.as<Tensor>(), block.as<Block>());
  //std::cout<<"Basic loop nest:\n"<<loop_nest<<std::endl;

  std::vector<Expr> fors = findAll<For>(loop_nest);

  Expr fused = LoopFuser::fuse(loop_nest, fors[0], fors[1]);

  fors = findAll<For>(fused);
  fused = LoopFuser::fuse(fused, fors[0], fors[1]);
  //std::cout<<"Fused:\n"<<fused<<std::endl;

  fors = findAll<For>(fused);
  auto Split = LoopSplitter::split(fused, fors[0], IntImm::make(128));
  //std::cout<<"Split:\n"<<Split<<std::endl;

  fors = findAll<For>(Split);
  auto bound = LoopBinder::bind(Split, fors[0], Thread::make(Thread::THREAD_TYPE::BIDx));
  fors = findAll<For>(bound);
  bound = LoopBinder::bind(bound, fors[1], Thread::make(Thread::THREAD_TYPE::TIDx));
  //std::cout<<"Bound to threads:\n"<<bound<<std::endl;

  //std::cout<<"CUDA code: "<<std::endl;

  std::vector<Expr> tensors;
  tensors.push_back(A);
  tensors.push_back(B);
  tensors.push_back(C);

  std::stringstream stream;
  CUDALower::lower(stream, bound, tensors, name);
  
  //std::cout<<std::endl;
  //std::cout<<"\nDone"<<std::endl;
  return stream.str();
}

}
