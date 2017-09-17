#pragma once

#include <THC/THC.h>

// We're defining THCStorage as a custom class
#undef THCStorage
#undef THStorage
#define THCRealStorage TH_CONCAT_3(TH,CReal,Storage)

#include "../Storage.hpp"
#include "../TraitsCuda.hpp"

namespace thpp {

template<typename real>
struct thc_storage_traits {};

#include "storages/generic/THCStorage.hpp"
#include <THC/THCGenerateAllTypes.h>


template<typename real>
struct THCStorage : public interface_traits<real>::storage_interface_type {
  template<typename U>
  friend struct THCTensor;

private:
  using interface_type = typename interface_traits<real>::storage_interface_type;
public:
  using storage_type = typename thc_storage_traits<real>::storage_type;
  using scalar_type = typename interface_type::scalar_type;

  THCStorage(THCState* state);
  THCStorage(THCState* state, storage_type *wrapped);
  THCStorage(THCState* state, std::size_t size);
  virtual ~THCStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual THCStorage& retain() override;
  virtual THCStorage& free() override;

  virtual THCStorage& resize(int64_t new_size) override;
  virtual THCStorage& fill(scalar_type value) override;
  virtual THCStorage& set(std::size_t ind, scalar_type value) override;
  virtual THCStorage& fast_set(std::size_t ind, scalar_type value) override;
  virtual scalar_type get(std::size_t ind) override;
  virtual scalar_type fast_get(std::size_t ind) override;

  virtual thpp::Type type() const override;
  virtual bool isCuda() const override;
  virtual int getDevice() const override;

  virtual std::unique_ptr<Tensor> newTensor() const override;
  virtual storage_type *getRaw() const;

protected:
  storage_type *storage;
  THCState* state;
};

} // namespace thpp
