#pragma once

#include <TH/TH.h>

// We're defining THStorage as a custom class
#undef THStorage
#define THRealStorage TH_CONCAT_3(TH,Real,Storage)

#include "../Storage.hpp"
#include "../Traits.hpp"
#include "../tensors/THTensor.hpp"

namespace thpp {

template<typename real>
struct th_storage_traits {};

#include "storages/generic/THStorage.hpp"
#include <TH/THGenerateAllTypes.h>


template<typename real>
struct THStorage : public interface_traits<real>::storage_interface_type {
  template<typename U>
  friend class THTensor;

private:
  using interface_type = typename interface_traits<real>::storage_interface_type;
public:
  using storage_type = typename th_storage_traits<real>::storage_type;
  using scalar_type = typename interface_type::scalar_type;

  THStorage();
  THStorage(storage_type *wrapped);
  THStorage(std::size_t size);
  virtual ~THStorage();

  virtual std::size_t elementSize() const override;
  virtual std::size_t size() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual THStorage& retain() override;
  virtual THStorage& free() override;

  virtual THStorage& resize(long new_size) override;
  virtual THStorage& fill(scalar_type value) override;
  virtual THStorage& set(std::size_t ind, scalar_type value) override;
  // Doesn't do bound checking
  virtual THStorage& fast_set(std::size_t ind, scalar_type value) override;
  virtual scalar_type get(std::size_t ind) override;
  // Doesn't do bound checking
  virtual scalar_type fast_get(std::size_t ind) override;

  virtual thpp::Type type() const override;
  virtual bool isCuda() const override;
  virtual int getDevice() const override;

  virtual std::unique_ptr<Tensor> newTensor() const override;
  virtual storage_type *getRaw() const;

protected:
  storage_type *storage;
};

} // namespace thpp
