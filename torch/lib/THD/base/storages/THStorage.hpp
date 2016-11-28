#pragma once

#include <TH/TH.h>

// We're defining THStorage as a custom class
#undef THStorage
#define THRealStorage TH_CONCAT_3(TH,Real,Storage)

#include "../Storage.hpp"
#include "../Traits.hpp"

namespace thd {

template<typename real>
struct th_storage_traits {};

#include "base/storages/generic/THStorage.hpp"
#include <TH/THGenerateAllTypes.h>


template<typename real>
struct THStorage : public interface_traits<real>::storage_interface_type {
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

  virtual thd::Type type() const override;

protected:
  storage_type *storage;
};

} // namespace thd

