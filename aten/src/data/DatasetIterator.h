#ifndef AT_DATASET_ITERATOR_H
#define AT_DATASET_ITERATOR_H

#include "Dataset.h"
#include <iterator>

class DatasetIterator : public std::iterator<std::forward_iterator_tag, Fields> {
private:
   uint64_t idx_ = 0;
   Dataset* dataset_;
public:

   DatasetIterator(Dataset& dataset)  {
      dataset_ = &dataset;
      idx_ = 0;
   }

   DatasetIterator(DatasetIterator& rhs) {
      DatasetIterator(*rhs.dataset_);
   }

   DatasetIterator& operator ++() {
      ++idx_;
      return *this;
   }

   DatasetIterator operator ++ (int) {
      DatasetIterator tmp(*this);
      ++idx_;
      return tmp;
   }

   friend bool operator == (const DatasetIterator& lhs, const DatasetIterator& rhs);
   friend bool operator != (const DatasetIterator& lhs, const DatasetIterator& rhs);

   Fields operator* () const {
      Fields sample;
      dataset_->get(idx_, sample);
      return sample;
   }

   Fields* operator-> () const {
      Fields sample;
      dataset_->get(idx_, sample);
      return &sample;
   }
};

bool operator == (const DatasetIterator& lhs, const DatasetIterator& rhs) {
   return lhs.dataset_ == rhs.dataset_;
}

bool operator != (const DatasetIterator& lhs, const DatasetIterator& rhs) {
   return lhs.dataset_ != rhs.dataset_;
}

typedef DatasetIterator iterator;
//typedef DatasetIterator<const Fields> const_iterator;

#endif

/**
iterator {
    iterator(const iterator&);
    ~iterator();
    iterator& operator=(const iterator&);
    iterator& operator++(); //prefix increment
    reference operator*() const;
    friend void swap(iterator& lhs, iterator& rhs); //C++11 I think
};

input_iterator : public virtual iterator {
    iterator operator++(int); //postfix increment
    value_type operator*() const;
    pointer operator->() const;
    friend bool operator==(const iterator&, const iterator&);
    friend bool operator!=(const iterator&, const iterator&);
};
//once an input iterator has been dereferenced, it is
//undefined to dereference one before that.

output_iterator : public virtual iterator {
    reference operator*() const;
    iterator operator++(int); //postfix increment
};
//dereferences may only be on the left side of an assignment
//once an input iterator has been dereferenced, it is
//undefined to dereference one before that.

forward_iterator : input_iterator, output_iterator {
    forward_iterator();
};
**/
