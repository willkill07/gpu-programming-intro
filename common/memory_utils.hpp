#ifndef MEMORY_UTILS_HPP
#define MEMORY_UTILS_HPP

#include <memory>
#include <mm_malloc.h>

namespace detail
{
  template <typename T>
  struct aligned_deleter {
    void operator()(T* ptr) {
      _mm_free(ptr);
    }
  };

  template <typename T>
  struct aligned_allocator {
    T* operator()(size_t elems) {
      return reinterpret_cast<T*>(_mm_malloc(sizeof(T) * elems, 64));
    }
  };

}

template <typename T>
std::unique_ptr<T[], detail::aligned_deleter<T>>
make_aligned_array(size_t elems) {
  static detail::aligned_deleter<T> del;
  return {detail::aligned_allocator<T>{}(elems), del};
}

#endif
