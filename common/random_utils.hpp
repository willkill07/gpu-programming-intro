#ifndef RANDOM_UTILS_HPP
#define RANDOM_UTILS_HPP

#include <random>
#include <type_traits>

namespace detail{

  template <typename T>
  struct uniform_distribution {
    static_assert(std::is_arithmetic<T>::value, "Type is not primitive numeric");
    using type = typename std::conditional<
      std::is_floating_point<T>::value,
      std::uniform_real_distribution<T>,
      std::uniform_int_distribution<T>>::type;
  };
}

template <typename T>
using uniform_distribution = typename detail::uniform_distribution<T>::type;

#endif
