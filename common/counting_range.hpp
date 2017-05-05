#ifndef COUNTING_RANGE_HPP
#define COUNTING_RANGE_HPP

#include <iterator>

template <typename T>
struct counting_iterator
    : public std::
          iterator<std::random_access_iterator_tag, T, T, T const *, T &>
{

  using iterator =
      std::iterator<std::random_access_iterator_tag, T, T, T const *, T &>;
  using typename iterator::value_type;
  using typename iterator::reference;
  using typename iterator::difference_type;

  counting_iterator()                          = default;
  counting_iterator(counting_iterator const &) = default;
  counting_iterator(counting_iterator &&)      = default;
  counting_iterator(T val)
      : value(val)
  {
  }
  counting_iterator &operator++()
  {
    ++value;
    return *this;
  }
  counting_iterator operator++(int)
  {
    return counting_iterator(value++);
  }
  counting_iterator &operator--()
  {
    --value;
    return *this;
  }
  counting_iterator operator--(int)
  {
    return counting_iterator(value--);
  }
  reference operator*()
  {
    return value;
  }
  reference const operator*() const
  {
    return value;
  }
  bool
  operator==(const counting_iterator &other) const
  {
    return value == other.value;
  }
  bool
  operator!=(const counting_iterator &other) const
  {
    return value != other.value;
  }
  bool
  operator<(const counting_iterator &other) const
  {
    return value < other.value;
  }
  bool
  operator>(const counting_iterator &other) const
  {
    return value > other.value;
  }
  bool
  operator<=(const counting_iterator &other) const
  {
    return value <= other.value;
  }
  bool
  operator>=(const counting_iterator &other) const
  {
    return value >= other.value;
  }
  template <typename U, typename = typename std::enable_if<std::is_convertible<U, T>::value>::type>
  counting_iterator
  operator+(U v) {
    return {value + v};
  }

  difference_type
  operator-(const counting_iterator &other) const
  {
    return value - other.value;
  }

private:
  T value{0};
};

template <typename T>
struct counting_range
{
  T low{0};
  T high{0};
  explicit counting_range(T max)
      : high(max)
  {
  }
  explicit counting_range(T min, T max)
      : low(min)
      , high(max)
  {
  }
  counting_iterator<T>
  begin() const
  {
    return {low};
  }
  counting_iterator<T>
  end() const
  {
    return {high};
  }
  T
  size() const
  {
    return (high - low);
  }
};

#endif
