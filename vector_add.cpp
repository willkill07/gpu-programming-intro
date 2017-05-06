#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "counting_range.hpp"
#include "memory_utils.hpp"
#include "random_utils.hpp"
#include "timer.hpp"

using DataType = float;

template <typename T>
void
vector_add(const int,
           const T* __restrict__,
           const T* __restrict__,
           T* __restrict__);

int
main(int argc, char* argv[])
{
  if (argc < 3)
    return EXIT_FAILURE;
  const int                 N = atoi(argv[1]);
  const counting_range<int> Range(N);

  // prefer malloc as it is non-initializing
  auto a = make_aligned_array<DataType>(N);
  auto b = make_aligned_array<DataType>(N);
  auto c = make_aligned_array<DataType>(N);

  // initialize
  {
    std::mt19937                   gen(atoi(argv[2]));
    uniform_distribution<DataType> dist(-1, 1);
    auto                           random = [&] { return dist(gen); };
    std::generate_n(a.get(), N, random);
    std::generate_n(b.get(), N, random);
  }

  timer<> t;
  vector_add(N, a.get(), b.get(), c.get());
  t.stop();

  double timeInMs = t.elapsed<std::milli>();
  std::cout << "Time:     " << timeInMs << "ms" << '\n';

  bool valid = std::all_of(
      Range.begin(), Range.end(), [&](int i) { return c[i] == (a[i] + b[i]); });

  std::cout << "Valid? -> " << std::boolalpha << valid << '\n';

  return EXIT_SUCCESS;
}

template <typename T>
void
vector_add(const int N,
           const T* __restrict__ a,
           const T* __restrict__ b,
           T* __restrict__ c)
{
#if defined(USE_SIMD)
#pragma simd
#elif defined(USE_OPENMP)
#pragma omp parallel for
#elif defined(USE_OPENMP_SIMD)
#pragma omp parallel for simd
#elif defined(USE_OPENACC)
#pragma acc kernels loop independent
#endif
  for (int i = 0; i < N; ++i)
    c[i] = a[i] + b[i];
}
