#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

#include "memory_utils.hpp"
#include "random_utils.hpp"
#include "timer.hpp"

using DataType = float;

template <typename T>
void
mmult(const int, const T* __restrict__, const T* __restrict__, T* __restrict__);

int
main(int argc, char* argv[])
{
  if (argc < 3)
    return EXIT_FAILURE;
  const int N = atoi(argv[1]);

  auto a = make_aligned_array<DataType>(N * N);
  auto b = make_aligned_array<DataType>(N * N);
  auto c = make_aligned_array<DataType>(N * N);

  // initialize
  {
    std::mt19937                   gen(atoi(argv[2]));
    uniform_distribution<DataType> dist(-1, 1);
    auto                           random = [&] { return dist(gen); };
    std::generate_n(a.get(), N * N, random);
    std::generate_n(b.get(), N * N, random);
  }

  timer<> t;
  mmult(N, a.get(), b.get(), c.get());
  t.stop();

  double timeInMs = t.elapsed<std::milli>();
  std::cout << "Time:     " << timeInMs << "ms" << '\n';

  double checksum =
      std::accumulate(c.get(), c.get() + N * N, 0.0f, std::plus<double>());
  std::cout << "Checksum: " << checksum << '\n';

  return EXIT_SUCCESS;
}

template <typename T>
void
mmult(const int N,
      const T* __restrict__ a,
      const T* __restrict__ b,
      T* __restrict__ c)
{

#define A(x, y) a[(x)*N + (y)]
#define B(x, y) b[(x)*N + (y)]
#define C(x, y) c[(x)*N + (y)]

#if defined(USE_OPENMP)

#pragma omp parallel for
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
    {
      T sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i, k) * B(k, i);
      C(i, j) = sum;
    }

#elif defined(USE_OPENMP_SIMD)

#pragma omp parallel for
  for (int i = 0; i < N; ++i)
#pragma omp simd
    for (int j = 0; j < N; ++j)
    {
      T sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i, k) * B(k, i);
      C(i, j) = sum;
    }

#elif defined(USE_OPENACC)

#pragma acc kernels loop independent
  for (int i = 0; i < N; ++i)
#pragma acc loop independent
    for (int j = 0; j < N; ++j)
    {
      T sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i, k) * B(k, i);
      C(i, j) = sum;
    }

#else // SIMD OR Sequential

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
    {
      T sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i, k) * B(k, i);
      C(i, j) = sum;
    }

#endif

#undef A
#undef B
#undef C
}
