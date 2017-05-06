#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

#include "memory_utils.hpp"
#include "random_utils.hpp"
#include "timer.hpp"

using DataType = float;

/*

   Given a grid, all x will be
   written to. Factors are
   indicated by their power-of-2
   representation:

   A(i,j) =
     B(i,j) / 4 +
     (B(i+1,j) +
      B(i-1,j) +
      B(i,j+1) +
      B(i,j-1)) / 8 +
     (B(i+1,j+1) +
      B(i-1,j-1) +
      B(i-1,j+1) +
      B(i+1,j-1)) / 16

   +---+---+---+---+---+---+---+
   |   |   |   |   |   |   |   |
   +---+---+---+---+---+---+---+
   |   | x | x | x | x | x |   |
   +---+---+---+---+---+---+---+
   |   | x |-4 |-3 |-4 | x |   |
   +---+---+---+---+---+---+---+
   |   | x |-3 |-2 |-3 | x |   |
   +---+---+---+---+---+---+---+
   |   | x |-4 |-3 |-4 | x |   |
   +---+---+---+---+---+---+---+
   |   | x | x | x | x | x |   |
   +---+---+---+---+---+---+---+
   |   |   |   |   |   |   |   |
   +---+---+---+---+---+---+---+

 */

template <typename T>
void
stencil(const int, T* __restrict__, T* __restrict__);

int
main(int argc, char* argv[])
{
  if (argc < 3)
    return EXIT_FAILURE;
  const int N = atoi(argv[1]);

  auto a = make_aligned_array<DataType>(N * N);
  auto b = make_aligned_array<DataType>(N * N);

  // initialize
  {
    std::mt19937                   gen(atoi(argv[2]));
    uniform_distribution<DataType> dist(-1, 1);
    auto                           random = [&] { return dist(gen); };
    std::generate_n(a.get(), N * N, random);
  }

  timer<> t;
  stencil(N, a.get(), b.get());
  t.stop();

  double timeInMs = t.elapsed<std::milli>();
  std::cout << "Time:     " << timeInMs << "ms" << '\n';

  double checksum =
      std::accumulate(a.get(), a.get() + N * N, 0.0f, std::plus<double>());
  std::cout << "Checksum: " << checksum << '\n';

  return EXIT_SUCCESS;
}

template <typename T>
void
stencil(const int N, T* __restrict__ a, T* __restrict__ b)
{

#define A(x, y) a[(x)*N + (y)]
#define B(x, y) b[(x)*N + (y)]
#define ABS(x) (((x) < 0) ? -(x) : (x))

#if defined(USE_OPENMP_SIMD) || defined(USE_OPENMP)

#pragma omp parallel
  {
#pragma omp for
    for (int i = 1; i < N - 1; ++i)
    {
      for (int j = 1; j < N - 1; ++j)
      {
        T res(0);
        for (int ii = -1; ii <= 1; ++ii)
          for (int jj = -1; jj <= 1; ++jj)
            res += A(i + ii, j + jj) / T(1 << (2 + ABS(ii) + ABS(jj)));
        B(i, j) = res;
      }
    }
#pragma omp for
    for (int i = 1; i < N - 1; ++i)
    {
      for (int j = 1; j < N - 1; ++j)
      {
        T res(0);
        for (int ii = -1; ii <= 1; ++ii)
          for (int jj = -1; jj <= 1; ++jj)
            res += B(i + ii, j + jj) / T(1 << (2 + ABS(ii) + ABS(jj)));
        A(i, j) = res;
      }
    }
  }

#elif defined(USE_OPENACC)

#pragma acc kernels loop independent
  for (int i = 1; i < N - 1; ++i)
  {
#pragma acc loop independent
    for (int j = 1; j < N - 1; ++j)
    {
      T res(0);
      for (int ii = -1; ii <= 1; ++ii)
        for (int jj = -1; jj <= 1; ++jj)
          res += A(i + ii, j + jj) / T(1 << (2 + ABS(ii) + ABS(jj)));
      B(i, j) = res;
    }
  }
#pragma acc kernels loop independent
  for (int i = 1; i < N - 1; ++i)
  {
#pragma acc loop independent
    for (int j = 1; j < N - 1; ++j)
    {
      T res(0);
      for (int ii = -1; ii <= 1; ++ii)
        for (int jj = -1; jj <= 1; ++jj)
          res += B(i + ii, j + jj) / T(1 << (2 + ABS(ii) + ABS(jj)));
      A(i, j) = res;
    }
  }

#else // SIMD or Sequential

  for (int i = 1; i < N - 1; ++i)
  {
    for (int j = 1; j < N - 1; ++j)
    {
      T res(0);
      for (int ii = -1; ii <= 1; ++ii)
        for (int jj = -1; jj <= 1; ++jj)
          res += A(i + ii, j + jj) / T(1 << (2 + ABS(ii) + ABS(jj)));
      B(i, j) = res;
    }
  }
  for (int i = 1; i < N - 1; ++i)
  {
    for (int j = 1; j < N - 1; ++j)
    {
      T res(0);
      for (int ii = -1; ii <= 1; ++ii)
        for (int jj = -1; jj <= 1; ++jj)
          res += B(i + ii, j + jj) / T(1 << (2 + ABS(ii) + ABS(jj)));
      A(i, j) = res;
    }
  }

#endif

#undef A
#undef B
#undef ABS
}
