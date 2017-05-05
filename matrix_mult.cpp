#include <algorithm>
#include <functional>
#include <numeric>
#include <iostream>

#include "random_utils.hpp"
#include "timer.hpp"

using DataType = float;
using Ptr = DataType*;

int main(int argc, char* argv[]) {
  if (argc < 3)
    return EXIT_FAILURE;
  const int N = atoi(argv[1]);

  // prefer malloc as it is non-initializing
  Ptr a = reinterpret_cast<Ptr>(malloc(sizeof(DataType) * N * N));
  Ptr b = reinterpret_cast<Ptr>(malloc(sizeof(DataType) * N * N));
  Ptr c = reinterpret_cast<Ptr>(malloc(sizeof(DataType) * N * N));

  // initialize
  {
    std::mt19937 gen(atoi(argv[2]));
    uniform_distribution<DataType> dist(-1,1);
    auto random = [&] { return dist(gen); };
    std::generate_n(a, N * N, random);
    std::generate_n(b, N * N, random);
  }

  timer<> t;

  #define A(x,y) a[(x) * N + (y)]
  #define B(x,y) b[(x) * N + (y)]
  #define C(x,y) c[(x) * N + (y)]

#if defined(USE_SIMD)

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      DataType sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i,k) * B(k,i);
      C(i,j) = sum;
    }

#elif defined(USE_OPENMP)

  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      DataType sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i,k) * B(k,i);
      C(i,j) = sum;
    }

#elif defined(USE_OPENMP_SIMD)

  #pragma omp parallel for
  for (int i = 0; i < N; ++i)
    #pragma omp simd
    for (int j = 0; j < N; ++j) {
      DataType sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i,k) * B(k,i);
      C(i,j) = sum;
    }

#elif defined(USE_OPENACC)

  #pragma acc kernels loop independent
  for (int i = 0; i < N; ++i)
    #pragma acc loop independent
    for (int j = 0; j < N; ++j) {
      DataType sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i,k) * B(k,i);
      C(i,j) = sum;
    }

#else

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      DataType sum(0);
      for (int k = 0; k < N; ++k)
        sum += A(i,k) * B(k,i);
      C(i,j) = sum;
    }

#endif

  #undef A
  #undef B
  #undef C

  t.stop();
  double timeInMs = t.elapsed<std::milli>();
  std::cout << "Time:     " << timeInMs << "ms" << '\n';

  double checksum = std::accumulate(c, c + N * N, 0.0f, std::plus<double>());
  std::cout << "Checksum: " << checksum << '\n';

  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
