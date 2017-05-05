#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "counting_range.hpp"
#include "random_utils.hpp"
#include "timer.hpp"

using DataType = float;
using Ptr = DataType*;

int main(int argc, char* argv[]) {
  if (argc < 3)
    return EXIT_FAILURE;
  const int N = atoi(argv[1]);
  const counting_range<int> Range(N);

  // prefer malloc as it is non-initializing
  Ptr a = reinterpret_cast<Ptr>(malloc(sizeof(DataType) * N));
  Ptr b = reinterpret_cast<Ptr>(malloc(sizeof(DataType) * N));
  Ptr c = reinterpret_cast<Ptr>(malloc(sizeof(DataType) * N));

  // initialize
  {
    std::mt19937 gen(atoi(argv[2]));
    uniform_distribution<DataType> dist(-1,1);
    auto random = [&] { return dist(gen); };
    std::generate_n(a, N, random);
    std::generate_n(b, N, random);
  }

  timer<> t;

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

  t.stop();
  double timeInMs = t.elapsed<std::milli>();
  std::cout << "Time:     " << timeInMs << "ms" << '\n';

  bool valid = std::all_of(Range.begin(), Range.end(), [&] (int i) {
    return c[i] == (a[i] + b[i]);
  });

  std::cout << "Valid? -> " << std::boolalpha << valid << '\n';

  // cleanup
  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
