#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

template <typename Clock = std::chrono::steady_clock>
struct timer {

  void start() {
    begin = Clock::now();
  }

  void stop() {
    end = Clock::now();
  }

  template <typename Ratio = std::milli>
  double elapsed() const {
    return std::chrono::duration<double, Ratio>(end - begin).count();
  }

private:
  mutable decltype(Clock::now()) begin{Clock::now()}, end;
};

#endif
