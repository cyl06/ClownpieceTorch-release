#ifndef META_H
#define META_H 1

#include <memory>
#include <vector>
#include <utility>
#include <random>

namespace at {
/*
Type Definitions
*/
template<class T>
using wptr=std::weak_ptr<T>;
template<class T>
using sptr=std::shared_ptr<T>;

using dtype=float;
using shape_t=std::vector<int>;
using stride_t=std::vector<int>;

using slice_t=std::pair<int, int>;

using veci=std::vector<int>;

template<class T>
using vec=std::vector<T>;

/*
Random
*/
namespace random {
  extern std::random_device random_device_rng;
  extern uint rand_seed;
  extern std::mt19937 mt19937_rng;

  dtype rand01(std::mt19937& rng);
  dtype randn(std::mt19937& rng, dtype mean, dtype stddev);
};

};

#endif
