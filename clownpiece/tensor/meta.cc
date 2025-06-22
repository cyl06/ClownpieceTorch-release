#include "meta.h"
namespace at{
namespace random {
/*
Random
*/
std::random_device random_device_rng = std::random_device();
uint rand_seed = random_device_rng();
std::mt19937 mt19937_rng = std::mt19937(rand_seed);

dtype rand01(std::mt19937& rng) {
    std::uniform_real_distribution<dtype> dist(0.0, 1.0);
    return dist(rng);
}

dtype randn(std::mt19937& rng, dtype mean, dtype stddev) {
    std::normal_distribution<dtype> dist(mean, stddev);
    return dist(rng);
}
};
};