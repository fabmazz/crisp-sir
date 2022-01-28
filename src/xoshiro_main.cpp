#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include "xoshiro.h"

int main()
{
    /* Initialize with some hard-coded state */
    Xoshiro::Xoshiro256PP rng;

    /* Can be seeded (also accepts a seed sequence) */
    rng = Xoshiro::Xoshiro256PP(1234);
    rng.seed(1234);

    /* Can burn out RNGs if desired */
    rng.discard(100);

    /* Now using it in place of std::default_random_engine */

    /* Shuffling a vector */
    auto n=200;
    std::vector<int> indices(n);
   
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    std::cout << "Sample randomly-shuffled vector: [ ";
    for (auto val : indices)
        std::cout << val << " ";
    std::cout << "]" << std::endl;

    /* Generating normally-distributed numbers */
    std::cout << "Sample random numbers ~Normal(1, 3): [ ";
    std::normal_distribution<double> rnorm(1, 3);
    for (int ix = 0; ix < n; ix++)
        std::cout << rnorm(rng) << " ";
    std::cout << "]" << std::endl;

    /* Can be serialized in the same way as standard generators */
    std::stringstream ss;
    ss << rng;
    std::cout << "\nNumbers generated right after serialization: [ ";
    for (int ix = 0; ix < 2; ix++)
        std::cout << rng() << " ";
    std::cout << "]" << std::endl;

    Xoshiro::Xoshiro256PP rng2;
    ss >> rng2;
    std::cout << "\nNumbers generated with de-serialized object: [ ";
    for (int ix = 0; ix < 2; ix++)
        std::cout << rng2() << " ";
    std::cout << "]" << std::endl;
    std::cout << "(should be the same as before)" << std::endl;

    /* Could also use the jumping functionality for parallel streams */
    Xoshiro::Xoshiro256PP rng_next = rng.jump();
    std::cout << std::setprecision(4) << std::fixed;
    std::uniform_real_distribution<double> runif(0, 1);
    std::cout << "\n(Jumping states)";
    std::cout << "\nNext random numbers (original): ";
    for (int ix = 0; ix < 5; ix++)
        std::cout << runif(rng) << " ";
    std::cout << "\nNext random numbers (parallel): ";
    for (int ix = 0; ix < 5; ix++)
        std::cout << runif(rng_next) << " ";
    std::cout << std::endl;
    return 0;
}