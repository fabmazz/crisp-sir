#include <cmath>

#include "types.h"

namespace Epi{

ArrayReal geometric_logp(const real_t& p, const uint& T){
    ArrayReal logp(0., T);
    for (uint t=1; t<T; t++){
        logp[t] = log(1. - p)*(t-1)+log(p);
    }
    return logp;
}

}