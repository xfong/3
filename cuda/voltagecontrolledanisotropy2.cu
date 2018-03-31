#include <stdint.h>
#include "float3.h"
#include "amul.h"

// Add voltage-controlled magnetic anisotropy field to B.
// https://www.nature.com/articles/s42005-019-0189-6.pdf
extern "C" __global__ void
addvoltagecontrolledanisotropy2(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                       float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
                       float* __restrict__ Ms_, float Ms_mul,
                       float* __restrict__ vcmaCoeff_, float vcmaCoeff_mul,
                       float* __restrict__ voltage_, float voltage_mul,
                       float* __restrict__ ux_, float ux_mul,
                       float* __restrict__ uy_, float uy_mul,
                       float* __restrict__ uz_, float uz_mul,
                       int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 u   = normalized(vmul(ux_, uy_, uz_, ux_mul, uy_mul, uz_mul, i));
        float invMs = inv_Msat(Ms_, Ms_mul, i);
        float  vcmaCoeff  = amul(vcmaCoeff_, vcmaCoeff_mul, i) * invMs;
        float  voltage  = amul(voltage_, voltage_mul, i) * invMs;
        float3 m   = {mx[i], my[i], mz[i]};
        float  mu  = dot(m, u);
        float3 Ba  = 2.0f*vcmaCoeff*voltage*    (mu)*u;

        Bx[i] += Ba.x;
        By[i] += Ba.y;
        Bz[i] += Ba.z;
    }
}

