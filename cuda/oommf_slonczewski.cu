// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013

#include <stdint.h>
#include "float3.h"
#include "constants.h"
#include "amul.h"

extern "C" __global__ void
addoommfslonczewskitorque(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                     float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                      float* __restrict__ Ms_,      float  Ms_mul,
                      float* __restrict__ jz_,      float  jz_mul,
                      float* __restrict__ px_,      float  px_mul,
                      float* __restrict__ py_,      float  py_mul,
                      float* __restrict__ pz_,      float  pz_mul,
                      float* __restrict__ alpha_,   float  alpha_mul,
                      float* __restrict__ pfix_,     float  pfix_mul,
                      float* __restrict__ pfree_,     float  pfree_mul,
                      float* __restrict__ lambdafix_,  float  lambdafix_mul,
                      float* __restrict__ lambdafree_,  float  lambdafree_mul,
                      float* __restrict__ epsPrime_,float  epsPrime_mul,
                      float* __restrict__ flt_,     float  flt_mul,
                     int N) {

	int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
	if (i < N) {

		float3 m = make_float3(mx[i], my[i], mz[i]);
		float  J = amul(jz_, jz_mul, i);
		float3 p = normalized(vmul(px_, py_, pz_, px_mul, py_mul, pz_mul, i));

        	float  Ms           = amul(Ms_, Ms_mul, i);
        	float  alpha        = amul(alpha_, alpha_mul, i);
        	float  flt          = amul(flt_, flt_mul, i);
        	float  pfix         = amul(pfix_, pfix_mul, i);
        	float  pfree        = amul(pfree_, pfix_mul, i);
		float  lambdafix    = amul(lambdafix_, lambdafix_mul, i);
		float  lambdafree   = amul(lambdafree_, lambdafree_mul, i);
	        float  epsilonPrime = amul(epsPrime_, epsPrime_mul, i);

		if (J == 0.0f || Ms == 0.0f) {
			return;
		}

		float beta    = (HBAR / QE) * (J / (2.0f *flt*Ms) );
		float lambdafix2 = lambdafix * lambdafix;
		float lambdafree2 = lambdafree * lambdafree;
		float lambdafreePlus = sqrt(lambdafree2 + 1.0f);
		float lambdafixPlus = sqrt(lambdafix2 + 1.0f);
		float lambdafreeMinus = sqrt(lambdafree2 - 1.0f);
		float lambdafixMinus = sqrt(lambdafix2 - 1.0f);
		float plus_ratio = lambdafreePlus / lambdafixPlus;
		float minus_ratio = 1.0f;
		if (lambdafreeMinus > 0) {
		   	minus_ratio = lambdafixMinus / lambdafreeMinus;
		}
		// Compute q_plus and q_minus
		float plus_factor = pfix * lambdafix2 * plus_ratio;
		float minus_factor = pfree * lambdafree2 * minus_ratio;
		float q_plus = plus_factor + minus_factor;
		float q_minus = plus_factor - minus_factor;
		float lplus2 = lambdafreePlus * lambdafixPlus;
		float lminus2 = lambdafreeMinus * lambdafixMinus;
		float pdotm = dot(p, m);
		float A_plus = lplus2 + (lminus2 * pdotm);
		float A_minus = lplus2 - (lminus2 * pdotm);
		float epsilon = (q_plus / A_plus) + (q_minus / A_minus);

		float A = beta * epsilon;
		float B = beta * epsilonPrime;

		float gilb     = 1.0f / (1.0f + alpha * alpha);
		float mxpxmFac = gilb * (A + alpha * B);
		float pxmFac   = gilb * (B - alpha * A);

		float3 pxm      = cross(p, m);
		float3 mxpxm    = cross(m, pxm);

		tx[i] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
		ty[i] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
		tz[i] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
	}
}
