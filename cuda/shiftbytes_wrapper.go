package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var shiftbytes_code cu.Function

type shiftbytes_args struct {
	arg_dst   unsafe.Pointer
	arg_src   unsafe.Pointer
	arg_Nx    int
	arg_Ny    int
	arg_Nz    int
	arg_shx   int
	arg_clamp byte
	argptr    [7]unsafe.Pointer
}

// Wrapper for shiftbytes CUDA kernel, asynchronous.
func k_shiftbytes_async(dst unsafe.Pointer, src unsafe.Pointer, Nx int, Ny int, Nz int, shx int, clamp byte, cfg *config, str cu.Stream) {
	if synchronous { // debug
		Sync()
	}

	if shiftbytes_code == 0 {
		shiftbytes_code = fatbinLoad(shiftbytes_map, "shiftbytes")
	}

	var _a_ shiftbytes_args

	_a_.arg_dst = dst
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_dst)
	_a_.arg_src = src
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_src)
	_a_.arg_Nx = Nx
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_Nx)
	_a_.arg_Ny = Ny
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_Ny)
	_a_.arg_Nz = Nz
	_a_.argptr[4] = unsafe.Pointer(&_a_.arg_Nz)
	_a_.arg_shx = shx
	_a_.argptr[5] = unsafe.Pointer(&_a_.arg_shx)
	_a_.arg_clamp = clamp
	_a_.argptr[6] = unsafe.Pointer(&_a_.arg_clamp)

	args := _a_.argptr[:]
	cu.LaunchKernel(shiftbytes_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, str, args)

	if synchronous { // debug
		Sync()
	}
}

// Wrapper for shiftbytes CUDA kernel, synchronized.
func k_shiftbytes_sync(dst unsafe.Pointer, src unsafe.Pointer, Nx int, Ny int, Nz int, shx int, clamp byte, cfg *config) {
	Sync()
	k_shiftbytes_async(dst, src, Nx, Ny, Nz, shx, clamp, cfg, stream0)
	Sync()
}

var shiftbytes_map = map[int]string{0: "",
	20: shiftbytes_ptx_20,
	30: shiftbytes_ptx_30,
	35: shiftbytes_ptx_35}

const (
	shiftbytes_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry shiftbytes(
	.param .u64 shiftbytes_param_0,
	.param .u64 shiftbytes_param_1,
	.param .u32 shiftbytes_param_2,
	.param .u32 shiftbytes_param_3,
	.param .u32 shiftbytes_param_4,
	.param .u32 shiftbytes_param_5,
	.param .u8 shiftbytes_param_6
)
{
	.reg .pred 	%p<9>;
	.reg .s16 	%rs<5>;
	.reg .s32 	%r<23>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd3, [shiftbytes_param_0];
	ld.param.u64 	%rd4, [shiftbytes_param_1];
	ld.param.u32 	%r7, [shiftbytes_param_2];
	ld.param.u32 	%r8, [shiftbytes_param_3];
	ld.param.u32 	%r10, [shiftbytes_param_4];
	ld.param.u32 	%r9, [shiftbytes_param_5];
	cvta.to.global.u64 	%rd1, %rd3;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 1 9 1
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	.loc 1 10 1
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	.loc 1 11 1
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	.loc 1 13 1
	setp.lt.s32	%p1, %r1, %r7;
	setp.lt.s32	%p2, %r2, %r8;
	and.pred  	%p3, %p1, %p2;
	.loc 1 13 1
	setp.lt.s32	%p4, %r3, %r10;
	and.pred  	%p5, %p3, %p4;
	ld.param.s8 	%rs4, [shiftbytes_param_6];
	.loc 1 13 1
	@!%p5 bra 	BB0_4;
	bra.uni 	BB0_1;

BB0_1:
	.loc 1 14 1
	sub.s32 	%r4, %r1, %r9;
	.loc 1 16 1
	setp.lt.s32	%p6, %r4, 0;
	setp.ge.s32	%p7, %r4, %r7;
	or.pred  	%p8, %p6, %p7;
	.loc 1 21 1
	mul.lo.s32 	%r5, %r3, %r8;
	add.s32 	%r6, %r5, %r2;
	.loc 1 16 1
	@%p8 bra 	BB0_3;

	.loc 1 19 1
	mad.lo.s32 	%r20, %r6, %r7, %r4;
	cvt.s64.s32	%rd5, %r20;
	add.s64 	%rd6, %rd2, %rd5;
	.loc 1 19 1
	ld.global.u8 	%rs4, [%rd6];

BB0_3:
	.loc 1 21 1
	mad.lo.s32 	%r22, %r6, %r7, %r1;
	cvt.s64.s32	%rd7, %r22;
	add.s64 	%rd8, %rd1, %rd7;
	.loc 1 21 1
	st.global.u8 	[%rd8], %rs4;

BB0_4:
	.loc 1 23 2
	ret;
}


`
	shiftbytes_ptx_30 = `
.version 3.2
.target sm_30
.address_size 64


.visible .entry shiftbytes(
	.param .u64 shiftbytes_param_0,
	.param .u64 shiftbytes_param_1,
	.param .u32 shiftbytes_param_2,
	.param .u32 shiftbytes_param_3,
	.param .u32 shiftbytes_param_4,
	.param .u32 shiftbytes_param_5,
	.param .u8 shiftbytes_param_6
)
{
	.reg .pred 	%p<9>;
	.reg .s16 	%rs<5>;
	.reg .s32 	%r<23>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd3, [shiftbytes_param_0];
	ld.param.u64 	%rd4, [shiftbytes_param_1];
	ld.param.u32 	%r7, [shiftbytes_param_2];
	ld.param.u32 	%r8, [shiftbytes_param_3];
	ld.param.u32 	%r10, [shiftbytes_param_4];
	ld.param.u32 	%r9, [shiftbytes_param_5];
	cvta.to.global.u64 	%rd1, %rd3;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 1 9 1
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	.loc 1 10 1
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	.loc 1 11 1
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	.loc 1 13 1
	setp.lt.s32	%p1, %r1, %r7;
	setp.lt.s32	%p2, %r2, %r8;
	and.pred  	%p3, %p1, %p2;
	.loc 1 13 1
	setp.lt.s32	%p4, %r3, %r10;
	and.pred  	%p5, %p3, %p4;
	ld.param.s8 	%rs4, [shiftbytes_param_6];
	.loc 1 13 1
	@!%p5 bra 	BB0_4;
	bra.uni 	BB0_1;

BB0_1:
	.loc 1 14 1
	sub.s32 	%r4, %r1, %r9;
	.loc 1 16 1
	setp.lt.s32	%p6, %r4, 0;
	setp.ge.s32	%p7, %r4, %r7;
	or.pred  	%p8, %p6, %p7;
	.loc 1 21 1
	mul.lo.s32 	%r5, %r3, %r8;
	add.s32 	%r6, %r5, %r2;
	.loc 1 16 1
	@%p8 bra 	BB0_3;

	.loc 1 19 1
	mad.lo.s32 	%r20, %r6, %r7, %r4;
	cvt.s64.s32	%rd5, %r20;
	add.s64 	%rd6, %rd2, %rd5;
	.loc 1 19 1
	ld.global.u8 	%rs4, [%rd6];

BB0_3:
	.loc 1 21 1
	mad.lo.s32 	%r22, %r6, %r7, %r1;
	cvt.s64.s32	%rd7, %r22;
	add.s64 	%rd8, %rd1, %rd7;
	.loc 1 21 1
	st.global.u8 	[%rd8], %rs4;

BB0_4:
	.loc 1 23 2
	ret;
}


`
	shiftbytes_ptx_35 = `
.version 3.2
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 66 3
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 71 3
	ret;
}

.visible .entry shiftbytes(
	.param .u64 shiftbytes_param_0,
	.param .u64 shiftbytes_param_1,
	.param .u32 shiftbytes_param_2,
	.param .u32 shiftbytes_param_3,
	.param .u32 shiftbytes_param_4,
	.param .u32 shiftbytes_param_5,
	.param .u8 shiftbytes_param_6
)
{
	.reg .pred 	%p<9>;
	.reg .s16 	%rs<5>;
	.reg .s32 	%r<23>;
	.reg .s64 	%rd<9>;


	ld.param.u64 	%rd3, [shiftbytes_param_0];
	ld.param.u64 	%rd4, [shiftbytes_param_1];
	ld.param.u32 	%r7, [shiftbytes_param_2];
	ld.param.u32 	%r8, [shiftbytes_param_3];
	ld.param.u32 	%r10, [shiftbytes_param_4];
	ld.param.u32 	%r9, [shiftbytes_param_5];
	cvta.to.global.u64 	%rd1, %rd3;
	cvta.to.global.u64 	%rd2, %rd4;
	.loc 1 9 1
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	.loc 1 10 1
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	.loc 1 11 1
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	.loc 1 13 1
	setp.lt.s32	%p1, %r1, %r7;
	setp.lt.s32	%p2, %r2, %r8;
	and.pred  	%p3, %p1, %p2;
	.loc 1 13 1
	setp.lt.s32	%p4, %r3, %r10;
	and.pred  	%p5, %p3, %p4;
	ld.param.s8 	%rs4, [shiftbytes_param_6];
	.loc 1 13 1
	@!%p5 bra 	BB2_4;
	bra.uni 	BB2_1;

BB2_1:
	.loc 1 14 1
	sub.s32 	%r4, %r1, %r9;
	.loc 1 16 1
	setp.lt.s32	%p6, %r4, 0;
	setp.ge.s32	%p7, %r4, %r7;
	or.pred  	%p8, %p6, %p7;
	.loc 1 21 1
	mul.lo.s32 	%r5, %r3, %r8;
	add.s32 	%r6, %r5, %r2;
	.loc 1 16 1
	@%p8 bra 	BB2_3;

	.loc 1 19 1
	mad.lo.s32 	%r20, %r6, %r7, %r4;
	cvt.s64.s32	%rd5, %r20;
	add.s64 	%rd6, %rd2, %rd5;
	.loc 1 19 1
	ld.global.nc.u8 	%rs4, [%rd6];

BB2_3:
	.loc 1 21 1
	mad.lo.s32 	%r22, %r6, %r7, %r1;
	cvt.s64.s32	%rd7, %r22;
	add.s64 	%rd8, %rd1, %rd7;
	.loc 1 21 1
	st.global.u8 	[%rd8], %rs4;

BB2_4:
	.loc 1 23 2
	ret;
}


`
)
