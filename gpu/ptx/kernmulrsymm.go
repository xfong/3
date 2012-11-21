package ptx

const KERNMULRSYMM = `
//
// Generated by NVIDIA NVVM Compiler
// Compiler built on Sat Sep 22 02:35:14 2012 (1348274114)
// Cuda compilation tools, release 5.0, V0.2.1221
//

.version 3.1
.target sm_30
.address_size 64

	.file	1 "/tmp/tmpxft_000021e8_00000000-9_kernmulrsymm.cpp3.i"
	.file	2 "/home/arne/src/code.google.com/p/nimble-cube/gpu/ptx/kernmulrsymm.cu"

.visible .entry kernmulRSymm(
	.param .u64 kernmulRSymm_param_0,
	.param .u64 kernmulRSymm_param_1,
	.param .u64 kernmulRSymm_param_2,
	.param .u64 kernmulRSymm_param_3,
	.param .u64 kernmulRSymm_param_4,
	.param .u64 kernmulRSymm_param_5,
	.param .u64 kernmulRSymm_param_6,
	.param .u64 kernmulRSymm_param_7,
	.param .u64 kernmulRSymm_param_8,
	.param .u32 kernmulRSymm_param_9
)
{
	.reg .pred 	%p<8>;
	.reg .s32 	%r<29>;
	.reg .f32 	%f<54>;
	.reg .s64 	%rd<39>;


	ld.param.u64 	%rd21, [kernmulRSymm_param_0];
	ld.param.u64 	%rd22, [kernmulRSymm_param_1];
	ld.param.u64 	%rd23, [kernmulRSymm_param_2];
	ld.param.u64 	%rd15, [kernmulRSymm_param_3];
	ld.param.u64 	%rd16, [kernmulRSymm_param_4];
	ld.param.u64 	%rd17, [kernmulRSymm_param_5];
	ld.param.u64 	%rd18, [kernmulRSymm_param_6];
	ld.param.u64 	%rd19, [kernmulRSymm_param_7];
	ld.param.u64 	%rd20, [kernmulRSymm_param_8];
	ld.param.u32 	%r3, [kernmulRSymm_param_9];
	cvta.to.global.u64 	%rd1, %rd19;
	cvta.to.global.u64 	%rd2, %rd18;
	cvta.to.global.u64 	%rd3, %rd17;
	cvta.to.global.u64 	%rd4, %rd16;
	cvta.to.global.u64 	%rd5, %rd15;
	cvta.to.global.u64 	%rd6, %rd23;
	cvta.to.global.u64 	%rd7, %rd22;
	cvta.to.global.u64 	%rd8, %rd21;
	.loc 2 13 1
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %ctaid.x;
	mad.lo.s32 	%r7, %r4, %r5, %r6;
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	.loc 2 14 1
	shl.b32 	%r2, %r1, 1;
	.loc 2 16 1
	setp.ge.s32 	%p1, %r1, %r3;
	@%p1 bra 	BB0_14;

	.loc 2 17 1
	mul.wide.s32 	%rd24, %r2, 4;
	add.s64 	%rd9, %rd8, %rd24;
	ld.global.f32 	%f1, [%rd9];
	.loc 2 18 1
	add.s32 	%r11, %r2, 1;
	mul.wide.s32 	%rd25, %r11, 4;
	add.s64 	%rd10, %rd8, %rd25;
	ld.global.f32 	%f2, [%rd10];
	.loc 2 20 1
	add.s64 	%rd11, %rd7, %rd24;
	ld.global.f32 	%f3, [%rd11];
	.loc 2 21 1
	add.s64 	%rd12, %rd7, %rd25;
	ld.global.f32 	%f4, [%rd12];
	.loc 2 23 1
	add.s64 	%rd13, %rd6, %rd24;
	ld.global.f32 	%f5, [%rd13];
	.loc 2 24 1
	add.s64 	%rd14, %rd6, %rd25;
	ld.global.f32 	%f6, [%rd14];
	.loc 2 26 1
	setp.eq.s64 	%p2, %rd15, 0;
	mov.f32 	%f19, 0f00000000;
	.loc 2 26 1
	mov.f32 	%f53, %f19;
	@%p2 bra 	BB0_3;

	mul.wide.s32 	%rd26, %r1, 4;
	add.s64 	%rd27, %rd5, %rd26;
	ld.global.f32 	%f7, [%rd27];
	mov.f32 	%f53, %f7;

BB0_3:
	.loc 2 26 1
	mov.f32 	%f8, %f53;
	.loc 2 27 1
	setp.eq.s64 	%p3, %rd16, 0;
	mov.f32 	%f52, %f19;
	@%p3 bra 	BB0_5;

	mul.wide.s32 	%rd28, %r1, 4;
	add.s64 	%rd29, %rd4, %rd28;
	ld.global.f32 	%f52, [%rd29];

BB0_5:
	.loc 2 28 1
	setp.eq.s64 	%p4, %rd17, 0;
	mov.f32 	%f51, %f19;
	@%p4 bra 	BB0_7;

	mul.wide.s32 	%rd30, %r1, 4;
	add.s64 	%rd31, %rd3, %rd30;
	ld.global.f32 	%f51, [%rd31];

BB0_7:
	.loc 2 30 1
	setp.eq.s64 	%p5, %rd18, 0;
	mov.f32 	%f50, %f19;
	@%p5 bra 	BB0_9;

	mul.wide.s32 	%rd32, %r1, 4;
	add.s64 	%rd33, %rd2, %rd32;
	ld.global.f32 	%f50, [%rd33];

BB0_9:
	.loc 2 31 1
	setp.eq.s64 	%p6, %rd19, 0;
	mov.f32 	%f49, %f19;
	@%p6 bra 	BB0_11;

	mul.wide.s32 	%rd34, %r1, 4;
	add.s64 	%rd35, %rd1, %rd34;
	ld.global.f32 	%f49, [%rd35];

BB0_11:
	.loc 2 32 1
	setp.eq.s64 	%p7, %rd20, 0;
	mov.f32 	%f48, %f19;
	@%p7 bra 	BB0_13;

	cvta.to.global.u64 	%rd36, %rd20;
	.loc 2 32 1
	mul.wide.s32 	%rd37, %r1, 4;
	add.s64 	%rd38, %rd36, %rd37;
	ld.global.f32 	%f48, [%rd38];

BB0_13:
	.loc 2 34 1
	mul.f32 	%f25, %f3, %f48;
	fma.rn.f32 	%f26, %f1, %f8, %f25;
	fma.rn.f32 	%f27, %f5, %f49, %f26;
	st.global.f32 	[%rd9], %f27;
	.loc 2 35 1
	mul.f32 	%f28, %f4, %f48;
	fma.rn.f32 	%f29, %f2, %f8, %f28;
	fma.rn.f32 	%f30, %f6, %f49, %f29;
	st.global.f32 	[%rd10], %f30;
	.loc 2 37 1
	mul.f32 	%f31, %f3, %f52;
	fma.rn.f32 	%f32, %f1, %f48, %f31;
	fma.rn.f32 	%f33, %f5, %f50, %f32;
	st.global.f32 	[%rd11], %f33;
	.loc 2 38 1
	mul.f32 	%f34, %f4, %f52;
	fma.rn.f32 	%f35, %f2, %f48, %f34;
	fma.rn.f32 	%f36, %f6, %f50, %f35;
	st.global.f32 	[%rd12], %f36;
	.loc 2 40 1
	mul.f32 	%f37, %f3, %f50;
	fma.rn.f32 	%f38, %f1, %f49, %f37;
	fma.rn.f32 	%f39, %f5, %f51, %f38;
	st.global.f32 	[%rd13], %f39;
	.loc 2 41 1
	mul.f32 	%f40, %f4, %f50;
	fma.rn.f32 	%f41, %f2, %f49, %f40;
	fma.rn.f32 	%f42, %f6, %f51, %f41;
	st.global.f32 	[%rd14], %f42;

BB0_14:
	.loc 2 43 2
	ret;
}


`
