/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: zhli@openailab.com
*/

// register definition
// x0        output start address
// x1        input start address
// x2        kernel start address
// x3        cin
// x10		t2		loop1 循环变量
// x9		t1		loop4 循环变量

// x9 ~ x10  temp loop counter

    .section .text,"ax"
    .align 5

    .type wino_sgemm_4x16 STT_FUNC
    .global wino_sgemm_4x16
    .hidden wino_sgemm_4x16
    
wino_sgemm_4x16:
	li				t1, 16
	vsetvli         x0, t1, e32
	#     // biases_initial
    beqz            a0, none_biases
    vlw.v           v0, (a0)
    vrgather.vi     v16, v0, 0
    vrgather.vi     v17, v0, 1
    vrgather.vi     v18, v0, 2
    vrgather.vi     v19, v0, 3
    addi            a0, a0, 0x10
    vlw.v           v0, (a0)
    vrgather.vi     v20, v0, 0
    vrgather.vi     v21, v0, 1
    vrgather.vi     v22, v0, 2
    vrgather.vi     v23, v0, 3
    addi            a0, a0, 0x10
    vlw.v           v0, (a0)
    vrgather.vi     v24, v0, 0
    vrgather.vi     v25, v0, 1
    vrgather.vi     v26, v0, 2
    vrgather.vi     v27, v0, 3
    addi            a0, a0, 0x10
    vlw.v           v0, (a0)
    vrgather.vi     v28, v0, 0
    vrgather.vi     v29, v0, 1
    vrgather.vi     v30, v0, 2
    vrgather.vi     v31, v0, 3

    j               start

none_biases:
    vmv.v.x         v16, x0
    vmv.v.x         v17, x0
    vmv.v.x         v18, x0
    vmv.v.x         v19, x0
    vmv.v.x         v20, x0
    vmv.v.x         v21, x0
    vmv.v.x         v22, x0
    vmv.v.x         v23, x0
    vmv.v.x         v24, x0
    vmv.v.x         v25, x0
    vmv.v.x         v26, x0
    vmv.v.x         v27, x0
    vmv.v.x         v28, x0
    vmv.v.x         v29, x0
    vmv.v.x         v30, x0
    vmv.v.x         v31, x0
	
start:
    vlw.v           v0, (a1)
    addi            t0, a2, 0
    vlw.v           v4, (t0)
    addi            t0, a2, 0x10
    vlw.v           v5, (t0)

    andi             t2, a3, 0x3
    bltz            t2, loop4_end
    srli            t1, a3, 0x2

loop4:  
    addi            t1, t1, -1
    addi            t0, a2, 0x20
    vlw.v           v6, (t0)
    addi            t0, a2, 0x30
    vlw.v           v7, (t0)

    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v0, v8
    vfmacc.vv       v17, v0, v9
    vfmacc.vv       v18, v0, v10
    vfmacc.vv       v19, v0, v11
    
    addi            t0, a1, 0x10
    vlw.v           v1, (t0)
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v0, v8
    vfmacc.vv       v21, v0, v9
    vfmacc.vv       v22, v0, v10
    vfmacc.vv       v23, v0, v11
    
    addi            t0, a2, 0x40
    vlw.v           v4, (t0)
    addi            t0, a2, 0x50
    vlw.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v0, v8
    vfmacc.vv       v25, v0, v9
    vfmacc.vv       v26, v0, v10
    vfmacc.vv       v27, v0, v11
    
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v0, v8
    vfmacc.vv       v29, v0, v9
    vfmacc.vv       v30, v0, v10
    vfmacc.vv       v31, v0, v11

    addi            t0, a2, 0x60
    vlw.v           v6, (t0)
    addi            t0, a2, 0x70
    vlw.v           v7, (t0)
    
    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v1, v8
    vfmacc.vv       v17, v1, v9
    vfmacc.vv       v18, v1, v10
    vfmacc.vv       v19, v1, v11
    
    addi            t0, a1, 0x20
    vlw.v           v0, (t0)
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v1, v8
    vfmacc.vv       v21, v1, v9
    vfmacc.vv       v22, v1, v10
    vfmacc.vv       v23, v1, v11
    
    addi            t0, a2, 0x80
    vlw.v           v4, (t0)
    addi            t0, a2, 0x90
    vlw.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v1, v8
    vfmacc.vv       v25, v1, v9
    vfmacc.vv       v26, v1, v10
    vfmacc.vv       v27, v1, v11
    
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v1, v8
    vfmacc.vv       v29, v1, v9
    vfmacc.vv       v30, v1, v10
    vfmacc.vv       v31, v1, v11
    
    addi            t0, a2, 0xa0
    vlw.v           v6, (t0)
    addi            t0, a2, 0xb0
    vlw.v           v7, (t0)
    
    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v0, v8
    vfmacc.vv       v17, v0, v9
    vfmacc.vv       v18, v0, v10
    vfmacc.vv       v19, v0, v11
    
    addi            t0, a1, 0x30
    vlw.v           v1, (t0)
    addi             a1, a1, 0x40
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v0, v8
    vfmacc.vv       v21, v0, v9
    vfmacc.vv       v22, v0, v10
    vfmacc.vv       v23, v0, v11
    
    addi            t0, a2, 0xc0
    vlw.v           v4, (t0)
    addi            t0, a2, 0xd0
    vlw.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v0, v8
    vfmacc.vv       v25, v0, v9
    vfmacc.vv       v26, v0, v10
    vfmacc.vv       v27, v0, v11
    
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v0, v8
    vfmacc.vv       v29, v0, v9
    vfmacc.vv       v30, v0, v10
    vfmacc.vv       v31, v0, v11
    
    addi            t0, a2, 0xe0
    vlw.v           v6, (t0)
    addi            t0, a2, 0xf0
    vlw.v           v7, (t0)
    addi            a2, a2, 0x100
    vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v1, v8
    vfmacc.vv       v17, v1, v9
    vfmacc.vv       v18, v1, v10
    vfmacc.vv       v19, v1, v11

    vlw.v           v0, (a1)
    
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v1, v8
    vfmacc.vv       v21, v1, v9
    vfmacc.vv       v22, v1, v10
    vfmacc.vv       v23, v1, v11
    
    addi            t0, a2, 0x0
    vlw.v           v4, (t0)
    addi            t0, a2, 0x10
    vlw.v           v5, (t0)
    
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v1, v8
    vfmacc.vv       v25, v1, v9
    vfmacc.vv       v26, v1, v10
    vfmacc.vv       v27, v1, v11

    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v1, v8
    vfmacc.vv       v29, v1, v9
    vfmacc.vv       v30, v1, v10
    vfmacc.vv       v31, v1, v11
    bnez            t1, loop4



loop4_end:
	# cbz	x10, save_result
	beqz            t2, save_result

loop1:
    #     ldp     q6, q7, [x2, 0x20]              // q6=k[b-8] q7=k[f-c]
    addi            t0, a2, 0x20
    vlw.v           v6, (t0)
    addi            t0, a2, 0x30
    vlw.v           v7, (t0)
    addi            a2, a2, 0x40
	
	# fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	# fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
    #     add     x2, x2, 0x40
	# fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	# fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	# add	x1, x1, 0x10
	vrgather.vi     v8, v4, 0
    vrgather.vi     v9, v4, 1
    vrgather.vi     v10, v4, 2
    vrgather.vi     v11, v4, 3
    vfmacc.vv       v16, v0, v8
    vfmacc.vv       v17, v0, v9
    vfmacc.vv       v18, v0, v10
    vfmacc.vv       v19, v0, v11
    addi            a1, a1, 0x10
    addi            t2, t2, -1
	# fmla	v20.4s, v0.4s,  v5.s[0]		// i[3-0]k[4]
	# fmla	v21.4s, v0.4s,  v5.s[1]		// i[3-0]k[5]
    #     subs    x10, x10 ,0x1
	# fmla	v22.4s, v0.4s,  v5.s[2]		// i[3-0]k[6]
	# fmla	v23.4s, v0.4s,  v5.s[3]		// i[3-0]k[7]
    vrgather.vi     v8,  v5, 0
    vrgather.vi     v9,  v5, 1
    vrgather.vi     v10, v5, 2
    vrgather.vi     v11, v5, 3
    vfmacc.vv       v20, v0, v8
    vfmacc.vv       v21, v0, v9
    vfmacc.vv       v22, v0, v10
    vfmacc.vv       v23, v0, v11
    addi            t0, a2, 0x0
    vlw.v           v4, (t0)
    addi            t0, a2, 0x10
	# ldp     q4, q5, [x2]                    // q4=k[3-0] q5=k[7-4]
	# fmla	v24.4s, v0.4s,  v6.s[0]		// i[3-0]k[8]
	# fmla	v25.4s, v0.4s,  v6.s[1]		// i[3-0]k[9]
	# fmla	v26.4s, v0.4s,  v6.s[2]		// i[3-0]k[a]
	# fmla	v27.4s, v0.4s,  v6.s[3]		// i[3-0]k[b]
	vlw.v           v5, (t0)
    vrgather.vi     v8,  v6, 0
    vrgather.vi     v9,  v6, 1
    vrgather.vi     v10, v6, 2
    vrgather.vi     v11, v6, 3
    vfmacc.vv       v24, v0, v8
    vfmacc.vv       v25, v0, v9
    vfmacc.vv       v26, v0, v10
    vfmacc.vv       v27, v0, v11
	# fmla	v28.4s, v0.4s,  v7.s[0]		// i[3-0]k[c]
	# fmla	v29.4s, v0.4s,  v7.s[1]		// i[3-0]k[d]
	# fmla	v30.4s, v0.4s,  v7.s[2]		// i[3-0]k[e]
	# fmla	v31.4s, v0.4s,  v7.s[3]		// i[3-0]k[f]
    vrgather.vi     v8,  v7, 0
    vrgather.vi     v9,  v7, 1
    vrgather.vi     v10, v7, 2
    vrgather.vi     v11, v7, 3
    vfmacc.vv       v28, v0, v8
    vfmacc.vv       v29, v0, v9
    vfmacc.vv       v30, v0, v10
    vfmacc.vv       v31, v0, v11
    #     ldr     q0, [x1]                        // q0=i[3-0]
    vlw.v           v0, (a1)

    # b.ne    loop1
	bnez			t2, loop1
	
save_result:
    # cmp     w4,0
    # beq     direct_save
	beqz            a4, direct_save

    stride_save:
    # mov x10,#0x240
    # lsl x11,x10,#2
	li				t5, 0x240
	slli			t0, t5, 2
	add				t6, t5, t0

    # add x6,x0,x11
    # add x7,x0,x11,lsl #1
	# add x8,x7,x11
	add				s6, a0, t6
	slli			t0, t6, 1
	add				s7, a0, t0
    add				s8, s6, t6

    # str q16,[x0]
    # str q17,[x0,#0x240]   //each line 36*4 data
    # str q18,[x0,#0x480]
    # str q19,[x0,#0x6c0]
	vsw.v			v16, (a0)
	addi			a0, a0, 0x240
	vsw.v			v17, (a0)
	addi			a0, a0, 0x240
	vsw.v			v18, (a0)
	addi			a0, a0, 0x240
	vsw.v			v19, (a0)
	addi			a0, a0, -0x6c0

    # str q20,[x6]
    # str q21,[x6,#0x240]   //each line 36*4 data
    # str q22,[x6,#0x480]
    # str q23,[x6,#0x6c0]
	vsw.v			v20, (s6)
	addi			s6, s6, 0x240
	vsw.v			v21, (s6)
	addi			s6, s6, 0x240
	vsw.v			v22, (s6)
	addi			s6, s6, 0x240
	vsw.v			v23, (s6)
	addi			s6, s6, -0x6c0

    # str q24,[x7]
    # str q25,[x7,#0x240]   //each line 36*4 data
    # str q26,[x7,#0x480]
    # str q27,[x7,#0x6c0]
	vsw.v			v24, (s7)
	addi			s7, s7, 0x240
	vsw.v			v25, (s7)
	addi			s7, s7, 0x240
	vsw.v			v26, (s7)
	addi			s7, s7, 0x240
	vsw.v			v27, (s7)
	addi			s7, s7, -0x6c0

    # str q28,[x8]
    # str q29,[x8,#0x240]   //each line 36*4 data
    # str q30,[x8,#0x480]
    # str q31,[x8,#0x6c0]
	vsw.v			v28, (s8)
	addi			s8, s8, 0x240
	vsw.v			v29, (s8)
	addi			s8, s8, 0x240
	vsw.v			v30, (s8)
	addi			s8, s8, 0x240
	vsw.v			v31, (s8)
	addi			s8, s8, -0x6c0

    # b end_func
	j end_func  # jump to end_func
	
    
    direct_save:
    # stp  q16,q17, [x0]
    # stp	 q18,q19, [x0, 0x20]
    # stp	 q20,q21, [x0, 0x40]
    # stp	 q22,q23, [x0, 0x60]
    # stp	 q24,q25, [x0, 0x80]
    # stp	 q26,q27, [x0, 0xa0]
    # stp	 q28,q29, [x0, 0xc0]
    # stp	 q30,q31, [x0, 0xe0]
	vsw.v       v16, (a0)
	addi		a0, a0, 16
    vsw.v       v17, (a0)
	addi		a0, a0, 16
    vsw.v       v18, (a0)
	addi		a0, a0, 16
    vsw.v       v19, (a0)
	addi		a0, a0, 16
    vsw.v       v20, (a0)
	addi		a0, a0, 16
    vsw.v       v21, (a0)
	addi		a0, a0, 16
    vsw.v       v22, (a0)
	addi		a0, a0, 16
    vsw.v       v23, (a0)
	addi		a0, a0, 16
    vsw.v       v24, (a0)
	addi		a0, a0, 16
    vsw.v       v25, (a0)
	addi		a0, a0, 16
    vsw.v       v26, (a0)
	addi		a0, a0, 16
    vsw.v       v27, (a0)
	addi		a0, a0, 16
    vsw.v       v28, (a0)
	addi		a0, a0, 16
    vsw.v       v29, (a0)
	addi		a0, a0, 16
    vsw.v       v30, (a0)
	addi		a0, a0, 16
    vsw.v       v31, (a0)
	addi		a0, a0, -0xf0

end_func:

	ret
        .end

