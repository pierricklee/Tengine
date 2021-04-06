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
// x0 a0        output start address
// x1 a1        input start address
// x2 a2        kernel start address
// x3 a3        cin
// x4 a4        direct save/stride save

// t2(x9) ~ t3(x10)  temp loop counter

    .section .text,"ax"
    .align 5

    .type wino_sgemm_4x4_A72 STT_FUNC
    .global wino_sgemm_4x4_A72
    .hidden wino_sgemm_4x4_A72
    
wino_sgemm_4x4_A72:
	# cmp	x3, 0x4
	vsetvli         t0, a0, e32, m1, d1

none_biases:
	# movi	d16, 0x0
	# movi	d17, 0x0
	# movi	d18, 0x0
	# movi	d19, 0x0
    vmv.v.x         v16, x0
    vmv.v.x         v17, x0
    vmv.v.x         v18, x0
    vmv.v.x         v19, x0

start:
	# and	x10,x3, 0x3
	# b.lt	loop4_end
	# lsr	x9, x3, 0x2
	addi            t3, a3, 0x3
	li              t0, 4
	blt             a3, t0, loop4_end
	srli			t2, a3, 0x2

loop4:  
	# ldr	q0, [x1]			// q0=i[3-0] q1=i[3-0]
	# ldr	q4, [x2]			// q4=k[3-0] q5=k[7-4] 

	# ldr	q1, [x1, 0x10]			// q0=i[3-0] q1=i[3-0]
	# ldr	q5, [x2, 0x10]			// q4=k[3-0] q5=k[7-4] 

	# ldp	q2, q3, [x1, 0x20]		// q2=i[3-0] q3=i[3-0]
	# ldp	q6, q7, [x2, 0x20]		// q6=k[b-8] q7=k[f-c]

	vlw.v           v0, (a1)
    addi            a1, a1, 16
    vlw.v           v1, (a1)
    addi            a1, a1, 16
    vlw.v           v2, (a1)
    addi            a1, a1, 16
    vlw.v           v3, (a1)
    addi            a1, a1, 16

    vlw.v           v4, (a2)
    addi            a2, a2, 16
    vlw.v           v5, (a2)
    addi            a2, a2, 16
    vlw.v           v6, (a2)
    addi            a2, a2, 16
    vlw.v           v7, (a2)
    addi            a2, a2, 16

    # subs	x9, x9, 0x1
	addi            t2, t2, -1

	# fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	# fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	# fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	# fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	vrgather.vi     v20, v4, 0
    vrgather.vi     v21, v4, 1
    vrgather.vi     v22, v4, 2
    vrgather.vi     v23, v4, 3
    vfmacc.vv       v16, v20, v0
    vfmacc.vv       v17, v21, v0
    vfmacc.vv       v18, v22, v0
    vfmacc.vv       v19, v23, v0

	# fmla	v16.4s, v1.4s,  v5.s[0]		// i[3-0]k[0]
	# fmla	v17.4s, v1.4s,  v5.s[1]		// i[3-0]k[1]
	# fmla	v18.4s, v1.4s,  v5.s[2]		// i[3-0]k[2]
	# fmla	v19.4s, v1.4s,  v5.s[3]		// i[3-0]k[3]
    vrgather.vi     v20, v5, 0
    vrgather.vi     v21, v5, 1
    vrgather.vi     v22, v5, 2
    vrgather.vi     v23, v5, 3
    vfmacc.vv       v16, v20, v1
    vfmacc.vv       v17, v21, v1
    vfmacc.vv       v18, v22, v1
    vfmacc.vv       v19, v23, v1

	# fmla	v16.4s, v2.4s, v6.s[0]		// i[3-0]k[0]
	# fmla	v17.4s, v2.4s, v6.s[1]		// i[3-0]k[1]
	# prfm	pldl1keep, [x1, 0x140]
    # fmla	v18.4s, v2.4s, v6.s[2]		// i[3-0]k[2]
	# prfm	pldl1keep, [x2, 0x140]
	# fmla	v19.4s, v2.4s, v6.s[3]		// i[3-0]k[3]
	vrgather.vi     v20, v6, 0
	vrgather.vi     v21, v6, 1
	vrgather.vi     v22, v6, 2
	vrgather.vi     v23, v6, 3
	vfmacc.vv       v16, v20, v2
	vfmacc.vv       v17, v21, v2
	vfmacc.vv       v18, v22, v2
	vfmacc.vv       v19, v23, v2

	# fmla	v16.4s, v3.4s, v7.s[0]		// i[3-0]k[0]
	# fmla	v17.4s, v3.4s, v7.s[1]		// i[3-0]k[1]
    # add	x1, x1, 0x40
	# fmla	v18.4s, v3.4s, v7.s[2]		// i[3-0]k[2]
    # add	x2, x2, 0x40
	# fmla	v19.4s, v3.4s, v7.s[3]		// i[3-0]k[3]
	vrgather.vi     v20, v7, 0
	vrgather.vi     v21, v7, 1
	vrgather.vi     v22, v7, 2
	vrgather.vi     v23, v7, 3
	vfmacc.vv       v16, v20, v3
	vfmacc.vv       v17, v21, v3
	vfmacc.vv       v18, v22, v3
	vfmacc.vv       v19, v23, v3

	# b.ne	loop4
	bnez            t2, loop4

loop4_end:
	# cbz	x10, save_result
	beqz		t3, save_result

loop1:
	# ldr     q0, [x1],0x10                    // q0=i[3-0]
    # ldr     q4, [x2], 0x10                  // q4=k[3-0]
	# subs	x10 ,x10 ,0x1
	vlw.v           v0, (a1)
    addi            a1, a1, 16
    vlw.v           v4, (a2)
    addi            a2, a2, 16
	addi            t3, t3, -1

	# fmla	v16.4s, v0.4s,  v4.s[0]		// i[0]k[3-0]
	# fmla	v17.4s, v0.4s,  v4.s[1]		// i[1]k[3-0]
	# fmla	v18.4s, v0.4s,  v4.s[2]		// i[2]k[3-0]
	# fmla	v19.4s, v0.4s,  v4.s[3]		// i[3]k[3-0]
	vrgather.vi     v20, v4, 0
    vrgather.vi     v21, v4, 1
    vrgather.vi     v22, v4, 2
    vrgather.vi     v23, v4, 3
    vfmacc.vv       v16, v20, v0
    vfmacc.vv       v17, v21, v0
    vfmacc.vv       v18, v22, v0
    vfmacc.vv       v19, v23, v0 
    # b.ne    loop1
    bnez            t3, loop1
	
save_result:
    # cmp     w4,0
    # beq     direct_save
	bnez			a4, direct_save

    stride_save:
    # mov x10,#0x240
	li				t3, 0x240
   
    # str q16,[x0]
    # str q17,[x0,#0x240]   //each line 36*4 data
    # str q18,[x0,#0x480]
    # str q19,[x0,#0x6c0]
	vsw.v           v16, (a0)
	addi			a0, a0, 0x240
    vsw.v           v17, (a0)
	addi			a0, a0, 0x240
    vsw.v           v18, (a0)
	addi			a0, a0, 0x240
    vsw.v           v19, (a0)
	addi			a0, a0, -0x6c0

    # b end_func
	j 				end_func  		# jump to end_func
	
    
    direct_save:
    # stp  q16,q17, [x0]
    # stp	 q18,q19, [x0, 0x20]
	vsw.v           v16, (a0)
	addi			a0, a0, 0x10
    vsw.v           v17, (a0)
	addi			a0, a0, 0x10
    vsw.v           v18, (a0)
	addi			a0, a0, 0x10
    vsw.v           v19, (a0)
	addi			a0, a0, -0x30


end_func:


	ret
        .end

