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


//x0: buffer
//x1: output
//x2: out_hw*sizeof(float)
//x3: ker
//x4: bias
//x5: activation

    .section .text,"ax"
    .align 5

    .type tran_out_4 STT_FUNC
    .global tran_out_4
    .hidden tran_out_4
    
tran_out_4:    # 暂时不保存vector寄存器
    # sub	sp, sp, 0x40
    li          t2, 4
    vsetvli     x0, t2, e32, m1, d1
    addi    sp, sp, -96
    # stp	d8, d9, [sp]
	# stp	d10,d11,[sp, 0x10]
	# stp	d12,d13,[sp, 0x20]
	# stp	d14,d15,[sp, 0x30]
    sd      s0, 0(sp)      
    sd      s1, 8(sp)               # x11
    sd      s2, 16(sp)              # x12
    sd      s3, 24(sp)              # x13
    sd      s4, 32(sp)      
    sd      s5, 40(sp)      
    sd      s6, 48(sp)     
    sd      s7, 56(sp)     
    sd      s8, 64(sp)     
    sd      s9, 72(sp)              # x9
    sd      s10, 80(sp)             # x10
    sd      s11, 88(sp) 
#           t1                      # x8
comput_idx:
    //str[x1,x11,x12,x13]
    # add x11,x1,x2      
    add     s1, a1, a2
    # add x12,x1,x2,LSL 1     
    slli    t0, a2, 1
    add     s2, a1, t0
    # add	x13,x11,x2, LSL 1  
    add     s3, s1, t0
    # //ldr[x0,x8,x9,x10]
    # add x8,x0,#0x60
    # add x9,x0,#0x120
    # add x10,x0,#0x1e0
    addi     t1, a0, 0x60
    addi     s9, a0, 0x120
    addi     s10, a0, 0x1e0

load:
    //load v0-v11
    //add:v20-v25
    //sub:v26-v31
    
    # ldp q0,q1,[x8]
    # ldp q2,q3,[x8,#0x20]
    # ldp q4,q5,[x8,#0x40]
    # ldp q6,q7,[x8,#0x60]
    # ldp q8,q9,[x8,#0x80]
    # ldp q10,q11,[x8,#0xa0]
    vlw.v           v0, (t1)
    addi            t1, t1, 16
    vlw.v           v1, (t1)
    addi            t1, t1, 16
    vlw.v           v2, (t1)
    addi            t1, t1, 16
    vlw.v           v3, (t1)
    addi            t1, t1, 16

    vlw.v           v4, (t1)
    addi            t1, t1, 16
    vlw.v           v5, (t1)
    addi            t1, t1, 16
    vlw.v           v6, (t1)
    addi            t1, t1, 16
    vlw.v           v7, (t1)
    addi            t1, t1, 16

    vlw.v           v8, (t1)
    addi            t1, t1, 16
    vlw.v           v9, (t1)
    addi            t1, t1, 16
    vlw.v           v10, (t1)
    addi            t1, t1, 16
    vlw.v           v11, (t1)
    addi            t1, t1, 16

    # fadd v20.4s,v0.4s,v6.4s
    # fadd v21.4s,v1.4s,v7.4s
    # fadd v22.4s,v2.4s,v8.4s
    # fadd v23.4s,v3.4s,v9.4s
    # fadd v24.4s,v4.4s,v10.4s
    # fadd v25.4s,v5.4s,v11.4s
    vfadd.vv	v20, v0, v6
    vfadd.vv	v21, v1, v7
    vfadd.vv	v22, v2, v8
    vfadd.vv	v23, v3, v9
    vfadd.vv	v24, v4, v10
    vfadd.vv	v25, v5, v11
    
    # fsub v26.4s,v0.4s,v6.4s
    # fsub v27.4s,v1.4s,v7.4s
    # fsub v28.4s,v2.4s,v8.4s
    # fsub v29.4s,v3.4s,v9.4s
    # fsub v30.4s,v4.4s,v10.4s
    # fsub v31.4s,v5.4s,v11.4s
    vfsub.vv	v26, v0, v6
    vfsub.vv	v27, v1, v7
    vfsub.vv	v28, v2, v8
    vfsub.vv	v29, v3, v9
    vfsub.vv	v30, v4, v10
    vfsub.vv	v31, v5, v11

    # //load:v0-v7
    # //add:v8-v13
    # //sub:v14-v19
    # ldp q0,q1,[x9]
    # ldp q2,q3,[x9,#0x20]
    vlw.v           v0, (s9)
    addi            s9, s9, 16
    vlw.v           v1, (s9)
    addi            s9, s9, 16
    vlw.v           v2, (s9)
    addi            s9, s9, 16
    vlw.v           v3, (s9)
    addi            s9, s9, 16
    addi            s9, s9, 0x20
    # ldp q4,q5,[x9,#0x60]
    # ldp q6,q7,[x9,#0x80]
    vlw.v           v4, (s9)
    addi            s9, s9, 16
    vlw.v           v5, (s9)
    addi            s9, s9, 16
    vlw.v           v6, (s9)
    addi            s9, s9, 16
    vlw.v           v7, (s9)
    addi            s9, s9, 16
    addi            s9, s9, -0x60

    # fadd v8.4s,v0.4s,v4.4s
    # fadd v9.4s,v1.4s,v5.4s
    # fadd v10.4s,v2.4s,v6.4s
    # fadd v11.4s,v3.4s,v7.4s
    vfadd.vv	v8, v0, v4
    vfadd.vv	v9, v1, v5
    vfadd.vv	v10, v2, v6
    vfadd.vv	v11, v3, v7
  
    # fsub v14.4s,v0.4s,v4.4s
    # fsub v15.4s,v1.4s,v5.4s
    # fsub v16.4s,v2.4s,v6.4s
    # fsub v17.4s,v3.4s,v7.4s
    vfsub.vv	v14, v0, v4
    vfsub.vv	v15, v1, v5
    vfsub.vv	v16, v2, v6
    vfsub.vv	v17, v3, v7

    # ldp q0,q1,[x9,#0x40]
    # ldp q2,q3,[x9,#0xa0]
    vlw.v           v0, (s9)
    addi            s9, s9, 16
    vlw.v           v1, (s9)
    addi            s9, s9, 0x50
    vlw.v           v2, (s9)
    addi            s9, s9, 16
    vlw.v           v3, (s9)
    addi            s9, s9, 16
    # fadd v12.4s,v0.4s,v2.4s
    # fadd v13.4s,v1.4s,v3.4s
    # fsub v18.4s,v0.4s,v2.4s
    # fsub v19.4s,v1.4s,v3.4s
    vfadd.vv	v12, v0, v2
    vfadd.vv	v13, v1, v3
    vfsub.vv	v18, v0, v2
    vfsub.vv	v19, v1, v3

    # ldr q0,[x3]
    flw     ft0, 0(a3)
    flw     ft1, 4(a3)
    flw     ft2, 8(a3)
    flw     ft3, 12(a3)

//line1: mid[v1,v4,v5,v6,v7,v(4)]
line1:
    # mov v1.16b,v26.16b
    # mov v2.16b,v27.16b
    # mov v3.16b,v28.16b
    vmv.v.v    v1, v26
    vmv.v.v    v2, v27
    vmv.v.v    v3, v28

    # fmla v1.4s,v14.4s,v0.s[0]
    # fmla v2.4s,v15.4s,v0.s[0]
    # fmla v3.4s,v16.4s,v0.s[0]
    vfmacc.vf	v1, ft0, v14
    vfmacc.vf	v2, ft0, v15
    vfmacc.vf	v3, ft0, v16
    # fadd v4.4s,v2.4s,v3.4s
    # fsub v5.4s,v2.4s,v3.4s
    vfadd.vv	v4, v2, v3
    vfsub.vv	v5, v2, v3
    # mov v2.16b,v29.16b
    # mov v3.16b,v30.16b
    vmv.v.v         v2, v29
    vmv.v.v         v3, v30
    # fmla v2.4s,v17.4s,v0.s[0]
    # fmla v3.4s,v18.4s,v0.s[0]
    vfmacc.vf	v2, ft0, v17
    vfmacc.vf	v3, ft0, v18
    # fadd v6.4s,v2.4s,v3.4s
    # fsub v7.4s,v2.4s,v3.4s
    vfadd.vv	v6, v2, v3
    vfsub.vv	v7, v2, v3

    # //end-mid ==========================
    # fadd v2.4s,v4.4s,v6.4s
    # mov  v3.16b,v4.16b
    # fadd v1.4s,v1.4s,v2.4s
    # fmla v3.4s,v6.4s,v0.s[1]
    vfadd.vv	v2, v4, v6
	vmv.v.v		v3, v4
    vfadd.vv	v1, v1, v2
    vfmacc.vf	v3, ft1, v6

    # cbz     x4, none_biases_1
    beqz        a4, none_biases_1

    # //bias
    # ld1r {v6.4s},[x4]
    vlw.v       v6, (a4)
    # mov v4.16b,v31.16b
    vmv.v.v		v4, v31
    # fadd v1.4s,v1.4s,v6.4s          //v1+bias
    # fmla v4.4s,v19.4s,v0.s[0]
    # fadd  v2.4s,v6.4s,v5.4s         //v2+bias
    # fadd  v4.4s,v4.4s,v5.4s 
    # fadd v3.4s,v3.4s,v6.4s          //v3+bias
    # fadd  v4.4s,v4.4s,v6.4s         //v4+bias
    # fmla v2.4s,v7.4s,v0.s[0]
    # fmla v4.4s,v7.4s,v0.s[2]
    vfadd.vv	v1, v1, v6          //v1+bias
    vfmacc.vf	v4, ft0, v19
    vfadd.vv	v2, v6, v5         //v2+bias
    vfadd.vv	v4, v4, v5 
    vfadd.vv	v3, v3, v6          //v3+bias
    vfadd.vv	v4, v4, v6         //v4+bias
    vfmacc.vf	v2, ft0, v7
    vfmacc.vf	v4, ft2, v7

    # b activation_1
    jal activation_1
    

    none_biases_1:
    # mov v4.16b,v31.16b
    # mov  v2.16b,v5.16b
    vmv.v.v		v4, v31
    vmv.v.v		v2, v5

    # fmla v4.4s,v19.4s,v0.s[0]
    # fmla v2.4s,v7.4s,v0.s[0]
    # fadd v4.4s,v4.4s,v5.4s
    # fmla v4.4s,v7.4s,v0.s[2]
    
    vfmacc.vf	v4, ft0, v19
    vfmacc.vf	v2, ft0, v7
    vfadd.vv	v4, v4, v5
    vfmacc.vf	v4, ft2, v7

    activation_1:
    # cmp     w5,0
    # blt     store_1
    blt         a5, x0, store_1

    # movi    d5, 0
    vmv.v.x		v5, x0
    # scvtf   s6,w5
    fcvt.s.w    ft6, a5
	

    # fmax    v1.4s, v1.4s, v5.4s
    # fmax    v2.4s, v2.4s, v5.4s
    # fmax    v3.4s, v3.4s, v5.4s
    # fmax    v4.4s, v4.4s, v5.4s
    vfmax.vv		v1, v1, v5
    vfmax.vv		v2, v2, v5
    vfmax.vv		v3, v3, v5
    vfmax.vv		v4, v4, v5

    # beq     store_1
    beqz        a5, store_1
    # dup     v6.4s,v6.s[0]
    vfmv.v.f        v6, ft6

    # fmin    v1.4s, v1.4s, v6.4s
    # fmin    v2.4s, v2.4s, v6.4s
    # fmin    v3.4s, v3.4s, v6.4s
    # fmin    v4.4s, v4.4s, v6.4s
    vfmin.vv		v1, v1, v6
    vfmin.vv		v2, v2, v6
    vfmin.vv		v3, v3, v6
    vfmin.vv		v4, v4, v6

    store_1:
    # st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x11]
    vssw.v          v1, (s1), t2
    addi            s1, s1, 1
    vssw.v          v2, (s1), t2
    addi            s1, s1, 1
    vssw.v          v3, (s1), t2
    addi            s1, s1, 1
    vssw.v          v4, (s1), t2
    addi            s1, s1, 1

//line2: mid[v1,v4,v5,v6,v7,v(4)]
line2:
    //v1
    # mov v1.16b,v20.16b
    # mov v2.16b,v21.16b
    # mov v3.16b,v22.16b
    vmv.v.v		v1, v20
    vmv.v.v		v2, v21
    vmv.v.v		v3, v22

    # fmla v1.4s,v8.4s,v0.s[1]
    # fmla v2.4s,v9.4s,v0.s[1]
    # fmla v3.4s,v10.4s,v0.s[1]
    vfmacc.vf	v1, ft1, v8
    vfmacc.vf	v2, ft1, v9
    vfmacc.vf	v3, ft1, v10

    # fadd v4.4s,v2.4s,v3.4s
    # fsub v5.4s,v2.4s,v3.4s
    vfadd.vv	v4, v2, v3
    vfsub.vv	v5, v2, v3

    # mov v2.16b,v23.16b
    # mov v3.16b,v24.16b
    vmv.v.v		v2, v23
    vmv.v.v		v3, v24

    # fmla v2.4s,v11.4s,v0.s[1]
    # fmla v3.4s,v12.4s,v0.s[1]
    # fadd v6.4s,v2.4s,v3.4s
    # fsub v7.4s,v2.4s,v3.4s
    vfmacc.vf	v2, ft1, v11
    vfmacc.vf	v3, ft1, v12
    vfadd.vv	v6, v2, v3
    vfsub.vv	v7, v2, v3
    # //end-mid ==========================
    # fadd v2.4s,v4.4s,v6.4s
    # mov  v3.16b,v4.16b
    # fadd v1.4s,v1.4s,v2.4s
    # fmla v3.4s,v6.4s,v0.s[1]
    vfadd.vv	v2, v4, v6
    vmv.v.v		v3, v4
    vfadd.vv	v1, v1, v2
    vfmacc.vf	v3, ft1, v6
    
    # cbz     x4, none_biases_2
    beqz        a4, none_biases_2

    # //bias
    # ld1r {v6.4s},[x4]
    vlw.v       v6, (a4)
    # mov v4.16b,v25.16b
    vmv.v.v		v4, v25

    # fadd v1.4s,v1.4s,v6.4s          //v1+bias
    # fmla v4.4s,v13.4s,v0.s[1]
    # fadd  v2.4s,v6.4s,v5.4s         //v2+bias
    # fadd  v4.4s,v4.4s,v5.4s 
    # fadd v3.4s,v3.4s,v6.4s          //v3+bias
    # fadd  v4.4s,v4.4s,v6.4s         //v4+bias
    # fmla v2.4s,v7.4s,v0.s[0]
    # fmla v4.4s,v7.4s,v0.s[2]
    vfadd.vv	v1, v1, v6          //v1+bias
    vfmacc.vf	v4, ft1, v13
    vfadd.vv	v2, v6, v5         //v2+bias
    vfadd.vv	v4, v4, v5 
    vfadd.vv	v3, v3, v6          //v3+bias
    vfadd.vv	v4, v4, v6         //v4+bias
    vfmacc.vf	v2, ft0, v7
    vfmacc.vf	v4, ft2, v7
    
    # b activation_2
    jal activation_2

    none_biases_2:
    # mov v4.16b,v25.16b
    # mov  v2.16b,v5.16b
    vmv.v.v		v4, v25
    vmv.v.v		v2, v5

    # fmla v4.4s,v13.4s,v0.s[1]
    # fmla v2.4s,v7.4s,v0.s[0]
    # fadd v4.4s,v4.4s,v5.4s
    # fmla v4.4s,v7.4s,v0.s[2]
    vfmacc.vf	v4, ft1, v13
    vfmacc.vf	v2, ft0, v7
    vfadd.vv	v4, v4, v5
    vfmacc.vf	v4, ft2, v7

    
    activation_2:
    # cmp     w5,0
    # blt     store_2
    blt         a5, x0, store_2

    # movi    d5, 0
    vmv.v.x     v5, x0
    # scvtf   s6,w5
	fcvt.s.w    ft6, a5
	

    # fmax    v1.4s, v1.4s, v5.4s
    # fmax    v2.4s, v2.4s, v5.4s
    # fmax    v3.4s, v3.4s, v5.4s
    # fmax    v4.4s, v4.4s, v5.4s
    vfmax.vv		v1, v1, v5
    vfmax.vv		v2, v2, v5
    vfmax.vv		v3, v3, v5
    vfmax.vv		v4, v4, v5

    # beq     store_2
    beqz        a5, store_2
    # dup     v6.4s,v6.s[0]
    vfmv.v.f        v6, ft6

    # fmin    v1.4s, v1.4s, v6.4s
    # fmin    v2.4s, v2.4s, v6.4s
    # fmin    v3.4s, v3.4s, v6.4s
    # fmin    v4.4s, v4.4s, v6.4s
    vfmin.vv		v1, v1, v6
    vfmin.vv		v2, v2, v6
    vfmin.vv		v3, v3, v6
    vfmin.vv		v4, v4, v6

    store_2:
    # st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x12]
    vssw.v          v1, (s2), t2
    addi            s2, s2, 1
    vssw.v          v2, (s2), t2
    addi            s2, s2, 1
    vssw.v          v3, (s2), t2
    addi            s2, s2, 1
    vssw.v          v4, (s2), t2
    addi            s2, s2, 1


//line0:
line0:
    // add 4 line,free(v8-v13)
    # fadd v20.4s,v20.4s,v8.4s
    # fadd v21.4s,v21.4s,v9.4s
    # fadd v22.4s,v22.4s,v10.4s
    # fadd v23.4s,v23.4s,v11.4s
    # fadd v24.4s,v24.4s,v12.4s
    # fadd v25.4s,v25.4s,v13.4s
    vfadd.vv	v20, v20, v8
    vfadd.vv	v21, v21, v9
    vfadd.vv	v22, v22, v10
    vfadd.vv	v23, v23, v11
    vfadd.vv	v24, v24, v12
    vfadd.vv	v25, v25, v13

    # //load v8-v13
    # ldp q8,q9,[x0]
    # ldp q10,q11,[x0,#0x20]
    # ldp q12,q13,[x0,#0x40]
    vlw.v       v8, (a0)
    addi        a0, a0, 16
    vlw.v       v9, (a0)
    addi        a0, a0, 16
    vlw.v       v10, (a0)
    addi        a0, a0, 16
    vlw.v       v11, (a0)
    addi        a0, a0, 16
    vlw.v       v12, (a0)
    addi        a0, a0, 16
    vlw.v       v13, (a0)
    addi        a0, a0, 16
    
    # //add get mid
    # fadd v1.4s,v20.4s,v8.4s
    # fadd v2.4s,v21.4s,v9.4s
    # fadd v3.4s,v22.4s,v10.4s
    # fadd v4.4s,v2.4s,v3.4s
    vfadd.vv	v1, v20, v8
    vfadd.vv	v2, v21, v9
    vfadd.vv	v3, v22, v10
    vfadd.vv	v4, v2, v3

    # fsub v5.4s,v2.4s,v3.4s
    # fadd v2.4s,v23.4s,v11.4s
    # fadd v3.4s,v24.4s,v12.4s
    # fadd v6.4s,v2.4s,v3.4s
    # fsub v7.4s,v2.4s,v3.4s
    vfsub.vv	v5, v2, v3
    vfadd.vv	v2, v23, v11
    vfadd.vv	v3, v24, v12
    vfadd.vv	v6, v2, v3
    vfsub.vv	v7, v2, v3

    # //end-mid ==========================
    # fadd v2.4s,v4.4s,v6.4s
    # mov  v3.16b,v4.16b
    # fadd v1.4s,v1.4s,v2.4s
    # fmla v3.4s,v6.4s,v0.s[1]
    vfadd.vv	v2, v4, v6
    vmv.v.v		v3, v4
    vfadd.vv	v1, v1, v2
    vfmacc.vf	v3, ft1, v6
    
    # cbz     x4, none_biases_3
    beqz        a4, none_biases_3
    # //bias
    # ld1r {v6.4s},[x4]
    vlw.v       v6, (a4)
    # fadd v4.4s,v25.4s,v13.4s
    # fadd v1.4s,v1.4s,v6.4s //v1+bias
    # fadd v4.4s,v4.4s,v5.4s
    # fadd v2.4s,v6.4s,v5.4s//v2+bias
    vfadd.vv	v4, v25, v13
    vfadd.vv	v1, v1, v6 //v1+bias
    vfadd.vv	v4, v4, v5
    vfadd.vv	v2, v6, v5//v2+bias
    # fmla v4.4s,v7.4s,v0.s[2]
    # fadd v3.4s,v3.4s,v6.4s//v3+bias
    # fmla v2.4s,v7.4s,v0.s[0]
    # fadd v4.4s,v4.4s,v6.4s //v4+bias
    vfmacc.vf	v4, ft2, v7
    vfadd.vv	v3, v3, v6//v3+bias
    vfmacc.vf	v2, ft0, v7
    vfadd.vv	v4, v4, v6 //v4+bias
    
    # b activation_3
    jal activation_3

    none_biases_3:
    # fadd v4.4s,v25.4s,v13.4s
    # mov  v2.16b,v5.16b
    # fadd v4.4s,v4.4s,v5.4s
    # fmla v2.4s,v7.4s,v0.s[0]
    # fmla v4.4s,v7.4s,v0.s[2]
    vfadd.vv	v4, v25, v13
    vmv.v.v		v2, v5
    vfadd.vv	v4, v4, v5
    vfmacc.vf	v2, ft0, v7
    vfmacc.vf	v4, ft2, v7

    
    activation_3:
    # cmp     w5,0
    # blt     store_3
    blt         a5, x0, store_3

    # movi    d5, 0
    vmv.v.x     v5, x0
    # scvtf   s6,w5
	fcvt.s.w    ft6, a5
	

    # fmax    v1.4s, v1.4s, v5.4s
    # fmax    v2.4s, v2.4s, v5.4s
    # fmax    v3.4s, v3.4s, v5.4s
    # fmax    v4.4s, v4.4s, v5.4s
    vfmax.vv		v1, v1, v5
    vfmax.vv		v2, v2, v5
    vfmax.vv		v3, v3, v5
    vfmax.vv		v4, v4, v5

    # beq     store_3
    beqz        a5, store_3
    # dup     v6.4s,v6.s[0]
    vfmv.v.f        v6, ft6

    # fmin    v1.4s, v1.4s, v6.4s
    # fmin    v2.4s, v2.4s, v6.4s
    # fmin    v3.4s, v3.4s, v6.4s
    # fmin    v4.4s, v4.4s, v6.4s
    vfmin.vv		v1, v1, v6
    vfmin.vv		v2, v2, v6
    vfmin.vv		v3, v3, v6
    vfmin.vv		v4, v4, v6

    store_3:
    # st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x1]
    vssw.v          v1, (a1), t2
    addi            a1, a1, 1
    vssw.v          v2, (a1), t2
    addi            a1, a1, 1
    vssw.v          v3, (a1), t2
    addi            a1, a1, 1
    vssw.v          v4, (a1), t2
    addi            a1, a1, 1

    
//line3:
line3:
    //load v8-v13
    # ldp q8,q9,[x10]
    # ldp q10,q11,[x10,#0x20]
    # ldp q12,q13,[x10,#0x40]
    vlw.v       v8, (s10)
    addi        s10, s10, 16
    vlw.v       v9, (s10)
    addi        s10, s10, 16
    vlw.v       v10, (s10)
    addi        s10, s10, 16
    vlw.v       v11, (s10)
    addi        s10, s10, 16
    vlw.v       v12, (s10)
    addi        s10, s10, 16
    vlw.v       v13, (s10)
    addi        s10, s10, 16
    
    # //v1
    # fadd v1.4s,v8.4s,v26.4s
    # fadd v2.4s,v9.4s,v27.4s
    # fadd v3.4s,v10.4s,v28.4s
    vfadd.vv	v1, v8, v26
    vfadd.vv	v2, v9, v27
    vfadd.vv	v3, v10, v28

    # fmla v1.4s,v14.4s,v0.s[2]
    # fmla v2.4s,v15.4s,v0.s[2]
    # fmla v3.4s,v16.4s,v0.s[2]
    vfmacc.vf	v1, ft2, v14
    vfmacc.vf	v2, ft2, v15
    vfmacc.vf	v3, ft2, v16

    # fadd v4.4s,v2.4s,v3.4s
    # fsub v5.4s,v2.4s,v3.4s
    # fadd v2.4s,v11.4s,v29.4s
    # fadd v3.4s,v12.4s,v30.4s
    vfadd.vv	v4, v2, v3
    vfsub.vv	v5, v2, v3
    vfadd.vv	v2, v11, v29
    vfadd.vv	v3, v12, v30

    # fmla v2.4s,v17.4s,v0.s[2]
    # fmla v3.4s,v18.4s,v0.s[2]
    vfmacc.vf	v2, ft2, v17
    vfmacc.vf	v3, ft2, v18

    # fadd v6.4s,v2.4s,v3.4s
    # fsub v7.4s,v2.4s,v3.4s
    vfadd.vv	v6, v2, v3
    vfsub.vv	v7, v2, v3
    # //end-mid ==========================
    # fadd v2.4s,v4.4s,v6.4s
    # mov  v3.16b,v4.16b
    # fadd v1.4s,v1.4s,v2.4s
    # fmla v3.4s,v6.4s,v0.s[1]
    vfadd.vv	v2, v4, v6
    vmv.v.v		v3, v4
    vfadd.vv	v1, v1, v2
    vfmacc.vf	v3, ft1, v6

    # cbz     x4, none_biases_4
    beqz        a4, none_biases_4
    # //bias
    # ld1r {v6.4s},[x4]
    vlw.v       v6, (a4)
    # fadd v4.4s,v13.4s,v31.4s
    # fadd v1.4s,v1.4s,v6.4s      //v1+bias
    # fmla v4.4s,v19.4s,v0.s[2]
    # fadd  v2.4s,v6.4s,v5.4s     //v2+bias
    vfadd.vv	v4, v13, v31
    vfadd.vv	v1, v1, v6      //v1+bias
    vfmacc.vf	v4, ft2, v19
    vfadd.vv	v2, v6, v5     //v2+bias

    # fmla v4.4s,v7.4s,v0.s[2]
    # fadd v3.4s,v3.4s,v6.4s      //v3+bias
    # fmla v2.4s,v7.4s,v0.s[0]
    # fadd v6.4s,v6.4s,v5.4s 
    # fadd v4.4s,v4.4s,v6.4s      //v4+bias
    vfmacc.vf	v4, ft2, v7
    vfadd.vv	v3, v3, v6      //v3+bias
    vfmacc.vf	v2, ft0, v7
    vfadd.vv	v6, v6, v5 
    vfadd.vv	v4, v4, v6      //v4+bias
    
    # b activation_4
    jal activation_4

    none_biases_4:
    # fadd v4.4s,v13.4s,v31.4s
    # mov  v2.16b,v5.16b
    # fmla v4.4s,v19.4s,v0.s[2]
    # fmla v2.4s,v7.4s,v0.s[0]
    # fadd v4.4s,v4.4s,v5.4s
    # fmla v4.4s,v7.4s,v0.s[2]
    vfadd.vv	v4, v13, v31
    vmv.v.v		v2, v5
    vfmacc.vf	v4, ft2, v19
    vfmacc.vf	v2, ft0, v7
    vfadd.vv	v4, v4, v5
    vfmacc.vf	v4, ft2, v7
    
    activation_4:
    # cmp     w5,0
    # blt     store_4
    blt         a5, x0, store_4

    # movi    d5, 0
    vmv.v.x     v5, x0
    # scvtf   s6,w5
	fcvt.s.w    ft6, a5
	

    # fmax    v1.4s, v1.4s, v5.4s
    # fmax    v2.4s, v2.4s, v5.4s
    # fmax    v3.4s, v3.4s, v5.4s
    # fmax    v4.4s, v4.4s, v5.4s
    vfmax.vv		v1, v1, v5
    vfmax.vv		v2, v2, v5
    vfmax.vv		v3, v3, v5
    vfmax.vv		v4, v4, v5

    # beq     store_4
    beqz        a5, store_4
    # dup     v6.4s,v6.s[0]
    vfmv.v.f        v6, ft6

    # fmin    v1.4s, v1.4s, v6.4s
    # fmin    v2.4s, v2.4s, v6.4s
    # fmin    v3.4s, v3.4s, v6.4s
    # fmin    v4.4s, v4.4s, v6.4s
    vfmin.vv		v1, v1, v6
    vfmin.vv		v2, v2, v6
    vfmin.vv		v3, v3, v6
    vfmin.vv		v4, v4, v6

    store_4:
    # st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x13]
    vssw.v          v1, (s3), t2
    addi            s3, s3, 1
    vssw.v          v2, (s3), t2
    addi            s3, s3, 1
    vssw.v          v3, (s3), t2
    addi            s3, s3, 1
    vssw.v          v4, (s3), t2
    addi            s3, s3, 1


return:
	ld          s0, 0(sp)
    ld          s1, 8(sp)
    ld          s2, 16(sp)
    ld          s3, 24(sp)
    ld          s4, 32(sp)
    ld          s5, 40(sp)
    ld          s6, 48(sp)
    ld          s7, 56(sp)
    ld          s8, 64(sp)
    ld          s9, 72(sp)
    ld          s10, 80(sp)
    ld          s11, 88(sp)
    addi        sp, sp, 96

	ret
        .end

