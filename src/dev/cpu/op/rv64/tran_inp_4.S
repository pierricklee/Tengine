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

/* Constants for vmerge masks */
.balign 16
vmerge_4:
  .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01   


//a0: inp
//a1: out = inp_ptr + c*4 
//a2: ker
//a3: inw
//a4: inc_4*(sizeof(float))
//a5: inhw (to prefetch next channel)

# x11  --  t1
# x12  --  t2
# x13  --  t3
# x14  --  t4
# x15  --  t5
# x16  --  t6
    .section .text,"ax"
    .align 5

    .type tran_inp_4 STT_FUNC
    .global tran_inp_4
    .hidden tran_inp_4
    
tran_inp_4:
    li          t2, 4
    vsetvli     x0, t2, e32, m1, d1
    addi    sp, sp, -96
    # sub	sp, sp, 0x40
    # stp	d8, d9, [sp]
	# stp	d10,d11,[sp, 0x10]
	# stp	d12,d13,[sp, 0x20]
	# stp	d14,d15,[sp, 0x30]
    sd      s0, 0(sp)              # 
    sd      s1, 8(sp)              # x11
    sd      s2, 16(sp)             # x12
    sd      s3, 24(sp)             # x13
    sd      s4, 32(sp)             # x14
    sd      s5, 40(sp)             # x15
    sd      s6, 48(sp)             # x16
    sd      s7, 56(sp)             # x17
    sd      s8, 64(sp)             # x18
    sd      s9, 72(sp)             # x9
    sd      s10, 80(sp)            # x10
    sd      s11, 88(sp) 

    la      s0, vmerge_4            # 1111 0000 0000 0000 
    
    
comput_idx:
    # lsl	x3, x3, 0x2 
    slli    a3, a3, 2
    # add x11,x0,x3       
    add     s1, a0, a3
    # add x12,x0,x3,LSL 1     
    slli    t0, a3, 1
    add     s2, a0, t0
    # add	x13,x11,x3, LSL 1  
    add     s3, s1, t0
    # add	x14,x0,x3,LSL 2    
    slli    t0, a3, 2
    add     s4, a0, t0

    # lsl x15,x4,0x2
    slli    s5, a4, 2
    # lsl x16,x4,0x1
    slli    s6, a4, 1
    # add x17,x16,x15 //[1,0]=1*6+0
    add     s7, s6, s5

load:
    //v0 1 2 3 4 5 
    # ld4 {v0.4s, v1.4s, v2.4s,v3.4s}, [x11]
    li              t0, 4
    vlsw.v          v24, (s1), t0
    addi            s1, s1, 4
    vlsw.v          v1, (s1), t0
    addi            s1, s1, 4
    vlsw.v          v2, (s1), t0
    addi            s1, s1, 4
    vlsw.v          v3, (s1), t0
    # ldr	q31, [x11, 0x40]	
    addi            s1, s1, 0x30
    vlw.v           v31, (s1)
    addi            s1, s1, -0x40
    
    # 放入 v24, v1, v2, v3, v4, v5
    # ext v4.16b,v0.16b,v31.16b,#4
    vslidedown.vi   v25, v24, 12
    vslideup.vi     v26, v31, 4
    vlb.v           v0, (s0)
    vmerge.vvm      v4, v26, v25, v0
    # ext v31.16b,v31.16b,v31.16b,#4
    vslidedown.vi   v25, v31, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v31, v26, v25, v0
    # ext v5.16b,v1.16b,v31.16b,#4
    vslidedown.vi   v25, v1, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v5, v26, v25, v0

    # free v14, v15, v0

    # //v6 7 8 9 10 11
    # load 放入 v6, v7, v8, v9, v10, v11
    # lsl	x5, x5, 0x2 
    slli    a5, a5, 2
    # ld4 {v6.4s, v7.4s, v8.4s,v9.4s}, [x12]
    vlsw.v  v6, (s2), t0
    addi    s2, s2, 4
    vlsw.v  v7, (s2), t0
    addi    s2, s2, 4
    vlsw.v  v8, (s2), t0
    addi    s2, s2, 4
    vlsw.v  v9, (s2), t0
    # ldr	q31, [x12, 0x40]	
    addi    s2, s2, 0x30
    vlw.v   v31, (s2)
    
    # 放入 v6, v7, v8, v9, v10, v11
    # ext v10.16b,v6.16b,v31.16b,#4
    vslidedown.vi           v25, v6, 12
    vslideup.vi             v26, v31, 4
    vmerge.vvm              v10, v26, v25, v0
    # ext v31.16b,v31.16b,v31.16b,#
    vslidedown.vi           v25, v31, 12
    vslideup.vi             v26, v31, 4
    vmerge.vvm              v31, v26, v25, v0
    # ext v11.16b,v7.16b,v31.16b,#4
    vslidedown.vi           v25, v7, 12
    vslideup.vi             v26, v31, 4
    vmerge.vvm              v11, v26, v25, v0

    # add x6,x5,#0x40
    addi      a6, a5, 0x40

    # //v12 13 14 15 16 17 
    # load 放入 v12, v13, v14, v15
    # ld4 {v12.4s, v13.4s, v14.4s,v15.4s}, [x13]
    # ldr	q31, [x13, 0x40]
    vlsw.v  v12, (s3), t0
    addi    s3, s3, 4
    vlsw.v  v13, (s3), t0
    addi    s3, s3, 4
    vlsw.v  v14, (s3), t0
    addi    s3, s3, 4
    vlsw.v  v15, (s3), t0	
    addi    s3, s3, 0x30
    vlw.v   v31, (s3)

    # add x18,x1,x17	
    add     s8, a1, s7
    # 放入 v12, v13, v14, v15, v16, v17
    # ext v16.16b,v12.16b,v31.16b,#4
    vslidedown.vi           v25, v12, 12
    vslideup.vi             v26, v31, 4
    vmerge.vvm              v16, v26, v25, v0
    # ext v31.16b,v31.16b,v31.16b,#
    vslidedown.vi           v25, v31, 12
    vslideup.vi             v26, v31, 4
    vmerge.vvm              v31, v26, v25, v0
    # ext v17.16b,v13.16b,v31.16b,#
    vslidedown.vi           v25, v13, 12
    vslideup.vi             v26, v31, 4
    vmerge.vvm              v17, v26, v25, v0

    # 放入 v18, v19, v20, v21, v22, v23
    # ld4 {v18.4s, v19.4s, v20.4s,v21.4s}, [x14]
    # ldr	q31, [x14, 0x40]	
    vlsw.v  v18, (s4), t0
    addi    s4, s4, 4
    vlsw.v  v19, (s4), t0
    addi    s4, s4, 4
    vlsw.v  v20, (s4), t0
    addi    s4, s4, 4
    vlsw.v  v21, (s4), t0	
    addi    s4, s4, 0x30
    vlw.v   v31, (s4)
    
    # ext v22.16b,v18.16b,v31.16b,#4
    vslidedown.vi       v25, v18, 12
    vslideup.vi         v26, v31, 4
    vmerge.vvm          v22, v26, v25, v0
    # ext v31.16b,v31.16b,v31.16b,#4
    vslidedown.vi       v25, v31, 12
    vslideup.vi         v26, v31, 4
    vmerge.vvm          v31, v26, v25, v0
    # ext v23.16b,v19.16b,v31.16b,#4
    vslidedown.vi       v25, v19, 12
    vslideup.vi         v26, v31, 4
    vmerge.vvm          v23, v26, v25, v0

    # ldr	q30, [x2]
    flw     ft0, 0(a2)
    flw     ft1, 4(a2)
    flw     ft2, 8(a2)
    flw     ft3, 12(a2)
    vmv.v.v  v0, v24

line1://add x18,x1,x17,[x18+=x4]
   
    # fadd	v24.4s, v13.4s, v19.4s
    # fadd	v25.4s, v14.4s, v20.4s
    # fadd	v26.4s, v15.4s, v21.4s
    # fadd	v27.4s, v16.4s, v22.4s
    vfadd.vv    v24, v13, v19
    vfadd.vv    v25, v14, v20
    vfadd.vv    v26, v15, v21
    vfadd.vv    v27, v16, v22

    # fmls	v24.4s, v1.4s,  v30.s[2]
    # fmls	v25.4s, v2.4s,  v30.s[2]
    # fmls	v26.4s, v3.4s,  v30.s[2]
    # fmls	v27.4s, v4.4s,  v30.s[2]
    vfnmsac.vf  v24, ft2, v1
    vfnmsac.vf  v25, ft2, v2
    vfnmsac.vf  v26, ft2, v3
    vfnmsac.vf  v27, ft2, v4 

    # movi	d28, 0x0
	vmv.v.i     v28, 0
    # fmls	v24.4s, v7.4s,  v30.s[2]
    # fmls	v25.4s, v8.4s,  v30.s[2]
    # fmls	v26.4s, v9.4s,  v30.s[2]
    # fmls	v27.4s, v10.4s,  v30.s[2]
    vfnmsac.vf  v24, ft2, v7
    vfnmsac.vf  v25, ft2, v8
    vfnmsac.vf  v26, ft2, v9
    vfnmsac.vf  v27, ft2, v10     
    
    # //v29 as inp0
    # fadd    v29.4s, v18.4s, v12.4s
    vfadd.vv    v29, v18, v12
    # fmls	v28.4s, v25.4s,  v30.s[3]
    vfnmsac.vf  v28, ft3, v25
    # fadd	v31.4s, v26.4s,  v27.4s
    vfadd.vv    v31, v26, v27

    # fmls	v29.4s, v0.4s,  v30.s[2]
    # fmla	v28.4s, v27.4s,  v30.s[0]
    # fmls	v29.4s, v6.4s,  v30.s[2]
    # fmls	v31.4s, v24.4s,  v30.s[2]
    # fmla	v28.4s, v29.4s,  v30.s[2]
    vfnmsac.vf  v29, ft2, v0
    vfmacc.vf   v28, ft0, v27
    vfnmsac.vf  v29, ft2, v6
    vfnmsac.vf  v31, ft2, v24
    vfmacc.vf   v28, ft2, v29
  
    # str q28, [x18]
    vlw.v   v28, (s8)

    # fmls	v31.4s, v25.4s,  v30.s[2]
    # add x18,x18,x4
    # fsub	v29.4s, v27.4s, v26.4s
    # str q31, [x18]
    vfnmsac.vf  v31, ft2, v25
    add         s8, s8, a4
    vsub.vv     v29, v27, v26 
    vlw.v       v31, (s8)
	
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # add x18,x18,x4
    # fmls	v29.4s, v25.4s,  v30.s[2]
    vfmacc.vf   v29, ft2, v24
    add         s8, s8, a4
    vfnmsac.vf  v29, ft2, v25
 
    # str q29, [x18]
    # movi	d28, 0x0
    # movi	d29, 0x0
    vlw.v       v29, (s8)
	vmv.v.i     v28, 0
    vmv.v.i     v29, 0

    # fsub    v31.4s,v27.4s, v25.4s
    # fmls	v28.4s, v24.4s,  v30.s[1]
    # fadd    v25.4s, v17.4s,v23.4s
    vfsub.vv    v31, v27, v25
    vfnmsac.vf  v28, ft1, v24
    vfadd.vv    v25, v17, v23

    # fadd    v28.4s,v28.4s,v31.4s
    # fmls	v25.4s, v5.4s,  v30.s[2]
    # fmla	v31.4s, v24.4s,  v30.s[1]
    vfadd.vv    v28, v28, v31
    vfnmsac.vf  v25, ft2, v5
    vfmacc.vf   v31, ft1, v24

    # fmla	v28.4s, v26.4s,  v30.s[1]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    vfmacc.vf   v28, ft1, v26
    vfmacc.vf   v29, ft2, v24
    
    # add x18,x18,x4
    # str q28, [x18] 
    add         s8, s8, a4
    vsw.v       v28, (s8)

    # fmls	v25.4s, v11.4s,  v30.s[2]
    # fmls	v29.4s, v26.4s,  v30.s[3]
    # fmls	v31.4s, v26.4s,  v30.s[1]
    vfnmsac.vf  v25, ft2, v11
    vfnmsac.vf  v29, ft3, v26
    vfnmsac.vf  v31, ft1, v26
    
    # add x18,x18,x4
    # str q31, [x18]
    add         s8, s8, a4
    vsw.v       v31, (s8)
    
    # //v25 as inp0
    # fmla	v29.4s, v25.4s,  v30.s[0]
    # add x18,x18,x4
    # str q29, [x18]
    vfmacc.vf   v29, ft0, v25
    add         s8, s8, a4
    vsw.v       v29, (s8)

line2://add x18,x1,x17,LSL 1  (add x9,x18,x17)
    # fsub    v24.4s, v19.4s, v13.4s
    # fsub    v25.4s, v20.4s, v14.4s
    # fsub    v26.4s, v21.4s, v15.4s
    # fsub    v27.4s, v22.4s, v16.4s
    vfsub.vv    v24, v19, v13
    vfsub.vv    v25, v20, v14
    vfsub.vv    v26, v21, v15
    vfsub.vv    v27, v22, v16

    # add x18,x1,x17,LSL 1 
    slli    t0, s7, 1
    add     s8, a1, t0

    # fmla	v24.4s, v1.4s,  v30.s[2]
    # fmla	v25.4s, v2.4s,  v30.s[2] 
    # fmla	v26.4s, v3.4s,  v30.s[2]
    # fmla	v27.4s, v4.4s,  v30.s[2]
    vfmacc.vf   v24, ft2, v1
    vfmacc.vf   v25, ft2, v2
    vfmacc.vf   v26, ft2, v3
    vfmacc.vf   v27, ft2, v4

    # fmls	v24.4s, v7.4s,  v30.s[2]
    # fmls	v25.4s, v8.4s,  v30.s[2]
    # fmls	v26.4s, v9.4s,  v30.s[2]
    # fmls	v27.4s, v10.4s,  v30.s[2]
    vfnmsac.vf  v24, ft2, v7
    vfnmsac.vf  v25, ft2, v8
    vfnmsac.vf  v26, ft2, v9
    vfnmsac.vf  v27, ft2, v10

    #  movi	d28, 0x0
    vmv.v.x         v28, x0
    # add x9,x18,x17
    add             s9, s8, s7

    # //v29 as inp0
    # fsub    v29.4s, v18.4s,  v12.4s
    # fmls	  v28.4s, v25.4s,  v30.s[3]
    # fadd	  v31.4s, v26.4s,  v27.4s
    vfsub.vv        v29, v18, v12
    vfnmsac.vf      v28, ft3, v25
    vfsub.vv        v31, v26, v27

    # fmla	v29.4s, v0.4s,  v30.s[2]
    # fmla	v28.4s, v27.4s,  v30.s[0]
    # fmls	v29.4s, v6.4s,  v30.s[2]
    vfmacc.vf       v29, ft2, v0
    vfmacc.vf       v28, ft0, v27
    vfnmsac.vf      v29, ft2, v30

    # fmls	v31.4s, v24.4s,  v30.s[2]
    # fmla	v28.4s, v29.4s,  v30.s[2]
    # str q28, [x18]
    vfnmsac.vf	v31, ft2, v24
	vfmacc.vf	v28, ft2, v29
	vsw.v		    v28, (s8)

    # fmls	v31.4s, v25.4s,  v30.s[2]
    # add x18,x18,x4
    # fsub	v29.4s, v27.4s,  v26.4s
	vfnmsac.vf	v31, ft2, v25
	add		    s8, s8,  a4
	vfsub.vv	v29, v27, v26

    # str q31, [x18]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # add x18,x18,x4
    # fmls	v29.4s, v25.4s,  v30.s[2]
    # str q29, [x18]
    vsw.v		    v31, (s8)
    vfmacc.vf	    v29, ft2, v24
    add             s8, s8, a4
    vfnmsac.vf	    v29, ft2, v25
    vsw.v		    v29, (s8)
	
    # movi	d28, 0x0
    # movi	d29, 0x0
	vmv.v.i     v28, 0
    vmv.v.i     v29, 0

    # fsub    v31.4s,v27.4s, v25.4s
    # fmls	v28.4s, v24.4s,  v30.s[1]
    # fsub    v25.4s,v23.4s,v17.4s
    # fadd    v28.4s,v28.4s,v31.4s    
	vfsub.vv	v31, v27, v25
	vfnmsac.vf	v28, ft1, v24
	vfsub.vv	v25, v23, v17
	vfadd.vv	v28, v28, v31
    
    # fmla	v25.4s, v5.4s,  v30.s[2]
    # fmla	v31.4s, v24.4s,  v30.s[1]
    # fmla	v28.4s, v26.4s,  v30.s[1]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    vfmacc.vf	v25, ft2, v5
	vfmacc.vf	v31, ft1, v24
	vfmacc.vf	v28, ft1, v26
	vfmacc.vf	v29, ft2, v24

    # add x18,x18,x4
    # str q28, [x18]
    add		    s8, s8,  a4
    vlw.v		v28, (s8)

    # fmls	v25.4s, v11.4s,  v30.s[2]
    # fmls	v29.4s, v26.4s,  v30.s[3]
    # fmls	v31.4s, v26.4s,  v30.s[1]
    vfnmsac.vf	v25, ft2, v11
	vfnmsac.vf	v29, ft3, v26
	vfnmsac.vf	v31, ft1, v26
    
    # add x18,x18,x4
    # str q31, [x18]
    add		    s8, s8, a4
	vsw.v		v31, (s8)

    # //v25 as inp0
    # fmla	v29.4s, v25.4s,  v30.s[0]
    # add x18,x18,x4
    # str q29, [x18]
	vfmacc.vf	v29, ft0, v25
	add		    s8, s8, a4
	vsw.v		v29, (s8)

line3://mov x18,x9 [x18+=x4]
    # fsub	v24.4s, v19.4s,  v7.4s
    # fsub	v25.4s, v20.4s,  v8.4s
    # fsub	v26.4s, v21.4s,  v9.4s
    # fsub	v27.4s, v22.4s,  v10.4s
    vfsub.vv	v24, v19, v7
    vfsub.vv	v25, v20, v8
    vfsub.vv	v26, v21, v9
    vfsub.vv	v27, v22, v10
    # mov x18,x9
    mv			s8,  s9
    # fmls	v24.4s, v1.4s,  v30.s[1]
    # fmls	v25.4s, v2.4s,  v30.s[1]
    # fmls	v26.4s, v3.4s,  v30.s[1]
    # fmls	v27.4s, v4.4s,  v30.s[1]
    vfnmsac.vf	v24, ft1, v1
    vfnmsac.vf	v25, ft1, v2
    vfnmsac.vf	v26, ft1, v3
    vfnmsac.vf	v27, ft1, v4
    
    # movi	d28, 0x0
    # fmla	v24.4s, v13.4s,  v30.s[1]
    # fmla	v25.4s, v14.4s,  v30.s[1]
    vmv.v.i     v28, 0
    vfmacc.vf	v24, ft1, v13
    vfmacc.vf	v25, ft1, v14

    # fmla	v26.4s, v15.4s,  v30.s[1]
    # fmla	v27.4s, v16.4s,  v30.s[1]
    vfmacc.vf	v26, ft1, v15
    vfmacc.vf	v27, ft1, v16


    //v29 as inp0
    # fsub    v29.4s,v18.4s,v6.4s
    # fmls	v28.4s, v25.4s,  v30.s[3]
    # fadd	v31.4s, v26.4s,  v27.4s
    vfsub.vv	v29, v18, v6
    vfnmsac.vf	v28, ft3, v25
    vfadd.vv	v31, v26, v27

    # fmla	v29.4s, v12.4s,  v30.s[1]
    # fmla	v28.4s, v27.4s,  v30.s[0]
    # fmls	v29.4s, v0.4s,  v30.s[1]
    vfmacc.vf	v29, ft1, v12
    vfmacc.vf	v28, ft0, v27
    vfnmsac.vf	v29, ft1, v0
    
    # fmls	v31.4s, v24.4s,  v30.s[2]
    # fmla	v28.4s, v29.4s,  v30.s[2]
    vfnmsac.vf	v31, ft2, v24
    vfmacc.vf	v28, ft2, v29

    # str q28, [x18]
    # fmls	v31.4s, v25.4s,  v30.s[2]
    # add x18,x18,x4
    # fsub	v29.4s, v27.4s, v26.4s
    # str q31, [x18]
    vlw.v		v28, (s8)
    vfnmsac.vf	v31, ft2, v25
    add		s8, s8,  a4
    vfsub.vv	v29, v27, v26
    vlw.v		v31, (s8)
    
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # add x18,x18,x4
    # fmls	v29.4s, v25.4s,  v30.s[2]
    # str q29, [x18]
    vfmacc.vf	v29, ft2, v24
    add		s8, s8,  a4
    vfnmsac.vf	v29, ft2, v25
    vlw.v		v29, (s8)
	
    # movi	d28, 0x0
    # movi	d29, 0x0
	vmv.v.i     v28, 0
    vmv.v.i     v29, 0
    # fsub    v31.4s,v27.4s, v25.4s
    # fmls	v28.4s, v24.4s,  v30.s[1]
    # fsub    v25.4s,v23.4s,v11.4s
    vfsub.vv	v31, v27, v25
    vfnmsac.vf	v28, ft1, v24
    vfsub.vv	v25, v23, v11

    # fadd    v28.4s,v28.4s,v31.4s
    # fmla	v25.4s, v17.4s,  v30.s[1]
    # fmla	v31.4s, v24.4s,  v30.s[1]
    vfadd.vv	v28, v28, v31
    vfmacc.vf	v25, ft1, v17
    vfmacc.vf	v31, ft1, v24

    # fmla	v28.4s, v26.4s,  v30.s[1]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # add x18,x18,x4
    # str q28, [x18]
    vfmacc.vf	v28, ft1, v26
    vfmacc.vf	v29, ft2, v24
    add		s8, s8,  a4
    vlw.v		v28, (s8)

    # fmls	v25.4s, v5.4s,  v30.s[1]
    # fmls	v29.4s, v26.4s,  v30.s[3]
    # fmls	v31.4s, v26.4s,  v30.s[1]
    vfnmsac.vf	v25, ft1, v5
    vfnmsac.vf	v29, ft3, v26
    vfnmsac.vf	v31, ft1, v26

    # add x18,x18,x4
    # str q31, [x18]
    # //v25 as inp0
    # fmla	v29.4s, v25.4s,  v30.s[0]
    add		s8, s8,  a4
    vlw.v		v31, (s8)
    vfmacc.vf	v29, ft0, v25

    # add x18,x18,x4
    # str q29, [x18]
    add		s8, s8,  a4
    vlw.v		v29, (s8)

line4://add x18,x1,x17,LSL 2  ((add x9,x18,x17))
    # fsub	v24.4s, v19.4s,  v7.4s
    # fsub	v25.4s, v20.4s,  v8.4s
    # fsub	v26.4s, v21.4s,  v9.4s
    # fsub	v27.4s, v22.4s,  v10.4s
    vfsub.vv	v24, v19, v7
    vfsub.vv	v25, v20, v8
    vfsub.vv	v26, v21, v9
    vfsub.vv	v27, v22, v10

    # add x18,x1,x17,LSL 2
    slli    t0, s7, 2
    add     s8, a1, t0

    # fmla	v24.4s, v1.4s,  v30.s[1]
    # fmla	v25.4s, v2.4s,  v30.s[1]
    vfmacc.vf	v24, ft1, v1
    vfmacc.vf	v25, ft1, v2
    
    # fmla	v26.4s, v3.4s,  v30.s[1]
    # fmla	v27.4s, v4.4s,  v30.s[1]
    vfmacc.vf	v26, ft1, v3
    vfmacc.vf	v27, ft1, v4
    # add x9,x18,x17
    add         s9, s8, s7
    # fmls	v24.4s, v13.4s,  v30.s[1]
    # fmls	v25.4s, v14.4s,  v30.s[1]
    vfnmsac.vf	v24, ft1, v13
    vfnmsac.vf	v25, ft1, v14
    # movi	d28, 0x0
	vmv.v.i     v28, 0
    # fmls	v26.4s, v15.4s,  v30.s[1]
    # fmls	v27.4s, v16.4s,  v30.s[1]
    vfnmsac.vf	v26, ft1, v15
    vfnmsac.vf	v27, ft1, v16

    # //v29 as inp0
    # fsub    v29.4s,v18.4s,v6.4s
    # fmla	v28.4s, v27.4s,  v30.s[0]
    # fadd	v31.4s, v26.4s,  v27.4s
    vfsub.vv	v29, v18, v6
    vfmacc.vf	v28, ft0, v27
    vfadd.vv	v31, v26, v27

    # fmla	v29.4s, V0.4s,  v30.s[1]
    # fmls	v28.4s, v25.4s,  v30.s[3]
    # fmls	v29.4s, v12.4s,  v30.s[1]
    vfmacc.vf	v29, ft1, v0
    vfnmsac.vf	v28, ft3, v25
    vfnmsac.vf	v29, ft1, v12

    # fmls	v31.4s, v24.4s,  v30.s[2]
    # fmla	v28.4s, v29.4s,  v30.s[2]
    vfnmsac.vf	v31, ft2, v24
    vfmacc.vf	v28, ft2, v29
    
    # str q28, [x18]
    vsw.v		v28, (s8)

    # //28 29 28 29
    # fmls	v31.4s, v25.4s,  v30.s[2]
    # add x18,x18,x4
    # fsub	v29.4s, v27.4s,v26.4s
    vfnmsac.vf	v31, ft2, v25
    add		s8, s8,  a4
    vfsub.vv	v29, v27, v26

    # str q31, [x18]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # add x18,x18,x4
    # fmls	v29.4s, v25.4s,  v30.s[2]
    # str q29, [x18]
    vsw.v		v31, (s8)
    vfmacc.vf	v29, ft2, v24
    add		    s8, s8,  a4
    vfnmsac.vf	v29, ft2, v25
    vsw.v		v29, (s8)
	# movi	d28, 0x0
	vmv.v.i     v28, 0
	# movi	d29, 0x0
    vmv.v.i     v29, 0
    
    
    # fsub    v31.4s,v27.4s, v25.4s
    # fmls	v28.4s, v24.4s,  v30.s[1]
    # fsub    v25.4s,v23.4s,v11.4s
    # fadd    v28.4s,v28.4s,v31.4s
    # fmla	v25.4s, v5.4s,  v30.s[1]
    # fmla	v31.4s, v24.4s,  v30.s[1]
    # fmla	v28.4s, v26.4s,  v30.s[1]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    vfsub.vv	v31, v27, v25
    vfnmsac.vf	v28, ft1, v24
    vfsub.vv	v25, v23, v11
    vfadd.vv	v28, v28, v31
    vfmacc.vf	v25, ft1, v5
    vfmacc.vf	v31, ft1, v24
    vfmacc.vf	v28, ft1, v26
    vfmacc.vf	v29, ft2, v24

    # add x18,x18,x4
    # str q28, [x18]
    # fmls	v25.4s, v17.4s,  v30.s[1]
    # fmls	v29.4s, v26.4s,  v30.s[3]
    # fmls	v31.4s, v26.4s,  v30.s[1]
    # add x18,x18,x4
    # str q31, [x18]
    add		    s8, s8,  a4
    vsw.v		v28, (s8)
    vfnmsac.vf	v25, ft1, v17
    vfnmsac.vf	v29, ft3, v26
    vfnmsac.vf	v31, ft1, v26
    add		    s8, s8,  a4
    vsw.v		v31, (s8)
    
    # //v25 as inp0
    # fmla	v29.4s, v25.4s,  v30.s[0]
    # add x18,x18,x4
    # str q29, [x18]
    vfmacc.vf	v29, ft0, v25
    add		    s8, s8,  a4
    vsw.v		v29, (s8)

line0: // addr:   str q28, [x1], add x18,x1,x4
    //v0 1 2 3 4 5 
    # ld4 {v0.4s, v1.4s, v2.4s,v3.4s}, [x0]
    li              t0, 4
    vlsw.v          v18, (a0), t0
    addi            a0, a0, 4
    vlsw.v          v19, (a0), t0
    addi            a0, a0, 4
    vlsw.v          v20, (a0), t0
    addi            a0, a0, 4
    vlsw.v          v21, (a0), t0
    addi            a0, a0, 0x30
    # ldr	q31, [x0, 0x40]	
    vlw.v           v31, (a0)
    addi            a0, a0, -0x40
    # ext v4.16b,v0.16b,v31.16b,#4
    vslidedown.vi   v25, v24, 12
    vslideup.vi     v26, v31, 4
    vlb.v           v0, (s0)
    vmerge.vvm      v4, v26, v25, v0
    # ext v31.16b,v31.16b,v31.16b,#4
    vslidedown.vi   v25, v31, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v31, v26, v25, v0
    # ext v5.16b,v1.16b,v31.16b,#4
    vslidedown.vi   v25, v1, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v5, v26, v25, v0
    
    # movi	d29, 0x0
    # movi	d24, 0x0
	# movi	d25, 0x0
	# movi	d26, 0x0
	# movi	d27, 0x0
    # movi	d28, 0x0
    # movi	d31, 0x0
    vmv.v.i     v29, 0
    vmv.v.i     v24, 0
    vmv.v.i     v25, 0
    vmv.v.i     v26, 0
    vmv.v.i     v27, 0
	vmv.v.i     v28, 0
    vmv.v.i     v31, 0


    # fmla	v29.4s, V0.4s,  v30.s[2]
    # fmla	v24.4s, v1.4s,  v30.s[2]
    vfmacc.vf	v29, ft2, v0
    vfmacc.vf	v24, ft2, v1
    
    # fmla	v25.4s, v2.4s,  v30.s[2]
    # fmls	v29.4s, v6.4s,  v30.s[3]
    vfmacc.vf	v25, ft2, v2
    vfnmsac.vf	v29, ft3, v6
    
    # fmla	v26.4s, v3.4s,  v30.s[2]
    # fmla	v27.4s, v4.4s,  v30.s[2]    
    # fmla	v29.4s, v18.4s,  v30.s[0]
    # fmls	v24.4s, v7.4s,  v30.s[3]
    # fmls	v25.4s, v8.4s,  v30.s[3]
    # fmls	v26.4s, v9.4s,  v30.s[3]
    # fmla	v28.4s, v29.4s,  v30.s[2]
    vfmacc.vf	v26, ft2, v3
    vfmacc.vf	v27, ft2, v4    
    vfmacc.vf	v29, ft0, v18
    vfnmsac.vf	v24, ft3, v7
    vfnmsac.vf	v25, ft3, v8
    vfnmsac.vf	v26, ft3, v9
    vfmacc.vf	v28, ft2, v29
   
    # fmls	v27.4s, v10.4s,  v30.s[3]
    # fmla	v25.4s, v20.4s,  v30.s[0]
    # fmla	v24.4s, v19.4s,  v30.s[0]
    # fmla	v27.4s, v22.4s,  v30.s[0]
    vfnmsac.vf	v27, ft3, v10
    vfmacc.vf	v25, ft0, v20
    vfmacc.vf	v24, ft0, v19
    vfmacc.vf	v27, ft0, v22

    # movi	d29, 0x0
    vmv.v.i     v29, 0
    # fmls	v28.4s, v25.4s,  v30.s[3]
    # fmls	v31.4s, v24.4s,  v30.s[2]
    # fmla	v26.4s, v21.4s,  v30.s[0]
    # fmls	v31.4s, v25.4s,  v30.s[2]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # fmla	v28.4s, v27.4s,  v30.s[0]
    vfnmsac.vf	v28, ft3, v25
    vfnmsac.vf	v31, ft2, v24
    vfmacc.vf	v26, ft0, v21
    vfnmsac.vf	v31, ft2, v25
    vfmacc.vf	v29, ft2, v24
    vfmacc.vf	v28, ft0, v27
    
    # fmls	v29.4s, v25.4s,  v30.s[2]
    # fmla	v31.4s, v26.4s,  v30.s[0]
    vfnmsac.vf	v29, ft2, v25
    vfmacc.vf	v31, ft0, v26
    # str q28, [x1]
    # add x18,x1,x4
    vsw.v       v28, (a1)
    add         s8, a1, a4
    # fmla	v31.4s, v27.4s,  v30.s[0]
    vfmacc.vf	v31, ft0, v27
    # movi	d28, 0x0
	vmv.v.i     v28, 0
    
	# fmls	v29.4s, v26.4s,  v30.s[0]
    # fmls	v28.4s, v24.4s,  v30.s[1]
    vfnmsac.vf	v29, ft0, v26
    vfnmsac.vf	v28, ft1, v24
    # str q31, [x18]
    vsw.v		v31, (s8)

    # fmla	v29.4s, v27.4s,  v30.s[0]
    # fmla	v28.4s, v26.4s,  v30.s[1]
    # add x18,x18,x4
    # fsub    v31.4s, v27.4s,  v25.4s
    # str q29, [x18]
    vfmacc.vf	v29, ft0, v27
    vfmacc.vf	v28, ft1, v26
    add		s8, s8,  a4
    vfsub.vv	v31, v27, v25
    vsw.v		v29, (s8)

    # fadd	v28.4s, v28.4s,  v31.4s
    vfadd.vv	v28, v28, v31
    # movi	d25, 0x0 
    # movi	d29, 0x0
    vmv.v.i     v25, 0
    vmv.v.i     v29, 0
    # add x18,x18,x4
    add		s8, s8,  a4

    # fmla	v25.4s, v5.4s,  v30.s[2]
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # fmla	v31.4s, v24.4s,  v30.s[1]
    # fmls	v25.4s, v11.4s,  v30.s[3]
    # fmls	v29.4s, v26.4s,  v30.s[3]
    # str q28, [x18]
    vfmacc.vf	v25, ft2, v5
    vfmacc.vf	v29, ft2, v24
    vfmacc.vf	v31, ft1, v24
    vfnmsac.vf	v25, ft3, v11
    vfnmsac.vf	v29, ft3, v26
    vsw.v		v28, (s8)

    # fmls	v31.4s, v26.4s,  v30.s[1]
    # fmla	v25.4s, v23.4s,  v30.s[0]
    # add x18,x18,x4
    # fmla	v29.4s, v25.4s,  v30.s[0]
    # str q31, [x18]
    # add x18,x18,x4
    # str q29, [x18]
    vfnmsac.vf	v31, ft1, v26
    vfmacc.vf	v25, ft0, v23
    add		s8, s8,  a4
    vfmacc.vf	v29, ft0, v25
    vsw.v		v31, (s8)
    add		s8, s8,  a4
    vsw.v		v29, (s8)


line5://
    //v0 1 2 3 4 5 
    # ld4 {v0.4s, v1.4s, v2.4s,v3.4s}, [x11]
    # ldr	q31, [x11, 0x40]	
    li              t0, 4
    vlsw.v          v24, (s1), t0
    addi            s1, s1, 4
    vlsw.v          v1, (s1), t0
    addi            s1, s1, 4
    vlsw.v          v2, (s1), t0
    addi            s1, s1, 4
    vlsw.v          v3, (s1), t0
    addi            s1, s1, 0x30
    vlw.v           v31, (s1)
    addi            s1, s1, -0x40

    # ext v4.16b,v0.16b,v31.16b,#4
    vslidedown.vi   v25, v24, 12
    vslideup.vi     v26, v31, 4
    vlb.v           v0, (s0)
    vmerge.vvm      v4, v26, v25, v0
    # ext v31.16b,v31.16b,v31.16b,#
    vslidedown.vi   v25, v31, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v31, v26, v25, v0  
    # ext v5.16b,v1.16b,v31.16b,#4
    vslidedown.vi   v25, v1, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v5, v26, v25, v0

    # //v18 19 20 21 22 23
    # add x14,x14,x3     
    add		s4, s4, a3 
    # ld4 {v18.4s, v19.4s, v20.4s,v21.4s}, [x14]
    li              t0, 4
    vlsw.v          v18, (s4), t0
    addi            s4, s4, 4
    vlsw.v          v19, (s4), t0
    addi            s4, s4, 4
    vlsw.v          v20, (s4), t0
    addi            s4, s4, 4
    vlsw.v          v21, (s4), t0
    addi            s4, s4, 0x30
    # ldr	q31, [x14, 0x40]	
    vlw.v           v31, (s4)
    addi            s4, s4, -0x40
    # ext v22.16b,v18.16b,v31.16b,#4
    vslidedown.vi   v25, v18, 12
    vslideup.vi     v26, v31, 4
    vlb.v           v0, (s0)
    vmerge.vvm      v22, v26, v25, v0
    # ext v31.16b,v31.16b,v31.16b,#4
    vslidedown.vi   v25, v31, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v31, v26, v25, v0  
    # ext v23.16b,v19.16b,v31.16b,#4
    vslidedown.vi   v25, v19, 12
    vslideup.vi     v26, v31, 4
    vmerge.vvm      v23, v26, v25, v0

    # movi	d24, 0x0
	# movi	d25, 0x0
	# movi	d26, 0x0
	# movi	d27, 0x0
    vmv.v.i     v24, 0
    vmv.v.i     v25, 0
    vmv.v.i     v26, 0
    vmv.v.i     v27, 0

    # fmla	v24.4s, v1.4s,  v30.s[2]
    # fmla	v25.4s, v2.4s,  v30.s[2]
    # fmla	v26.4s, v3.4s,  v30.s[2]
    # fmla	v27.4s, v4.4s,  v30.s[2]
    vfmacc.vf	v24, ft2, v1
    vfmacc.vf	v25, ft2, v2
    vfmacc.vf	v26, ft2, v3
    vfmacc.vf	v27, ft2, v4

    # fmls	v24.4s, v13.4s,  v30.s[3]
    # fmls	v25.4s, v14.4s,  v30.s[3]
    vfnmsac.vf	v24, ft3, v13
    vfnmsac.vf	v25, ft3, v14
    
    # fmls	v26.4s, v15.4s,  v30.s[3]
    # fmls	v27.4s, v16.4s,  v30.s[3]
    vfnmsac.vf	v26, ft3, v15
    vfnmsac.vf	v27, ft3, v16
    
    # fmla	v24.4s, v19.4s,  v30.s[0]
    # fmla	v25.4s, v20.4s,  v30.s[0]
    # fmla	v26.4s, v21.4s,  v30.s[0]
    # fmla	v27.4s, v22.4s,  v30.s[0]
    vfmacc.vf	v24, ft0, v19
    vfmacc.vf	v25, ft0, v20
    vfmacc.vf	v26, ft0, v21
    vfmacc.vf	v27, ft0, v22

    # //v29 as inp0
    #  movi	d29, 0x0
    vmv.v.i     v29, 0
    # movi	d28, 0x0
	vmv.v.i     v28, 0
    
    # fmla	v29.4s, V0.4s,  v30.s[2]
    # fmls	v29.4s, v12.4s,  v30.s[3]
    # fmla	v29.4s, v18.4s,  v30.s[0]
    vfmacc.vf	v29, ft2, v0
    vfnmsac.vf	v29, ft3, v12
    vfmacc.vf	v29, ft0, v18
    
    # fmla	v28.4s, v29.4s,  v30.s[2]
    # fmls	v28.4s, v25.4s,  v30.s[3]
    # fmla	v28.4s, v27.4s,  v30.s[0]
    vfmacc.vf	v28, ft2, v29
    vfnmsac.vf	v28, ft3, v25
    vfmacc.vf	v28, ft0, v27
    # str q28, [x9]
    vsw.v     v28, (s9)
 
    # //28 29 28 29
    # movi	d28, 0x0
	vmv.v.i     v28, 0
	# movi	d29, 0x0
    vmv.v.i     v29, 0
    
    # fmls	v28.4s, v24.4s,  v30.s[2]
    # fmls	v28.4s, v25.4s,  v30.s[2]
    # fmla	v28.4s, v26.4s,  v30.s[0]
    # fmla	v28.4s, v27.4s,  v30.s[0]
    vfnmsac.vf	v28, ft2, v24
    vfnmsac.vf	v28, ft2, v25
    vfmacc.vf	v28, ft0, v26
    vfmacc.vf	v28, ft0, v27
    
    # add x18,x9,x4
    # str q28, [x18]
    add     s8, s9, a4
    vsw.v		v28, (s8)
    # fmla	v29.4s, v24.4s,  v30.s[2]
    # fmls	v29.4s, v25.4s,  v30.s[2]
    # fmls	v29.4s, v26.4s,  v30.s[0]
    # fmla	v29.4s, v27.4s,  v30.s[0]
    vfmacc.vf	v29, ft2, v24
    vfnmsac.vf	v29, ft2, v25
    vfnmsac.vf	v29, ft0, v26
    vfmacc.vf	v29, ft0, v27
    # add x18,x18,x4
    # str q29, [x18]
    add		s8, s8,  a4
    vsw.v		v29, (s8)

	# movi	d28, 0x0
	vmv.v.i     v28, 0
	# movi	d29, 0x0
    vmv.v.i     v29, 0
    
    # fmls	v28.4s, v24.4s,  v30.s[1]
    # fmls	v28.4s, v25.4s,  v30.s[0]
    # fmla	v28.4s, v26.4s,  v30.s[1]
    # fmla	v28.4s, v27.4s,  v30.s[0]
    vfnmsac.vf	v28, ft1, v24
    vfnmsac.vf	v28, ft0, v25
    vfmacc.vf	v28, ft1, v26
    vfmacc.vf	v28, ft0, v27
    # add x18,x18,x4
    # str q28, [x18]
    add		s8, s8,  a4
    vsw.v		v28, (s8)

    # fmla	v29.4s, v24.4s,  v30.s[1]
    # fmls	v29.4s, v25.4s,  v30.s[0]
    # fmls	v29.4s, v26.4s,  v30.s[1]
    # fmla	v29.4s, v27.4s,  v30.s[0]
    vfmacc.vf	v29, ft1, v24
    vfnmsac.vf	v29, ft0, v25
    vfnmsac.vf	v29, ft1, v26
    vfmacc.vf	v29, ft0, v27
    # add x18,x18,x4
    # str q29, [x18]
    add		    s8, s8,  a4
    vsw.v		v29, (s8)

    #     //v25 as inp0
    #  movi	d29, 0x0
    vmv.v.i     v29, 0 
    # fmla	v29.4s, v5.4s,  v30.s[2]
    # fmls	v29.4s, v17.4s,  v30.s[3]
    # fmla	v29.4s, v23.4s,  v30.s[0]
    vfmacc.vf	v29, ft2, v5
    vfnmsac.vf	v29, ft3, v17
    vfmacc.vf	v29, ft0, v23
    # movi	d28, 0x0
    vmv.v.i     v28, 0
    # fmla	v28.4s, v24.4s,  v30.s[2]
    # fmls	v28.4s, v26.4s,  v30.s[3]
    # fmla	v28.4s, v29.4s,  v30.s[0]
    vfmacc.vf	v28, ft2, v24
    vfnmsac.vf	v28, ft3, v26
    vfmacc.vf	v28, ft0, v29
    # add x18,x18,x4
    # str q28, [x18]
    add		    s8, s8,  a4
    vsw.v		v28, (s8)
//
return:
	# ldp	d8,  d9,  [sp]
	# ldp	d10, d11, [sp, 0x10]
	# ldp	d12, d13, [sp, 0x20]
	# ldp	d14, d15, [sp, 0x30]
	# add	sp, sp, 0x40
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


//stp   q24,q25, [x1]
//stp	 q26,q27, [x1, 0x20]
//stp	 q28,q29, [x1, 0x40]
