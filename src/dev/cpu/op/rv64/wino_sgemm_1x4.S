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
// a0        output start address
// a1        input start address
// a2        kernel start address
// a3        cin

// x9 ~ x10  temp loop counter

    .section .text,"ax"
    .align 5

    .type wino_sgemm_1x4 STT_FUNC
    .global wino_sgemm_1x4
    .hidden wino_sgemm_1x4
    
wino_sgemm_1x4:
    // bring some code ahead to reduce dependency
    # prfm    pldl1keep, [x1]
    # cmp    x3, 0x4
    vsetvli	        x0, a0, e32, m1, d1

none_biases:
    # movi    d16, 0x0
    vmv.v.x         v16, x0

start:
    # and    x10,x3, 0x3
    addi            t3, a3, 0x3
    # b.lt    loop4_end
    li              t0, 4
	blt             a3, t0, loop4_end
    # lsr    x9, x3, 0x2
    srli			t2, a3, 0x2

loop4:  
    # ldr    q0, [x1]
	vlw.v           v0, (a1)
    addi            a1, a1, 16
    # ldp    q4, q5, [x2]
    # fmla    v16.4s, v4.4s,  v0.s[0]
    # ldp    q6, q7, [x2, 0x20]
    vlw.v           v4, (a2)
    addi            a2, a2, 16
    vlw.v           v5, (a2)
    addi            a2, a2, 16
    vlw.v           v6, (a2)
    addi            a2, a2, 16
    vlw.v           v7, (a2)
    addi            a2, a2, 16
    # fmla    v16.4s, v5.4s,  v0.s[1]
    # prfm    pldl1keep, [x2, 0x50]
    # fmla    v16.4s, v6.4s,  v0.s[2]
    # prfm    pldl1keep, [x1, 0x20]
    # fmla    v16.4s, v7.4s,  v0.s[3]
    vrgather.vi     v20, v0, 0
    vrgather.vi     v21, v0, 1
    vrgather.vi     v22, v0, 2
    vrgather.vi     v23, v0, 3
    vfmacc.vv       v16, v20, v4
    vfmacc.vv       v17, v21, v4
    vfmacc.vv       v18, v22, v4
    vfmacc.vv       v19, v23, v4

    # subs    x9, x9, 0x1
    addi            t2, t2, -1
    # add    x1, x1, 0x10
    # add    x2, x2, 0x40
    # b.ne    loop4
    bnez            t2, loop4

loop4_end:
    # cbz    x10, save_result
    beqz		t3, save_result

loop1:
    # ldr    s0,[x1], 0x4
    # ldr    q4,[x2], 0x10
    vlw.v           v0, (a1)
    flw              f6, 0(a1) # 
    addi            a1, a1, 4
    vlw.v           v4, (a2)
    addi            a2, a2, 16
    # fmla    v16.4s, v4.4s,  v0.s[0]
    vrgather.vi     v20, v0, 0
    vfmacc.vv       v16, v20, v4
    # //add     x2, x2, 0x10
    # //add     x1, x1, 0x4
    # subs    x10, x10 ,0x1
    addi            t3, t3, -1
    # b.ne    loop1
    bnez            t3, loop1
    
save_result:
    # str  q16, [x0]
    vsw.v           v16, (a0)



    ret
        .end

