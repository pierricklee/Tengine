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
 * Copyright (c) 2021, Open AI Lab
 * Author: hhchen@openailab.com
 */

#include "timvx_executor.hpp"

extern "C"
{
#include "tengine_op.h"
#include "pooling_param.h"
}


bool VXEngine::AddPoolingNode(struct ir_node* ir_node)
{
    TLOG_INFO("Tengine TIM-VX: Support OP(%d) OP_POOL.\n", ir_node->idx);
    struct ir_graph* ir_graph = ir_node->graph;

    struct pool_param* param = (struct pool_param*)ir_node->op.param_mem;

    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    tim::vx::PoolType pooltype;
    if (param->pool_method == 0)
    {
        pooltype = tim::vx::PoolType::MAX;
    }
    else
    {
        pooltype = tim::vx::PoolType::AVG;
    }

    tim::vx::PadType padtype;

    int h = input_tensor->dims[2];
    int out_h = (h - 1) / param->stride_h + 1;
    int total_len_h = (out_h - 1) * param->stride_h + param->kernel_h;
    int pad_num_h = total_len_h - h;
    int pad_h0 = 0;
    if (param->pad_h0 == pad_num_h / 2 && param->pad_h1 == pad_num_h - pad_num_h / 2)
    {
        pad_h0 = -1;
    }

    int w = input_tensor->dims[3];
    int out_w = (w - 1) / param->stride_w + 1;
    int total_len_w = (out_w - 1) * param->stride_w + param->kernel_w;
    int pad_num_w = total_len_w - w;
    int pad_w0 = 0;
    if (param->pad_w0 == pad_num_w / 2 && param->pad_w1 == pad_num_w - pad_num_w / 2)
    {
        pad_w0 = -1;
    }

    if (pad_h0 == -1 && pad_w0 == -1)
    {
        TLOG_INFO("Log:tim::vx::PadType::SAME\n");
        padtype = tim::vx::PadType::SAME;
    }
    else if(param->pad_h0 == 0 && param->pad_w0 == 0)
    {
        TLOG_INFO("Log:tim::vx::PadType::VALID\n");
        padtype = tim::vx::PadType::VALID;
    }

    auto pool = graph->CreateOperation<tim::vx::ops::Pool2d>(
        pooltype, padtype,
        std::array<uint32_t, 2>({ (unsigned int)param->kernel_h, (unsigned int)param->kernel_w}),
           std::array<uint32_t, 2>({(unsigned int)param->stride_h, (unsigned int)param->stride_w}));

    (*pool).BindInputs({ this->vx_tensor_map[input_tensor->idx] })
           .BindOutputs({ this->vx_tensor_map[output_tensor->idx] });

    return true;
}
