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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: zhli@openailab.com
 */

#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include "test_op.h"
#include "tengine_c_api.h"
#include "tengine_c_api_ex.h"
#include "common.h"

/* 定义默认函数运行参数 */
#define DEFAULT_LOOP_COUNT      1
#define DEFAULT_THREAD_COUNT    1

/* 定义默认CPU为 所有CPU（优先big核）*/
#define DEFAULT_CLUSTER         TENGINE_CLUSTER_ALL

/* Kernel大小，如为3默认优先走winograd */
#define kernel_size             3

/* 是否进入DepthWidth，非0则进入 */
#define DW_TEST                 0

/* 定义默认Conv参数 */
#define INPUT_CHANNEL           128
#define OUT_CHANNEL             32
#define IMAGE_SIZE              56
#define STRIDE					1
#define PAD						1

/* 是否加入 warm_up，是否进入Debug*/
#define WARM_UP                 0
#define DEBUG                   0


int loop_counts = DEFAULT_LOOP_COUNT;
int num_threads = DEFAULT_THREAD_COUNT;
int power       = DEFAULT_CLUSTER;

int allocated_num = 0;
void** record_ptr = NULL;

void record_allocated_buf(void* buf)
{
    allocated_num++;
    record_ptr = realloc(record_ptr, sizeof(void*) * allocated_num);
    record_ptr[allocated_num - 1] = buf;
}


void free_allocated_buf(void)
{
    for(int i = 0; i < allocated_num; i++)
        free(record_ptr[i]);

    if(record_ptr)
        free(record_ptr);
}


void init_buffer(void* buf, int elem_num, int elem_size, int val)
{
    for(int i = 0; i < elem_num; i++)
    {
        float val0;
        float* fp;
        int16_t* i16;
        char* c;

        if(val > 0)
            val0 = val;
        else
            val0 = i;

        switch(elem_size)
        {
            case 4:
                fp = ( float* )buf;
                fp[i] = val0;
                break;
            case 2:
                i16 = ( int16_t* )buf;
                i16[i] = val0;
                break;
            case 1:
                c = ( char* )buf;
                c[i] = val0;
                break;
        }
    }
}

/* 填入Conv数据，分配内存 */
void fill_conv_node(node_t node)
{
    tensor_t filter = get_node_input_tensor(node, 1);
    int dims[4];

    get_tensor_shape(filter, dims, 4);

    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];
    int elem_size = 4;

    // weight data init
    void* filter_buf = malloc(elem_num * elem_size);

    init_buffer(filter_buf, elem_num, elem_size, 1);

    set_tensor_buffer(filter, filter_buf, elem_num * elem_size);

    record_allocated_buf(filter_buf);

    release_graph_tensor(filter);

    tensor_t bias = get_node_input_tensor(node, 2);

    if(bias == NULL)
        return;

    get_tensor_shape(bias, dims, 1);

    elem_num = dims[0];

    // bias data init
    void* bias_buf = malloc(elem_num * elem_size);

    init_buffer(bias_buf, elem_num, elem_size, 1);

    set_tensor_buffer(bias, bias_buf, elem_num * elem_size);

    record_allocated_buf(bias_buf);

    release_graph_tensor(bias);
}

/* 预留deconv节点 */
void fill_deconv_node(node_t node)
{
    tensor_t filter = get_node_input_tensor(node, 1);
    int dims[4];

    get_tensor_shape(filter, dims, 4);

    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];
    int elem_size = 4;

    void* filter_buf = malloc(elem_num * elem_size);

    init_buffer(filter_buf, elem_num, elem_size, 1);

    set_tensor_buffer(filter, filter_buf, elem_num * elem_size);

    record_allocated_buf(filter_buf);

    release_graph_tensor(filter);

    tensor_t bias = get_node_input_tensor(node, 2);

    if(bias == NULL)
        return;

    get_tensor_shape(bias, dims, 1);

    elem_num = dims[0];

    void* bias_buf = malloc(elem_num * elem_size);

    init_buffer(bias_buf, elem_num, elem_size, 1);

    set_tensor_buffer(bias, bias_buf, elem_num * elem_size);

    record_allocated_buf(bias_buf);

    release_graph_tensor(bias);
}

/* 填入Conv参数，创建graph */
void fill_graph_param(graph_t graph)
{
    int node_num = get_graph_node_num(graph);

    for(int i = 0; i < node_num; i++)
    {
        node_t node = get_graph_node_by_idx(graph, i);

        const char* node_op = get_node_op(node);

        if(!strcmp(node_op, "Convolution"))
        {
            fill_conv_node(node);
        }else if(!strcmp(node_op, "DeConv"))
        {
            fill_deconv_node(node);
        }

        release_graph_node(node);
    }
}

/* 创建Conv节点 */
int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad, int in_c, int out_c, int group)
{
    char* weight_name = malloc(strlen(node_name) + 16);
    sprintf(weight_name, "%s/weight", node_name);

    char* bias_name = malloc(strlen(node_name) + 16);
    sprintf(bias_name, "%s/bias", node_name);

    /* weight */
    node_t w_node = create_graph_node(graph, weight_name, "Const");
    if (NULL == w_node)
    {
        free(bias_name);
        free(weight_name);

        fprintf(stderr, "create test weight node failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    tensor_t w_tensor = create_graph_tensor(graph, weight_name, TENGINE_DT_FP32);
    if (NULL == w_tensor)
    {
        free(bias_name);
        free(weight_name);

        fprintf(stderr, "create graph weight tensor failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    int w_dims[] = {out_c, in_c, k_size, k_size};
    set_tensor_shape(w_tensor, w_dims, 4);

    /* bias */
    node_t b_node = create_graph_node(graph, bias_name, "Const");
    if (NULL == b_node)
    {
        free(bias_name);
        free(weight_name);

        fprintf(stderr, "create test bias node failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    tensor_t b_tensor = create_graph_tensor(graph, bias_name, TENGINE_DT_FP32);
    if (NULL == b_tensor)
    {
        free(bias_name);
        free(weight_name);

        fprintf(stderr, "create graph bias tensor failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);

    int b_dims[] = {out_c};
    set_tensor_shape(b_tensor, b_dims, 1);

    /* conv */
    node_t conv_node = create_graph_node(graph, node_name, "Convolution");
    if (NULL == conv_node)
    {
        free(bias_name);
        free(weight_name);

        fprintf(stderr, "create test conv node failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    if (NULL == input_tensor)
    {
        free(bias_name);
        free(weight_name);

        fprintf(stderr, "get graph input tensor failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    set_node_input_tensor(conv_node, 2, b_tensor);
    set_node_input_tensor(conv_node, 1, w_tensor);
    set_node_input_tensor(conv_node, 0, input_tensor);

    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    if (NULL == output_tensor)
    {
        free(bias_name);
        free(weight_name);

        fprintf(stderr, "create graph output tensor failed. ERRNO: %d.\n", get_tengine_errno());
        return -1;
    }

    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    free(bias_name);
    free(weight_name);

    /* attr */
    set_node_attr_int(conv_node, "kernel_h", &k_size);
    set_node_attr_int(conv_node, "kernel_w", &k_size);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h0", &pad);
    set_node_attr_int(conv_node, "pad_h1", &pad);
    set_node_attr_int(conv_node, "pad_w0", &pad);
    set_node_attr_int(conv_node, "pad_w1", &pad);
    set_node_attr_int(conv_node, "output_channel", &out_c);
    set_node_attr_int(conv_node, "input_channel", &in_c);
    set_node_attr_int(conv_node, "group", &group);

    release_graph_node(conv_node);

    return 0;
}

graph_t create_test_graph(int image_c, int image_h, int image_w, int out_c)
{
    graph_t graph = create_graph(NULL, NULL, NULL);
    if(NULL == graph)
    {
        fprintf(stderr, "get graph failed. ERRNO: %d.\n", get_tengine_errno());
        return NULL;
    }

    const char* input_name = "data";
    const char* conv_name = "conv";
    // const char* deconv_name = "deconv";

    if(create_node(graph, input_name, 1, image_c, image_h, image_w, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW) < 0)
    {
        fprintf(stderr, "create input failed\n");
        return NULL;
    }

    if(!DW_TEST)
    {
        // 创建非DepthWidth节点
        if(create_conv_node(graph, conv_name, input_name, kernel_size, STRIDE, PAD, image_c, out_c, 1) < 0)
        {
            fprintf(stderr, "create test conv node failed. ERRNO: %d.\n", get_tengine_errno());
            return NULL;
        }
        fprintf(stderr, "kernel_size: %d.\n", kernel_size);
    }

    else
    {
        // 按DepthWidth创建Node，即 group = image_c
        if(create_conv_node(graph, conv_name, input_name, kernel_size, STRIDE, PAD, image_c, out_c, image_c) < 0)
        {
            fprintf(stderr, "create test conv node failed. ERRNO: %d.\n", get_tengine_errno());
            return NULL;
        }
    }

    const char* inputs[] = {input_name};
    const char* outputs[] = {conv_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed. ERRNO: %d.\n", get_tengine_errno());
        return NULL;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed: ERRNO: %d\n", get_tengine_errno());
        return NULL;
    }

    return graph;
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n  [-r loop_count] [-g image_c,image_h,image_w,image_n] [-t thread_count]\n [-p cpu affinity, 0:auto, 1:big, 2:middle, 3:little]\n");
}


int main(int argc, char* argv[])
{
    struct options opt;

    float img_hw[4] = {0.f};
    int image_c = INPUT_CHANNEL;
    int image_h = IMAGE_SIZE;
    int image_w = IMAGE_SIZE;
    int out_c = OUT_CHANNEL;
    int res;

    while((res = getopt(argc,argv,"r:t:p:g:h")) !=-1 )
    {
        switch (res)
        {
            case 'r':
                loop_counts = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'p':
                power = atoi(optarg);
                break;
            case 'g':
                split(img_hw, optarg, ",");
                image_c = ( int )img_hw[0];
                image_h = ( int )img_hw[1];
                image_w = ( int )img_hw[2];
                out_c   = ( int )img_hw[3];
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    opt.num_thread = num_threads;
    opt.cluster = power;
    opt.precision = TENGINE_MODE_FP32;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Engine init failed. ERRNO: %d.", get_tengine_errno());

    graph_t graph = create_test_graph(image_c, image_h, image_w, out_c);

    if(NULL == graph)
        return 1;

    fill_graph_param(graph);

    //* fill input
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int dims[4];
    int dim_num = get_tensor_shape(input_tensor, dims, 4);

    int elem_num = 1;
    int elem_size = 4;

    for(int i = 0; i < dim_num; i++)
        elem_num *= dims[i];

    void* input_buf = malloc(elem_num * elem_size);

    init_buffer(input_buf, elem_num, elem_size, 1);
    record_allocated_buf(input_buf);

    set_tensor_buffer(input_tensor, input_buf, elem_num * elem_size);
    release_graph_tensor(input_tensor);

    // multithread pre-run
    // if (prerun_graph_multithread(graph, opt) < 0)
    // {
    //     fprintf(stderr, "Prerun multithread graph failed.\n");
    //     return -1;
    // }

    if(prerun_graph(graph) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }
    
    if(DEBUG)
    { 
        dump_graph(graph);
    }
    
    /* 通过‘宏’控制，warm up框架 */
    if(WARM_UP)
    {   
        for(int t=0;t<5;t++)
        {
            if (run_graph(graph, 1) < 0)
            {
                fprintf(stderr, "Run graph failed\n");
                return -1;
            }
        }
    }

    // run graph
    double min_time = __DBL_MAX__;  
    double max_time = -__DBL_MAX__;
    double total_time = 0.;
    for (int i = 0; i < loop_counts; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }

	/* 通过‘宏’控制，打印 tensor到shell */
    if(DEBUG)
    {
        //dump input node
        int input_node_count = get_graph_input_node_number(graph);
        for(int i = 0; i < input_node_count; i++)
        {
            node_t input = get_graph_input_node(graph, i);
            dump_node_output(input, 0);
        }

        // dump output node
        int output_node_count = get_graph_output_node_number(graph);
        for(int i = 0; i < output_node_count; i++)
        {
            node_t output = get_graph_output_node(graph, i);
            dump_node_output(output, 0);
        }
    }

	/* 打印关键参数 */
    fprintf(stderr, " image_c = %d, image_h = %d, image_w = %d, out_c = %d .\n", image_c, image_h, image_w, out_c);
    fprintf(stderr, " loop_count = %d, num_threads = %d, power =  %d\n", loop_counts, num_threads, power);
    fprintf(stderr, " min = %7.2f ms, max = %7.2f ms, avg = %7.2f ms\n\n", min_time, max_time, total_time / loop_counts);

    // exit
    free_allocated_buf();

    test_graph_release(graph);

    return 0;
}
