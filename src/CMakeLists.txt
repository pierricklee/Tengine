# add include path
list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op)
list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm)
if(TENGINE_USE_TINY_SERIALIZER)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tiny)
endif()


# add lib files, for utility
if (TENGINE_BUILD_CPP_API)
    file(GLOB_RECURSE TENGINE_LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c ${CMAKE_CURRENT_SOURCE_DIR}/lib/tengine_cpp_api.cpp)
else()
    file(GLOB_RECURSE TENGINE_LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c)
endif()

# add front end files
file(GLOB_RECURSE TENGINE_FRONT_END_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op/*.c)

# add serializer files
file(GLOB_RECURSE TENGINE_SERIALIZER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm/*.c ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tm/op/*.c)
if(TENGINE_USE_TINY_SERIALIZER)
    file(GLOB_RECURSE TENGINE_TINY_SERIALIZER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tiny/*.c)
endif()

# add core srcs
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_allocator.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_device.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_module.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_node_ops.c")
list(APPEND TENGINE_BACKEND_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/cpu_probe.c")

# arm
if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
    if (TENGINE_ARCH_ARM_82)
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*hcl.c"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*fp16_arm82.c")
    else()
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*hcl.c"
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c")
    endif()

    if (TENGINE_TARGET_PROCESSOR_32Bit)
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch32/*.S")
    else()
        if (TENGINE_ARCH_ARM_82)
            list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -march=armv8.2-a+fp16)
            list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -march=armv8.2-a+fp16)
            list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -march=armv8.2-a+fp16)
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch64/*.S"
                                                            "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/armv8_2/*.S")
        else()
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/aarch64/*.S")
        endif()
    endif()
endif()

# x86
if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    if (TENGINE_ARCH_X86_AVX)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE /arch:AVX2)
            list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE /arch:AVX2)
            list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE /arch:AVX2)
        else()
            list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -mfma)
            list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -mf16c)

            list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -mfma)
            list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -mf16c)
        endif()
    endif()
endif()
file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/rv64/*.S")
# RV64
if (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
    file(GLOB_RECURSE TENGINE_BACKEND_HCL_ASM_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/rv64/*.S")
endif()

# add operator files
if (TENGINE_DYNAMIC_COMPILE AND EXISTS "${CMAKE_SOURCE_DIR}/cmake/operators.cmake")
    # macro for adding include op dir
    macro (ADD_OP name)
        # add reference operator files 
        set (REF_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*ref.c")
        list (APPEND REF_SOURCE ${REF_OP_PATH})
        
        # add hcl operator files
        # arm
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
            if (TENGINE_ARCH_ARM_82)
                set (ARM_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*arm.c")
                set (ARM82_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*fp16_arm82.c")
            else()
                set (ARM_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*arm.c")
            endif()
            list (APPEND HCL_SOURCE ${ARM_OP_PATH})
            list (APPEND HCL_SOURCE ${ARM82_OP_PATH})
        endif()

        # X86
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
            set (X86_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*x86.c")
            list (APPEND HCL_SOURCE ${X86_OP_PATH})
        endif()

        # MIPS64
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
            set (MIPS_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*mips.c")
            list (APPEND HCL_SOURCE ${MIPS_OP_PATH})
        endif()

        # RV64
        if (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
            set (RV64_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/${name}/*rv64.c")
            list (APPEND HCL_SOURCE ${RV64_OP_PATH})
        endif()
    endmacro()
    include(${CMAKE_SOURCE_DIR}/cmake/operators.cmake)

    # add reference operator files
    file(GLOB_RECURSE TENGINE_BACKEND_REF_OPS ${REF_SOURCE})

    # add hcl operator files
    # arm or x86 or mips64 or rv64
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM" OR ${TENGINE_TARGET_PROCESSOR} MATCHES "X86" OR ${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS" OR ${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   ${HCL_SOURCE})
    endif()
else()
    # add reference operator files
    file(GLOB_RECURSE TENGINE_BACKEND_REF_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*ref.c")

    # add hcl operator files
    # arm
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
        if (TENGINE_ARCH_ARM_82)
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c"
                                                        "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*fp16_arm82.c")
        else()
            file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*arm.c")
        endif()                                                    
    endif()

    # X86
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*x86.c")
    endif()

    # MIPS64
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*mips.c")
    endif()

    # RV64
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
        file(GLOB_RECURSE TENGINE_BACKEND_HCL_OPS   "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*rv64.c")
    endif()
endif()

# add cmsis operator files
file(GLOB_RECURSE TENGINE_BACKEND_CMSIS_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cpu/op/*cmsis.c")
# add ACL GPU support
if (TENGINE_ENABLE_ACL)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dev/acl)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/acl/include)
    list(APPEND TENGINE_ACL_LIB_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/acl/lib)

    #link_directories(${TENGINE_ACL_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_ACL_LIB_DIRS' is ${TENGINE_ACL_LIB_DIRS}.")
    endif()

    file(GLOB_RECURSE TENGINE_BACKEND_ACL_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/acl/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/dev/acl/*.c")
endif ()


# add vulkan support
if (TENGINE_ENABLE_VULKAN)
    add_subdirectory(dev/vulkan)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dev/vulkan)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/dev/vulkan)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${Vulkan_INCLUDE_DIR})
    list(APPEND TENGINE_VULKAN_LIB_DIRS $ENV{VULKAN_SDK}/lib)
    
    link_directories(${TENGINE_VULKAN_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_VULKAN_LIB_DIRS' is ${TENGINE_VULKAN_LIB_DIRS}.")
    endif()
endif()

# add opencl support
if (TENGINE_ENABLE_OPENCL)
    message(${OPENCL_INCLUDE_DIRS})
    message(${OPENCL_LIBRARY})
    include_directories(${OPENCL_INCLUDE_DIRS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dev/opencl/cl_kernel)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dev/opencl)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/dev/opencl/op)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE ${OPENCL_INCLUDE_DIRS})

    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${OPENCL_LIBRARY})

    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'OPENCL_LIBRARY' is ${OPENCL_LIBRARY}.")
    endif()

    file(GLOB_RECURSE TENGINE_BACKEND_OPENCL_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/opencl/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/dev/opencl/*.cpp")
    file(GLOB_RECURSE TENGINE_BACKEND_OPENCL_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/opencl/op/*.cpp")
endif()

# add nVIDIA TensorRT support
if (TENGINE_ENABLE_TENSORRT)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/tensorrt)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/tensorrt/op)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${TENSORRT_INCLUDE_DIR})

    list(APPEND TENGINE_TENSORRT_LIB_DIRS ${TENSORRT_LIBRARY_DIR})

    if(DEFINED CUDA_INCLUDE_DIR)
        list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CUDA_INCLUDE_DIR})
    endif()

    if(DEFINED CUDA_LIBARAY_DIR)
        list(APPEND TENGINE_TENSORRT_LIB_DIRS  ${CUDA_LIBARAY_DIR})
    endif()

    message(STATUS "TENGINE_INCLUDE_DIRS_PRIVATE=${TENGINE_INCLUDE_DIRS_PRIVATE}")

    link_directories(${TENGINE_TENSORRT_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_TENSORRT_LIB_DIRS' is ${TENGINE_TENSORRT_LIB_DIRS}.")
    endif()

    file(GLOB_RECURSE TENGINE_BACKEND_TENSORRT_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/tensorrt/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/dev/tensorrt/*.cpp")
    file(GLOB_RECURSE TENGINE_BACKEND_TENSORRT_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/tensorrt/op/*.cpp")
endif ()

if (TENGINE_ENABLE_TIM_VX)
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
        set(TIM_VX_ARCH "aarch64")
    elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
        set(TIM_VX_ARCH "x86_64")
    else()
        message(FATAL_ERROR "Tengine: Unsupported OS:${TENGINE_TARGET_PROCESSOR}")
    endif()

    if (TENGINE_ENABLE_TIM_VX_INTEGRATION)
        set(VSI_TIM_NAME "tim_vx_internal")
        set(VSI_TIM_VX_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/src/tim/vx")

        aux_source_directory(${VSI_TIM_VX_BASE} VSI_TIM_VX_SRC)
        aux_source_directory(${VSI_TIM_VX_BASE}/ops VSI_TIM_OPS_SRC)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src VSI_TIM_INTERNAL_SRC)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/kernel VSI_TIM_INTERNAL_KERNEL)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/kernel/cl VSI_TIM_INTERNAL_KERNEL_CL)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/kernel/cpu VSI_TIM_INTERNAL_KERNEL_CPU)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/kernel/evis VSI_TIM_INTERNAL_KERNEL_EVIS)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/kernel/vx VSI_TIM_INTERNAL_KERNEL_VX)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/ops VSI_TIM_INTERNAL_OPS)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/client VSI_TIM_INTERNAL_CLIENT)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/libnnext VSI_TIM_INTERNAL_LIBNNEXT)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/libnnext/ops/kernel VSI_TIM_INTERNAL_LIBNNEXT_OPS_KERNEL)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/quantization VSI_TIM_INTERNAL_QUANTIZATION)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/custom/ops VSI_TIM_INTERNAL_CUSTOM_OPS)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/custom/ops/kernel VSI_TIM_INTERNAL_CUSTOM_OPS_KERNEL)
        aux_source_directory(${VSI_TIM_VX_BASE}/internal/src/utils VSI_TIM_INTERNAL_UTILS)

        list(APPEND VSI_TIM_VX_ALL_SRC
            ${VSI_TIM_VX_SRC}
            ${VSI_TIM_OPS_SRC}
            ${VSI_TIM_INTERNAL_SRC}
            ${VSI_TIM_INTERNAL_KERNEL}
            ${VSI_TIM_INTERNAL_KERNEL_CL}
            ${VSI_TIM_INTERNAL_KERNEL_CPU}
            ${VSI_TIM_INTERNAL_KERNEL_EVIS}
            ${VSI_TIM_INTERNAL_KERNEL_VX}
            ${VSI_TIM_INTERNAL_OPS}
            ${VSI_TIM_INTERNAL_CLIENT}
            ${VSI_TIM_INTERNAL_LIBNNEXT}
            ${VSI_TIM_INTERNAL_LIBNNEXT_OPS_KERNEL}
            ${VSI_TIM_INTERNAL_QUANTIZATION}
            ${VSI_TIM_INTERNAL_CUSTOM_OPS}
            ${VSI_TIM_INTERNAL_CUSTOM_OPS_KERNEL}
            ${VSI_TIM_INTERNAL_UTILS}
            )

        #message("VSI_TIM_VX_ALL_SRC=${VSI_TIM_VX_ALL_SRC}")

        add_library(${VSI_TIM_NAME} STATIC ${VSI_TIM_VX_ALL_SRC})
        target_link_directories(${VSI_TIM_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/lib/${TIM_VX_ARCH})
        target_link_libraries(${VSI_TIM_NAME} PRIVATE CLC GAL OpenVX OpenVXU VSC ArchModelSw NNArchPerf)
        target_include_directories(${VSI_TIM_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/include)
        target_include_directories(${VSI_TIM_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/include)
        target_include_directories(${VSI_TIM_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/include/tim/vx)
        target_include_directories(${VSI_TIM_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/src/tim/vx)
        target_include_directories(${VSI_TIM_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/src/tim/vx/internal/include)
        set_target_properties(${VSI_TIM_NAME} PROPERTIES CXX_STANDARD_REQUIRED 14)
        set_target_properties(${VSI_TIM_NAME} PROPERTIES CXX_STANDARD 14)
        set(VSI_TIM_OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
        target_compile_definitions(${VSI_TIM_NAME} PRIVATE "-DOVXLIB_API=${VSI_TIM_OVXLIB_API_ATTR}")
        target_compile_options(${VSI_TIM_NAME} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fPIC>)
        target_compile_options(${VSI_TIM_NAME} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-O0>)
        target_compile_options(${VSI_TIM_NAME} PRIVATE $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-g>)
    endif()

    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/op)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/include)

    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/include)

    list(APPEND TENGINE_TIM_VX_LIB_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/tim-vx/lib/${TIM_VX_ARCH})

    file(GLOB TENGINE_BACKEND_TIM_VX_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/*.cc")
    file(GLOB TENGINE_BACKEND_TIM_VX_OPS  "${CMAKE_CURRENT_SOURCE_DIR}/dev/tim-vx/op/*.cc")
endif ()


# add nVIDIA cudabackend support
if (TENGINE_ENABLE_CUDABACKEND)
    enable_language(CUDA)

    include(${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda/cmake/FindcuDNN.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda/cmake/FindcuBLAS.cmake)

    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda/op)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CUDABACKEND_INCLUDE_DIR})
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CUDA_INCLUDE_DIRS})
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CUDNN_INCLUDE_DIR})
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CUBLAS_INCLUDE_DIRS})

    list(APPEND TENGINE_CUDABACKEND_LIB_DIRS ${CUDA_LIBRARIES})
    list(APPEND TENGINE_CUDABACKEND_LIB_DIRS ${CUDNN_LIBRARY})
    list(APPEND TENGINE_CUDABACKEND_LIB_DIRS ${CUBLAS_LIBRARIES})

    message(STATUS "TENGINE_INCLUDE_DIRS_PRIVATE=${TENGINE_INCLUDE_DIRS_PRIVATE}")

    link_directories(${TENGINE_CUDABACKEND_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_CUDABACKEND_LIB_DIRS' is ${TENGINE_CUDABACKEND_LIB_DIRS}.")
    endif()

    file(GLOB_RECURSE TENGINE_BACKEND_CUDABACKEND_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda/*.cpp")
    file(GLOB_RECURSE TENGINE_BACKEND_CUDABACKEND_OPS "${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda/op/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/dev/cuda/op/*.cu")
endif ()


# add HiSilicon NNIE support
if (TENGINE_ENABLE_NNIE)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/include)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/core)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/common)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/driver)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/operator)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/serializer)
    list(APPEND TENGINE_INCLUDE_DIRS_PRIVATE  ${CMAKE_SOURCE_DIR}/3rdparty/nnie/include)

    list(APPEND TENGINE_NNIE_LIB_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/nnie/lib)

    link_directories(${TENGINE_NNIE_LIB_DIRS})
    if(TENGINE_VERBOSE)
        message (STATUS "TENGINE: 'TENGINE_NNIE_LIB_DIRS' is ${TENGINE_NNIE_LIB_DIRS}.")
    endif()

    file(GLOB_RECURSE TENGINE_BACKEND_NNIE_BASE "${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/dev/nnie/*.cpp")

endif ()


# add libraries path
list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_ACL_LIB_DIRS})
list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_VULKAN_LIB_DIRS})
list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_CUDABACKEND_LIB_DIRS})
set(TENGINE_COMMON_LIB_DIRS ${TENGINE_LINKING_LIBRARIES_PRIVATE} CACHE INTERNAL "" FORCE)
link_directories(${TENGINE_LINKING_LIBRARIES_PRIVATE})


# add standalone hcl lib
set (TENGINE_STANDALONE_HCL_LIB_NAME hclcpu)
if (TENGINE_STANDALONE_HCL)
    if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
        add_library (
                ${TENGINE_STANDALONE_HCL_LIB_NAME} SHARED
                ${TENGINE_BACKEND_HCL_OPS}
                ${TENGINE_BACKEND_HCL_ASM_OPS}
        )
    elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
        add_library (
                ${TENGINE_STANDALONE_HCL_LIB_NAME} SHARED
                ${TENGINE_BACKEND_HCL_OPS}
        )
    elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
        list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -mips64r2)
        list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -mabi=64)
        list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -mmsa)

        list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -mips64r2)
        list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -mabi=64)
        list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -mmsa)

        list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -mips64r2)
        list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -mabi=64)
        list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -mmsa)
        add_library (
                ${TENGINE_STANDALONE_HCL_LIB_NAME} SHARED
                ${TENGINE_BACKEND_HCL_OPS}
        )
    elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
        message (STATUS "TENGINE RV64 TENGINE_BACKEND_HCL_ASM_OPS.----------------------------------------------")
        list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -rv64imafdcvxtheadc)
        list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -mabi=lp64dv)
        list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -mtune=c910)

        list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -rv64imafdcvxtheadc)
        list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -mabi=lp64dv)
        list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -mtune=c910)

        list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -rv64imafdcvxtheadc)
        list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -mabi=lp64dv)
        list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -mtune=c910)
        add_library (
                ${TENGINE_STANDALONE_HCL_LIB_NAME} SHARED
                ${TENGINE_BACKEND_HCL_OPS}
                ${TENGINE_BACKEND_HCL_ASM_OPS}
        )
    endif()
    

    if (TENGINE_BACKEND_HCL_OPS)
        unset(TENGINE_BACKEND_HCL_OPS)
    endif()

    if (TENGINE_BACKEND_HCL_ASM_OPS)
        unset(TENGINE_BACKEND_HCL_ASM_OPS)
    endif()

    if (TENGINE_STANDALONE_HCL_AUTO_LOAD)
        LIST (APPEND TENGINE_COMPILE_DEFINITION_CXX_PRIVATE TENGINE_AUTO_LOAD_HCL)
        LIST (APPEND TENGINE_COMPILE_DEFINITION_CUDA_PRIVATE TENGINE_AUTO_LOAD_HCL)
    endif()
endif()


# add advanced hcl source files
if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS}
        ${TENGINE_BACKEND_HCL_ASM_OPS}
        ${TENGINE_BACKEND_ACL_BASE}
        ${TENGINE_BACKEND_VULKAN_BASE}
        ${TENGINE_BACKEND_VULKAN_OPS}
        ${TENGINE_BACKEND_OPENCL_BASE}
        ${TENGINE_BACKEND_OPENCL_OPS}
        ${TENGINE_BACKEND_TENSORRT_BASE}
        ${TENGINE_BACKEND_TENSORRT_OPS}
        ${TENGINE_BACKEND_CUDABACKEND_BASE}
        ${TENGINE_BACKEND_CUDABACKEND_OPS}
        ${TENGINE_BACKEND_TIM_VX_BASE}
        ${TENGINE_BACKEND_TIM_VX_OPS}
        ${TENGINE_BACKEND_NNIE_BASE})
elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS}
        ${TENGINE_BACKEND_VULKAN_BASE}
        ${TENGINE_BACKEND_VULKAN_OPS}
        ${TENGINE_BACKEND_OPENCL_BASE}
        ${TENGINE_BACKEND_OPENCL_OPS}
        ${TENGINE_BACKEND_TENSORRT_BASE}
        ${TENGINE_BACKEND_TENSORRT_OPS}
        ${TENGINE_BACKEND_CUDABACKEND_BASE}
        ${TENGINE_BACKEND_CUDABACKEND_OPS}
        ${TENGINE_BACKEND_TIM_VX_BASE}
        ${TENGINE_BACKEND_TIM_VX_OPS})
elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "MIPS")
    add_definitions(-mips64r2)
    add_definitions(-mabi=64)
    add_definitions(-mmsa)
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS})
elseif (${TENGINE_TARGET_PROCESSOR} MATCHES "lp64dv")
    add_definitions(-march=rv64imafdcvxtheadc)
    add_definitions(-mabi=lp64dv)
    add_definitions(-mtune=c910)
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS} ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS}
        ${TENGINE_BACKEND_HCL_OPS}
        ${TENGINE_BACKEND_HCL_ASM_OPS})
else()
    add_library(${CMAKE_PROJECT_NAME} SHARED
        ${TENGINE_LIB_SRCS}
        ${TENGINE_FRONT_END_SRCS}
        ${TENGINE_SERIALIZER_SRCS}
        ${TENGINE_TINY_SERIALIZER_SRCS}
        ${TENGINE_BACKEND_COMMON}
        ${TENGINE_BACKEND_REF_OPS})
endif()


if (NOT TENGINE_FORCE_SKIP_OPENMP)
    TENGINE_USE_LIB_OPENMP(${CMAKE_PROJECT_NAME})
endif()

# show linking libraries
if(TENGINE_VERBOSE)
    message (STATUS "TENGINE: 'TENGINE_LINKING_LIBRARIES_PRIVATE' is ${TENGINE_LINKING_LIBRARIES_PRIVATE}.")
endif()

if (NOT TENGINE_ENABLE_ALL_SYMBOL)
    if (NOT (CMAKE_SYSTEM_NAME MATCHES "Windows"))
        set_property (TARGET ${CMAKE_PROJECT_NAME} PROPERTY COMPILE_FLAGS "-fvisibility=hidden")
        set_property (TARGET ${CMAKE_PROJECT_NAME} PROPERTY LINK_FLAGS    "-fvisibility=hidden")
    endif()
endif()

if (ANDROID)
    # list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE android)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_LINKING_LIBS})
elseif(MSVC)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_LINKING_LIBS})
elseif(OHOS)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE dl)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE m)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_LINKING_LIBS})
else()
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE pthread)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE dl)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE m)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_LINKING_LIBS})
endif()

if (TENGINE_ENABLE_ACL)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE arm_compute)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE arm_compute_core)
endif ()

if (TENGINE_ENABLE_VULKAN)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${Vulkan_LIBRARY})
    message("===== vulkan library === ${Vulkan_LIBRARY}")
    add_dependencies(${CMAKE_PROJECT_NAME} generate-spirv)
endif ()

if (TENGINE_STANDALONE_HCL)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${TENGINE_STANDALONE_HCL_LIB_NAME})
endif()

if (TENGINE_ENABLE_TENSORRT)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE nvinfer)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE cudart)
endif()

if (TENGINE_ENABLE_TIM_VX)
    target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC ${TENGINE_TIM_VX_LIB_DIRS})
    if  (TENGINE_ENABLE_TIM_VX_INTEGRATION)
        list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE ${VSI_TIM_NAME})
        add_dependencies(${CMAKE_PROJECT_NAME} ${VSI_TIM_NAME})
        target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        target_link_options(${CMAKE_PROJECT_NAME} PRIVATE -Wl,--whole-archive -l ${VSI_TIM_NAME} -Wl,--no-whole-archive)
    else()
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE -Wl,--whole-archive -ltim-vx -Wl,--no-whole-archive)
    endif()
endif()

if (TENGINE_ENABLE_CUDABACKEND)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${TENGINE_COMPILE_DEFINITION_CUDA_PRIVATE}>)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE}>)
endif ()

if (TENGINE_ENABLE_NNIE)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE libnnie.a libVoiceEngine.a libupvqe.a libdnvqe.a libmpi.a libsecurec.a)
    list(APPEND TENGINE_LINKING_LIBRARIES_PRIVATE dl stdc++)
endif()

# add target compile include link
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:ASM>: ${TENGINE_COMPILE_DEFINITION_CXX_PRIVATE}>)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>: ${TENGINE_COMPILE_DEFINITION_CXX_PRIVATE}>)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: ${TENGINE_COMPILE_DEFINITION_CXX_PRIVATE}>)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:ASM>: ${TENGINE_COMPILE_OPTIONS_C_PRIVATE}>)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C>: ${TENGINE_COMPILE_OPTIONS_C_PRIVATE}>)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: ${TENGINE_COMPILE_OPTIONS_CXX_PRIVATE}>)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${TENGINE_INCLUDE_DIRS_PRIVATE})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${TENGINE_LINKING_LIBRARIES_PRIVATE})

if (ANDROID)
    target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC android)
endif()

if(TENGINE_COVERAGE)
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")    
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE -coverage -lgcov)
endif()

# add install files
install (TARGETS ${CMAKE_PROJECT_NAME} DESTINATION lib)
if (TENGINE_STANDALONE_HCL)
    install (TARGETS ${TENGINE_STANDALONE_HCL_LIB_NAME} DESTINATION lib)
endif()
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tengine_c_api.h DESTINATION include)
if (TENGINE_BUILD_CPP_API)
    install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/tengine_cpp_api.h DESTINATION include)
endif()
