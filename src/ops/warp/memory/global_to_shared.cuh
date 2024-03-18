#pragma once

#include <cuda/pipeline>

#include "../../../common/common.cuh"
#include "../../../types/types.cuh"

namespace kittens {

// ----------- ROW LAYOUTS ----------

template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void load(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height * dst.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        *(float4*)(&dst[{row, col}]) = *(float4*)(&src[row*row_stride + col]);
    }
}
template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void store(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride) {

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height * src.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        *(float4*)(&dst[row*row_stride + col]) = *(float4*)(&src[{row, col}]);
    }
}


template<ducks::sv::all ST, typename U>
__device__ inline static void load(ST &dst, const U *src) {

    int laneid = threadIdx.x % 32;
    int total_calls = (dst.length+31)/32; // allowing it to round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {
        int idx = i * 32 + laneid;
        if (idx < dst.length) { dst[idx] = src[idx]; }
    }
}

template<ducks::sv::all ST, typename U>
__device__ inline static void store(U *dst, const ST &src) {

    int laneid = threadIdx.x % 32;
    int total_calls = (src.length+31)/32; // allowing it to round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) { 
        int idx = i * 32 + laneid;     
        if (idx < src.length) { dst[idx] = src[idx]; }
    }
}


template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void load_async(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height * dst.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        cuda::memcpy_async(
            (void*)(&dst[{row, col}]),
            (void*)(&src[row*row_stride + col]),
            cuda::aligned_size_t<16>(sizeof(float4)),
            barrier
        );
    }
}
template<int height, int width, ducks::st_layout::row layout>
__device__ static inline void store_async(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride, cuda::barrier<cuda::thread_scope_block> &barrier) {
    // each thread needs to do 1 call per width*height
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % 32;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(bf16);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height * src.width;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        cuda::memcpy_async(
            (void*)(&dst[row*row_stride + col]),
            (void*)(&src[{row, col}]),
            cuda::aligned_size_t<16>(sizeof(float4)),
            barrier
        );
    }
}


// ----------- COL LAYOUTS ----------

template<int height, int width, ducks::st_layout::col layout>
__device__ static inline void load(st<bf16, height, width, layout> &dst, const bf16 *src, const int row_stride) {
    
    int laneid = threadIdx.x % 32;

    // in column mode we unfortunately can only transfer one element at at time.
    int elem_per_memcpy = 1;
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = dst.height * dst.width * 8;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        dst[{row, col}] = src[row*row_stride + col];
    }
}
template<int height, int width, ducks::st_layout::col layout>
__device__ static inline void store(bf16 *dst, const st<bf16, height, width, layout> &src, const int row_stride) {

    int laneid = threadIdx.x % 32;

    // in column mode we unfortunately can only transfer one element at at time.
    int elem_per_memcpy = 1;
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = src.height * src.width * 8;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * 32 + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        dst[row*row_stride + col] = src[{row, col}];
    }
}

}