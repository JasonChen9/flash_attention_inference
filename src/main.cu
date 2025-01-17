// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attention main

#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

#define FLASH_ATTENTION_FUNC(name)                                                                      \
    void name(Tensor<cutlass::half_t> *Q, Tensor<cutlass::half_t> *K, Tensor<cutlass::half_t> *V,       \
              Tensor<cutlass::half_t> *O, int *cu_seq_q, int *cu_seq_k, size_t batch, size_t max_seq_q, \
              size_t max_seq_k, bool is_causal, int num_splits, bool is_alibi, cudaDeviceProp *dev_prop)

FLASH_ATTENTION_FUNC(flash_attn);
FLASH_ATTENTION_FUNC(flash_attn_v2);

DEFINE_uint32(b, 2, "batch size");
DEFINE_uint32(sq, 256, "q seq len");
DEFINE_uint32(sk, 256, "k seq len");
DEFINE_uint32(hq, 32, "q head num");
DEFINE_uint32(hk, 32, "k head num");
DEFINE_uint32(d, 128, "head dim");
DEFINE_bool(is_causal, true, "causal mask");
DEFINE_int32(num_splits, 0, "num splits of seq q len for flash attn");
DEFINE_bool(is_alibi, false, "enable alibi");
DEFINE_bool(is_hybrid, false, "hybrid mode");
DEFINE_uint32(prefill_fraction, 0, "percentage occupied by prefill in hybrid mode, the value ranges from 0 to 100");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the CPU result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    omp_set_num_threads(FLAGS_cpu_procs);
    FAI_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

    cudaDeviceProp dev_prop;
    FAI_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    FLOG("MHA start with %u CPU processes on the %u-th GPU: %s", FLAGS_cpu_procs, FLAGS_gpu_rank, dev_prop.name);

    int driver_version = 0;
    int runtime_version = 0;
    FAI_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    FAI_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    FLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);

    FLOG("CUDA capability major/minor version number: %d.%d", dev_prop.major, dev_prop.minor);
    FLOG("%d multiprocessors, %d CUDA cores/MP: %d CUDA cores", dev_prop.multiProcessorCount,
         convert_SM_to_cores(dev_prop.major, dev_prop.minor),
         convert_SM_to_cores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
    FLOG("GPU max clock rate: %.0f MHz (%0.2f GHz)", dev_prop.clockRate * 1e-3f, dev_prop.clockRate * 1e-6f);
    FLOG("Memory clock rate: %.0f MHz (%0.2f GHz)", dev_prop.memoryClockRate * 1e-3f, dev_prop.memoryClockRate * 1e-6f);
    FLOG("Memory bus width: %d-bit", dev_prop.memoryBusWidth);
    FLOG("Total amount of global memory: %.0f MBytes (%llu Bytes)",
         static_cast<float>(dev_prop.totalGlobalMem / 1048576.0f), (unsigned long long)dev_prop.totalGlobalMem);
    FLOG("Total amount of constant memory: %.0f KBytes (%zu Bytes)",
         static_cast<float>(dev_prop.totalConstMem / 1024.0f), dev_prop.totalConstMem);
    FLOG("Total amount of shared memory per block: %.0f KBytes (%zu Bytes)",
         static_cast<float>(dev_prop.sharedMemPerBlock / 1024.0f), dev_prop.sharedMemPerBlock);
    FLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
         static_cast<float>(dev_prop.sharedMemPerMultiprocessor / 1024.0f), dev_prop.sharedMemPerMultiprocessor);
    FLOG("L2 cache size: %.0f KBytes (%d Bytes)", static_cast<float>(dev_prop.l2CacheSize / 1024.0f),
         dev_prop.l2CacheSize);
    FLOG("Total number of registers available per block: %d", dev_prop.regsPerBlock);
    FLOG("Warp size: %d", dev_prop.warpSize);
    FLOG("Max number of threads per multiprocessor: %d", dev_prop.maxThreadsPerMultiProcessor);
    FLOG("Max number of threads per block: %d", dev_prop.maxThreadsPerBlock);
    FLOG("Max dimension size of a thread block (x,y,z): (%d, %d, %d)", dev_prop.maxThreadsDim[0],
         dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    FLOG("Max dimension size of a grid size (x,y,z): (%d, %d, %d)", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
         dev_prop.maxGridSize[2]);

    FLOG(
        "MHA: Softmax (Q (%u x %u x %u x %u) * K^T (%u x %u x %u x %u)) * V (%u x %u x %u x %u) = O (%u x %u x %u x "
        "%u)",
        FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk,
        FLAGS_d, FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_d);
    FLOG(
        "Profiling: is causal: %d, num splits: %d, is alibi: %d, is hybrid: %d, prefill fraction: %u, warmup "
        "iterations: %u, profiling iterations: %u, sleep duration: %u ms, enable check: %d",
        FLAGS_is_causal, FLAGS_num_splits, FLAGS_is_alibi, FLAGS_is_hybrid, FLAGS_prefill_fraction,
        FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);

    Tester tester(FLAGS_b, FLAGS_sq, FLAGS_sk, FLAGS_hq, FLAGS_hk, FLAGS_d, FLAGS_is_causal, FLAGS_num_splits,
                  FLAGS_is_alibi, FLAGS_is_hybrid, FLAGS_prefill_fraction, &dev_prop, FLAGS_warmup_iterations,
                  FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);
    tester.evaluate(flash_attn, "Flash-Attention");
    tester.evaluate(flash_attn_v2, "Flash-Attention-V2");

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    FLOG("Done");

    return 0;
}
