#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-reduce.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/top-k-mask.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_env_set(const char * name) {
    return getenv(name) != nullptr;
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_set("GGML_CUDA_P2P");
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(const char * devName) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    // Extract the base architecture model ("gfx908:sramecc+:xnack-" -> "908")
    std::string archName = devName;
    archName.erase(0, 3);
    archName = archName.substr(0, archName.find(':'));

    const auto hex = [](const std::string & s) { return (int) strtoul(s.c_str(), nullptr, 16); };

    int cc = 0;

    // tease out the version information
    if (archName.size() > 8) {
        // versions labeled generic use '-' as delimiter
        if (archName.compare(archName.size() - 8, 8, "-generic") == 0) {
            archName.resize(archName.size() - 8);
            const size_t sep = archName.find('-');
            cc = 0x100 * hex(archName.substr(0, sep));
            if (sep != std::string::npos) {
                cc += 0x10 * hex(archName.substr(sep + 1));
            }
        }
    } else if (archName.size() >= 3) {
        // ordinary IDs are the version in hex
        cc = hex(archName);
    }

    return GGML_CUDA_CC_OFFSET_AMD + cc;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    ggml_backend_cuda_context & ctx;
    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    ggml_cuda_pool_leg(ggml_backend_cuda_context & ctx, int device) :
        ctx(ctx),
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    // captured CUDA graphs can still use freed buffers, so they must be captured again
    void invalidate_cuda_graphs() {
#ifdef USE_CUDA_GRAPH
        int n_captured = 0;
        for (const auto & [key, graph] : ctx.cuda_graphs) {
            n_captured += graph->instances.size();
        }
        if (n_captured > 0) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: memory freed, %d captured CUDA graphs must be captured again\n", device, n_captured);
        }
        ctx.cuda_graphs.clear();
#endif // USE_CUDA_GRAPH
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            invalidate_cuda_graphs();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
        invalidate_cuda_graphs();
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context & ctx,
                                                                               int                         device,
                                                                               [[maybe_unused]] int        stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(ctx, device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    // pool blocks held across a graph go back to the pool before the pools are destroyed below
    fwht_q8_context.reset();

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // very large pinned allocations can fail (e.g. on Windows) and fall back to pageable memory, smaller ones succeed
    return 4ull*1024*1024*1024;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ ggml_backend_cuda_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ldc));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static bool ggml_cuda_pq2_cublas_i8_enabled() {
    static const bool env = []() {
        const char * v = getenv("GGML_CUDA_PQ2_CUBLAS_I8");
        return v != nullptr && atoi(v) != 0;
    }();
    return env;
}

// =================================================================================================
// Lossless Direct PQ2_0 (exact 2-bit ternary + exact per-128 half d) -> Turing m8n8k16 INT8 Tensor Core
// Per-128-block INT8 activations [-127, +127]: 0.000 PPL degradation, 0.2656 B/w (6.46 GB) weight traffic!
// =================================================================================================
static bool ggml_cuda_pq2_narrow_int8_enabled() {
    static const bool en = []() {
        const char * v = getenv("GGML_CUDA_PQ2_NARROW_INT8");
        return v != nullptr && atoi(v) != 0;
    }();
    return en;
}

// Prefill-only (N >= 256, M <= 65536) High-Precision INT4 via Turing m8n8k32 (217.5 TOPS):
// Exact ternary weights + exact per-128 half d (reuses 6.46 GB A_qs/A_d cache) x per-128 RMS-clipped INT4 activations.
static bool ggml_cuda_pq2_prefill_int4_enabled() {
    static const bool en = []() {
        const char * v = getenv("GGML_CUDA_PQ2_PREFILL_INT4");
        return v != nullptr && atoi(v) != 0;
    }();
    return en;
}

static float ggml_cuda_pq2_prefill_int4_clip() {
    static const float clip = []() {
        const char * v = getenv("GGML_CUDA_PQ2_PREFILL_INT4_CLIP");
        return v != nullptr ? (float) atof(v) : 2.35f;
    }();
    return clip;
}

__device__ __forceinline__ void pq2_mma_s4_m8n8k32(int32_t * d, uint32_t a, uint32_t b, const int32_t * c) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMPERE_MMA_AVAILABLE)
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a), "r"(b), "r"(c[0]), "r"(c[1]));
#else
    GGML_UNUSED_VARS(d, a, b, c);
#endif
}

__global__ static void repack_pq2_to_tc_i8_kernel(
        const block_pq2_0 * __restrict__ src, uint2 * __restrict__ dst_qs, half * __restrict__ dst_d,
        const int M, const int K) {
    const int n_kblk = K >> 7;
    const int rg = blockIdx.x;
    const int kb = blockIdx.y;
    const int lane = threadIdx.x; // 0..31
    const int lane_r = lane >> 2; // 0..7
    const int lane_k = lane & 3;  // 0..3
    const int row = rg * 8 + lane_r;
    if (row >= M) { return; }
    const block_pq2_0 & blk = src[(int64_t) row * n_kblk + kb];
    auto pack_pair = [&](int ks0, int ks1) -> uint32_t {
        const int off0 = ks0 * 8 + lane_k * 2;
        const int off1 = ks1 * 8 + lane_k * 2;
        const uint16_t s0 = (uint16_t) blk.qs[off0] | ((uint16_t) blk.qs[off0 + 1] << 8);
        const uint16_t s1 = (uint16_t) blk.qs[off1] | ((uint16_t) blk.qs[off1 + 1] << 8);
        uint32_t w = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t lo = (s0 >> (b * 4)) & 0xFu;
            const uint32_t hi = (s1 >> (b * 4)) & 0xFu;
            w |= (lo | (hi << 4)) << (b * 8);
        }
        return w;
    };
    dst_qs[((int64_t) rg * n_kblk + kb) * 32 + lane] = make_uint2(pack_pair(0, 1), pack_pair(2, 3));
    if (lane_k == 0) {
        dst_d[((int64_t) rg * n_kblk + kb) * 8 + lane_r] = blk.d;
    }
}

struct pq2_tc_i8_entry {
    uint2 * qs = nullptr;
    half  * d  = nullptr;
    int64_t M  = 0;
    int64_t K  = 0;
};

static pq2_tc_i8_entry get_or_repack_pq2_tc_i8(
        const block_pq2_0 * src0, const int64_t M, const int64_t K, cudaStream_t stream, bool allow_alloc) {
    static std::mutex mu;
    static std::unordered_map<const void *, pq2_tc_i8_entry> cache;
    static std::vector<pq2_tc_i8_entry> retired;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(src0);
    if (it != cache.end() && it->second.M >= M && it->second.K == K) {
        return it->second;
    }
    if (!allow_alloc) {
        return {};
    }
    const int64_t n_rg   = (M + 7) >> 3;
    const int64_t n_kblk = K >> 7;
    const size_t  need_qs = (size_t) n_rg * (size_t) n_kblk * 32 * sizeof(uint2);
    const size_t  need_d  = (size_t) n_rg * (size_t) n_kblk * 8  * sizeof(half);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b < need_qs + need_d + (2ULL << 30)) {
        return {};
    }
    pq2_tc_i8_entry e;
    e.M = M; e.K = K;
    if (cudaMalloc(&e.qs, need_qs) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    if (cudaMalloc(&e.d, need_d) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFree(e.qs);
        return {};
    }
    dim3 grid((unsigned) n_rg, (unsigned) n_kblk);
    repack_pq2_to_tc_i8_kernel<<<grid, 32, 0, stream>>>(src0, e.qs, e.d, (int) M, (int) K);
    CUDA_CHECK(cudaGetLastError());
    if (it != cache.end()) {
        retired.push_back(it->second);
    }
    cache[src0] = e;
    return e;
}

// 1-pass ZERO-SMEM F32 -> exact 2-nibble INT8 [-120, +119] (b = 16*b_hi + b_lo) + per-128 B_sum + B_scales
// Stored per 128-block as 16 uint32_t of b_hi followed by 16 uint32_t of b_lo (32 uint32_t/block).
__global__ static void quantize_f32_to_i8_sum128_kernel(
        const float * __restrict__ src1, uint32_t * __restrict__ dst_i8,
        int32_t * __restrict__ dst_sum128, float * __restrict__ dst_scales128,
        const int N, const int N_pad, const int K, const int64_t s11) {
    const int col = blockIdx.x;
    if (col >= N_pad) { return; }
    const int tid = threadIdx.x; // 0..511 (half-warp = 16 threads = 128 elements!)
    const int K8 = K >> 3;
    const int K4 = K >> 2;
    const int n_kblk = K >> 7;
    uint32_t * out32  = dst_i8        + (int64_t) col * K4;
    int32_t  * out_s  = dst_sum128    + (int64_t) col * n_kblk;
    float    * out_sc = dst_scales128 + (int64_t) col * n_kblk;
    if (col >= N) {
        for (int i = tid; i < K4; i += 512) { out32[i] = 0u; }
        for (int i = tid; i < n_kblk; i += 512) { out_s[i] = 0; out_sc[i] = 0.0f; }
        return;
    }
    const float4 * in4 = (const float4 *) (src1 + (int64_t) col * s11);
    for (int i = tid; i < K8; i += 512) {
        const float4 a = in4[2 * i + 0];
        const float4 b = in4[2 * i + 1];
        float bmax = fmaxf(
            fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))),
            fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFF, bmax, off));
        }
        const float inv_scale = bmax > 0.0f ? (119.0f / bmax) : 0.0f;
        int q[8];
        q[0] = max(-120, min(119, __float2int_rn(a.x * inv_scale)));
        q[1] = max(-120, min(119, __float2int_rn(a.y * inv_scale)));
        q[2] = max(-120, min(119, __float2int_rn(a.z * inv_scale)));
        q[3] = max(-120, min(119, __float2int_rn(a.w * inv_scale)));
        q[4] = max(-120, min(119, __float2int_rn(b.x * inv_scale)));
        q[5] = max(-120, min(119, __float2int_rn(b.y * inv_scale)));
        q[6] = max(-120, min(119, __float2int_rn(b.z * inv_scale)));
        q[7] = max(-120, min(119, __float2int_rn(b.w * inv_scale)));
        uint32_t w_hi = 0u, w_lo = 0u;
        int s = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int lo = ((q[j] + 8) & 15) - 8;
            const int hi = (q[j] - lo) >> 4;
            w_hi |= ((uint32_t)(hi & 0xF)) << (j * 4);
            w_lo |= ((uint32_t)(lo & 0xF)) << (j * 4);
            s += q[j];
        }
        const int kb = i >> 4;
        const int kw = ((i & 3) << 2) | ((i >> 2) & 3);
        out32[(kb << 5) + kw]      = w_hi;
        out32[(kb << 5) + 16 + kw] = w_lo;
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            s += __shfl_xor_sync(0xFFFFFFFF, s, off);
        }
        if ((tid & 15) == 0) {
            out_s[kb]  = s;
            out_sc[kb] = bmax / 119.0f;
        }
    }
}

template <int N_PAD, int WARPS, int R_TILES>
__global__ static void __launch_bounds__(WARPS * 32)
pq2_narrow_int8_v2_kernel(
        const uint2    * __restrict__ A_qs,
        const half     * __restrict__ A_d,
        const uint32_t * __restrict__ B_i8,
        const int32_t  * __restrict__ B_sum,
        const float    * __restrict__ B_scales,
        const float    * __restrict__ add_src,
        float          * __restrict__ C,
        const int M, const int N, const int K, const int64_t s_dst) {
    constexpr int N_TILES = N_PAD / 8;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg0  = (blockIdx.x * WARPS + warp) * R_TILES;
    const int col_base = blockIdx.y * N_PAD;

    const int n_kblk = K >> 7;
    const int K4     = K >> 2;
    const int n_sk   = gridDim.z;
    const int sk_id  = blockIdx.z;
    const int kb_per = (n_kblk + n_sk - 1) / n_sk;
    const int kb_beg = sk_id * kb_per;
    const int kb_end = min(n_kblk, kb_beg + kb_per);
    if (kb_beg >= kb_end) { return; }

    __shared__ uint32_t sB[2][N_PAD][33]; // [0..15] = b_hi, [16..31] = b_lo (+1 pad: zero bank conflicts!)
    __shared__ int32_t  sBsum[2][N_PAD];
    __shared__ float    sBsc[2][N_PAD];

    float acc[R_TILES][N_TILES][2];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt)
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            acc[rt][nt][0] = 0.0f;
            acc[rt][nt][1] = 0.0f;
        }

    const int lane_r = lane >> 2;
    const int lane_k = lane & 3;
    const int c0_off = lane_k * 2;
    const int c1_off = c0_off + 1;

    constexpr int TOTAL_B_WORDS = N_PAD * 32;
    constexpr int B_PER_THR     = (TOTAL_B_WORDS + WARPS * 32 - 1) / (WARPS * 32);
    for (int idx = tid; idx < TOTAL_B_WORDS; idx += WARPS * 32) {
        sB[0][idx >> 5][idx & 31] = B_i8[(int64_t)(col_base + (idx >> 5)) * K4 + (kb_beg << 5) + (idx & 31)];
    }
    if (tid < N_PAD) {
        sBsum[0][tid] = B_sum[(int64_t)(col_base + tid) * n_kblk + kb_beg];
        sBsc[0][tid]  = B_scales[(int64_t)(col_base + tid) * n_kblk + kb_beg];
    }
    __syncthreads();

    uint2 uA[R_TILES];
    float dA[R_TILES];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int rg = rg0 + rt;
        if (rg * 8 < M) {
            uA[rt] = A_qs[((int64_t) rg * n_kblk + kb_beg) * 32 + lane];
            dA[rt] = __half2float(A_d[((int64_t) rg * n_kblk + kb_beg) * 8 + lane_r]);
        } else {
            uA[rt] = make_uint2(0u, 0u);
            dA[rt] = 0.0f;
        }
    }

    for (int kb = kb_beg; kb < kb_end; ++kb) {
        const int cur = (kb - kb_beg) & 1;
        const int nxt = cur ^ 1;
        const bool has_nxt = (kb + 1 < kb_end);

        uint32_t a0[R_TILES], a1[R_TILES], a2[R_TILES], a3[R_TILES];
#pragma unroll
        for (int rt = 0; rt < R_TILES; ++rt) {
            a0[rt] = (uA[rt].x & 0x03030303u) | ((uA[rt].x & 0x0C0C0C0Cu) << 2);
            a1[rt] = ((uA[rt].x >> 4) & 0x03030303u) | ((uA[rt].x >> 2) & 0x30303030u);
            a2[rt] = (uA[rt].y & 0x03030303u) | ((uA[rt].y & 0x0C0C0C0Cu) << 2);
            a3[rt] = ((uA[rt].y >> 4) & 0x03030303u) | ((uA[rt].y >> 2) & 0x30303030u);
        }

        // Software-pipeline NEXT k-block A_qs/A_d AND B_i8 loads into registers BEFORE MMAs!
        uint2 uA_nxt[R_TILES];
        float dA_nxt[R_TILES];
        uint32_t gB[B_PER_THR];
        int32_t  gBsum = 0;
        float    gBsc  = 0.0f;
        if (has_nxt) {
            const int kb1 = kb + 1;
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                const int rg = rg0 + rt;
                if (rg * 8 < M) {
                    uA_nxt[rt] = A_qs[((int64_t) rg * n_kblk + kb1) * 32 + lane];
                    dA_nxt[rt] = __half2float(A_d[((int64_t) rg * n_kblk + kb1) * 8 + lane_r]);
                } else {
                    uA_nxt[rt] = make_uint2(0u, 0u);
                    dA_nxt[rt] = 0.0f;
                }
            }
#pragma unroll
            for (int bi = 0; bi < B_PER_THR; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_B_WORDS) {
                    gB[bi] = B_i8[(int64_t)(col_base + (idx >> 5)) * K4 + (kb1 << 5) + (idx & 31)];
                }
            }
            if (tid < N_PAD) {
                gBsum = B_sum[(int64_t)(col_base + tid) * n_kblk + kb1];
                gBsc  = B_scales[(int64_t)(col_base + tid) * n_kblk + kb1];
            }
        }

#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            const int b_col = nt * 8 + lane_r;
            const uint32_t bh0 = sB[cur][b_col][(lane_k << 2) + 0];
            const uint32_t bh1 = sB[cur][b_col][(lane_k << 2) + 1];
            const uint32_t bh2 = sB[cur][b_col][(lane_k << 2) + 2];
            const uint32_t bh3 = sB[cur][b_col][(lane_k << 2) + 3];
            const uint32_t bl0 = sB[cur][b_col][16 + (lane_k << 2) + 0];
            const uint32_t bl1 = sB[cur][b_col][16 + (lane_k << 2) + 1];
            const uint32_t bl2 = sB[cur][b_col][16 + (lane_k << 2) + 2];
            const uint32_t bl3 = sB[cur][b_col][16 + (lane_k << 2) + 3];
            const int32_t bs0 = sBsum[cur][nt * 8 + c0_off];
            const int32_t bs1 = sBsum[cur][nt * 8 + c1_off];
            const float bsc0  = sBsc[cur][nt * 8 + c0_off];
            const float bsc1  = sBsc[cur][nt * 8 + c1_off];
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                int32_t dhi[2] = { 0, 0 }, dlo[2] = { 0, 0 };
                pq2_mma_s4_m8n8k32(dhi, a0[rt], bh0, dhi);
                pq2_mma_s4_m8n8k32(dhi, a1[rt], bh1, dhi);
                pq2_mma_s4_m8n8k32(dhi, a2[rt], bh2, dhi);
                pq2_mma_s4_m8n8k32(dhi, a3[rt], bh3, dhi);
                pq2_mma_s4_m8n8k32(dlo, a0[rt], bl0, dlo);
                pq2_mma_s4_m8n8k32(dlo, a1[rt], bl1, dlo);
                pq2_mma_s4_m8n8k32(dlo, a2[rt], bl2, dlo);
                pq2_mma_s4_m8n8k32(dlo, a3[rt], bl3, dlo);
                acc[rt][nt][0] = fmaf((float)((dhi[0] << 4) + dlo[0] - bs0), dA[rt] * bsc0, acc[rt][nt][0]);
                acc[rt][nt][1] = fmaf((float)((dhi[1] << 4) + dlo[1] - bs1), dA[rt] * bsc1, acc[rt][nt][1]);
            }
        }

        if (has_nxt) {
#pragma unroll
            for (int bi = 0; bi < B_PER_THR; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_B_WORDS) {
                    sB[nxt][idx >> 5][idx & 31] = gB[bi];
                }
            }
            if (tid < N_PAD) {
                sBsum[nxt][tid] = gBsum;
                sBsc[nxt][tid]  = gBsc;
            }
            __syncthreads();
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                uA[rt] = uA_nxt[rt];
                dA[rt] = dA_nxt[rt];
            }
        }
    }

#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int out_row = (rg0 + rt) * 8 + lane_r;
        if (out_row < M) {
#pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                const int col0 = col_base + nt * 8 + c0_off;
                const int col1 = col_base + nt * 8 + c1_off;
                if (col0 < N) {
                    float v = acc[rt][nt][0];
                    if (n_sk == 1) {
                        if (add_src) { v += add_src[(int64_t) col0 * s_dst + out_row]; }
                        C[(int64_t) col0 * s_dst + out_row] = v;
                    } else {
                        atomicAdd(&C[(int64_t) col0 * s_dst + out_row], v);
                    }
                }
                if (col1 < N) {
                    float v = acc[rt][nt][1];
                    if (n_sk == 1) {
                        if (add_src) { v += add_src[(int64_t) col1 * s_dst + out_row]; }
                        C[(int64_t) col1 * s_dst + out_row] = v;
                    } else {
                        atomicAdd(&C[(int64_t) col1 * s_dst + out_row], v);
                    }
                }
            }
        }
    }
}

template <int N_PAD, int WARPS, int R_TILES>
__global__ static void __launch_bounds__(WARPS * 32)
pq2_narrow_int8_swiglu_v2_kernel(
        const uint2    * __restrict__ G_qs,
        const half     * __restrict__ G_d,
        const uint2    * __restrict__ U_qs,
        const half     * __restrict__ U_d,
        const uint32_t * __restrict__ B_i8,
        const int32_t  * __restrict__ B_sum,
        const float    * __restrict__ B_scales,
        float          * __restrict__ C,
        const int M, const int N, const int K, const int64_t s_dst) {
    constexpr int N_TILES = N_PAD / 8;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rg0  = (blockIdx.x * WARPS + warp) * R_TILES;
    const int col_base = blockIdx.y * N_PAD;

    const int n_kblk = K >> 7;
    const int K4     = K >> 2;
    __shared__ uint32_t sB[2][N_PAD][33];
    __shared__ int32_t  sBsum[2][N_PAD];
    __shared__ float    sBsc[2][N_PAD];

    float acc_g[R_TILES][N_TILES][2];
    float acc_u[R_TILES][N_TILES][2];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt)
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            acc_g[rt][nt][0] = 0.0f; acc_g[rt][nt][1] = 0.0f;
            acc_u[rt][nt][0] = 0.0f; acc_u[rt][nt][1] = 0.0f;
        }

    const int lane_r = lane >> 2;
    const int lane_k = lane & 3;
    const int c0_off = lane_k * 2;
    const int c1_off = c0_off + 1;

    constexpr int TOTAL_B_WORDS = N_PAD * 32;
    constexpr int B_PER_THR     = (TOTAL_B_WORDS + WARPS * 32 - 1) / (WARPS * 32);
    for (int idx = tid; idx < TOTAL_B_WORDS; idx += WARPS * 32) {
        sB[0][idx >> 5][idx & 31] = B_i8[(int64_t)(col_base + (idx >> 5)) * K4 + (idx & 31)];
    }
    if (tid < N_PAD) {
        sBsum[0][tid] = B_sum[(int64_t)(col_base + tid) * n_kblk];
        sBsc[0][tid]  = B_scales[(int64_t)(col_base + tid) * n_kblk];
    }
    __syncthreads();

    uint2 uG[R_TILES], uU[R_TILES];
    float dG[R_TILES], dU[R_TILES];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int rg = rg0 + rt;
        if (rg * 8 < M) {
            const int64_t qoff = ((int64_t) rg * n_kblk) * 32 + lane;
            const int64_t doff = ((int64_t) rg * n_kblk) * 8 + lane_r;
            uG[rt] = G_qs[qoff]; dG[rt] = __half2float(G_d[doff]);
            uU[rt] = U_qs[qoff]; dU[rt] = __half2float(U_d[doff]);
        } else {
            uG[rt] = make_uint2(0u, 0u); dG[rt] = 0.0f;
            uU[rt] = make_uint2(0u, 0u); dU[rt] = 0.0f;
        }
    }

    for (int kb = 0; kb < n_kblk; ++kb) {
        const int cur = kb & 1;
        const int nxt = cur ^ 1;
        const bool has_nxt = (kb + 1 < n_kblk);

        uint32_t g0[R_TILES], g1[R_TILES], g2[R_TILES], g3[R_TILES];
        uint32_t p0[R_TILES], p1[R_TILES], p2[R_TILES], p3[R_TILES];
#pragma unroll
        for (int rt = 0; rt < R_TILES; ++rt) {
            g0[rt] = (uG[rt].x & 0x03030303u) | ((uG[rt].x & 0x0C0C0C0Cu) << 2);
            g1[rt] = ((uG[rt].x >> 4) & 0x03030303u) | ((uG[rt].x >> 2) & 0x30303030u);
            g2[rt] = (uG[rt].y & 0x03030303u) | ((uG[rt].y & 0x0C0C0C0Cu) << 2);
            g3[rt] = ((uG[rt].y >> 4) & 0x03030303u) | ((uG[rt].y >> 2) & 0x30303030u);
            p0[rt] = (uU[rt].x & 0x03030303u) | ((uU[rt].x & 0x0C0C0C0Cu) << 2);
            p1[rt] = ((uU[rt].x >> 4) & 0x03030303u) | ((uU[rt].x >> 2) & 0x30303030u);
            p2[rt] = (uU[rt].y & 0x03030303u) | ((uU[rt].y & 0x0C0C0C0Cu) << 2);
            p3[rt] = ((uU[rt].y >> 4) & 0x03030303u) | ((uU[rt].y >> 2) & 0x30303030u);
        }

        uint2 uG_nxt[R_TILES], uU_nxt[R_TILES];
        float dG_nxt[R_TILES], dU_nxt[R_TILES];
        uint32_t gB[B_PER_THR];
        int32_t  gBsum = 0;
        float    gBsc  = 0.0f;
        if (has_nxt) {
            const int kb1 = kb + 1;
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                const int rg = rg0 + rt;
                if (rg * 8 < M) {
                    const int64_t qoff = ((int64_t) rg * n_kblk + kb1) * 32 + lane;
                    const int64_t doff = ((int64_t) rg * n_kblk + kb1) * 8 + lane_r;
                    uG_nxt[rt] = G_qs[qoff]; dG_nxt[rt] = __half2float(G_d[doff]);
                    uU_nxt[rt] = U_qs[qoff]; dU_nxt[rt] = __half2float(U_d[doff]);
                } else {
                    uG_nxt[rt] = make_uint2(0u, 0u); dG_nxt[rt] = 0.0f;
                    uU_nxt[rt] = make_uint2(0u, 0u); dU_nxt[rt] = 0.0f;
                }
            }
#pragma unroll
            for (int bi = 0; bi < B_PER_THR; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_B_WORDS) {
                    gB[bi] = B_i8[(int64_t)(col_base + (idx >> 5)) * K4 + (kb1 << 5) + (idx & 31)];
                }
            }
            if (tid < N_PAD) {
                gBsum = B_sum[(int64_t)(col_base + tid) * n_kblk + kb1];
                gBsc  = B_scales[(int64_t)(col_base + tid) * n_kblk + kb1];
            }
        }

#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
            const int b_col = nt * 8 + lane_r;
            const uint32_t bh0 = sB[cur][b_col][(lane_k << 2) + 0];
            const uint32_t bh1 = sB[cur][b_col][(lane_k << 2) + 1];
            const uint32_t bh2 = sB[cur][b_col][(lane_k << 2) + 2];
            const uint32_t bh3 = sB[cur][b_col][(lane_k << 2) + 3];
            const uint32_t bl0 = sB[cur][b_col][16 + (lane_k << 2) + 0];
            const uint32_t bl1 = sB[cur][b_col][16 + (lane_k << 2) + 1];
            const uint32_t bl2 = sB[cur][b_col][16 + (lane_k << 2) + 2];
            const uint32_t bl3 = sB[cur][b_col][16 + (lane_k << 2) + 3];
            const int32_t bs0 = sBsum[cur][nt * 8 + c0_off];
            const int32_t bs1 = sBsum[cur][nt * 8 + c1_off];
            const float bsc0  = sBsc[cur][nt * 8 + c0_off];
            const float bsc1  = sBsc[cur][nt * 8 + c1_off];
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                int32_t dgh[2] = { 0, 0 }, dgl[2] = { 0, 0 };
                pq2_mma_s4_m8n8k32(dgh, g0[rt], bh0, dgh);
                pq2_mma_s4_m8n8k32(dgh, g1[rt], bh1, dgh);
                pq2_mma_s4_m8n8k32(dgh, g2[rt], bh2, dgh);
                pq2_mma_s4_m8n8k32(dgh, g3[rt], bh3, dgh);
                pq2_mma_s4_m8n8k32(dgl, g0[rt], bl0, dgl);
                pq2_mma_s4_m8n8k32(dgl, g1[rt], bl1, dgl);
                pq2_mma_s4_m8n8k32(dgl, g2[rt], bl2, dgl);
                pq2_mma_s4_m8n8k32(dgl, g3[rt], bl3, dgl);
                acc_g[rt][nt][0] = fmaf((float)((dgh[0] << 4) + dgl[0] - bs0), dG[rt] * bsc0, acc_g[rt][nt][0]);
                acc_g[rt][nt][1] = fmaf((float)((dgh[1] << 4) + dgl[1] - bs1), dG[rt] * bsc1, acc_g[rt][nt][1]);

                int32_t duh[2] = { 0, 0 }, dul[2] = { 0, 0 };
                pq2_mma_s4_m8n8k32(duh, p0[rt], bh0, duh);
                pq2_mma_s4_m8n8k32(duh, p1[rt], bh1, duh);
                pq2_mma_s4_m8n8k32(duh, p2[rt], bh2, duh);
                pq2_mma_s4_m8n8k32(duh, p3[rt], bh3, duh);
                pq2_mma_s4_m8n8k32(dul, p0[rt], bl0, dul);
                pq2_mma_s4_m8n8k32(dul, p1[rt], bl1, dul);
                pq2_mma_s4_m8n8k32(dul, p2[rt], bl2, dul);
                pq2_mma_s4_m8n8k32(dul, p3[rt], bl3, dul);
                acc_u[rt][nt][0] = fmaf((float)((duh[0] << 4) + dul[0] - bs0), dU[rt] * bsc0, acc_u[rt][nt][0]);
                acc_u[rt][nt][1] = fmaf((float)((duh[1] << 4) + dul[1] - bs1), dU[rt] * bsc1, acc_u[rt][nt][1]);
            }
        }

        if (has_nxt) {
#pragma unroll
            for (int bi = 0; bi < B_PER_THR; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_B_WORDS) {
                    sB[nxt][idx >> 5][idx & 31] = gB[bi];
                }
            }
            if (tid < N_PAD) {
                sBsum[nxt][tid] = gBsum;
                sBsc[nxt][tid]  = gBsc;
            }
            __syncthreads();
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                uG[rt] = uG_nxt[rt]; dG[rt] = dG_nxt[rt];
                uU[rt] = uU_nxt[rt]; dU[rt] = dU_nxt[rt];
            }
        }
    }

#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int out_row = (rg0 + rt) * 8 + lane_r;
        if (out_row < M) {
#pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
                const int col0 = col_base + nt * 8 + c0_off;
                const int col1 = col_base + nt * 8 + c1_off;
                if (col0 < N) {
                    const float gv0 = acc_g[rt][nt][0];
                    C[(int64_t) col0 * s_dst + out_row] = (gv0 / (1.0f + __expf(-gv0))) * acc_u[rt][nt][0];
                }
                if (col1 < N) {
                    const float gv1 = acc_g[rt][nt][1];
                    C[(int64_t) col1 * s_dst + out_row] = (gv1 / (1.0f + __expf(-gv1))) * acc_u[rt][nt][1];
                }
            }
        }
    }
}

// 2D Warp-Tiled Lossless 2-Nibble INT8 GEMM (WARPS_M x WARPS_N warp grid, 128-bit LDG.128/STS.128/LDS.128)
template <int WARPS_M, int WARPS_N, int R_TILES, int N_TILES_W, bool SINGLE_BUF = false>
__global__ static void __launch_bounds__(WARPS_M * WARPS_N * 32)
pq2_narrow2d_i8_kernel(
        const uint2   * __restrict__ A_qs,
        const half    * __restrict__ A_d,
        const uint4   * __restrict__ B_i8_u4,
        const int32_t * __restrict__ B_sum,
        const float   * __restrict__ B_sc,
        const float   * __restrict__ add_src,
        float         * __restrict__ C,
        const int M, const int N, const int K, const int64_t s_dst) {
    constexpr int WARPS  = WARPS_M * WARPS_N;
    constexpr int N_PAD  = WARPS_N * N_TILES_W * 8;
    constexpr int N_BUFS = SINGLE_BUF ? 1 : 2;
    const int tid      = threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;
    int bx = blockIdx.x, by = blockIdx.y;
    if (gridDim.y >= 2) {
        const int gy = gridDim.y;
        const int sw = (gy >= 4) ? 4 : 2;
        const int linear = blockIdx.y * gridDim.x + blockIdx.x;
        const int grp_tiles = gridDim.x * sw;
        const int grp = linear / grp_tiles;
        const int rem = linear - grp * grp_tiles;
        const int y0  = grp * sw;
        const int sw_cur = min(sw, gy - y0);
        by = y0 + (rem % sw_cur);
        bx = rem / sw_cur;
    }
    const int rg0      = (bx * WARPS_M + warp_m) * R_TILES;
    const int col_base = by * N_PAD;
    const int wn_col0  = warp_n * N_TILES_W * 8;

    const int n_kblk   = K >> 7;
    const int split_k  = gridDim.z;
    const int sk_id    = blockIdx.z;
    const int kb_start = (n_kblk * sk_id) / split_k;
    const int kb_end   = (n_kblk * (sk_id + 1)) / split_k;
    if (kb_start >= kb_end) { return; }

    __shared__ uint4   sB[N_BUFS][N_PAD][9];
    __shared__ int32_t sBsum[N_BUFS][N_PAD];
    __shared__ float   sBsc[N_BUFS][N_PAD];

    float acc[R_TILES][N_TILES_W][2];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt)
#pragma unroll
        for (int nt = 0; nt < N_TILES_W; ++nt) { acc[rt][nt][0] = 0.0f; acc[rt][nt][1] = 0.0f; }

    const int lane_r = lane >> 2;
    const int lane_k = lane & 3;
    const int c0_off = lane_k * 2;
    const int c1_off = c0_off + 1;

    constexpr int TOTAL_U4  = N_PAD * 8;
    constexpr int U4_PER_TH = (TOTAL_U4 + WARPS * 32 - 1) / (WARPS * 32);
#pragma unroll
    for (int bi = 0; bi < U4_PER_TH; ++bi) {
        const int idx = tid + bi * (WARPS * 32);
        if (idx < TOTAL_U4) {
            const int c = idx >> 3;
            const int w = idx & 7;
            sB[0][c][w] = B_i8_u4[((int64_t)(col_base + c) * n_kblk + kb_start) * 8 + w];
        }
    }
    if (tid < N_PAD) {
        sBsum[0][tid] = B_sum[(int64_t)(col_base + tid) * n_kblk + kb_start];
        sBsc[0][tid]  = B_sc[(int64_t)(col_base + tid) * n_kblk + kb_start];
    }
    __syncthreads();

    uint2 uA[R_TILES];
    float dA[R_TILES];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int rg = rg0 + rt;
        if (rg * 8 < M) {
            uA[rt] = A_qs[((int64_t) rg * n_kblk + kb_start) * 32 + lane];
            dA[rt] = __half2float(A_d[((int64_t) rg * n_kblk + kb_start) * 8 + lane_r]);
        } else { uA[rt] = make_uint2(0u, 0u); dA[rt] = 0.0f; }
    }

    for (int kb = kb_start; kb < kb_end; ++kb) {
        const int cur = SINGLE_BUF ? 0 : ((kb - kb_start) & 1);
        const int nxt = SINGLE_BUF ? 0 : (cur ^ 1);
        const bool has_nxt = (kb + 1 < kb_end);

        uint32_t a0[R_TILES], a1[R_TILES], a2[R_TILES], a3[R_TILES];
#pragma unroll
        for (int rt = 0; rt < R_TILES; ++rt) {
            a0[rt] = (uA[rt].x & 0x03030303u) | ((uA[rt].x & 0x0C0C0C0Cu) << 2);
            a1[rt] = ((uA[rt].x >> 4) & 0x03030303u) | ((uA[rt].x >> 2) & 0x30303030u);
            a2[rt] = (uA[rt].y & 0x03030303u) | ((uA[rt].y & 0x0C0C0C0Cu) << 2);
            a3[rt] = ((uA[rt].y >> 4) & 0x03030303u) | ((uA[rt].y >> 2) & 0x30303030u);
        }

        uint2 uA_nxt[R_TILES];
        float dA_nxt[R_TILES];
        uint4 gB[U4_PER_TH];
        int32_t gBsum = 0;
        float   gBsc  = 0.0f;
        if (has_nxt) {
            const int kb1 = kb + 1;
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                const int rg = rg0 + rt;
                if (rg * 8 < M) {
                    uA_nxt[rt] = A_qs[((int64_t) rg * n_kblk + kb1) * 32 + lane];
                    dA_nxt[rt] = __half2float(A_d[((int64_t) rg * n_kblk + kb1) * 8 + lane_r]);
                } else { uA_nxt[rt] = make_uint2(0u, 0u); dA_nxt[rt] = 0.0f; }
            }
#pragma unroll
            for (int bi = 0; bi < U4_PER_TH; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_U4) {
                    const int c = idx >> 3;
                    const int w = idx & 7;
                    gB[bi] = B_i8_u4[((int64_t)(col_base + c) * n_kblk + kb1) * 8 + w];
                }
            }
            if (tid < N_PAD) {
                gBsum = B_sum[(int64_t)(col_base + tid) * n_kblk + kb1];
                gBsc  = B_sc[(int64_t)(col_base + tid) * n_kblk + kb1];
            }
        }

#pragma unroll
        for (int nt = 0; nt < N_TILES_W; ++nt) {
            const int b_col = wn_col0 + nt * 8 + lane_r;
            const uint4 bh = sB[cur][b_col][lane_k];
            const uint4 bl = sB[cur][b_col][4 + lane_k];
            const int32_t bs0 = sBsum[cur][wn_col0 + nt * 8 + c0_off];
            const int32_t bs1 = sBsum[cur][wn_col0 + nt * 8 + c1_off];
            const float   bsc0 = sBsc[cur][wn_col0 + nt * 8 + c0_off];
            const float   bsc1 = sBsc[cur][wn_col0 + nt * 8 + c1_off];
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                int32_t dhi[2] = { 0, 0 }, dlo[2] = { 0, 0 };
                pq2_mma_s4_m8n8k32(dhi, a0[rt], bh.x, dhi);
                pq2_mma_s4_m8n8k32(dhi, a1[rt], bh.y, dhi);
                pq2_mma_s4_m8n8k32(dhi, a2[rt], bh.z, dhi);
                pq2_mma_s4_m8n8k32(dhi, a3[rt], bh.w, dhi);
                pq2_mma_s4_m8n8k32(dlo, a0[rt], bl.x, dlo);
                pq2_mma_s4_m8n8k32(dlo, a1[rt], bl.y, dlo);
                pq2_mma_s4_m8n8k32(dlo, a2[rt], bl.z, dlo);
                pq2_mma_s4_m8n8k32(dlo, a3[rt], bl.w, dlo);
                acc[rt][nt][0] = fmaf((float)((dhi[0] << 4) + dlo[0] - bs0), dA[rt] * bsc0, acc[rt][nt][0]);
                acc[rt][nt][1] = fmaf((float)((dhi[1] << 4) + dlo[1] - bs1), dA[rt] * bsc1, acc[rt][nt][1]);
            }
        }
        if (has_nxt) {
            if constexpr (SINGLE_BUF) { __syncthreads(); }
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) { uA[rt] = uA_nxt[rt]; dA[rt] = dA_nxt[rt]; }
#pragma unroll
            for (int bi = 0; bi < U4_PER_TH; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_U4) { sB[nxt][idx >> 3][idx & 7] = gB[bi]; }
            }
            if (tid < N_PAD) { sBsum[nxt][tid] = gBsum; sBsc[nxt][tid] = gBsc; }
            __syncthreads();
        }
    }

#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int row = (rg0 + rt) * 8 + lane_r;
        if (row < M) {
#pragma unroll
            for (int nt = 0; nt < N_TILES_W; ++nt) {
                const int c0 = col_base + wn_col0 + nt * 8 + c0_off;
                const int c1 = col_base + wn_col0 + nt * 8 + c1_off;
                if (split_k == 1) {
                    if (c0 < N) {
                        const int64_t off = (int64_t) c0 * s_dst + row;
                        C[off] = add_src ? (acc[rt][nt][0] + add_src[off]) : acc[rt][nt][0];
                    }
                    if (c1 < N) {
                        const int64_t off = (int64_t) c1 * s_dst + row;
                        C[off] = add_src ? (acc[rt][nt][1] + add_src[off]) : acc[rt][nt][1];
                    }
                } else {
                    if (c0 < N) { atomicAdd(&C[(int64_t) c0 * s_dst + row], acc[rt][nt][0]); }
                    if (c1 < N) { atomicAdd(&C[(int64_t) c1 * s_dst + row], acc[rt][nt][1]); }
                }
            }
        }
    }
}

template <int WARPS_M, int WARPS_N, int R_TILES, int N_TILES_W, bool SINGLE_BUF = false>
__global__ static void __launch_bounds__(WARPS_M * WARPS_N * 32)
pq2_narrow2d_swiglu_kernel(
        const uint2   * __restrict__ G_qs, const half * __restrict__ G_d,
        const uint2   * __restrict__ U_qs, const half * __restrict__ U_d,
        const uint4   * __restrict__ B_i8_u4, const int32_t * __restrict__ B_sum, const float * __restrict__ B_sc,
        float * __restrict__ C, const int M, const int N, const int K, const int64_t s_dst) {
    constexpr int WARPS  = WARPS_M * WARPS_N;
    constexpr int N_PAD  = WARPS_N * N_TILES_W * 8;
    constexpr int N_BUFS = SINGLE_BUF ? 1 : 2;
    const int tid      = threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;
    int bx = blockIdx.x, by = blockIdx.y;
    if (gridDim.y >= 2) {
        const int gy = gridDim.y;
        const int sw = (gy >= 4) ? 4 : 2;
        const int linear = blockIdx.y * gridDim.x + blockIdx.x;
        const int grp_tiles = gridDim.x * sw;
        const int grp = linear / grp_tiles;
        const int rem = linear - grp * grp_tiles;
        const int y0  = grp * sw;
        const int sw_cur = min(sw, gy - y0);
        by = y0 + (rem % sw_cur);
        bx = rem / sw_cur;
    }
    const int rg0      = (bx * WARPS_M + warp_m) * R_TILES;
    const int col_base = by * N_PAD;
    const int wn_col0  = warp_n * N_TILES_W * 8;
    const int n_kblk   = K >> 7;

    __shared__ uint4   sB[N_BUFS][N_PAD][9];
    __shared__ int32_t sBsum[N_BUFS][N_PAD];
    __shared__ float   sBsc[N_BUFS][N_PAD];

    float acc_g[R_TILES][N_TILES_W][2], acc_u[R_TILES][N_TILES_W][2];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt)
#pragma unroll
        for (int nt = 0; nt < N_TILES_W; ++nt) {
            acc_g[rt][nt][0] = 0.0f; acc_g[rt][nt][1] = 0.0f;
            acc_u[rt][nt][0] = 0.0f; acc_u[rt][nt][1] = 0.0f;
        }

    const int lane_r = lane >> 2;
    const int lane_k = lane & 3;
    const int c0_off = lane_k * 2;
    const int c1_off = c0_off + 1;
    constexpr int TOTAL_U4  = N_PAD * 8;
    constexpr int U4_PER_TH = (TOTAL_U4 + WARPS * 32 - 1) / (WARPS * 32);

#pragma unroll
    for (int bi = 0; bi < U4_PER_TH; ++bi) {
        const int idx = tid + bi * (WARPS * 32);
        if (idx < TOTAL_U4) { sB[0][idx >> 3][idx & 7] = B_i8_u4[((int64_t)(col_base + (idx >> 3)) * n_kblk) * 8 + (idx & 7)]; }
    }
    if (tid < N_PAD) {
        sBsum[0][tid] = B_sum[(int64_t)(col_base + tid) * n_kblk];
        sBsc[0][tid]  = B_sc[(int64_t)(col_base + tid) * n_kblk];
    }
    __syncthreads();

    uint2 uG[R_TILES], uU[R_TILES];
    float dG[R_TILES], dU[R_TILES];
#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int rg = rg0 + rt;
        if (rg * 8 < M) {
            const int64_t qoff = ((int64_t) rg * n_kblk) * 32 + lane;
            const int64_t doff = ((int64_t) rg * n_kblk) * 8 + lane_r;
            uG[rt] = G_qs[qoff]; dG[rt] = __half2float(G_d[doff]);
            uU[rt] = U_qs[qoff]; dU[rt] = __half2float(U_d[doff]);
        } else { uG[rt] = make_uint2(0u, 0u); dG[rt] = 0.0f; uU[rt] = make_uint2(0u, 0u); dU[rt] = 0.0f; }
    }

    for (int kb = 0; kb < n_kblk; ++kb) {
        const int cur = SINGLE_BUF ? 0 : (kb & 1);
        const int nxt = SINGLE_BUF ? 0 : (cur ^ 1);
        const bool has_nxt = (kb + 1 < n_kblk);

        uint2 uG_nxt[R_TILES], uU_nxt[R_TILES];
        float dG_nxt[R_TILES], dU_nxt[R_TILES];
        uint4 gB[U4_PER_TH];
        int32_t gBsum = 0; float gBsc = 0.0f;
        if (has_nxt) {
            const int kb1 = kb + 1;
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                const int rg = rg0 + rt;
                if (rg * 8 < M) {
                    const int64_t qoff = ((int64_t) rg * n_kblk + kb1) * 32 + lane;
                    const int64_t doff = ((int64_t) rg * n_kblk + kb1) * 8 + lane_r;
                    uG_nxt[rt] = G_qs[qoff]; dG_nxt[rt] = __half2float(G_d[doff]);
                    uU_nxt[rt] = U_qs[qoff]; dU_nxt[rt] = __half2float(U_d[doff]);
                } else { uG_nxt[rt] = make_uint2(0u, 0u); dG_nxt[rt] = 0.0f; uU_nxt[rt] = make_uint2(0u, 0u); dU_nxt[rt] = 0.0f; }
            }
#pragma unroll
            for (int bi = 0; bi < U4_PER_TH; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_U4) { gB[bi] = B_i8_u4[((int64_t)(col_base + (idx >> 3)) * n_kblk + kb1) * 8 + (idx & 7)]; }
            }
            if (tid < N_PAD) {
                gBsum = B_sum[(int64_t)(col_base + tid) * n_kblk + kb1];
                gBsc  = B_sc[(int64_t)(col_base + tid) * n_kblk + kb1];
            }
        }

#pragma unroll
        for (int nt = 0; nt < N_TILES_W; ++nt) {
            const int b_col = wn_col0 + nt * 8 + lane_r;
            const uint4 bh = sB[cur][b_col][lane_k];
            const uint4 bl = sB[cur][b_col][4 + lane_k];
            const int32_t bs0 = sBsum[cur][wn_col0 + nt * 8 + c0_off];
            const int32_t bs1 = sBsum[cur][wn_col0 + nt * 8 + c1_off];
            const float   bsc0 = sBsc[cur][wn_col0 + nt * 8 + c0_off];
            const float   bsc1 = sBsc[cur][wn_col0 + nt * 8 + c1_off];
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) {
                const uint32_t g0 = (uG[rt].x & 0x03030303u) | ((uG[rt].x & 0x0C0C0C0Cu) << 2);
                const uint32_t g1 = ((uG[rt].x >> 4) & 0x03030303u) | ((uG[rt].x >> 2) & 0x30303030u);
                const uint32_t g2 = (uG[rt].y & 0x03030303u) | ((uG[rt].y & 0x0C0C0C0Cu) << 2);
                const uint32_t g3 = ((uG[rt].y >> 4) & 0x03030303u) | ((uG[rt].y >> 2) & 0x30303030u);
                int32_t dgh[2] = { 0, 0 }, dgl[2] = { 0, 0 };
                pq2_mma_s4_m8n8k32(dgh, g0, bh.x, dgh);
                pq2_mma_s4_m8n8k32(dgh, g1, bh.y, dgh);
                pq2_mma_s4_m8n8k32(dgh, g2, bh.z, dgh);
                pq2_mma_s4_m8n8k32(dgh, g3, bh.w, dgh);
                pq2_mma_s4_m8n8k32(dgl, g0, bl.x, dgl);
                pq2_mma_s4_m8n8k32(dgl, g1, bl.y, dgl);
                pq2_mma_s4_m8n8k32(dgl, g2, bl.z, dgl);
                pq2_mma_s4_m8n8k32(dgl, g3, bl.w, dgl);
                acc_g[rt][nt][0] = fmaf((float)((dgh[0] << 4) + dgl[0] - bs0), dG[rt] * bsc0, acc_g[rt][nt][0]);
                acc_g[rt][nt][1] = fmaf((float)((dgh[1] << 4) + dgl[1] - bs1), dG[rt] * bsc1, acc_g[rt][nt][1]);

                const uint32_t p0 = (uU[rt].x & 0x03030303u) | ((uU[rt].x & 0x0C0C0C0Cu) << 2);
                const uint32_t p1 = ((uU[rt].x >> 4) & 0x03030303u) | ((uU[rt].x >> 2) & 0x30303030u);
                const uint32_t p2 = (uU[rt].y & 0x03030303u) | ((uU[rt].y & 0x0C0C0C0Cu) << 2);
                const uint32_t p3 = ((uU[rt].y >> 4) & 0x03030303u) | ((uU[rt].y >> 2) & 0x30303030u);
                int32_t duh[2] = { 0, 0 }, dul[2] = { 0, 0 };
                pq2_mma_s4_m8n8k32(duh, p0, bh.x, duh);
                pq2_mma_s4_m8n8k32(duh, p1, bh.y, duh);
                pq2_mma_s4_m8n8k32(duh, p2, bh.z, duh);
                pq2_mma_s4_m8n8k32(duh, p3, bh.w, duh);
                pq2_mma_s4_m8n8k32(dul, p0, bl.x, dul);
                pq2_mma_s4_m8n8k32(dul, p1, bl.y, dul);
                pq2_mma_s4_m8n8k32(dul, p2, bl.z, dul);
                pq2_mma_s4_m8n8k32(dul, p3, bl.w, dul);
                acc_u[rt][nt][0] = fmaf((float)((duh[0] << 4) + dul[0] - bs0), dU[rt] * bsc0, acc_u[rt][nt][0]);
                acc_u[rt][nt][1] = fmaf((float)((duh[1] << 4) + dul[1] - bs1), dU[rt] * bsc1, acc_u[rt][nt][1]);
            }
        }
        if (has_nxt) {
            if constexpr (SINGLE_BUF) { __syncthreads(); }
#pragma unroll
            for (int rt = 0; rt < R_TILES; ++rt) { uG[rt] = uG_nxt[rt]; dG[rt] = dG_nxt[rt]; uU[rt] = uU_nxt[rt]; dU[rt] = dU_nxt[rt]; }
#pragma unroll
            for (int bi = 0; bi < U4_PER_TH; ++bi) {
                const int idx = tid + bi * (WARPS * 32);
                if (idx < TOTAL_U4) { sB[nxt][idx >> 3][idx & 7] = gB[bi]; }
            }
            if (tid < N_PAD) { sBsum[nxt][tid] = gBsum; sBsc[nxt][tid] = gBsc; }
            __syncthreads();
        }
    }

#pragma unroll
    for (int rt = 0; rt < R_TILES; ++rt) {
        const int row = (rg0 + rt) * 8 + lane_r;
        if (row < M) {
#pragma unroll
            for (int nt = 0; nt < N_TILES_W; ++nt) {
                const int c0 = col_base + wn_col0 + nt * 8 + c0_off;
                const int c1 = col_base + wn_col0 + nt * 8 + c1_off;
                if (c0 < N) { const float gv0 = acc_g[rt][nt][0]; C[(int64_t) c0 * s_dst + row] = (gv0 / (1.0f + __expf(-gv0))) * acc_u[rt][nt][0]; }
                if (c1 < N) { const float gv1 = acc_g[rt][nt][1]; C[(int64_t) c1 * s_dst + row] = (gv1 / (1.0f + __expf(-gv1))) * acc_u[rt][nt][1]; }
            }
        }
    }
}

static void pq2_narrow_i8_cache_invalidate(cudaStream_t stream);
static void pq2_narrow_i8_cache_invalidate_if_overlap(cudaStream_t stream, const void * dst_data, size_t dst_nbytes);
static void pq2_prefill_i4_cache_invalidate(cudaStream_t stream);
static void pq2_prefill_i4_cache_invalidate_if_overlap(cudaStream_t stream, const void * dst_data, size_t dst_nbytes);

__device__ __forceinline__ void pq2_ldmatrix_x4(uint32_t * regs, const void * smem_ptr) {
#if defined(TURING_MMA_AVAILABLE) || defined(AMPERE_MMA_AVAILABLE)
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
#else
    GGML_UNUSED_VARS(regs, smem_ptr);
#endif
}

// Repack PQ2_0 weights into row-major signed-INT4 [-7,+7] [M, K/2] + per-row F32 scale [M]
__global__ static void repack_pq2_to_s4_rm_kernel(
        const block_pq2_0 * __restrict__ src, uint32_t * __restrict__ dst_qs, float * __restrict__ dst_d,
        const int M, const int K) {
    const int row = blockIdx.x;
    if (row >= M) { return; }
    const int tid    = threadIdx.x; // 0..255
    const int n_kblk = K >> 7;
    const int K8     = K >> 3;
    const block_pq2_0 * blks = src + (int64_t) row * n_kblk;

    float d_max = 0.0f;
    for (int b = tid; b < n_kblk; b += 256) {
        d_max = fmaxf(d_max, fabsf(__half2float(blks[b].d)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        d_max = fmaxf(d_max, __shfl_xor_sync(0xFFFFFFFF, d_max, off));
    }
    __shared__ float smax[8];
    if ((tid & 31) == 0) { smax[tid >> 5] = d_max; }
    __syncthreads();
    if (tid == 0) {
        float m = smax[0];
#pragma unroll
        for (int i = 1; i < 8; ++i) { m = fmaxf(m, smax[i]); }
        smax[0] = m;
        dst_d[row] = m * (1.0f / 7.0f);
    }
    __syncthreads();
    const float inv_d = smax[0] > 0.0f ? (7.0f / smax[0]) : 0.0f;

    uint32_t * out = dst_qs + (int64_t) row * K8;
    for (int i = tid; i < K8; i += 256) {
        const int b  = i >> 4;
        const int kw = i & 15;
        const float bd = __half2float(blks[b].d) * inv_d;
        const uint16_t s = (uint16_t) blks[b].qs[kw * 2] | ((uint16_t) blks[b].qs[kw * 2 + 1] << 8);
        uint32_t w = 0u;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int code = (int)((s >> (j * 2)) & 3u) - 1;
            const int q    = max(-7, min(7, __float2int_rn((float) code * bd)));
            w |= ((uint32_t)(q & 0xF)) << (j * 4);
        }
        out[i] = w;
    }
}

struct pq2_s4_rm_entry {
    uint8_t * qs = nullptr;
    float   * d  = nullptr;
    int64_t   M  = 0;
    int64_t   K  = 0;
};

static pq2_s4_rm_entry get_or_repack_pq2_s4_rm(
        const block_pq2_0 * src0, const int64_t M, const int64_t K, cudaStream_t stream, bool allow_alloc) {
    static std::mutex mu;
    static std::unordered_map<const void *, pq2_s4_rm_entry> cache;
    static std::vector<pq2_s4_rm_entry> retired;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(src0);
    if (it != cache.end() && it->second.M >= M && it->second.K == K) {
        return it->second;
    }
    if (!allow_alloc) { return {}; }
    const size_t need_qs = (size_t) M * (size_t) (K >> 1);
    const size_t need_d  = (size_t) M * sizeof(float);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b < need_qs + need_d + (3584ULL << 20)) {
        return {};
    }
    pq2_s4_rm_entry e;
    e.M = M; e.K = K;
    if (cudaMalloc(&e.qs, need_qs) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    if (cudaMalloc(&e.d, need_d) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFree(e.qs);
        return {};
    }
    repack_pq2_to_s4_rm_kernel<<<(unsigned) M, 256, 0, stream>>>(src0, (uint32_t *) e.qs, e.d, (int) M, (int) K);
    CUDA_CHECK(cudaGetLastError());
    if (it != cache.end()) {
        retired.push_back(it->second);
    }
    cache[src0] = e;
    return e;
}

// 1-pass F32 -> per-column RMS-clipped INT4 [-8, +7] + col_scales (eliminates 4.0-sigma outlier blowup!)
__global__ static void quantize_f32_to_s4_colwise_clipped_kernel(
        const float * __restrict__ src1, uint32_t * __restrict__ dst_i4,
        float * __restrict__ dst_scales,
        const int N, const int N_pad, const int K, const int64_t s11, const float clip_sigma) {
    const int col = blockIdx.x;
    if (col >= N_pad) { return; }
    const int tid = threadIdx.x; // 0..511
    const int K8  = K >> 3;
    uint32_t * out32 = dst_i4 + (int64_t) col * K8;
    if (col >= N) {
        for (int i = tid; i < K8; i += 512) { out32[i] = 0u; }
        if (tid == 0) { dst_scales[col] = 0.0f; }
        return;
    }
    const float4 * in4 = (const float4 *) (src1 + (int64_t) col * s11);
    float bmax = 0.0f, sq = 0.0f;
    for (int i = tid; i < (K >> 2); i += 512) {
        const float4 v = in4[i];
        bmax = fmaxf(bmax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        sq  += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFF, bmax, off));
        sq  += __shfl_xor_sync(0xFFFFFFFF, sq,   off);
    }
    __shared__ float smax[16], ssq[16];
    if ((tid & 31) == 0) { smax[tid >> 5] = bmax; ssq[tid >> 5] = sq; }
    __syncthreads();
    if (tid == 0) {
        float m = smax[0], s = ssq[0];
#pragma unroll
        for (int i = 1; i < 16; ++i) { m = fmaxf(m, smax[i]); s += ssq[i]; }
        const float rms = sqrtf(s / (float) K);
        const float c   = (clip_sigma > 0.0f) ? fminf(m, clip_sigma * rms) : m;
        smax[0] = c > 0.0f ? (7.0f / c) : 0.0f;
        dst_scales[col] = c * (1.0f / 7.0f);
    }
    __syncthreads();
    const float inv_scale = smax[0];
    for (int i = tid; i < K8; i += 512) {
        const float4 a = in4[2 * i + 0];
        const float4 b = in4[2 * i + 1];
        uint32_t w = 0u;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(a.x * inv_scale))) & 0xF)) << 0;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(a.y * inv_scale))) & 0xF)) << 4;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(a.z * inv_scale))) & 0xF)) << 8;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(a.w * inv_scale))) & 0xF)) << 12;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(b.x * inv_scale))) & 0xF)) << 16;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(b.y * inv_scale))) & 0xF)) << 20;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(b.z * inv_scale))) & 0xF)) << 24;
        w |= ((uint32_t)(max(-8, min(7, __float2int_rn(b.w * inv_scale))) & 0xF)) << 28;
        out32[i] = w;
    }
}

// 217.5 TOPS zero-spill Turing sm_75 INT4 GEMM (128x128x256, 64KB dynamic smem, dblbuf frags, L2 swizzle, coalesced SMEM epilogue!)
// EPI_MODE: 0 = F32 (+ optional add_src), 1 = write FP16 gate (dst_f16), 2 = read FP16 gate (gate_f16) -> silu(g)*up -> F32 dst
template <int EPI_MODE = 0>
__global__ static void __launch_bounds__(256, 1)
pq2_prefill_s4_128x128x256_ldm_kernel(
        const uint8_t * __restrict__ A,
        const float   * __restrict__ row_scales,
        const uint8_t * __restrict__ B,
        const float   * __restrict__ col_scales,
        const void    * __restrict__ aux_src,
        void          * __restrict__ dst_out,
        const int M, const int N, const int N_pad, const int K, const int64_t s_dst) {
    constexpr int BM = 128, BN = 128, BK = 256, BK_B = 128;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t * smA0 = smem;
    uint8_t * smA1 = smem + BM * BK_B;
    uint8_t * smB0 = smem + 2 * BM * BK_B;
    uint8_t * smB1 = smem + 2 * BM * BK_B + BN * BK_B;

    const int grid_m = M / BM;
    const int grid_n = N_pad / BN;
    int bm, bn;
    if (((grid_m & 3) == 0) && ((grid_n & 3) == 0)) {
        const int lin = blockIdx.x + blockIdx.y * grid_m;
        const int super_idx = lin >> 4;
        const int local = lin & 15;
        const int n_sm = grid_m >> 2;
        bm = ((super_idx % n_sm) * 4 + (local & 3)) * BM;
        bn = ((super_idx / n_sm) * 4 + (local >> 2)) * BN;
    } else {
        const int group_m = 8;
        const int lin = blockIdx.x + blockIdx.y * grid_m;
        const int blocks_per_group = group_m * grid_n;
        const int gid = lin / blocks_per_group;
        const int first_m = gid * group_m;
        const int gm_size = min(group_m, grid_m - first_m);
        const int rem = lin % blocks_per_group;
        bm = (first_m + rem % gm_size) * BM;
        bn = (rem / gm_size) * BN;
    }

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = (warp >> 1) * 32;
    const int wn   = (warp & 1)  * 64;

    const int K_B      = K >> 1;
    const int n_ktiles = K / BK;

    const int ld_c  = tid & 7;
    const int ld_r  = tid >> 3; // 0..31
    const int sw_c0 = ld_c ^ ((ld_r +  0) & 7);
    const int sw_c1 = ld_c ^ ((ld_r + 32) & 7);
    const int sw_c2 = ld_c ^ ((ld_r + 64) & 7);
    const int sw_c3 = ld_c ^ ((ld_r + 96) & 7);

    int32_t acc[4][8][2] = {};
    const int ldm_row_a  = wm + lane;
    const int ldm_row_b0 = wn + lane;
    const int ldm_row_b1 = wn + 32 + lane;

    {
        const int k_off = ld_c * 16;
        *(int4 *) (smA0 + (ld_r +  0) * BK_B + sw_c0 * 16) = *(const int4 *) (A + (int64_t) (bm + ld_r +  0) * K_B + k_off);
        *(int4 *) (smA0 + (ld_r + 32) * BK_B + sw_c1 * 16) = *(const int4 *) (A + (int64_t) (bm + ld_r + 32) * K_B + k_off);
        *(int4 *) (smA0 + (ld_r + 64) * BK_B + sw_c2 * 16) = *(const int4 *) (A + (int64_t) (bm + ld_r + 64) * K_B + k_off);
        *(int4 *) (smA0 + (ld_r + 96) * BK_B + sw_c3 * 16) = *(const int4 *) (A + (int64_t) (bm + ld_r + 96) * K_B + k_off);
        *(int4 *) (smB0 + (ld_r +  0) * BK_B + sw_c0 * 16) = *(const int4 *) (B + (int64_t) (bn + ld_r +  0) * K_B + k_off);
        *(int4 *) (smB0 + (ld_r + 32) * BK_B + sw_c1 * 16) = *(const int4 *) (B + (int64_t) (bn + ld_r + 32) * K_B + k_off);
        *(int4 *) (smB0 + (ld_r + 64) * BK_B + sw_c2 * 16) = *(const int4 *) (B + (int64_t) (bn + ld_r + 64) * K_B + k_off);
        *(int4 *) (smB0 + (ld_r + 96) * BK_B + sw_c3 * 16) = *(const int4 *) (B + (int64_t) (bn + ld_r + 96) * K_B + k_off);
    }
    __syncthreads();

    uint32_t a_frag[2][4];
    uint32_t b_frag[2][8];

    for (int kt = 0; kt < n_ktiles; ++kt) {
        uint8_t * sAc = (kt & 1) ? smA1 : smA0;
        uint8_t * sBc = (kt & 1) ? smB1 : smB0;
        uint8_t * sAn = (kt & 1) ? smA0 : smA1;
        uint8_t * sBn = (kt & 1) ? smB0 : smB1;

        pq2_ldmatrix_x4(a_frag[0],     sAc + ldm_row_a  * BK_B + (0 ^ (ldm_row_a  & 7)) * 16);
        pq2_ldmatrix_x4(b_frag[0] + 0, sBc + ldm_row_b0 * BK_B + (0 ^ (ldm_row_b0 & 7)) * 16);
        pq2_ldmatrix_x4(b_frag[0] + 4, sBc + ldm_row_b1 * BK_B + (0 ^ (ldm_row_b1 & 7)) * 16);

        const bool has_nxt = (kt + 1 < n_ktiles);
        const int k_off = (kt + 1) * BK_B + ld_c * 16;
        int4 gA0, gA1, gA2, gA3, gB0, gB1, gB2, gB3;
        if (has_nxt) {
            gA0 = *(const int4 *) (A + (int64_t) (bm + ld_r +  0) * K_B + k_off);
            gA1 = *(const int4 *) (A + (int64_t) (bm + ld_r + 32) * K_B + k_off);
            gA2 = *(const int4 *) (A + (int64_t) (bm + ld_r + 64) * K_B + k_off);
            gA3 = *(const int4 *) (A + (int64_t) (bm + ld_r + 96) * K_B + k_off);
        }

#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int fc = ks & 1;
            const int fn = fc ^ 1;
            if (ks == 4 && has_nxt) {
                *(int4 *) (sAn + (ld_r +  0) * BK_B + sw_c0 * 16) = gA0;
                *(int4 *) (sAn + (ld_r + 32) * BK_B + sw_c1 * 16) = gA1;
                *(int4 *) (sAn + (ld_r + 64) * BK_B + sw_c2 * 16) = gA2;
                *(int4 *) (sAn + (ld_r + 96) * BK_B + sw_c3 * 16) = gA3;
                gB0 = *(const int4 *) (B + (int64_t) (bn + ld_r +  0) * K_B + k_off);
                gB1 = *(const int4 *) (B + (int64_t) (bn + ld_r + 32) * K_B + k_off);
                gB2 = *(const int4 *) (B + (int64_t) (bn + ld_r + 64) * K_B + k_off);
                gB3 = *(const int4 *) (B + (int64_t) (bn + ld_r + 96) * K_B + k_off);
            }
            if (ks + 1 < 8) {
                pq2_ldmatrix_x4(a_frag[fn],     sAc + ldm_row_a  * BK_B + ((ks + 1) ^ (ldm_row_a  & 7)) * 16);
                pq2_ldmatrix_x4(b_frag[fn] + 0, sBc + ldm_row_b0 * BK_B + ((ks + 1) ^ (ldm_row_b0 & 7)) * 16);
                pq2_ldmatrix_x4(b_frag[fn] + 4, sBc + ldm_row_b1 * BK_B + ((ks + 1) ^ (ldm_row_b1 & 7)) * 16);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
                for (int ni = 0; ni < 8; ++ni) {
                    pq2_mma_s4_m8n8k32(acc[mi][ni], a_frag[fc][mi], b_frag[fc][ni], acc[mi][ni]);
                }
            }
        }

        if (has_nxt) {
            *(int4 *) (sBn + (ld_r +  0) * BK_B + sw_c0 * 16) = gB0;
            *(int4 *) (sBn + (ld_r + 32) * BK_B + sw_c1 * 16) = gB1;
            *(int4 *) (sBn + (ld_r + 64) * BK_B + sw_c2 * 16) = gB2;
            *(int4 *) (sBn + (ld_r + 96) * BK_B + sw_c3 * 16) = gB3;
            __syncthreads();
        }
    }

    __syncthreads();
    int32_t * smC = (int32_t *) smA0;
    int32_t * wC  = smC + warp * (8 * 36);
    const int c_row     = lane >> 2;
    const int c_col     = (lane & 3) * 2;
    const int col_local = lane >> 3;
    const int row4_off  = (lane & 7) * 4;
    const float4 rs4    = *(const float4 *) (row_scales + bm + wm + row4_off);
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            wC[(c_col + 0) * 36 + mi * 8 + c_row] = acc[mi][ni][0];
            wC[(c_col + 1) * 36 + mi * 8 + c_row] = acc[mi][ni][1];
        }
        __syncwarp();
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int col = bn + wn + ni * 8 + pass * 4 + col_local;
            if (col < N) {
                const float cs = __ldg(col_scales + col);
                const int4  q4 = *(const int4 *) (wC + (pass * 4 + col_local) * 36 + row4_off);
                float4 v4;
                v4.x = (float) q4.x * (rs4.x * cs);
                v4.y = (float) q4.y * (rs4.y * cs);
                v4.z = (float) q4.z * (rs4.z * cs);
                v4.w = (float) q4.w * (rs4.w * cs);
                const int64_t idx = (int64_t) col * s_dst + bm + wm + row4_off;
                if constexpr (EPI_MODE == 1) {
                    const half2 h0 = __floats2half2_rn(v4.x, v4.y);
                    const half2 h1 = __floats2half2_rn(v4.z, v4.w);
                    uint2 u;
                    u.x = *(const uint32_t *) &h0;
                    u.y = *(const uint32_t *) &h1;
                    *(uint2 *) (((half *) dst_out) + idx) = u;
                } else if constexpr (EPI_MODE == 2 || EPI_MODE == 3) {
                    const uint2 u = *(const uint2 *) (((const half *) aux_src) + idx);
                    const half2 h0 = *(const half2 *) &u.x;
                    const half2 h1 = *(const half2 *) &u.y;
                    const float2 g01 = __half22float2(h0);
                    const float2 g23 = __half22float2(h1);
                    v4.x *= __fdividef(g01.x, 1.0f + __expf(-g01.x));
                    v4.y *= __fdividef(g01.y, 1.0f + __expf(-g01.y));
                    v4.z *= __fdividef(g23.x, 1.0f + __expf(-g23.x));
                    v4.w *= __fdividef(g23.y, 1.0f + __expf(-g23.y));
                    if constexpr (EPI_MODE == 3) {
                        const half2 o0 = __floats2half2_rn(v4.x, v4.y);
                        const half2 o1 = __floats2half2_rn(v4.z, v4.w);
                        uint2 ou;
                        ou.x = *(const uint32_t *) &o0;
                        ou.y = *(const uint32_t *) &o1;
                        *(uint2 *) (((half *) dst_out) + idx) = ou;
                    } else {
                        *(float4 *) (((float *) dst_out) + idx) = v4;
                    }
                } else {
                    if (aux_src) {
                        const float4 a4 = *(const float4 *) (((const float *) aux_src) + idx);
                        v4.x += a4.x; v4.y += a4.y; v4.z += a4.z; v4.w += a4.w;
                    }
                    *(float4 *) (((float *) dst_out) + idx) = v4;
                }
            }
        }
        __syncwarp();
    }
}

template <int EPI_MODE>
static void launch_pq2_prefill_s4_128x128x256(
        dim3 grid, cudaStream_t stream,
        const uint8_t * A, const float * rsc, const uint8_t * B, const float * csc,
        const void * aux_src, void * dst_out, int M, int N, int N_pad, int K, int64_t s_dst) {
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(
            pq2_prefill_s4_128x128x256_ldm_kernel<EPI_MODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
        attr_set = true;
    }
    pq2_prefill_s4_128x128x256_ldm_kernel<EPI_MODE><<<grid, 256, 65536, stream>>>(
        A, rsc, B, csc, aux_src, dst_out, M, N, N_pad, K, s_dst);
}

struct pq2_prefill_i4_scratch {
    uint32_t   * B_i4        = nullptr;
    float      * B_scales    = nullptr;
    half       * gate_f16    = nullptr;
    half       * mul_w_f16   = nullptr;
    size_t       cap_b       = 0;
    size_t       cap_sc      = 0;
    size_t       cap_gate    = 0;
    size_t       cap_mw_f16  = 0;
    const void * src1_data   = nullptr;
    size_t       src1_nbytes = 0;
    int64_t      N           = 0;
    int64_t      K           = 0;
    int64_t      s11         = 0;
    uint64_t     gen         = 0;
    bool         valid       = false;
    const void * mw_data     = nullptr;
    size_t       mw_nbytes   = 0;
    int64_t      mw_N        = 0;
    int64_t      mw_K        = 0;
    bool         mw_valid    = false;
};

static pq2_prefill_i4_scratch & pq2_prefill_i4_scratch_for_stream(cudaStream_t stream) {
    static std::mutex mu;
    static std::unordered_map<cudaStream_t, pq2_prefill_i4_scratch> map;
    std::lock_guard<std::mutex> lock(mu);
    return map[stream];
}

static bool ggml_cuda_mul_mat_pq2_prefill_int4(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * add_src = nullptr) {
    if (!ggml_cuda_pq2_prefill_int4_enabled()) { return false; }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) { return false; }
    if (src0->type != GGML_TYPE_PQ2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) { return false; }
    if (add_src && (add_src->type != GGML_TYPE_F32 || !ggml_are_same_shape(add_src, dst) ||
            add_src->nb[0] != sizeof(float) || add_src->nb[1] != dst->nb[1])) { return false; }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (N < 256 || M > 65536 || (K % 256) != 0 || (M % 128) != 0 ||
            !ggml_is_contiguous(src0) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    (void) cudaStreamIsCapturing(stream, &cap_status);
    const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);

    auto ae = get_or_repack_pq2_s4_rm((const block_pq2_0 *) src0->data, M, K, stream, allow_alloc);
    ggml_cuda_pool_alloc<uint8_t> tmp_qs(ctx.pool());
    ggml_cuda_pool_alloc<float>   tmp_d(ctx.pool());
    if (!ae.qs || !ae.d) {
        tmp_qs.alloc((size_t) M * (size_t) (K >> 1));
        tmp_d.alloc((size_t) M);
        repack_pq2_to_s4_rm_kernel<<<(unsigned) M, 256, 0, stream>>>(
            (const block_pq2_0 *) src0->data, (uint32_t *) tmp_qs.get(), tmp_d.get(), (int) M, (int) K);
        ae.qs = tmp_qs.get();
        ae.d  = tmp_d.get();
    }

    const int64_t N_pad = ((N + 127) / 128) * 128;
    const int64_t s11   = src1->nb[1] / sizeof(float);
    const int64_t s_dst = dst->nb[1] / sizeof(float);
    auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
    const size_t need_b  = (size_t) N_pad * (size_t) (K >> 3) * sizeof(uint32_t);
    const size_t need_sc = (size_t) N_pad * sizeof(float);
    if (sc.cap_b < need_b || sc.cap_sc < need_sc) {
        if (!allow_alloc) { return false; }
        if (sc.B_i4)     { CUDA_CHECK(cudaFree(sc.B_i4)); }
        if (sc.B_scales) { CUDA_CHECK(cudaFree(sc.B_scales)); }
        CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));
        CUDA_CHECK(cudaMalloc(&sc.B_scales, need_sc));
        sc.cap_b = need_b; sc.cap_sc = need_sc;
        sc.valid = false;
    }
    if (!(sc.valid && sc.src1_data == src1->data && sc.N == N && sc.K == K && sc.s11 == s11)) {
        quantize_f32_to_s4_colwise_clipped_kernel<<<(unsigned) N_pad, 512, 0, stream>>>(
            (const float *) src1->data, sc.B_i4, sc.B_scales,
            (int) N, (int) N_pad, (int) K, s11, ggml_cuda_pq2_prefill_int4_clip());
        sc.src1_data   = src1->data;
        sc.src1_nbytes = ggml_nbytes(src1);
        sc.N = N; sc.K = K; sc.s11 = s11; sc.valid = true;
    }
    dim3 grid((unsigned) (M / 128), (unsigned) (N_pad / 128));
    launch_pq2_prefill_s4_128x128x256<0>(
        grid, stream, ae.qs, ae.d, (const uint8_t *) sc.B_i4, sc.B_scales,
        add_src ? (const void *) add_src->data : nullptr,
        (void *) dst->data, (int) M, (int) N, (int) N_pad, (int) K, s_dst);
    pq2_prefill_i4_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// 1-pass ZERO-SMEM half2 (128B/warp LDG.32 + 32B sector STG.32) warp-parallel MUL(signs) + FWHT(1024) -> RMS-clipped INT4 B_s4!
__global__ static void __launch_bounds__(576, 1)
mul_signs_fwht1024_to_s4_f16_kernel(
        const half * __restrict__ src_f16, const float * __restrict__ signs,
        uint32_t * __restrict__ dst_s4, float * __restrict__ out_col_scales,
        const int M, const int N, const int N_pad, const int64_t s11, const float clip_sigma) {
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int n_blk = M >> 10;
    uint32_t * out32 = dst_s4 + (int64_t) j * (M >> 3);
    if (j >= N) {
        if (tid == 0) { out_col_scales[j] = 0.0f; }
        for (int i = tid; i < (M >> 3); i += blockDim.x) { out32[i] = 0u; }
        return;
    }
    const half2  * in_col2 = (const half2  *) (src_f16 + (int64_t) j * s11);
    const float2 * signs2  = (const float2 *) signs;
    float v0[16], v1[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int idx2 = warp * 512 + k * 32 + lane;
        const float2 xf = __half22float2(in_col2[idx2]);
        const float2 sf = __ldg(signs2 + idx2);
        const float x0 = xf.x * sf.x;
        const float x1 = xf.y * sf.y;
        // Stage 0 (bit 0, stride 1) in-register!
        v0[k] = x0 + x1;
        v1[k] = x0 - x1;
    }
    // Stages 6..9 (bits 6..9, across k = 0..15) in-register!
#pragma unroll
    for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 16; k += (h << 1)) {
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a0 = v0[k + r], b0 = v0[k + r + h];
                v0[k + r] = a0 + b0; v0[k + r + h] = a0 - b0;
                const float a1 = v1[k + r], b1 = v1[k + r + h];
                v1[k + r] = a1 + b1; v1[k + r + h] = a1 - b1;
            }
        }
    }
    // Stages 1..5 (bits 1..5, across lane = 0..31) via warp shuffle!
    float cmax = 0.0f, csq = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        float x0 = v0[k], x1 = v1[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float y0 = __shfl_xor_sync(0xFFFFFFFF, x0, h);
            const float y1 = __shfl_xor_sync(0xFFFFFFFF, x1, h);
            x0 = ((lane & h) == 0) ? (x0 + y0) : (y0 - x0);
            x1 = ((lane & h) == 0) ? (x1 + y1) : (y1 - x1);
        }
        v0[k] = x0; v1[k] = x1;
        cmax = fmaxf(cmax, fmaxf(fabsf(x0), fabsf(x1)));
        csq += x0 * x0 + x1 * x1;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
        csq += __shfl_xor_sync(0xFFFFFFFF, csq,  off);
    }
    __shared__ float smax[24], ssq[24];
    if (lane == 0) { smax[warp] = cmax; ssq[warp] = csq; }
    __syncthreads();
    float max_v = 0.0f, tot_sq = 0.0f;
    for (int w = 0; w < n_blk; ++w) {
        max_v  = fmaxf(max_v, smax[w]);
        tot_sq += ssq[w];
    }
    const float rms = sqrtf(tot_sq / (float) M);
    const float c   = (clip_sigma > 0.0f) ? fminf(max_v, clip_sigma * rms) : max_v;
    const float inv_scale = c > 0.0f ? (7.0f / c) : 0.0f;
    if (tid == 0) { out_col_scales[j] = (c * (1.0f / 7.0f)) * (1.0f / 32.0f); }
    uint32_t * out_blk = out32 + warp * 128;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        uint32_t q = ((uint32_t)(max(-8, min(7, __float2int_rn(v0[k] * inv_scale))) & 0xF))
                   | (((uint32_t)(max(-8, min(7, __float2int_rn(v1[k] * inv_scale))) & 0xF)) << 4);
        q |= __shfl_down_sync(0xFFFFFFFF, q, 1) << 8;
        q |= __shfl_down_sync(0xFFFFFFFF, q, 2) << 16;
        if ((lane & 3) == 0) { out_blk[k * 8 + (lane >> 2)] = q; }
    }
}

static bool ggml_cuda_mul_mat_pq2_prefill_int4_swiglu(
        ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
        const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * fwht_signs = nullptr, const ggml_tensor * fwht_out = nullptr, int fwht_H = 0) {
    if (!ggml_cuda_pq2_prefill_int4_enabled()) { return false; }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) { return false; }
    if (gate_w->type != GGML_TYPE_PQ2_0 || up_w->type != GGML_TYPE_PQ2_0 ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    if (!ggml_are_same_shape(gate_w, up_w) || gate_w->ne[2] != 1 || gate_w->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1) { return false; }
    const int64_t K = gate_w->ne[0], M = gate_w->ne[1], N = src1->ne[1];
    if (N < 256 || M > 65536 || (K % 256) != 0 || (M % 128) != 0 ||
            !ggml_is_contiguous(gate_w) || !ggml_is_contiguous(up_w) ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            ggml_backend_buffer_get_usage(gate_w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    const bool fuse_fwht_s4 = (fwht_signs != nullptr && fwht_out != nullptr && fwht_H == 1024 &&
                               (M % 1024) == 0 && (M >> 10) <= 24);
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    (void) cudaStreamIsCapturing(stream, &cap_status);
    const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);

    auto ge = get_or_repack_pq2_s4_rm((const block_pq2_0 *) gate_w->data, M, K, stream, allow_alloc);
    auto ue = get_or_repack_pq2_s4_rm((const block_pq2_0 *) up_w->data,   M, K, stream, allow_alloc);
    ggml_cuda_pool_alloc<uint8_t> tmp_g_qs(ctx.pool());
    ggml_cuda_pool_alloc<float>   tmp_g_d(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> tmp_u_qs(ctx.pool());
    ggml_cuda_pool_alloc<float>   tmp_u_d(ctx.pool());
    if (!ge.qs || !ge.d) {
        tmp_g_qs.alloc((size_t) M * (size_t) (K >> 1));
        tmp_g_d.alloc((size_t) M);
        repack_pq2_to_s4_rm_kernel<<<(unsigned) M, 256, 0, stream>>>(
            (const block_pq2_0 *) gate_w->data, (uint32_t *) tmp_g_qs.get(), tmp_g_d.get(), (int) M, (int) K);
        ge.qs = tmp_g_qs.get(); ge.d = tmp_g_d.get();
    }
    if (!ue.qs || !ue.d) {
        tmp_u_qs.alloc((size_t) M * (size_t) (K >> 1));
        tmp_u_d.alloc((size_t) M);
        repack_pq2_to_s4_rm_kernel<<<(unsigned) M, 256, 0, stream>>>(
            (const block_pq2_0 *) up_w->data, (uint32_t *) tmp_u_qs.get(), tmp_u_d.get(), (int) M, (int) K);
        ue.qs = tmp_u_qs.get(); ue.d = tmp_u_d.get();
    }

    const int64_t N_pad = ((N + 127) / 128) * 128;
    const int64_t s11   = src1->nb[1] / sizeof(float);
    const int64_t s_dst = dst->nb[1] / sizeof(float);
    auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
    const size_t need_b  = (size_t) N_pad * (size_t) (K >> 3) * sizeof(uint32_t);
    const size_t need_sc = (size_t) N_pad * sizeof(float);
    if (sc.cap_b < need_b || sc.cap_sc < need_sc) {
        if (!allow_alloc) { return false; }
        if (sc.B_i4)     { CUDA_CHECK(cudaFree(sc.B_i4)); }
        if (sc.B_scales) { CUDA_CHECK(cudaFree(sc.B_scales)); }
        CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));
        CUDA_CHECK(cudaMalloc(&sc.B_scales, need_sc));
        sc.cap_b = need_b; sc.cap_sc = need_sc;
        sc.valid = false;
    }
    if (!(sc.valid && sc.src1_data == src1->data && sc.N == N && sc.K == K && sc.s11 == s11)) {
        quantize_f32_to_s4_colwise_clipped_kernel<<<(unsigned) N_pad, 512, 0, stream>>>(
            (const float *) src1->data, sc.B_i4, sc.B_scales,
            (int) N, (int) N_pad, (int) K, s11, ggml_cuda_pq2_prefill_int4_clip());
        sc.src1_data   = src1->data;
        sc.src1_nbytes = ggml_nbytes(src1);
        sc.N = N; sc.K = K; sc.s11 = s11; sc.valid = true;
    }
    const size_t need_gate = (size_t) N * (size_t) s_dst * sizeof(half);
    if (sc.cap_gate < need_gate) {
        if (!allow_alloc) { return false; }
        if (sc.gate_f16) { CUDA_CHECK(cudaFree(sc.gate_f16)); }
        CUDA_CHECK(cudaMalloc(&sc.gate_f16, need_gate));
        sc.cap_gate = need_gate;
    }
    dim3 grid((unsigned) (M / 128), (unsigned) (N_pad / 128));
    launch_pq2_prefill_s4_128x128x256<1>(
        grid, stream, ge.qs, ge.d, (const uint8_t *) sc.B_i4, sc.B_scales,
        nullptr, (void *) sc.gate_f16, (int) M, (int) N, (int) N_pad, (int) K, s_dst);
    if (fuse_fwht_s4) {
        // EPI_MODE == 3: write SwiGLU(gate, up) as FP16 in-place into sc.gate_f16 (2x less DRAM write!), then FWHT(1024)->B_s4!
        launch_pq2_prefill_s4_128x128x256<3>(
            grid, stream, ue.qs, ue.d, (const uint8_t *) sc.B_i4, sc.B_scales,
            (const void *) sc.gate_f16, (void *) sc.gate_f16, (int) M, (int) N, (int) N_pad, (int) K, s_dst);
        const size_t need_b_out = (size_t) N_pad * (size_t) (M >> 3) * sizeof(uint32_t);
        if (sc.cap_b < need_b_out) {
            if (!allow_alloc) { return false; }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (sc.B_i4) { CUDA_CHECK(cudaFree(sc.B_i4)); }
            CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b_out));
            sc.cap_b = need_b_out;
        }
        mul_signs_fwht1024_to_s4_f16_kernel<<<(unsigned) N_pad, (unsigned) ((M >> 10) * 32), 0, stream>>>(
            sc.gate_f16, (const float *) fwht_signs->data, sc.B_i4, sc.B_scales,
            (int) M, (int) N, (int) N_pad, s_dst, ggml_cuda_pq2_prefill_int4_clip());
        sc.src1_data   = fwht_out->data;
        sc.src1_nbytes = ggml_nbytes(fwht_out);
        sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true;
    } else {
        launch_pq2_prefill_s4_128x128x256<2>(
            grid, stream, ue.qs, ue.d, (const uint8_t *) sc.B_i4, sc.B_scales,
            (const void *) sc.gate_f16, (void *) dst->data, (int) M, (int) N, (int) N_pad, (int) K, s_dst);
        pq2_prefill_i4_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// 1-pass ZERO-SMEM float2 (LDG.64 + 32B full-sector STG.32) warp-parallel [ADD +] RMS_NORM + MUL(w) + MUL(signs) + FWHT(1024) -> B_s4!
__global__ static void rms_mul_signs_fwht1024_to_s4_kernel(
        const float * __restrict__ src, const float * __restrict__ opt_add_b, float * __restrict__ opt_add_out,
        const float * __restrict__ rms_w, const float eps,
        const float * __restrict__ signs, float * __restrict__ opt_mul_w_out, half * __restrict__ opt_mul_w_f16,
        uint32_t * __restrict__ dst_s4, float * __restrict__ out_col_scales,
        const int M, const int N, const int N_pad, const int64_t s11, const float clip_sigma) {
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int n_blk = M >> 10;
    uint32_t * out32 = dst_s4 + (int64_t) j * (M >> 3);
    if (j >= N) {
        if (tid == 0) { out_col_scales[j] = 0.0f; }
        for (int i = tid; i < (M >> 3); i += blockDim.x) { out32[i] = 0u; }
        return;
    }
    const float2 * in_col2  = (const float2 *) (src + (int64_t) j * s11);
    const float2 * add_b2   = opt_add_b   ? (const float2 *) (opt_add_b   + (int64_t) j * M) : nullptr;
    float2       * add_out2 = opt_add_out ? (float2       *) (opt_add_out + (int64_t) j * M) : nullptr;
    const float2 * rw2      = (const float2 *) rms_w;
    const float2 * sgn2     = (const float2 *) signs;
    float v0[16], v1[16];
    float sq_in = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int idx2 = warp * 512 + k * 32 + lane;
        float2 x = in_col2[idx2];
        if (add_b2) {
            const float2 b = add_b2[idx2];
            x.x += b.x; x.y += b.y;
            if (add_out2) { add_out2[idx2] = x; }
        }
        v0[k] = x.x; v1[k] = x.y;
        sq_in += x.x * x.x + x.y * x.y;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { sq_in += __shfl_xor_sync(0xFFFFFFFF, sq_in, off); }
    __shared__ float smax[24], ssq[24];
    if (lane == 0) { ssq[warp] = sq_in; }
    __syncthreads();
    float tot_in_sq = 0.0f;
    for (int w = 0; w < n_blk; ++w) { tot_in_sq += ssq[w]; }
    const float irms = rsqrtf(tot_in_sq / (float) M + eps);
    float2 * mw_col2     = opt_mul_w_out ? (float2 *) (opt_mul_w_out + (int64_t) j * M) : nullptr;
    half2  * mw_f16_col2 = opt_mul_w_f16 ? (half2  *) (opt_mul_w_f16 + (int64_t) j * M) : nullptr;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int idx2 = warp * 512 + k * 32 + lane;
        const float2 w = __ldg(rw2 + idx2);
        const float2 s = __ldg(sgn2 + idx2);
        const float mw0 = v0[k] * irms * w.x;
        const float mw1 = v1[k] * irms * w.y;
        if (mw_col2)     { mw_col2[idx2]     = make_float2(mw0, mw1); }
        if (mw_f16_col2) { mw_f16_col2[idx2] = __floats2half2_rn(mw0, mw1); }
        const float x0 = mw0 * s.x;
        const float x1 = mw1 * s.y;
        v0[k] = x0 + x1;
        v1[k] = x0 - x1;
    }
#pragma unroll
    for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 16; k += (h << 1)) {
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a0 = v0[k + r], b0 = v0[k + r + h];
                v0[k + r] = a0 + b0; v0[k + r + h] = a0 - b0;
                const float a1 = v1[k + r], b1 = v1[k + r + h];
                v1[k + r] = a1 + b1; v1[k + r + h] = a1 - b1;
            }
        }
    }
    float cmax = 0.0f, csq = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        float x0 = v0[k], x1 = v1[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float y0 = __shfl_xor_sync(0xFFFFFFFF, x0, h);
            const float y1 = __shfl_xor_sync(0xFFFFFFFF, x1, h);
            x0 = ((lane & h) == 0) ? (x0 + y0) : (y0 - x0);
            x1 = ((lane & h) == 0) ? (x1 + y1) : (y1 - x1);
        }
        v0[k] = x0; v1[k] = x1;
        cmax = fmaxf(cmax, fmaxf(fabsf(x0), fabsf(x1)));
        csq += x0 * x0 + x1 * x1;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
        csq += __shfl_xor_sync(0xFFFFFFFF, csq,  off);
    }
    __syncthreads();
    if (lane == 0) { smax[warp] = cmax; ssq[warp] = csq; }
    __syncthreads();
    float max_v = 0.0f, tot_sq = 0.0f;
    for (int w = 0; w < n_blk; ++w) {
        max_v  = fmaxf(max_v, smax[w]);
        tot_sq += ssq[w];
    }
    const float rms = sqrtf(tot_sq / (float) M);
    const float c   = (clip_sigma > 0.0f) ? fminf(max_v, clip_sigma * rms) : max_v;
    const float inv_scale = c > 0.0f ? (7.0f / c) : 0.0f;
    if (tid == 0) { out_col_scales[j] = (c * (1.0f / 7.0f)) * (1.0f / 32.0f); }
    uint32_t * out_blk = out32 + warp * 128;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        uint32_t q = ((uint32_t)(max(-8, min(7, __float2int_rn(v0[k] * inv_scale))) & 0xF))
                   | (((uint32_t)(max(-8, min(7, __float2int_rn(v1[k] * inv_scale))) & 0xF)) << 4);
        q |= __shfl_down_sync(0xFFFFFFFF, q, 1) << 8;
        q |= __shfl_down_sync(0xFFFFFFFF, q, 2) << 16;
        if ((lane & 3) == 0) { out_blk[k * 8 + (lane >> 2)] = q; }
    }
}

// 1-pass ZERO-SMEM warp-parallel MUL(signs) + FWHT(1024) -> RMS-clipped INT4 B_s4 (1 warp per 1024-block, M <= 18432!)
__global__ static void mul_signs_fwht1024_to_s4_kernel(
        const float * __restrict__ src, const float * __restrict__ signs,
        uint32_t * __restrict__ dst_s4, float * __restrict__ out_col_scales,
        const int M, const int N, const int N_pad, const int64_t s11, const float clip_sigma) {
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int n_blk = M >> 10;
    uint32_t * out32 = dst_s4 + (int64_t) j * (M >> 3);
    if (j >= N) {
        if (tid == 0) { out_col_scales[j] = 0.0f; }
        for (int i = tid; i < (M >> 3); i += blockDim.x) { out32[i] = 0u; }
        return;
    }
    const float * in_col = src + (int64_t) j * s11;
    float v[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int idx = warp * 1024 + k * 32 + lane;
        v[k] = in_col[idx] * signs[idx];
    }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 32; k += (h << 1)) {
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a = v[k + r], b = v[k + r + h];
                v[k + r] = a + b; v[k + r + h] = a - b;
            }
        }
    }
    float cmax = 0.0f, csq = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x = v[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float x2 = __shfl_xor_sync(0xFFFFFFFF, x, h);
            x = ((lane & h) == 0) ? (x + x2) : (x2 - x);
        }
        v[k] = x;
        cmax = fmaxf(cmax, fabsf(x));
        csq += x * x;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
        csq += __shfl_xor_sync(0xFFFFFFFF, csq,  off);
    }
    __shared__ float smax[24], ssq[24];
    if (lane == 0) { smax[warp] = cmax; ssq[warp] = csq; }
    __syncthreads();
    float max_v = 0.0f, tot_sq = 0.0f;
    for (int w = 0; w < n_blk; ++w) {
        max_v  = fmaxf(max_v, smax[w]);
        tot_sq += ssq[w];
    }
    const float rms = sqrtf(tot_sq / (float) M);
    const float c   = (clip_sigma > 0.0f) ? fminf(max_v, clip_sigma * rms) : max_v;
    const float inv_scale = c > 0.0f ? (7.0f / c) : 0.0f;
    if (tid == 0) { out_col_scales[j] = (c * (1.0f / 7.0f)) * (1.0f / 32.0f); }
    uint32_t * out_blk = out32 + warp * 128;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        uint32_t q = (uint32_t)(max(-8, min(7, __float2int_rn(v[k] * inv_scale))) & 0xF);
        q |= __shfl_down_sync(0xFFFFFFFF, q, 1) << 4;
        q |= __shfl_down_sync(0xFFFFFFFF, q, 2) << 8;
        q |= __shfl_down_sync(0xFFFFFFFF, q, 4) << 16;
        if ((lane & 7) == 0) { out_blk[k * 4 + (lane >> 3)] = q; }
    }
}

struct pq2_narrow_i8_scratch {
    uint32_t   * B_i8        = nullptr;
    int32_t    * B_sum       = nullptr;
    float      * B_scales    = nullptr;
    size_t       cap_b       = 0;
    size_t       cap_sum     = 0;
    size_t       cap_sc      = 0;
    const void * src1_data   = nullptr;
    size_t       src1_nbytes = 0;
    int64_t      N           = 0;
    int64_t      K           = 0;
    int64_t      s11         = 0;
    uint64_t     gen         = 0;
    bool         valid       = false;
};

static pq2_narrow_i8_scratch & pq2_narrow_i8_scratch_for_stream(cudaStream_t stream) {
    static std::mutex mu;
    static std::unordered_map<cudaStream_t, pq2_narrow_i8_scratch> map;
    std::lock_guard<std::mutex> lock(mu);
    return map[stream];
}

static void pq2_narrow_i8_cache_invalidate(cudaStream_t stream) {
    pq2_narrow_i8_scratch_for_stream(stream).valid = false;
}

static void pq2_narrow_i8_cache_invalidate_if_overlap(cudaStream_t stream, const void * dst_data, size_t dst_nbytes) {
    if (!dst_data || dst_nbytes == 0) { return; }
    const uintptr_t b0 = (uintptr_t) dst_data, b1 = b0 + dst_nbytes;
    auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
    if (sc.valid) {
        const uintptr_t a0 = (uintptr_t) sc.src1_data, a1 = a0 + sc.src1_nbytes;
        if (a0 < b1 && b0 < a1) { sc.valid = false; }
    }
}

static void pq2_prefill_i4_cache_invalidate(cudaStream_t stream) {
    auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
    sc.valid = false;
    sc.mw_valid = false;
}

static void pq2_prefill_i4_cache_invalidate_if_overlap(cudaStream_t stream, const void * dst_data, size_t dst_nbytes) {
    if (!dst_data || dst_nbytes == 0) { return; }
    const uintptr_t b0 = (uintptr_t) dst_data, b1 = b0 + dst_nbytes;
    auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
    if (sc.valid) {
        const uintptr_t a0 = (uintptr_t) sc.src1_data, a1 = a0 + sc.src1_nbytes;
        if (a0 < b1 && b0 < a1) { sc.valid = false; }
    }
    if (sc.mw_valid) {
        const uintptr_t a0 = (uintptr_t) sc.mw_data, a1 = a0 + sc.mw_nbytes;
        if (a0 < b1 && b0 < a1) { sc.mw_valid = false; }
    }
}

__global__ static void f32_to_f16_1d_kernel(const float * __restrict__ src, half * __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i] = __float2half(src[i]); }
}

static bool ggml_cuda_mul_mat_f32_narrow_f16_tc(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) { return false; }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) { return false; }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (N < 17 || N > 64 || M > 64 || K < 512 || (K % 4) != 0 ||
            strstr(src0->name, "lora") != nullptr ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) { return false; }
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    (void) cudaStreamIsCapturing(stream, &cap);
    const bool allow_alloc = (cap == cudaStreamCaptureStatusNone);

    static std::mutex w_mu;
    static std::unordered_map<const void *, half *> w_f16_cache;
    half * w_f16 = nullptr;
    {
        std::lock_guard<std::mutex> lk(w_mu);
        auto it = w_f16_cache.find(src0->data);
        if (it != w_f16_cache.end()) {
            w_f16 = it->second;
        } else {
            if (!allow_alloc) { return false; }
            const int64_t n_el = M * K;
            CUDA_CHECK(cudaMalloc(&w_f16, (size_t) n_el * sizeof(half)));
            f32_to_f16_1d_kernel<<<(unsigned) ((n_el + 255) / 256), 256, 0, stream>>>((const float *) src0->data, w_f16, n_el);
            w_f16_cache[src0->data] = w_f16;
        }
    }
    struct x_f16_buf { half * ptr = nullptr; size_t cap = 0; const void * src1 = nullptr; int64_t N = 0, K = 0; };
    static std::mutex x_mu;
    static std::unordered_map<cudaStream_t, x_f16_buf> x_map;
    x_f16_buf * xb = nullptr;
    {
        std::lock_guard<std::mutex> lk(x_mu);
        xb = &x_map[stream];
    }
    const size_t need_el = (size_t) N * (size_t) K;
    if (xb->cap < need_el) {
        if (!allow_alloc) { return false; }
        const size_t alloc_el = std::max(need_el, (size_t) 64 * 8192);
        if (xb->ptr) { CUDA_CHECK(cudaFree(xb->ptr)); }
        CUDA_CHECK(cudaMalloc(&xb->ptr, alloc_el * sizeof(half)));
        xb->cap = alloc_el;
    }
    f32_to_f16_1d_kernel<<<(unsigned) ((need_el + 255) / 256), 256, 0, stream>>>((const float *) src1->data, xb->ptr, (int64_t) need_el);
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        (int) M, (int) N, (int) K,
        &alpha, w_f16, CUDA_R_16F, (int) K,
        xb->ptr, CUDA_R_16F, (int) K,
        &beta, dst->data, CUDA_R_32F, (int) M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return true;
}

static bool ggml_cuda_mul_mat_f32_prefill_f16_tc(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_pq2_prefill_int4_enabled()) { return false; }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) { return false; }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) { return false; }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (N < 256 || M > 64 || strstr(src0->name, "lora") != nullptr ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) { return false; }
    cudaStream_t stream = ctx.stream();
    auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
    if (!(sc.mw_valid && sc.mw_data == src1->data && sc.mw_N == N && sc.mw_K == K && sc.mul_w_f16)) { return false; }
    static std::mutex w_mu;
    static std::unordered_map<const void *, half *> w_f16_cache;
    half * w_f16 = nullptr;
    {
        std::lock_guard<std::mutex> lk(w_mu);
        auto it = w_f16_cache.find(src0->data);
        if (it != w_f16_cache.end()) {
            w_f16 = it->second;
        } else {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            (void) cudaStreamIsCapturing(stream, &cap);
            if (cap != cudaStreamCaptureStatusNone) { return false; }
            const int64_t n_el = M * K;
            CUDA_CHECK(cudaMalloc(&w_f16, (size_t) n_el * sizeof(half)));
            f32_to_f16_1d_kernel<<<(unsigned) ((n_el + 255) / 256), 256, 0, stream>>>((const float *) src0->data, w_f16, n_el);
            w_f16_cache[src0->data] = w_f16;
        }
    }
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        (int) M, (int) N, (int) K,
        &alpha, w_f16, CUDA_R_16F, (int) K,
        sc.mul_w_f16, CUDA_R_16F, (int) K,
        &beta, dst->data, CUDA_R_32F, (int) M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    pq2_prefill_i4_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
    return true;
}

static int64_t pq2_narrow_i8_max_n() {
    static const int64_t env_n = []() -> int64_t {
        const char * v = getenv("GGML_CUDA_PQ2_NARROW_MAX_N");
        return v ? (int64_t) atoi(v) : 0;
    }();
    if (env_n > 0) { return env_n; }
    return ggml_cuda_pq2_prefill_int4_enabled() ? 255 : 512;
}

static int64_t pq2_narrow_i8_n_pad(int64_t N) {
    return (N <= 8) ? 8 : ((N <= 16) ? 16 : ((N <= 24) ? 24 : ((N <= 32) ? 32 : ((N + 63) / 64) * 64)));
}

__device__ __forceinline__ void fwht1024_pack_2nibble_warp(
        const float v[32], const int warp, const int lane,
        uint32_t * __restrict__ out32, int32_t * __restrict__ out_s, float * __restrict__ out_sc) {
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const float x0 = v[b * 4 + 0];
        const float x1 = v[b * 4 + 1];
        const float x2 = v[b * 4 + 2];
        const float x3 = v[b * 4 + 3];
        float bmax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFF, bmax, off)); }
        const float inv_s = bmax > 0.0f ? (119.0f / bmax) : 0.0f;
        const int q0 = max(-120, min(119, __float2int_rn(x0 * inv_s)));
        const int q1 = max(-120, min(119, __float2int_rn(x1 * inv_s)));
        const int q2 = max(-120, min(119, __float2int_rn(x2 * inv_s)));
        const int q3 = max(-120, min(119, __float2int_rn(x3 * inv_s)));
        int s = q0 + q1 + q2 + q3;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { s += __shfl_xor_sync(0xFFFFFFFF, s, off); }
        const int kb = warp * 8 + b;
        if (lane == 0) {
            out_s[kb]  = s;
            out_sc[kb] = (bmax / 119.0f) * (1.0f / 32.0f);
        }
        const int qs[4] = { q0, q1, q2, q3 };
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int q  = qs[c];
            const int lo = ((q + 8) & 15) - 8;
            const int hi = (q - lo) >> 4;
            uint32_t wh = (uint32_t)(hi & 0xF);
            uint32_t wl = (uint32_t)(lo & 0xF);
            wh |= __shfl_down_sync(0xFFFFFFFF, wh, 1) << 4;
            wl |= __shfl_down_sync(0xFFFFFFFF, wl, 1) << 4;
            wh |= __shfl_down_sync(0xFFFFFFFF, wh, 2) << 8;
            wl |= __shfl_down_sync(0xFFFFFFFF, wl, 2) << 8;
            wh |= __shfl_down_sync(0xFFFFFFFF, wh, 4) << 16;
            wl |= __shfl_down_sync(0xFFFFFFFF, wl, 4) << 16;
            if ((lane & 7) == 0) {
                const int kw = ((lane >> 3) << 2) | c;
                out32[(kb << 5) + kw]      = wh;
                out32[(kb << 5) + 16 + kw] = wl;
            }
        }
    }
}

__global__ static void mul_signs_fwht1024_to_narrow_i8_kernel(
        const float * __restrict__ src, const float * __restrict__ signs,
        uint32_t * __restrict__ dst_i8, int32_t * __restrict__ dst_sum128, float * __restrict__ dst_scales128,
        const int M, const int N, const int N_pad, const int64_t s11) {
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int K4 = M >> 2, n_kblk = M >> 7;
    uint32_t * out32  = dst_i8        + (int64_t) j * K4;
    int32_t  * out_s  = dst_sum128    + (int64_t) j * n_kblk;
    float    * out_sc = dst_scales128 + (int64_t) j * n_kblk;
    if (j >= N) {
        for (int i = tid; i < K4; i += blockDim.x) { out32[i] = 0u; }
        for (int i = tid; i < n_kblk; i += blockDim.x) { out_s[i] = 0; out_sc[i] = 0.0f; }
        return;
    }
    const float * in_col = src + (int64_t) j * s11;
    float v[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int idx = warp * 1024 + k * 32 + lane;
        v[k] = in_col[idx] * __ldg(signs + idx);
    }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int k = 0; k < 32; k += (h << 1))
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a = v[k + r], b = v[k + r + h];
                v[k + r] = a + b; v[k + r + h] = a - b;
            }
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x = v[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float x2 = __shfl_xor_sync(0xFFFFFFFF, x, h);
            x = ((lane & h) == 0) ? (x + x2) : (x2 - x);
        }
        v[k] = x;
    }
    fwht1024_pack_2nibble_warp(v, warp, lane, out32, out_s, out_sc);
}

__global__ static void rms_mul_signs_fwht1024_to_narrow_i8_kernel(
        const float * __restrict__ src, const float * __restrict__ opt_add_b, float * __restrict__ opt_add_out,
        const float * __restrict__ rms_w, const float eps,
        const float * __restrict__ signs, float * __restrict__ opt_mul_w_out,
        uint32_t * __restrict__ dst_i8, int32_t * __restrict__ dst_sum128, float * __restrict__ dst_scales128,
        const int M, const int N, const int N_pad, const int64_t s11) {
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n_blk = M >> 10, K4 = M >> 2, n_kblk = M >> 7;
    uint32_t * out32  = dst_i8        + (int64_t) j * K4;
    int32_t  * out_s  = dst_sum128    + (int64_t) j * n_kblk;
    float    * out_sc = dst_scales128 + (int64_t) j * n_kblk;
    if (j >= N) {
        for (int i = tid; i < K4; i += blockDim.x) { out32[i] = 0u; }
        for (int i = tid; i < n_kblk; i += blockDim.x) { out_s[i] = 0; out_sc[i] = 0.0f; }
        return;
    }
    const float * in_col  = src + (int64_t) j * s11;
    const float * add_b   = opt_add_b   ? (opt_add_b   + (int64_t) j * M) : nullptr;
    float       * add_out = opt_add_out ? (opt_add_out + (int64_t) j * M) : nullptr;
    float v[32];
    float sq_in = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int idx = warp * 1024 + k * 32 + lane;
        float x = in_col[idx];
        if (add_b) {
            x += add_b[idx];
            if (add_out) { add_out[idx] = x; }
        }
        v[k] = x;
        sq_in += x * x;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { sq_in += __shfl_xor_sync(0xFFFFFFFF, sq_in, off); }
    __shared__ float ssq[24];
    if (lane == 0) { ssq[warp] = sq_in; }
    __syncthreads();
    float tot_in_sq = 0.0f;
    for (int w = 0; w < n_blk; ++w) { tot_in_sq += ssq[w]; }
    const float irms = rsqrtf(tot_in_sq / (float) M + eps);
    float * mw_col = opt_mul_w_out ? (opt_mul_w_out + (int64_t) j * M) : nullptr;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int idx = warp * 1024 + k * 32 + lane;
        const float mw = v[k] * irms * __ldg(rms_w + idx);
        if (mw_col) { mw_col[idx] = mw; }
        v[k] = mw * __ldg(signs + idx);
    }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int k = 0; k < 32; k += (h << 1))
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a = v[k + r], b = v[k + r + h];
                v[k + r] = a + b; v[k + r + h] = a - b;
            }
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x = v[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float x2 = __shfl_xor_sync(0xFFFFFFFF, x, h);
            x = ((lane & h) == 0) ? (x + x2) : (x2 - x);
        }
        v[k] = x;
    }
    fwht1024_pack_2nibble_warp(v, warp, lane, out32, out_s, out_sc);
}

static bool pq2_narrow_i8_alloc_bufs(
        pq2_narrow_i8_scratch & sc, const int64_t N_pad, const int64_t K, cudaStream_t stream) {
    const size_t need_b   = (size_t) N_pad * (size_t) (K >> 2) * sizeof(uint32_t);
    const size_t need_sum = (size_t) N_pad * (size_t) (K >> 7) * sizeof(int32_t);
    const size_t need_sc  = (size_t) N_pad * (size_t) (K >> 7) * sizeof(float);
    if (sc.cap_b < need_b || sc.cap_sum < need_sum || sc.cap_sc < need_sc) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        (void) cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) { return false; }
        const size_t alloc_b   = std::max(need_b,   (size_t) 128 * (20480 / 4) * sizeof(uint32_t));
        const size_t alloc_sum = std::max(need_sum, (size_t) 128 * (20480 / 128) * sizeof(int32_t));
        const size_t alloc_sc  = std::max(need_sc,  (size_t) 128 * (20480 / 128) * sizeof(float));
        if (sc.B_i8)     { CUDA_CHECK(cudaFree(sc.B_i8)); }
        if (sc.B_sum)    { CUDA_CHECK(cudaFree(sc.B_sum)); }
        if (sc.B_scales) { CUDA_CHECK(cudaFree(sc.B_scales)); }
        CUDA_CHECK(cudaMalloc(&sc.B_i8, alloc_b));
        CUDA_CHECK(cudaMalloc(&sc.B_sum, alloc_sum));
        CUDA_CHECK(cudaMalloc(&sc.B_scales, alloc_sc));
        sc.cap_b = alloc_b; sc.cap_sum = alloc_sum; sc.cap_sc = alloc_sc;
        sc.valid = false;
    }
    return true;
}

static bool pq2_narrow_i8_ensure_b(
        pq2_narrow_i8_scratch & sc, const ggml_tensor * src1,
        const int64_t N, const int64_t N_pad, const int64_t K, const int64_t s11,
        cudaStream_t stream, bool allow_alloc) {
    const size_t need_b   = (size_t) N_pad * (size_t) (K >> 2) * sizeof(uint32_t);
    const size_t need_sum = (size_t) N_pad * (size_t) (K >> 7) * sizeof(int32_t);
    const size_t need_sc  = (size_t) N_pad * (size_t) (K >> 7) * sizeof(float);
    if (sc.cap_b < need_b || sc.cap_sum < need_sum || sc.cap_sc < need_sc) {
        if (!allow_alloc) { return false; }
        const size_t alloc_b   = std::max(need_b,   (size_t) 128 * (20480 / 4) * sizeof(uint32_t));
        const size_t alloc_sum = std::max(need_sum, (size_t) 128 * (20480 / 128) * sizeof(int32_t));
        const size_t alloc_sc  = std::max(need_sc,  (size_t) 128 * (20480 / 128) * sizeof(float));
        if (sc.B_i8)     { CUDA_CHECK(cudaFree(sc.B_i8)); }
        if (sc.B_sum)    { CUDA_CHECK(cudaFree(sc.B_sum)); }
        if (sc.B_scales) { CUDA_CHECK(cudaFree(sc.B_scales)); }
        CUDA_CHECK(cudaMalloc(&sc.B_i8, alloc_b));
        CUDA_CHECK(cudaMalloc(&sc.B_sum, alloc_sum));
        CUDA_CHECK(cudaMalloc(&sc.B_scales, alloc_sc));
        sc.cap_b = alloc_b; sc.cap_sum = alloc_sum; sc.cap_sc = alloc_sc;
        sc.valid = false;
    }
    if (!(sc.valid && sc.src1_data == src1->data && sc.N == N && sc.K == K && sc.s11 == s11)) {
        quantize_f32_to_i8_sum128_kernel<<<(unsigned) N_pad, 512, 0, stream>>>(
            (const float *) src1->data, sc.B_i8, sc.B_sum, sc.B_scales, (int) N, (int) N_pad, (int) K, s11);
        sc.src1_data   = src1->data;
        sc.src1_nbytes = ggml_nbytes(src1);
        sc.N           = N;
        sc.K           = K;
        sc.s11         = s11;
        sc.valid       = true;
    }
    return true;
}

static bool ggml_cuda_mul_mat_pq2_narrow_int8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * add_src = nullptr) {
    if (!ggml_cuda_pq2_narrow_int8_enabled()) { return false; }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) { return false; }
    if (src0->type != GGML_TYPE_PQ2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) { return false; }
    if (add_src && (add_src->type != GGML_TYPE_F32 || !ggml_are_same_shape(add_src, dst) ||
            add_src->nb[0] != sizeof(float) || add_src->nb[1] != dst->nb[1])) { return false; }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (N < 5 || N > pq2_narrow_i8_max_n() || M > (N >= 256 ? 65536 : 262144) || (K % 128) != 0 || (M % 8) != 0 ||
            !ggml_is_contiguous(src0) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    (void) cudaStreamIsCapturing(stream, &cap_status);
    const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);

    const auto ae = get_or_repack_pq2_tc_i8((const block_pq2_0 *) src0->data, M, K, stream, allow_alloc);
    if (!ae.qs || !ae.d) { return false; }

    const int64_t N_pad = pq2_narrow_i8_n_pad(N);
    auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
    const int64_t s11   = src1->nb[1] / sizeof(float);
    const int64_t s_dst = dst->nb[1] / sizeof(float);
    if (!pq2_narrow_i8_ensure_b(sc, src1, N, N_pad, K, s11, stream, allow_alloc)) { return false; }
    const float * add_ptr = add_src ? (const float *) add_src->data : nullptr;
    // W4 R4 + Split-K for N<=32; 2D warp-tiled uint4 (128 regs, 16 warps/SM) for N=33..128 (100% lossless!)
    unsigned split_k = 1u;
    if (M <= 65536 && K >= 2048) {
        if (N_pad <= 32) {
            split_k = (M <= 8192) ? 4u : 2u;
        } else if (N <= 64 && M <= 8192) {
            split_k = 2u;
        } else if (N >= 129 && M <= 12288 && K >= 4096) {
            // Wave-balanced split-K to eliminate tail-wave SM underutilization on N=129..512
            const int64_t wave_slots = (int64_t) std::max(1, ggml_cuda_info().devices[ctx.device].nsm) * 2;
            const int64_t grid_xy = ((M + 127) / 128) * ((N + 63) / 64);
            const int64_t n_kblk  = K >> 7;
            auto wave_eff = [&](unsigned sk) -> double {
                const int64_t tot   = grid_xy * sk;
                const int64_t waves = (tot + wave_slots - 1) / wave_slots;
                return (double) tot / (double) (waves * wave_slots);
            };
            const double eff1 = wave_eff(1u);
            double best_eff = eff1;
            for (unsigned sk : { 2u, 3u }) {
                if (n_kblk >= (int64_t) (12u * sk)) {
                    const double e = wave_eff(sk);
                    if (e > eff1 + 0.12 && e > best_eff + 0.03) {
                        best_eff = e;
                        split_k  = sk;
                    }
                }
            }
        }
    }
    if (split_k > 1u) {
        const size_t out_bytes = (size_t) N * (size_t) s_dst * sizeof(float);
        if (!add_ptr) {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, out_bytes, stream));
        } else if (add_ptr != (const float *) dst->data) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, add_ptr, out_bytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    if (N_pad == 8) {
        dim3 grid((unsigned) ((M + 127) / 128), 1, split_k);
        pq2_narrow_int8_v2_kernel<8, 4, 4><<<grid, 128, 0, stream>>>(
            ae.qs, ae.d, sc.B_i8, sc.B_sum, sc.B_scales, add_ptr, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else if (N_pad == 16) {
        dim3 grid((unsigned) ((M + 127) / 128), 1, split_k);
        pq2_narrow_int8_v2_kernel<16, 4, 4><<<grid, 128, 0, stream>>>(
            ae.qs, ae.d, sc.B_i8, sc.B_sum, sc.B_scales, add_ptr, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else if (N_pad == 24) {
        dim3 grid((unsigned) ((M + 127) / 128), 1, split_k);
        pq2_narrow_int8_v2_kernel<24, 4, 4><<<grid, 128, 0, stream>>>(
            ae.qs, ae.d, sc.B_i8, sc.B_sum, sc.B_scales, add_ptr, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else if (N_pad == 32) {
        dim3 grid((unsigned) ((M + 127) / 128), 1, split_k);
        pq2_narrow_int8_v2_kernel<32, 4, 4><<<grid, 128, 0, stream>>>(
            ae.qs, ae.d, sc.B_i8, sc.B_sum, sc.B_scales, add_ptr, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else if (N >= 129 || (N <= 64 && M <= 8192)) {
        dim3 grid((unsigned) ((M + 127) / 128), (unsigned) ((N + 63) / 64), split_k);
        pq2_narrow2d_i8_kernel<4, 2, 4, 4, true><<<grid, 256, 0, stream>>>(
            ae.qs, ae.d, (const uint4 *) sc.B_i8, sc.B_sum, sc.B_scales, add_ptr, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else {
        dim3 grid((unsigned) ((M + 63) / 64), (unsigned) ((N + 63) / 64), 1);
        pq2_narrow2d_i8_kernel<2, 2, 4, 4, false><<<grid, 128, 0, stream>>>(
            ae.qs, ae.d, (const uint4 *) sc.B_i8, sc.B_sum, sc.B_scales, add_ptr, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    }
    pq2_narrow_i8_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool ggml_cuda_mul_mat_pq2_narrow_int8_swiglu(
        ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
        const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_pq2_narrow_int8_enabled()) { return false; }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) { return false; }
    if (gate_w->type != GGML_TYPE_PQ2_0 || up_w->type != GGML_TYPE_PQ2_0 ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    if (!ggml_are_same_shape(gate_w, up_w) || gate_w->ne[2] != 1 || gate_w->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1) { return false; }
    const int64_t K = gate_w->ne[0], M = gate_w->ne[1], N = src1->ne[1];
    if (N < 5 || N > pq2_narrow_i8_max_n() || M > 262144 || (K % 128) != 0 || (M % 8) != 0 ||
            !ggml_is_contiguous(gate_w) || !ggml_is_contiguous(up_w) ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            ggml_backend_buffer_get_usage(gate_w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    (void) cudaStreamIsCapturing(stream, &cap_status);
    const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);

    const auto ge = get_or_repack_pq2_tc_i8((const block_pq2_0 *) gate_w->data, M, K, stream, allow_alloc);
    const auto ue = get_or_repack_pq2_tc_i8((const block_pq2_0 *) up_w->data,   M, K, stream, allow_alloc);
    if (!ge.qs || !ge.d || !ue.qs || !ue.d) { return false; }

    const int64_t N_pad = pq2_narrow_i8_n_pad(N);
    auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
    const int64_t s11   = src1->nb[1] / sizeof(float);
    const int64_t s_dst = dst->nb[1] / sizeof(float);
    if (!pq2_narrow_i8_ensure_b(sc, src1, N, N_pad, K, s11, stream, allow_alloc)) { return false; }

    dim3 grid((unsigned) ((M + 63) / 64));
    if (N_pad == 8) {
        pq2_narrow_int8_swiglu_v2_kernel<8, 4, 2><<<grid, 128, 0, stream>>>(
            ge.qs, ge.d, ue.qs, ue.d, sc.B_i8, sc.B_sum, sc.B_scales, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else if (N_pad == 16) {
        pq2_narrow_int8_swiglu_v2_kernel<16, 4, 2><<<grid, 128, 0, stream>>>(
            ge.qs, ge.d, ue.qs, ue.d, sc.B_i8, sc.B_sum, sc.B_scales, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else if (N_pad == 24) {
        pq2_narrow_int8_swiglu_v2_kernel<24, 4, 2><<<grid, 128, 0, stream>>>(
            ge.qs, ge.d, ue.qs, ue.d, sc.B_i8, sc.B_sum, sc.B_scales, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else if (N_pad == 32) {
        pq2_narrow_int8_swiglu_v2_kernel<32, 4, 2><<<grid, 128, 0, stream>>>(
            ge.qs, ge.d, ue.qs, ue.d, sc.B_i8, sc.B_sum, sc.B_scales, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    } else {
        dim3 grid2((unsigned) ((M + 63) / 64), (unsigned) ((N + 63) / 64));
        pq2_narrow2d_swiglu_kernel<4, 2, 2, 4, true><<<grid2, 256, 0, stream>>>(
            ge.qs, ge.d, ue.qs, ue.d, (const uint4 *) sc.B_i8, sc.B_sum, sc.B_scales, (float *) dst->data, (int) M, (int) N, (int) K, s_dst);
    }
    pq2_narrow_i8_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
    CUDA_CHECK(cudaGetLastError());
    return true;
}


__global__ static void compute_pq2_row_scales_kernel(
        const block_pq2_0 * __restrict__ src0, float * __restrict__ row_scales, float * __restrict__ row_inv_scales,
        const int M, const int K) {
    const int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    const int nkb = K / QK_PQ2_0;
    const block_pq2_0 * row_blk = src0 + (int64_t) row * nkb;
    const int tid = threadIdx.x;

    float rmax = 0.0f;
    for (int b = tid; b < nkb; b += blockDim.x) {
        rmax = fmaxf(rmax, fabsf(__half2float(row_blk[b].d)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, off));
    }
    __shared__ float smax[8];
    if ((tid & 31) == 0) {
        smax[tid >> 5] = rmax;
    }
    __syncthreads();
    if (tid < 8) {
        float v = smax[tid];
        v = fmaxf(v, __shfl_xor_sync(0xFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    if (tid == 0) {
        const float max_d = smax[0];
        row_scales[row]     = max_d / 127.0f;
        row_inv_scales[row] = max_d > 0.0f ? (127.0f / max_d) : 0.0f;
    }
}

__global__ static void dequant_pq2_to_i8_1pass_kernel(
        const block_pq2_0 * __restrict__ src0, const float * __restrict__ row_inv_scales,
        int8_t * __restrict__ dst_i8, const int M, const int nkb) {
    const int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    const float inv_scale = row_inv_scales[row];
    const block_pq2_0 * row_blk = src0 + (int64_t) row * nkb;
    int8_t * out_row = dst_i8 + (int64_t) row * (nkb * QK_PQ2_0);
    const int total_q = nkb * 4;
    for (int idx = threadIdx.x; idx < total_q; idx += blockDim.x) {
        const int b  = idx >> 2;
        const int kq = idx & 3;
        const block_pq2_0 * blk = row_blk + b;
        const float df = __half2float(blk->d);
        const int s_i8  = max(-127, min(127, __float2int_rn(df * inv_scale)));
        const int s_neg = max(-127, min(127, -s_i8));
        const int s_2   = max(-127, min(127, 2 * s_i8));
        const uint32_t lut =
            ((uint32_t) (uint8_t) (int8_t) s_neg) |
            (0u << 8) |
            (((uint32_t) (uint8_t) (int8_t) s_i8) << 16) |
            (((uint32_t) (uint8_t) (int8_t) s_2)  << 24);

        const uint8_t * qsi = blk->qs + kq * 8;
        const uint32_t mis  = (uint32_t) ((uintptr_t) qsi & 3);
        const uint32_t * qwi = (const uint32_t *) (qsi - mis);
        const uint32_t w0 = qwi[0], w1 = qwi[1], w2 = qwi[1 + mis / 2];
        const uint32_t sh = 8 * mis;
        const uint32_t qw[2] = { __funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh) };

        int v[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = (j & 1) == 0 ? int(int16_t(qw[j / 2] & 0xFFFFu)) : int(qw[j / 2]) >> 16;
            const int qe = __byte_perm(lut, lut, q >> 0);
            const int qo = __byte_perm(lut, lut, q >> 2);
            v[2 * j + 0] = __byte_perm(qe, qo, 0x5140);
            v[2 * j + 1] = __byte_perm(qe, qo, 0x7362);
        }
        int4 * out4 = (int4 *) (out_row + b * 128 + kq * 32);
        out4[0] = make_int4(v[0], v[1], v[2], v[3]);
        out4[1] = make_int4(v[4], v[5], v[6], v[7]);
    }
}

struct pq2_row_scale_cache_entry {
    float  * scales;
    float  * inv_scales;
    int8_t * A_i8;
    int64_t  M;
    int64_t  K;
};

static size_t ggml_cuda_pq2_cache_i8_max_bytes() {
    static const size_t max_bytes = []() -> size_t {
        const char * v = getenv("GGML_CUDA_PQ2_CACHE_MB");
        if (v != nullptr) {
            long mb = atol(v);
            return mb > 0 ? (size_t) mb * 1024ull * 1024ull : 0ull;
        }
        const char * en = getenv("GGML_CUDA_PQ2_CACHE_I8");
        if (en != nullptr && atoi(en) != 0) {
            return 26000ull * 1024ull * 1024ull; // 26 GB default cap when enabled
        }
        return 0ull;
    }();
    return max_bytes;
}

static pq2_row_scale_cache_entry get_or_compute_pq2_row_scales(
        const block_pq2_0 * src0_data, const int64_t M, const int64_t K, cudaStream_t stream, bool allow_alloc = true) {
    static std::mutex mu;
    static std::unordered_map<const void *, pq2_row_scale_cache_entry> cache;
    static std::vector<pq2_row_scale_cache_entry> retired;
    static size_t cached_i8_bytes = 0;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(src0_data);
    if (it != cache.end() && it->second.M >= M && it->second.K == K) {
        return it->second;
    }
    if (!allow_alloc) {
        return {};
    }
    pq2_row_scale_cache_entry e{};
    e.M = M;
    e.K = K;
    e.A_i8 = nullptr;
    if (cudaMalloc(&e.scales, (size_t) M * sizeof(float)) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    if (cudaMalloc(&e.inv_scales, (size_t) M * sizeof(float)) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFree(e.scales);
        return {};
    }
    compute_pq2_row_scales_kernel<<<(unsigned) M, 256, 0, stream>>>(src0_data, e.scales, e.inv_scales, (int) M, (int) K);
    const size_t need_bytes = (size_t) M * (size_t) K;
    const size_t max_bytes  = ggml_cuda_pq2_cache_i8_max_bytes();
    size_t free_b = 0, total_b = 0;
    const bool have_headroom = (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) || (free_b >= need_bytes + (3584ULL << 20));
    if (max_bytes > 0 && have_headroom && cached_i8_bytes + need_bytes <= max_bytes) {
        int8_t * ptr = nullptr;
        if (cudaMalloc(&ptr, need_bytes) == cudaSuccess) {
            const int nkb = (int) (K / QK_PQ2_0);
            dequant_pq2_to_i8_1pass_kernel<<<(unsigned) M, 256, 0, stream>>>(
                src0_data, e.inv_scales, ptr, (int) M, nkb);
            e.A_i8 = ptr;
            cached_i8_bytes += need_bytes;
        } else {
            (void) cudaGetLastError();
        }
    }
    if (it != cache.end()) {
        retired.push_back(it->second);
    }
    cache[src0_data] = e;
    return e;
}

// 1-entry activation (B_i8 + col_scales) cache per stream across sibling matmuls
// (e.g. attn_qkv + attn_gate, attn_q + attn_k + attn_v, ssm_alpha + ssm_beta).
struct pq2_b_i8_cache_state {
    const void * src1_data = nullptr;
    size_t       src1_nbytes = 0;
    int64_t      N = 0;
    int64_t      K = 0;
    int64_t      s11 = 0;
    int8_t     * B_i8 = nullptr;
    size_t       B_i8_cap = 0;
    float      * col_scales = nullptr;
    size_t       col_scales_cap = 0;
    bool         valid = false;
};

static pq2_b_i8_cache_state & pq2_b_i8_cache_for_stream(cudaStream_t stream) {
    static std::mutex mu;
    static std::unordered_map<cudaStream_t, pq2_b_i8_cache_state> map;
    std::lock_guard<std::mutex> lock(mu);
    return map[stream];
}

static void pq2_b_i8_cache_invalidate(cudaStream_t stream) {
    pq2_b_i8_cache_for_stream(stream).valid = false;
}

static void pq2_b_i8_cache_invalidate_if_overlap(cudaStream_t stream, const void * dst_data, size_t dst_nbytes) {
    auto & st = pq2_b_i8_cache_for_stream(stream);
    if (!st.valid || !dst_data || dst_nbytes == 0) {
        return;
    }
    const char * a0 = (const char *) st.src1_data;
    const char * a1 = a0 + st.src1_nbytes;
    const char * b0 = (const char *) dst_data;
    const char * b1 = b0 + dst_nbytes;
    if (a0 < b1 && b0 < a1) {
        st.valid = false;
    }
}

static bool pq2_b_i8_cache_enabled() {
    static const bool en = []() {
        const char * v = getenv("GGML_CUDA_PQ2_CACHE_B");
        return v == nullptr || atoi(v) != 0;
    }();
    return en;
}

// Quantize static thin F32 weight [K, M] (e.g. 5120x48 ssm_alpha/beta) to row-wise INT8 ONCE and cache it!
__global__ static void quantize_f32_weight_to_i8_rowwise_kernel(
        const float * __restrict__ src0, int8_t * __restrict__ dst_i8, float * __restrict__ row_scales,
        const int M, const int K) {
    const int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    const float4 * in4 = (const float4 *) (src0 + (int64_t) row * K);
    const int K4 = K >> 2;
    const int tid = threadIdx.x;
    float rmax = 0.0f;
    for (int i = tid; i < K4; i += blockDim.x) {
        const float4 x = in4[i];
        rmax = fmaxf(rmax, fabsf(x.x));
        rmax = fmaxf(rmax, fabsf(x.y));
        rmax = fmaxf(rmax, fabsf(x.z));
        rmax = fmaxf(rmax, fabsf(x.w));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        rmax = fmaxf(rmax, __shfl_xor_sync(0xFFFFFFFF, rmax, off));
    }
    __shared__ float smax[8];
    if ((tid & 31) == 0) {
        smax[tid >> 5] = rmax;
    }
    __syncthreads();
    if (tid < 8) {
        float v = smax[tid];
        v = fmaxf(v, __shfl_xor_sync(0xFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    const float max_v = smax[0];
    const float scale = max_v / 127.0f;
    const float inv_s = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        row_scales[row] = scale;
    }
    char4 * out4 = (char4 *) (dst_i8 + (int64_t) row * K);
    for (int i = tid; i < K4; i += blockDim.x) {
        const float4 x = in4[i];
        char4 q;
        q.x = (int8_t) __float2int_rn(x.x * inv_s);
        q.y = (int8_t) __float2int_rn(x.y * inv_s);
        q.z = (int8_t) __float2int_rn(x.z * inv_s);
        q.w = (int8_t) __float2int_rn(x.w * inv_s);
        out4[i] = q;
    }
}

struct thin_f32_i8_cache_entry {
    int8_t * A_i8;
    float  * scales;
    int64_t  M;
    int64_t  K;
};

static thin_f32_i8_cache_entry get_or_quantize_thin_f32(
        const float * src0_data, const int64_t M, const int64_t K, cudaStream_t stream, bool allow_alloc = true) {
    static std::mutex mu;
    static std::unordered_map<const void *, thin_f32_i8_cache_entry> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(src0_data);
    if (it != cache.end() && it->second.M == M && it->second.K == K) {
        return it->second;
    }
    if (!allow_alloc) {
        return {};
    }
    thin_f32_i8_cache_entry e{};
    e.M = M;
    e.K = K;
    CUDA_CHECK(cudaMalloc(&e.A_i8,   (size_t) M * K));
    CUDA_CHECK(cudaMalloc(&e.scales, (size_t) M * sizeof(float)));
    quantize_f32_weight_to_i8_rowwise_kernel<<<(unsigned) M, 256, 0, stream>>>(src0_data, e.A_i8, e.scales, (int) M, (int) K);
    cache[src0_data] = e;
    return e;
}

__global__ static void quantize_f32_to_i8_colwise_kernel(
        const float * __restrict__ src1, int8_t * __restrict__ dst_i8, float * __restrict__ col_scales,
        const int N, const int K, const int64_t s11);

static void get_or_quantize_b_i8(
        const ggml_tensor * src1, const int64_t N, const int64_t N_pad, const int64_t K, const int64_t s11,
        cudaStream_t stream, int8_t * & B_i8_out, float * & col_scales_out) {
    auto & st = pq2_b_i8_cache_for_stream(stream);
    if (pq2_b_i8_cache_enabled() && st.valid &&
            st.src1_data == src1->data && st.N == N && st.K == K && st.s11 == s11) {
        B_i8_out = st.B_i8;
        col_scales_out = st.col_scales;
        return;
    }
    const size_t need_b = (size_t) N_pad * (size_t) K;
    const size_t need_s = (size_t) N_pad * sizeof(float);
    if (st.B_i8_cap < need_b) {
        if (st.B_i8) { CUDA_CHECK(cudaFree(st.B_i8)); }
        CUDA_CHECK(cudaMalloc(&st.B_i8, need_b));
        st.B_i8_cap = need_b;
    }
    if (st.col_scales_cap < need_s) {
        if (st.col_scales) { CUDA_CHECK(cudaFree(st.col_scales)); }
        CUDA_CHECK(cudaMalloc(&st.col_scales, need_s));
        st.col_scales_cap = need_s;
    }
    quantize_f32_to_i8_colwise_kernel<<<(unsigned) N, 512, 0, stream>>>(
        (const float *) src1->data, st.B_i8, st.col_scales, (int) N, (int) K, s11);
    st.src1_data   = src1->data;
    st.src1_nbytes = ggml_nbytes(src1);
    st.N           = N;
    st.K           = K;
    st.s11         = s11;
    st.valid       = pq2_b_i8_cache_enabled();
    B_i8_out       = st.B_i8;
    col_scales_out = st.col_scales;
}

__global__ static void quantize_f32_to_i8_colwise_kernel(
        const float * __restrict__ src1, int8_t * __restrict__ dst_i8, float * __restrict__ col_scales,
        const int N, const int K, const int64_t s11) {
    const int col = blockIdx.x;
    if (col >= N) {
        return;
    }
    const float * in_col = src1 + (int64_t) col * s11;
    const int tid = threadIdx.x;
    const int K4 = K >> 2;
    const float4 * in4 = (const float4 *) in_col;

    // Single-pass register cache for K <= 20480 (512 threads * 10 float4 = 20480 floats):
    // reads src1 ONCE from DRAM instead of TWICE!
    constexpr int MAX_REG = 10;
    float4 reg[MAX_REG];
    const bool use_reg = (K4 <= (int) blockDim.x * MAX_REG);

    float cmax = 0.0f;
    if (use_reg) {
#pragma unroll
        for (int r = 0; r < MAX_REG; ++r) {
            const int i = r * blockDim.x + tid;
            if (i < K4) {
                const float4 x = in4[i];
                reg[r] = x;
                cmax = fmaxf(cmax, fabsf(x.x));
                cmax = fmaxf(cmax, fabsf(x.y));
                cmax = fmaxf(cmax, fabsf(x.z));
                cmax = fmaxf(cmax, fabsf(x.w));
            }
        }
    } else {
        for (int i = tid; i < K4; i += blockDim.x) {
            const float4 x = in4[i];
            cmax = fmaxf(cmax, fabsf(x.x));
            cmax = fmaxf(cmax, fabsf(x.y));
            cmax = fmaxf(cmax, fabsf(x.z));
            cmax = fmaxf(cmax, fabsf(x.w));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    __shared__ float smax[16];
    if ((tid & 31) == 0) {
        smax[tid >> 5] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = (tid < (int)(blockDim.x >> 5)) ? smax[tid] : 0.0f;
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    const float max_v = smax[0];
    const float scale_col = max_v / 127.0f;
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        col_scales[col] = scale_col;
    }

    char4 * out4 = (char4 *) (dst_i8 + (int64_t) col * K);
    if (use_reg) {
#pragma unroll
        for (int r = 0; r < MAX_REG; ++r) {
            const int i = r * blockDim.x + tid;
            if (i < K4) {
                const float4 x = reg[r];
                char4 q;
                q.x = (int8_t) __float2int_rn(x.x * inv_scale);
                q.y = (int8_t) __float2int_rn(x.y * inv_scale);
                q.z = (int8_t) __float2int_rn(x.z * inv_scale);
                q.w = (int8_t) __float2int_rn(x.w * inv_scale);
                out4[i] = q;
            }
        }
    } else {
        for (int i = tid; i < K4; i += blockDim.x) {
            const float4 x = in4[i];
            char4 q;
            q.x = (int8_t) __float2int_rn(x.x * inv_scale);
            q.y = (int8_t) __float2int_rn(x.y * inv_scale);
            q.z = (int8_t) __float2int_rn(x.z * inv_scale);
            q.w = (int8_t) __float2int_rn(x.w * inv_scale);
            out4[i] = q;
        }
    }
}

__global__ static void dequant_i32_to_f32_epilogue_kernel(
        const int32_t * __restrict__ src_i32, const float * __restrict__ row_scales, const float * __restrict__ col_scales,
        const float * __restrict__ add_src, float * __restrict__ dst, const int M, const int N, const int64_t s_dst) {
    const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int j  = blockIdx.y;
    if (i4 >= M || j >= N) {
        return;
    }
    const float cs  = col_scales[j];
    const float4 rs = *(const float4 *) (row_scales + i4);
    const int4   c  = *(const int4 *)   (src_i32 + (int64_t) j * M + i4);
    float4 out;
    out.x = (float) c.x * (rs.x * cs);
    out.y = (float) c.y * (rs.y * cs);
    out.z = (float) c.z * (rs.z * cs);
    out.w = (float) c.w * (rs.w * cs);
    if (add_src) {
        const float4 a = *(const float4 *) (add_src + (int64_t) j * s_dst + i4);
        out.x += a.x;
        out.y += a.y;
        out.z += a.z;
        out.w += a.w;
    }
    *(float4 *) (dst + (int64_t) j * s_dst + i4) = out;
}

__global__ static void dequant_i32_swiglu_epilogue_kernel(
        const int32_t * __restrict__ gate_i32, const float * __restrict__ gate_rs,
        const int32_t * __restrict__ up_i32,   const float * __restrict__ up_rs,
        const float * __restrict__ col_scales, float * __restrict__ dst,
        const int M, const int N, const int64_t s_dst) {
    const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int j  = blockIdx.y;
    if (i4 >= M || j >= N) {
        return;
    }
    const float cs   = col_scales[j];
    const float4 grs = *(const float4 *) (gate_rs + i4);
    const float4 urs = *(const float4 *) (up_rs   + i4);
    const int4   gc  = *(const int4 *)   (gate_i32 + (int64_t) j * M + i4);
    const int4   uc  = *(const int4 *)   (up_i32   + (int64_t) j * M + i4);
    auto silu = [](float x) { return x / (1.0f + __expf(-x)); };
    float4 out;
    out.x = silu((float) gc.x * (grs.x * cs)) * ((float) uc.x * (urs.x * cs));
    out.y = silu((float) gc.y * (grs.y * cs)) * ((float) uc.y * (urs.y * cs));
    out.z = silu((float) gc.z * (grs.z * cs)) * ((float) uc.z * (urs.z * cs));
    out.w = silu((float) gc.w * (grs.w * cs)) * ((float) uc.w * (urs.w * cs));
    *(float4 *) (dst + (int64_t) j * s_dst + i4) = out;
}

// Fused SwiGLU + col-wise INT8 quantization directly into B_i8 cache (zero 142 MB F32 intermediate!).
// 1 CTA per column j, 512 threads, up to M <= 18432 (512 * 4 * 9) held 100% in registers.
__global__ static void dequant_i32_swiglu_to_i8_epilogue_kernel(
        const int32_t * __restrict__ gate_i32, const float * __restrict__ gate_rs,
        const int32_t * __restrict__ up_i32,   const float * __restrict__ up_rs,
        const float * __restrict__ in_col_scales,
        int8_t * __restrict__ dst_i8, float * __restrict__ out_col_scales,
        const int M, const int N) {
    const int j = blockIdx.x;
    if (j >= N) {
        return;
    }
    const int tid = threadIdx.x;
    const int M4 = M >> 2;
    const float cs = in_col_scales[j];
    const int4 * g4 = (const int4 *) (gate_i32 + (int64_t) j * M);
    const int4 * u4 = (const int4 *) (up_i32   + (int64_t) j * M);
    const float4 * grs4 = (const float4 *) gate_rs;
    const float4 * urs4 = (const float4 *) up_rs;

    constexpr int MAX_REG = 9;
    float4 reg[MAX_REG];
    auto silu = [](float x) { return x / (1.0f + __expf(-x)); };
    float cmax = 0.0f;
#pragma unroll
    for (int r = 0; r < MAX_REG; ++r) {
        const int i = r * 512 + tid;
        if (i < M4) {
            const float4 grs = grs4[i];
            const float4 urs = urs4[i];
            const int4   gc  = g4[i];
            const int4   uc  = u4[i];
            float4 y;
            y.x = silu((float) gc.x * (grs.x * cs)) * ((float) uc.x * (urs.x * cs));
            y.y = silu((float) gc.y * (grs.y * cs)) * ((float) uc.y * (urs.y * cs));
            y.z = silu((float) gc.z * (grs.z * cs)) * ((float) uc.z * (urs.z * cs));
            y.w = silu((float) gc.w * (grs.w * cs)) * ((float) uc.w * (urs.w * cs));
            reg[r] = y;
            cmax = fmaxf(cmax, fabsf(y.x));
            cmax = fmaxf(cmax, fabsf(y.y));
            cmax = fmaxf(cmax, fabsf(y.z));
            cmax = fmaxf(cmax, fabsf(y.w));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    __shared__ float smax[16];
    if ((tid & 31) == 0) {
        smax[tid >> 5] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = smax[tid];
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    const float max_v = smax[0];
    const float scale_col = max_v / 127.0f;
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        out_col_scales[j] = scale_col;
    }

    char4 * out4 = (char4 *) (dst_i8 + (int64_t) j * M);
#pragma unroll
    for (int r = 0; r < MAX_REG; ++r) {
        const int i = r * 512 + tid;
        if (i < M4) {
            const float4 y = reg[r];
            char4 q;
            q.x = (int8_t) __float2int_rn(y.x * inv_scale);
            q.y = (int8_t) __float2int_rn(y.y * inv_scale);
            q.z = (int8_t) __float2int_rn(y.z * inv_scale);
            q.w = (int8_t) __float2int_rn(y.w * inv_scale);
            out4[i] = q;
        }
    }
}

// Fused SwiGLU + MUL(Hadamard signs) + FWHT(H) + col-wise INT8 quantization directly into B_i8 cache!
// Eliminates 142.6 MB SwiGLU F32 write + 142.6 MB MUL/FWHT read+write + 142.6 MB quantize read per layer!
template <int H>
__global__ static void dequant_i32_swiglu_fwht_to_i8_kernel(
        const int32_t * __restrict__ gate_i32, const float * __restrict__ gate_rs,
        const int32_t * __restrict__ up_i32,   const float * __restrict__ up_rs,
        const float * __restrict__ in_col_scales,
        const float * __restrict__ signs,
        int8_t * __restrict__ dst_i8, float * __restrict__ out_col_scales,
        const int M, const int N) {
    const int j = blockIdx.x;
    if (j >= N) {
        return;
    }
    const int tid = threadIdx.x; // 0 .. H-1
    const int n_blk = M / H;
    const float cs = in_col_scales[j];
    const int32_t * gc_col = gate_i32 + (int64_t) j * M;
    const int32_t * uc_col = up_i32   + (int64_t) j * M;

    __shared__ float s[H];
    constexpr int MAX_BLK = 36; // supports M <= 36 * 512 = 18432
    float reg[MAX_BLK];
    auto silu = [](float x) { return x / (1.0f + __expf(-x)); };
    float cmax = 0.0f;

    for (int b = 0; b < n_blk; ++b) {
        const int idx = b * H + tid;
        float v = silu((float) gc_col[idx] * (gate_rs[idx] * cs)) *
                  ((float) uc_col[idx] * (up_rs[idx] * cs)) * signs[idx];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float v2 = __shfl_xor_sync(0xFFFFFFFF, v, h);
            v = ((tid & h) == 0) ? (v + v2) : (v2 - v);
        }
#pragma unroll
        for (int h = 32; h < H; h <<= 1) {
            s[tid] = v;
            __syncthreads();
            const float v2 = s[tid ^ h];
            v = ((tid & h) == 0) ? (v + v2) : (v2 - v);
            __syncthreads();
        }
        reg[b] = v;
        cmax = fmaxf(cmax, fabsf(v));
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    __shared__ float smax[16];
    constexpr int nwarps = H / 32;
    if ((tid & 31) == 0) {
        smax[tid >> 5] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = (tid < nwarps) ? smax[tid] : 0.0f;
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    const float max_v = smax[0];
    const float fwht_scale = rsqrtf((float) H);
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        out_col_scales[j] = (max_v / 127.0f) * fwht_scale;
    }

    int8_t * out_col = dst_i8 + (int64_t) j * M;
    for (int b = 0; b < n_blk; ++b) {
        out_col[b * H + tid] = (int8_t) __float2int_rn(reg[b] * inv_scale);
    }
}

// Lossless warp-parallel ZERO-SMEM-TRANSPOSE GDN build_norm_gated (+ 4D head permute) + FWHT(1024) -> B_i8!
template <int N_BLK, bool TO_S4 = false>
__global__ static void gdn_norm_gated_fwht1024_to_i8_kernel(
        const float * __restrict__ src,
        const float * __restrict__ rms_w,
        const float eps,
        const float * __restrict__ z_gate,
        const float * __restrict__ signs,
        int8_t * __restrict__ dst_i8,
        float * __restrict__ out_col_scales,
        const int N, const int N_pad, const int perm_nk, const int perm_rep, const float clip_sigma = 0.0f) {
    constexpr int M = N_BLK * 1024;
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid  = threadIdx.x; // 0 .. N_BLK*32-1
    const int warp = tid >> 5;
    const int lane = tid & 31;
    uint32_t * out32 = TO_S4 ? ((uint32_t *) dst_i8 + (int64_t) j * (M >> 3))
                             : ((uint32_t *) (dst_i8 + (int64_t) j * M));
    if (j >= N) {
        if (tid == 0) { out_col_scales[j] = 0.0f; }
        const int n_w = TO_S4 ? (M >> 3) : (M >> 2);
        for (int i = tid; i < n_w; i += N_BLK * 32) { out32[i] = 0u; }
        return;
    }
    const float * s_col = src    + (int64_t) j * M;
    const float * z_col = z_gate + (int64_t) j * M;
    const int n_heads_blk = 8; // 1024 / 128 = 8 heads per warp
    float v[32];
    const int h_base = warp * n_heads_blk;
#pragma unroll
    for (int h = 0; h < n_heads_blk; ++h) {
        const int out_h = h_base + h;
        const int in_h  = (perm_rep > 1) ? ((out_h % perm_rep) * perm_nk + (out_h / perm_rep)) : out_h;
        const int in_base  = in_h  * 128;
        const int out_base = out_h * 128;
        float x0 = s_col[in_base + lane +  0];
        float x1 = s_col[in_base + lane + 32];
        float x2 = s_col[in_base + lane + 64];
        float x3 = s_col[in_base + lane + 96];
        float ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { ssq += __shfl_xor_sync(0xFFFFFFFF, ssq, off); }
        const float irms = rsqrtf(ssq * (1.0f / 128.0f) + eps);
        const float g0 = z_col[in_base + lane +  0];
        const float g1 = z_col[in_base + lane + 32];
        const float g2 = z_col[in_base + lane + 64];
        const float g3 = z_col[in_base + lane + 96];
        v[h * 4 + 0] = (x0 * irms * rms_w[lane +  0]) * (g0 / (1.0f + __expf(-g0))) * signs[out_base + lane +  0];
        v[h * 4 + 1] = (x1 * irms * rms_w[lane + 32]) * (g1 / (1.0f + __expf(-g1))) * signs[out_base + lane + 32];
        v[h * 4 + 2] = (x2 * irms * rms_w[lane + 64]) * (g2 / (1.0f + __expf(-g2))) * signs[out_base + lane + 64];
        v[h * 4 + 3] = (x3 * irms * rms_w[lane + 96]) * (g3 / (1.0f + __expf(-g3))) * signs[out_base + lane + 96];
    }
    // 2D FWHT(1024) = H_32(in-reg) kron H_32(warp-shfl), zero smem!
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 32; k += (h << 1)) {
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a = v[k + r], b = v[k + r + h];
                v[k + r] = a + b; v[k + r + h] = a - b;
            }
        }
    }
    float cmax = 0.0f, csq = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x = v[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float x2 = __shfl_xor_sync(0xFFFFFFFF, x, h);
            x = ((lane & h) == 0) ? (x + x2) : (x2 - x);
        }
        v[k] = x;
        cmax = fmaxf(cmax, fabsf(x));
        if constexpr (TO_S4) { csq += x * x; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
        if constexpr (TO_S4) { csq += __shfl_xor_sync(0xFFFFFFFF, csq, off); }
    }
    __shared__ float smax[N_BLK], ssq_sh[N_BLK];
    if (lane == 0) { smax[warp] = cmax; if constexpr (TO_S4) { ssq_sh[warp] = csq; } }
    __syncthreads();
    float max_v = 0.0f, tot_sq = 0.0f;
#pragma unroll
    for (int w = 0; w < N_BLK; ++w) { max_v = fmaxf(max_v, smax[w]); if constexpr (TO_S4) { tot_sq += ssq_sh[w]; } }
    if constexpr (TO_S4) {
        const float rms = sqrtf(tot_sq / (float) M);
        const float c   = (clip_sigma > 0.0f) ? fminf(max_v, clip_sigma * rms) : max_v;
        const float inv_scale = c > 0.0f ? (7.0f / c) : 0.0f;
        if (tid == 0) { out_col_scales[j] = (c * (1.0f / 7.0f)) * (1.0f / 32.0f); }
        uint32_t * out_blk = out32 + warp * 128;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t q = (uint32_t)(max(-8, min(7, __float2int_rn(v[k] * inv_scale))) & 0xF);
            q |= __shfl_down_sync(0xFFFFFFFF, q, 1) << 4;
            q |= __shfl_down_sync(0xFFFFFFFF, q, 2) << 8;
            q |= __shfl_down_sync(0xFFFFFFFF, q, 4) << 16;
            if ((lane & 7) == 0) { out_blk[k * 4 + (lane >> 3)] = q; }
        }
    } else {
        const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
        if (tid == 0) { out_col_scales[j] = (max_v / 127.0f) * (1.0f / 32.0f); }
        uint32_t * out_blk = out32 + warp * 256;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t q = (uint32_t)(uint8_t)(int8_t) __float2int_rn(v[k] * inv_scale);
            q |= __shfl_down_sync(0xFFFFFFFF, q, 1) << 8;
            q |= __shfl_down_sync(0xFFFFFFFF, q, 2) << 16;
            if ((lane & 3) == 0) { out_blk[k * 8 + (lane >> 2)] = q; }
        }
    }
}

// Lossless warp-parallel ZERO-SMEM-TRANSPOSE Full-Attn Gate: SIGMOID(gate) * attn_out * signs + FWHT(1024) -> B_i8 / B_s4!
template <int N_BLK, bool TO_S4 = false>
__global__ static void sigmoid_mul_fwht1024_to_i8_kernel(
        const float * __restrict__ attn_out,
        const float * __restrict__ gate,
        const float * __restrict__ signs,
        int8_t * __restrict__ dst_i8,
        float * __restrict__ out_col_scales,
        const int N, const int N_pad, const bool strided_gate, const float clip_sigma = 0.0f) {
    constexpr int M = N_BLK * 1024;
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    uint32_t * out32 = TO_S4 ? ((uint32_t *) dst_i8 + (int64_t) j * (M >> 3))
                             : ((uint32_t *) (dst_i8 + (int64_t) j * M));
    if (j >= N) {
        if (tid == 0) { out_col_scales[j] = 0.0f; }
        const int n_w = TO_S4 ? (M >> 3) : (M >> 2);
        for (int i = tid; i < n_w; i += N_BLK * 32) { out32[i] = 0u; }
        return;
    }
    const float * a_col = attn_out + (int64_t) j * M;
    const float * g_col = strided_gate ? (gate + (int64_t) j * (M * 2)) : (gate + (int64_t) j * M);
    float v[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int idx = warp * 1024 + k * 32 + lane;
        const int g_idx = strided_gate ? (((idx >> 8) * 512) + 256 + (idx & 255)) : idx;
        const float gv = g_col[g_idx];
        v[k] = a_col[idx] * (1.0f / (1.0f + __expf(-gv))) * signs[idx];
    }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 32; k += (h << 1)) {
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a = v[k + r], b = v[k + r + h];
                v[k + r] = a + b; v[k + r + h] = a - b;
            }
        }
    }
    float cmax = 0.0f, csq = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x = v[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float x2 = __shfl_xor_sync(0xFFFFFFFF, x, h);
            x = ((lane & h) == 0) ? (x + x2) : (x2 - x);
        }
        v[k] = x;
        cmax = fmaxf(cmax, fabsf(x));
        if constexpr (TO_S4) { csq += x * x; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
        if constexpr (TO_S4) { csq += __shfl_xor_sync(0xFFFFFFFF, csq, off); }
    }
    __shared__ float smax[N_BLK], ssq_sh[N_BLK];
    if (lane == 0) { smax[warp] = cmax; if constexpr (TO_S4) { ssq_sh[warp] = csq; } }
    __syncthreads();
    float max_v = 0.0f, tot_sq = 0.0f;
#pragma unroll
    for (int w = 0; w < N_BLK; ++w) { max_v = fmaxf(max_v, smax[w]); if constexpr (TO_S4) { tot_sq += ssq_sh[w]; } }
    if constexpr (TO_S4) {
        const float rms = sqrtf(tot_sq / (float) M);
        const float c   = (clip_sigma > 0.0f) ? fminf(max_v, clip_sigma * rms) : max_v;
        const float inv_scale = c > 0.0f ? (7.0f / c) : 0.0f;
        if (tid == 0) { out_col_scales[j] = (c * (1.0f / 7.0f)) * (1.0f / 32.0f); }
        uint32_t * out_blk = out32 + warp * 128;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t q = (uint32_t)(max(-8, min(7, __float2int_rn(v[k] * inv_scale))) & 0xF);
            q |= __shfl_down_sync(0xFFFFFFFF, q, 1) << 4;
            q |= __shfl_down_sync(0xFFFFFFFF, q, 2) << 8;
            q |= __shfl_down_sync(0xFFFFFFFF, q, 4) << 16;
            if ((lane & 7) == 0) { out_blk[k * 4 + (lane >> 3)] = q; }
        }
    } else {
        const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
        if (tid == 0) { out_col_scales[j] = (max_v / 127.0f) * (1.0f / 32.0f); }
        uint32_t * out_blk = out32 + warp * 256;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            uint32_t q = (uint32_t)(uint8_t)(int8_t) __float2int_rn(v[k] * inv_scale);
            q |= __shfl_down_sync(0xFFFFFFFF, q, 1) << 8;
            q |= __shfl_down_sync(0xFFFFFFFF, q, 2) << 16;
            if ((lane & 3) == 0) { out_blk[k * 8 + (lane >> 2)] = q; }
        }
    }
}

static thread_local std::unordered_map<const ggml_tensor *, const ggml_tensor *> tl_skipped_gate_cont_i8;

template <int N_BLK>
__global__ static void gdn_norm_gated_fwht1024_to_narrow_i8_kernel(
        const float * __restrict__ src, const float * __restrict__ rms_w, const float eps,
        const float * __restrict__ z_gate, const float * __restrict__ signs,
        uint32_t * __restrict__ dst_i8, int32_t * __restrict__ dst_sum128, float * __restrict__ dst_scales128,
        const int N, const int N_pad, const int perm_nk, const int perm_rep) {
    constexpr int M = N_BLK * 1024;
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int K4 = M >> 2, n_kblk = M >> 7;
    uint32_t * out32  = dst_i8        + (int64_t) j * K4;
    int32_t  * out_s  = dst_sum128    + (int64_t) j * n_kblk;
    float    * out_sc = dst_scales128 + (int64_t) j * n_kblk;
    if (j >= N) {
        for (int i = tid; i < K4; i += N_BLK * 32) { out32[i] = 0u; }
        for (int i = tid; i < n_kblk; i += N_BLK * 32) { out_s[i] = 0; out_sc[i] = 0.0f; }
        return;
    }
    const float * s_col = src    + (int64_t) j * M;
    const float * z_col = z_gate + (int64_t) j * M;
    float v[32];
    const int h_base = warp * 8;
#pragma unroll
    for (int h = 0; h < 8; ++h) {
        const int out_h = h_base + h;
        const int in_h  = (perm_rep > 1) ? ((out_h % perm_rep) * perm_nk + (out_h / perm_rep)) : out_h;
        const int in_base = in_h * 128, out_base = out_h * 128;
        const float x0 = s_col[in_base + lane +  0], x1 = s_col[in_base + lane + 32];
        const float x2 = s_col[in_base + lane + 64], x3 = s_col[in_base + lane + 96];
        float ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { ssq += __shfl_xor_sync(0xFFFFFFFF, ssq, off); }
        const float irms = rsqrtf(ssq * (1.0f / 128.0f) + eps);
        const float g0 = z_col[in_base + lane +  0], g1 = z_col[in_base + lane + 32];
        const float g2 = z_col[in_base + lane + 64], g3 = z_col[in_base + lane + 96];
        v[h * 4 + 0] = (x0 * irms * rms_w[lane +  0]) * __fdividef(g0, 1.0f + __expf(-g0)) * signs[out_base + lane +  0];
        v[h * 4 + 1] = (x1 * irms * rms_w[lane + 32]) * __fdividef(g1, 1.0f + __expf(-g1)) * signs[out_base + lane + 32];
        v[h * 4 + 2] = (x2 * irms * rms_w[lane + 64]) * __fdividef(g2, 1.0f + __expf(-g2)) * signs[out_base + lane + 64];
        v[h * 4 + 3] = (x3 * irms * rms_w[lane + 96]) * __fdividef(g3, 1.0f + __expf(-g3)) * signs[out_base + lane + 96];
    }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int k = 0; k < 32; k += (h << 1))
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a = v[k + r], b = v[k + r + h];
                v[k + r] = a + b; v[k + r + h] = a - b;
            }
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x = v[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float x2 = __shfl_xor_sync(0xFFFFFFFF, x, h);
            x = ((lane & h) == 0) ? (x + x2) : (x2 - x);
        }
        v[k] = x;
    }
    fwht1024_pack_2nibble_warp(v, warp, lane, out32, out_s, out_sc);
}

static bool ggml_cuda_gdn_norm_gated_fwht1024_to_i8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * rms_w, float eps,
        const ggml_tensor * z_gate, const ggml_tensor * signs, const ggml_tensor * fwht_out,
        int64_t M, int64_t N, int perm_nk, int perm_rep) {
    if (M != 6144 && M != 4096) { return false; }
    cudaStream_t stream = ctx.stream();
    if (ggml_cuda_pq2_narrow_int8_enabled() && N >= 5 && N <= pq2_narrow_i8_max_n()) {
        auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
        const int64_t N_pad = pq2_narrow_i8_n_pad(N);
        if (!pq2_narrow_i8_alloc_bufs(sc, N_pad, M, stream)) { return false; }
        if (M == 6144) {
            gdn_norm_gated_fwht1024_to_narrow_i8_kernel<6><<<(unsigned) N_pad, 6 * 32, 0, stream>>>(
                (const float *) src->data, (const float *) rms_w->data, eps,
                (const float *) z_gate->data, (const float *) signs->data,
                sc.B_i8, sc.B_sum, sc.B_scales, (int) N, (int) N_pad, perm_nk, perm_rep);
        } else {
            gdn_norm_gated_fwht1024_to_narrow_i8_kernel<4><<<(unsigned) N_pad, 4 * 32, 0, stream>>>(
                (const float *) src->data, (const float *) rms_w->data, eps,
                (const float *) z_gate->data, (const float *) signs->data,
                sc.B_i8, sc.B_sum, sc.B_scales, (int) N, (int) N_pad, perm_nk, perm_rep);
        }
        sc.src1_data = fwht_out->data; sc.src1_nbytes = ggml_nbytes(fwht_out);
        sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true; ++sc.gen;
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (ggml_cuda_pq2_prefill_int4_enabled() && N >= 256) {
        auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
        const int64_t N_pad = ((N + 127) / 128) * 128;
        const size_t need_b = (size_t) N_pad * (size_t) (M >> 3) * sizeof(uint32_t);
        const size_t need_s = (size_t) N_pad * sizeof(float);
        if (sc.cap_b < need_b)  { if (sc.B_i4)     CUDA_CHECK(cudaFree(sc.B_i4));     CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));     sc.cap_b = need_b; }
        if (sc.cap_sc < need_s) { if (sc.B_scales) CUDA_CHECK(cudaFree(sc.B_scales)); CUDA_CHECK(cudaMalloc(&sc.B_scales, need_s)); sc.cap_sc = need_s; }
        const float clip = ggml_cuda_pq2_prefill_int4_clip();
        if (M == 6144) {
            gdn_norm_gated_fwht1024_to_i8_kernel<6, true><<<(unsigned) N_pad, 6 * 32, 0, stream>>>(
                (const float *) src->data, (const float *) rms_w->data, eps,
                (const float *) z_gate->data, (const float *) signs->data,
                (int8_t *) sc.B_i4, sc.B_scales, (int) N, (int) N_pad, perm_nk, perm_rep, clip);
        } else {
            gdn_norm_gated_fwht1024_to_i8_kernel<4, true><<<(unsigned) N_pad, 4 * 32, 0, stream>>>(
                (const float *) src->data, (const float *) rms_w->data, eps,
                (const float *) z_gate->data, (const float *) signs->data,
                (int8_t *) sc.B_i4, sc.B_scales, (int) N, (int) N_pad, perm_nk, perm_rep, clip);
        }
        sc.src1_data = fwht_out->data; sc.src1_nbytes = ggml_nbytes(fwht_out);
        sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true; ++sc.gen;
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (!ggml_cuda_pq2_cublas_i8_enabled() || !pq2_b_i8_cache_enabled() || N < 129) { return false; }
    auto & st = pq2_b_i8_cache_for_stream(stream);
    const int64_t N_pad = (N + 3) & ~int64_t(3);
    const size_t need_b = (size_t) N_pad * (size_t) M;
    const size_t need_s = (size_t) N_pad * sizeof(float);
    if (st.B_i8_cap < need_b) { if (st.B_i8) CUDA_CHECK(cudaFree(st.B_i8)); CUDA_CHECK(cudaMalloc(&st.B_i8, need_b)); st.B_i8_cap = need_b; }
    if (st.col_scales_cap < need_s) { if (st.col_scales) CUDA_CHECK(cudaFree(st.col_scales)); CUDA_CHECK(cudaMalloc(&st.col_scales, need_s)); st.col_scales_cap = need_s; }
    if (M == 6144) {
        gdn_norm_gated_fwht1024_to_i8_kernel<6, false><<<(unsigned) N_pad, 6 * 32, 0, stream>>>(
            (const float *) src->data, (const float *) rms_w->data, eps,
            (const float *) z_gate->data, (const float *) signs->data,
            st.B_i8, st.col_scales, (int) N, (int) N_pad, perm_nk, perm_rep);
    } else {
        gdn_norm_gated_fwht1024_to_i8_kernel<4, false><<<(unsigned) N_pad, 4 * 32, 0, stream>>>(
            (const float *) src->data, (const float *) rms_w->data, eps,
            (const float *) z_gate->data, (const float *) signs->data,
            st.B_i8, st.col_scales, (int) N, (int) N_pad, perm_nk, perm_rep);
    }
    st.src1_data = fwht_out->data; st.src1_nbytes = ggml_nbytes(fwht_out);
    st.N = N; st.K = M; st.s11 = M; st.valid = true;
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int N_BLK>
__global__ static void sigmoid_mul_fwht1024_to_narrow_i8_kernel(
        const float * __restrict__ attn_out, const float * __restrict__ gate, const float * __restrict__ signs,
        uint32_t * __restrict__ dst_i8, int32_t * __restrict__ dst_sum128, float * __restrict__ dst_scales128,
        const int N, const int N_pad, const bool strided_gate) {
    constexpr int M = N_BLK * 1024;
    const int j = blockIdx.x;
    if (j >= N_pad) { return; }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int K4 = M >> 2, n_kblk = M >> 7;
    uint32_t * out32  = dst_i8        + (int64_t) j * K4;
    int32_t  * out_s  = dst_sum128    + (int64_t) j * n_kblk;
    float    * out_sc = dst_scales128 + (int64_t) j * n_kblk;
    if (j >= N) {
        for (int i = tid; i < K4; i += N_BLK * 32) { out32[i] = 0u; }
        for (int i = tid; i < n_kblk; i += N_BLK * 32) { out_s[i] = 0; out_sc[i] = 0.0f; }
        return;
    }
    const float * a_col = attn_out + (int64_t) j * M;
    const float * g_col = gate     + (int64_t) j * (strided_gate ? (2 * M) : M);
    float v[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        const int idx = warp * 1024 + k * 32 + lane;
        const int g_idx = strided_gate ? ((idx >> 8) * 512 + 256 + (idx & 255)) : idx;
        const float g = g_col[g_idx];
        v[k] = a_col[idx] * __fdividef(1.0f, 1.0f + __expf(-g)) * __ldg(signs + idx);
    }
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int k = 0; k < 32; k += (h << 1))
#pragma unroll
            for (int r = 0; r < h; ++r) {
                const float a = v[k + r], b = v[k + r + h];
                v[k + r] = a + b; v[k + r + h] = a - b;
            }
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        float x = v[k];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float x2 = __shfl_xor_sync(0xFFFFFFFF, x, h);
            x = ((lane & h) == 0) ? (x + x2) : (x2 - x);
        }
        v[k] = x;
    }
    fwht1024_pack_2nibble_warp(v, warp, lane, out32, out_s, out_sc);
}

static bool ggml_cuda_sigmoid_mul_fwht1024_to_i8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * attn_out, const ggml_tensor * gate,
        const ggml_tensor * signs, const ggml_tensor * fwht_out, int64_t M, int64_t N, bool strided_gate) {
    if (M != 6144 && M != 4096) { return false; }
    cudaStream_t stream = ctx.stream();
    if (ggml_cuda_pq2_narrow_int8_enabled() && N >= 5 && N <= pq2_narrow_i8_max_n()) {
        auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
        const int64_t N_pad = pq2_narrow_i8_n_pad(N);
        if (!pq2_narrow_i8_alloc_bufs(sc, N_pad, M, stream)) { return false; }
        if (M == 6144) {
            sigmoid_mul_fwht1024_to_narrow_i8_kernel<6><<<(unsigned) N_pad, 6 * 32, 0, stream>>>(
                (const float *) attn_out->data, (const float *) gate->data, (const float *) signs->data,
                sc.B_i8, sc.B_sum, sc.B_scales, (int) N, (int) N_pad, strided_gate);
        } else {
            sigmoid_mul_fwht1024_to_narrow_i8_kernel<4><<<(unsigned) N_pad, 4 * 32, 0, stream>>>(
                (const float *) attn_out->data, (const float *) gate->data, (const float *) signs->data,
                sc.B_i8, sc.B_sum, sc.B_scales, (int) N, (int) N_pad, strided_gate);
        }
        sc.src1_data = fwht_out->data; sc.src1_nbytes = ggml_nbytes(fwht_out);
        sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true; ++sc.gen;
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (ggml_cuda_pq2_prefill_int4_enabled() && N >= 256) {
        auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
        const int64_t N_pad = ((N + 127) / 128) * 128;
        const size_t need_b = (size_t) N_pad * (size_t) (M >> 3) * sizeof(uint32_t);
        const size_t need_s = (size_t) N_pad * sizeof(float);
        if (sc.cap_b < need_b)  { if (sc.B_i4)     CUDA_CHECK(cudaFree(sc.B_i4));     CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));     sc.cap_b = need_b; }
        if (sc.cap_sc < need_s) { if (sc.B_scales) CUDA_CHECK(cudaFree(sc.B_scales)); CUDA_CHECK(cudaMalloc(&sc.B_scales, need_s)); sc.cap_sc = need_s; }
        const float clip = ggml_cuda_pq2_prefill_int4_clip();
        if (M == 6144) {
            sigmoid_mul_fwht1024_to_i8_kernel<6, true><<<(unsigned) N_pad, 6 * 32, 0, stream>>>(
                (const float *) attn_out->data, (const float *) gate->data, (const float *) signs->data,
                (int8_t *) sc.B_i4, sc.B_scales, (int) N, (int) N_pad, strided_gate, clip);
        } else {
            sigmoid_mul_fwht1024_to_i8_kernel<4, true><<<(unsigned) N_pad, 4 * 32, 0, stream>>>(
                (const float *) attn_out->data, (const float *) gate->data, (const float *) signs->data,
                (int8_t *) sc.B_i4, sc.B_scales, (int) N, (int) N_pad, strided_gate, clip);
        }
        sc.src1_data = fwht_out->data; sc.src1_nbytes = ggml_nbytes(fwht_out);
        sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true; ++sc.gen;
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (!ggml_cuda_pq2_cublas_i8_enabled() || !pq2_b_i8_cache_enabled() || N < 129) { return false; }
    auto & st = pq2_b_i8_cache_for_stream(stream);
    const int64_t N_pad = (N + 3) & ~int64_t(3);
    const size_t need_b = (size_t) N_pad * (size_t) M;
    const size_t need_s = (size_t) N_pad * sizeof(float);
    if (st.B_i8_cap < need_b) { if (st.B_i8) CUDA_CHECK(cudaFree(st.B_i8)); CUDA_CHECK(cudaMalloc(&st.B_i8, need_b)); st.B_i8_cap = need_b; }
    if (st.col_scales_cap < need_s) { if (st.col_scales) CUDA_CHECK(cudaFree(st.col_scales)); CUDA_CHECK(cudaMalloc(&st.col_scales, need_s)); st.col_scales_cap = need_s; }
    if (M == 6144) {
        sigmoid_mul_fwht1024_to_i8_kernel<6, false><<<(unsigned) N_pad, 6 * 32, 0, stream>>>(
            (const float *) attn_out->data, (const float *) gate->data, (const float *) signs->data,
            st.B_i8, st.col_scales, (int) N, (int) N_pad, strided_gate);
    } else {
        sigmoid_mul_fwht1024_to_i8_kernel<4, false><<<(unsigned) N_pad, 4 * 32, 0, stream>>>(
            (const float *) attn_out->data, (const float *) gate->data, (const float *) signs->data,
            st.B_i8, st.col_scales, (int) N, (int) N_pad, strided_gate);
    }
    st.src1_data = fwht_out->data; st.src1_nbytes = ggml_nbytes(fwht_out);
    st.N = N; st.K = M; st.s11 = M; st.valid = true;
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// Fused MUL(Hadamard signs) + FWHT(H) + col-wise INT8 quantization directly into B_i8 cache!
template <int H>
__global__ static void mul_signs_fwht_to_i8_kernel(
        const float * __restrict__ src, const float * __restrict__ signs,
        int8_t * __restrict__ dst_i8, float * __restrict__ out_col_scales,
        const int M, const int N, const int64_t s11) {
    const int j = blockIdx.x;
    if (j >= N) {
        return;
    }
    const int tid = threadIdx.x; // 0 .. H-1
    const int n_blk = M / H;
    const float * in_col = src + (int64_t) j * s11;

    __shared__ float s[H];
    constexpr int MAX_BLK = 36;
    float reg[MAX_BLK];
    float cmax = 0.0f;

    for (int b = 0; b < n_blk; ++b) {
        const int idx = b * H + tid;
        float v = in_col[idx] * signs[idx];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float v2 = __shfl_xor_sync(0xFFFFFFFF, v, h);
            v = ((tid & h) == 0) ? (v + v2) : (v2 - v);
        }
#pragma unroll
        for (int h = 32; h < H; h <<= 1) {
            s[tid] = v;
            __syncthreads();
            const float v2 = s[tid ^ h];
            v = ((tid & h) == 0) ? (v + v2) : (v2 - v);
            __syncthreads();
        }
        reg[b] = v;
        cmax = fmaxf(cmax, fabsf(v));
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    __shared__ float smax[16];
    constexpr int nwarps = H / 32;
    if ((tid & 31) == 0) {
        smax[tid >> 5] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = (tid < nwarps) ? smax[tid] : 0.0f;
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    const float max_v = smax[0];
    const float fwht_scale = rsqrtf((float) H);
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        out_col_scales[j] = (max_v / 127.0f) * fwht_scale;
    }

    int8_t * out_col = dst_i8 + (int64_t) j * M;
    for (int b = 0; b < n_blk; ++b) {
        out_col[b * H + tid] = (int8_t) __float2int_rn(reg[b] * inv_scale);
    }
}

// H=1024 specialization using H_1024 = H_32 kron H_32 (512 threads = 16 warps, 2 elems/thread):
// Only 1 __syncthreads() per 1024-block in the main loop (transposed registers),
// and coalesced 32-bit STG.32 (int32_t / 4x int8_t) stores at the end!
__global__ static void dequant_i32_swiglu_fwht1024_to_i8_kernel(
        const int32_t * __restrict__ gate_i32, const float * __restrict__ gate_rs,
        const int32_t * __restrict__ up_i32,   const float * __restrict__ up_rs,
        const float * __restrict__ in_col_scales,
        const float * __restrict__ signs,
        int8_t * __restrict__ dst_i8, float * __restrict__ out_col_scales,
        const int M, const int N) {
    const int j = blockIdx.x;
    if (j >= N) {
        return;
    }
    const int tid  = threadIdx.x; // 0 .. 511
    const int warp = tid >> 5;    // 0 .. 15
    const int lane = tid & 31;    // 0 .. 31
    const int n_blk = M >> 10;    // M / 1024
    const float cs = in_col_scales[j];
    const int32_t * gc_col = gate_i32 + (int64_t) j * M;
    const int32_t * uc_col = up_i32   + (int64_t) j * M;

    __shared__ float s[32][33];
    constexpr int MAX_BLK = 18; // supports M <= 18432
    float reg0[MAX_BLK];
    float reg1[MAX_BLK];
    auto silu = [](float x) { return x / (1.0f + __expf(-x)); };
    float cmax = 0.0f;

    for (int b = 0; b < n_blk; ++b) {
        const int i0 = b * 1024 + tid;
        const int i1 = i0 + 512;
        float v0 = silu((float) gc_col[i0] * (gate_rs[i0] * cs)) *
                   ((float) uc_col[i0] * (up_rs[i0] * cs)) * signs[i0];
        float v1 = silu((float) gc_col[i1] * (gate_rs[i1] * cs)) *
                   ((float) uc_col[i1] * (up_rs[i1] * cs)) * signs[i1];
        // Stage 1: H_32 along columns (lane = 0..31)
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float a2 = __shfl_xor_sync(0xFFFFFFFF, v0, h);
            const float b2 = __shfl_xor_sync(0xFFFFFFFF, v1, h);
            v0 = ((lane & h) == 0) ? (v0 + a2) : (a2 - v0);
            v1 = ((lane & h) == 0) ? (v1 + b2) : (b2 - v1);
        }
        // Transpose into s[col][row]
        s[lane][warp]      = v0;
        s[lane][warp + 16] = v1;
        __syncthreads();
        // Stage 2: H_32 along rows (lane = row 0..31, warp/warp+16 = col)
        float u0 = s[warp][lane];
        float u1 = s[warp + 16][lane];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float a2 = __shfl_xor_sync(0xFFFFFFFF, u0, h);
            const float b2 = __shfl_xor_sync(0xFFFFFFFF, u1, h);
            u0 = ((lane & h) == 0) ? (u0 + a2) : (a2 - u0);
            u1 = ((lane & h) == 0) ? (u1 + b2) : (b2 - u1);
        }
        reg0[b] = u0;
        reg1[b] = u1;
        cmax = fmaxf(cmax, fmaxf(fabsf(u0), fabsf(u1)));
        __syncthreads();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    __shared__ float smax[16];
    if (lane == 0) {
        smax[warp] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = smax[tid];
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    const float max_v = smax[0];
    const float fwht_scale = 1.0f / 32.0f; // 1 / sqrt(1024)
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        out_col_scales[j] = (max_v / 127.0f) * fwht_scale;
    }

    // Transpose quantized int8_t back via s_i8[32][36] and write coalesced 32-bit char4 (STG.32)!
    int8_t * s_i8 = (int8_t *) s; // 32 * 36 = 1152 bytes <= 4224 bytes of s!
    int32_t * out32 = (int32_t *) (dst_i8 + (int64_t) j * M);
    for (int b = 0; b < n_blk; ++b) {
        s_i8[lane * 36 + warp]      = (int8_t) __float2int_rn(reg0[b] * inv_scale);
        s_i8[lane * 36 + warp + 16] = (int8_t) __float2int_rn(reg1[b] * inv_scale);
        __syncthreads();
        if (tid < 256) {
            const int r = tid >> 3;      // 0..31
            const int c4 = (tid & 7) * 4; // 0, 4, ..., 28
            out32[b * 256 + tid] = *(const int32_t *) (s_i8 + r * 36 + c4);
        }
        __syncthreads();
    }
}

// 7-op super-fusion: dequant_i32 + ADD(residual) + RMS_NORM + MUL(rms_weight) [+ MUL(hadamard_signs) + FWHT(1024)] -> B_i8 cache!
// 1 CTA per column j, 512 threads, M=5120 (5 blocks of 1024 -> 10 floats/thread in registers!).
__global__ static void dequant_i32_add_rms_mul_fwht1024_to_i8_kernel(
        const int32_t * __restrict__ src_i32, const float * __restrict__ row_scales, const float * __restrict__ in_col_scales,
        const float * __restrict__ add_src, float * __restrict__ add_dst,
        const float * __restrict__ rms_weight, const float eps,
        const float * __restrict__ hadamard_signs,
        float * __restrict__ norm_f32_dst,
        int8_t * __restrict__ dst_i8, float * __restrict__ out_col_scales,
        const int M, const int N, const int64_t s_dst) {
    const int j = blockIdx.x;
    if (j >= N) {
        return;
    }
    const int tid  = threadIdx.x; // 0..511
    const int warp = tid >> 5;    // 0..15
    const int lane = tid & 31;    // 0..31
    const int n_blk = M >> 10;    // M / 1024 (e.g. 5 for 5120)
    const float cs = in_col_scales[j];
    const int32_t * c_col  = src_i32 + (int64_t) j * M;
    const float   * a_col  = add_src + (int64_t) j * s_dst;
    float         * ad_col = add_dst + (int64_t) j * s_dst;

    constexpr int MAX_BLK = 8; // supports M <= 8192
    float reg0[MAX_BLK];
    float reg1[MAX_BLK];
    float sum_sq = 0.0f;

    for (int b = 0; b < n_blk; ++b) {
        const int i0 = b * 1024 + tid;
        const int i1 = i0 + 512;
        const float v0 = (float) c_col[i0] * (row_scales[i0] * cs) + a_col[i0];
        const float v1 = (float) c_col[i1] * (row_scales[i1] * cs) + a_col[i1];
        ad_col[i0] = v0;
        ad_col[i1] = v1;
        reg0[b] = v0;
        reg1[b] = v1;
        sum_sq += v0 * v0 + v1 * v1;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    }
    __shared__ float sbuf[16];
    if (lane == 0) {
        sbuf[warp] = sum_sq;
    }
    __syncthreads();
    if (tid < 16) {
        float v = sbuf[tid];
        v += __shfl_xor_sync(0xFFFF, v, 8);
        v += __shfl_xor_sync(0xFFFF, v, 4);
        v += __shfl_xor_sync(0xFFFF, v, 2);
        v += __shfl_xor_sync(0xFFFF, v, 1);
        sbuf[0] = v;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(sbuf[0] / (float) M + eps);

    __shared__ float s[32][33];
    float cmax = 0.0f;
    for (int b = 0; b < n_blk; ++b) {
        const int i0 = b * 1024 + tid;
        const int i1 = i0 + 512;
        float v0 = reg0[b] * inv_rms * rms_weight[i0];
        float v1 = reg1[b] * inv_rms * rms_weight[i1];
        if (norm_f32_dst) {
            norm_f32_dst[(int64_t) j * s_dst + i0] = v0;
            norm_f32_dst[(int64_t) j * s_dst + i1] = v1;
        }
        v0 *= hadamard_signs[i0];
        v1 *= hadamard_signs[i1];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float a2 = __shfl_xor_sync(0xFFFFFFFF, v0, h);
            const float b2 = __shfl_xor_sync(0xFFFFFFFF, v1, h);
            v0 = ((lane & h) == 0) ? (v0 + a2) : (a2 - v0);
            v1 = ((lane & h) == 0) ? (v1 + b2) : (b2 - v1);
        }
        s[lane][warp]      = v0;
        s[lane][warp + 16] = v1;
        __syncthreads();
        float u0 = s[warp][lane];
        float u1 = s[warp + 16][lane];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float a2 = __shfl_xor_sync(0xFFFFFFFF, u0, h);
            const float b2 = __shfl_xor_sync(0xFFFFFFFF, u1, h);
            u0 = ((lane & h) == 0) ? (u0 + a2) : (a2 - u0);
            u1 = ((lane & h) == 0) ? (u1 + b2) : (b2 - u1);
        }
        reg0[b] = u0;
        reg1[b] = u1;
        cmax = fmaxf(cmax, fmaxf(fabsf(u0), fabsf(u1)));
        __syncthreads();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    if (lane == 0) {
        sbuf[warp] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = sbuf[tid];
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        sbuf[0] = v;
    }
    __syncthreads();
    const float max_v = sbuf[0];
    const float fwht_scale = 1.0f / 32.0f; // 1 / sqrt(1024)
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        out_col_scales[j] = (max_v / 127.0f) * fwht_scale;
    }

    int8_t * s_i8 = (int8_t *) s;
    int32_t * out32 = (int32_t *) (dst_i8 + (int64_t) j * M);
    for (int b = 0; b < n_blk; ++b) {
        s_i8[lane * 36 + warp]      = (int8_t) __float2int_rn(reg0[b] * inv_scale);
        s_i8[lane * 36 + warp + 16] = (int8_t) __float2int_rn(reg1[b] * inv_scale);
        __syncthreads();
        if (tid < 256) {
            const int r = tid >> 3;
            const int c4 = (tid & 7) * 4;
            out32[b * 256 + tid] = *(const int32_t *) (s_i8 + r * 36 + c4);
        }
        __syncthreads();
    }
}

// Fused dequant_i32 + ADD(residual) + RMS_NORM + MUL(weight) -> direct B_i8 cache!
// 1 CTA per column j, 512 threads, M <= 512 * 4 * 4 = 8192 (M=5120 -> 3 float4s/thread in registers!).
__global__ static void dequant_i32_add_rms_mul_to_i8_kernel(
        const int32_t * __restrict__ src_i32, const float * __restrict__ row_scales, const float * __restrict__ in_col_scales,
        const float * __restrict__ add_src, float * __restrict__ add_dst,
        const float * __restrict__ rms_weight, const float eps,
        float * __restrict__ norm_f32_dst,
        int8_t * __restrict__ dst_i8, float * __restrict__ out_col_scales,
        const int M, const int N, const int64_t s_dst) {
    const int j = blockIdx.x;
    if (j >= N) {
        return;
    }
    const int tid = threadIdx.x;
    const int M4 = M >> 2;
    const float cs = in_col_scales[j];
    const int4   * c4   = (const int4 *)   (src_i32 + (int64_t) j * M);
    const float4 * rs4  = (const float4 *) row_scales;
    const float4 * a4   = (const float4 *) (add_src + (int64_t) j * s_dst);
    float4       * ad4  = (float4 *)       (add_dst + (int64_t) j * s_dst);
    const float4 * w4   = (const float4 *) rms_weight;

    constexpr int MAX_REG = 4; // 512 * 4 * 4 = 8192 floats
    float4 reg[MAX_REG];
    float sum_sq = 0.0f;
#pragma unroll
    for (int r = 0; r < MAX_REG; ++r) {
        const int i = r * 512 + tid;
        if (i < M4) {
            const int4   c  = c4[i];
            const float4 rs = rs4[i];
            const float4 a  = a4[i];
            float4 v;
            v.x = (float) c.x * (rs.x * cs) + a.x;
            v.y = (float) c.y * (rs.y * cs) + a.y;
            v.z = (float) c.z * (rs.z * cs) + a.z;
            v.w = (float) c.w * (rs.w * cs) + a.w;
            ad4[i] = v;
            reg[r] = v;
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, off);
    }
    __shared__ float sbuf[16];
    if ((tid & 31) == 0) {
        sbuf[tid >> 5] = sum_sq;
    }
    __syncthreads();
    if (tid < 16) {
        float v = sbuf[tid];
        v += __shfl_xor_sync(0xFFFF, v, 8);
        v += __shfl_xor_sync(0xFFFF, v, 4);
        v += __shfl_xor_sync(0xFFFF, v, 2);
        v += __shfl_xor_sync(0xFFFF, v, 1);
        sbuf[0] = v;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(sbuf[0] / (float) M + eps);

    float cmax = 0.0f;
#pragma unroll
    for (int r = 0; r < MAX_REG; ++r) {
        const int i = r * 512 + tid;
        if (i < M4) {
            const float4 w = w4[i];
            float4 v = reg[r];
            v.x = v.x * inv_rms * w.x;
            v.y = v.y * inv_rms * w.y;
            v.z = v.z * inv_rms * w.z;
            v.w = v.w * inv_rms * w.w;
            reg[r] = v;
            if (norm_f32_dst) {
                ((float4 *) (norm_f32_dst + (int64_t) j * s_dst))[i] = v;
            }
            cmax = fmaxf(cmax, fabsf(v.x));
            cmax = fmaxf(cmax, fabsf(v.y));
            cmax = fmaxf(cmax, fabsf(v.z));
            cmax = fmaxf(cmax, fabsf(v.w));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    if ((tid & 31) == 0) {
        sbuf[tid >> 5] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = sbuf[tid];
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        sbuf[0] = v;
    }
    __syncthreads();
    const float max_v = sbuf[0];
    const float scale_col = max_v / 127.0f;
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        out_col_scales[j] = scale_col;
    }

    char4 * out4 = (char4 *) (dst_i8 + (int64_t) j * M);
#pragma unroll
    for (int r = 0; r < MAX_REG; ++r) {
        const int i = r * 512 + tid;
        if (i < M4) {
            const float4 v = reg[r];
            char4 q;
            q.x = (int8_t) __float2int_rn(v.x * inv_scale);
            q.y = (int8_t) __float2int_rn(v.y * inv_scale);
            q.z = (int8_t) __float2int_rn(v.z * inv_scale);
            q.w = (int8_t) __float2int_rn(v.w * inv_scale);
            out4[i] = q;
        }
    }
}

__global__ static void mul_signs_fwht1024_to_i8_kernel(
        const float * __restrict__ src, const float * __restrict__ signs,
        int8_t * __restrict__ dst_i8, float * __restrict__ out_col_scales,
        const int M, const int N, const int64_t s11) {
    const int j = blockIdx.x;
    if (j >= N) {
        return;
    }
    const int tid  = threadIdx.x; // 0 .. 511
    const int warp = tid >> 5;    // 0 .. 15
    const int lane = tid & 31;    // 0 .. 31
    const int n_blk = M >> 10;    // M / 1024
    const float * in_col = src + (int64_t) j * s11;

    __shared__ float s[32][33];
    constexpr int MAX_BLK = 18;
    float reg0[MAX_BLK];
    float reg1[MAX_BLK];
    float cmax = 0.0f;

    for (int b = 0; b < n_blk; ++b) {
        const int i0 = b * 1024 + tid;
        const int i1 = i0 + 512;
        float v0 = in_col[i0] * signs[i0];
        float v1 = in_col[i1] * signs[i1];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float a2 = __shfl_xor_sync(0xFFFFFFFF, v0, h);
            const float b2 = __shfl_xor_sync(0xFFFFFFFF, v1, h);
            v0 = ((lane & h) == 0) ? (v0 + a2) : (a2 - v0);
            v1 = ((lane & h) == 0) ? (v1 + b2) : (b2 - v1);
        }
        s[lane][warp]      = v0;
        s[lane][warp + 16] = v1;
        __syncthreads();
        float u0 = s[warp][lane];
        float u1 = s[warp + 16][lane];
#pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float a2 = __shfl_xor_sync(0xFFFFFFFF, u0, h);
            const float b2 = __shfl_xor_sync(0xFFFFFFFF, u1, h);
            u0 = ((lane & h) == 0) ? (u0 + a2) : (a2 - u0);
            u1 = ((lane & h) == 0) ? (u1 + b2) : (b2 - u1);
        }
        reg0[b] = u0;
        reg1[b] = u1;
        cmax = fmaxf(cmax, fmaxf(fabsf(u0), fabsf(u1)));
        __syncthreads();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmax = fmaxf(cmax, __shfl_xor_sync(0xFFFFFFFF, cmax, off));
    }
    __shared__ float smax[16];
    if (lane == 0) {
        smax[warp] = cmax;
    }
    __syncthreads();
    if (tid < 16) {
        float v = smax[tid];
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 8));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 4));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 2));
        v = fmaxf(v, __shfl_xor_sync(0xFFFF, v, 1));
        smax[0] = v;
    }
    __syncthreads();
    const float max_v = smax[0];
    const float fwht_scale = 1.0f / 32.0f;
    const float inv_scale = max_v > 0.0f ? (127.0f / max_v) : 0.0f;
    if (tid == 0) {
        out_col_scales[j] = (max_v / 127.0f) * fwht_scale;
    }

    int8_t * s_i8 = (int8_t *) s;
    int32_t * out32 = (int32_t *) (dst_i8 + (int64_t) j * M);
    for (int b = 0; b < n_blk; ++b) {
        s_i8[lane * 36 + warp]      = (int8_t) __float2int_rn(reg0[b] * inv_scale);
        s_i8[lane * 36 + warp + 16] = (int8_t) __float2int_rn(reg1[b] * inv_scale);
        __syncthreads();
        if (tid < 256) {
            const int r = tid >> 3;
            const int c4 = (tid & 7) * 4;
            out32[b * 256 + tid] = *(const int32_t *) (s_i8 + r * 36 + c4);
        }
        __syncthreads();
    }
}

// Fast thin F32 matmul for decode/verify (M <= 64, N <= 8, K >= 512, e.g. 5120x48 ssm_alpha/beta):
// 1 CTA per output row, reads src0 row ONCE as float4 and dots against all N columns in registers.
template <int N_COLS>
__global__ static void thin_f32_decode_kernel(
        const float * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int M, const int K, const int64_t s11, const int64_t s_dst) {
    const int row = blockIdx.x;
    if (row >= M) {
        return;
    }
    const int tid = threadIdx.x;
    const int K4 = K >> 2;
    const float4 * w4 = (const float4 *) (src0 + (int64_t) row * K);
    float acc[N_COLS] = {};
    for (int i = tid; i < K4; i += blockDim.x) {
        const float4 w = w4[i];
#pragma unroll
        for (int j = 0; j < N_COLS; ++j) {
            const float4 x = ((const float4 *) (src1 + (int64_t) j * s11))[i];
            acc[j] += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
        }
    }
    __shared__ float ssum[8][N_COLS];
#pragma unroll
    for (int j = 0; j < N_COLS; ++j) {
        float v = acc[j];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, off);
        }
        if ((tid & 31) == 0) {
            ssum[tid >> 5][j] = v;
        }
    }
    __syncthreads();
    if (tid < N_COLS) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            sum += ssum[w][tid];
        }
        dst[(int64_t) tid * s_dst + row] = sum;
    }
}

// P1.2: Fast thin F32 matmul for N = 9..64 (1 CTA per (row, 8-col tile)) - eliminates 96 cublasSgemm calls in -np 8 verify!
__global__ static void thin_f32_narrow_kernel(
        const float * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int M, const int N, const int K, const int64_t s11, const int64_t s_dst) {
    const int row = blockIdx.x;
    const int c0  = blockIdx.y * 8;
    if (row >= M || c0 >= N) { return; }
    const int n_c = min(8, N - c0);
    const int tid = threadIdx.x;
    const int K4  = K >> 2;
    const float4 * w4 = (const float4 *) (src0 + (int64_t) row * K);
    float acc[8] = {};
    for (int i = tid; i < K4; i += blockDim.x) {
        const float4 w = w4[i];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_c) {
                const float4 x = ((const float4 *) (src1 + (int64_t) (c0 + j) * s11))[i];
                acc[j] += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
            }
        }
    }
    __shared__ float ssum[8][8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = acc[j];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFF, v, off);
        }
        if ((tid & 31) == 0) {
            ssum[tid >> 5][j] = v;
        }
    }
    __syncthreads();
    if (tid < n_c) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            sum += ssum[w][tid];
        }
        dst[(int64_t) (c0 + tid) * s_dst + row] = sum;
    }
}

static bool ggml_cuda_mul_mat_pq2_cublas_i8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * add_src = nullptr,
        const ggml_tensor * rms_weight = nullptr, const float rms_eps = 0.0f,
        ggml_tensor * norm_mul_dst = nullptr, const bool skip_norm_f32_write = false,
        const ggml_tensor * fwht_signs = nullptr, const ggml_tensor * fwht_out = nullptr) {
    if (!ggml_cuda_pq2_cublas_i8_enabled()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_PQ2_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const int64_t min_n_cb = ggml_cuda_pq2_narrow_int8_enabled() ? (M > 65536 ? 256 : (pq2_narrow_i8_max_n() + 1)) : 129;
    if (N < min_n_cb || (K % QK_PQ2_0) != 0 || (M % 4) != 0 || !ggml_is_contiguous(src0) ||
            (src1->nb[0] != sizeof(float)) || (dst->nb[0] != sizeof(float))) {
        return false;
    }
    if (ggml_cuda_pq2_prefill_int4_enabled() && N >= 256 && M <= 65536) {
        return false;
    }
    if (add_src && (add_src->type != GGML_TYPE_F32 || !ggml_are_same_shape(add_src, dst) ||
            add_src->nb[0] != sizeof(float) || add_src->nb[1] != dst->nb[1])) {
        return false;
    }

    const int64_t N_pad = (N + 3) & ~int64_t(3);
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    (void) cudaStreamIsCapturing(stream, &cap_status);
    const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);

    const auto rs_entry = get_or_compute_pq2_row_scales((const block_pq2_0 *) src0->data, M, K, stream, allow_alloc);
    if (!rs_entry.scales || !rs_entry.inv_scales) {
        return false;
    }
    auto & b_st = pq2_b_i8_cache_for_stream(stream);
    if (!allow_alloc && (b_st.B_i8_cap < (size_t) N_pad * (size_t) K || b_st.col_scales_cap < (size_t) N_pad * sizeof(float))) {
        return false;
    }
    ggml_cuda_pool_alloc<int8_t> A_i8_tmp(ctx.pool());
    int8_t * A_i8_ptr = rs_entry.A_i8;
    if (A_i8_ptr == nullptr) {
        A_i8_tmp.alloc((size_t) M * K);
        A_i8_ptr = A_i8_tmp.get();
        const int nkb = (int) (K / QK_PQ2_0);
        dequant_pq2_to_i8_1pass_kernel<<<(unsigned) M, 256, 0, stream>>>(
            (const block_pq2_0 *) src0->data, rs_entry.inv_scales, A_i8_ptr, (int) M, nkb);
    }

    int8_t * B_i8_ptr = nullptr;
    float  * col_scales_ptr = nullptr;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    get_or_quantize_b_i8(src1, N, N_pad, K, s11, stream, B_i8_ptr, col_scales_ptr);

    ggml_cuda_pool_alloc<int32_t> C_i32(ctx.pool(), (size_t) M * N_pad);

    const int32_t alpha_i = 1;
    const int32_t beta_i  = 0;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        (int) M, (int) N_pad, (int) K,
        &alpha_i,
        A_i8_ptr,    CUDA_R_8I,  (int) K,
        B_i8_ptr,    CUDA_R_8I,  (int) K,
        &beta_i,
        C_i32.get(), CUDA_R_32I, (int) M,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    const int64_t s_dst = dst->nb[1] / sizeof(float);
    if (add_src && rms_weight && norm_mul_dst && fwht_signs && fwht_out && pq2_b_i8_cache_enabled() &&
            (M % 1024) == 0 && (M / 1024) <= 8 && s_dst == M) {
        auto & st = pq2_b_i8_cache_for_stream(stream);
        const size_t need_b = (size_t) N_pad * (size_t) M;
        const size_t need_s = (size_t) N_pad * sizeof(float);
        ggml_cuda_pool_alloc<float> in_cs_copy(ctx.pool(), (size_t) N_pad);
        CUDA_CHECK(cudaMemcpyAsync(in_cs_copy.get(), col_scales_ptr, (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (st.B_i8_cap < need_b) {
            if (st.B_i8) { CUDA_CHECK(cudaFree(st.B_i8)); }
            CUDA_CHECK(cudaMalloc(&st.B_i8, need_b));
            st.B_i8_cap = need_b;
        }
        if (st.col_scales_cap < need_s) {
            if (st.col_scales) { CUDA_CHECK(cudaFree(st.col_scales)); }
            CUDA_CHECK(cudaMalloc(&st.col_scales, need_s));
            st.col_scales_cap = need_s;
        }
        dequant_i32_add_rms_mul_fwht1024_to_i8_kernel<<<(unsigned) N, 512, 0, stream>>>(
            C_i32.get(), rs_entry.scales, in_cs_copy.get(),
            (const float *) add_src->data, (float *) dst->data,
            (const float *) rms_weight->data, rms_eps,
            (const float *) fwht_signs->data,
            skip_norm_f32_write ? nullptr : (float *) norm_mul_dst->data,
            st.B_i8, st.col_scales, (int) M, (int) N, s_dst);
        st.src1_data   = fwht_out->data;
        st.src1_nbytes = ggml_nbytes(fwht_out);
        st.N           = N;
        st.K           = M;
        st.s11         = M;
        st.valid       = true;
    } else if (add_src && rms_weight && norm_mul_dst && pq2_b_i8_cache_enabled() && M <= 512 * 4 * 4 && s_dst == M) {
        auto & st = pq2_b_i8_cache_for_stream(stream);
        const size_t need_b = (size_t) N_pad * (size_t) M;
        const size_t need_s = (size_t) N_pad * sizeof(float);
        ggml_cuda_pool_alloc<float> in_cs_copy(ctx.pool(), (size_t) N_pad);
        CUDA_CHECK(cudaMemcpyAsync(in_cs_copy.get(), col_scales_ptr, (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (st.B_i8_cap < need_b) {
            if (st.B_i8) { CUDA_CHECK(cudaFree(st.B_i8)); }
            CUDA_CHECK(cudaMalloc(&st.B_i8, need_b));
            st.B_i8_cap = need_b;
        }
        if (st.col_scales_cap < need_s) {
            if (st.col_scales) { CUDA_CHECK(cudaFree(st.col_scales)); }
            CUDA_CHECK(cudaMalloc(&st.col_scales, need_s));
            st.col_scales_cap = need_s;
        }
        dequant_i32_add_rms_mul_to_i8_kernel<<<(unsigned) N, 512, 0, stream>>>(
            C_i32.get(), rs_entry.scales, in_cs_copy.get(),
            (const float *) add_src->data, (float *) dst->data,
            (const float *) rms_weight->data, rms_eps,
            skip_norm_f32_write ? nullptr : (float *) norm_mul_dst->data,
            st.B_i8, st.col_scales, (int) M, (int) N, s_dst);
        st.src1_data   = norm_mul_dst->data;
        st.src1_nbytes = ggml_nbytes(norm_mul_dst);
        st.N           = N;
        st.K           = M;
        st.s11         = M;
        st.valid       = true;
    } else {
        const dim3 epi_grid((unsigned) ((M / 4 + 255) / 256), (unsigned) N, 1);
        dequant_i32_to_f32_epilogue_kernel<<<epi_grid, 256, 0, stream>>>(
            C_i32.get(), rs_entry.scales, col_scales_ptr,
            add_src ? (const float *) add_src->data : nullptr,
            (float *) dst->data, (int) M, (int) N, s_dst);
        pq2_b_i8_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool ggml_cuda_mul_mat_pq2_cublas_i8_swiglu(
        ggml_backend_cuda_context & ctx, const ggml_tensor * gate_w, const ggml_tensor * up_w,
        const ggml_tensor * src1, ggml_tensor * dst, const bool direct_to_b_cache = false,
        const ggml_tensor * fwht_signs = nullptr, const ggml_tensor * fwht_out = nullptr, const int fwht_H = 0) {
    if (!ggml_cuda_pq2_cublas_i8_enabled()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!turing_mma_available(cc)) {
        return false;
    }
    if (gate_w->type != GGML_TYPE_PQ2_0 || up_w->type != GGML_TYPE_PQ2_0 ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(gate_w, up_w) || gate_w->ne[2] != 1 || gate_w->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t K = gate_w->ne[0];
    const int64_t M = gate_w->ne[1];
    const int64_t N = src1->ne[1];
    const int64_t min_n_cb = ggml_cuda_pq2_narrow_int8_enabled() ? (M > 65536 ? 256 : (pq2_narrow_i8_max_n() + 1)) : 129;
    if (N < min_n_cb || (K % QK_PQ2_0) != 0 || (M % 4) != 0 ||
            !ggml_is_contiguous(gate_w) || !ggml_is_contiguous(up_w) ||
            (src1->nb[0] != sizeof(float)) || (dst->nb[0] != sizeof(float))) {
        return false;
    }
    if (ggml_cuda_pq2_prefill_int4_enabled() && N >= 256 && M <= 65536) {
        return false;
    }

    const int64_t N_pad = (N + 3) & ~int64_t(3);
    cudaStream_t stream = ctx.stream();
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    (void) cudaStreamIsCapturing(stream, &cap_status);
    const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);

    const auto gate_rs = get_or_compute_pq2_row_scales((const block_pq2_0 *) gate_w->data, M, K, stream, allow_alloc);
    const auto up_rs   = get_or_compute_pq2_row_scales((const block_pq2_0 *) up_w->data,   M, K, stream, allow_alloc);
    if (!gate_rs.scales || !gate_rs.inv_scales || !up_rs.scales || !up_rs.inv_scales) {
        return false;
    }
    auto & b_st = pq2_b_i8_cache_for_stream(stream);
    if (!allow_alloc && (b_st.B_i8_cap < (size_t) N_pad * (size_t) K || b_st.col_scales_cap < (size_t) N_pad * sizeof(float))) {
        return false;
    }

    ggml_cuda_pool_alloc<int8_t> A_i8_tmp(ctx.pool());
    if (gate_rs.A_i8 == nullptr || up_rs.A_i8 == nullptr) {
        A_i8_tmp.alloc((size_t) M * K);
    }
    ggml_cuda_pool_alloc<int32_t> C_gate(ctx.pool(), (size_t) M * N_pad);
    ggml_cuda_pool_alloc<int32_t> C_up(ctx.pool(), (size_t) M * N_pad);

    // Quantize src1 ONCE (or reuse cached B_i8)!
    int8_t * B_i8_ptr = nullptr;
    float  * col_scales_ptr = nullptr;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    get_or_quantize_b_i8(src1, N, N_pad, K, s11, stream, B_i8_ptr, col_scales_ptr);

    const int32_t alpha_i = 1;
    const int32_t beta_i  = 0;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));

    const int nkb = (int) (K / QK_PQ2_0);

    // Gate GEMM
    int8_t * gate_A = gate_rs.A_i8;
    if (gate_A == nullptr) {
        gate_A = A_i8_tmp.get();
        dequant_pq2_to_i8_1pass_kernel<<<(unsigned) M, 256, 0, stream>>>(
            (const block_pq2_0 *) gate_w->data, gate_rs.inv_scales, gate_A, (int) M, nkb);
    }
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        (int) M, (int) N_pad, (int) K,
        &alpha_i, gate_A, CUDA_R_8I, (int) K, B_i8_ptr, CUDA_R_8I, (int) K,
        &beta_i, C_gate.get(), CUDA_R_32I, (int) M,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Up GEMM
    int8_t * up_A = up_rs.A_i8;
    if (up_A == nullptr) {
        up_A = A_i8_tmp.get();
        dequant_pq2_to_i8_1pass_kernel<<<(unsigned) M, 256, 0, stream>>>(
            (const block_pq2_0 *) up_w->data, up_rs.inv_scales, up_A, (int) M, nkb);
    }
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        (int) M, (int) N_pad, (int) K,
        &alpha_i, up_A, CUDA_R_8I, (int) K, B_i8_ptr, CUDA_R_8I, (int) K,
        &beta_i, C_up.get(), CUDA_R_32I, (int) M,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Fused SwiGLU (+ optional Hadamard signs + FWHT) -> direct INT8 into B_i8 cache!
    const int64_t s_dst = dst->nb[1] / sizeof(float);
    if (fwht_signs && fwht_out && pq2_b_i8_cache_enabled() &&
            (fwht_H == 1024 || fwht_H == 512 || fwht_H == 256 || fwht_H == 128) &&
            (M % fwht_H) == 0 && (fwht_H == 1024 ? (M / 1024) <= 18 : (M / fwht_H) <= 36)) {
        auto & st = pq2_b_i8_cache_for_stream(stream);
        const size_t need_b = (size_t) N_pad * (size_t) M;
        const size_t need_s = (size_t) N_pad * sizeof(float);
        ggml_cuda_pool_alloc<float> in_cs_copy(ctx.pool(), (size_t) N_pad);
        CUDA_CHECK(cudaMemcpyAsync(in_cs_copy.get(), col_scales_ptr, (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (st.B_i8_cap < need_b) {
            if (st.B_i8) { CUDA_CHECK(cudaFree(st.B_i8)); }
            CUDA_CHECK(cudaMalloc(&st.B_i8, need_b));
            st.B_i8_cap = need_b;
        }
        if (st.col_scales_cap < need_s) {
            if (st.col_scales) { CUDA_CHECK(cudaFree(st.col_scales)); }
            CUDA_CHECK(cudaMalloc(&st.col_scales, need_s));
            st.col_scales_cap = need_s;
        }
        if (fwht_H == 1024) {
            dequant_i32_swiglu_fwht1024_to_i8_kernel<<<(unsigned) N, 512, 0, stream>>>(
                C_gate.get(), gate_rs.scales, C_up.get(), up_rs.scales, in_cs_copy.get(),
                (const float *) fwht_signs->data, st.B_i8, st.col_scales, (int) M, (int) N);
        } else if (fwht_H == 512) {
            dequant_i32_swiglu_fwht_to_i8_kernel<512><<<(unsigned) N, 512, 0, stream>>>(
                C_gate.get(), gate_rs.scales, C_up.get(), up_rs.scales, in_cs_copy.get(),
                (const float *) fwht_signs->data, st.B_i8, st.col_scales, (int) M, (int) N);
        } else if (fwht_H == 256) {
            dequant_i32_swiglu_fwht_to_i8_kernel<256><<<(unsigned) N, 256, 0, stream>>>(
                C_gate.get(), gate_rs.scales, C_up.get(), up_rs.scales, in_cs_copy.get(),
                (const float *) fwht_signs->data, st.B_i8, st.col_scales, (int) M, (int) N);
        } else {
            dequant_i32_swiglu_fwht_to_i8_kernel<128><<<(unsigned) N, 128, 0, stream>>>(
                C_gate.get(), gate_rs.scales, C_up.get(), up_rs.scales, in_cs_copy.get(),
                (const float *) fwht_signs->data, st.B_i8, st.col_scales, (int) M, (int) N);
        }
        st.src1_data   = fwht_out->data;
        st.src1_nbytes = ggml_nbytes(fwht_out);
        st.N           = N;
        st.K           = M;
        st.s11         = M;
        st.valid       = true;
    } else if (direct_to_b_cache && pq2_b_i8_cache_enabled() && M <= 512 * 4 * 9 && s_dst == M) {
        auto & st = pq2_b_i8_cache_for_stream(stream);
        const size_t need_b = (size_t) N_pad * (size_t) M;
        const size_t need_s = (size_t) N_pad * sizeof(float);
        // Ensure we don't alias the input B_i8/col_scales we just read (they are done after GEMMs,
        // EXCEPT in_col_scales is still read by the epilogue, so copy/keep in_col_scales in pool!)
        ggml_cuda_pool_alloc<float> in_cs_copy(ctx.pool(), (size_t) N_pad);
        CUDA_CHECK(cudaMemcpyAsync(in_cs_copy.get(), col_scales_ptr, (size_t) N * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if (st.B_i8_cap < need_b) {
            if (st.B_i8) { CUDA_CHECK(cudaFree(st.B_i8)); }
            CUDA_CHECK(cudaMalloc(&st.B_i8, need_b));
            st.B_i8_cap = need_b;
        }
        if (st.col_scales_cap < need_s) {
            if (st.col_scales) { CUDA_CHECK(cudaFree(st.col_scales)); }
            CUDA_CHECK(cudaMalloc(&st.col_scales, need_s));
            st.col_scales_cap = need_s;
        }
        dequant_i32_swiglu_to_i8_epilogue_kernel<<<(unsigned) N, 512, 0, stream>>>(
            C_gate.get(), gate_rs.scales, C_up.get(), up_rs.scales, in_cs_copy.get(),
            st.B_i8, st.col_scales, (int) M, (int) N);
        st.src1_data   = dst->data;
        st.src1_nbytes = ggml_nbytes(dst);
        st.N           = N;
        st.K           = M;
        st.s11         = M;
        st.valid       = true;
    } else {
        const dim3 epi_grid((unsigned) ((M / 4 + 255) / 256), (unsigned) N, 1);
        dequant_i32_swiglu_epilogue_kernel<<<epi_grid, 256, 0, stream>>>(
            C_gate.get(), gate_rs.scales, C_up.get(), up_rs.scales, col_scales_ptr,
            (float *) dst->data, (int) M, (int) N, s_dst);
        pq2_b_i8_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16 && src0->type != GGML_TYPE_PQ1_0) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32 || src0->type == GGML_TYPE_PQ1_0) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1 (and PQ2_0 multi-column 1..8)
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1 &&
        !(src0->type == GGML_TYPE_PQ2_0 && ggml_cuda_pq2_multicol(dst->ne[1]))) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    if (ggml_cuda_pq2_narrow_int8_enabled() && src0->type == GGML_TYPE_PQ2_0 &&
            src1->ne[1] >= 5 && src1->ne[1] <= pq2_narrow_i8_max_n() &&
            src0->ne[1] <= (src1->ne[1] >= 256 ? 65536 : 262144)) {
        return false;
    }

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1 (and PQ2_0 multi-column 1..8)
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1 &&
        !(src0->type == GGML_TYPE_PQ2_0 && ggml_cuda_pq2_multicol(dst->ne[1]))) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

bool ggml_cuda_mul_mat_q1_hopper(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// PQ2_0 mat-vec for Pascal, 1-4 columns, bit-identical to the generic mmvq there. The generic kernel (4 warps,
// 32 blocks per iteration) gives thread t = 32*w + lane the 32-chunks t, t+128, ...: it sums them in order into
// v[w], then warp 0 adds ((v0 + v1) + v2) + v3 and does the warp butterfly. Here ONE warp keeps that order for a
// whole row (lane owns chunks lane + 32*w + 128*i), so there is no shared memory and no block barrier, and does 4
// rows at once. Activations use the decode-free PQ2 layout (quantize.cu): sum(q*(c-1)) = sum(q*c) - sum(q) is the
// same integer as the generic byte-perm decode, and d2*d8*sumi is the same float product.
template <int ncols>
static __global__ void __launch_bounds__(128) mul_mat_vec_pq2_0_ordered(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst) {
    constexpr int R = 4;
    const int lane = threadIdx.x & 31;
    const int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * R;
    if (row0 >= nrows_x) {
        return;
    }
    const int    nchunk    = ncols_x / QK8_1;
    const size_t col_bytes = (size_t) stride_col_y * sizeof(block_q8_1);
    const size_t ds_off    = (size_t) stride_col_y * QK8_1;
    const char * y         = (const char *) vy;

    float s[R][ncols];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        float v[R][ncols] = {};
        for (int c = 32*w + lane; c < nchunk; c += 128) {
            uint32_t w0[R], w1[R];
            float    d2[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = min(row0 + r, nrows_x - 1); // clamped rows are computed but not stored
                const block_pq2_0 * blk = (const block_pq2_0 *) vx + (int64_t) row*stride_row_x + (c >> 2);
                const uint16_t    * qp  = (const uint16_t *) (blk->qs + (c & 3)*8); // blocks are only 2-byte aligned
                d2[r] = blk->d;
                w0[r] = (uint32_t) qp[0] | ((uint32_t) qp[1] << 16);
                w1[r] = (uint32_t) qp[2] | ((uint32_t) qp[3] << 16);
            }
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const char * yc = y + j*col_bytes;
                const int4  a0  = *(const int4 *) (yc + c*QK8_1);
                const int4  a1  = *(const int4 *) (yc + c*QK8_1 + 16);
                const half2 ds  = *(const half2 *) (yc + ds_off + c*sizeof(half2));
                const float d8  = __low2float(ds);
                const int   qs  = __half_as_short(__high2half(ds));
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int sumi = 0;
                    sumi = ggml_cuda_dp4a((int) ((w0[r] >> 0) & 0x03030303u), a0.x, sumi);
                    sumi = ggml_cuda_dp4a((int) ((w0[r] >> 2) & 0x03030303u), a0.y, sumi);
                    sumi = ggml_cuda_dp4a((int) ((w0[r] >> 4) & 0x03030303u), a0.z, sumi);
                    sumi = ggml_cuda_dp4a((int) ((w0[r] >> 6) & 0x03030303u), a0.w, sumi);
                    sumi = ggml_cuda_dp4a((int) ((w1[r] >> 0) & 0x03030303u), a1.x, sumi);
                    sumi = ggml_cuda_dp4a((int) ((w1[r] >> 2) & 0x03030303u), a1.y, sumi);
                    sumi = ggml_cuda_dp4a((int) ((w1[r] >> 4) & 0x03030303u), a1.z, sumi);
                    sumi = ggml_cuda_dp4a((int) ((w1[r] >> 6) & 0x03030303u), a1.w, sumi);
                    sumi -= qs;
                    v[r][j] += d2[r] * d8 * sumi;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                s[r][j] = w == 0 ? v[r][j] : s[r][j] + v[r][j];
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float sum = warp_reduce_sum<WARP_SIZE>(s[r][j]);
            if (lane == 0 && row0 + r < nrows_x) {
                dst[(int64_t) j*stride_col_dst + row0 + r] = sum;
            }
        }
    }
}

static void ggml_cuda_mul_mat_pq2_0_ordered(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS
    cudaStream_t stream = ctx.stream();
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne11*ne10_padded*sizeof(block_q8_1)/QK8_1);
    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), GGML_CUDA_Q8_1_PQ2, ne10,
        nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float), ne10_padded, ne11, 1, 1, stream);
    const dim3 blocks((ne01 + 15) / 16), threads(128);
    const int sx = nb01/ggml_type_size(src0->type), sy = ne10_padded/QK8_1, sd = nb1/sizeof(float);
    const void * x = src0->data;
    const char * yq = src1_q8_1.get();
    float * d = (float *) dst->data;
    switch (ne11) {
        case 1: mul_mat_vec_pq2_0_ordered<1><<<blocks, threads, 0, stream>>>(x, yq, d, ne00, ne01, sx, sy, sd); break;
        case 2: mul_mat_vec_pq2_0_ordered<2><<<blocks, threads, 0, stream>>>(x, yq, d, ne00, ne01, sx, sy, sd); break;
        case 3: mul_mat_vec_pq2_0_ordered<3><<<blocks, threads, 0, stream>>>(x, yq, d, ne00, ne01, sx, sy, sd); break;
        case 4: mul_mat_vec_pq2_0_ordered<4><<<blocks, threads, 0, stream>>>(x, yq, d, ne00, ne01, sx, sy, sd); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// Buffer- and batch-size independent, so graph_optimize produces the same graph topology for every ubatch size (see #29986).
static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device) {
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];

    if (up->op != GGML_OP_MUL_MAT || !ggml_cuda_mmq_fusion_supported(x_up->type) || y->type != GGML_TYPE_F32) {
        return false;
    }

    // The fused kernel is only compiled for weights whose rows fill whole tiles (no fallback).
    if (ggml_cuda_mmq_needs_fallback(x_up->ne[1])) {
        return false;
    }

    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu)) {
        return false;
    }

    // The fused write-back applies SWIGLU_OAI with the default alpha and limit of ggml_cuda_op_swiglu_oai_single.
    if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI && (ggml_get_op_params_f32(glu, 2) != 1.702f || ggml_get_op_params_f32(glu, 3) != 7.0f)) {
        return false;
    }

    // NVIDIA only: the fused write-back relies on the MMA accumulator layout.
    return turing_mma_available(ggml_cuda_info().devices[device].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu) {
    // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
    if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
        return false;
    }

    // The padding of a compute-buffer weight that is a view cannot be cleared, see ggml_cuda_mul_mat.
    const auto bad_padding_clear = [](const ggml_tensor * x) {
        return ggml_backend_buffer_get_usage(x->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(x) != ggml_backend_buffer_get_alloc_size(x->buffer, x) && x->view_src;
    };
    const int device = ggml_cuda_get_device();
    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, device) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
        return false;
    }

    // The batch size checks are done here, not in ggml_cuda_mul_mat_q_fusion_matches.
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];
    const int           cc   = ggml_cuda_info().devices[device].cc;
    if (ggml_cuda_should_use_mmvq(x_up->type, cc, y->ne[1]) || !ggml_cuda_should_use_mmq(x_up->type, cc, y->ne[1], 0)) {
        return false;
    }

    // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, x_up->ne[1], y->ne[1], y->ne[2]*y->ne[3], device);
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ne11 >= 1 && ne11 <= 8 && ne00 >= 512 && (ne00 % 4) == 0 && ne01 <= 64 &&
            ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float)) {
        const int64_t K = ne00, M = ne01, s11 = src1->nb[1] / sizeof(float), s_dst = dst->nb[1] / sizeof(float);
        cudaStream_t stream = ctx.stream();
        switch (ne11) {
            case 1: thin_f32_decode_kernel<1><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
            case 2: thin_f32_decode_kernel<2><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
            case 3: thin_f32_decode_kernel<3><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
            case 4: thin_f32_decode_kernel<4><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
            case 5: thin_f32_decode_kernel<5><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
            case 6: thin_f32_decode_kernel<6><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
            case 7: thin_f32_decode_kernel<7><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
            case 8: thin_f32_decode_kernel<8><<<(unsigned) M, 256, 0, stream>>>((const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) K, s11, s_dst); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ggml_cuda_mul_mat_f32_narrow_f16_tc(ctx, src0, src1, dst)) {
        return;
    }
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ne11 >= 9 && ne11 <= 64 && ne00 >= 512 && (ne00 % 4) == 0 && ne01 <= 64 &&
            ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float)) {
        const int64_t K = ne00, M = ne01, N = ne11, s11 = src1->nb[1] / sizeof(float), s_dst = dst->nb[1] / sizeof(float);
        dim3 grid((unsigned) M, (unsigned) ((N + 7) / 8));
        thin_f32_narrow_kernel<<<grid, 256, 0, ctx.stream()>>>(
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) N, (int) K, s11, s_dst);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ggml_cuda_mul_mat_f32_prefill_f16_tc(ctx, src0, src1, dst)) {
        return;
    }
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
            ggml_cuda_pq2_cublas_i8_enabled() && turing_mma_available(cc) &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ne11 >= 65 && ne00 >= 512 && (ne00 % 4) == 0 && ne01 <= 64 && (ne01 % 4) == 0 &&
            (strstr(src0->name, "ssm_alpha") != nullptr || strstr(src0->name, "ssm_beta") != nullptr) &&
            ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float)) {
        const int64_t K = ne00, M = ne01, N = ne11;
        const int64_t N_pad = (N + 3) & ~int64_t(3);
        cudaStream_t stream = ctx.stream();
        cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
        (void) cudaStreamIsCapturing(stream, &cap_status);
        const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);
        const auto ae = get_or_quantize_thin_f32((const float *) src0->data, M, K, stream, allow_alloc);
        auto & b_st = pq2_b_i8_cache_for_stream(stream);
        if (ae.A_i8 != nullptr && (allow_alloc || (b_st.B_i8_cap >= (size_t) N_pad * (size_t) K && b_st.col_scales_cap >= (size_t) N_pad * sizeof(float)))) {
            int8_t * B_i8_ptr = nullptr;
            float  * col_scales_ptr = nullptr;
            const int64_t s11 = src1->nb[1] / sizeof(float);
            get_or_quantize_b_i8(src1, N, N_pad, K, s11, stream, B_i8_ptr, col_scales_ptr);
            ggml_cuda_pool_alloc<int32_t> C_i32(ctx.pool(), (size_t) M * N_pad);
            const int32_t alpha_i = 1, beta_i = 0;
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
            CUBLAS_CHECK(cublasGemmEx(
                ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, (int) M, (int) N_pad, (int) K,
                &alpha_i, ae.A_i8, CUDA_R_8I, (int) K, B_i8_ptr, CUDA_R_8I, (int) K,
                &beta_i, C_i32.get(), CUDA_R_32I, (int) M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            const int64_t s_dst = dst->nb[1] / sizeof(float);
            const dim3 epi_grid((unsigned) ((M / 4 + 255) / 256), (unsigned) N, 1);
            dequant_i32_to_f32_epilogue_kernel<<<epi_grid, 256, 0, stream>>>(
                C_i32.get(), ae.scales, col_scales_ptr, nullptr, (float *) dst->data, (int) M, (int) N, s_dst);
            pq2_b_i8_cache_invalidate_if_overlap(stream, dst->data, ggml_nbytes(dst));
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // Pascal PQ2_0 at 1-4 columns (decode, MTP verify and draft head): same sums as the generic mmvq, faster
    if (src0->type == GGML_TYPE_PQ2_0 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA &&
            !ggml_cuda_batch_invariant() && ne11 <= 4 && ne00 % QK_PQ2_0 == 0 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && ggml_is_contiguous(src0) &&
            ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        ggml_cuda_mul_mat_pq2_0_ordered(ctx, src0, src1, dst);
        return;
    }
    if (ggml_cuda_mul_mat_pq2_narrow_int8(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_mul_mat_pq2_prefill_int4(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_mul_mat_pq2_cublas_i8(ctx, src0, src1, dst)) {
        return;
    }
    if ((src0->type == GGML_TYPE_Q1_0 || src0->type == GGML_TYPE_PQ2_0) && ne11 >= 128
            && ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)
            && ggml_cuda_mul_mat_q1_hopper(ctx, src0, src1, dst)) {
        // handled by the opt-in Hopper wgmma path (returns false to fall through when unsupported)
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET_REPLAY:
            ggml_cuda_op_gated_delta_net_replay(ctx, dst, nullptr);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_TOP_K_MASK:
            ggml_cuda_op_top_k_mask(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// A graph is cached under its first node. Speculative decoding (the MTP draft step) builds graphs whose first
// node keeps its address but changes shape from one call to the next (1 to K+1 positions), and every change
// reset the warmup of the one cached graph, so those steps ran eagerly. Mixing the first node's shape and the
// node count into the key gives each shape its own cached graph; the property check still guards each one.
static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    const ggml_tensor * n0 = cgraph->nodes[0];
    uint64_t h = (uint64_t) (uintptr_t) n0;
    const auto mix = [&h](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        mix((uint64_t) n0->ne[d]);
    }
    mix((uint64_t) cgraph->n_nodes);
    return (const void *) (uintptr_t) h;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(cudaGraphExec_t & instance, cudaGraph_t cuda_graph) {
#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(instance));
        instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&instance, cuda_graph, nullptr, nullptr, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

// number of ggml nodes per CUDA graph chunk, 0 captures the whole ggml graph as one CUDA graph
static int ggml_cuda_graph_chunk_size() {
    static const int chunk_size = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_CHUNK_SIZE");
        return env != nullptr ? atoi(env) : 512;
    }();
    return chunk_size;
}

// end the capture of chunk `idx`, update or create its executable and launch it
static void ggml_cuda_graph_launch_chunk(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph, size_t idx) {
    cudaGraph_t cuda_graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph));

    if (idx < graph->instances.size()) {
        ggml_cuda_graph_update_executable(graph->instances[idx], cuda_graph);
    } else {
        GGML_ASSERT(idx == graph->instances.size());
        graph->instances.emplace_back(nullptr);
        CUDA_CHECK(cudaGraphInstantiate(&graph->instances[idx], cuda_graph, nullptr, nullptr, 0));
    }
    CUDA_CHECK(cudaGraphDestroy(cuda_graph));

    CUDA_CHECK(cudaGraphLaunch(graph->instances[idx], cuda_ctx->stream()));
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/multi shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// Every use of the Hadamard transform output at node i_mm (directly or through a reshape view of the whole
// tensor) must be src1 of a PTQ1_0 (or, with accept_pq2, PQ2_0) MUL_MAT that ggml_cuda_mul_mat routes to
// ggml_cuda_mul_mat_vec_q, all consumers of one weight type so they share one q8_1 layout. The consumers'
// src1 shape [K, ncols] is what gets quantized. Fills u (the output and its views, K, ncols, weight type).
struct ggml_cuda_fwht_q8_uses {
    const ggml_tensor * aliases[8] = {};
    int       n_aliases = 0;
    int64_t   K         = 0;
    int64_t   ncols     = 0;
    ggml_type wtype     = GGML_TYPE_COUNT;
};

static bool ggml_cuda_fwht_q8_find_uses(const ggml_cgraph * cgraph, const int i_mm, const int cc, const bool accept_pq2,
                                        ggml_cuda_fwht_q8_uses & u) {
    const ggml_tensor * mm = cgraph->nodes[i_mm];
    const ggml_tensor ** aliases = u.aliases;
    int &     n_aliases = u.n_aliases;
    int64_t & K         = u.K;
    int64_t & ncols     = u.ncols;
    ggml_type & wtype   = u.wtype;
    aliases[0] = mm;
    n_aliases  = 1;
    int uses_expected = ggml_node_get_use_count(cgraph, i_mm);
    int uses_found    = 0;
    for (int j = i_mm + 1; j < cgraph->n_nodes; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = t->src[s];
            if (!src) {
                continue;
            }
            bool is_alias = false;
            for (int a = 0; a < n_aliases; ++a) {
                if (src == aliases[a]) {
                    is_alias = true;
                    break;
                }
            }
            if (!is_alias) {
                continue;
            }
            uses_found++;
            if (t->op == GGML_OP_RESHAPE && t->view_src == mm && t->data == mm->data &&
                    ggml_nelements(t) == ggml_nelements(mm) && ggml_is_contiguous(t)) {
                if (n_aliases == 8 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    return false; // an output view of the transform must stay F32
                }
                aliases[n_aliases++] = t;
                uses_expected += ggml_node_get_use_count(cgraph, j);
                continue;
            }
            const ggml_type ct = t->src[0] ? t->src[0]->type : GGML_TYPE_COUNT;
            const bool type_ok = ct == GGML_TYPE_PTQ1_0 || (accept_pq2 && ct == GGML_TYPE_PQ2_0);
            if (t->op != GGML_OP_MUL_MAT || s != 1 || !type_ok || (wtype != GGML_TYPE_COUNT && ct != wtype) ||
                    ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD ||
                    t->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) ||
                    src->ne[2] != 1 || src->ne[3] != 1 ||
                    (K != 0 && (src->ne[0] != K || src->ne[1] != ncols)) ||
                    !ggml_cuda_should_use_mmvq(ct, cc, src->ne[1])) {
                return false;
            }
            if (ct == GGML_TYPE_PQ2_0 && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA) {
                return false; // ggml_cuda_mul_mat_pq2_0_ordered quantizes src1 itself
            }
            K     = src->ne[0];
            ncols = src->ne[1];
            wtype = ct;
            // ggml_cuda_mul_mat sends a padded compute-buffer view to cuBLAS instead
            const ggml_tensor * w = t->src[0];
            if (ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                    ggml_nbytes(w) != ggml_backend_buffer_get_alloc_size(w->buffer, w) && w->view_src) {
                return false;
            }
        }
    }
    if (uses_found == 0 || uses_found != uses_expected || K == 0) {
        return false;
    }
    return true;
}

// GGML_CUDA_FWHT_MMQ: like ggml_cuda_fwht_q8_find_uses, but every use must be an MMQ mat-mul (prefill widths) of
// PTQ1_0/PQ2_0 weights (q8_1 D4 layout), so the rotation can write their shared block_q8_1_mmq rows.
static bool ggml_cuda_fwht_mmq_find_uses(const ggml_cgraph * cgraph, const int i_mm, const int cc, ggml_cuda_fwht_q8_uses & u) {
    const ggml_tensor * mm = cgraph->nodes[i_mm];
    u = ggml_cuda_fwht_q8_uses();
    u.aliases[0] = mm;
    u.n_aliases  = 1;
    int uses_expected = ggml_node_get_use_count(cgraph, i_mm);
    int uses_found    = 0;
    for (int j = i_mm + 1; j < cgraph->n_nodes; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = t->src[s];
            if (!src) {
                continue;
            }
            bool is_alias = false;
            for (int a = 0; a < u.n_aliases; ++a) {
                if (src == u.aliases[a]) {
                    is_alias = true;
                    break;
                }
            }
            if (!is_alias) {
                continue;
            }
            uses_found++;
            if (t->op == GGML_OP_RESHAPE && t->view_src == mm && t->data == mm->data &&
                    ggml_nelements(t) == ggml_nelements(mm) && ggml_is_contiguous(t)) {
                if (u.n_aliases == 8 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    return false; // an output view of the transform must stay F32 (as in ggml_cuda_fwht_q8_find_uses)
                }
                u.aliases[u.n_aliases++] = t;
                uses_expected += ggml_node_get_use_count(cgraph, j);
                continue;
            }
            const ggml_type ct = t->src[0] ? t->src[0]->type : GGML_TYPE_COUNT;
            if (t->op != GGML_OP_MUL_MAT || s != 1 || (ct != GGML_TYPE_PTQ1_0 && ct != GGML_TYPE_PQ2_0) ||
                    ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD ||
                    t->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) ||
                    src->ne[2] != 1 || src->ne[3] != 1 || t->src[0]->ne[2] != 1 || t->src[0]->ne[3] != 1 ||
                    (u.K != 0 && (src->ne[0] != u.K || src->ne[1] != u.ncols)) ||
                    src->ne[0] % 512 != 0 ||  // no padding blocks (ne10_padded == K)
                    ggml_cuda_should_use_mmvq(ct, cc, src->ne[1]) || !ggml_cuda_should_use_mmq(ct, cc, src->ne[1], 0)) {
                return false;
            }
            const ggml_tensor * w = t->src[0];
            if (ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                return false;
            }
            u.K     = src->ne[0];
            u.ncols = src->ne[1];
            u.wtype = ct;
        }
        if (uses_found == uses_expected) {
            break; // every use seen (host time: the uses sit a few nodes after the rotation)
        }
    }
    return uses_found > 0 && uses_found == uses_expected && u.K != 0;
}

// GGML_CUDA_FWHT_MMQ_GLU: GLU(swiglu, split gate/up, not swapped) -> MUL(signs) -> RESHAPE -> MUL_MAT(Hadamard) whose
// uses are all MMQ mat-muls: one launch computes silu(gate)*up, rotates and quantizes. Returns nodes consumed (4) or 0.
static int ggml_cuda_try_glu_fwht_mmq(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    // GGML_CUDA_FWHT_MMQ and GGML_CUDA_FWHT_MMQ_GLU: 1 = on, 0 = off, unset = follow GGML_CUDA_PREFILL_OPT
    static const int mmq_env = ggml_cuda_env_tristate("GGML_CUDA_FWHT_MMQ");
    static const int glu_env = ggml_cuda_env_tristate("GGML_CUDA_FWHT_MMQ_GLU");
    const int  cc  = ggml_cuda_info().devices[ctx.device].cc;
    const bool def = ggml_cuda_prefill_opt(cc);
    const bool on  = (mmq_env >= 0 ? mmq_env == 1 : def) && (glu_env >= 0 ? glu_env == 1 : def);
    if (!on || !ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_GLU, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { node_idx + 3 })) {
        return 0;
    }
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_TURING) {
        return 0;
    }
    const ggml_tensor * glu     = cgraph->nodes[node_idx];
    const ggml_tensor * mul     = cgraph->nodes[node_idx + 1];
    const ggml_tensor * reshape = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mm      = cgraph->nodes[node_idx + 3];
    const ggml_tensor * g0      = glu->src[0];
    const ggml_tensor * g1      = glu->src[1];
    const ggml_tensor * signs   = mul->src[1];
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || g1 == nullptr || ggml_get_op_params_i32(glu, 1) != 0 ||
            glu->type != GGML_TYPE_F32 || g0->type != GGML_TYPE_F32 || g1->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(glu) || !ggml_is_contiguous(g0) || !ggml_is_contiguous(g1) ||
            !ggml_are_same_shape(g0, glu) || !ggml_are_same_shape(g1, glu) ||
            mul->src[0] != glu || mul->type != GGML_TYPE_F32 ||
            ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD || mm->src[1] != reshape || reshape->src[0] != mul ||
            signs->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs) || signs->ne[1] != 1 || signs->ne[2] != 1 ||
            signs->ne[3] != 1 || signs->ne[0] != glu->ne[0] ||
            mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || (mm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_nelements(mm) != ggml_nelements(glu)) {
        return 0;
    }
    const int n = (int) mm->src[0]->ne[0];
    ggml_cuda_fwht_q8_uses u;
    if (signs->ne[0] % n != 0 || !ggml_cuda_fwht_mmq_find_uses(cgraph, node_idx + 3, cc, u) || u.K % n != 0 ||
            signs->ne[0] != u.K) {
        return 0;
    }
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data;
        const char * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    const size_t q8_bytes = (size_t) (u.ncols * u.K) / 128 * 144 + 256*1024;
    std::unique_ptr<ggml_cuda_pool_alloc<char>> blk;
    void * out = mm->data; // stable address (CUDA graphs) unless it overlaps an input
    if (overlaps(g0, mm) || overlaps(g1, mm) || overlaps(signs, mm) || q8_bytes > ggml_nbytes(mm)) {
        blk = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), q8_bytes);
        out = blk->get();
    }
    if (!ggml_cuda_fwht_quantize_mmq_d4(g0->data, g0->type, (const float *) signs->data, n, out, u.K, u.ncols, ctx.stream(),
                                        (const float *) g1->data)) {
        return 0;
    }
    if (blk) {
        ctx.fwht_q8().held.push_back(std::move(blk));
    }
    ggml_cuda_fwht_q8 e;
    e.layout = GGML_CUDA_Q8_1_MMQ_D4;
    e.ne0    = u.K;
    e.ncols  = u.ncols;
    e.data   = out;
    for (int a = 0; a < u.n_aliases; ++a) {
        ctx.fwht_q8().set(u.aliases[a], e);
    }
    return 4;
}

// A Hadamard rotation of an activation ([MUL signs, RESHAPE,] MUL_MAT with GGML_HINT_SRC0_IS_HADAMARD)
// whose every use is a PTQ1_0 mat-vec on the mmvq path does not need to exist in F32: each of those
// mat-vecs quantizes it to q8_1 straight away, so the transform kernel quantizes as it goes and writes
// the q8_1 rows (in the layout the mat-vecs expect) into its own output buffer; the mat-vecs then skip
// their quantize launch (ggml_cuda_fwht_q8). On Bonsai 2 that removes ~390 launches per decode step.
// PQ2_0 consumers are accepted too, on the q8_1 AoS layout or the PQ2 multi-column layout: the same butterfly and the
// same per-32 quantizer as the unfused pair, so the q8_1 rows are bit-identical. GGML_CUDA_FWHT_FUSION_PQ2=0 limits
// the fusion to PTQ1_0 consumers and transforms of at most 2048 (the previous default). It does not affect the q8_1
// output of the fused norm + transform kernel (GGML_CUDA_NORM_FWHT_Q8=0 turns that off) or the PQ2_0 multi-column
// mat-vec range (GGML_CUDA_PQ2_MC_MIN).
// The check walks every later node, so any other use of the output (or of a view of it) keeps the
// plain transform, and use counts guard against consumers outside this graph. Returns the number of
// nodes consumed (3 or 1) or 0 to fall through. GGML_CUDA_FWHT_FUSION=0 disables it.
static int ggml_cuda_try_fwht_q8(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool disabled = getenv("GGML_CUDA_FWHT_FUSION") != nullptr &&
                                 atoi(getenv("GGML_CUDA_FWHT_FUSION")) == 0;
    if (disabled) {
        return 0;
    }
    static const bool pq2_enabled = getenv("GGML_CUDA_FWHT_FUSION_PQ2") == nullptr ||
                                    atoi(getenv("GGML_CUDA_FWHT_FUSION_PQ2")) != 0;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_TURING) {
        return 0; // the PTQ1_0 mmvq predicate below is only meaningful there
    }

    const ggml_tensor * x     = nullptr;
    const ggml_tensor * signs = nullptr;
    ggml_tensor *       mm    = nullptr;
    int consumed = 0;
    int i_mm     = -1;

    if (ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { node_idx + 2 })) {
        const ggml_tensor * mul     = cgraph->nodes[node_idx];
        const ggml_tensor * reshape = cgraph->nodes[node_idx + 1];
        mm    = cgraph->nodes[node_idx + 2];
        x     = mul->src[0];
        signs = mul->src[1];
        const bool pattern_ok = ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
            mm->src[1] == reshape && reshape->src[0] == mul &&
            signs->ne[1] == 1 && signs->ne[2] == 1 && signs->ne[3] == 1 &&
            signs->type == GGML_TYPE_F32 && mul->type == x->type &&
            ggml_is_contiguous(x) && ggml_is_contiguous(signs) &&
            signs->ne[0] == x->ne[0] && signs->ne[0] % mm->src[0]->ne[0] == 0;
        if (!pattern_ok) {
            return 0;
        }
        consumed = 3;
        i_mm     = node_idx + 2;
    } else {
        mm = cgraph->nodes[node_idx];
        if (mm->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
            return 0;
        }
        x = mm->src[1];
        if (!ggml_is_contiguous(x) || !ggml_are_same_shape(x, mm)) {
            return 0;
        }
        consumed = 1;
        i_mm     = node_idx;
    }

    const int n = (int) mm->src[0]->ne[0];
    if (mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || (mm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16) || ggml_nelements(mm) != ggml_nelements(x)) {
        return 0;
    }

    // The fused kernel reads x (and signs) in the same launch that writes its output. The separate transform
    // and quantize launches never needed those apart, and ggml-alloc does hand mm the block of x when x is a
    // reshape view (no in-place reuse through a view parent, so the viewed tensor is released at the MUL and
    // mm takes it two nodes later - the output-projection input on every Bonsai layer). Blocks storing q8_1
    // rows early would then overwrite f32 input other blocks have not loaded yet: sporadic garbage that on a
    // hybrid model lands in the recurrent state and poisons the rest of the sequence. In that case the rows
    // go to a pool block held until the next graph evaluation instead of mm's buffer.
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data;
        const char * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data;
        const char * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    const bool out_aliases_in = overlaps(x, mm) || (signs && overlaps(signs, mm));

    // x is only known to be contiguous with the same element count as the consumers' src1 [K, ncols];
    // in the unsigned pattern it is the [n, rows] reshape
    ggml_cuda_fwht_q8_uses u;
    if (!ggml_cuda_fwht_q8_find_uses(cgraph, i_mm, cc, pq2_enabled, u)) {
        static const int fwht_mmq_env = ggml_cuda_env_tristate("GGML_CUDA_FWHT_MMQ");
        const bool fwht_mmq = fwht_mmq_env >= 0 ? fwht_mmq_env == 1 : ggml_cuda_prefill_opt(cc);
        if (!fwht_mmq || !ggml_cuda_fwht_mmq_find_uses(cgraph, i_mm, cc, u) || u.K % n != 0 ||
                (signs && signs->ne[0] != u.K)) {
            return 0;
        }
        if (u.wtype == GGML_TYPE_PQ2_0 &&
                ((ggml_cuda_pq2_narrow_int8_enabled() && u.ncols >= 5 && u.ncols <= 128) ||
                 (ggml_cuda_pq2_cublas_i8_enabled() && u.ncols >= 129))) {
            return 0;
        }
        // rows in the rotation's own F32 output (4 bytes per value vs 1.125: room for the MMQ tile reads past the
        // last column; a stable address, so CUDA graphs stay valid), or in a pool block held for this graph when
        // that buffer overlaps the input (see out_aliases_in above)
        const size_t q8_bytes = (size_t) (u.ncols * u.K) / 128 * 144 + 256*1024;
        std::unique_ptr<ggml_cuda_pool_alloc<char>> blk;
        void * out = mm->data;
        if (out_aliases_in || q8_bytes > ggml_nbytes(mm)) {
            blk = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), q8_bytes);
            out = blk->get();
        }
        if (!ggml_cuda_fwht_quantize_mmq_d4(x->data, x->type, signs ? (const float *) signs->data : nullptr, n, out,
                                            u.K, u.ncols, ctx.stream())) {
            return 0;
        }
        if (blk) {
            ctx.fwht_q8().held.push_back(std::move(blk));
        }
        ggml_cuda_fwht_q8 e;
        e.layout = GGML_CUDA_Q8_1_MMQ_D4;
        e.ne0    = u.K;
        e.ncols  = u.ncols;
        e.data   = out;
        for (int a = 0; a < u.n_aliases; ++a) {
            ctx.fwht_q8().set(u.aliases[a], e);
        }
        return consumed;
    }
    const int64_t K     = u.K;
    const int64_t ncols = u.ncols;
    const ggml_type wtype = u.wtype;
    if (ggml_cuda_pq2_narrow_int8_enabled() && wtype == GGML_TYPE_PQ2_0 && ncols >= 5 && ncols <= 128) {
        return 0;
    }
    int64_t ne0_padded = 0;
    ggml_cuda_q8_1_layout layout = GGML_CUDA_Q8_1_AOS;
    if (K % n != 0 || !ggml_cuda_fwht_quantize_supported(n, K) || (signs && signs->ne[0] != K)) {
        return 0;
    }
    if (n > 2048 && !pq2_enabled) {
        return 0; // 4096-wide fused transforms go with the PQ2_0 switch
    }

    // same (type, ncols, ids) -> layout and padding as ggml_cuda_mul_mat_vec_q
    layout     = ggml_cuda_q8_1_layout_host(wtype, (int) ncols, false);
    if (layout != GGML_CUDA_Q8_1_AOS && layout != GGML_CUDA_Q8_1_PT && layout != GGML_CUDA_Q8_1_SOA_ISUM &&
            layout != GGML_CUDA_Q8_1_PQ2) {
        return 0;
    }
    ne0_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    if (layout == GGML_CUDA_Q8_1_SOA_ISUM) {
        ne0_padded = GGML_PAD(ne0_padded, GGML_CUDA_PTQ1_K_PAD);
    }
    if (ne0_padded % n != 0) {
        return 0; // the fused quantizer writes pad blocks in transform-width units
    }
    const size_t q8_bytes = (size_t) (ncols * ne0_padded) * sizeof(block_q8_1) / QK8_1;
    void * out = mm->data;
    if (out_aliases_in) {
        auto blk = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), q8_bytes);
        out = blk->get();
        ctx.fwht_q8().held.push_back(std::move(blk));
    } else if ((int64_t) q8_bytes > (int64_t) ggml_nbytes(mm)) {
        return 0; // the quantized rows must fit in the F32 output allocation
    }

    fwht_quantize_row_q8_1_cuda(x->data, x->type, signs ? (const float *) signs->data : nullptr, n,
                                out, layout, K, ne0_padded, ncols, ctx.stream());

    ggml_cuda_fwht_q8 e;
    e.layout = layout;
    e.ne0    = ne0_padded;
    e.ncols  = ncols;
    e.data   = out;
    for (int a = 0; a < u.n_aliases; ++a) {
        ctx.fwht_q8().set(u.aliases[a], e);
    }
    return consumed;
}

// GDN conv chain (on by default, GGML_CUDA_GDN_FUSION=0 disables it). For n_t tokens of one sequence (a decode step, n_t = 1, or a
// speculative verify step, n_t = 2..GGML_CUDA_GDN_CONV_MAX_TOKENS), a hybrid-model gated delta net layer runs these
// launch-bound kernels between the qkv projection and the GDN op:
//   GET_ROWS(conv cache, s_copy) -> RESHAPE -> CONCAT(state, qkv^T)
//   -> CPY(window -> conv cache row), once, or once per rollback snapshot (1 + n_rs_seq rows, #379 batches them)
//   -> SSM_CONV + SILU -> L2_NORM(q/k heads of the SILU output)
// ggml_cuda_ssm_conv_gdn_fused computes all of them in one launch with the same arithmetic per element (bit-identical
// outputs). The GET_ROWS is skipped where it stands and registered under its CONCAT (the conv cache is read there
// instead: nothing in between may write the cache or the ids); the CONCAT launches the fused kernel and marks the CPYs,
// SSM_CONV, SILU and L2_NORM done. The CONCAT and SSM_CONV outputs are never materialised, so they must have no other
// readers, and every node evaluated between the CONCAT and the L2_NORM (after the fused launch instead of before parts
// of it) must not touch any memory the fused kernel reads or writes. Megakernel-style launch removal in the spirit of
// Hazy Research's "no bubbles" low-latency decode work, applied as a graph-level fusion rather than a persistent kernel.
static bool ggml_cuda_gdn_fusion_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_GDN_FUSION");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

struct ggml_cuda_gdn_conv_plan {
    int                 i_gather = -1;
    int                 n_cpy    = 0;
    int                 i_cpy[GGML_CUDA_GDN_CONV_MAX_SNAP] = {};
    int                 i_conv   = -1; // the SILU is i_conv + 1
    int                 i_l2     = -1;
    const ggml_tensor * gather   = nullptr;
    ggml_cuda_ssm_conv_gdn_fused_args args;
};

struct ggml_cuda_byte_range {
    uintptr_t beg = 0;
    uintptr_t end = 0;
};

static ggml_cuda_byte_range ggml_cuda_tensor_range(const ggml_tensor * t) {
    ggml_cuda_byte_range r;
    r.beg = (uintptr_t) t->data;
    r.end = r.beg + (t->data ? ggml_nbytes(t) : 0);
    return r;
}

static bool ggml_cuda_ranges_overlap(const ggml_cuda_byte_range & a, const ggml_cuda_byte_range & b) {
    return a.beg < a.end && b.beg < b.end && a.beg < b.end && b.beg < a.end;
}

static int ggml_cuda_find_node(const ggml_cgraph * cgraph, const ggml_tensor * t, int from, int to) {
    for (int j = std::max(from, 0); j < to; ++j) {
        if (cgraph->nodes[j] == t) {
            return j;
        }
    }
    return -1;
}

static bool ggml_cuda_gdn_conv_match(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int ic,
                                     ggml_cuda_gdn_conv_plan & p) {
    const int n_nodes = cgraph->n_nodes;
    const ggml_tensor * cc = cgraph->nodes[ic];
    if (cc->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cc, 0) != 0 || cc->type != GGML_TYPE_F32 ||
            (cc->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(cc) || cc->ne[2] != 1 || cc->ne[3] != 1) {
        return false;
    }

    // conv state: RESHAPE(GET_ROWS(cache, ids)) of one cache row, [d_conv - 1, C]
    const ggml_tensor * s0 = cc->src[0];
    const ggml_tensor * gr = s0 ? s0->src[0] : nullptr;
    if (!s0 || s0->op != GGML_OP_RESHAPE || s0->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t d_conv = s0->ne[0] + 1;
    const int64_t C      = cc->ne[1];
    const int64_t n_t    = cc->ne[0] - (d_conv - 1);
    const bool prefill_conv = (n_t > GGML_CUDA_GDN_CONV_MAX_TOKENS && n_t <= 8192 && d_conv == 4);
    if (d_conv < 3 || d_conv > 5 || C > INT_MAX / 8 || n_t < 1 || (!prefill_conv && n_t > GGML_CUDA_GDN_CONV_MAX_TOKENS)) {
        return false;
    }
    if (s0->ne[1] != C || !gr || gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 ||
            (gr->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(gr) || gr->ne[0] != (d_conv - 1) * C ||
            ggml_nrows(gr) != 1) {
        return false;
    }
    const ggml_tensor * cache = gr->src[0];
    const ggml_tensor * ids   = gr->src[1];
    if (cache->type != GGML_TYPE_F32 || cache->data == nullptr || cache->nb[0] != sizeof(float) ||
            cache->ne[0] != gr->ne[0] || cache->nb[1] % sizeof(float) != 0 || cache->ne[2] != 1 || cache->ne[3] != 1 ||
            ids->type != GGML_TYPE_I32 || ids->data == nullptr || !ggml_is_contiguous(ids) || ggml_nelements(ids) != 1) {
        return false;
    }
    const int i_gr = ggml_cuda_find_node(cgraph, gr, 0, ic);
    const int i_s0 = ggml_cuda_find_node(cgraph, s0, i_gr + 1, ic);
    if (i_gr < 0 || i_s0 < 0 || ggml_node_get_use_count(cgraph, i_gr) != 1 || ggml_node_get_use_count(cgraph, i_s0) != 1) {
        return false;
    }

    // new tokens: [n_t, C] (the transposed projection output), token t of channel c at x + t*nb[0] + c*nb[1]
    const ggml_tensor * x = cc->src[1];
    if (x->type != GGML_TYPE_F32 || x->data == nullptr || x->ne[0] != n_t || x->ne[1] != C || x->ne[2] != 1 ||
            x->ne[3] != 1 || x->nb[1] % sizeof(float) != 0 || (n_t > 1 && x->nb[0] % sizeof(float) != 0)) {
        return false;
    }
    const int64_t x_stride     = (int64_t) (x->nb[1] / sizeof(float));
    const int64_t x_tok_stride = n_t > 1 ? (int64_t) (x->nb[0] / sizeof(float)) : 0;

    // the next real nodes: CPYs of d_conv - 1 consecutive conv inputs back into whole cache rows, each row written once
    // (one CPY at a plain step, one per rollback snapshot when the memory keeps n_rs_seq > 0 snapshots)
    const ggml_cuda_byte_range r_cache = ggml_cuda_tensor_range(cache);
    ggml_cuda_byte_range r_dsts[GGML_CUDA_GDN_CONV_MAX_SNAP];
    int n_cpy = 0;
    int i_last_cpy = ic;
    for (int j = ic + 1; j < n_nodes; ++j) {
        const ggml_tensor * cpy = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(cpy)) {
            continue;
        }
        if (cpy->op != GGML_OP_CPY || cpy->src[0]->op != GGML_OP_VIEW || cpy->src[0]->view_src != cc) {
            break;
        }
        const ggml_tensor * win = cpy->src[0];
        const ggml_tensor * dst = cpy->src[1];
        if (n_cpy == GGML_CUDA_GDN_CONV_MAX_SNAP || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                win->view_offs % sizeof(float) != 0 || win->view_offs / sizeof(float) > (size_t) n_t ||
                win->type != GGML_TYPE_F32 || win->ne[0] != d_conv - 1 || win->ne[1] != C || win->ne[2] != 1 ||
                win->ne[3] != 1 || win->nb[0] != sizeof(float) || win->nb[1] != cc->nb[1] ||
                dst->type != GGML_TYPE_F32 || dst->data == nullptr || !ggml_is_contiguous(dst) ||
                ggml_nelements(dst) != (d_conv - 1) * C) {
            return false;
        }
        const int i_win = ggml_cuda_find_node(cgraph, win, ic + 1, j);
        if (i_win < 0 || ggml_node_get_use_count(cgraph, i_win) != 1) {
            return false;
        }
        // one whole cache row (the row the state is read from, or another one), not written by an earlier CPY
        const ggml_cuda_byte_range r_dst = ggml_cuda_tensor_range(dst);
        if (r_dst.beg < r_cache.beg || r_dst.end > r_cache.end || (r_dst.beg - r_cache.beg) % cache->nb[1] != 0 ||
                r_dst.end - r_dst.beg != (uintptr_t) cache->ne[0] * sizeof(float)) {
            return false;
        }
        for (int k = 0; k < n_cpy; ++k) {
            if (ggml_cuda_ranges_overlap(r_dst, r_dsts[k])) {
                return false;
            }
        }
        r_dsts[n_cpy] = r_dst;
        p.i_cpy[n_cpy] = j;
        p.args.snap_offs[n_cpy] = (int64_t) ((r_dst.beg - r_cache.beg) / sizeof(float));
        p.args.snap_pos [n_cpy] = (int) (win->view_offs / sizeof(float));
        ++n_cpy;
        i_last_cpy = j;
    }
    if (n_cpy == 0 || ggml_node_get_use_count(cgraph, ic) != n_cpy + 1) {
        return false;
    }

    // SSM_CONV(concat) followed by its SILU
    int i_conv = -1;
    for (int j = i_last_cpy + 1; j < n_nodes - 1 && j < i_last_cpy + 64; ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_SSM_CONV && cgraph->nodes[j]->src[0] == cc) {
            i_conv = j;
            break;
        }
    }
    if (i_conv < 0) {
        return false;
    }
    const ggml_tensor * conv = cgraph->nodes[i_conv];
    const ggml_tensor * silu = cgraph->nodes[i_conv + 1];
    const ggml_tensor * w    = conv->src[1];
    if (conv->type != GGML_TYPE_F32 || (conv->flags & GGML_TENSOR_FLAG_OUTPUT) || conv->ne[0] != C ||
            conv->ne[1] != n_t || conv->ne[2] != 1 || ggml_node_get_use_count(cgraph, i_conv) != 1 ||
            w->type != GGML_TYPE_F32 || w->data == nullptr || w->ne[0] != d_conv || w->ne[1] != C ||
            w->nb[0] != sizeof(float) || w->nb[1] % sizeof(float) != 0 ||
            silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || silu->src[0] != conv ||
            silu->type != GGML_TYPE_F32 || (silu->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(silu) ||
            !ggml_are_same_shape(silu, conv)) {
        return false;
    }

    // L2_NORM over rows of a leading view of each token's SILU output (the q and k heads)
    int i_l2 = i_conv + 2;
    while (i_l2 < n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[i_l2])) {
        ++i_l2;
    }
    if (i_l2 >= n_nodes) {
        return false;
    }
    const ggml_tensor * l2  = cgraph->nodes[i_l2];
    const ggml_tensor * qkv = l2->src[0];
    if (l2->op != GGML_OP_L2_NORM || l2->type != GGML_TYPE_F32 || (l2->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !ggml_is_contiguous(l2) || qkv->op != GGML_OP_VIEW || qkv->view_src != silu || qkv->view_offs != 0 ||
            !ggml_are_same_shape(qkv, l2) || qkv->nb[0] != sizeof(float) ||
            qkv->nb[1] != qkv->ne[0] * sizeof(float) || qkv->ne[2] != n_t || qkv->ne[3] != 1 ||
            (n_t > 1 && qkv->nb[2] != silu->nb[1])) {
        return false;
    }
    const int64_t row_len = qkv->ne[0];
    const int64_t n_l2    = qkv->ne[0] * qkv->ne[1];
    float eps;
    memcpy(&eps, l2->op_params, sizeof(float));
    if (row_len % WARP_SIZE != 0 || row_len >= 1024 || C % row_len != 0 || n_l2 > C || eps < 0.0f ||
            (!prefill_conv && ((d_conv - 1 + 2*n_t)*row_len + n_t)*sizeof(float) > 48*1024)) {
        return false;
    }

    // memory: outputs against inputs (the SILU may overwrite the new tokens in place, element for element)
    const uintptr_t x_span = (uintptr_t) ((n_t - 1) * x_tok_stride * sizeof(float) + (C - 1) * x->nb[1] + sizeof(float));
    const ggml_cuda_byte_range r_x    = { (uintptr_t) x->data, (uintptr_t) x->data + x_span };
    const ggml_cuda_byte_range r_w    = ggml_cuda_tensor_range(w);
    const ggml_cuda_byte_range r_ids  = ggml_cuda_tensor_range(ids);
    const ggml_cuda_byte_range r_y    = ggml_cuda_tensor_range(silu);
    const ggml_cuda_byte_range r_l2   = ggml_cuda_tensor_range(l2);
    const bool y_in_place = silu->data == x->data && x_stride == 1 && (n_t == 1 || x_tok_stride == C);
    if ((!y_in_place && ggml_cuda_ranges_overlap(r_y, r_x)) || ggml_cuda_ranges_overlap(r_y, r_cache) ||
            ggml_cuda_ranges_overlap(r_y, r_w) || ggml_cuda_ranges_overlap(r_y, r_ids) || ggml_cuda_ranges_overlap(r_y, r_l2) ||
            ggml_cuda_ranges_overlap(r_l2, r_x) || ggml_cuda_ranges_overlap(r_l2, r_cache) ||
            ggml_cuda_ranges_overlap(r_l2, r_w) || ggml_cuda_ranges_overlap(r_l2, r_ids)) {
        return false;
    }
    for (int k = 0; k < n_cpy; ++k) {
        if (ggml_cuda_ranges_overlap(r_dsts[k], r_x) || ggml_cuda_ranges_overlap(r_dsts[k], r_w) ||
                ggml_cuda_ranges_overlap(r_dsts[k], r_ids)) {
            return false;
        }
    }

    // between the GET_ROWS and the CONCAT nothing may write the cache or the ids (the read moves to the CONCAT)
    for (int j = i_gr + 1; j < ic; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        const ggml_cuda_byte_range r = ggml_cuda_tensor_range(t);
        if (ggml_cuda_ranges_overlap(r, r_cache) || ggml_cuda_ranges_overlap(r, r_ids)) {
            return false;
        }
    }
    // nodes evaluated after the fused launch instead of before parts of it. The SILU and L2_NORM outputs are written
    // early (at the CONCAT instead of at their own nodes), so no node in between may write or read their memory (the
    // allocator may lend it out until those nodes run); the cache rows, weights and ids must not be written either,
    // and the cache must not be read. The new tokens are read at the CONCAT position as before, and the CONCAT and
    // SSM_CONV buffers are never written by the fused launch, so nodes in between may reuse that memory freely.
    for (int j = ic + 1; j < i_l2; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (j <= i_last_cpy || j == i_conv || j == i_conv + 1 || ggml_cuda_is_view_or_noop(t)) {
            continue; // (ic, i_last_cpy] holds only the CPYs and their views
        }
        const ggml_cuda_byte_range r = ggml_cuda_tensor_range(t);
        for (const ggml_cuda_byte_range & o : { r_y, r_l2, r_cache, r_w, r_ids }) {
            if (ggml_cuda_ranges_overlap(r, o)) {
                return false;
            }
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s] == nullptr) {
                continue;
            }
            const ggml_cuda_byte_range rs = ggml_cuda_tensor_range(t->src[s]);
            for (const ggml_cuda_byte_range & o : { r_y, r_l2, r_cache }) {
                if (ggml_cuda_ranges_overlap(rs, o)) {
                    return false;
                }
            }
        }
    }
    // concurrent streams would run other nodes alongside
    for (const auto & [fork_node, event] : ctx.stream_context().concurrent_events) {
        GGML_UNUSED(fork_node);
        for (const ggml_tensor * t : { gr, cc, conv, silu, l2 }) {
            if (event.stream_mapping.find(t) != event.stream_mapping.end()) {
                return false;
            }
        }
        for (int k = 0; k < n_cpy; ++k) {
            if (event.stream_mapping.find(cgraph->nodes[p.i_cpy[k]]) != event.stream_mapping.end()) {
                return false;
            }
        }
    }

    p.i_gather = i_gr;
    p.n_cpy    = n_cpy;
    p.i_conv   = i_conv;
    p.i_l2     = i_l2;
    p.gather   = gr;

    ggml_cuda_ssm_conv_gdn_fused_args & a = p.args;
    a.cache        = (const float *) cache->data;
    a.ids          = (const int32_t *) ids->data;
    a.row_stride   = (int64_t) (cache->nb[1] / sizeof(float));
    a.x            = (const float *) x->data;
    a.x_stride     = x_stride;
    a.x_tok_stride = x_tok_stride;
    a.w            = (const float *) w->data;
    a.w_stride     = (int) (w->nb[1] / sizeof(float));
    a.cache_dst    = (float *) cache->data;
    a.n_snap       = n_cpy;
    a.y            = (float *) silu->data;
    a.l2           = (float *) l2->data;
    a.n_l2         = (int) n_l2;
    a.n_channels   = (int) C;
    a.n_t          = (int) n_t;
    a.row_len      = (int) row_len;
    a.d_conv       = (int) d_conv;
    a.eps          = eps;
    return true;
}

// at the conv-state GET_ROWS: skip it when its CONCAT will run the fused chain
static bool ggml_cuda_try_gdn_conv_gather_skip(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_cuda_gdn_fusion_enabled() || !GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    for (int j = node_idx + 1; j < cgraph->n_nodes && j < node_idx + 128; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op != GGML_OP_CONCAT || t->src[0] == nullptr || t->src[0]->op != GGML_OP_RESHAPE || t->src[0]->src[0] != gr) {
            continue;
        }
        ggml_cuda_gdn_conv_plan p;
        if (!ggml_cuda_gdn_conv_match(ctx, cgraph, j, p) || p.i_gather != node_idx) {
            return false;
        }
        ctx.gdn_conv().gathers[t] = gr;
        return true;
    }
    return false;
}

// Tiled 2D race-free prefill fused GDN conv1d (d_conv=4) + SiLU + snapshot:
// Pass 1 saves only the 3 boundary halo tokens per T_TILE=64 tile (3.9 MB) + cache snapshots.
// Pass 2 launches (C/128) x (n_t/64) = 2,560 CTAs saturating all SMs, 1 coalesced read + 1 write/token,
// 100% safe even when y == x in-place!
constexpr int GDN_PREFILL_CONV_TILE = 64;

__global__ static void gdn_prefill_conv_halo_kernel(
        ggml_cuda_ssm_conv_gdn_fused_args a, float * __restrict__ halo, const int n_tiles) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= a.n_channels) {
        return;
    }
    const int row_idx = a.ids[0];
    const float * c_row = a.cache + (int64_t) row_idx * a.row_stride + (int64_t) c * 3;
    const float * x_c   = a.x + (int64_t) c * a.x_stride;
    const int64_t x_ts  = a.x_tok_stride;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int t0 = tile * GDN_PREFILL_CONV_TILE;
        float * h = halo + ((int64_t) tile * a.n_channels + c) * 3;
        if (tile == 0) {
            h[0] = c_row[0];
            h[1] = c_row[1];
            h[2] = c_row[2];
        } else {
            h[0] = x_c[(int64_t) (t0 - 3) * x_ts];
            h[1] = x_c[(int64_t) (t0 - 2) * x_ts];
            h[2] = x_c[(int64_t) (t0 - 1) * x_ts];
        }
    }
    for (int k = 0; k < a.n_snap; ++k) {
        float * dst = a.cache_dst + a.snap_offs[k] + (int64_t) c * 3;
        const int sp = a.snap_pos[k];
        float vals[3];
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) {
            const int pos = sp + tap;
            vals[tap] = (pos < 3) ? c_row[pos] : x_c[(int64_t) (pos - 3) * x_ts];
        }
        dst[0] = vals[0];
        dst[1] = vals[1];
        dst[2] = vals[2];
    }
}

__global__ static void gdn_prefill_conv_silu_tiled_kernel(
        ggml_cuda_ssm_conv_gdn_fused_args a, const float * __restrict__ halo) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile = blockIdx.y;
    const int t0 = tile * GDN_PREFILL_CONV_TILE;
    if (c >= a.n_channels || t0 >= a.n_t) {
        return;
    }
    const float * wc = a.w + (int64_t) c * a.w_stride;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float * h = halo + ((int64_t) tile * a.n_channels + c) * 3;
    float r0 = h[0], r1 = h[1], r2 = h[2];

    const float * x_c = a.x + (int64_t) c * a.x_stride;
    float * y_c = a.y + c;
    const int64_t x_ts = a.x_tok_stride;
    const int64_t y_ts = a.n_channels;
    const int t_end = min(t0 + GDN_PREFILL_CONV_TILE, a.n_t);

    const bool do_l2 = (a.row_len == 128) && (blockIdx.x * 128 < (unsigned) a.n_l2);
    __shared__ float sw[2][4];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float eps2 = a.eps * a.eps;
    float * l2_c = a.l2 + c;

    for (int t = t0; t < t_end; ++t) {
        const float r3 = x_c[(int64_t) t * x_ts];
        const float v = r0 * w0 + r1 * w1 + r2 * w2 + r3 * w3;
        const float sv = v / (1.0f + __expf(-v));
        if (do_l2) {
            const int b = t & 1;
            float sq = sv * sv;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sq += __shfl_xor_sync(0xFFFFFFFF, sq, off);
            }
            if (lane == 0) {
                sw[b][warp] = sq;
            }
            __syncthreads();
            const float tot = sw[b][0] + sw[b][1] + sw[b][2] + sw[b][3];
            l2_c[(int64_t) t * a.n_l2] = sv * rsqrtf(fmaxf(tot, eps2));
        } else {
            y_c[(int64_t) t * y_ts] = sv;
        }
        r0 = r1;
        r1 = r2;
        r2 = r3;
    }
}

// at the CONCAT whose GET_ROWS was skipped: one launch for the chain
static bool ggml_cuda_try_gdn_conv_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    auto & reg = ctx.gdn_conv();
    const ggml_tensor * cc = cgraph->nodes[node_idx];
    const auto it = reg.gathers.find(cc);
    if (it == reg.gathers.end()) {
        return false;
    }
    ggml_cuda_gdn_conv_plan p;
    if (!ggml_cuda_gdn_conv_match(ctx, cgraph, node_idx, p) || p.gather != it->second) {
        // not expected (the match is a pure function of the graph): materialise the skipped gather, run unfused
        ggml_cuda_op_get_rows(ctx, const_cast<ggml_tensor *>(it->second));
        return false;
    }
    if (p.args.n_t > GGML_CUDA_GDN_CONV_MAX_TOKENS) {
        cudaStream_t stream = ctx.stream();
        const int n_tiles = (p.args.n_t + GDN_PREFILL_CONV_TILE - 1) / GDN_PREFILL_CONV_TILE;
        ggml_cuda_pool_alloc<float> halo(ctx.pool(), (size_t) n_tiles * (size_t) p.args.n_channels * 3);
        const unsigned c_blocks = (unsigned) ((p.args.n_channels + 127) / 128);
        gdn_prefill_conv_halo_kernel<<<c_blocks, 128, 0, stream>>>(p.args, halo.get(), n_tiles);
        gdn_prefill_conv_silu_tiled_kernel<<<dim3(c_blocks, (unsigned) n_tiles, 1), 128, 0, stream>>>(p.args, halo.get());
        CUDA_CHECK(cudaGetLastError());
        for (int k = 0; k < p.n_cpy; ++k) {
            reg.done.insert(cgraph->nodes[p.i_cpy[k]]);
        }
        reg.done.insert(cgraph->nodes[p.i_conv]);
        reg.done.insert(cgraph->nodes[p.i_conv + 1]);
        if (p.args.row_len == 128) {
            reg.done.insert(cgraph->nodes[p.i_l2]);
        }
        return true;
    }
    ggml_cuda_ssm_conv_gdn_fused(ctx, p.args);
    for (int k = 0; k < p.n_cpy; ++k) {
        reg.done.insert(cgraph->nodes[p.i_cpy[k]]);
    }
    reg.done.insert(cgraph->nodes[p.i_conv]);
    reg.done.insert(cgraph->nodes[p.i_conv + 1]);
    reg.done.insert(cgraph->nodes[p.i_l2]);
    return true;
}

// GET_ROWS and let the kernel index the cache row directly. Single-sequence only: with several
// sequences a gathered row may alias a row another sequence writes in the same op.
// GGML_CUDA_GDN_GATHER_FUSION=0 disables it. The registration lives in the evaluating context
// (ctx.gdn_gathers()), so concurrent contexts or threads never see each other's skips.
static bool ggml_cuda_try_gdn_gather_skip(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    static const bool disabled = getenv("GGML_CUDA_GDN_GATHER_FUSION") != nullptr &&
                                 atoi(getenv("GGML_CUDA_GDN_GATHER_FUSION")) == 0;
    if (disabled) {
        return false;
    }
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !ggml_is_contiguous(gr)) {
        return false;
    }
    const ggml_tensor * cache = gr->src[0];
    const ggml_tensor * ids   = gr->src[1];
    if (cache->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || cache->data == nullptr || ids->data == nullptr ||
        cache->nb[0] != sizeof(float) || cache->nb[1] % sizeof(float) != 0 || !ggml_is_contiguous(ids) ||
        ids->ne[0] != 1 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        gr->ne[1] != 1 || gr->ne[2] != 1 || gr->ne[3] != 1 || gr->ne[0] != cache->ne[0]) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, node_idx) != 1) {
        return false;
    }
    const ggml_tensor * cur = gr;
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if ((n->op == GGML_OP_GATED_DELTA_NET || n->op == GGML_OP_GATED_DELTA_NET_REPLAY) && n->src[5] == cur) {
            const ggml_tensor * v = n->src[2];
            const int64_t       D = v->ne[0] * v->ne[0] * v->ne[1];
            if (gr->ne[0] != D || v->ne[3] != 1 || ggml_nelements(cur) != D) {
                return false;
            }
            // ggml-alloc sees the GET_ROWS as the last reader of ids, so a later node can reuse ids memory before the GDN kernel reads it
            const char * ids_beg = (const char *) ids->data;
            const char * ids_end = ids_beg + ggml_nbytes(ids);
            for (int k = node_idx + 1; k <= j; ++k) {
                const ggml_tensor * t = cgraph->nodes[k];
                const char *        b = (const char *) t->data;
                if (b != nullptr && b < ids_end && ids_beg < b + ggml_nbytes(t)) {
                    return false;
                }
            }
            // on a concurrent stream, other nodes can run at the same time as the GDN op
            for (const auto & [fork_node, event] : ctx.stream_context().concurrent_events) {
                if (event.stream_mapping.find(n) != event.stream_mapping.end()) {
                    return false;
                }
            }
            ggml_cuda_gated_delta_net_gather gather;
            gather.base       = (const float *) cache->data;
            gather.ids        = (const int32_t *) ids->data;
            gather.row_stride = (int64_t) (cache->nb[1] / sizeof(float));
            ctx.gdn_gathers().set(n, gather);
            return true;
        }
        if (n->op == GGML_OP_RESHAPE && n->src[0] == cur) {
            if (ggml_node_get_use_count(cgraph, j) != 1) {
                return false;
            }
            cur = n;
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == cur || (n->view_src != nullptr && n->view_src == gr)) {
                return false;
            }
        }
    }
    return false;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// Match GATED_DELTA_NET_REPLAY + CPY (base -> plane 0) (+ SET_ROWS for n_seqs == 1).
// For n_seqs > 1, fuses CPY (eliminating the D*n_seqs copy) while leaving SET_ROWS unfused for cell-swap safety.
static int ggml_cuda_try_gdn_replay_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gdn_replay_fused & fused) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    if (gdn->op != GGML_OP_GATED_DELTA_NET_REPLAY || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * src_v    = gdn->src[2];
    const int64_t       S_v      = src_v->ne[0];
    const int64_t       H        = src_v->ne[1];
    const int64_t       n_tokens = src_v->ne[2];
    const int64_t       n_seqs   = src_v->ne[3];
    const int64_t       K        = ggml_get_op_params_i32(gdn, 0);
    const int64_t       D        = S_v * S_v * H;
    const int64_t       R        = H * (2 * S_v + 1);
    const size_t        base_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);
    const size_t        rec_off  = base_off + ggml_row_size(GGML_TYPE_F32, D * n_seqs);

    const ggml_tensor * cpy  = nullptr;
    const ggml_tensor * sr   = nullptr;
    int                 skip = 0;
    const bool          want_sr = (n_seqs == 1);
    int                 n_sr_multi = 0;
    const ggml_tensor * sr_first = nullptr;
    int64_t             sr_cell_stride_bytes = 0;
    const int64_t       rec_elems = R * (K - 1);
    const size_t        rec_bytes = ggml_row_size(GGML_TYPE_F32, rec_elems);
    int                 skip_after_cpy = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->flags & GGML_TENSOR_FLAG_OUTPUT) {
            if (cpy != nullptr) {
                n_sr_multi = -1;
                break;
            }
            return 0;
        }
        if (n->op == GGML_OP_CPY && cpy == nullptr) {
            cpy = n;
            skip = j - node_idx;
            skip_after_cpy = skip;
        } else if (want_sr && n->op == GGML_OP_SET_ROWS && sr == nullptr) {
            sr = n;
            skip = j - node_idx;
            break;
        } else if (!want_sr && cpy != nullptr && n->op == GGML_OP_SET_ROWS && n_sr_multi < n_seqs &&
                   n_seqs <= ggml_cuda_gdn_replay_fused::MAX_BATCH_SEQS) {
            const ggml_tensor * rs = n->src[0];
            const ggml_tensor * ri = n->src[1];
            const int64_t si = n_sr_multi;
            bool ok = rs->op == GGML_OP_VIEW && rs->view_src == gdn &&
                      rs->view_offs == rec_off + (size_t) si * rec_bytes &&
                      ggml_is_contiguous(rs) && rs->ne[0] == rec_elems && ggml_nrows(rs) == 1 &&
                      n->type == GGML_TYPE_F32 && n->data != nullptr &&
                      n->ne[0] == rec_elems && n->nb[0] == sizeof(float) && n->nb[1] == rec_bytes &&
                      ri->type == GGML_TYPE_I64 && ggml_nelements(ri) == 1 && ri->data != nullptr &&
                      !(ri->buffer && ggml_backend_buffer_is_host(ri->buffer));
            if (ok && si == 0) {
                sr_first = n;
            } else if (ok && si == 1) {
                sr_cell_stride_bytes = (int64_t) ((const char *) n->data - (const char *) sr_first->data);
                ok = sr_cell_stride_bytes > 0 && (sr_cell_stride_bytes % (int64_t) sizeof(float)) == 0;
            } else if (ok && si > 1) {
                ok = ((const char *) n->data - (const char *) sr_first->data) == si * sr_cell_stride_bytes;
            }
            if (!ok) {
                n_sr_multi = -1;
                break;
            }
            fused.rec_batch_wrows[si] = (const int64_t *) ri->data;
            ++n_sr_multi;
            if (n_sr_multi == n_seqs) {
                skip = j - node_idx;
                break;
            }
        } else {
            break;
        }
    }
    if (cpy == nullptr) {
        return 0;
    }
    const ggml_tensor * cs = cpy->src[0];
    const ggml_tensor * cd = cpy->src[1];
    if (cs->op != GGML_OP_VIEW || cs->view_src != gdn || cs->view_offs != base_off || !ggml_is_contiguous(cs) ||
        ggml_nelements(cs) != D * n_seqs || cd->type != GGML_TYPE_F32 || cd->data == nullptr ||
        !ggml_is_contiguous(cd)) {
        return 0;
    }
    fused.base = (float *) cd->data;
    fused.rec  = nullptr;
    if (want_sr && sr != nullptr) {
        const ggml_tensor * rs = sr->src[0];
        const ggml_tensor * ri = sr->src[1];
        if (rs->op == GGML_OP_VIEW && rs->view_src == gdn && rs->view_offs == rec_off && ggml_is_contiguous(rs) &&
            rs->ne[0] == R * (K - 1) && ggml_nrows(rs) == 1 && sr->type == GGML_TYPE_F32 && sr->data != nullptr &&
            sr->ne[0] == R * (K - 1) && sr->nb[0] == sizeof(float) && sr->nb[1] % sizeof(float) == 0 &&
            ri->type == GGML_TYPE_I64 && ggml_nelements(ri) == 1 && ri->data != nullptr &&
            !(ri->buffer && ggml_backend_buffer_is_host(ri->buffer))) {
            fused.rec            = (float *) sr->data;
            fused.rec_rows       = (const int64_t *) ri->data;
            fused.rec_row_stride = (int64_t) (sr->nb[1] / sizeof(float));
        } else {
            return 0;
        }
    } else if (!want_sr && n_sr_multi == n_seqs && sr_first != nullptr) {
        fused.rec_batch_dst         = (float *) sr_first->data;
        fused.rec_batch_n_seqs      = (int) n_seqs;
        fused.rec_batch_cell_stride = sr_cell_stride_bytes / (int64_t) sizeof(float);
        fused.rec_batch_bank_stride = rec_elems;
    } else if (!want_sr) {
        skip = skip_after_cpy;
    }
    return skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_UNUSED_VARS(cgraph, node_idx, selection, args);
    return false;
#else
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
#endif // defined(GGML_USE_HIP) && HIP_VERSION < 60300000
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false,
                                                 const bool          allow_exact_alias = false) {
    // An exact alias is a destination that shares the base pointer, type and
    // contiguous layout of a source, i.e. element k of the destination is
    // element k of the source. Fusions whose kernels are elementwise within the
    // unit they parallelize over can opt into accepting those; a partial
    // overlap is still rejected for everyone.
    auto is_exact_alias = [&](const ggml_tensor * a, const ggml_tensor * b) {
        return allow_exact_alias && a->data == b->data && a->type == b->type &&
               ggml_are_same_shape(a, b) && ggml_is_contiguous(a) && ggml_is_contiguous(b);
    };

    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_exact_alias(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


// GGML_CUDA_MOE_EP_VERIFY=1: compare against the existing fused reduction (REPEAT/GET_ROWS result input);
// =2: compare against the fully unfused MUL, MUL, VIEW, ADD nodes
static int ggml_cuda_moe_ep_verify_mode() {
    static const int m = getenv("GGML_CUDA_MOE_EP_VERIFY") ? atoi(getenv("GGML_CUDA_MOE_EP_VERIFY")) : 0;
    return m;
}

// GGML_CUDA_MOE_EP_FUSION bit mask: 1 = scale gather into the weighted reduction, 2 = router RMS_NORM+SCALE+MUL, 4 = router matvec + top-k
static int ggml_cuda_moe_ep_bits() {
    static const int bits = getenv("GGML_CUDA_MOE_EP_FUSION") ? atoi(getenv("GGML_CUDA_MOE_EP_FUSION")) : 0;
    return bits;
}
static bool ggml_cuda_moe_ep_fusion_enabled() {
    return (ggml_cuda_moe_ep_bits() & 1) != 0;
}

// REPEAT(w_s [1,n_expert,1] -> [1,n_expert,n_tokens]) -> GET_ROWS(.., ids) -> MUL(experts, .) -> weighted reduction.
// The gather and the repeat only feed the expert-scale MUL, so the reduction kernel can read w_s[ids] directly.
static bool ggml_cuda_match_moe_scale_gather(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match,
        const ggml_tensor *& table,
        const ggml_tensor *& ids) {
    if (node_idx + 3 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * rep = cgraph->nodes[node_idx];
    const ggml_tensor * get = cgraph->nodes[node_idx + 1];
    if (rep->op != GGML_OP_REPEAT || get->op != GGML_OP_GET_ROWS || get->src[0] != rep ||
            rep->type != GGML_TYPE_F32 || get->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * w_s = rep->src[0];
    ids = get->src[1];
    if (!w_s || !ids || w_s->type != GGML_TYPE_F32 || !ggml_is_contiguous(w_s) || w_s->ne[0] != 1 || w_s->ne[2] != 1 ||
            w_s->ne[3] != 1 || ids->type != GGML_TYPE_I32 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 ||
            rep->ne[0] != 1 || rep->ne[1] != w_s->ne[1] || rep->ne[3] != 1 ||
            get->ne[0] != 1 || get->ne[1] != ids->ne[0] || get->ne[2] != ids->ne[1] || get->ne[3] != 1 ||
            rep->ne[2] != ids->ne[1] || !ggml_is_contiguous(rep) || !ggml_is_contiguous(get)) {
        return false;
    }
    if (!ggml_cuda_match_moe_weighted_reduction(cgraph, node_idx + 2, match) || match.expert_scale != get) {
        return false;
    }
    const int count = match.node_count + 2;
    std::vector<ggml_op> ops(count);
    for (int j = 0; j < count; ++j) {
        ops[j] = cgraph->nodes[node_idx + j]->op;
    }
    const int output_idx = node_idx + count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, count, ops.data(), &output_idx, 1)) {
        return false;
    }
    table = w_s;
    return true;
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    std::initializer_list<enum ggml_op> hc_post_gate_ops = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

    if (is_equal(hc_post_gate_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            ggml_get_op_params_f32(scale_in, 1) == 0.0f && ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
            scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(scale_in->src[0], scale_out)) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

static bool ggml_cuda_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const int64_t a_start = (int64_t) a->data;
    const int64_t a_end   = a_start + (int64_t) ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

    const int64_t b_start = (int64_t) b->data;
    const int64_t b_end   = b_start + (int64_t) ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

    return (b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end);
}

// MoE weighted-expert reduce, as emitted by llm_graph_context::build_moe_ffn:
//     MUL(down_out, router_weights)  ->  n_exp x VIEW  ->  (n_exp - 1) x ADD
// The MUL writes a second full-size [n_embd, n_expert_used, n_tokens] F32 tensor that only
// exists so the add chain can read it back, which at large ubatch is tens of MB written and
// read again per layer. Folding the weighting into the reduction removes that round trip and
// is bit-identical to the unfused sequence.
static bool ggml_cuda_should_fuse_moe_weighted_reduce(const ggml_cgraph * cgraph,
                                                      int                 node_idx,
                                                      int                 n_exp,
                                                      bool &              stage_weights) {
    const ggml_tensor * mul = cgraph->nodes[node_idx];
    const ggml_tensor * x   = mul->src[0];
    const ggml_tensor * w   = mul->src[1];

    stage_weights = false;

    if (!x || !w) {
        return false;
    }

    if (mul->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(mul) || !ggml_is_contiguous(x) || !ggml_are_same_shape(mul, x)) {
        return false;
    }

    const int64_t ne0   = mul->ne[0];
    const int64_t n_tok = mul->ne[2];

    // the router weights must be a per (expert, token) scalar, broadcast along ne0
    if (w->ne[0] != 1 || w->ne[1] != n_exp || w->ne[2] != n_tok || w->ne[3] != 1) {
        return false;
    }

    // one block row per token
    if (n_tok > 65535) {
        return false;
    }

    // n_exp views of the weighted tensor, one per expert slot, in slot order
    for (int e = 0; e < n_exp; ++e) {
        const ggml_tensor * v = cgraph->nodes[node_idx + 1 + e];

        if (v->view_src != mul || v->type != GGML_TYPE_F32) {
            return false;
        }
        if (v->ne[0] != ne0 || v->ne[1] != n_tok || v->ne[2] != 1 || v->ne[3] != 1) {
            return false;
        }
        if (v->nb[0] != mul->nb[0] || v->nb[1] != mul->nb[2]) {
            return false;
        }
        if ((const char *) v->data != (const char *) mul->data + e*mul->nb[1]) {
            return false;
        }
    }

    // left-to-right add chain over those views
    for (int k = 0; k < n_exp - 1; ++k) {
        const ggml_tensor * add = cgraph->nodes[node_idx + 1 + n_exp + k];

        const ggml_tensor * expected_a = (k == 0) ? cgraph->nodes[node_idx + 1] : cgraph->nodes[node_idx + n_exp + k];
        const ggml_tensor * expected_b = cgraph->nodes[node_idx + 2 + k];

        if (add->src[0] != expected_a || add->src[1] != expected_b) {
            return false;
        }
        if (add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add)) {
            return false;
        }
        if (add->ne[0] != ne0 || add->ne[1] != n_tok || add->ne[2] != 1 || add->ne[3] != 1) {
            return false;
        }
    }

    // Aliasing. The fused kernel reads x and w while it is writing dst, so anything the
    // graph allocator has placed under dst is a cross-block race. ggml_cuda_check_fusion_
    // memory_ranges is not used here because it ignores sources whose op is GGML_OP_NONE
    // and because it can only reject, while the weights case has a cheap repair.
    const ggml_tensor * dst = cgraph->nodes[node_idx + 2*n_exp - 1];

    if (ggml_cuda_tensors_overlap(dst, x)) {
        return false;
    }

    if (ggml_cuda_tensors_overlap(dst, w)) {
        // The router weights die at the MUL, so the allocator routinely packs the add-chain
        // output over them. They are only n_expert_used*n_tokens floats, so the op snapshots
        // them into pool scratch rather than giving up the fusion.
        if (!ggml_is_contiguous(w)) {
            return false;
        }

        stage_weights = true;
    }

    return true;
}

// try and fuse nodes and return the number of nodes to skip
// extent [first, last byte + 1) of a (possibly strided, non-negative stride) tensor view
static void ggml_cuda_view_extent(const ggml_tensor * t, uintptr_t & beg, uintptr_t & end) {
    size_t span = ggml_type_size(t->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        span += (size_t) (t->ne[d] - 1) * t->nb[d];
    }
    beg = (uintptr_t) t->data;
    end = beg + span;
}

// A run of independent F32 -> F32 CPY nodes that share shape and strides (only the data pointers differ), with only
// view nodes in between: the recurrent rollback writes one conv-state snapshot per slot this way. One launch instead
// of one per copy. Returns the number of extra nodes consumed (0: not applicable).
static int ggml_cuda_try_batched_cpy(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * first = cgraph->nodes[i];
    auto matches = [&](const ggml_tensor * n) {
        if (n->op != GGML_OP_CPY || n->src[0]->type != GGML_TYPE_F32 || n->src[1]->type != GGML_TYPE_F32 ||
                (n->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_nelements(n->src[0]) > INT_MAX) {
            return false;
        }
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (n->src[0]->ne[d] != first->src[0]->ne[d] || n->src[0]->nb[d] != first->src[0]->nb[d] ||
                    n->src[1]->ne[d] != first->src[1]->ne[d] || n->src[1]->nb[d] != first->src[1]->nb[d]) {
                return false;
            }
        }
        return true;
    };
    if (!matches(first)) {
        return 0;
    }

    int idx[CUDA_CPY_BATCH_MAX];
    int n = 0;
    int last = i;
    for (int j = i; j < cgraph->n_nodes && n < CUDA_CPY_BATCH_MAX; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (matches(node)) {
            // independent of the copies before it: no earlier destination overlaps this source or destination
            uintptr_t sb, se, db, de;
            ggml_cuda_view_extent(node->src[0], sb, se);
            ggml_cuda_view_extent(node->src[1], db, de);
            bool ok = true;
            for (int k = 0; k < n && ok; ++k) {
                uintptr_t pb, pe;
                ggml_cuda_view_extent(cgraph->nodes[idx[k]]->src[1], pb, pe);
                ok = (pe <= sb || se <= pb) && (pe <= db || de <= pb);
            }
            if (!ok) {
                break;
            }
            idx[n++] = j;
            last = j;
        } else if (!ggml_cuda_is_view_or_noop(node)) {
            break;
        }
    }
    if (n < 2) {
        return 0;
    }

    ggml_cuda_cpy_batch batch;
    for (int k = 0; k < n; ++k) {
        batch.src[k] = (const char *) cgraph->nodes[idx[k]]->src[0]->data;
        batch.dst[k] = (char *) cgraph->nodes[idx[k]]->src[1]->data;
    }
    ggml_cuda_cpy_f32_batched(*cuda_ctx, first->src[0], first->src[1], batch, n);
    return last - i;
}

// Two consecutive independent F32/F16/BF16 mat-vecs on the same activation (the GDN alpha and beta projections, which
// the fused GDN op consumes raw) as one launch (on by default, GGML_CUDA_GDN_FUSION=0 disables it). Returns the
// number of nodes consumed.
static int ggml_cuda_try_mmvf_pair(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * mm0 = cgraph->nodes[i];
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        ++j;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mm1 = cgraph->nodes[j];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    for (const ggml_tensor * mm : { mm0, mm1 }) {
        const ggml_tensor * w = mm->src[0];
        if (mm->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD || !w || !w->buffer ||
                ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                !ggml_cuda_should_use_mmvf(w->type, cc, ggml_cuda_info().devices[ctx.device].warp_size, w->ne, w->nb, mm->src[1]->ne[1])) {
            return 0;
        }
    }
    if (mm1->src[1] != mm0->src[1] || mm1->src[0] == mm0 || mm1->src[1] == mm0) {
        return 0;
    }
    // leave a pair whose second product starts another fusion (bias add, GLU) to that fusion
    int k = j + 1;
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        ++k;
    }
    if (k < cgraph->n_nodes && (cgraph->nodes[k]->op == GGML_OP_ADD || cgraph->nodes[k]->op == GGML_OP_GLU)) {
        return 0;
    }
    const ggml_cuda_byte_range r0 = ggml_cuda_tensor_range(mm0);
    const ggml_cuda_byte_range r1 = ggml_cuda_tensor_range(mm1);
    for (const ggml_tensor * t : { mm0->src[0], mm1->src[0], mm0->src[1] }) {
        const ggml_cuda_byte_range r = ggml_cuda_tensor_range(t);
        if (ggml_cuda_ranges_overlap(r, r0) || ggml_cuda_ranges_overlap(r, r1)) {
            return 0;
        }
    }
    if (ggml_cuda_ranges_overlap(r0, r1)) {
        return 0;
    }
    for (const auto & [fork_node, event] : ctx.stream_context().concurrent_events) {
        GGML_UNUSED(fork_node);
        if (event.stream_mapping.find(mm0) != event.stream_mapping.end() ||
                event.stream_mapping.find(mm1) != event.stream_mapping.end()) {
            return 0;
        }
    }
    if (!ggml_cuda_mul_mat_vec_f_pair(ctx, mm0, mm1)) {
        return 0;
    }
    return j - i;
}

// Whether the Hadamard transform at node i_mm feeds a PTQ1_0 mat-mul (directly or through a reshape of it) within the
// next few nodes. ggml_cuda_try_fwht_q8 fuses those transforms with their q8_1 quantization by default, so the
// norm + transform fusions leave them to it.
static bool ggml_cuda_hadamard_feeds_ptq1(const ggml_cgraph * cgraph, int i_mm) {
    const ggml_tensor * mm = cgraph->nodes[i_mm];
    const ggml_tensor * alias = mm;
    for (int k = i_mm + 1; k < cgraph->n_nodes && k <= i_mm + 8; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (t->op == GGML_OP_RESHAPE && t->src[0] == mm) {
            alias = t;
            continue;
        }
        if (t->op == GGML_OP_MUL_MAT && (t->src[1] == mm || t->src[1] == alias) && t->src[0] &&
                t->src[0]->type == GGML_TYPE_PTQ1_0) {
            return true;
        }
    }
    return false;
}

// The norm + transform fusions below can also quantize the transformed rows when every use of the transform output
// (node i_mm, width n) is a PQ2_0 mat-vec on the mmvq path over the full row, with a q8_1 layout the fused kernel
// writes: AOS (1 column) or PQ2 (the multi-column path). The mat-vecs then find the rows through ctx.fwht_q8() and
// skip their quantize launch. GGML_CUDA_NORM_FWHT_Q8=0 disables it.
struct ggml_cuda_norm_fwht_q8_plan {
    ggml_cuda_fwht_q8_uses uses;
    ggml_cuda_q8_1_layout  layout = GGML_CUDA_Q8_1_AOS;
};

// srcs: the tensors the fused kernel reads (residual inputs or the norm input). The F32 output may exactly alias one of
// them (each block reads its row before writing it), but the q8_1 rows are packed: block r writes bytes that lie in
// earlier rows of the input, which other blocks may not have loaded yet. So with several rows any overlap keeps the F32
// output. A single row is safe: add_rms_norm_fwht_f32 runs one 1024-thread block per row, the norm needs the sum of
// squares of the whole row, so the block loads all of its input (a, b) before the block reduce's __syncthreads, and
// every q8_1 store comes after it. That holds with and without the add (the lookahead site) and for every Hadamard block
// width (only the butterfly depends on it). Decode thus keeps the q8_1 output when the allocator reuses an input buffer.
static bool ggml_cuda_norm_fwht_q8_prepare(const ggml_cgraph * cgraph, const int i_mm, const int64_t n,
                                           const ggml_tensor * const * srcs, const int n_srcs,
                                           ggml_cuda_norm_fwht_q8_plan & plan) {
    static const bool disabled = getenv("GGML_CUDA_NORM_FWHT_Q8") != nullptr && atoi(getenv("GGML_CUDA_NORM_FWHT_Q8")) == 0;
    if (disabled) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_TURING) {
        return false;
    }
    const ggml_tensor * mm = cgraph->nodes[i_mm];
    if (ggml_cuda_pq2_narrow_int8_enabled() && ggml_nelements(mm) >= 5 * n) {
        return false;
    }
    const char * m0 = (const char *) mm->data;
    const char * m1 = m0 + ggml_nbytes(mm);
    for (int k = 0; k < n_srcs && ggml_nelements(mm) > n; ++k) {
        const char * s0 = (const char *) srcs[k]->data;
        const char * s1 = s0 + ggml_nbytes(srcs[k]);
        if (m0 < s1 && s0 < m1) {
            return false;
        }
    }
    ggml_cuda_fwht_q8_uses & u = plan.uses;
    if (!ggml_cuda_fwht_q8_find_uses(cgraph, i_mm, cc, true, u) || u.wtype != GGML_TYPE_PQ2_0 || u.K != n ||
            u.K * u.ncols != ggml_nelements(mm) || GGML_PAD(u.K, MATRIX_ROW_PADDING) != u.K) {
        return false;
    }
    plan.layout = ggml_cuda_q8_1_layout_host(u.wtype, (int) u.ncols, false);
    return plan.layout == GGML_CUDA_Q8_1_AOS || plan.layout == GGML_CUDA_Q8_1_PQ2;
}

static void ggml_cuda_norm_fwht_q8_register(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
                                            const ggml_cuda_norm_fwht_q8_plan & plan) {
    ggml_cuda_fwht_q8 e;
    e.layout = plan.layout;
    e.ne0    = plan.uses.K;
    e.ncols  = plan.uses.ncols;
    e.data   = mm->data;
    for (int a = 0; a < plan.uses.n_aliases; ++a) {
        ctx.fwht_q8().set(plan.uses.aliases[a], e);
    }
}

struct ggml_cuda_tail_verify_entry {
    float * scratch; int tag;
    // q8_1 check of the attn_norm output (GGML_CUDA_TAIL_FUSION=3): the fused rows and a scratch for the reference quantize
    const void * q8_fused = nullptr; void * q8_ref = nullptr; const ggml_tensor * norm_out = nullptr;
    int64_t ne0_pad = 0; int64_t nrows = 0;
};
static std::map<const ggml_tensor *, ggml_cuda_tail_verify_entry> & ggml_cuda_tail_verify_pending() {
    static std::map<const ggml_tensor *, ggml_cuda_tail_verify_entry> m;
    return m;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }
    static const bool dual_rms_q8_enabled = [] {
        const char * env = getenv("GGML_CUDA_GB10_DUAL_RMS_Q8");
        return !env || std::atoi(env) != 0;
    }();

    ggml_tensor * node = cgraph->nodes[i];

    // Gemma 4 MoE layer tail: [RMS_NORM, MUL] [RMS_NORM, MUL, ADD] [RMS_NORM, MUL, ADD] MUL(layer scalar) -> one launch.
    // With GGML_CUDA_TAIL_FUSION=2 the next layer's attn_norm [RMS_NORM, MUL] joins the same launch; =3 also writes its q8_1 rows for the mat-vecs.
    // GGML_CUDA_TAIL_FUSION=1/2 enables; GGML_CUDA_TAIL_VERIFY=1 additionally checks it bit-for-bit against the unfused nodes.
    if (node->op == GGML_OP_RMS_NORM && i + 8 < cgraph->n_nodes) {
        static const int  tail_level  = getenv("GGML_CUDA_TAIL_FUSION") != nullptr ? atoi(getenv("GGML_CUDA_TAIL_FUSION")) : 0;
        static const bool tail_verify = getenv("GGML_CUDA_TAIL_VERIFY") != nullptr && atoi(getenv("GGML_CUDA_TAIL_VERIFY")) != 0;
        if (tail_level > 0) {
            ggml_tensor * const * n = cgraph->nodes + i;
            static const ggml_op ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD,
                                           GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL };
            bool ok = true;
            for (int q = 0; q < 9 && ok; ++q) ok = n[q]->op == ops[q] && n[q]->type == GGML_TYPE_F32;
            auto has = [](const ggml_tensor * t, const ggml_tensor * s) { return t->src[0] == s || t->src[1] == s; };
            auto other = [](const ggml_tensor * t, const ggml_tensor * s) { return t->src[0] == s ? t->src[1] : t->src[0]; };
            if (ok) {
                ok = has(n[1], n[0]) && has(n[3], n[2]) && has(n[4], n[1]) && has(n[4], n[3]) && n[5]->src[0] == n[4] &&
                     has(n[6], n[5]) && has(n[7], n[6]) && has(n[8], n[7]);
            }
            if (ok) {
                const int64_t nc = n[0]->src[0]->ne[0];
                const ggml_tensor * ws[3] = { other(n[1], n[0]), other(n[3], n[2]), other(n[6], n[5]) };
                const ggml_tensor * rs = other(n[7], n[6]);
                const ggml_tensor * ls = other(n[8], n[7]);
                float e0, e2, e5;
                memcpy(&e0, n[0]->op_params, 4); memcpy(&e2, n[2]->op_params, 4); memcpy(&e5, n[5]->op_params, 4);
                ok = nc <= 3*1024 && nc % 1 == 0 && e0 == e2 && e0 == e5 &&
                     n[0]->src[0]->ne[2] == 1 && n[0]->src[0]->ne[3] == 1 && n[2]->src[0]->ne[0] == nc &&
                     n[2]->src[0]->ne[1] == n[0]->src[0]->ne[1] && n[2]->src[0]->ne[2] == 1 && n[2]->src[0]->ne[3] == 1 &&
                     n[0]->src[0]->type == GGML_TYPE_F32 && n[2]->src[0]->type == GGML_TYPE_F32 &&
                     n[0]->src[0]->nb[0] == 4 && n[2]->src[0]->nb[0] == 4 &&
                     ggml_are_same_shape(n[4], n[0]) && ggml_are_same_shape(n[8], n[0]) && ggml_are_same_shape(rs, n[0]) &&
                     rs->type == GGML_TYPE_F32 && rs->nb[0] == 4 && ggml_is_contiguous(n[8]) &&
                     ls->type == GGML_TYPE_F32 && ggml_nelements(ls) == 1;
                for (int q = 0; q < 3 && ok; ++q) {
                    ok = ws[q]->type == GGML_TYPE_F32 && ggml_is_contiguous(ws[q]) && ws[q]->ne[0] == nc && ggml_nrows(ws[q]) == 1;
                }
                // the next layer's attn_norm: RMS_NORM(l_out) * weight, not followed by a sign MUL (that is the
                // norm+Hadamard fusion, which keeps its own path)
                int n_fused = 9;
                if (ok && tail_level >= 2 && i + 10 < cgraph->n_nodes && n[9]->op == ops[9] && n[10]->op == ops[10] &&
                        n[9]->src[0] == n[8] && has(n[10], n[9]) && n[9]->type == GGML_TYPE_F32 && n[10]->type == GGML_TYPE_F32 &&
                        strncmp(n[10]->name, "attn_norm", 9) == 0 && ggml_are_same_shape(n[10], n[8]) && ggml_is_contiguous(n[10])) {
                    const ggml_tensor * w4 = other(n[10], n[9]);
                    float e9;
                    memcpy(&e9, n[9]->op_params, 4);
                    bool ext = e9 == e0 && w4->type == GGML_TYPE_F32 && ggml_is_contiguous(w4) && w4->ne[0] == nc && ggml_nrows(w4) == 1;
                    for (int k = i + 11; ext && k < cgraph->n_nodes && k <= i + 26; ++k) {
                        ext = !(cgraph->nodes[k]->op == GGML_OP_MUL && has(cgraph->nodes[k], n[10]));
                    }
                    if (ext) n_fused = 11;
                }
                int outs[] = { i + 8, i + 10 };
                if (n_fused == 11 && !(ggml_can_fuse_subgraph(cgraph, i, 11, ops, outs, 2) &&
                                       ggml_cuda_check_fusion_memory_ranges(cgraph, i, 11, outs, 2, false, true))) {
                    n_fused = 9;
                }
                if (n_fused == 9) {
                    ok = ggml_can_fuse_subgraph(cgraph, i, 9, ops, outs, 1) && ggml_cuda_check_fusion_memory_ranges(cgraph, i, 9, outs, 1, false, true);
                }
                // the fused kernel reproduces either reduction order of the unfused norms, not a mix of them
                if (ok && n_fused == 11 && !ggml_cuda_tail_fused_exact(n, 11)) {
                    n_fused = 9;
                    ok = ggml_can_fuse_subgraph(cgraph, i, 9, ops, outs, 1) && ggml_cuda_check_fusion_memory_ranges(cgraph, i, 9, outs, 1, false, true);
                }
                ok = ok && ggml_cuda_tail_fused_exact(n, n_fused);
                // level 3: also write the q8_1 rows of the normalized activation that the mat-vecs of the next
                // attention block would quantize (all consumers are plain MUL_MATs with the AOS layout)
                bool with_q8 = false;
                int64_t ne0_pad = 0;
                if (ok && n_fused == 11 && tail_level >= 3) {
                    const int64_t nrows = n[0]->src[0]->ne[1];
                    ne0_pad = GGML_PAD(nc, MATRIX_ROW_PADDING);
                    int uses_found = 0;
                    with_q8 = nc % QK8_1 == 0 && ne0_pad <= 3*1024 && nrows <= 8 && n[10]->ne[2] == 1 && n[10]->ne[3] == 1;
                    for (int k = i + 11; with_q8 && k < cgraph->n_nodes && k <= i + 40; ++k) {
                        const ggml_tensor * t = cgraph->nodes[k];
                        for (int q = 0; with_q8 && q < GGML_MAX_SRC; ++q) {
                            if (t->src[q] != n[10]) continue;
                            ++uses_found;
                            with_q8 = t->op == GGML_OP_MUL_MAT && q == 1 && t->src[0]->ne[2] == 1 && t->src[0]->ne[3] == 1 &&
                                      ggml_cuda_q8_1_layout_host(t->src[0]->type, (int) nrows, false) == GGML_CUDA_Q8_1_AOS;
                        }
                    }
                    with_q8 = with_q8 && uses_found > 0 && uses_found == ggml_node_get_use_count(cgraph, i + 10);
                }
                if (ok) {
                    if (tail_verify) {
                        static float * scratch = nullptr; static int tag = 0;
                        const size_t cap = (size_t) 32*1024*1024;
                        if (!scratch) { CUDA_CHECK(cudaMalloc(&scratch, 4*cap)); }
                        const size_t need = ggml_nelements(n[8])*sizeof(float);
                        if (need <= cap && tag < 1000000) {
                            float * scratch2 = (float *) ((char *) scratch + cap);
                            char * q8_fused = (char *) scratch + 2*cap;
                            char * q8_ref   = (char *) scratch + 3*cap;
                            ggml_cuda_op_tail_fused(*cuda_ctx, n, n_fused, scratch, scratch2, with_q8 ? (block_q8_1 *) q8_fused : nullptr, (int) ne0_pad);
                            ggml_cuda_tail_verify_pending()[n[8]] = { scratch, tag };
                            if (n_fused == 11) {
                                ggml_cuda_tail_verify_entry e = { scratch2, 1000000 + tag };
                                if (with_q8) {
                                    e.q8_fused = q8_fused; e.q8_ref = q8_ref; e.norm_out = n[10]; e.ne0_pad = ne0_pad; e.nrows = n[0]->src[0]->ne[1];
                                }
                                ggml_cuda_tail_verify_pending()[n[10]] = e;
                            }
                            static int n_with_norm = 0, n_with_q8 = 0;
                            n_with_norm += n_fused == 11;
                            n_with_q8 += with_q8;
                            ++tag;
                            if ((tag & (tag - 1)) == 0) fprintf(stderr, "TAILVERIFY groups verified: %d, of which with the next attn_norm: %d, with q8_1: %d\n", tag, n_with_norm, n_with_q8);
                        }
                        return 0;
                    }
                    if (with_q8) {
                        const size_t q8_bytes = (size_t) (n[0]->src[0]->ne[1] * ne0_pad) * sizeof(block_q8_1) / QK8_1;
                        auto blk = std::make_unique<ggml_cuda_pool_alloc<char>>(cuda_ctx->pool(), q8_bytes);
                        char * q8 = blk->get();
                        cuda_ctx->fwht_q8().held.push_back(std::move(blk));
                        ggml_cuda_op_tail_fused(*cuda_ctx, n, n_fused, nullptr, nullptr, (block_q8_1 *) q8, (int) ne0_pad);
                        ggml_cuda_fwht_q8 e;
                        e.layout = GGML_CUDA_Q8_1_AOS;
                        e.ne0    = ne0_pad;
                        e.ncols  = n[0]->src[0]->ne[1];
                        e.data   = q8;
                        cuda_ctx->fwht_q8().set(n[10], e);
                    } else {
                        ggml_cuda_op_tail_fused(*cuda_ctx, n, n_fused, nullptr, nullptr, nullptr, 0);
                    }
                    return n_fused - 1;
                }
            }
        }
    }

    if (node->op == GGML_OP_CPY) {
        static const bool batched_cpy = [] {
            const char * env = getenv("GGML_CUDA_BATCHED_CPY");
            return !env || std::atoi(env) != 0;
        }();
        if (batched_cpy) {
            if (const int n = ggml_cuda_try_batched_cpy(cuda_ctx, cgraph, i)) {
                return n;
            }
        }
    }

    // GB10 prefill: apply SWIGLU while quantizing the activation consumed by the
    // following low-bit down projection. This avoids materializing and rereading
    // the GLU output and removes one launch per transformer block.
    if (node->op == GGML_OP_GLU && i + 1 < cgraph->n_nodes) {
        ggml_tensor * mm = cgraph->nodes[i + 1];
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const ggml_op ops[] = { GGML_OP_GLU, GGML_OP_MUL_MAT };
        const int out_nodes[] = { i + 1 };
        if (cc == GGML_CUDA_CC_DGX_SPARK && mm->op == GGML_OP_MUL_MAT && mm->src[1] == node &&
                node->src[0] && node->src[1] && node->src[0]->type == GGML_TYPE_F32 &&
                node->src[1]->type == GGML_TYPE_F32 && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU &&
                ggml_get_op_params_i32(node, 1) == 0 && ggml_are_same_shape(node->src[0], node->src[1]) &&
                ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node->src[1]) &&
                mm->src[1]->ne[1] >= 32 &&
                ggml_cuda_should_use_mmq(mm->src[0]->type, cc, mm->src[1]->ne[1], 0) &&
                ggml_can_fuse_subgraph(cgraph, i, 2, ops, out_nodes, 1) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, out_nodes, 1)) {
            ggml_cuda_mul_mat_q(*cuda_ctx, mm->src[0], node->src[1], nullptr, mm, node->src[0]);
            return 1;
        }
    }

    // Post-attention normalization feeds adjacent gate and up projections.
    // Quantize the normalized residual once and reuse it for both MMQs.
    if (node->op == GGML_OP_ADD && i + 4 < cgraph->n_nodes) {
        ggml_tensor * rms_norm = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        ggml_tensor * mm_a = cgraph->nodes[i + 3];
        ggml_tensor * mm_b = cgraph->nodes[i + 4];
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const ggml_op ops[] = { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT };
        const int out_nodes[] = { i, i + 3, i + 4 };
        if (dual_rms_q8_enabled && cc == GGML_CUDA_CC_DGX_SPARK && rms_norm->op == GGML_OP_RMS_NORM &&
                mul->op == GGML_OP_MUL && mm_a->op == GGML_OP_MUL_MAT && mm_b->op == GGML_OP_MUL_MAT &&
                mm_a->src[1] == mul && mm_b->src[1] == mul &&
                (mul->src[0] == rms_norm || mul->src[1] == rms_norm) && rms_norm->src[0] == node &&
                node->src[0] && node->src[1] && node->type == GGML_TYPE_F32 &&
                node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32 &&
                rms_norm->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(node->src[0], node->src[1]) && ggml_are_same_shape(node, rms_norm) &&
                ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node->src[1]) &&
                ggml_is_contiguous(node) && ggml_is_contiguous(rms_norm) && ggml_is_contiguous(mul) &&
                mul->ne[1] >= 32 && mm_a->src[0]->type == mm_b->src[0]->type &&
                (mm_a->src[0]->type == GGML_TYPE_Q1_0 ||
                 mm_a->src[0]->type == GGML_TYPE_Q2_0 ||
                 mm_a->src[0]->type == GGML_TYPE_PQ2_0) &&
                ggml_cuda_should_use_mmq(mm_a->src[0]->type, cc, mul->ne[1], 0) &&
                ggml_cuda_should_use_mmq(mm_b->src[0]->type, cc, mul->ne[1], 0) &&
                ggml_can_fuse_subgraph(cgraph, i, 5, ops, out_nodes, 3) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 5, out_nodes, 3)) {
            const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
            if (weight && weight->type == GGML_TYPE_F32 && ggml_is_contiguous(weight) &&
                    weight->ne[0] == node->ne[0] && ggml_nrows(weight) == 1) {
                ggml_cuda_pool_alloc<float> row_scale(cuda_ctx->pool(), ggml_nrows(node));
                ggml_cuda_op_add_rms_norm_scale_fused(*cuda_ctx, node, rms_norm, row_scale.get());
                ggml_cuda_mul_mat_q_fused_two(*cuda_ctx, mm_a->src[0], mm_b->src[0], node,
                                               mm_a, mm_b, weight, row_scale.get());
                return 4;
            }
        }
    }

    // Residual ADD -> RMS_NORM -> MUL(weight) -> MUL(Hadamard signs) -> RESHAPE -> MUL_MAT(Hadamard hint):
    // the input of every Bonsai 2 FFN. One launch instead of three (add, norm, signed transform), bit-identical
    // to them. The residual sum and the rotated activation are the only outputs; the normalized row is not
    // needed anywhere else (ggml_can_fuse_subgraph checks). GGML_CUDA_NORM_FWHT_FUSION=0 disables it.
    if (node->op == GGML_OP_ADD && i + 5 < cgraph->n_nodes) {
        static const bool arf_disabled = getenv("GGML_CUDA_NORM_FWHT_FUSION") != nullptr &&
                                         atoi(getenv("GGML_CUDA_NORM_FWHT_FUSION")) == 0;
        ggml_tensor * rms_norm = cgraph->nodes[i + 1];
        ggml_tensor * mul_w    = cgraph->nodes[i + 2];
        ggml_tensor * mul_s    = cgraph->nodes[i + 3];
        ggml_tensor * reshape  = cgraph->nodes[i + 4];
        ggml_tensor * mm       = cgraph->nodes[i + 5];
        const ggml_op ops[]       = { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT };
        const int     out_nodes[] = { i, i + 5 };
        if (!arf_disabled && GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) &&
                rms_norm->op == GGML_OP_RMS_NORM && mul_w->op == GGML_OP_MUL && mul_s->op == GGML_OP_MUL &&
                reshape->op == GGML_OP_RESHAPE && mm->op == GGML_OP_MUL_MAT &&
                rms_norm->src[0] == node && (mul_w->src[0] == rms_norm || mul_w->src[1] == rms_norm) &&
                mul_s->src[0] == mul_w && reshape->src[0] == mul_s && mm->src[1] == reshape &&
                ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            const ggml_tensor * a      = node->src[0];
            const ggml_tensor * b      = node->src[1];
            const ggml_tensor * weight = mul_w->src[0] == rms_norm ? mul_w->src[1] : mul_w->src[0];
            const ggml_tensor * signs  = mul_s->src[1];
            const int64_t ncols = node->ne[0];
            if (a && b && weight && signs &&
                    a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                    rms_norm->type == GGML_TYPE_F32 && mul_w->type == GGML_TYPE_F32 && mul_s->type == GGML_TYPE_F32 &&
                    weight->type == GGML_TYPE_F32 && signs->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
                    ggml_are_same_shape(a, b) && ggml_are_same_shape(a, node) && ggml_are_same_shape(node, rms_norm) &&
                    ggml_are_same_shape(rms_norm, mul_w) && ggml_are_same_shape(mul_w, mul_s) &&
                    ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(node) &&
                    ggml_is_contiguous(weight) && ggml_is_contiguous(signs) && ggml_is_contiguous(mm) &&
                    weight->ne[0] == ncols && ggml_nrows(weight) == 1 &&
                    signs->ne[0] == ncols && ggml_nrows(signs) == 1 &&
                    mm->ne[0] == mm->src[0]->ne[0] && ggml_nelements(mm) == ggml_nelements(node) &&
                    ggml_cuda_add_rms_norm_fwht_supported((int) ncols, (int) mm->src[0]->ne[0]) &&
                    !ggml_cuda_hadamard_feeds_ptq1(cgraph, i + 5) &&
                    weight->data != node->data && signs->data != node->data &&
                    weight->data != mm->data && signs->data != mm->data &&
                    ggml_can_fuse_subgraph(cgraph, i, 6, ops, out_nodes, 2) &&
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, out_nodes, 2, false, true)) {
                // Narrow INT8 (N=5..128) fast path: fuse ADD + RMS_NORM + MUL(w) + MUL(signs) + FWHT(1024) directly into 2-nibble B_i8!
                if (ggml_cuda_pq2_narrow_int8_enabled() && (int) mm->src[0]->ne[0] == 1024 &&
                        (ncols % 1024) == 0 && (ncols >> 10) <= 24 && node->ne[1] >= 5 && node->ne[1] <= pq2_narrow_i8_max_n() &&
                        node->ne[2] == 1 && node->ne[3] == 1 && (mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                    bool fwht_only_pq2 = true;
                    int n_pq2 = 0;
                    for (int jj = i + 6; jj < cgraph->n_nodes && fwht_only_pq2; ++jj) {
                        const ggml_tensor * cn = cgraph->nodes[jj];
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            if (cn->src[s] == mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == mm)) {
                                if (cn->op == GGML_OP_RESHAPE) { continue; }
                                if (s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0 &&
                                        cn->src[0]->ne[1] <= (node->ne[1] >= 256 ? 65536 : 262144)) {
                                    ++n_pq2;
                                } else {
                                    fwht_only_pq2 = false; break;
                                }
                            }
                        }
                    }
                    if (fwht_only_pq2 && n_pq2 >= 1) {
                        cudaStream_t stream = cuda_ctx->stream();
                        const int64_t N = node->ne[1], N_pad = pq2_narrow_i8_n_pad(N);
                        auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
                        if (pq2_narrow_i8_alloc_bufs(sc, N_pad, ncols, stream)) {
                            float eps = 0.0f; memcpy(&eps, rms_norm->op_params, sizeof(float));
                            rms_mul_signs_fwht1024_to_narrow_i8_kernel<<<(unsigned) N_pad, (unsigned) ((ncols >> 10) * 32), 0, stream>>>(
                                (const float *) a->data, (const float *) b->data, (float *) node->data,
                                (const float *) weight->data, eps, (const float *) signs->data, nullptr,
                                sc.B_i8, sc.B_sum, sc.B_scales, (int) ncols, (int) N, (int) N_pad, ncols);
                            CUDA_CHECK(cudaGetLastError());
                            sc.src1_data = mm->data; sc.src1_nbytes = ggml_nbytes(mm);
                            sc.N = N; sc.K = ncols; sc.s11 = ncols; sc.valid = true; ++sc.gen;
                            return 5;
                        }
                    }
                }
                // Prefill-only INT4 fast path: fuse ADD + RMS_NORM + MUL(w) + MUL(signs) + FWHT(1024) directly into B_s4!
                if (ggml_cuda_pq2_prefill_int4_enabled() && (int) mm->src[0]->ne[0] == 1024 &&
                        (ncols % 1024) == 0 && (ncols >> 10) <= 24 && node->ne[1] >= 256 &&
                        node->ne[2] == 1 && node->ne[3] == 1 && (mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                    bool fwht_only_pq2 = true;
                    int n_pq2 = 0;
                    for (int jj = i + 6; jj < cgraph->n_nodes && fwht_only_pq2; ++jj) {
                        const ggml_tensor * cn = cgraph->nodes[jj];
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            if (cn->src[s] == mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == mm)) {
                                if (cn->op == GGML_OP_RESHAPE) { continue; }
                                if (s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0 && cn->src[0]->ne[1] <= 65536) {
                                    ++n_pq2;
                                } else {
                                    fwht_only_pq2 = false; break;
                                }
                            }
                        }
                    }
                    if (fwht_only_pq2 && n_pq2 >= 1) {
                        cudaStream_t stream = cuda_ctx->stream();
                        const int64_t N = node->ne[1], N_pad = ((N + 127) / 128) * 128;
                        auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
                        const size_t need_b = (size_t) N_pad * (size_t) (ncols >> 3) * sizeof(uint32_t);
                        const size_t need_s = (size_t) N_pad * sizeof(float);
                        if (sc.cap_b < need_b)  { if (sc.B_i4)     CUDA_CHECK(cudaFree(sc.B_i4));     CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));     sc.cap_b = need_b; }
                        if (sc.cap_sc < need_s) { if (sc.B_scales) CUDA_CHECK(cudaFree(sc.B_scales)); CUDA_CHECK(cudaMalloc(&sc.B_scales, need_s)); sc.cap_sc = need_s; }
                        float eps = 0.0f; memcpy(&eps, rms_norm->op_params, sizeof(float));
                        rms_mul_signs_fwht1024_to_s4_kernel<<<(unsigned) N_pad, (unsigned) ((ncols >> 10) * 32), 0, stream>>>(
                            (const float *) a->data, (const float *) b->data, (float *) node->data,
                            (const float *) weight->data, eps, (const float *) signs->data, nullptr, nullptr,
                            sc.B_i4, sc.B_scales, (int) ncols, (int) N, (int) N_pad, ncols, ggml_cuda_pq2_prefill_int4_clip());
                        CUDA_CHECK(cudaGetLastError());
                        sc.src1_data = mm->data; sc.src1_nbytes = ggml_nbytes(mm);
                        sc.N = N; sc.K = ncols; sc.s11 = ncols; sc.valid = true;
                        return 5;
                    }
                }
                ggml_cuda_norm_fwht_q8_plan plan;
                const ggml_tensor * q8_srcs[2] = { a, b };
                if (ggml_cuda_norm_fwht_q8_prepare(cgraph, i + 5, ncols, q8_srcs, 2, plan)) {
                    ggml_cuda_op_add_rms_norm_fwht_fused(*cuda_ctx, node, rms_norm, weight, signs, nullptr, mm, mm->data, plan.layout);
                    ggml_cuda_norm_fwht_q8_register(*cuda_ctx, mm, plan);
                } else {
                    ggml_cuda_op_add_rms_norm_fwht_fused(*cuda_ctx, node, rms_norm, weight, signs, nullptr, mm);
                }
                return 5;
            }
        }
    }

    // RMS_NORM -> MUL(weight) ... MUL(Hadamard signs) -> RESHAPE -> MUL_MAT(Hadamard hint), with unrelated nodes in
    // between: the attention input at the start of every Bonsai 2 layer, where the recurrent conv-state reads sit
    // between the norm and the transform, and the normalized row also feeds ssm_alpha/ssm_beta. The norm, the
    // normalized row and the transform are computed here in one launch; the sign MUL, RESHAPE and transform nodes are
    // recorded as precomputed and skipped when the loop reaches them. Bit-identical to rms_norm_f32<1024> with the
    // multiply followed by the signed transform. Only taken when none of the nodes in between touches the
    // transform's output buffer (it is written early) and no node stream concurrency is active.
    if (node->op == GGML_OP_RMS_NORM && i + 4 < cgraph->n_nodes &&
            cuda_ctx->stream_context().concurrent_events.empty()) {
        static const bool rf_disabled = getenv("GGML_CUDA_NORM_FWHT_FUSION") != nullptr &&
                                        atoi(getenv("GGML_CUDA_NORM_FWHT_FUSION")) == 0;
        ggml_tensor * mul_w = cgraph->nodes[i + 1];
        const ggml_tensor * x = node->src[0];
        const int64_t ncols = node->ne[0];
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (!rf_disabled && GGML_CUDA_CC_IS_NVIDIA(cc) && mul_w->op == GGML_OP_MUL &&
                (mul_w->src[0] == node || mul_w->src[1] == node) && x && x->type == GGML_TYPE_F32 &&
                node->type == GGML_TYPE_F32 && mul_w->type == GGML_TYPE_F32 && ggml_is_contiguous(x) &&
                ggml_is_contiguous(node) && ggml_is_contiguous(mul_w) && ggml_are_same_shape(x, node) &&
                ggml_are_same_shape(node, mul_w) && ggml_cuda_add_rms_norm_fwht_supported((int) ncols, 512) &&
                ggml_node_get_use_count(cgraph, i) == 1) {
            const ggml_tensor * weight = mul_w->src[0] == node ? mul_w->src[1] : mul_w->src[0];
            // the sign MUL of this normalized row, a few nodes later
            int j = -1;
            for (int k = i + 2; k + 2 < cgraph->n_nodes && k <= i + 16; ++k) {
                const ggml_tensor * t = cgraph->nodes[k];
                if (t->op == GGML_OP_MUL && t->src[0] == mul_w) {
                    j = k;
                    break;
                }
            }
            ggml_tensor * mul_s   = j >= 0 ? cgraph->nodes[j]     : nullptr;
            ggml_tensor * reshape = j >= 0 ? cgraph->nodes[j + 1] : nullptr;
            ggml_tensor * mm      = j >= 0 ? cgraph->nodes[j + 2] : nullptr;
            const ggml_tensor * signs = mul_s ? mul_s->src[1] : nullptr;
            // weight and signs are read now: they must be leaves (model tensors), not results of the nodes in between
            bool ok = mul_s && weight && signs && weight->op == GGML_OP_NONE && signs->op == GGML_OP_NONE &&
                reshape->op == GGML_OP_RESHAPE && reshape->src[0] == mul_s &&
                mm->op == GGML_OP_MUL_MAT && mm->src[1] == reshape &&
                ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                mul_s->type == GGML_TYPE_F32 && ggml_are_same_shape(mul_s, mul_w) &&
                weight->type == GGML_TYPE_F32 && ggml_is_contiguous(weight) && weight->ne[0] == ncols && ggml_nrows(weight) == 1 &&
                signs->type == GGML_TYPE_F32 && ggml_is_contiguous(signs) && signs->ne[0] == ncols && ggml_nrows(signs) == 1 &&
                mm->type == GGML_TYPE_F32 && ggml_is_contiguous(mm) && mm->ne[0] == mm->src[0]->ne[0] &&
                ggml_cuda_add_rms_norm_fwht_supported((int) ncols, (int) mm->src[0]->ne[0]) &&
                ggml_nelements(mm) == ggml_nelements(node) && !(mm->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                ggml_node_get_use_count(cgraph, j) == 1 && ggml_node_get_use_count(cgraph, j + 1) == 1 &&
                !ggml_cuda_hadamard_feeds_ptq1(cgraph, j + 2);
            auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
                if (!a || !b || !a->data || !b->data) {
                    return false;
                }
                const char * a0 = (const char *) a->data;
                const char * b0 = (const char *) b->data;
                return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
            };
            // mm is written now instead of at j + 2: it must not share memory with the norm's inputs and outputs
            // or with anything the nodes in between read or write
            ok = ok && !overlaps(mm, x) && !overlaps(mm, mul_w) && !overlaps(mm, weight) && !overlaps(mm, signs);
            for (int k = i + 2; ok && k < j; ++k) {
                const ggml_tensor * t = cgraph->nodes[k];
                if (overlaps(mm, t)) {
                    ok = false;
                }
                for (int s = 0; ok && s < GGML_MAX_SRC; ++s) {
                    if (t->src[s] == mul_w || t->src[s] == node) {
                        continue; // reading the normalized row (alpha/beta) is fine, it is written here too
                    }
                    if (overlaps(mm, t->src[s])) {
                        ok = false;
                    }
                }
            }
            if (ok) {
                bool did_s4 = false;
                if (ggml_cuda_pq2_prefill_int4_enabled() && (int) mm->src[0]->ne[0] == 1024 &&
                        (ncols % 1024) == 0 && (ncols >> 10) <= 24 && x->ne[1] >= 256 &&
                        x->ne[2] == 1 && x->ne[3] == 1) {
                    bool safe_s4 = true;
                    for (int k = i + 2; k < j && safe_s4; ++k) {
                        const ggml_tensor * t = cgraph->nodes[k];
                        if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[0]->type == GGML_TYPE_PQ2_0) {
                            safe_s4 = false;
                        }
                    }
                    int n_pq2 = 0;
                    for (int jj = j + 3; jj < cgraph->n_nodes && safe_s4; ++jj) {
                        const ggml_tensor * cn = cgraph->nodes[jj];
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            if (cn->src[s] == mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == mm)) {
                                if (cn->op == GGML_OP_RESHAPE) { continue; }
                                if (s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0 && cn->src[0]->ne[1] <= 65536) {
                                    ++n_pq2;
                                } else {
                                    safe_s4 = false; break;
                                }
                            }
                        }
                    }
                    if (safe_s4 && n_pq2 >= 1) {
                        cudaStream_t stream = cuda_ctx->stream();
                        const int64_t N = x->ne[1], N_pad = ((N + 127) / 128) * 128;
                        auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
                        const size_t need_b = (size_t) N_pad * (size_t) (ncols >> 3) * sizeof(uint32_t);
                        const size_t need_s = (size_t) N_pad * sizeof(float);
                        if (sc.cap_b < need_b)  { if (sc.B_i4)     CUDA_CHECK(cudaFree(sc.B_i4));     CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));     sc.cap_b = need_b; }
                        if (sc.cap_sc < need_s) { if (sc.B_scales) CUDA_CHECK(cudaFree(sc.B_scales)); CUDA_CHECK(cudaMalloc(&sc.B_scales, need_s)); sc.cap_sc = need_s; }
                        const bool need_mw = (ggml_node_get_use_count(cgraph, i + 1) > 1) || ((mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) != 0);
                        bool mw_only_small_f32_mm = need_mw && ((mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
                        if (mw_only_small_f32_mm) {
                            for (int jj = i + 2; jj < cgraph->n_nodes && mw_only_small_f32_mm; ++jj) {
                                const ggml_tensor * cn = cgraph->nodes[jj];
                                if (cn == mul_s) { continue; }
                                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                                    if (cn->src[s] == mul_w) {
                                        if (!(s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] &&
                                                cn->src[0]->type == GGML_TYPE_F32 && cn->src[0]->ne[1] <= 64 &&
                                                strstr(cn->src[0]->name, "lora") == nullptr &&
                                                ggml_is_contiguous(cn->src[0]) && ggml_is_contiguous(cn))) {
                                            mw_only_small_f32_mm = false; break;
                                        }
                                    }
                                }
                            }
                        }
                        half * mw_f16_ptr = nullptr;
                        if (mw_only_small_f32_mm) {
                            const size_t need_mw_f16 = (size_t) N * (size_t) ncols * sizeof(half);
                            if (sc.cap_mw_f16 < need_mw_f16) {
                                if (sc.mul_w_f16) { CUDA_CHECK(cudaFree(sc.mul_w_f16)); }
                                CUDA_CHECK(cudaMalloc(&sc.mul_w_f16, need_mw_f16));
                                sc.cap_mw_f16 = need_mw_f16;
                            }
                            mw_f16_ptr = sc.mul_w_f16;
                        }
                        float eps = 0.0f; memcpy(&eps, node->op_params, sizeof(float));
                        rms_mul_signs_fwht1024_to_s4_kernel<<<(unsigned) N_pad, (unsigned) ((ncols >> 10) * 32), 0, stream>>>(
                            (const float *) x->data, nullptr, nullptr,
                            (const float *) weight->data, eps, (const float *) signs->data,
                            (need_mw && !mw_f16_ptr) ? (float *) mul_w->data : nullptr, mw_f16_ptr,
                            sc.B_i4, sc.B_scales, (int) ncols, (int) N, (int) N_pad, ncols, ggml_cuda_pq2_prefill_int4_clip());
                        CUDA_CHECK(cudaGetLastError());
                        sc.src1_data = mm->data; sc.src1_nbytes = ggml_nbytes(mm);
                        sc.N = N; sc.K = ncols; sc.s11 = ncols; sc.valid = true;
                        if (mw_f16_ptr) {
                            sc.mw_data = mul_w->data; sc.mw_nbytes = ggml_nbytes(mul_w);
                            sc.mw_N = N; sc.mw_K = ncols; sc.mw_valid = true;
                        }
                        did_s4 = true;
                    }
                }
                if (!did_s4 && ggml_cuda_pq2_narrow_int8_enabled() && (int) mm->src[0]->ne[0] == 1024 &&
                        (ncols % 1024) == 0 && (ncols >> 10) <= 24 && x->ne[1] >= 5 && x->ne[1] <= pq2_narrow_i8_max_n() &&
                        x->ne[2] == 1 && x->ne[3] == 1) {
                    bool safe_ni8 = true;
                    for (int k = i + 2; k < j && safe_ni8; ++k) {
                        const ggml_tensor * t = cgraph->nodes[k];
                        if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[0]->type == GGML_TYPE_PQ2_0) {
                            safe_ni8 = false;
                        }
                    }
                    int n_pq2 = 0;
                    for (int jj = j + 3; jj < cgraph->n_nodes && safe_ni8; ++jj) {
                        const ggml_tensor * cn = cgraph->nodes[jj];
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            if (cn->src[s] == mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == mm)) {
                                if (cn->op == GGML_OP_RESHAPE) { continue; }
                                if (s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0 &&
                                        cn->src[0]->ne[1] <= (x->ne[1] >= 256 ? 65536 : 262144)) {
                                    ++n_pq2;
                                } else {
                                    safe_ni8 = false; break;
                                }
                            }
                        }
                    }
                    if (safe_ni8 && n_pq2 >= 1) {
                        cudaStream_t stream = cuda_ctx->stream();
                        const int64_t N = x->ne[1], N_pad = pq2_narrow_i8_n_pad(N);
                        auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
                        if (pq2_narrow_i8_alloc_bufs(sc, N_pad, ncols, stream)) {
                            const bool need_mw = (ggml_node_get_use_count(cgraph, i + 1) > 1) || ((mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) != 0);
                            float eps = 0.0f; memcpy(&eps, node->op_params, sizeof(float));
                            rms_mul_signs_fwht1024_to_narrow_i8_kernel<<<(unsigned) N_pad, (unsigned) ((ncols >> 10) * 32), 0, stream>>>(
                                (const float *) x->data, nullptr, nullptr,
                                (const float *) weight->data, eps, (const float *) signs->data,
                                need_mw ? (float *) mul_w->data : nullptr,
                                sc.B_i8, sc.B_sum, sc.B_scales, (int) ncols, (int) N, (int) N_pad, ncols);
                            CUDA_CHECK(cudaGetLastError());
                            sc.src1_data = mm->data; sc.src1_nbytes = ggml_nbytes(mm);
                            sc.N = N; sc.K = ncols; sc.s11 = ncols; sc.valid = true; ++sc.gen;
                            did_s4 = true;
                        }
                    }
                }
                if (!did_s4) {
                    ggml_cuda_norm_fwht_q8_plan plan;
                    const ggml_tensor * q8_srcs[1] = { x };
                    if (ggml_cuda_norm_fwht_q8_prepare(cgraph, j + 2, ncols, q8_srcs, 1, plan)) {
                        ggml_cuda_op_add_rms_norm_fwht_fused(*cuda_ctx, nullptr, node, weight, signs, mul_w, mm, mm->data, plan.layout);
                        ggml_cuda_norm_fwht_q8_register(*cuda_ctx, mm, plan);
                    } else {
                        ggml_cuda_op_add_rms_norm_fwht_fused(*cuda_ctx, nullptr, node, weight, signs, mul_w, mm);
                    }
                }
                
                auto & pre = cuda_ctx->fwht_q8().precomputed;
                pre.insert(mul_s);
                pre.insert(reshape);
                pre.insert(mm);
                return 1;
            }
        }
    }

    // Preserve the residual sum for later graph consumers while normalizing it
    // in the same launch. This removes a full read of the residual tensor.
    //
    // add_rms_norm_f32 uses no architecture-specific instructions, so this runs
    // on every NVIDIA device rather than GB10 only. It is elementwise within a
    // row -- each thread reads a[col] and b[col] before writing sum[col], and
    // block_reduce synchronizes the block before dst[col] is written -- so the
    // in-place residual add and the norm output reusing an input buffer are
    // both safe, hence allow_exact_alias below.
    if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
        ggml_tensor * rms_norm = cgraph->nodes[i + 1];
        ggml_tensor * mul = cgraph->nodes[i + 2];
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const ggml_op ops[] = { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
        const int out_nodes[] = { i, i + 2 };
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && rms_norm->op == GGML_OP_RMS_NORM &&
                mul->op == GGML_OP_MUL && (mul->src[0] == rms_norm || mul->src[1] == rms_norm) &&
                rms_norm->src[0] == node && node->src[0] && node->src[1] &&
                node->type == GGML_TYPE_F32 && node->src[0]->type == GGML_TYPE_F32 &&
                node->src[1]->type == GGML_TYPE_F32 && rms_norm->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(node->src[0], node->src[1]) &&
                ggml_are_same_shape(node, rms_norm) && ggml_is_contiguous(node->src[0]) &&
                ggml_is_contiguous(node->src[1]) && ggml_is_contiguous(node) &&
                ggml_is_contiguous(rms_norm) && ggml_is_contiguous(mul) &&
                ggml_can_fuse_subgraph(cgraph, i, 3, ops, out_nodes, 2) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 3, out_nodes, 2, false, true)) {
            const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
            if (weight && weight->type == GGML_TYPE_F32 && ggml_is_contiguous(weight) &&
                    weight->ne[0] == node->ne[0] && ggml_nrows(weight) == 1 &&
                    weight->data != node->data && weight->data != rms_norm->data &&
                    weight->data != mul->data) {
                ggml_cuda_op_add_rms_norm_fused(*cuda_ctx, node, rms_norm, mul);
                return 2;
            }
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    // The fork's exact MoE weighted reduce (#107) is tried first: it keeps the serial expert-add order, so outputs
    // match the unfused graph bit for bit. Shapes it does not take fall through to upstream's reduction below.
    // MoE router weighting + expert reduction  [kill switch: GGML_CUDA_MOE_REDUCE_DISABLE=1]
    static bool disable_moe_reduce = getenv("GGML_CUDA_MOE_REDUCE_DISABLE") != nullptr &&
                                     std::atoi(getenv("GGML_CUDA_MOE_REDUCE_DISABLE"));

    if (!disable_moe_reduce && node->op == GGML_OP_MUL && node->ne[3] == 1 &&
        node->ne[1] >= 2 && node->ne[1] <= GGML_CUDA_MOE_REDUCE_MAX_EXPERTS) {

        const int n_exp   = (int) node->ne[1];
        const int n_nodes = 2*n_exp;  // MUL + n_exp views + (n_exp - 1) adds

        std::vector<ggml_op> ops;
        ops.reserve(n_nodes);
        ops.push_back(GGML_OP_MUL);
        ops.insert(ops.end(), n_exp,     GGML_OP_VIEW);
        ops.insert(ops.end(), n_exp - 1, GGML_OP_ADD);

        const int out_node = i + n_nodes - 1;

        bool stage_weights = false;

        if (ggml_can_fuse_subgraph(cgraph, i, n_nodes, ops.data(), &out_node, 1) &&
            ggml_cuda_should_fuse_moe_weighted_reduce(cgraph, i, n_exp, stage_weights)) {

            static bool debug_moe_reduce = getenv("GGML_CUDA_MOE_REDUCE_DEBUG") != nullptr;
            if (debug_moe_reduce) {
                GGML_LOG_INFO("%s: fused moe weighted reduce: n_expert_used=%d ne0=%d n_tokens=%d staged=%d\n",
                              __func__, n_exp, (int) node->ne[0], (int) node->ne[2], (int) stage_weights);
            }

            ggml_cuda_op_moe_weighted_reduce(*cuda_ctx, node->src[0], node->src[1], cgraph->nodes[out_node],
                                             n_exp, stage_weights);
            return n_nodes - 1;
        }
    }

    // GGML_CUDA_MOE_EP_FUSION=1: fold the per-expert scale gather (REPEAT + GET_ROWS of down_exps_s by the selected ids)
    // into the weighted reduction kernel
    if (node->op == GGML_OP_REPEAT && ggml_cuda_moe_ep_fusion_enabled()) {
        ggml_cuda_moe_weighted_reduction_match match;
        const ggml_tensor * table = nullptr;
        const ggml_tensor * ids   = nullptr;
        if (ggml_cuda_match_moe_scale_gather(cgraph, i, match, table, ids)) {
            const int count      = match.node_count + 2;
            const int output_idx = i + count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, count, &output_idx, 1)) {
                // GGML_CUDA_MOE_EP_VERIFY=1: run the fused kernel into scratch, run the unfused nodes, compare bit-for-bit at the output
                if (ggml_cuda_moe_ep_verify_mode() != 0) {
                    static float * scratch = nullptr; static int tag = 0;
                    if (!scratch) { CUDA_CHECK(cudaMalloc(&scratch, (size_t) 32*1024*1024)); }
                    const size_t need = ggml_nelements(match.dst)*sizeof(float);
                    if (need <= (size_t) 32*1024*1024 && tag < 1000000) {
                        ggml_tensor tmp = *match.dst;
                        tmp.data = scratch;
                        ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, match.experts, nullptr, match.weights, &tmp, table, ids);
                        ggml_cuda_tail_verify_pending()[match.dst] = { scratch, tag++ };
                    }
                    return 0;
                }
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, nullptr, match.weights, match.dst, table, ids);
                return count - 1;
            }
        }
    }

    if (node->op == GGML_OP_MUL && !(ggml_cuda_moe_ep_verify_mode() == 2 && ggml_cuda_moe_ep_fusion_enabled())) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                if (!ggml_cuda_tail_verify_pending().empty()) {
                    auto it = ggml_cuda_tail_verify_pending().find(match.dst);
                    if (it != ggml_cuda_tail_verify_pending().end()) {
                        ggml_cuda_op_tail_verify_cmp(*cuda_ctx, match.dst, it->second.scratch, it->second.tag);
                        static int nl = 0;
                        if ((++nl & (nl - 1)) == 0) fprintf(stderr, "TAILVERIFY compare launches captured/issued: %d\n", nl);
                    }
                }
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_GATED_DELTA_NET_REPLAY) {
        ggml_cuda_gdn_replay_fused fused_rpl;
        const int nodes_to_skip = ggml_cuda_try_gdn_replay_fusion(cgraph, i, fused_rpl);
        if (nodes_to_skip > 0) {
            ggml_cuda_op_gated_delta_net_replay(*cuda_ctx, node, &fused_rpl);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                ops.push_back(GGML_OP_RESHAPE);
                if (args.prob_bias) {
                    ops.push_back(GGML_OP_ADD);
                    bias = cgraph->nodes[i + ops.size() - 1]->src[1];
                }
                if (args.n_groups > 0) {
                    for (const auto & p : topk_moe_group_pattern) {
                        ops.push_back(p.op);
                    }
                }
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                out_nodes[0] = i + ops.size() - 2;
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    // Hadamard sign flip + reshape + FWHT-hint matmul: multiply the sign
    // vector during the transform's load instead of a separate full pass
    if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { i + 2 })) {
        const ggml_tensor * mul     = cgraph->nodes[i];
        const ggml_tensor * reshape = cgraph->nodes[i + 1];
        ggml_tensor *       mm      = cgraph->nodes[i + 2];

        const ggml_tensor * x     = mul->src[0];
        const ggml_tensor * signs = mul->src[1];

        const bool pattern_ok = ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
            mm->src[1] == reshape && reshape->src[0] == mul &&
            signs->ne[1] == 1 && signs->ne[2] == 1 && signs->ne[3] == 1 &&
            signs->type == GGML_TYPE_F32 &&
            (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16) &&
            // ggml_mul keeps src0's type, so an F16 x gives an F16 mul
            mul->type == x->type &&
            ggml_is_contiguous(x) && ggml_is_contiguous(signs) &&
            signs->ne[0] == x->ne[0] && signs->ne[0] % mm->src[0]->ne[0] == 0;

        if (pattern_ok && ggml_cuda_op_fwht_signed(*cuda_ctx, x, signs, mm)) {
            return 2;
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (op == GGML_OP_MUL_MAT && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                    ggml_get_op_params_i32(glu, 1) == 0 && gate->src[1] == up->src[1]) {
                // Check for SwiGLU -> MUL(signs) -> RESHAPE -> MUL_MAT(HADAMARD) -> [RESHAPE] -> PQ2_0 ffn_down!
                const ggml_tensor * fwht_signs = nullptr;
                const ggml_tensor * fwht_out   = nullptr;
                int fwht_H = 0;
                int extra_fused = 0;
                if (i + 5 < cgraph->n_nodes &&
                        cgraph->nodes[i + 3]->op == GGML_OP_MUL &&
                        cgraph->nodes[i + 4]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i + 5]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(cgraph->nodes[i + 5], 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                    ggml_tensor * mul_s   = cgraph->nodes[i + 3];
                    ggml_tensor * rsh     = cgraph->nodes[i + 4];
                    ggml_tensor * fwht_mm = cgraph->nodes[i + 5];
                    const ggml_tensor * sgn = (mul_s->src[0] == glu) ? mul_s->src[1] : ((mul_s->src[1] == glu) ? mul_s->src[0] : nullptr);
                    const int H = (int) fwht_mm->src[0]->ne[0];
                    if (sgn && sgn->type == GGML_TYPE_F32 && ggml_is_contiguous(sgn) &&
                            sgn->ne[0] == glu->ne[0] && ggml_nrows(sgn) == 1 &&
                            rsh->src[0] == mul_s && fwht_mm->src[1] == rsh &&
                            (H == 1024 || H == 512 || H == 256 || H == 128) &&
                            ggml_node_get_use_count(cgraph, i + 2) == 1 &&
                            ggml_node_get_use_count(cgraph, i + 3) == 1 &&
                            ggml_node_get_use_count(cgraph, i + 4) == 1 &&
                            (glu->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                            (mul_s->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                            (fwht_mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                        // Verify fwht_mm feeds a PQ2_0 MUL_MAT (directly or via a single RESHAPE) and nothing else!
                        int j_mm = i + 6;
                        const ggml_tensor * cur_t = fwht_mm;
                        while (j_mm < cgraph->n_nodes && cgraph->nodes[j_mm]->op == GGML_OP_RESHAPE &&
                                cgraph->nodes[j_mm]->src[0] == cur_t && ggml_node_get_use_count(cgraph, j_mm - 1) == 1) {
                            cur_t = cgraph->nodes[j_mm];
                            ++j_mm;
                        }
                        if (j_mm < cgraph->n_nodes && cgraph->nodes[j_mm]->op == GGML_OP_MUL_MAT &&
                                cgraph->nodes[j_mm]->src[1] == cur_t &&
                                cgraph->nodes[j_mm]->src[0]->type == GGML_TYPE_PQ2_0 &&
                                ggml_node_get_use_count(cgraph, j_mm - 1) == 1) {
                            fwht_signs  = sgn;
                            fwht_out    = fwht_mm;
                            fwht_H      = H;
                            extra_fused = 3; // also consume mul_s, rsh, fwht_mm!
                        }
                    }
                }
                int j_next = i + 3;
                while (j_next < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j_next])) {
                    ++j_next;
                }
                const bool direct_b = (extra_fused == 0) && (j_next < cgraph->n_nodes) &&
                    cgraph->nodes[j_next]->op == GGML_OP_MUL_MAT &&
                    cgraph->nodes[j_next]->src[1] == glu &&
                    cgraph->nodes[j_next]->src[0]->type == GGML_TYPE_PQ2_0 &&
                    ggml_node_get_use_count(cgraph, i + 2) == 1 &&
                    (glu->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
                if (ggml_cuda_mul_mat_pq2_prefill_int4_swiglu(*cuda_ctx, gate->src[0], up->src[0], src1, glu, fwht_signs, fwht_out, fwht_H)) {
                    fused_mul_mat_vec = true;
                    fused_node_count  = 3 + ((fwht_signs && fwht_out && fwht_H == 1024 && (glu->ne[0] % 1024) == 0 && (glu->ne[0] >> 10) <= 24) ? extra_fused : 0);
                    break;
                }
                if (ggml_cuda_mul_mat_pq2_cublas_i8_swiglu(*cuda_ctx, gate->src[0], up->src[0], src1, glu, direct_b, fwht_signs, fwht_out, fwht_H)) {
                    fused_mul_mat_vec = true;
                    fused_node_count  = 3 + extra_fused;
                    break;
                }
            }

            if (op == GGML_OP_MUL_MAT && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                    ggml_get_op_params_i32(glu, 1) == 0 && gate->src[1] == up->src[1] &&
                    ggml_cuda_mul_mat_pq2_narrow_int8_swiglu(*cuda_ctx, gate->src[0], up->src[0], src1, glu)) {
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
            const bool skip_mmvq_for_narrow_i8 = ggml_cuda_pq2_narrow_int8_enabled() &&
                src0->type == GGML_TYPE_PQ2_0 && src1->type == GGML_TYPE_F32 &&
                src1->ne[1] >= 5 && src1->ne[1] <= 64 && src0->ne[1] <= 262144;
            if (!skip_mmvq_for_narrow_i8 && ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (!skip_mmvq_for_narrow_i8 && ggml_cuda_should_fuse_mul_mat_q(up, gate, glu)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        } else if (op == GGML_OP_MUL_MAT && (ggml_cuda_pq2_cublas_i8_enabled() || ggml_cuda_pq2_narrow_int8_enabled()) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD }, { i + 1 })) {
            static const int fuse_add_mode = []() {
                const char * v = getenv("GGML_CUDA_PQ2_FUSE_ADD");
                return v != nullptr ? atoi(v) : 1; // 0=off, 1=7-op or non-RMS 2-op, 2=always 2-op
            }();
            const bool next_is_rms = (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_RMS_NORM);
            if (fuse_add_mode != 0) {
                int out_nodes[] = { i + 1 };
                if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, out_nodes, 1)) {
                    ggml_tensor * mm  = cgraph->nodes[i];
                    ggml_tensor * add = cgraph->nodes[i + 1];
                    const ggml_tensor * res = (add->src[0] == mm) ? add->src[1] : ((add->src[1] == mm) ? add->src[0] : nullptr);
                    // Try 4-op super-fusion: MUL_MAT + ADD + RMS_NORM + MUL(weight) -> B_i8 cache!
                    if (res && i + 3 < cgraph->n_nodes &&
                            cgraph->nodes[i + 2]->op == GGML_OP_RMS_NORM &&
                            cgraph->nodes[i + 3]->op == GGML_OP_MUL) {
                        ggml_tensor * rms   = cgraph->nodes[i + 2];
                        ggml_tensor * mul_w = cgraph->nodes[i + 3];
                        const ggml_tensor * w = (mul_w->src[0] == rms) ? mul_w->src[1] : ((mul_w->src[1] == rms) ? mul_w->src[0] : nullptr);
                        if (rms->src[0] == add && w && w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) &&
                                w->ne[0] == add->ne[0] && ggml_nrows(w) == 1 &&
                                rms->type == GGML_TYPE_F32 && mul_w->type == GGML_TYPE_F32 &&
                                ggml_are_same_shape(add, rms) && ggml_are_same_shape(rms, mul_w) &&
                                ggml_is_contiguous(add) && ggml_is_contiguous(mul_w) &&
                                ggml_node_get_use_count(cgraph, i + 2) == 1 &&
                                (rms->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                (mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                w->data != add->data && w->data != mul_w->data && add->data != mul_w->data) {
                            float eps = 0.0f;
                            memcpy(&eps, rms->op_params, sizeof(float));
                            // Try 7-op super-fusion: MUL_MAT + ADD + RMS_NORM + MUL(w) + MUL(signs) + RESHAPE + FWHT(1024) -> B_i8!
                            if (i + 6 < cgraph->n_nodes &&
                                    cgraph->nodes[i + 4]->op == GGML_OP_MUL &&
                                    cgraph->nodes[i + 5]->op == GGML_OP_RESHAPE &&
                                    cgraph->nodes[i + 6]->op == GGML_OP_MUL_MAT &&
                                    ggml_get_op_params_i32(cgraph->nodes[i + 6], 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                                ggml_tensor * mul_s   = cgraph->nodes[i + 4];
                                ggml_tensor * rsh     = cgraph->nodes[i + 5];
                                ggml_tensor * fwht_mm = cgraph->nodes[i + 6];
                                const ggml_tensor * sgn = (mul_s->src[0] == mul_w) ? mul_s->src[1] : ((mul_s->src[1] == mul_w) ? mul_s->src[0] : nullptr);
                                const int H = (int) fwht_mm->src[0]->ne[0];
                                if (sgn && sgn->type == GGML_TYPE_F32 && ggml_is_contiguous(sgn) &&
                                        sgn->ne[0] == add->ne[0] && ggml_nrows(sgn) == 1 && H == 1024 &&
                                        rsh->src[0] == mul_s && fwht_mm->src[1] == rsh &&
                                        ggml_node_get_use_count(cgraph, i + 4) == 1 &&
                                        ggml_node_get_use_count(cgraph, i + 5) == 1 &&
                                        (mul_s->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                        (fwht_mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                                    // Check if mul_w has any other consumer besides mul_s (e.g. ssm_alpha/beta!)
                                    const bool mul_w_only_sgn = (ggml_node_get_use_count(cgraph, i + 3) == 1);
                                    // Verify all consumers of fwht_mm (through optional RESHAPE) are PQ2_0 MUL_MATs
                                    bool fwht_only_pq2 = true;
                                    for (int j = i + 7; j < cgraph->n_nodes && fwht_only_pq2; ++j) {
                                        const ggml_tensor * cn = cgraph->nodes[j];
                                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                                            if (cn->src[s] == fwht_mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == fwht_mm)) {
                                                if (cn->op == GGML_OP_RESHAPE) { continue; }
                                                if (!(s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0)) {
                                                    fwht_only_pq2 = false; break;
                                                }
                                            }
                                        }
                                    }
                                    if (fwht_only_pq2 && ggml_cuda_mul_mat_pq2_cublas_i8(
                                            *cuda_ctx, mm->src[0], mm->src[1], add, res, w, eps, mul_w, mul_w_only_sgn, sgn, fwht_mm)) {
                                        fused_mul_mat_vec = true;
                                        fused_node_count  = 7;
                                        break;
                                    }
                                }
                            }
                            // Only fall back to 4-op fusion if mul_w is NOT followed by Hadamard (otherwise let MUL_MAT+ADD run)
                            bool all_i8_mm = true;
                            for (int j = i + 4; j < cgraph->n_nodes && all_i8_mm; ++j) {
                                const ggml_tensor * cn = cgraph->nodes[j];
                                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                                    if (cn->src[s] == mul_w) {
                                        const bool ok_mm = (s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] &&
                                            (cn->src[0]->type == GGML_TYPE_PQ2_0 ||
                                             (cn->src[0]->type == GGML_TYPE_F32 && cn->src[0]->ne[1] <= 64 &&
                                              strstr(cn->src[0]->name, "lora") == nullptr)));
                                        if (!ok_mm) { all_i8_mm = false; break; }
                                    }
                                }
                            }
                            if (all_i8_mm && ggml_cuda_mul_mat_pq2_cublas_i8(*cuda_ctx, mm->src[0], mm->src[1], add, res, w, eps, mul_w, true)) {
                                fused_mul_mat_vec = true;
                                fused_node_count  = 4;
                                break;
                            }
                        }
                    }
                    if (fuse_add_mode == 2 || !next_is_rms) {
                        if ((!next_is_rms || (fuse_add_mode == 2 && mm->src[1]->ne[1] > 32)) && res &&
                                ggml_cuda_mul_mat_pq2_narrow_int8(*cuda_ctx, mm->src[0], mm->src[1], add, res)) {
                            fused_mul_mat_vec = true;
                            fused_node_count  = 2;
                            break;
                        }
                        if (res && ggml_cuda_mul_mat_pq2_prefill_int4(*cuda_ctx, mm->src[0], mm->src[1], add, res)) {
                            fused_mul_mat_vec = true;
                            fused_node_count  = 2;
                            break;
                        }
                        if (res && ggml_cuda_mul_mat_pq2_cublas_i8(*cuda_ctx, mm->src[0], mm->src[1], add, res)) {
                            fused_mul_mat_vec = true;
                            fused_node_count  = 2;
                            break;
                        }
                    }
                }
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // Lossless Fused Full-Attn Gate: SIGMOID(gate) + MUL(attn_out, sig) + MUL(signs) + RESHAPE + FWHT(1024) -> B_i8!
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
            ggml_cuda_pq2_cublas_i8_enabled() && pq2_b_i8_cache_enabled() &&
            i + 4 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL &&
            cgraph->nodes[i + 3]->op == GGML_OP_RESHAPE &&
            cgraph->nodes[i + 4]->op == GGML_OP_MUL_MAT &&
            ggml_get_op_params_i32(cgraph->nodes[i + 4], 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ggml_tensor * sig     = cgraph->nodes[i];
        ggml_tensor * mul_g   = cgraph->nodes[i + 1];
        ggml_tensor * mul_s   = cgraph->nodes[i + 2];
        ggml_tensor * rsh     = cgraph->nodes[i + 3];
        ggml_tensor * fwht_mm = cgraph->nodes[i + 4];
        const ggml_tensor * gate = sig->src[0];
        const ggml_tensor * aout = (mul_g->src[0] == sig) ? mul_g->src[1] : ((mul_g->src[1] == sig) ? mul_g->src[0] : nullptr);
        const ggml_tensor * sgn  = (mul_s->src[0] == mul_g) ? mul_s->src[1] : ((mul_s->src[1] == mul_g) ? mul_s->src[0] : nullptr);
        const int H = (int) fwht_mm->src[0]->ne[0];
        if (gate && aout && sgn && H == 1024 &&
                gate->type == GGML_TYPE_F32 && aout->type == GGML_TYPE_F32 && sgn->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(gate) && ggml_is_contiguous(aout) && ggml_is_contiguous(sgn) &&
                ggml_are_same_shape(gate, aout) && sgn->ne[0] == gate->ne[0] && ggml_nrows(sgn) == 1 &&
                rsh->src[0] == mul_s && fwht_mm->src[1] == rsh &&
                ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                ggml_node_get_use_count(cgraph, i + 2) == 1 && ggml_node_get_use_count(cgraph, i + 3) == 1 &&
                (sig->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (mul_g->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                (mul_s->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (fwht_mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            bool fwht_only_pq2 = true;
            for (int j = i + 5; j < cgraph->n_nodes && fwht_only_pq2; ++j) {
                const ggml_tensor * cn = cgraph->nodes[j];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (cn->src[s] == fwht_mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == fwht_mm)) {
                        if (cn->op == GGML_OP_RESHAPE) { continue; }
                        if (!(s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0)) {
                            fwht_only_pq2 = false; break;
                        }
                    }
                }
            }
            auto it_skip = tl_skipped_gate_cont_i8.find(gate);
            const bool is_strided = (it_skip != tl_skipped_gate_cont_i8.end());
            const ggml_tensor * gate_src = is_strided ? it_skip->second : gate;
            if (fwht_only_pq2 && ggml_cuda_sigmoid_mul_fwht1024_to_i8(
                    *cuda_ctx, aout, gate_src, sgn, fwht_mm, gate->ne[0], gate->ne[1], is_strided)) {
                if (is_strided) { tl_skipped_gate_cont_i8.erase(it_skip); }
                return 4;
            }
        }
    }

    // Lossless Fused GDN build_norm_gated (+ 4D head permute) + FWHT(1024) -> B_i8!
    if (node->op == GGML_OP_RMS_NORM && ggml_cuda_pq2_cublas_i8_enabled() && pq2_b_i8_cache_enabled() &&
            i + 6 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            cgraph->nodes[i + 2]->op == GGML_OP_GLU &&
            cgraph->nodes[i + 3]->op == GGML_OP_RESHAPE) {
        ggml_tensor * rms = cgraph->nodes[i]; ggml_tensor * mul_w = cgraph->nodes[i + 1];
        ggml_tensor * glu = cgraph->nodes[i + 2]; ggml_tensor * rsh1 = cgraph->nodes[i + 3];
        int perm_nk = 1, perm_rep = 1, idx_ms = i + 4;
        const ggml_tensor * pre_ms = rsh1;
        bool perm_ok = true;
        if (i + 10 < cgraph->n_nodes &&
                cgraph->nodes[i + 4]->op == GGML_OP_RESHAPE && cgraph->nodes[i + 5]->op == GGML_OP_PERMUTE &&
                cgraph->nodes[i + 6]->op == GGML_OP_CONT && cgraph->nodes[i + 7]->op == GGML_OP_RESHAPE) {
            ggml_tensor * rsh_p = cgraph->nodes[i + 4]; ggml_tensor * perm = cgraph->nodes[i + 5];
            ggml_tensor * cont_p = cgraph->nodes[i + 6]; ggml_tensor * rsh_p2 = cgraph->nodes[i + 7];
            if (rsh_p->src[0] == rsh1 && perm->src[0] == rsh_p && cont_p->src[0] == perm && rsh_p2->src[0] == cont_p &&
                    rsh_p->ne[0] == 128 && perm->ne[0] == 128 && perm->ne[1] == rsh_p->ne[2] && perm->ne[2] == rsh_p->ne[1] &&
                    rsh_p->ne[2] > 1 && rsh_p->ne[0] * rsh_p->ne[1] * rsh_p->ne[2] == rsh1->ne[0] &&
                    ggml_node_get_use_count(cgraph, i + 4) == 1 && ggml_node_get_use_count(cgraph, i + 5) == 1 &&
                    ggml_node_get_use_count(cgraph, i + 6) == 1 && ggml_node_get_use_count(cgraph, i + 7) == 1 &&
                    (cont_p->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                perm_nk = (int) rsh_p->ne[1]; perm_rep = (int) rsh_p->ne[2]; pre_ms = rsh_p2; idx_ms = i + 8;
            } else { perm_ok = false; }
        }
        if (perm_ok && idx_ms + 2 < cgraph->n_nodes &&
                cgraph->nodes[idx_ms]->op == GGML_OP_MUL && cgraph->nodes[idx_ms + 1]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[idx_ms + 2]->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(cgraph->nodes[idx_ms + 2], 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            ggml_tensor * mul_s = cgraph->nodes[idx_ms]; ggml_tensor * rsh2 = cgraph->nodes[idx_ms + 1];
            ggml_tensor * fwht_mm = cgraph->nodes[idx_ms + 2];
            const ggml_tensor * src = rms->src[0];
            const ggml_tensor * rw  = (mul_w->src[0] == rms) ? mul_w->src[1] : ((mul_w->src[1] == rms) ? mul_w->src[0] : nullptr);
            const ggml_tensor * z   = (glu->src[1] == mul_w) ? glu->src[0] : nullptr;
            const ggml_tensor * sgn = (mul_s->src[0] == pre_ms) ? mul_s->src[1] : ((mul_s->src[1] == pre_ms) ? mul_s->src[0] : nullptr);
            const int H = (int) fwht_mm->src[0]->ne[0];
            if (src && rw && z && sgn && H == 1024 &&
                    ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(glu, 1) == 0 &&
                    rms->ne[0] == 128 && src->type == GGML_TYPE_F32 && rw->type == GGML_TYPE_F32 &&
                    z->type == GGML_TYPE_F32 && sgn->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(src) && ggml_is_contiguous(rw) && ggml_is_contiguous(z) && ggml_is_contiguous(sgn) &&
                    rw->ne[0] == 128 && ggml_nrows(rw) == 1 && ggml_are_same_shape(src, z) &&
                    rsh1->src[0] == glu && rsh2->src[0] == mul_s && fwht_mm->src[1] == rsh2 &&
                    sgn->ne[0] == rsh1->ne[0] && ggml_nrows(sgn) == 1 &&
                    ggml_node_get_use_count(cgraph, i) == 1 && ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                    ggml_node_get_use_count(cgraph, i + 2) == 1 && ggml_node_get_use_count(cgraph, i + 3) == 1 &&
                    ggml_node_get_use_count(cgraph, idx_ms) == 1 && ggml_node_get_use_count(cgraph, idx_ms + 1) == 1 &&
                    (rms->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                    (glu->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (mul_s->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                    (fwht_mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                bool fwht_only_pq2 = true;
                for (int j = idx_ms + 3; j < cgraph->n_nodes && fwht_only_pq2; ++j) {
                    const ggml_tensor * cn = cgraph->nodes[j];
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        if (cn->src[s] == fwht_mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == fwht_mm)) {
                            if (cn->op == GGML_OP_RESHAPE) { continue; }
                            if (!(s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0)) {
                                fwht_only_pq2 = false; break;
                            }
                        }
                    }
                }
                float eps = 0.0f; memcpy(&eps, rms->op_params, sizeof(float));
                if (fwht_only_pq2 && ggml_cuda_gdn_norm_gated_fwht1024_to_i8(
                        *cuda_ctx, src, rw, eps, z, sgn, fwht_mm, rsh1->ne[0], rsh1->ne[1], perm_nk, perm_rep)) {
                    return (idx_ms + 2) - i;
                }
            }
        }
    }

    // 5-op Fused RMS_NORM + MUL(w) + MUL(signs) + RESHAPE + FWHT(1024) -> B_s4 (eliminates ALL 127 RMS_NORM + MUL + FWHT pairs!)
    if (node->op == GGML_OP_RMS_NORM && ggml_cuda_pq2_prefill_int4_enabled() &&
            i + 4 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL &&
            cgraph->nodes[i + 3]->op == GGML_OP_RESHAPE &&
            cgraph->nodes[i + 4]->op == GGML_OP_MUL_MAT &&
            ggml_get_op_params_i32(cgraph->nodes[i + 4], 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ggml_tensor * rms     = cgraph->nodes[i];
        ggml_tensor * mul_w   = cgraph->nodes[i + 1];
        ggml_tensor * mul_s   = cgraph->nodes[i + 2];
        ggml_tensor * rsh     = cgraph->nodes[i + 3];
        ggml_tensor * fwht_mm = cgraph->nodes[i + 4];
        const ggml_tensor * x   = rms->src[0];
        const ggml_tensor * rw  = (mul_w->src[0] == rms)   ? mul_w->src[1] : ((mul_w->src[1] == rms)   ? mul_w->src[0] : nullptr);
        const ggml_tensor * sgn = (mul_s->src[0] == mul_w) ? mul_s->src[1] : ((mul_s->src[1] == mul_w) ? mul_s->src[0] : nullptr);
        const int H = (int) fwht_mm->src[0]->ne[0];
        if (x && rw && sgn && H == 1024 &&
                x->type == GGML_TYPE_F32 && rw->type == GGML_TYPE_F32 && sgn->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(rw) && ggml_is_contiguous(sgn) &&
                rw->ne[0] == x->ne[0] && ggml_nrows(rw) == 1 &&
                sgn->ne[0] == x->ne[0] && ggml_nrows(sgn) == 1 &&
                x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) &&
                x->ne[1] >= 256 && (x->ne[0] % 1024) == 0 && (x->ne[0] >> 10) <= 24 &&
                rsh->src[0] == mul_s && fwht_mm->src[1] == rsh &&
                ggml_node_get_use_count(cgraph, i) == 1 &&
                ggml_node_get_use_count(cgraph, i + 2) == 1 &&
                ggml_node_get_use_count(cgraph, i + 3) == 1 &&
                (rms->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                (mul_s->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                (fwht_mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            bool fwht_only_pq2 = true;
            int n_pq2 = 0;
            for (int j = i + 5; j < cgraph->n_nodes && fwht_only_pq2; ++j) {
                const ggml_tensor * cn = cgraph->nodes[j];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (cn->src[s] == fwht_mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == fwht_mm)) {
                        if (cn->op == GGML_OP_RESHAPE) { continue; }
                        if (s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0 && cn->src[0]->ne[1] <= 65536) {
                            ++n_pq2;
                        } else {
                            fwht_only_pq2 = false; break;
                        }
                    }
                }
            }
            if (fwht_only_pq2 && n_pq2 >= 1) {
                cudaStream_t stream = cuda_ctx->stream();
                const int64_t M = x->ne[0], N = x->ne[1], s11 = x->nb[1] / sizeof(float);
                const int64_t N_pad = ((N + 127) / 128) * 128;
                auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
                const size_t need_b = (size_t) N_pad * (size_t) (M >> 3) * sizeof(uint32_t);
                const size_t need_s = (size_t) N_pad * sizeof(float);
                if (sc.cap_b < need_b)  { if (sc.B_i4)     CUDA_CHECK(cudaFree(sc.B_i4));     CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));     sc.cap_b = need_b; }
                if (sc.cap_sc < need_s) { if (sc.B_scales) CUDA_CHECK(cudaFree(sc.B_scales)); CUDA_CHECK(cudaMalloc(&sc.B_scales, need_s)); sc.cap_sc = need_s; }
                const bool need_mw = (ggml_node_get_use_count(cgraph, i + 1) > 1) || ((mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) != 0);
                float eps = 0.0f; memcpy(&eps, rms->op_params, sizeof(float));
                const int n_blk = (int) (M >> 10);
                rms_mul_signs_fwht1024_to_s4_kernel<<<(unsigned) N_pad, (unsigned) (n_blk * 32), 0, stream>>>(
                    (const float *) x->data, nullptr, nullptr, (const float *) rw->data, eps, (const float *) sgn->data,
                    need_mw ? (float *) mul_w->data : nullptr, nullptr,
                    sc.B_i4, sc.B_scales, (int) M, (int) N, (int) N_pad, s11, ggml_cuda_pq2_prefill_int4_clip());
                CUDA_CHECK(cudaGetLastError());
                sc.src1_data = fwht_mm->data; sc.src1_nbytes = ggml_nbytes(fwht_mm);
                sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true;
                return 4;
            }
        }
    }

    // Fused MUL(Hadamard signs) -> RESHAPE -> MUL_MAT(HADAMARD) -> [RESHAPE] -> PQ2_0 MUL_MAT into B_s4, narrow_i8, or B_i8 cache!
    if (node->op == GGML_OP_MUL && (ggml_cuda_pq2_narrow_int8_enabled() || (ggml_cuda_pq2_cublas_i8_enabled() && pq2_b_i8_cache_enabled())) &&
            i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL_MAT &&
            ggml_get_op_params_i32(cgraph->nodes[i + 2], 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ggml_tensor * mul_s   = node;
        ggml_tensor * rsh     = cgraph->nodes[i + 1];
        ggml_tensor * fwht_mm = cgraph->nodes[i + 2];
        const ggml_tensor * x   = mul_s->src[0];
        const ggml_tensor * sgn = mul_s->src[1];
        const int H = (int) fwht_mm->src[0]->ne[0];
        if (x && sgn && x->type == GGML_TYPE_F32 && sgn->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(sgn) && sgn->ne[0] == x->ne[0] && ggml_nrows(sgn) == 1 &&
                x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) &&
                ((x->ne[1] >= 5 && x->ne[1] <= pq2_narrow_i8_max_n() && H == 1024 && ggml_cuda_pq2_narrow_int8_enabled()) ||
                 (x->ne[1] >= 129 && (H == 1024 || H == 512 || H == 256 || H == 128))) &&
                (x->ne[0] % H) == 0 && (H == 1024 ? (x->ne[0] / 1024) <= 18 : (x->ne[0] / H) <= 36) &&
                rsh->src[0] == mul_s && fwht_mm->src[1] == rsh &&
                ggml_node_get_use_count(cgraph, i) == 1 &&
                ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                (mul_s->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                (fwht_mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            // Verify ALL consumers of fwht_mm (directly or via RESHAPE) are PQ2_0 MUL_MATs (allows multi-consumer like attn_qkv + attn_gate!)
            bool fwht_only_pq2 = true;
            int n_pq2_consumers = 0;
            int64_t min_consumer_m = INT64_MAX;
            int64_t max_consumer_m = 0;
            for (int j = i + 3; j < cgraph->n_nodes && fwht_only_pq2; ++j) {
                const ggml_tensor * cn = cgraph->nodes[j];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (cn->src[s] == fwht_mm || (cn->src[s] && cn->src[s]->op == GGML_OP_RESHAPE && cn->src[s]->src[0] == fwht_mm)) {
                        if (cn->op == GGML_OP_RESHAPE) { continue; }
                        if (s == 1 && cn->op == GGML_OP_MUL_MAT && cn->src[0] && cn->src[0]->type == GGML_TYPE_PQ2_0) {
                            ++n_pq2_consumers;
                            min_consumer_m = std::min(min_consumer_m, cn->src[0]->ne[1]);
                            max_consumer_m = std::max(max_consumer_m, cn->src[0]->ne[1]);
                        } else {
                            fwht_only_pq2 = false; break;
                        }
                    }
                }
            }
            if (fwht_only_pq2 && n_pq2_consumers >= 1) {
                cudaStream_t stream = cuda_ctx->stream();
                cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
                (void) cudaStreamIsCapturing(stream, &cap_status);
                const bool allow_alloc = (cap_status == cudaStreamCaptureStatusNone);
                const int64_t M = x->ne[0];
                const int64_t N = x->ne[1];
                const int64_t s11 = x->nb[1] / sizeof(float);
                if (ggml_cuda_pq2_narrow_int8_enabled() && N >= 5 && N <= pq2_narrow_i8_max_n() &&
                        max_consumer_m <= (N >= 256 ? 65536 : 262144) && H == 1024 && (M >> 10) <= 24) {
                    auto & sc = pq2_narrow_i8_scratch_for_stream(stream);
                    const int64_t N_pad = pq2_narrow_i8_n_pad(N);
                    if (pq2_narrow_i8_alloc_bufs(sc, N_pad, M, stream)) {
                        const int n_blk = (int) (M >> 10);
                        mul_signs_fwht1024_to_narrow_i8_kernel<<<(unsigned) N_pad, (unsigned) (n_blk * 32), 0, stream>>>(
                            (const float *) x->data, (const float *) sgn->data,
                            sc.B_i8, sc.B_sum, sc.B_scales, (int) M, (int) N, (int) N_pad, s11);
                        CUDA_CHECK(cudaGetLastError());
                        sc.src1_data = fwht_mm->data; sc.src1_nbytes = ggml_nbytes(fwht_mm);
                        sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true; ++sc.gen;
                        return 2;
                    }
                }
                if (ggml_cuda_pq2_prefill_int4_enabled() && N >= 256 && max_consumer_m <= 65536 && H == 1024 && (M >> 10) <= 24) {
                    auto & sc = pq2_prefill_i4_scratch_for_stream(stream);
                    const int64_t N_pad = ((N + 127) / 128) * 128;
                    const size_t need_b = (size_t) N_pad * (size_t) (M >> 3) * sizeof(uint32_t);
                    const size_t need_s = (size_t) N_pad * sizeof(float);
                    if (allow_alloc || (sc.cap_b >= need_b && sc.cap_sc >= need_s)) {
                        if (sc.cap_b < need_b)  { if (sc.B_i4)     CUDA_CHECK(cudaFree(sc.B_i4));     CUDA_CHECK(cudaMalloc(&sc.B_i4, need_b));     sc.cap_b = need_b; }
                        if (sc.cap_sc < need_s) { if (sc.B_scales) CUDA_CHECK(cudaFree(sc.B_scales)); CUDA_CHECK(cudaMalloc(&sc.B_scales, need_s)); sc.cap_sc = need_s; }
                        const int n_blk = (int) (M >> 10);
                        mul_signs_fwht1024_to_s4_kernel<<<(unsigned) N_pad, (unsigned) (n_blk * 32), 0, stream>>>(
                            (const float *) x->data, (const float *) sgn->data, sc.B_i4, sc.B_scales,
                            (int) M, (int) N, (int) N_pad, s11, ggml_cuda_pq2_prefill_int4_clip());
                        CUDA_CHECK(cudaGetLastError());
                        sc.src1_data   = fwht_mm->data;
                        sc.src1_nbytes = ggml_nbytes(fwht_mm);
                        sc.N = N; sc.K = M; sc.s11 = M; sc.valid = true;
                        return 2;
                    }
                }
                const bool all_consumers_cublas_i8 =
                    (min_consumer_m > 65536 && N >= 256) ||
                    (N >= 129 && (!ggml_cuda_pq2_narrow_int8_enabled() || N > pq2_narrow_i8_max_n()) &&
                     !(ggml_cuda_pq2_prefill_int4_enabled() && N >= 256));
                if (ggml_cuda_pq2_cublas_i8_enabled() && pq2_b_i8_cache_enabled() && all_consumers_cublas_i8) {
                    auto & st = pq2_b_i8_cache_for_stream(stream);
                    const int64_t N_pad = (N + 3) & ~int64_t(3);
                    const size_t need_b = (size_t) N_pad * (size_t) M;
                    const size_t need_s = (size_t) N_pad * sizeof(float);
                    if (allow_alloc || (st.B_i8_cap >= need_b && st.col_scales_cap >= need_s)) {
                        if (st.B_i8_cap < need_b) {
                            if (st.B_i8) { CUDA_CHECK(cudaFree(st.B_i8)); }
                            CUDA_CHECK(cudaMalloc(&st.B_i8, need_b));
                            st.B_i8_cap = need_b;
                        }
                        if (st.col_scales_cap < need_s) {
                            if (st.col_scales) { CUDA_CHECK(cudaFree(st.col_scales)); }
                            CUDA_CHECK(cudaMalloc(&st.col_scales, need_s));
                            st.col_scales_cap = need_s;
                        }
                        if (H == 1024) {
                            mul_signs_fwht1024_to_i8_kernel<<<(unsigned) N, 512, 0, stream>>>(
                                (const float *) x->data, (const float *) sgn->data, st.B_i8, st.col_scales, (int) M, (int) N, s11);
                        } else if (H == 512) {
                            mul_signs_fwht_to_i8_kernel<512><<<(unsigned) N, 512, 0, stream>>>(
                                (const float *) x->data, (const float *) sgn->data, st.B_i8, st.col_scales, (int) M, (int) N, s11);
                        } else if (H == 256) {
                            mul_signs_fwht_to_i8_kernel<256><<<(unsigned) N, 256, 0, stream>>>(
                                (const float *) x->data, (const float *) sgn->data, st.B_i8, st.col_scales, (int) M, (int) N, s11);
                        } else {
                            mul_signs_fwht_to_i8_kernel<128><<<(unsigned) N, 128, 0, stream>>>(
                                (const float *) x->data, (const float *) sgn->data, st.B_i8, st.col_scales, (int) M, (int) N, s11);
                        }
                        CUDA_CHECK(cudaGetLastError());
                        st.src1_data   = fwht_mm->data;
                        st.src1_nbytes = ggml_nbytes(fwht_mm);
                        st.N           = N;
                        st.K           = M;
                        st.s11         = M;
                        st.valid       = true;
                        return 2; // consumed mul_s, rsh, fwht_mm!
                    }
                }
            }
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    const int ep_out2 = i + 2;
    if ((ggml_cuda_moe_ep_bits() & 2) && i + 2 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_SCALE &&
            cgraph->nodes[i + 1]->src[0] == node && cgraph->nodes[i + 2]->op == GGML_OP_MUL &&
            (cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1] || cgraph->nodes[i + 2]->src[1] == cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_MUL }, { i + 2 }) &&
            ggml_cuda_check_fusion_memory_ranges(cgraph, i, 3, &ep_out2, 1)) {
        if (ggml_cuda_moe_ep_verify_mode() != 0) {
            static float * scratch = nullptr; static int tag = 0;
            if (!scratch) { CUDA_CHECK(cudaMalloc(&scratch, (size_t) 32*1024*1024)); }
            ggml_tensor * mul_n = cgraph->nodes[i + 2];
            if (ggml_nelements(mul_n)*sizeof(float) <= (size_t) 32*1024*1024 && tag < 1000000 &&
                    ggml_cuda_op_rms_norm_scale_mul_fused(*cuda_ctx, node, cgraph->nodes[i + 1], mul_n, scratch)) {
                ggml_cuda_tail_verify_pending()[mul_n] = { scratch, tag++ };
            }
        } else if (ggml_cuda_op_rms_norm_scale_mul_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
            return 2;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_dsv4_hc_post(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    if (node->op == GGML_OP_MUL_MAT && ggml_cuda_gdn_fusion_enabled() &&
            GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
        if (const int n = ggml_cuda_try_mmvf_pair(*cuda_ctx, cgraph, i)) {
            return n;
        }
    }

    return 0;
}

// GGML_CUDA_OP_TIMING=1 (with CUDA graphs off): an event on the main stream before every node that launches work
// (views and no-ops get none); the time to the next event is charged to that node (a fused group to its first
// node). Events come from a pool and are read back only for the report, every 100 graphs, so the stream does not
// stop after each graph. The key of a node is kept as numbers: the graph's tensors can be gone at report time.
struct ggml_cuda_op_timing {
    struct rec {
        cudaEvent_t  ev;
        const char * desc;   // nullptr: the end of a graph
        ggml_type    type0;
        int64_t      ne00;
        int64_t      ne01;
        int64_t      ne1;
        bool         mul_mat;
    };

    std::vector<cudaEvent_t> pool;
    std::vector<rec>         recs;
    int64_t                  n_graphs = 0;

    cudaEvent_t next_event() {
        const size_t i = recs.size();
        if (i == pool.size()) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreate(&e));
            pool.push_back(e);
        }
        return pool[i];
    }

    void record(const ggml_tensor * t, cudaStream_t stream) {
        rec r = { next_event(), ggml_op_desc(t), t->src[0] ? t->src[0]->type : GGML_TYPE_COUNT,
                  t->src[0] ? t->src[0]->ne[0] : 0, t->src[0] ? t->src[0]->ne[1] : 0, t->ne[1], t->op == GGML_OP_MUL_MAT && t->src[0] };
        CUDA_CHECK(cudaEventRecord(r.ev, stream));
        recs.push_back(r);
    }

    void end_graph(cudaStream_t stream) {
        rec r = { next_event(), nullptr, GGML_TYPE_COUNT, 0, 0, 0, false };
        CUDA_CHECK(cudaEventRecord(r.ev, stream));
        recs.push_back(r);
        static const int report_n = getenv("GGML_CUDA_OP_TIMING_N") ? atoi(getenv("GGML_CUDA_OP_TIMING_N")) : 100;
        if (++n_graphs < report_n) {
            return;
        }
        CUDA_CHECK(cudaEventSynchronize(r.ev));
        struct acc { double ms = 0; int64_t n = 0; };
        std::map<std::string, acc> totals;
        double ms_graphs = 0.0;
        for (size_t k = 0; k + 1 < recs.size(); ++k) {
            const rec & a = recs[k];
            if (a.desc == nullptr) {
                continue;
            }
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a.ev, recs[k + 1].ev));
            char key[160];
            if (a.mul_mat) {
                snprintf(key, sizeof(key), "MUL_MAT %s %lldx%lld cols=%lld", ggml_type_name(a.type0),
                    (long long) a.ne00, (long long) a.ne01, (long long) a.ne1);
            } else {
                snprintf(key, sizeof(key), "%s%s%s cols=%lld", a.desc, a.type0 != GGML_TYPE_COUNT ? " " : "",
                    a.type0 != GGML_TYPE_COUNT ? ggml_type_name(a.type0) : "", (long long) a.ne1);
            }
            auto & s = totals[key];
            s.ms += ms;
            s.n  += 1;
            ms_graphs += ms;
        }
        std::vector<std::pair<std::string, acc>> v(totals.begin(), totals.end());
        std::sort(v.begin(), v.end(), [](auto & x, auto & y) { return x.second.ms > y.second.ms; });
        fprintf(stderr, "op timing over %lld graphs: %.3f ms per graph\n", (long long) n_graphs, ms_graphs / n_graphs);
        for (size_t k = 0; k < v.size() && k < 30; ++k) {
            fprintf(stderr, "  %8.3f ms/graph %6.1f%%  x%-7.1f %s\n", v[k].second.ms / n_graphs,
                100.0 * v[k].second.ms / ms_graphs, (double) v[k].second.n / n_graphs, v[k].first.c_str());
        }
        fflush(stderr);
        recs.clear();
        n_graphs = 0;
    }
};

static ggml_cuda_op_timing & ggml_cuda_op_timing_of(int device) {
    static ggml_cuda_op_timing state[GGML_CUDA_MAX_DEVICES];
    return state[device];
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    struct gb10_shared_q8_entry {
        const ggml_tensor * src1;
        ggml_type type;
        ggml_cuda_pool_alloc<char> * data;
        bool quantized = false;
        int remaining = 0;
    };
    std::vector<gb10_shared_q8_entry> gb10_shared_q8;

    struct gb10_virtual_rms_entry {
        const ggml_tensor * normalized;
        ggml_tensor * residual;
        const ggml_tensor * weight;
        ggml_type type;
        ggml_cuda_pool_alloc<char> * row_scale;
        int remaining;
    };
    std::vector<gb10_virtual_rms_entry> gb10_virtual_rms;
    // Keep every persistent pool allocation in true allocation order. The VMM
    // scratch pool is stack-like, so all releases happen globally in reverse.
    std::vector<std::unique_ptr<ggml_cuda_pool_alloc<char>>> gb10_pool_allocations;

    // only read while the nodes are evaluated or captured: not built when a captured CUDA graph is replayed
    std::map<const ggml_tensor *, std::array<int, GGML_TYPE_COUNT>> gb10_shared_q8_consumer_counts;
    if (!use_cuda_graph || cuda_graph_update_required) {
        for (int j = 0; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * candidate = cgraph->nodes[j];
            if (candidate->op == GGML_OP_MUL_MAT && candidate->src[0] && candidate->src[1]) {
                ++gb10_shared_q8_consumer_counts[candidate->src[1]][candidate->src[0]->type];
            }
        }
    }
    const auto gb10_shared_q8_consumer_count = [&](const ggml_tensor * src1, ggml_type type) {
        const auto it = gb10_shared_q8_consumer_counts.find(src1);
        return it == gb10_shared_q8_consumer_counts.end() ? 0 : it->second[type];
    };

    static const bool virtual_rms_q8_enabled = [] {
        const char * env = getenv("GGML_CUDA_GB10_VIRTUAL_RMS_Q8");
        return !env || std::atoi(env) != 0;
    }();
    static const bool shared_q8_enabled = [] {
        const char * env = getenv("GGML_CUDA_GB10_SHARED_Q8");
        return !env || std::atoi(env) != 0;
    }();
#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph * graph      = use_cuda_graph ? cuda_ctx->cuda_graph(graph_key) : nullptr;
    const int         chunk_size = ggml_cuda_graph_chunk_size();
    int               chunk_end  = chunk_size;
    size_t            n_chunks   = 0;
#endif // USE_CUDA_GRAPH

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        for (auto & entry : gb10_shared_q8) {
            entry.quantized = false;
            entry.remaining = gb10_shared_q8_consumer_count(entry.src1, entry.type);
        }
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            cuda_ctx->gdn_gathers().reset();
            cuda_ctx->fwht_q8().reset();
            cuda_ctx->gdn_conv().reset();
            pq2_b_i8_cache_invalidate(cuda_ctx->stream());
            pq2_narrow_i8_cache_invalidate(cuda_ctx->stream());
            pq2_prefill_i4_cache_invalidate(cuda_ctx->stream());


            // GGML_CUDA_OP_TIMING=1: see ggml_cuda_op_timing
            static const bool op_timing = getenv("GGML_CUDA_OP_TIMING") != nullptr;
            const bool timing = op_timing && !use_cuda_graph;

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
#ifdef USE_CUDA_GRAPH
                // fused nodes are skipped together, so a cut never splits a fusion
                // a concurrent region must join before its capture can end
                if (use_cuda_graph && chunk_size > 0 && i >= chunk_end && !is_concurrent_event_active) {
                    ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);
                    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
                    chunk_end = i + chunk_size;
                }
#endif // USE_CUDA_GRAPH
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (!cuda_ctx->fwht_q8().precomputed.empty() && cuda_ctx->fwht_q8().precomputed.count(node)) {
                    continue; // computed ahead of its position by a look-ahead fusion in ggml_cuda_try_fuse
                }

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                if (is_concurrent_event_active) {
                    pq2_b_i8_cache_invalidate(cuda_ctx->stream());
                    pq2_narrow_i8_cache_invalidate(cuda_ctx->stream());
                    pq2_prefill_i4_cache_invalidate(cuda_ctx->stream());
                } else {
                    if (node->op != GGML_OP_MUL_MAT || !node->src[0] || node->src[0]->type != GGML_TYPE_PQ2_0) {
                        pq2_narrow_i8_cache_invalidate_if_overlap(cuda_ctx->stream(), node->data, ggml_nbytes(node));
                        pq2_b_i8_cache_invalidate_if_overlap(cuda_ctx->stream(), node->data, ggml_nbytes(node));
                        pq2_prefill_i4_cache_invalidate_if_overlap(cuda_ctx->stream(), node->data, ggml_nbytes(node));
                    }
                }

                if (timing) {
                    ggml_cuda_op_timing_of(cuda_ctx->device).record(node, cuda_ctx->stream(cuda_ctx->device, 0));
                }

                // GDN decode conv chain: the conv-state gather moves into the fused launch at its CONCAT, which
                // also computes the conv-state CPY, SSM_CONV + SILU and the q/k L2_NORM (marked done)
                if (!cuda_ctx->gdn_conv().done.empty() && cuda_ctx->gdn_conv().done.count(node)) {
                    continue;
                }
                if (node->op == GGML_OP_GET_ROWS && !is_concurrent_event_active &&
                        ggml_cuda_try_gdn_conv_gather_skip(*cuda_ctx, cgraph, i)) {
                    continue;
                }
                if (node->op == GGML_OP_CONCAT && !is_concurrent_event_active &&
                        ggml_cuda_try_gdn_conv_fused(*cuda_ctx, cgraph, i)) {
                    continue;
                }

                // recurrent-state gather folded into the GDN kernel (no temp, no kernel)
                if (node->op == GGML_OP_GET_ROWS && !is_concurrent_event_active &&
                        ggml_cuda_try_gdn_gather_skip(*cuda_ctx, cgraph, i)) {
                    continue;
                }

                if (node->op == GGML_OP_GLU && !is_concurrent_event_active) {
                    const int consumed = ggml_cuda_try_glu_fwht_mmq(*cuda_ctx, cgraph, i);
                    if (consumed > 0) {
                        i += consumed - 1;
                        continue;
                    }
                }
                // Hadamard rotation that quantizes its own output; its PTQ1_0 mat-vecs skip quantize
                if ((node->op == GGML_OP_MUL || node->op == GGML_OP_MUL_MAT) && !is_concurrent_event_active) {
                    const int consumed = ggml_cuda_try_fwht_q8(*cuda_ctx, cgraph, i);
                    if (consumed > 0) {
                        i += consumed - 1;
                        continue;
                    }
                }

                // The normalized pre-attention residual is consumed only by a
                // group of low-bit projections. Preserve residual + one scale per
                // row and let their shared Q8 quantizer apply the norm weight.
                const int virtual_rms_cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
                if (virtual_rms_q8_enabled && virtual_rms_cc == GGML_CUDA_CC_DGX_SPARK && !is_concurrent_event_active &&
                        node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
                    ggml_tensor * rms = cgraph->nodes[i + 1];
                    ggml_tensor * mul = cgraph->nodes[i + 2];
                    const ggml_op ops[] = { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
                    const int out_nodes[] = { i, i + 2 };
                    const ggml_tensor * weight = mul->src[0] == rms ? mul->src[1] : mul->src[0];
                    int consumers = 0;
                    ggml_type consumer_type = GGML_TYPE_COUNT;
                    bool consumers_ok = true;
                    for (int j = i + 3; j < cgraph->n_nodes; ++j) {
                        ggml_tensor * consumer = cgraph->nodes[j];
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            if (consumer->src[s] != mul) {
                                continue;
                            }
                            const bool is_mmq = s == 1 && consumer->op == GGML_OP_MUL_MAT && consumer->src[0] &&
                                (consumer->src[0]->type == GGML_TYPE_Q1_0 ||
                                 consumer->src[0]->type == GGML_TYPE_Q2_0 ||
                                 consumer->src[0]->type == GGML_TYPE_PQ2_0) &&
                                ggml_cuda_should_use_mmq(consumer->src[0]->type, virtual_rms_cc, mul->ne[1], 0);
                            if (!is_mmq || (consumers > 0 && consumer_type != consumer->src[0]->type)) {
                                consumers_ok = false;
                            } else {
                                consumer_type = consumer->src[0]->type;
                                ++consumers;
                            }
                        }
                    }
                    if (consumers_ok && consumers > 1 && rms->op == GGML_OP_RMS_NORM &&
                            mul->op == GGML_OP_MUL && (mul->src[0] == rms || mul->src[1] == rms) &&
                            rms->src[0] == node && node->src[0] && node->src[1] && weight &&
                            node->type == GGML_TYPE_F32 && node->src[0]->type == GGML_TYPE_F32 &&
                            node->src[1]->type == GGML_TYPE_F32 && rms->type == GGML_TYPE_F32 &&
                            mul->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 &&
                            ggml_are_same_shape(node->src[0], node->src[1]) &&
                            ggml_are_same_shape(node, rms) && ggml_is_contiguous(node->src[0]) &&
                            ggml_is_contiguous(node->src[1]) && ggml_is_contiguous(node) &&
                            ggml_is_contiguous(mul) && ggml_is_contiguous(weight) &&
                            weight->ne[0] == node->ne[0] && ggml_nrows(weight) == 1 && node->ne[1] >= 32 &&
                            ggml_can_fuse_subgraph(cgraph, i, 3, ops, out_nodes, 2) &&
                            ggml_cuda_check_fusion_memory_ranges(cgraph, i, 3, out_nodes, 2)) {
                        auto row_scale = std::make_unique<ggml_cuda_pool_alloc<char>>(
                            cuda_ctx->pool(), ggml_nrows(node)*sizeof(float));
                        ggml_cuda_pool_alloc<char> * row_scale_ptr = row_scale.get();
                        gb10_pool_allocations.push_back(std::move(row_scale));
                        ggml_cuda_op_add_rms_norm_scale_fused(
                            *cuda_ctx, node, rms, (float *) row_scale_ptr->get());
                        gb10_virtual_rms.push_back({ mul, node, weight, consumer_type, row_scale_ptr, consumers });
                        i += 2;
                        continue;
                    }
                }

                // Skip full-attn gate CONT when our lossless SIGMOID+MUL+FWHT(1024)->B_i8 will read strided Qcur_full directly!
                if (node->op == GGML_OP_CONT && ggml_cuda_pq2_cublas_i8_enabled() && pq2_b_i8_cache_enabled() &&
                        node->type == GGML_TYPE_F32 && node->ne[0] == 6144 && node->ne[1] >= 129 &&
                        node->src[0] && node->src[0]->op == GGML_OP_VIEW && node->src[0]->view_src &&
                        node->src[0]->ne[0] == 256 && node->src[0]->ne[1] == 24 && node->src[0]->ne[2] == node->ne[1] &&
                        node->src[0]->nb[0] == 4 && node->src[0]->nb[1] == 2048 && node->src[0]->view_offs == 1024 &&
                        ggml_node_get_use_count(cgraph, i) == 1 && (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                    const ggml_tensor * qfull = node->src[0]->view_src;
                    bool can_skip = false;
                    for (int j = i + 1; j + 4 < cgraph->n_nodes && j <= i + 32; ++j) {
                        const ggml_tensor * sj = cgraph->nodes[j];
                        if (sj->op == GGML_OP_UNARY && ggml_get_unary_op(sj) == GGML_UNARY_OP_SIGMOID && sj->src[0] == node &&
                                cgraph->nodes[j + 1]->op == GGML_OP_MUL &&
                                cgraph->nodes[j + 2]->op == GGML_OP_MUL &&
                                cgraph->nodes[j + 3]->op == GGML_OP_RESHAPE &&
                                cgraph->nodes[j + 4]->op == GGML_OP_MUL_MAT &&
                                ggml_get_op_params_i32(cgraph->nodes[j + 4], 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                                ggml_node_get_use_count(cgraph, j) == 1 &&
                                ggml_node_get_use_count(cgraph, j + 1) == 1 &&
                                ggml_node_get_use_count(cgraph, j + 2) == 1 &&
                                ggml_node_get_use_count(cgraph, j + 3) == 1) {
                            can_skip = true;
                            for (int k = i + 1; k < j; ++k) {
                                const ggml_tensor * tk = cgraph->nodes[k];
                                if (!ggml_cuda_is_view_or_noop(tk) && tk->data == qfull->data) {
                                    can_skip = false; break;
                                }
                            }
                            break;
                        }
                    }
                    if (can_skip) {
                        tl_skipped_gate_cont_i8[node] = qfull;
                        continue;
                    }
                }

                const uint64_t ni8_gen_before = pq2_narrow_i8_scratch_for_stream(cuda_ctx->stream()).gen;
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    const auto & ni8_post = pq2_narrow_i8_scratch_for_stream(cuda_ctx->stream());
                    if (ni8_post.gen == ni8_gen_before) {
                        for (int f = 1; f <= nodes_to_skip; ++f) {
                            pq2_narrow_i8_cache_invalidate_if_overlap(cuda_ctx->stream(), cgraph->nodes[i + f]->data, ggml_nbytes(cgraph->nodes[i + f]));
                        }
                    }
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }

                // Several Qwen3.5 attention projections consume the exact same
                // normalized activation. Quantize it once per graph execution and
                // reuse the Q8 tile for the later MMQs on the same CUDA stream.
                const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
                if (shared_q8_enabled && cc == GGML_CUDA_CC_DGX_SPARK && !is_concurrent_event_active &&
                        node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[1] &&
                        node->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(node->src[1]) &&
                        node->src[1]->ne[1] >= 32 &&
                        (node->src[0]->type == GGML_TYPE_Q1_0 ||
                         node->src[0]->type == GGML_TYPE_Q2_0 ||
                         node->src[0]->type == GGML_TYPE_PQ2_0) &&
                        ggml_cuda_should_use_mmq(node->src[0]->type, cc, node->src[1]->ne[1], 0) &&
                        gb10_shared_q8_consumer_count(node->src[1], node->src[0]->type) > 1) {
                    auto virtual_it = std::find_if(gb10_virtual_rms.begin(), gb10_virtual_rms.end(), [&](const auto & entry) {
                        return entry.normalized == node->src[1] && entry.type == node->src[0]->type;
                    });
                    const bool virtual_rms = virtual_it != gb10_virtual_rms.end() && virtual_it->row_scale;
                    auto it = std::find_if(gb10_shared_q8.begin(), gb10_shared_q8.end(), [&](const auto & entry) {
                        return entry.src1 == node->src[1] && entry.type == node->src[0]->type;
                    });
                    if (it == gb10_shared_q8.end()) {
                        auto data = std::make_unique<ggml_cuda_pool_alloc<char>>(cuda_ctx->pool(),
                                ggml_cuda_mul_mat_q_q8_size(node->src[0], node->src[1]));
                        ggml_cuda_pool_alloc<char> * data_ptr = data.get();
                        gb10_pool_allocations.push_back(std::move(data));
                        gb10_shared_q8.push_back({ node->src[1], node->src[0]->type, data_ptr, false,
                                gb10_shared_q8_consumer_count(node->src[1], node->src[0]->type) });
                        it = std::prev(gb10_shared_q8.end());
                    }
                    ggml_cuda_mul_mat_q(*cuda_ctx, node->src[0], virtual_rms ? virtual_it->residual : node->src[1],
                            nullptr, node, nullptr, virtual_rms ? virtual_it->weight : nullptr, nullptr,
                            it->data->get(), !it->quantized,
                            virtual_rms ? (float *) virtual_it->row_scale->get() : nullptr);
                    it->quantized = true;
                    --it->remaining;
                    if (virtual_rms) {
                        --virtual_it->remaining;
                    }
                    if (!is_concurrent_event_active) {
                        try_launch_concurrent_event(node);
                    }
                    continue;
                }

#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!ggml_cuda_tail_verify_pending().empty()) {
                    auto it = ggml_cuda_tail_verify_pending().find(node);
                    if (it != ggml_cuda_tail_verify_pending().end()) {
                        ggml_cuda_op_tail_verify_cmp(*cuda_ctx, node, it->second.scratch, it->second.tag);
                        if (it->second.q8_fused) {
                            const ggml_cuda_tail_verify_entry & ve = it->second;
                            quantize_row_q8_1_cuda((const float *) ve.norm_out->data, nullptr, ve.q8_ref, GGML_CUDA_Q8_1_AOS, ve.norm_out->ne[0],
                                                   ve.norm_out->nb[1] / sizeof(float), 0, 0, ve.ne0_pad, ve.nrows, 1, 1, cuda_ctx->stream());
                            ggml_cuda_op_tail_verify_cmp_q8(*cuda_ctx, ve.q8_ref, ve.q8_fused, (size_t) (ve.nrows * ve.ne0_pad) * sizeof(block_q8_1) / QK8_1, ve.tag);
                        }
                        static int nlaunch = 0;
                        ++nlaunch;
                        if ((nlaunch & (nlaunch - 1)) == 0) fprintf(stderr, "TAILVERIFY compare launches captured/issued: %d\n", nlaunch);
                    }
                }

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }

            if (timing) {
                ggml_cuda_op_timing_of(cuda_ctx->device).end_graph(cuda_ctx->stream(cuda_ctx->device, 0));
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);

            // drop the chunks left over from a longer graph
            for (size_t i = n_chunks; i < graph->instances.size(); ++i) {
                CUDA_CHECK(cudaGraphExecDestroy(graph->instances[i]));
            }
            graph->instances.resize(n_chunks);
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph && !cuda_graph_update_required) {
        // the chunks were launched during capture, otherwise launch them here
        for (cudaGraphExec_t instance : graph->instances) {
            CUDA_CHECK(cudaGraphLaunch(instance, cuda_ctx->stream()));
        }
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }

    // The VMM scratch pool is stack-like, so release all persistent allocations
    // explicitly in reverse order across both shared-Q8 and row-scale buffers.
    for (auto it = gb10_pool_allocations.rbegin(); it != gb10_pool_allocations.rend(); ++it) {
        it->reset();
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->instances.empty()) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch && ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        // graph->uid is only set for a compatible graph (ggml_cuda_graph_update_required), and an unchanged uid means
        // the same nodes: skip the scan of the nodes
        const bool graph_compatible = (cgraph->uid != 0 && cgraph->uid == graph->uid) || ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    }
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    }
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instances.empty();
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Pre-warm PQ2_0 narrow INT8, prefill INT4, and cuBLAS INT8 caches & scratch buffers BEFORE CUDA graph capture
        // so cudaMalloc/cudaFuncSetAttribute never execute during capture and large weights (e.g. lm_head) hit cache.
        if (turing_mma_available(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
            cudaStream_t st = cuda_ctx->stream();
            const bool en_ni8 = ggml_cuda_pq2_narrow_int8_enabled();
            const bool en_i4  = ggml_cuda_pq2_prefill_int4_enabled();
            const bool en_cb  = ggml_cuda_pq2_cublas_i8_enabled();
            auto & sc_ni8 = pq2_narrow_i8_scratch_for_stream(st);
            auto & sc_i4  = pq2_prefill_i4_scratch_for_stream(st);
            auto & sc_cb  = pq2_b_i8_cache_for_stream(st);
            size_t max_i4_b = 0, max_i4_sc = 0, max_i4_gate = 0;
            size_t max_cb_b = 0, max_cb_sc = 0;
            bool need_i4_attr = false;
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * nd = cgraph->nodes[i];
                if (nd->op == GGML_OP_MUL_MAT && nd->src[0] && nd->src[1] &&
                        nd->src[0]->type == GGML_TYPE_PQ2_0 && nd->src[1]->type == GGML_TYPE_F32 &&
                        nd->src[0]->ne[2] == 1 && nd->src[0]->ne[3] == 1 &&
                        nd->src[1]->ne[2] == 1 && nd->src[1]->ne[3] == 1 &&
                        ggml_is_contiguous(nd->src[0]) &&
                        ggml_backend_buffer_get_usage(nd->src[0]->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                    const int64_t K = nd->src[0]->ne[0], M = nd->src[0]->ne[1], N = nd->src[1]->ne[1];
                    if (en_ni8 && N >= 5 && N <= pq2_narrow_i8_max_n() &&
                            M <= (N >= 256 ? 65536 : 262144) && (K % 128) == 0 && (M % 8) == 0) {
                        (void) get_or_repack_pq2_tc_i8((const block_pq2_0 *) nd->src[0]->data, M, K, st, true);
                        (void) pq2_narrow_i8_alloc_bufs(sc_ni8, pq2_narrow_i8_n_pad(N), K, st);
                    } else if (en_i4 && N >= 256 && M <= 65536 && (K % 256) == 0 && (M % 128) == 0) {
                        (void) get_or_repack_pq2_s4_rm((const block_pq2_0 *) nd->src[0]->data, M, K, st, true);
                        const int64_t N_pad = ((N + 127) / 128) * 128;
                        max_i4_b    = std::max(max_i4_b,    (size_t) N_pad * (size_t) (K >> 3) * sizeof(uint32_t));
                        max_i4_sc   = std::max(max_i4_sc,   (size_t) N_pad * sizeof(float));
                        max_i4_gate = std::max(max_i4_gate, (size_t) N * (size_t) M * sizeof(half));
                        need_i4_attr = true;
                    } else if (en_cb && (K % QK_PQ2_0) == 0 && (M % 4) == 0) {
                        const int64_t min_n_cb = en_ni8 ? (M > 65536 ? 256 : (pq2_narrow_i8_max_n() + 1)) : 129;
                        if (N >= min_n_cb) {
                            (void) get_or_compute_pq2_row_scales((const block_pq2_0 *) nd->src[0]->data, M, K, st, true);
                            const int64_t N_pad = (N + 3) & ~int64_t(3);
                            max_cb_b  = std::max(max_cb_b,  (size_t) N_pad * (size_t) K);
                            max_cb_sc = std::max(max_cb_sc, (size_t) N_pad * sizeof(float));
                        }
                    }
                }
            }
            if (need_i4_attr) {
                CUDA_CHECK(cudaFuncSetAttribute(pq2_prefill_s4_128x128x256_ldm_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
                CUDA_CHECK(cudaFuncSetAttribute(pq2_prefill_s4_128x128x256_ldm_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
                CUDA_CHECK(cudaFuncSetAttribute(pq2_prefill_s4_128x128x256_ldm_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
                CUDA_CHECK(cudaFuncSetAttribute(pq2_prefill_s4_128x128x256_ldm_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
            }
            if (max_i4_b > sc_i4.cap_b) {
                if (sc_i4.B_i4) { CUDA_CHECK(cudaFree(sc_i4.B_i4)); }
                CUDA_CHECK(cudaMalloc(&sc_i4.B_i4, max_i4_b));
                sc_i4.cap_b = max_i4_b; sc_i4.valid = false;
            }
            if (max_i4_sc > sc_i4.cap_sc) {
                if (sc_i4.B_scales) { CUDA_CHECK(cudaFree(sc_i4.B_scales)); }
                CUDA_CHECK(cudaMalloc(&sc_i4.B_scales, max_i4_sc));
                sc_i4.cap_sc = max_i4_sc; sc_i4.valid = false;
            }
            if (max_i4_gate > sc_i4.cap_gate) {
                if (sc_i4.gate_f16) { CUDA_CHECK(cudaFree(sc_i4.gate_f16)); }
                CUDA_CHECK(cudaMalloc(&sc_i4.gate_f16, max_i4_gate));
                sc_i4.cap_gate = max_i4_gate;
            }
            if (max_cb_b > sc_cb.B_i8_cap) {
                if (sc_cb.B_i8) { CUDA_CHECK(cudaFree(sc_cb.B_i8)); }
                CUDA_CHECK(cudaMalloc(&sc_cb.B_i8, max_cb_b));
                sc_cb.B_i8_cap = max_cb_b; sc_cb.valid = false;
            }
            if (max_cb_sc > sc_cb.col_scales_cap) {
                if (sc_cb.col_scales) { CUDA_CHECK(cudaFree(sc_cb.col_scales)); }
                CUDA_CHECK(cudaMalloc(&sc_cb.col_scales, max_cb_sc));
                sc_cb.col_scales_cap = max_cb_sc; sc_cb.valid = false;
            }
        }
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    const bool disable_fusion = ggml_cuda_fusion_disabled();

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // Keep the inputs of the fused gate/up/GLU MMQ kernel alive until the GLU so that they cannot alias its output (under op offload the weights are buffer copies).
            if (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
                ggml_tensor * glu  = cgraph->nodes[i + 2];
                ggml_tensor * gate = glu->src[0];
                ggml_tensor * up   = glu->src[1];
                const bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);
                if (ok && ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, cuda_ctx->device)) {
                    add_alloc_deps(i, i + 2);
                    i += 2;
                    continue;
                }
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                    if (ggml_cuda_moe_ep_fusion_enabled() && match.expert_scale->op == GGML_OP_GET_ROWS && match.expert_scale->src[1]) {
                        params->add_alloc_dep(
                            params->user_data, const_cast<ggml_tensor *>(match.expert_scale->src[1]), match.dst);
                    }
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    // A single attribute query, not cudaGetDeviceProperties: this is called on hot host
    // paths (graph construction queries the device type per layer) and the full property
    // read costs ~0.7 ms per call, the attribute read ~20 ns.
    int integrated = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, ggml_cuda_get_physical_device(ctx->device)));

    return integrated ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_PQ1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_PQ2_0:
                    case GGML_TYPE_PTQ1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_PQ1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_PQ2_0:
                    case GGML_TYPE_PTQ1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL ||
                               op->type == GGML_TYPE_PQ2_0M) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            }
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            // rows-indexed state read (src[6]) not implemented on CUDA yet;
            // reject so it falls back instead of silently reading src[5] as a scratch
            if (op->src[6] != NULL) {
                return false;
            }
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_GATED_DELTA_NET_REPLAY:
#ifdef GGML_USE_MUSA
            return false;
#else
            return op->src[2] != NULL && op->src[3] != NULL &&
                   op->src[2]->ne[0] == 128 && op->src[3]->ne[0] == 1;
#endif
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_TOP_K_MASK:
            return op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

__global__ static void cuda_argmax_f32_rows_kernel(
        const float * __restrict__ logits, int32_t * __restrict__ out_ids, const int n_vocab) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * src = logits + (int64_t) row * n_vocab;

    float best_v = -INFINITY;
    int   best_i = 0;
    const int n4 = n_vocab >> 2;
    const float4 * src4 = (const float4 *) src;
    for (int i = tid; i < n4; i += 256) {
        const float4 v = src4[i];
        const int base = i << 2;
        if (v.x > best_v || (v.x == best_v && base + 0 < best_i)) { best_v = v.x; best_i = base + 0; }
        if (v.y > best_v || (v.y == best_v && base + 1 < best_i)) { best_v = v.y; best_i = base + 1; }
        if (v.z > best_v || (v.z == best_v && base + 2 < best_i)) { best_v = v.z; best_i = base + 2; }
        if (v.w > best_v || (v.w == best_v && base + 3 < best_i)) { best_v = v.w; best_i = base + 3; }
    }
    for (int i = (n4 << 2) + tid; i < n_vocab; i += 256) {
        const float v = src[i];
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFF, best_v, off);
        const int   oi = __shfl_xor_sync(0xFFFFFFFF, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[8];
    __shared__ int   si[8];
    if ((tid & 31) == 0) {
        sv[tid >> 5] = best_v;
        si[tid >> 5] = best_i;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = sv[0];
        int   bi = si[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        out_ids[row] = bi;
    }
}

bool ggml_backend_cuda_argmax_f32_rows_async(
        ggml_backend_t backend, const struct ggml_tensor * t_logits,
        int32_t * host_ids, int64_t n_rows, int64_t n_vocab) {
    if (!backend || !t_logits || !host_ids || n_rows <= 0 || n_vocab <= 0) { return false; }
    if (!ggml_backend_is_cuda(backend)) { return false; }
    if (t_logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(t_logits) || !t_logits->data) { return false; }
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx->device);
    cudaStream_t stream = ctx->stream();

    struct argmax_buf { int32_t * d_ids = nullptr; int64_t cap = 0; };
    static std::mutex mu;
    static std::unordered_map<cudaStream_t, argmax_buf> map;
    argmax_buf * ab = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        ab = &map[stream];
    }
    if (ab->cap < n_rows) {
        const int64_t alloc_n = std::max<int64_t>(n_rows, 2048);
        if (ab->d_ids) { CUDA_CHECK(cudaFree(ab->d_ids)); }
        CUDA_CHECK(cudaMalloc(&ab->d_ids, (size_t) alloc_n * sizeof(int32_t)));
        ab->cap = alloc_n;
    }
    cuda_argmax_f32_rows_kernel<<<(unsigned) n_rows, 256, 0, stream>>>(
        (const float *) t_logits->data, ab->d_ids, (int) n_vocab);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(host_ids, ab->d_ids, (size_t) n_rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    return true;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
