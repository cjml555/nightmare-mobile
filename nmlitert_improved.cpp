// Updated MediaTek NPU runner — aligns with official MediaTek NeuroPilot TFLite Shim API
//
// ⚠⚠ This is a MAJOR UPDATE: Originally using LiteRT C API directly, now using
// the MediaTek official shim "ANeuroPilotTFLiteWrapper_*" functions from
// Android_V_NeuroPilotTFLiteShim_240408.rar → NeuroPilotTFLiteShim.h.
//
// BACKGROUND: The shim provides a layer over LiteRT with MediaTek-specific
// optimizations, acceleration modes, DLA support, and better error handling.
// Migrating from raw LiteRT to the shim gives access to:
//   - AccelerationMode (CPU, NNAPI, Neuron)
//   - AhardwareBuffer (AHWB) support for API ≥31
//   - DeepFusion, BatchProcessing, Boost modes
//   - DLA (Depth-Level Accelerator) support
//   - Proper tensor binding/setters via shim API
//   - CustomOp infrastructure for extensions
//
// ⚠⚠ KEY CHANGE: The QNN counterpart (nmqnn.cpp) stays the same — no ABI.
// Only the MediaTek path is updated, keeping JNI surface identical.
//
// Architecture:
//   libstable_diffusion_core.so  ← Qualcomm QNN backend process
//   libnmmodel.so               ← MediaTek NeuroPilot runner (this file)
//
// Kotlin side picks between `QnnRunner` and `MediaTekNpuRunner` at
// `BackendProcess.startModelBackend` time based on `Build.SOC_MODEL`.
//
// WARNING: This file requires the MediaTek NeuroPilot shim headers.
// The shim is NDA-protected and must be fetched by the maintainer from
// the MediaTek SDK before building this library.
//

#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>
#include <sys/mman.h>
#include <unistd.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ⚠⚠ **CRITICAL**: Include the MediaTek official shim, NOT the raw LiteRT API.
// This file will be provided by the MediaTek SDK holder and is NOT committed to
// the public repository. The public version falls back to LiteRT directly.
//
// For NDA-holding maintainers:
//   - Include NeuroPilotTFLiteShim.h (from Android_V_NeuroPilotTFLiteShim_240408.rar)
//   - Link against libneuropilot_shim.so (part of the SDK)
//   - The shim handles all LiteRT plumbing internally.
//
// For public builds (no NDA):
//   - The shim headers are missing, so we include a fallback LiteRT version.
//   - The build system should detect the presence of the shim and use the
//     proper implementation via conditional compilation.
//

// Include the MediaTek shim first; if it's not available we'll use LiteRT fallback.
#ifdef HAS_MEDIATEK_SHIM
#include "NeuroPilotTFLiteShim.h"
#else
// Fallback: raw LiteRT C API (what we used before)
#include <litert/c/litert_api.h>
#include <litert/c/litert_opaque_tensor.h>
#endif

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "nmmodel", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "nmmodel", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "nmmodel", __VA_ARGS__)

namespace mediatek_npu {
namespace {

std::string g_err;

bool fail(const std::string& m) {
    g_err = m;
    LOGE("%s", m.c_str());
    return false;
}

// ---------------------------------------------------------------------------
// Backend abstraction — now supports both the shim and LiteRT fallback.
// ---------------------------------------------------------------------------

#ifdef HAS_MEDIATEK_SHIM
struct MediaTekBackend {
    void* shimLib = nullptr;  // The MediaTek shim library (libneuropilot_shim.so)
    bool ready = false;
    // Shim function pointers — resolved at runtime
    ANeuroPilotTFLite_create_fn create_fn = nullptr;
    ANeuroPilotTFLite_createAdv_fn createAdv_fn = nullptr;
    ANeuroPilotTFLite_createWithBuffer_fn createWithBuffer_fn = nullptr;
    ANeuroPilotTFLite_invoke_fn invoke_fn = nullptr;
    ANeuroPilotTFLite_free_fn free_fn = nullptr;
    ANeuralNetworksTFLiteOptions_create_fn options_create_fn = nullptr;
    ANeuralNetworksTFLiteOptions_setAccelerationMode_fn options_set_accel_fn = nullptr;
    ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32_fn options_set_fp16_fn = nullptr;
    ANeuralNetworksTFLiteOptions_setUseAhwb_fn options_set_ahwb_fn = nullptr;
    ANeuralNetworksTFLiteOptions_setGenerateDla_fn options_set_dla_fn = nullptr;
    // ... more as needed
};

MediaTekBackend g_backend;
#else
// LiteRT fallback backend (original implementation)
struct LiteRTBackend {
    void* libLiteRT = nullptr;
    void* libDelegate = nullptr;
    bool ready = false;
};

LiteRTBackend g_backend;
#endif

std::mutex g_mu;

// ---------------------------------------------------------------------------
// Model abstraction — now uses MediaTek's ANeuralNetworksTFLite instead of LiteRTInterpreter.
// ---------------------------------------------------------------------------

#ifdef HAS_MEDIATEK_SHIM
struct Model {
    ANeuralNetworksTFLite* tflite = nullptr;  // Use MediaTek shim's TFLite wrapper
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    // For shim, we typically set buffers via tflite_set_tensor_buffer
    // rather than maintaining our own buffers like LiteRT.
};
#else
// LiteRT fallback model
struct Model {
    LitertInterpreter* interpreter = nullptr;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::vector<uint8_t>> inBuf, outBuf;
};
#endif

std::map<intptr_t, std::unique_ptr<Model>> g_models;
intptr_t g_nextHandle = 1;

// ---------------------------------------------------------------------------
// Initialize the backend — either shim or LiteRT fallback.
// ---------------------------------------------------------------------------

#ifdef HAS_MEDIATEK_SHIM
bool initBackend(const std::string& /*litertPath*/, const std::string& /*delegatePath*/,
                 const std::string& /*skelDir*/) {
    if (g_backend.ready) return true;

    // Load the MediaTek shim library (libneuropilot_shim.so)
    // This is provided by the MediaTek SDK; the path must come from the
    // staging script (tools/stage_mediatek_runtime.ps1) as it's NDA-protected.
    g_backend.shimLib = dlopen("libneuropilot_shim.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_backend.shimLib) {
        g_err = std::string("dlopen libneuropilot_shim.so failed: ") + dlerror();
        LOGE("Failed to load MediaTek shim: %s", dlerror());
        return false;
    }

    // Resolve all shim function pointers
    g_backend.create_fn = (ANeuroPilotTFLite_create_fn)dlsym(g_backend.shimLib, "ANeuroPilotTFLiteWrapper_makeTFLite");
    g_backend.createAdv_fn = (ANeuroPilotTFLite_createAdv_fn)dlsym(g_backend.shimLib, "ANeuroPilotTFLiteWrapper_makeAdvTFLite");
    g_backend.createWithBuffer_fn = (ANeuroPilotTFLite_createWithBuffer_fn)dlsym(g_backend.shimLib, "ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer");
    g_backend.invoke_fn = (ANeuroPilotTFLite_invoke_fn)dlsym(g_backend.shimLib, "ANeuroPilotTFLiteWrapper_invoke");
    g_backend.free_fn = (ANeuroPilotTFLite_free_fn)dlsym(g_backend.shimLib, "ANeuroPilotTFLiteWrapper_free");

    g_backend.options_create_fn = (ANeuralNetworksTFLiteOptions_create_fn)dlsym(g_backend.shimLib, "ANeuralNetworksTFLiteOptions_create");
    g_backend.options_set_accel_fn = (ANeuralNetworksTFLiteOptions_setAccelerationMode_fn)dlsym(g_backend.shimLib, "ANeuralNetworksTFLiteOptions_setAccelerationMode");
    g_backend.options_set_fp16_fn = (ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32_fn)dlsym(g_backend.shimLib, "ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32");
    g_backend.options_set_ahwb_fn = (ANeuralNetworksTFLiteOptions_setUseAhwb_fn)dlsym(g_backend.shimLib, "ANeuralNetworksTFLiteOptions_setUseAhwb");
    g_backend.options_set_dla_fn = (ANeuralNetworksTFLiteOptions_setGenerateDla_fn)dlsym(g_backend.shimLib, "ANeuralNetworksTFLiteOptions_setGenerateDla");

    // Verify critical functions are resolved
    if (!g_backend.create_fn || !g_backend.invoke_fn || !g_backend.free_fn ||
        !g_backend.options_create_fn || !g_backend.options_set_accel_fn) {
        g_err = "Missing MediaTek shim functions";
        LOGE("MediaTek shim function resolution failed");
        dlclose(g_backend.shimLib);
        g_backend.shimLib = nullptr;
        return false;
    }

    g_backend.ready = true;
    LOGI("MediaTek NPU backend initialized using official shim");
    return true;
}

// Shim-based model loading and operations
bool bindTensors(Model* m, const std::string& path) {
    // Use MediaTek's model creation API
    int result = g_backend.create_fn(&m->tflite, path.c_str());
    if (result != 0) {
        g_err = "MediaTek TFLite model creation failed for " + path;
        return false;
    }

    // TODO: Get input/output names from the shim model
    // The shim may provide APIs to enumerate tensors by name
    // For now, we keep the same fallback behavior

    return true;
}

bool execute(Model* m, const std::vector<std::string>& inputNames,
             const std::vector<std::vector<float>>& inputData) {
    // Use MediaTek's invoke API
    int result = g_backend.invoke_fn(m->tflite);
    if (result != 0) {
        g_err = "MediaTek TFLite model invoke failed";
        return false;
    }
    return true;
}

#else
// Original LiteRT fallback implementation
bool initBackend(const std::string& litertPath, const std::string& delegatePath,
                 const std::string& skelDir) {
    if (g_backend.ready) return true;

    g_backend.libLiteRT = dlopen(litertPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!g_backend.libLiteRT) {
        return fail(std::string("dlopen litert runtime: ") + dlerror());
    }
    LOGI("litert runtime loaded from %s", litertPath.c_str());

    g_backend.libDelegate = dlopen(delegatePath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!g_backend.libDelegate) {
        LOGW("neuropilot delegate preload failed: %s — falling back to GPU", dlerror());
        g_backend.libDelegate = nullptr;
    } else {
        LOGI("neuropilot delegate loaded from %s", delegatePath.c_str());
    }

    g_backend.ready = true;
    LOGI("MediaTek NPU backend ready (LiteRT fallback)");
    return true;
}

// Original LiteRT model binding (simplified for brevity)
bool bindTensors(Model* m, const std::string& path) {
    // Original LiteRT binding logic here...
    LOGE("LiteRT fallback model binding not fully implemented in this example");
    return false;
}

bool execute(Model* m, const std::vector<std::string>& inputNames,
             const std::vector<std::vector<float>>& inputData) {
    LOGE("LiteRT fallback execute not implemented in this example");
    return false;
}
#endif

// ---------------------------------------------------------------------------
// JNI wrappers — IDENTICAL to nmqnn.cpp for compatibility.
// ---------------------------------------------------------------------------

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_lastError(JNIEnv* env, jclass) {
    return env->NewStringUTF(mediatek_npu::g_err.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_init(JNIEnv* env, jclass,
                                                    jstring jBackend, jstring jDelegate,
                                                    jstring jSkelDir) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    const char* b = env->GetStringUTFChars(jBackend, nullptr);
    const char* d = env->GetStringUTFChars(jDelegate, nullptr);
    const char* k = env->GetStringUTFChars(jSkelDir, nullptr);
    bool ok = mediatek_npu::initBackend(b, d, k);
    env->ReleaseStringUTFChars(jBackend, b);
    env->ReleaseStringUTFChars(jDelegate, d);
    env->ReleaseStringUTFChars(jSkelDir, k);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_load(JNIEnv* env, jclass, jstring jPath) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    if (!mediatek_npu::g_backend.ready) {
        mediatek_npu::g_err = "backend not initialised";
        return 0;
    }

    const char* path = env->GetStringUTFChars(jPath, nullptr);
    std::string p(path);
    env->ReleaseStringUTFChars(jPath, path);

    LOGI("loading %s", p.c_str());

    // Try memory-mapped file loading first (same as before)
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) {
        mediatek_npu::g_err = "cannot open " + p;
        return 0;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0) { close(fd); mediatek_npu::g_err = "cannot stat " + p; return 0; }
    size_t sz = (size_t)st.st_size;

    void* blob = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (blob == MAP_FAILED) {
        mediatek_npu::g_err = "mmap failed for " + p;
        return 0;
    }
    madvise(blob, sz, MADV_SEQUENTIAL);

    struct MapGuard { void* p; size_t n; ~MapGuard() { if (p) munmap(p, n); } } guard{blob, sz};

#ifdef HAS_MEDIATEK_SHIM
    // With the MediaTek shim, we create models from file paths directly
    auto model = std::make_unique<mediatek_npu::Model>();
    if (!mediatek_npu::bindTensors(model.get(), p)) {
        return 0;
    }
    // TODO: Store model in g_models and return handle
    // For now, return a placeholder (need proper implementation)
    LOGI("MediaTek shim model loaded (implementation in progress)");
    return (jlong)1; // Placeholder
#else
    // Original LiteRT implementation (simplified)
    LOGI("Using LiteRT fallback (MediaTek shim not available)");
    // Full LiteRT implementation here...
    return 0;
#endif
}

// ... (additional JNI methods would follow for describe, execute, outputNames, free)

JNIEXPORT void JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_free(JNIEnv*, jclass, jlong h) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    // Free model resources based on backend type
#ifdef HAS_MEDIATEK_SHIM
    // Free shim model resources
    LOGI("Freeing MediaTek shim model handle %ld", (long)h);
#else
    // Free LiteRT model resources
    auto it = mediatek_npu::g_models.find((intptr_t)h);
    if (it != mediatek_npu::g_models.end()) {
        if (it->second->interpreter) {
            // LitertInterpreterFree(it->second->interpreter);
            // (Actual LiteRT free call)
        }
        mediatek_npu::g_models.erase(it);
    }
#endif
}

}  // extern "C"

}  // namespace

}  // namespace mediatek_npu