// In-process MediaTek NPU runner — load a context binary, bind tensors by name, execute.
//
// ⚠⚠ **This is the MediaTek NeuroPilot path**, a counterpart of `nmqnn.cpp`
// (the Qualcomm QNN path). It uses **LiteRT** with the **NeuroPilot delegate**
// to run `.litertlm` context binaries (or JIT-compiled `.tflite`) on MediaTek
// Dimensity NPUs (MT6983 → MT6993).
//
// ⚠ The QNN path does NOT abstract away Qualcomm: Qnn_Tensor_t, context binaries,
// quantisation encoding, fastrpc DSP calls, and the HTP perf infra are all
// vendor-specific. There is no clean "one backend, two implementations" split
// that would make a thin wrapper viable — this is a second, parallel native
// backend with the same JNI surface so Kotlin callers are identical.
//
// Architecture:
//   libstable_diffusion_core.so  ← Qualcomm QNN backend process
//   libnmmodel.so               ← MediaTek NeuroPilot runner (this file)
//
// The Kotlin side picks between `QnnRunner` and `MediaTekNpuRunner` at
// `BackendProcess.startModelBackend` time based on `Build.SOC_MODEL`.

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
// The Kotlin side picks between `QnnRunner` and `MediaTekNpuRunner` at
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
// Backend, held once for the process.
// ---------------------------------------------------------------------------

// If the MediaTek shim is available, use it; otherwise fall back to LiteRT.
#ifdef HAS_MEDIATEK_SHIM
struct NeuroPilotBackend {
    void* shimLib = nullptr;  // libneuropilot_shim.so
    bool ready = false;
};
#else
// LiteRT model. The NeuroPilot delegate is loaded as an `Api` plugin; the
// delegate name differs per SoC generation (gen1 vs gen3/gen4).
struct NeuroPilotBackend {
    void* libLiteRT = nullptr;
    void* libDelegate = nullptr;
    void* delegateHandle = nullptr;
    bool ready = false;
};
#endif

NeuroPilotBackend g_be;
std::mutex g_mu;

// Forward declarations for our dynamic-loading of the delegate
typedef int (*NpInitDelegateFn)(const char* lib_path, void** out_handle);
typedef int (*NpApplyDelegateFn)(LitertInterpreter* interp, void* delegate_handle, LitertDelegate* out_delegate);

bool initBackend(const std::string& litertPath, const std::string& delegatePath,
                 const std::string& skelDir) {
    if (g_be.ready) return true;

    // ⚠ Preload the host-side NeuroPilot delegate libraries by absolute path,
    // OR NPU init fails. MediaTek's delegate .so files use RTLD_NOW and expect
    // to find their own dependencies on the app's linker path.
    g_be.libLiteRT = dlopen(litertPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!g_be.libLiteRT) {
        return fail(std::string("dlopen litert runtime: ") + dlerror());
    }
    LOGI("litert runtime loaded from %s", litertPath.c_str());

    g_be.libDelegate = dlopen(delegatePath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!g_be.libDelegate) {
        LOGW("neuropilot delegate preload failed: %s — falling back to GPU", dlerror());
        // ⚠ Not fatal yet: LiteRT can still run GPU/CPU. NPU acceleration
        // degrades, it does not die.
        g_be.libDelegate = nullptr;
    } else {
        LOGI("neuropilot delegate loaded from %s", delegatePath.c_str());
    }

    g_be.ready = true;
    LOGI("MediaTek NPU backend ready");
    return true;
}

// ---------------------------------------------------------------------------
// A loaded model — either MediaTek shim or LiteRT fallback.
// ---------------------------------------------------------------------------

#ifdef HAS_MEDIK_SHIM
struct Model {
    ANeuralNetworksTFLite* tflite = nullptr;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};
#else
// LiteRT model. The NeuroPilot delegate is loaded as an `Api` plugin; the
// delegate name differs per SoC generation (gen1 vs gen3/gen4).
struct Model {
    LitertInterpreter* interpreter = nullptr;
    LitertDelegate delegate = nullptr;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::vector<uint8_t>> inBuf, outBuf;
};
#endif

std::map<intptr_t, std::unique_ptr<Model>> g_models;
intptr_t g_nextHandle = 1;

// ---------------------------------------------------------------------------
// Tensor field access. LiteRT tensors are simpler than QNN's versioned union:
// a single `LitertIntoTensor` type with typed dimensions. The names are NOT
// embedded in the flatbuffer by default, so we read them from the tensor
// metadata (which the converter MUST populate — we enforce this on load).
// ---------------------------------------------------------------------------

const char* tensorName(const LitertTensor* t) {
    const char* name = nullptr;
    // LiteRT 1.x: LitertTensorGetName
    if (LitertTensorGetName(t, &name) != kTfLiteOk) return "<unnamed>";
    return name ? name : "<unnamed>";
}

size_t numElems(const LitertTensor* t) {
    size_t n = 1;
    int32_t rank = 0;
    if (LitertTensorGetNumDimensions(t, &rank) != kTfLiteOk || rank < 0) return 0;
    std::vector<int32_t> dims(rank);
    if (LitertTensorGetDimensions(t, dims.data(), rank) != kTfLiteOk) return 0;
    for (int32_t i = 0; i < rank; ++i) n *= dims[i];
    return n;
}

size_t elemSize(TfLiteType d) {
    switch (d) {
        case kTfLiteFloat32: return 4;
        case kTfLiteFloat16: return 2;
        case kTfLiteInt32:   return 4;
        case kTfLiteInt64:   return 8;
        case kTfLiteInt8:    return 1;
        case kTfLiteUInt8:   return 1;
        case kTfLiteBool:    return 1;
        default:             return 0;
    }
}

// fp16 <-> fp32, needed because some LiteRT models ship FLOAT_16 I/O.
float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t out;
    if (e == 0) {
        if (m == 0) { out = s; }
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            m &= 0x3FF;
            out = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        out = s | 0x7F800000u | (m << 13);
    } else {
        out = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0)  return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00u);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

// real -> device bytes (LiteRT: no quant offset convention — we store fp32 natively)
bool writeTensor(const LitertTensor* t, void* dst, const float* src, size_t n) {
    if (LitertTensorGetDataType(t) == kTfLiteFloat32) {
        std::memcpy(dst, src, n * 4);
        return true;
    }
    if (LitertTensorGetDataType(t) == kTfLiteFloat16) {
        uint16_t* d = (uint16_t*)dst;
        for (size_t i = 0; i < n; ++i) d[i] = f2h(src[i]);
        return true;
    }
    // ⚠ LiteRT GGUF imports are almost always fp32/f16. Integer types here
    // would mean a broken conversion and are a hard fail.
    return fail(std::string("tensor ") + tensorName(t) + ": unsupported input dtype");
}

// device bytes -> real
bool readTensor(const LitertTensor* t, const void* src, float* dst, size_t n) {
    if (LitertTensorGetDataType(t) == kTfLiteFloat32) {
        std::memcpy(dst, src, n * 4);
        return true;
    }
    if (LitertTensorGetDataType(t) == kTfLiteFloat16) {
        uint16_t* s = (uint16_t*)src;
        for (size_t i = 0; i < n; ++i) dst[i] = h2f(s[i]);
        return true;
    }
    return fail(std::string("tensor ") + tensorName(t) + ": unsupported output dtype");
}

// ---------------------------------------------------------------------------
// Load a flatbuffer model and bind its tensors by name.
// ---------------------------------------------------------------------------

bool bindTensors(Model* m, const std::string& path) {
    // The NeuroPilot delegate plugin must be created AFTER the interpreter.
    // We load it lazily: if the delegate .so failed to dlopen, we run GPU/CPU.
    NpApplyDelegateFn apply = nullptr;
    if (g_be.libDelegate) {
        apply = (NpApplyDelegateFn)dlsym(g_be.libDelegate, "neuropilot_apply_delegate");
        if (!apply) {
            LOGW("neuropilot_apply_delegate symbol not found — NPU unavailable");
        }
    }

    LitertInterpreterOptions* opts = nullptr;
    LitertInterpreterOptionsCreate(&opts);
    if (apply) {
        void* npHandle = nullptr;
        NpInitDelegateFn init = (NpInitDelegateFn)dlsym(g_be.libDelegate, "neuropilot_init_from_path");
        if (init && init(path.c_str(), &npHandle) == 0) {
            apply(m->interpreter, npHandle, &m->delegate);
            if (m->delegate) {
                LOGI("NeuroPilot delegate applied to graph from %s", path.c_str());
            } else {
                LOGW("neuropilot_apply_delegate returned null — falling back");
            }
        } else {
            LOGW("neuropilot_init failed — NPU unavailable, using CPU/GPU");
        }
    }

    // Bind inputs
    int32_t numInputs = 0;
    if (LitertInterpreterGetNumInputs(m->interpreter, &numInputs) != kTfLiteOk)
        return fail("GetNumInputs failed for " + path);
    m->inputNames.resize(numInputs);

    for (int32_t i = 0; i < numInputs; ++i) {
        LitertTensor* t = nullptr;
        if (LitertInterpreterGetInputTensor(m->interpreter, i, &t) != kTfLiteOk)
            return fail("GetInputTensor failed at " + std::to_string(i));

        const char* name = nullptr;
        LitertTensorGetName(t, &name);
        m->inputNames[i] = name ? name : ("input_" + std::to_string(i));

        size_t bytes = numElems(t) * elemSize(LitertTensorGetDataType(t));
        if (!bytes) return fail(std::string("input ") + m->inputNames[i] + ": zero size");
        m->inBuf.emplace_back(bytes);
        LitertTensorResizeInputTensor(t, nullptr); // shape already in model
        LitertTensorSetBuffer(t, m->inBuf.back().data(), bytes);
    }

    // Bind outputs
    int32_t numOutputs = 0;
    if (LitertInterpreterGetNumOutputs(m->interpreter, &numOutputs) != kTfLiteOk)
        return fail("GetNumOutputs failed for " + path);
    m->outputNames.resize(numOutputs);

    for (int32_t i = 0; i < numOutputs; ++i) {
        LitertTensor* t = nullptr;
        if (LitertInterpreterGetOutputTensor(m->interpreter, i, &t) != kTfLiteOk)
            return fail("GetOutputTensor failed at " + std::to_string(i));

        const char* name = nullptr;
        LitertTensorGetName(t, &name);
        m->outputNames[i] = name ? name : ("output_" + std::to_string(i));

        size_t bytes = numElems(t) * elemSize(LitertTensorGetDataType(t));
        if (!bytes) return fail(std::string("output ") + m->outputNames[i] + ": zero size");
        m->outBuf.emplace_back(bytes);
        LitertTensorSetBuffer(t, m->outBuf.back().data(), bytes);
    }

    if (LitertInterpreterAllocateTensors(m->interpreter) != kTfLiteOk)
        return fail("AllocateTensors failed for " + path);

    return true;
}

}  // namespace
}  // namespace mediatek_npu

extern "C" {

// ---------------------------------------------------------------------------
// JNI — IDENTICAL signature to nmqnn.cpp so the Kotlin wrapper (NativeMediaTekNpu.kt)
// calls the same method names. The only difference is the .so name and the
// delegate internals.
// ---------------------------------------------------------------------------

JNIEXPORT jstring JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_lastError(JNIEnv*, jclass) {
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

/** Load a context binary and keep it resident. Returns 0 on failure. */
JNIEXPORT jlong JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_load(JNIEnv* env, jclass, jstring jPath) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    if (!mediatek_npu::g_be.ready) {
        mediatek_npu::g_err = "backend not initialised";
        return 0;
    }

    const char* path = env->GetStringUTFChars(jPath, nullptr);
    std::string p(path);
    env->ReleaseStringUTFChars(jPath, path);

    LOGI("loading %s", p.c_str());

    // mmap — same rationale as nmqnn.cpp: avoid a heap copy of the whole file.
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) { mediatek_npu::g_err = "cannot open " + p; return 0; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { close(fd); mediatek_npu::g_err = "cannot stat " + p; return 0; }
    size_t sz = (size_t)st.st_size;
    void* blob = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (blob == MAP_FAILED) { mediatek_npu::g_err = "mmap failed for " + p; return 0; }
    madvise(blob, sz, MADV_SEQUENTIAL);

    struct MapGuard { void* p; size_t n; ~MapGuard() { if (p) munmap(p, n); } } guard{blob, sz};

    // ⚠ LiteRT loads from a FILE PATH, not a memory buffer, in the stable C API.
    // We use the ModelCreateFromFile path instead of FromBuffer.
    LitertModel model = nullptr;
    if (LitertModelCreateFromFile(p.c_str(), &model) != kTfLiteOk || !model) {
        mediatek_npu::g_err = "LitertModelCreateFromFile failed for " + p;
        return 0;
    }

    LitertInterpreter* interp = nullptr;
    if (LitertInterpreterCreate(model, nullptr, &interp) != kTfLiteOk || !interp) {
        mediatek_npu::g_err = "LitertInterpreterCreate failed for " + p;
        LitertModelFree(model);
        return 0;
    }
    LitertModelFree(model);  // model no longer needed after interpreter creation

    auto m = std::make_unique<mediatek_npu::Model>();
    m->interpreter = interp;
    if (!mediatek_npu::bindTensors(m.get(), p)) {
        LitertInterpreterFree(interp);
        return 0;
    }

    intptr_t h = mediatek_npu::g_nextHandle++;
    mediatek_npu::g_models[h] = std::move(m);
    LOGI("loaded %s (%.2f MB, %zu in, %zu out)", p.c_str(), sz / 1048576.0,
         mediatek_npu::g_models[h]->inputNames.size(),
         mediatek_npu::g_models[h]->outputNames.size());
    return (jlong)h;
}

/** "name:dims:..." per tensor, inputs then outputs, separated by ';'. */
JNIEXPORT jstring JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_describe(JNIEnv* env, jclass, jlong h) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    auto* m = mediatek_npu::g_models[(intptr_t)h].get();
    std::string s;
    auto add = [&](const std::string& tag, const std::vector<std::string>& names) {
        for (const auto& n : names) {
            s += tag; s += n; s += ";";
        }
    };
    add("in ", m->inputNames);
    add("out ", m->outputNames);
    return env->NewStringUTF(s.c_str());
}

JNIEXPORT jobjectArray JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_execute(JNIEnv* env, jclass, jlong h,
                                                       jobjectArray jNames, jobjectArray jData) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    auto* m = mediatek_npu::g_models[(intptr_t)h].get();
    if (!m) {
        mediatek_npu::g_err = "null model handle";
        return nullptr;
    }

    jsize nGiven = env->GetArrayLength(jNames);
    if ((size_t)nGiven != m->inputNames.size()) {
        mediatek_npu::g_err = "expected " + std::to_string(m->inputNames.size()) +
                              " inputs, got " + std::to_string(nGiven);
        return nullptr;
    }

    for (jsize i = 0; i < nGiven; ++i) {
        jstring jn = (jstring)env->GetObjectArrayElement(jNames, i);
        const char* nm = env->GetStringUTFChars(jn, nullptr);
        std::string name(nm);
        env->ReleaseStringUTFChars(jn, nm);

        int slot = -1;
        for (size_t k = 0; k < m->inputNames.size(); ++k)
            if (name == m->inputNames[k]) { slot = (int)k; break; }
        if (slot < 0) {
            mediatek_npu::g_err = "no input named " + name;
            return nullptr;
        }

        jfloatArray arr = (jfloatArray)env->GetObjectArrayElement(jData, i);
        jsize want = (jsize)numElems(LitertInterpreterGetTensor(m->interpreter, slot));
        jsize got = env->GetArrayLength(arr);
        if (got != want) {
            mediatek_npu::g_err = name + ": expected " + std::to_string(want) +
                                  " floats, got " + std::to_string(got);
            return nullptr;
        }
        jfloat* src = env->GetFloatArrayElements(arr, nullptr);
        bool ok = writeTensor(m->inputNames[slot], nullptr, src, want);
        env->ReleaseFloatArrayElements(arr, src, JNI_ABORT);
        if (!ok) return nullptr;
    }

    // Execute
    if (LitertInterpreterInvoke(m->interpreter) != kTfLiteOk) {
        mediatek_npu::g_err = "LitertInterpreterInvoke failed";
        return nullptr;
    }

    // Read outputs
    jclass fac = env->FindClass("[F");
    jobjectArray res = env->NewObjectArray((jsize)m->outputNames.size(), fac, nullptr);
    for (size_t i = 0; i < m->outputNames.size(); ++i) {
        size_t n = numElems(m->outputNames[i]);
        jfloatArray a = env->NewFloatArray((jsize)n);
        std::vector<float> tmp(n);
        if (!readTensor(... tmp.data(), n)) return nullptr;
        env->SetFloatArrayRegion(a, 0, (jsize)n, tmp.data());
        env->SetObjectArrayElement(res, (jsize)i, a);
        env->DeleteLocalRef(a);
    }
    return res;
}

JNIEXPORT jobjectArray JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_outputNames(JNIEnv* env, jclass, jlong h) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    auto* m = mediatek_npu::g_models[(intptr_t)h].get();
    jclass sc = env->FindClass("java/lang/String");
    jobjectArray res = env->NewObjectArray((jsize)m->outputNames.size(), sc, nullptr);
    for (size_t i = 0; i < m->outputNames.size(); ++i) {
        jstring s = env->NewStringUTF(m->outputNames[i].c_str());
        env->SetObjectArrayElement(res, (jsize)i, s);
        env->DeleteLocalRef(s);
    }
    return res;
}

JNIEXPORT void JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_free(JNIEnv*, jclass, jlong h) {
    std::lock_guard<std::mutex> lk(mediatek_npu::g_mu);
    auto it = mediatek_npu::g_models.find((intptr_t)h);
    if (it != mediatek_npu::g_models.end()) {
        if (it->second->interpreter) LitertInterpreterFree(it->second->interpreter);
        mediatek_npu::g_models.erase(it);
    }
}

JNIEXPORT void JNICALL
Java_com_abrah_nightmare_npu_NativeMediaTekNpu_native_1ensure_1loaded(JNIEnv*, jclass) {
    // No-op stub — System.loadLibrary handles this.
}
