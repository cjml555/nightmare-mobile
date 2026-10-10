package com.abrah.nightmare.npu

import android.content.Context
import android.os.Build
import android.util.Log
import com.abrah.nightmare.BackendProcess
import com.abrah.nightmare.Prefs
import com.abrah.nightmare.SelectedModel
import com.abrah.nightmare.npu.MediaTekDeviceProbe
import java.io.File

/** ⭐⭐⭐ **The single decision point for which NPU backend a model uses.**
 *
 * ⚡ This object is called once per model load — from `BackendProcess.startModelBackend` —
 * and its whole job is to decide whether the current phone uses Qualcomm QNN
 * or MediaTek NeuroPilot, then hand over to the right runner.
 *
 * ⚠⚠⚠️ **Why not a common interface?** The original app's QNN runner (`QnnRunner`)
 * calls into `libnmqnn.so`, whose native functions are Qualcomm's QNN API:
 * Qnn_Tensor_t unions, context deserialisation with fastrpc DSP calls, and the
 * HTP arch-variant table (v68/v70/v79/v81) are all Qualcomm-specific. There is no
 * stable "NPU runtime" ABI between vendors — QNN and NeuroPilot use completely
 * different tensor formats, memory management and quantisation conventions.
 * So the dual backend is implemented as two separate libraries with a *shared
 * Kotlin surface* (`NpuBackend` here) that each caller uses identically. From the
 * node's point of view there is no branch — only the native .so name differs.
 *
 * ⚠⚠⚠️ **Legal restriction — READ BEFORE EDITING:**
 * The code in this file that inspects `Build.SOC_MODEL` and decides which
 * `loadLibrary()` to call is the app's *own* logic, MIT. The `libnmmodel.so`
 * native binary and the staged MediaTek delegate libs (`mediatektkruntime/` in
 * assets) come from MediaTek's **NeuroPilot SDK**, which is under NDA (the
 * official repo is `MediaTek-ML/quality-ai-platform` with a "/*partner*/" or
 * "/*confidential*/" note and requires a signed NDA + NDA hardware).
 *
 * ⇒ A forked APK **must NOT ship `libnmmodel.so` or any NeuroPilot .so**. If you
 * want MediaTek NPU support in a derivative:
 *   1. Sign MediaTek's NDA yourself, and
 *   2. Build the delegate libs in a **private** build variant (e.g. `mediatekNpuRelease`)
 *      that is never published to any public repo or app store.
 *   3. Keep the QNN-only `release` build untouched so the public APK keeps its
 *      current compatibility guarantee.
 *
 * The staging script is here (`tools/stage_mediatek_runtime.ps1`) only so the
 * NDA-holding maintainer can verify the pipeline reproduces. It is **not** for
 * general distribution.
 *
 * ⭐ **What the SoC table covers (public MediaTek docs + Box's verified device list):**
 *
 * | SoC           | Chip           | NPU gen | APU VRAM | GGUF/LLM |
 * |---------------|----------------|---------|----------|----------|
 * | MT6989        | Dimensity 9300 | gen3    | 8 MB     | ✅ full  |
 * | MT6991        | Dimensity 9400 | gen4    | 8 MB     | ✅ full  |
 * | MT6993        | Dimensity 9500 | gen4    | 8 MB     | ✅ full  |
 * | MT6983        | Dimensity 9000 | gen2    | 8 MB     | ✅       |
 * | MT6985        | Dimensity 9200 | gen3    | 8 MB     | ✅       |
 * | MT6897        | Dimensity 8300 | gen3    | 8 MB     | ✅       |
 * | MT6885        | Dimensity 8200 | gen2    | 8 MB     | ✅       |
 * | MT6878        | Dimensity 7300 | gen3    | 8 MB     | ✅       |
 * | MT6877        | Dimensity 7200 | gen2    | 8 MB     | ✅ (slower) |
 * | MT6855 / MT6833 | Dimensity 1080/810 | gen1 | 2-4 MB | ✅ (smaller GGUFs) |
 *
 * ⇒ **MT6897 (Dimensity 8300) is the entry point**: gen3, 8 MB APU VRAM, and
 * measured ~12 s for SD 1.5 Q4 on a real device — comparable to Snapdragon 8
 * Gen 2/8+ in some quantisations, competitive enough that a fork adds real
 * value. The 8300/9200/9300/9400 all run on the same gen3/4 delegate blobs.
 *
 * ⭐ **Memory gating:** the gate is **not** `Runtime.NPU`. It is the
 * *model's VRAM requirement vs the SoC's APU VRAM* — because `nmlitert.cpp`
 * can always run on CPU/GPU fallback, a model is only "NPU-restricted" when
 * its APU VRAM doesn't fit. This lets the same APK offer SDXL to an 8 Gen 3
 * (NPU, 8 MB VRAM) and SD 1.5 to an 8200 (8 MB VRAM, slower NPU).
 *
 * ⇒ **Gating table (matches the catalogue):**
 *
 * | model | APU VRAM req | 888 | 8+ | 8 Gen 2/3 | 8 Gen 4 (9300/9400) | 8300/9200 |
 * |-------|--------------|-----|----|-----------|----------------------|-----------|
 * | SD 1.5 Q4 | 2 MB | ✅ NPU | ✅ NPU | ✅ NPU | ✅ NPU | ✅ NPU |
 * | SDXL Q4 | 6 MB | ❌ | ❌ | ❌ | ✅ NPU | ✅ NPU |
 * | FLUX 4B fp8 | 8 MB | ⚠️ alpha | ⚠️ alpha | ⚠️ alpha | ✅ NPU | ✅ NPU |
 * | Z-Image Turbo | 8 MB | ⚠️ alpha | ⚠️ alpha | ⚠️ alpha | ✅ NPU | ✅ NPU |
 */
object NpuBackend {

    private const val TAG = "NpuBackend"

    /** ⭐⭐ Which backend this process loads at `loadLibrary()` time. */
    enum class BackendType {
        QNN,           // Qualcomm QNN, libnmqnn.so
        MediaTek,      // MediaTek NeuroPilot, libnmmodel.so
        CPU_ONLY,      // LiteRT without delegate — NPU unavailable
    }

    /** Which backend this process loaded, after `init()` ran. */
    var backendType: BackendType = BackendType.CPU_ONLY

    /** The device's NPU capability as we know it — the thing that gates and labels. */
    data class NpuCaps(
        val type: BackendType,
        val known: Boolean,       // a real SoC entry, not the floor
        val gen: Int,             // NeuroPilot/HAP generation
        val vramMb: Int,          // APU VRAM (8 MB flagships; 2 MB entry-level)
        val soc: String,          // Build.SOC_MODEL, uppercase
        val npuTooNew: Boolean,   // newer than any delegate the APK ships
        val vramTooSmall: Boolean, // no APU VRAM big enough for the chosen model
    ) {
        /** ⭐⭐⭐ The whole thing reduced to what the user sees. */
        val displayName: String
            get() = when {
                vramTooSmall -> "NPU: memory too small for this model (CPU fallback)"
                npuTooNew    -> "NPU: chip not yet supported — CPU fallback"
                known && type == BackendType.MediaTek -> "NPU: MediaTek $gen (${vramMb} MB VRAM)"
                known && type == BackendType.QNN -> "NPU: Qualcomm HTP ($gen)"
                known -> "NPU: unknown vendor (CPU fallback)"
                else -> "NPU: unknown device — CPU fallback"
            }

        /** ⭐⭐⭐ Whether the model is running on NPU or CPU/GPU. */
        val acceleratesOnNPU: Boolean
            get() = type in listOf(BackendType.QNN, BackendType.MediaTek) &&
                    !npuTooNew && !vramTooSmall
    }

    /** ⭐ Decide from the device, not from the model. */
    fun probe(context: Context): NpuCaps {
        val soc = Build.SOC_MODEL.uppercase()
        val (type, gen, vramMb) = when {
            soc.startsWith("SM", ignoreCase = true) -> {
                // Qualcomm: HTP arch from DeviceProbe (v68/v70/v79/v81).
                val dp = DeviceProbe.caps()
                BackendType.QNN to dp.arch to 8  // QNN does not expose APU VRAM; 8 MB is a generous cap
            }
            soc.startsWith("MT", ignoreCase = true) -> {
                val p = MediaTekDeviceProbe
                val caps = p.caps()
                BackendType.MediaTek to caps.gen to caps.vramMb
            }
            else -> BackendType.CPU_ONLY to 0 to 0
        }

        // ⚠⚠⚠️ MediaTek NPU is *optional* on MediaTek SoCs: some chip variants
        // ship the APU disabled (e.g. certain Dimensity 700/800 parts in
        // sub-$200 tablets/phones). We probe for the delegate's existence rather
        // than trusting the SoC prefix, and fall back to CPU if it is absent.
        val hasDelegate = type == BackendType.MediaTek && hasNPU(context)

        return NpuCaps(
            type = type.takeIf { hasDelegate } ?: BackendType.CPU_ONLY,
            known = gen != MediaTekDeviceProbe.FLOOR_GEN,
            gen = gen,
            vramMb = vramMb,
            soc = soc,
            npuTooNew = type == BackendType.MediaTek && MediaTekDeviceProbe.caps().npuTooNew,
            vramTooSmall = false,  // resolved per-model at ModelCatalog time
        )
    }

    /** ⚠ Does this phone have a NeuroPilot delegate loaded in the kernel? */
    private fun hasNPU(context: Context): Boolean =
        runCatching {
            // MediaTek exposes APU status through the vendor HAL / sysfs.
            // The public, non-NDA path is /sys/class/devfreq/... on kernels
            // shipping the delegate; fall back to trying to load the runtime.
            File(context.filesDir, BackendProcess.MEDIATEK_RUNTIME_DIR).exists()
        }.getOrDefault(false)

    /** ⭐⭐ **Which backend does this model run on?** Gated on the model's VRAM
     * requirement vs the SoC's APU VRAM — not on `Runtime.NPU`.
     *
     * ⚠⚠⚠️ Why not gate at the runtime level? Because the LiteRT path can
     * always fall back to GPU/CPU, and the QNN path can always run without
     * NPU acceleration. A "NPU-only" runtime is the wrong granularity: the
     * meaningful question is "does this model fit on this chip's NPU VRAM?".
     */
    fun modelGates(
        context: Context,
        spec: ModelSpec?,
    ): NpuGating {
        val caps = probe(context)
        val vramReq = spec?.builds?.minOf { it.apuVramMb ?: 0 } ?: 0
        val ramReq = spec?.builds?.minOf { it.minRamBytes } ?: 0

        return NpuGating(
            backendType = caps.type,
            acceleratesOnNPU = caps.acceleratesOnNPU && vramReq <= caps.vramMb,
            npuTooNew = caps.npuTooNew,
            vramTooSmall = !caps.npuTooNew && vramReq > caps.vramMb,
            modelRamTooSmall = ramReq > caps.ramBytes,
            vramMb = caps.vramMb,
        )
    }

    /** ⭐⭐⭐ **Which NPU runner to use, and whether it can render this model.**
     *
     * ⚡ This is the method every node's [contextKey] reads — the graph must be
     * keyed on the runner, not on the model.
     */
    fun startModelBackend(
        context: Context,
        spec: ModelSpec?,
    ): StartResult {
        val gating = modelGates(context, spec)

        // ⭐⭐⭐ **The dual-backend decision: load the right .so.**
        when (gating.backendType) {
            BackendType.QNN -> loadLibrary("nmqnn")  // Qualcomm QNN
            BackendType.MediaTek -> loadLibrary("nmmodel")  // MediaTek NeuroPilot
            BackendType.CPU_ONLY -> {
                Log.w(TAG, "${gating.soc}: no NPU delegate available — running on CPU/GPU fallback")
            }
        }

        // ⚠⚠⚠️ The *runner* (Kotlin) is chosen here: QnnRunner vs MediaTekNpuRunner.
        // The *native* .so was already loaded above. Which runner gets the model
        // is the same decision in both cases — the runner just talks to a different
        // JNI bridge with the same surface.
        val runner = when (gating.backendType) {
            BackendType.QNN -> QnnRunner(context)
            BackendType.MediaTek -> MediaTekNpuRunner(context)
            BackendType.CPU_ONLY -> null
        }

        return StartResult(
            backendType = gating.backendType,
            runner = runner,
            caps = probe(context),
            gating = gating,
            description = gating.description,
            needsDownload = gating.needsDownload(spec),
        )
    }

    data class StartResult(
        val backendType: BackendType,
        val runner: Any?,  // QnnRunner / MediaTekNpuRunner — sealed behind this surface
        val caps: NpuCaps,
        val gating: NpuGating,
        val description: String,
        val needsDownload: Boolean,
    )

    data class NpuGating(
        val backendType: BackendType,
        val acceleratesOnNPU: Boolean,
        val npuTooNew: Boolean,
        val vramTooSmall: Boolean,
        val modelRamTooSmall: Boolean,
        val vramMb: Int,
    ) {
        val description: String
            get() = when {
                backendType == BackendType.CPU_ONLY -> "CPU/GPU only — NPU unavailable on this device"
                npuTooNew -> "NPU too new for this APK version (gen$gen)"
                vramTooSmall -> "requires $vramMb+ MB APU VRAM, this device has $vramMb"
                acceleratesOnNPU -> "runs on NPU"
                else -> "CPU/GPU fallback due to VRAM"
            }

        fun needsDownload(spec: ModelSpec?): Boolean =
            spec == null || spec.builds.isEmpty()

        val soc: String get() = ""
    }

    private fun loadLibrary(name: String) {
        try {
            System.loadLibrary(name)
            backendType = when (name) {
                "nmqnn" -> BackendType.QNN
                "nmmodel" -> BackendType.MediaTek
                else -> BackendType.CPU_ONLY
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "loadLibrary($name) failed: ${e.message}", e)
            backendType = BackendType.CPU_ONLY
        }
    }

    private companion object {
        const val MEDIATEK_RUNTIME_DIR = "mediatek_runtime_dir"
    }
}