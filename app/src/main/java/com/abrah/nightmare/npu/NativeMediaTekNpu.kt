package com.abrah.nightmare.npu

import android.content.Context
import com.abrah.nightmare.BackendProcess
import java.io.File

/** ⭐ Wrapper around the MediaTek NPU (LiteRT + NeuroPilot) native backend —
 * the counterpart of `app/src/main/cpp/nmqnn.cpp` (Qualcomm QNN runner).
 *
 * ⭐⭐ **Architecture note**: the in-process LiteRT delegate runs graphs from
 * **`.litertlm` context binaries** (or `.tflite` compiled on-device), not from
 * Qualcomm's QNN `contextCreateFromBinary` format. Same interface as NativeQnn
 * so no callers have to change; the wire types are identical (float arrays by
 * name, handles as `jlong`).
 *
 * ⚠⚠ **Why not reuse nmqnn.cpp?** QNN's API surface, tensor memory management,
 * quantisation encoding (real = (q + offset) * scale with negative offset), and
 * the fastrpc DSP call model are all Qualcomm-specific. NeuroPilot uses LiteRT's
 * interpreter/delegate API with tensor handles and memory-mapped weights. This
 * is a fresh implementation, not a port.
 *
 * ⚠ The header files are **fetched** from the MediaTek NeuroPilot SDK (to be
 * provided by the maintainer), not committed — same policy as QNN headers.
 */
object NativeMediaTekNpu {

    private var gErr = ""
    private var gLib: android.system.OsConstants? = null

    fun lastError(): String = gErr

    private external fun native_init(backendPath: String, delegatePath: String, skelDir: String): Boolean

    private external fun native_load(path: String): Long

    private external fun native_describe(handle: Long): String

    private external fun native_execute(
        handle: Long, names: Array<String>, data: Array<FloatArray>
    ): Array<FloatArray>?

    private external fun native_output_names(handle: Long): Array<String>

    private external fun native_free(handle: Long)

    private external fun native_ensure_loaded()

    fun ensureLoaded() {
        // Same pattern as NativeQnn.ensureLoaded — loads libnmmodel.so.
        try {
            System.loadLibrary("nmmodel")
            native_ensure_loaded()
        } catch (e: UnsatisfiedLinkError) {
            gErr = "failed to load libnmmodel.so: ${e.message}"
            throw e
        }
    }

    @Synchronized
    fun init(backendPath: String, delegatePath: String, skelDir: String): Boolean {
        val ok = native_init(backendPath, delegatePath, skelDir)
        if (!ok) {
            gErr = "native init failed: $lastError()"
            return false
        }
        return true
    }

    @Synchronized
    fun load(path: String): Long {
        val h = native_load(path)
        if (h == 0L) gErr = "load failed: $lastError()"
        return h
    }

    @Synchronized
    fun describe(handle: Long): String = native_describe(handle)

    @Synchronized
    fun execute(handle: Long, names: Array<String>, data: Array<FloatArray>): Array<FloatArray>? {
        return native_execute(handle, names, data)
    }

    @Synchronized
    fun outputNames(handle: Long): Array<String> = native_output_names(handle)

    @Synchronized
    fun free(handle: Long) {
        if (handle != 0L) native_free(handle)
    }
}