package com.abrah.nightmare.npu

import android.content.Context
import android.util.Log
import java.io.File

/** ⭐⭐ A cache of loaded NPU graphs — the thing that makes running a context
 * binary from a node affordable, for the MediaTek NeuroPilot path.
 *
 * ⚠⚠ **Residency is the whole point.** Just as QnnRunner, this keeps context
 * binaries resident between calls instead of re-reading every inference.
 * A 1.4 GB context deserialised per call would be a render cost, not a cache hit.
 *
 * ⚠ Keyed by **absolute path**, not by a short name — same rule as QnnRunner.
 *
 * ⚠⚠ Every method blocks and several of them for seconds. Call from a worker.
 *
 * ⇒ Uses **LiteRT** with a **NeuroPilot delegate** (not Qualcomm QNN). The
 * native side is `app/src/main/cpp/nmlitert.cpp` (to be created).
 */
class MediaTekNpuRunner(private val context: Context) : AutoCloseable {

    private val handles = HashMap<String, Long>()
    private val runnerPaths = HashMap<String, String>()

    @Volatile
    private var initialised = false

    /** Bring the NeuroPilot delegate up. Idempotent. */
    @Synchronized
    fun initBackend() {
        if (initialised) return
        NativeMediaTekNpu.ensureLoaded()
        val runtime = BackendProcess.prepareRuntime(context)
        val ok = NativeMediaTekNpu.init(
            File(runtime, "liblitert_runtime.so").absolutePath,
            File(runtime, "libmtk-neuropilot.so").absolutePath,
            runtime.absolutePath,
        )
        check(ok) { "MediaTek NPU init failed: ${NativeMediaTekNpu.lastError()}" }
        initialised = true
        Log.i(TAG, "MediaTek NPU ready (runtime ${runtime.absolutePath})")
    }

    /** Load (and cache) a context binary. ⚠ Throws rather than returning 0. */
    @Synchronized
    fun handleOf(bin: File): Long {
        initBackend()
        handles[bin.absolutePath]?.let { return it }
        check(bin.isFile && bin.length() > 0) { "no context binary at ${bin.absolutePath}" }
        val h = NativeMediaTekNpu.load(bin.absolutePath)
        check(h != 0L) { "load ${bin.name} failed: ${NativeMediaTekNpu.lastError()}" }
        handles[bin.absolutePath] = h
        runnerPaths[bin.absolutePath] = bin.absolutePath
        Resident.add(bin)
        return h
    }

    /** The binary's own tensor list: `in <name>:dims:dtype;out …;`. */
    fun describe(bin: File): String = NativeMediaTekNpu.describe(handleOf(bin))

    /** Execute with named real-valued tensors, returning outputs by name. */
    fun run(bin: File, inputs: Map<String, FloatArray>): Map<String, FloatArray> {
        val h = handleOf(bin)
        val names = inputs.keys.toTypedArray()
        val data = names.map { inputs.getValue(it) }.toTypedArray()
        val out = NativeMediaTekNpu.execute(h, names, data)
            ?: error("${bin.name}: execute failed: ${NativeMediaTekNpu.lastError()}")
        val outNames = NativeMediaTekNpu.outputNames(h)
        return outNames.indices.associate { outNames[it] to out[it] }
    }

    /** ⚠ Frees the context. The next [run] reloads it from flash. */
    @Synchronized
    fun release(bin: File) {
        handles.remove(bin.absolutePath)?.let { NativeMediaTekNpu.free(it) }
        Resident.remove(bin)
    }

    @Synchronized
    fun releaseAll() {
        handles.values.forEach { NativeMediaTekNpu.free(it) }
        handles.keys.forEach { Resident.remove(File(it)) }
        handles.clear()
    }

    /** ⭐⭐ What the NPU is holding **right now**, across every runner in the process. */
    object Resident {
        private val counts = LinkedHashMap<String, Int>()

        @Synchronized
        internal fun add(bin: File) {
            counts[bin.name] = (counts[bin.name] ?: 0) + 1
        }

        @Synchronized
        internal fun remove(bin: File) {
            val n = (counts[bin.name] ?: 0) - 1
            if (n > 0) counts[bin.name] = n else counts.remove(bin.name)
        }

        /** How many context binaries are mapped. 0 when the NPU is idle. */
        @Synchronized
        fun count(): Int = counts.size

        /** ⚠ For a readout and a log line, newest last. */
        @Synchronized
        fun names(): List<String> = counts.keys.toList()
    }

    /** ⚠ For a readout: which binaries are holding memory right now. */
    @Synchronized
    fun resident(): List<String> = handles.keys.map { File(it).name }.sorted()

    // ---- by NAME ----------------------------------------------------------

    /** ⚠ A name, not a descriptor. */
    @JvmInline
    value class Graph(val name: String)

    fun fileOf(name: String): File = NpuFiles.contextFile(context, name)
        ?: error("model \"$name\" is not on this device (${NpuFiles.ctxDir(context)})")

    fun isReady(name: String): Boolean = NpuFiles.isReady(context, name)
    fun handleOf(name: String): Long = handleOf(fileOf(name))
    fun describe(name: String): String = describe(fileOf(name))
    fun release(name: String) {
        NpuFiles.contextFile(context, name)?.let { release(it) }
    }

    fun run(
        g: Graph,
        floats: Map<String, FloatArray> = emptyMap(),
        intInputs: Map<String, IntArray> = emptyMap(),
    ): Map<String, FloatArray> {
        val all = LinkedHashMap<String, FloatArray>(floats)
        intInputs.forEach { (k, v) -> all[k] = FloatArray(v.size) { v[it].toFloat() } }
        return run(fileOf(g.name), all)
    }

    override fun close() = releaseAll()

    private companion object {
        const val TAG = "MediaTekNpuRunner"
    }
}