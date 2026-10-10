package com.abrah.nightmare.npu

import android.os.Build
import android.util.Log

/** ⭐ What this phone's MediaTek NPU actually is — inferred from the SoC model.
 *
 * ⚠⚠ Unlike Qualcomm where QNN exposes exact arch/VTCM, MediaTek's NeuroPilot
 * SDK does not ship a user-space probe binary in the APK. We fall back to a
 * SoC-to-capability table built from MediaTek's public specs and the Box app's
 * verified device list (MT6989, MT6991, MT6993, MT6897, MT6878, MT6885).
 *
 * ⇒ Two numbers decide everything here: the NeuroPilot **gen** (≈ LiteRT tier)
 * and its **APU VRAM** in MB. A measurement path (via `lite::Interpreter` with
 * NeuroPilot delegate) would be better, but LiteRT JIT compiles on first run
 * and there is no pre-run probe equivalent to QNN's `--device_info`.
 *
 * ⚠ The table is conservative: an unknown chip gets the floor (gen 1, 2 MB VRAM)
 * which still runs SD 1.5 GGUF on LiteRT CPU/GPU fallback. Over-guessing
 * would hand a user a model their NPU refuses, which is worse.
 */
object MediaTekDeviceProbe {

    private const val TAG = "MediaTekDeviceProbe"

    /** ⭐⭐ SoC → NeuroPilot generation token (≈ LiteRT delegate tier).
     *
     * ⚠⚠ Never guess an entry. A wrong tier loads the wrong delegate and
     * NPU init fails with a generic "delegate not found" that points nowhere.
     * An SoC absent here is safe: the runner then falls back to LiteRT GPU/CPU.
     */
    internal val SOC_TO_GEN = mapOf(
        // Flagships (2023-2025)
        "MT6989" to 3,   // Dimensity 9300 - NPU 790 (Gen 3)
        "MT6991" to 4,   // Dimensity 9400 - NPU 890 (Gen 4)
        "MT6993" to 4,   // Dimensity 9500 - NPU 890 (Gen 4)
        "MT6983" to 2,   // Dimensity 9000 - NPU 690 (Gen 2)
        "MT6985" to 3,   // Dimensity 9200 - NPU 790 (Gen 3)
        // Premium/Upper-mid
        "MT6897" to 3,   // Dimensity 8300 - NPU 790 (Gen 3)
        "MT6885" to 2,   // Dimensity 8200 - NPU 690 (Gen 2)
        "MT6878" to 3,   // Dimensity 7300 - NPU 790 (Gen 3)
        // Mid-range (NPU present but smaller VRAM)
        "MT6877" to 2,   // Dimensity 7200 - NPU 690 (Gen 2)
        "MT6855" to 1,   // Dimensity 1080 - NPU 490 (Gen 1)
        "MT6833" to 1,   // Dimensity 810 - NPU 490 (Gen 1)
    )

    /** ⭐ SoCs known to carry **8 MB** of APU VRAM.
     *
     * ⚠ Deliberately separate from [SOC_TO_GEN]. That table decides which
     * delegate blob to load; this one only decides whether a model that needs
     * 8 MB VRAM is offered for download. A wrong VRAM guess costs a wasted
     * download; a wrong gen guess breaks NPU init.
     */
    private val SOC_HAS_8MB_APU_VRAM = setOf(
        "MT6989", "MT6991", "MT6993", "MT6983", "MT6985", // Flagships
        "MT6897", "MT6878",                                // Premium with 8 MB
    )

    /** Every gen whose delegate blobs the APK carries. ⚠ Keep in step with
     * the assets staged by `tools/stage_mediatek_runtime.ps1` (to be created). */
    val STAGED_GENS = listOf(1, 2, 3, 4)

    /** The APU as far as we know it. */
    data class Caps(
        val gen: Int,
        val vramMb: Int,
        val measured: Boolean,
        val soc: String,
        val ramBytes: Long = 0L,
    ) {
        /** ⚠ The APK has to carry this gen's delegate, or NPU init fails. */
        val staged: Boolean get() = gen in STAGED_GENS

        /** ⭐⭐⭐ Do we actually KNOW this chip, or is [gen] the floor? */
        val known: Boolean get() = measured || gen != FLOOR_GEN

        /** A floor, never an allowlist: newer gens run older delegates — but only a gen the APK ships. */
        fun supportsGen(min: Int): Boolean = staged && gen >= min

        /** A chip whose NPU is newer than every gen this APK's runtime knows. */
        val npuTooNew: Boolean get() = gen > (STAGED_GENS.maxOrNull() ?: 0)
    }

    /** The SoC as Android reports it, uppercased — the key for both tables. */
    fun soc(): String = (Build.SOC_MODEL ?: "").uppercase()

    /** ⭐⭐ What to gate and label with. */
    fun caps(): Caps {
        val soc = soc()
        val gen = SOC_TO_GEN[soc] ?: genFromPartNumber(soc) ?: FLOOR_GEN
        val vramMb = when {
            soc in SOC_HAS_8MB_APU_VRAM -> 8
            genFromPartNumber(soc) != null -> 8
            else -> FLOOR_VRAM_MB
        }
        return Caps(
            gen = gen,
            vramMb = vramMb,
            measured = false,
            soc = soc,
            ramBytes = totalRamBytes(),
        )
    }

    /** ⭐ An gen for a chip no table knows, read out of its part number.
     * Only used UPWARD for a part number at or above the newest known. */
    internal fun genFromPartNumber(soc: String): Int? {
        if (!soc.startsWith("MT")) return null
        val n = soc.dropWhile { !it.isDigit() }.takeWhile { it.isDigit() }.toIntOrNull()
            ?: return null
        val newestKnown = SOC_TO_GEN.keys.mapNotNull { partNumberOf(it) }.maxOrNull() ?: return null
        if (n < newestKnown) return null
        if (n / 100 > newestKnown / 100) return NEWER_THAN_STAGED
        return SOC_TO_GEN.values.maxOrNull()
    }

    /** ⚠ Not a gen: "newer than every delegate this APK ships". */
    const val NEWER_THAN_STAGED = 99

    private fun partNumberOf(soc: String): Int? =
        soc.dropWhile { !it.isDigit() }.takeWhile { it.isDigit() }.toIntOrNull()

    /** True when the guess above is doing real work and should be replaced by a measurement. */
    fun needsMeasuring(): Boolean = soc() !in SOC_TO_GEN

    /** Gen 1 / 2 MB — what runs anywhere (LiteRT CPU/GPU fallback). */
    const val FLOOR_GEN = 1
    const val FLOOR_VRAM_MB = 2

    /** The phone's RAM from /proc/meminfo (0 when unknown). */
    fun totalRamBytes(): Long = runCatching {
        java.io.File("/proc/meminfo").useLines { lines ->
            lines.first { it.startsWith("MemTotal:") }
                .filter { it.isDigit() }.toLong() * 1024L
        }
    }.getOrDefault(0L)

    /** Which NeuroPilot delegate libs this device needs unpacked. */
    fun delegateLibs(all: List<String>): List<String> {
        val gen = SOC_TO_GEN[soc()] ?: measured?.gen ?: return all
        val token = "gen$gen"
        return all.filter { n ->
            !DELEGATE_TOKEN.containsMatchIn(n) || n.contains(token)
        }
    }

    private val DELEGATE_TOKEN = Regex("gen\\d{1}")

    /** ⚠ Placeholder for a future `--device_info` equivalent using LiteRT. */
    suspend fun measure(context: android.content.Context): Caps? = null
}