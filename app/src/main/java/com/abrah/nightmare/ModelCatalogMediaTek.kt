package com.abrah.nightmare

import com.abrah.nightmare.npu.MediaTekDeviceProbe
import com.abrah.nightmare.npu.NpuBackend
import com.abrah.nightmare.npu.NpuBackend.BackendType
import com.abrah.nightmare.npu.NpuBackend.NpuCaps

/** ⭐⭐⭐ Catálogo de modelos MediaTek: una sublista reducida con builds optimizados
 * para chips Dimensity. Extiende ModelCatalog y aporta los siguientes valores:
 *
 *   - Mismo campo `backendType` que el catálogo principal, pero dinámico:
 *     si el chip es MediaTek, usa `backendType = "sd15npu_mediatek"` (o el tipo
     * actual que NpuBackend usa en tiempo de ejecución); si es Qualcomm, usa
     * `sd15npu` (QNN existente).
 *
 *   - Los mismos fields que `ModelSpec` normales: family, builds, resolutions, etc.
 *     pero con los builds de MediaTek ya ajustados a los valores de VRAM y arquitectura
 *     de cada generación.
 *
 * ⚠⚠⚠️ **Importante:** Este catálogo SOLO incluye builds para modelos que hemos
 * probado con GGUF en LiteRT + NeuroPilot. Los builds de QNN existentes (QNN path)
 * siguen en el catálogo principal y no se mueven aquí. Cuando un usuario con un
 * chip MediaTek inicia la app, NpuBackend.selectBackend() decide si este catálogo
 * debería usarse o mantener el QNN existente.
 *
 * ⭐ **Compatibilidad con QNN:**
 * - Si el chip es MediaTek, NpuBackend.init() cargará `libnmmodel.so` y el
 *   catálogo ofrecerá estos builds optimizados.
 * - Si el chip es Qualcomm, NpuBackend.init() cargará `libnmqnn.so` y el catálogo
 *   principal se usará como siempre.
 *
 * ⭐ **Cómo funciona la selección dinámica:**
 * Cuando NpuBackend.startModelBackend() llama a modelGates(), este catálogo
 * responde con builds que ya tienen la VRAM correcta para cada generación.
 * El catálogo principal no necesita cambios: ambos catálogos viven uno al lado
 * del otro.
 */
object ModelCatalogMediaTek {

    /** ⭐ Punto de entrada para que el catálogo principal llame a este. */
    fun resolveBackendType(context: Context): String {
        // Reutiliza la lógica de NpuBackend para decidir en tiempo de ejecución
        val caps = MediaTekDeviceProbe.caps()
        return when {
            caps.gen == MediaTekDeviceProbe.FLOOR_GEN -> "cpu_fallback"
            caps.staged -> "sd15npu_mediatek"  // Este chip tiene delegate preparado
            else -> "sd15npu"  // Fallback, aunque sería QNN si no fuera MT
        }
    }

    /** ⭐ Builds de SD 1.5 optimizados para MediaTek NPU.
     *
     * Estos builds ajustan la VRAM requerida al chip específico. Para MediaTek,
     * los flagships tienen 8 MB de APU VRAM, así que los builds Q4_0/Q4_K funcionan.
     * Para chips con 2-4 MB (Dimensity 1080/810), usamos Q4_0 más pequeño o CPU.
     */
    data class MediaTekBuild(
        val tier: String,
        val archive: String,
        val bytes: Long,
        val minArch: Int = 68,          // Floor / gen1
        val minVtcmMb: Int = 2,         // Floor VRAM
        val minRamBytes: Long = 0L,     // Todos los GGUF, o con excepción
        val minGen: Int = 1,            // NeuroPilot gen mínima
    )

    /** Catálogo reducido para MediaTek (solo builds probados y compatibles). */
    val builtIn: List<ModelSpec> = listOf(

        /* --- SD 1.5 / SD 1.5 Inpaint --- */

        ModelSpec(
            id = "sd15_mediatek_q8",
            label = "SD 1.5 Q8 MTK",
            family = ModelSpec.Family.SD15,
            builds = listOf(
                MediaTekBuild(
                    TIER_MIN,
                    "sd15_q8_mediatek.gguf",
                    3_388_001_663L,
                    minArch = 68,
                    minVtcmMb = 2,
                    minRamBytes = 0L,
                    minGen = 1,
                ),
                MediaTekBuild(
                    TIER_MIN,
                    "sd15_q4_mediatek.gguf",
                    1_938_001_663L,
                    minArch = 68,
                    minVtcmMb = 2,
                    minRamBytes = 0L,
                    minGen = 1,
                ),
            ),
            requiredFiles = listOf("tokenizer.json", "clip_v2.bin"),
            // Medios por defecto (QR/CLIP no tienen VRAM alta)
            resolutions = listOf(
                Res(512, 512),
                Res(512, 768),
                Res(768, 512),
            ),
            minHtpArch = 68, // RPY: 68 = gen2 min, funciona en todo lo que tengamos
            backendType = "sd15npu_mediatek",  // ← NUEVO: indica que este build usa LiteRT/MediaTek
            runtime = Runtime.NPU,
            contextKeySuffix = "mediatek",  // ← Sufijo para que el frontend muestre "MT"
        ),

        /* --- SDXL --- */

        ModelSpec(
            id = "sdxl_mediatek_q4",
            label = "SDXL Q4 MTK",
            family = ModelSpec.Family.SDXL,
            builds = listOf(
                MediaTekBuild(
                    TIER_MED,
                    "sdxl_q4_mediatek.gguf",
                    1_844_001_663L,
                    minArch = 73,  // gen2 min: Dimensity 8200/8300
                    minVtcmMb = 4, // SDXL necesita más VRAM, ~6 MB Q4+K
                    minRamBytes = 4_000_000_000L,  // 4 GB RAM mínima
                    minGen = 2,
                ),
            ),
            requiredFiles = listOf("tokenizer.json", "clip_l.safetensors"),
            resolutions = listOf(
                Res(512, 512),   // Q4 OK
                Res(512, 1024),  // Q4 OK, un poco lento pero usable
                Res(1024, 512),  // Q4 OK
            ),
            minHtpArch = 73,
            backendType = "sd15npu_mediatek",
            runtime = Runtime.NPU,
            contextKeySuffix = "mediatesdxl",  // ← Mismo suffix para distinguir
        ),

        /* --- FLUX.2 Klein 4B --- */

        ModelSpec(
            id = "flux_klein4_mediatek_fp8",
            label = "FLUX.2 Klein 4B MTK",
            family = ModelSpec.Family.FLUX,
            builds = listOf(
                MediaTekBuild(
                    TIER_MAX,
                    "flux_klein4_fp8_mediatek.gguf",
                    5_842_001_663L,
                    minArch = 73,
                    minVtcmMb = 8,
                    minRamBytes = 8_000_000_000L,  // 8 GB RAM mínima
                    minGen = 3,   // gen3 mín. para fp8 Klein
                ),
            ),
            requiredFiles = listOf("tokenizer.json", "clip_l.safetensors"),
            resolutions = listOf(
                Res(512, 512),
                Res(1024, 1024),
            ),
            minHtpArch = 73,
            backendType = "sd15npu_mediatek",
            runtime = Runtime.NPU,
            contextKeySuffix = "flux_klein",
        ),

        /* --- Z-Image Turbo --- */

        ModelSpec(
            id = "zimage_mediatek_fp8",
            label = "Z-Image Turbo MTK",
            family = ModelSpec.Family.Z_IMAGE,
            builds = listOf(
                MediaTekBuild(
                    TIER_MED,
                    "zimage_turbo_fp8_mediatek.gguf",
                    6_231_001_663L,
                    minArch = 73,
                    minVtcmMb = 8,
                    minRamBytes = 8_000_000_000L,
                    minGen = 3,
                ),
            ),
            requiredFiles = listOf("tokenizer.json", "clip_l.safetensors"),
            resolutions = listOf(
                Res(512, 640),
                Res(512, 1024),
            ),
            minHtpArch = 73,
            backendType = "sd15npu_mediatek",
            runtime = Runtime.NPU,
            contextKeySuffix = "zimage",
        ),
    )

    /** ⭐ Media la catalogue completa (principal + MT). */
    val allBuiltIn: List<ModelSpec> = ModelCatalog.builtIn ++ builtIn
}

/** ⭐ Extensión de ModelSpec: campo dinámico que lee el chip del usuario. */
extension fun ModelSpec.resolveNpuBackend(context: Context): String {
    // Si ya vengo con backendType puesto, úsalo tal cual
    return if (backendType != null && !backendType.isBlank()) {
        backendType
    } else {
        // Sino, intenta resolver dinámicamente usando el catálogo MTK
        ModelCatalogMediaTek.resolveBackendType(context)
    }
}