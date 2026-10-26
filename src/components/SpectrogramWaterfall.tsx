"use client";

import React, { useEffect, useRef } from "react";
import { SUB_BANDS, SubBandEmitterConfig } from "@/lib/rfSynthesizer";
import { BlindDSPResult } from "@/lib/dspEngine";
import { ColorMapName, mapNormalizedToRgb } from "@/lib/colormaps";

interface SpectrogramWaterfallProps {
  widebandRowDb: number[]; // 512 bins across 2400 - 2480 MHz
  activeSubBand: number; // 0..7
  dspResult: BlindDSPResult | null;
  colorMap: ColorMapName;
  onSelectColorMap: (cm: ColorMapName) => void;
  onManualSubBandClick: (idx: number) => void;
  activeEmitters: SubBandEmitterConfig[];
  snrDb: number;
}

export default function SpectrogramWaterfall({
  widebandRowDb,
  activeSubBand,
  dspResult,
  colorMap,
  onSelectColorMap,
  onManualSubBandClick,
  activeEmitters,
}: SpectrogramWaterfallProps) {
  const waterfallCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const psdCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Draw scrolling wideband waterfall (512 frequency bins x 180 time rows)
  useEffect(() => {
    const canvas = waterfallCanvasRef.current;
    if (!canvas || widebandRowDb.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Shift existing waterfall down by 2 pixels
    const shiftPx = 2;
    const prevImage = ctx.getImageData(0, 0, width, height - shiftPx);
    ctx.putImageData(prevImage, 0, shiftPx);

    // Render new top strip (shiftPx rows tall)
    const rowImg = ctx.createImageData(width, shiftPx);
    const data = rowImg.data;
    const numBins = widebandRowDb.length;

    const minDb = -36;
    const maxDb = 8;

    for (let x = 0; x < width; x++) {
      const binIdx = Math.min(numBins - 1, Math.floor((x / width) * numBins));
      const db = widebandRowDb[binIdx];
      const norm = Math.max(0, Math.min(1, (db - minDb) / (maxDb - minDb)));
      const [r, g, b] = mapNormalizedToRgb(norm, colorMap);

      for (let y = 0; y < shiftPx; y++) {
        const idx = (y * width + x) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(rowImg, 0, 0);
  }, [widebandRowDb, colorMap]);

  // Draw active dwelled sub-band PSD + 2D CA-CFAR adaptive threshold & envelope markers
  useEffect(() => {
    const canvas = psdCanvasRef.current;
    if (!canvas || !dspResult) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Tactical grid background
    ctx.strokeStyle = "rgba(30, 41, 59, 0.75)";
    ctx.lineWidth = 1;
    for (let gx = 1; gx < 8; gx++) {
      const x = (gx / 8) * w;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let gy = 1; gy < 4; gy++) {
      const y = (gy / 4) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const psd = dspResult.psdDb;
    const cfar = dspResult.cfarThresholdDb;
    const N = psd.length;
    const minDb = -38;
    const maxDb = 12;

    const dbToY = (db: number) => {
      const norm = (db - minDb) / (maxDb - minDb);
      return h - Math.max(4, Math.min(h - 4, norm * (h - 14) + 6));
    };

    // Highlight detected -10 dB spectral envelope region if signal detected
    if (dspResult.detected) {
      const x1 = (dspResult.lowerFreqBin / N) * w;
      const x2 = ((dspResult.upperFreqBin + 1) / N) * w;
      ctx.fillStyle = "rgba(16, 185, 129, 0.14)";
      ctx.fillRect(x1, 0, Math.max(4, x2 - x1), h);

      ctx.strokeStyle = "rgba(16, 185, 129, 0.65)";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x1, 0);
      ctx.lineTo(x1, h);
      ctx.moveTo(x2, 0);
      ctx.lineTo(x2, h);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw PSD filled area
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * w;
      const y = dbToY(psd[i]);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(6, 182, 212, 0.42)");
    grad.addColorStop(1, "rgba(6, 182, 212, 0.02)");
    ctx.fillStyle = grad;
    ctx.fill();

    // Draw PSD trace
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * w;
      const y = dbToY(psd[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw 2D CA-CFAR adaptive threshold trace
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * w;
      const y = dbToY(cfar[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw Centroid fc marker
    if (dspResult.detected) {
      const band = SUB_BANDS[activeSubBand] ?? SUB_BANDS[0];
      const normOffset = (dspResult.centerFreqMHz - band.centerMHz) / 10.0 + 0.5;
      const cx = Math.max(8, Math.min(w - 8, normOffset * w));
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, h);
      ctx.stroke();

      ctx.fillStyle = "#10b981";
      ctx.font = "10px monospace";
      ctx.fillText(`fc=${dspResult.centerFreqMHz.toFixed(2)}MHz`, Math.min(w - 110, cx + 4), 14);
    }
  }, [dspResult, activeSubBand]);

  const currentBand = SUB_BANDS[activeSubBand] ?? SUB_BANDS[0];

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 shadow-lg">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2.5">
          <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse" />
          <div>
            <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-semibold">
              Wideband Surveillance Waterfall &amp; Active Receiver Dwell Overlay (2400–2480 MHz)
            </h2>
            <p className="text-[11px] text-slate-400 font-mono">
              Hann-Windowed STFT (128-pt Radix-2 FFT) • Click any sub-band arm to force receiver lock
            </p>
          </div>
        </div>

        {/* Colormap Selector */}
        <div className="flex items-center gap-1.5 bg-slate-950/80 p-1 rounded-lg border border-slate-800">
          <span className="text-[10px] font-mono text-slate-400 px-1.5">LUT:</span>
          {(["TURBO", "VIRIDIS", "JET"] as ColorMapName[]).map((cm) => (
            <button
              key={cm}
              onClick={() => onSelectColorMap(cm)}
              className={`px-2 py-0.5 text-[10px] font-mono rounded transition ${
                colorMap === cm
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {cm}
            </button>
          ))}
        </div>
      </div>

      {/* Sub-Band Arm Selector Bar & Active Sweep Indicator */}
      <div className="grid grid-cols-8 gap-1">
        {SUB_BANDS.map((band) => {
          const isDwelled = band.index === activeSubBand;
          const emittersHere = activeEmitters.filter((e) => e.subBandIndex === band.index);
          return (
            <button
              key={band.index}
              onClick={() => onManualSubBandClick(band.index)}
              className={`relative flex flex-col items-center py-1.5 px-1 rounded border text-left transition-all ${
                isDwelled
                  ? "bg-cyan-950/70 border-cyan-400 shadow-[0_0_12px_rgba(6,182,212,0.35)]"
                  : "bg-slate-950/60 border-slate-800/80 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between w-full px-1">
                <span
                  className={`text-[10px] font-mono font-bold ${
                    isDwelled ? "text-cyan-300" : "text-slate-400"
                  }`}
                >
                  {band.label}
                </span>
                {isDwelled && (
                  <span className="text-[9px] font-mono px-1 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-400/40">
                    LOCK
                  </span>
                )}
              </div>
              <span className="text-[9px] font-mono text-slate-500">
                {band.startMHz}-{band.endMHz}
              </span>
              <div className="mt-1 flex items-center gap-1 h-3">
                {emittersHere.length > 0 ? (
                  emittersHere.map((em) => (
                    <span
                      key={em.id}
                      className={`text-[8px] font-mono px-1 rounded ${
                        em.isHopping
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                          : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      }`}
                    >
                      {em.modulation}
                    </span>
                  ))
                ) : (
                  <span className="text-[8px] font-mono text-slate-600">CLEAR</span>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Scrolling Wideband Waterfall Canvas with Active Dwell Highlight Column */}
      <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
        <canvas
          ref={waterfallCanvasRef}
          width={640}
          height={185}
          className="w-full h-[185px] block"
        />
        {/* Active Receiver Sweep Overlay Box */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 border-x-2 border-cyan-400/90 bg-cyan-400/5 transition-all duration-150"
          style={{
            left: `${(activeSubBand / 8) * 100}%`,
            width: `${100 / 8}%`,
          }}
        >
          <div className="bg-cyan-500/90 text-slate-950 text-[9px] font-mono font-bold px-1 py-0.5 text-center truncate">
            RX DWELL #{activeSubBand + 1}
          </div>
        </div>
      </div>

      {/* Instantaneous Sub-Band PSD + 2D CA-CFAR Threshold Overlay */}
      <div className="bg-slate-950/90 border border-slate-800/90 rounded-lg p-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono font-semibold text-cyan-300">
              ACTIVE DWELL SPECTRUM: {currentBand.label} ({currentBand.startMHz}–{currentBand.endMHz} MHz)
            </span>
            {dspResult?.detected ? (
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                CA-CFAR PULSE DETECTED (SNR: {dspResult.estimatedSnrDb} dB)
              </span>
            ) : (
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                NOISE FLOOR ({dspResult?.noiseFloorDb ?? -24.0} dB)
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 text-[10px] font-mono">
            <span className="flex items-center gap-1 text-cyan-300">
              <span className="w-2.5 h-0.5 bg-cyan-400 inline-block" /> PSD (dB)
            </span>
            <span className="flex items-center gap-1 text-amber-300">
              <span className="w-2.5 h-0.5 bg-amber-400 inline-block" /> 2D CA-CFAR Threshold
            </span>
            <span className="flex items-center gap-1 text-emerald-300">
              <span className="w-2.5 h-2 bg-emerald-500/30 border border-emerald-400 inline-block" /> -10dB Envelope
            </span>
          </div>
        </div>

        <canvas
          ref={psdCanvasRef}
          width={640}
          height={105}
          className="w-full h-[105px] block rounded"
        />

        {/* Blind Parameter Telemetry Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-2 pt-2 border-t border-slate-800/80 text-[11px] font-mono">
          <div className="bg-slate-900/80 px-2.5 py-1 rounded border border-slate-800">
            <span className="text-slate-400 block text-[9px]">CENTER FREQ (fc)</span>
            <span className="text-cyan-300 font-semibold">
              {dspResult?.detected ? `${dspResult.centerFreqMHz.toFixed(3)} MHz` : "---"}
            </span>
          </div>
          <div className="bg-slate-900/80 px-2.5 py-1 rounded border border-slate-800">
            <span className="text-slate-400 block text-[9px]">BW (-3dB / -10dB)</span>
            <span className="text-emerald-300 font-semibold">
              {dspResult?.detected
                ? `${dspResult.bandwidth3dBMHz.toFixed(2)} / ${dspResult.bandwidth10dBMHz.toFixed(2)} MHz`
                : "---"}
            </span>
          </div>
          <div className="bg-slate-900/80 px-2.5 py-1 rounded border border-slate-800">
            <span className="text-slate-400 block text-[9px]">PULSE WIDTH (PW)</span>
            <span className="text-amber-300 font-semibold">
              {dspResult?.detected ? `${dspResult.pulseWidthUs.toFixed(1)} μs` : "---"}
            </span>
          </div>
          <div className="bg-slate-900/80 px-2.5 py-1 rounded border border-slate-800">
            <span className="text-slate-400 block text-[9px]">ESTIMATED PRI</span>
            <span className="text-purple-300 font-semibold">
              {dspResult?.detected ? `${dspResult.priUs} μs` : "---"}
            </span>
          </div>
          <div className="bg-slate-900/80 px-2.5 py-1 rounded border border-slate-800">
            <span className="text-slate-400 block text-[9px]">CFAR HIT RATIO</span>
            <span className="text-slate-200 font-semibold">
              {dspResult ? `${(dspResult.cfarHitRatio * 100).toFixed(1)}%` : "0.0%"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
