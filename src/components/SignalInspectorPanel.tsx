"use client";

import React, { useEffect, useRef } from "react";
import { IQBurst } from "@/lib/rfSynthesizer";
import { BlindDSPResult } from "@/lib/dspEngine";
import { AMRPredictionResult, MODULATION_CLASSES } from "@/lib/amrNeuralNet";

interface SignalInspectorPanelProps {
  iqBurst: IQBurst | null;
  dspResult: BlindDSPResult | null;
  amrResult: AMRPredictionResult | null;
}

export default function SignalInspectorPanel({
  iqBurst,
  dspResult,
  amrResult,
}: SignalInspectorPanelProps) {
  const iqCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const constellationCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cfarGridCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // 1. Render Time-Domain I[n] & Q[n] with CA-CFAR Pulse Edge Markers
  useEffect(() => {
    const canvas = iqCanvasRef.current;
    if (!canvas || !iqBurst) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Center line
    ctx.strokeStyle = "rgba(51, 65, 85, 0.65)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Draw CA-CFAR detected pulse window shading
    if (dspResult?.detected) {
      const xLead = dspResult.leadingEdgeRatio * w;
      const xTrail = dspResult.trailingEdgeRatio * w;
      ctx.fillStyle = "rgba(16, 185, 129, 0.10)";
      ctx.fillRect(xLead, 0, Math.max(4, xTrail - xLead), h);

      ctx.strokeStyle = "#f59e0b";
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(xLead, 0);
      ctx.lineTo(xLead, h);
      ctx.moveTo(xTrail, 0);
      ctx.lineTo(xTrail, h);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const numSamples = Math.min(256, iqBurst.i.length);
    const scaleY = h * 0.36;

    // Draw I[n] (In-phase - Cyan)
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let n = 0; n < numSamples; n++) {
      const x = (n / (numSamples - 1)) * w;
      const y = h / 2 - Math.max(-1.2, Math.min(1.2, iqBurst.i[n] * 0.45)) * scaleY;
      if (n === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Q[n] (Quadrature - Emerald)
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let n = 0; n < numSamples; n++) {
      const x = (n / (numSamples - 1)) * w;
      const y = h / 2 - Math.max(-1.2, Math.min(1.2, iqBurst.q[n] * 0.45)) * scaleY;
      if (n === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [iqBurst, dspResult]);

  // 2. Render Complex Baseband I/Q Constellation Diagram
  useEffect(() => {
    const canvas = constellationCanvasRef.current;
    if (!canvas || !iqBurst) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Axes & reference unit circle
    ctx.strokeStyle = "rgba(51, 65, 85, 0.7)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(w / 2, h / 2, w * 0.34, 0, 2 * Math.PI);
    ctx.strokeStyle = "rgba(6, 182, 212, 0.2)";
    ctx.stroke();

    const N = iqBurst.i.length;
    const start = Math.floor((dspResult?.leadingEdgeRatio ?? 0.15) * N);
    const end = Math.floor((dspResult?.trailingEdgeRatio ?? 0.85) * N);

    ctx.fillStyle = "rgba(34, 211, 238, 0.82)";
    for (let n = start; n < end; n += 2) {
      const valI = iqBurst.i[n];
      const valQ = iqBurst.q[n];
      const px = w / 2 + valI * (w * 0.28);
      const py = h / 2 - valQ * (h * 0.28);
      ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
    }
  }, [iqBurst, dspResult]);

  // 3. Render 2D CA-CFAR Time-Frequency Detection Matrix
  useEffect(() => {
    const canvas = cfarGridCanvasRef.current;
    if (!canvas || !dspResult) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const mask = dspResult.cfarMask2D;
    const T = mask.length;
    const F = mask[0]?.length ?? 128;

    const cellW = w / F;
    const cellH = h / Math.max(1, T);

    for (let t = 0; t < T; t++) {
      for (let f = 0; f < F; f++) {
        if (mask[t][f]) {
          ctx.fillStyle = "#10b981";
        } else {
          const db = dspResult.stftMatrixDb[t]?.[f] ?? -35;
          const norm = Math.max(0, Math.min(1, (db + 36) / 42));
          const shade = Math.floor(14 + norm * 38);
          ctx.fillStyle = `rgb(${shade}, ${shade + 8}, ${shade + 22})`;
        }
        ctx.fillRect(f * cellW, t * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
  }, [dspResult]);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-3.5 shadow-lg">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div>
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-200 font-semibold">
            Baseband I/Q Signal Inspector &amp; 1D-CNN Modulation Classifier
          </h2>
          <p className="text-[11px] font-mono text-slate-400">
            Time-Domain Envelope • 2D CA-CFAR Cell Matrix • Deep Learning AMR Softmax Output
          </p>
        </div>
        {amrResult && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold">
              PRED: {amrResult.predictedClass} ({(amrResult.confidence * 100).toFixed(1)}%)
            </span>
          </div>
        )}
      </div>

      {/* Time-Domain Baseband I[n] + jQ[n] Waveform */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2.5">
        <div className="flex items-center justify-between text-[10px] font-mono mb-1">
          <span className="text-slate-300 font-semibold">
            COMPLEX BASEBAND STREAM I[n] + jQ[n]
          </span>
          <div className="flex items-center gap-3">
            <span className="text-cyan-300">■ I[n] In-Phase</span>
            <span className="text-emerald-400">■ Q[n] Quadrature</span>
            <span className="text-amber-400">┆ CA-CFAR Pulse Edges</span>
          </div>
        </div>
        <canvas
          ref={iqCanvasRef}
          width={480}
          height={82}
          className="w-full h-[82px] block rounded"
        />
      </div>

      {/* Constellation + 2D CA-CFAR Detection Matrix */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2 flex flex-col items-center">
          <div className="w-full flex items-center justify-between text-[10px] font-mono mb-1">
            <span className="text-slate-300 font-semibold">I/Q CONSTELLATION</span>
            <span className="text-slate-500">Phase Plane</span>
          </div>
          <canvas
            ref={constellationCanvasRef}
            width={140}
            height={115}
            className="w-[140px] h-[115px] block rounded bg-slate-950"
          />
        </div>

        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2 flex flex-col">
          <div className="w-full flex items-center justify-between text-[10px] font-mono mb-1">
            <span className="text-slate-300 font-semibold">2D CA-CFAR MATRIX</span>
            <span className="text-emerald-400">■ Hit Cells</span>
          </div>
          <canvas
            ref={cfarGridCanvasRef}
            width={180}
            height={115}
            className="w-full h-[115px] block rounded bg-slate-950"
          />
        </div>
      </div>

      {/* 1D-CNN Modulation Recognition Softmax Distribution */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-3 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-mono font-semibold text-slate-200">
            1D-CNN + CUMULANT NEURAL CLASSIFIER (8-CLASS SOFTMAX)
          </span>
          <span className="text-[10px] font-mono text-amber-300">
            Uncertainty H(p): {amrResult ? amrResult.shannonEntropy.toFixed(3) : "0.000"}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 mt-0.5">
          {MODULATION_CLASSES.map((cls) => {
            const prob = amrResult?.probabilities[cls] ?? 0.125;
            const pct = (prob * 100).toFixed(1);
            const isTop = amrResult?.predictedClass === cls;
            return (
              <div key={cls} className="flex flex-col gap-0.5">
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className={isTop ? "text-cyan-300 font-bold" : "text-slate-400"}>
                    {cls}
                  </span>
                  <span className={isTop ? "text-cyan-300 font-bold" : "text-slate-400"}>
                    {pct}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-150 ${
                      isTop ? "bg-cyan-400" : "bg-slate-600"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(2, prob * 100))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Neural Feature Layer Telemetry */}
        {amrResult && (
          <div className="grid grid-cols-4 gap-1.5 pt-2 mt-1 border-t border-slate-800/80 text-[10px] font-mono">
            <div className="bg-slate-900 px-2 py-1 rounded">
              <span className="text-slate-500 block text-[8px]">|C40| CUMULANT</span>
              <span className="text-slate-200">{amrResult.layerTelemetry.cumulantC40Mag}</span>
            </div>
            <div className="bg-slate-900 px-2 py-1 rounded">
              <span className="text-slate-500 block text-[8px]">C42 CUMULANT</span>
              <span className="text-slate-200">{amrResult.layerTelemetry.cumulantC42}</span>
            </div>
            <div className="bg-slate-900 px-2 py-1 rounded">
              <span className="text-slate-500 block text-[8px]">CHIRP SLOPE β</span>
              <span className="text-slate-200">{amrResult.layerTelemetry.instFreqSlope}</span>
            </div>
            <div className="bg-slate-900 px-2 py-1 rounded">
              <span className="text-slate-500 block text-[8px]">NN LATENCY</span>
              <span className="text-emerald-300">{amrResult.inferenceTimeMs} ms</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
