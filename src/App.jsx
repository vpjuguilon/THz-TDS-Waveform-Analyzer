import React, { useState, useMemo, useCallback, useRef } from 'react';
import Papa from 'papaparse';
import * as math from 'mathjs';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea, ReferenceLine, Customized,
} from 'recharts';
import { Upload, Trash2, Eye, EyeOff, Sparkles, X, Download, ZoomIn, Move, RotateCcw, Pencil, Camera, GripVertical } from 'lucide-react';

const COLORS = ['#0d9488', '#d97706', '#db2777', '#4f46e5', '#65a30d', '#ea580c', '#0284c7', '#dc2626'];

// Default frequency-domain view range (THz) used before the user sets an explicit range or zooms.
const DEFAULT_FREQ_DOMAIN = [0, 6];

// Common strong atmospheric water-vapor absorption lines in the THz range (THz), commonly cited in THz-TDS work.
const WATER_VAPOR_LINES = [0.557, 0.752, 0.988, 1.097, 1.113, 1.163, 1.208, 1.229, 1.412, 1.602, 1.669, 1.718, 1.796, 1.867];

// ---------- signal processing helpers ----------

function nextPow2(n) {
  return Math.pow(2, Math.ceil(Math.log2(Math.max(2, n))));
}

function applyWindow(data, type) {
  const N = data.length;
  if (type === 'none' || N < 2) return data.slice();
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    let w = 1;
    if (type === 'hann') w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));
    else if (type === 'hamming') w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (N - 1));
    else if (type === 'blackman') {
      w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (N - 1));
    }
    out[i] = data[i] * w;
  }
  return out;
}

const UNIT_TO_PS = { fs: 0.001, ps: 1, ns: 1000, s: 1e12 };

function estimateDt(time) {
  const diffs = [];
  for (let i = 1; i < time.length; i++) diffs.push(time[i] - time[i - 1]);
  diffs.sort((a, b) => a - b);
  return diffs[Math.floor(diffs.length / 2)] || 1;
}

function computeFFT(time, amplitude, opts) {
  const { windowType, zeroPadFactor, timeUnit } = opts;
  const scale = UNIT_TO_PS[timeUnit] ?? 1;
  const dt_ps = estimateDt(time) * scale;

  const N = amplitude.length;
  const windowed = applyWindow(amplitude, windowType);
  const mean = windowed.reduce((a, b) => a + b, 0) / windowed.length;
  const centered = windowed.map((v) => v - mean);

  const paddedLen = nextPow2(N * zeroPadFactor);
  const padded = new Array(paddedLen).fill(0);
  for (let i = 0; i < N; i++) padded[i] = centered[i];

  const spectrum = math.fft(padded);
  const half = Math.floor(paddedLen / 2);
  const freqs = new Array(half);
  const mags = new Array(half);
  for (let k = 0; k < half; k++) {
    freqs[k] = k / (paddedLen * dt_ps); // THz, since dt_ps is in picoseconds
    const c = spectrum[k];
    mags[k] = Math.hypot(c.re, c.im);
  }
  return { freqs, mags };
}

function toDB(mags) {
  return mags.map((m) => 20 * Math.log10(Math.max(m, 1e-12)));
}

function findPeakIndex(mags) {
  let idx = 0;
  for (let i = 1; i < mags.length; i++) if (mags[i] > mags[idx]) idx = i;
  return idx;
}

function interpCrossing(f1, m1, f2, m2, target) {
  if (m2 === m1) return f1;
  const t = (target - m1) / (m2 - m1);
  return f1 + t * (f2 - f1);
}

function computeBandwidth(freqs, magsDB, peakIndex, thresholdDB) {
  let lo = freqs[0];
  let hi = freqs[freqs.length - 1];
  for (let i = peakIndex; i > 0; i--) {
    if (magsDB[i] >= thresholdDB && magsDB[i - 1] < thresholdDB) {
      lo = interpCrossing(freqs[i - 1], magsDB[i - 1], freqs[i], magsDB[i], thresholdDB);
      break;
    }
  }
  for (let i = peakIndex; i < magsDB.length - 1; i++) {
    if (magsDB[i] >= thresholdDB && magsDB[i + 1] < thresholdDB) {
      hi = interpCrossing(freqs[i], magsDB[i], freqs[i + 1], magsDB[i + 1], thresholdDB);
      break;
    }
  }
  return { lo, hi, width: Math.max(0, hi - lo) };
}

function computeNoiseFloorDB(time, amplitude, region, fraction, opts) {
  const N = time.length;
  const n = Math.max(8, Math.floor(N * fraction));
  const segTime = region === 'end' ? time.slice(N - n) : time.slice(0, n);
  const segAmp = region === 'end' ? amplitude.slice(N - n) : amplitude.slice(0, n);
  const { mags } = computeFFT(segTime, segAmp, opts);
  const magsDB = toDB(mags);
  return magsDB.reduce((a, b) => a + b, 0) / magsDB.length;
}

// ---------- sample data ----------

function generateSample(name, color, params) {
  const { tau, f0, cycles, noiseLevel, t0 } = params;
  const dt = 0.02;
  const time = [];
  const amplitude = [];
  for (let t = -5; t <= 30; t += dt) {
    const x = (t - t0) / tau;
    let e;
    if (cycles > 1) {
      e = Math.exp(-x * x) * Math.sin(2 * Math.PI * f0 * (t - t0));
    } else {
      e = x * Math.exp(-x * x);
    }
    e += (Math.random() * 2 - 1) * noiseLevel;
    time.push(Number(t.toFixed(4)));
    amplitude.push(e);
  }
  return {
    id: `${name}_${Math.random().toString(36).slice(2)}`,
    name, color, visible: true, width: 1.4, time, amplitude,
  };
}

function makeSampleSet() {
  return [
    generateSample('Emitter A (broadband)', COLORS[0], { tau: 0.12, f0: 0, cycles: 1, noiseLevel: 0.004, t0: 5 }),
    generateSample('Emitter B (narrowband)', COLORS[1], { tau: 3, f0: 0.55, cycles: 3, noiseLevel: 0.006, t0: 6 }),
    generateSample('Emitter C (noisy)', COLORS[2], { tau: 0.25, f0: 0, cycles: 1, noiseLevel: 0.03, t0: 4.5 }),
  ];
}

// ---------- file parsing ----------

function parseFileText(text) {
  let rows = Papa.parse(text.trim(), { skipEmptyLines: true, dynamicTyping: true }).data;
  if (rows.length && (!Array.isArray(rows[0]) || rows[0].length < 2)) {
    rows = text.trim().split('\n').map((line) => line.trim().split(/[\s,;]+/).map(Number));
  }
  let startIdx = 0;
  if (rows.length) {
    const r0 = rows[0];
    const looksLikeHeader = r0.some((v) => typeof v !== 'number' || Number.isNaN(v));
    if (looksLikeHeader) startIdx = 1;
  }
  const time = [];
  const amplitude = [];
  for (let i = startIdx; i < rows.length; i++) {
    const r = rows[i];
    if (!r || r.length < 2) continue;
    const t = Number(r[0]);
    const a = Number(r[1]);
    if (Number.isNaN(t) || Number.isNaN(a)) continue;
    time.push(t);
    amplitude.push(a);
  }
  return { time, amplitude };
}

// Draws a full black rectangle around the plot area (recharts only draws the bottom/left axis lines by default).
function ChartBorder({ offset }) {
  if (!offset) return null;
  return (
    <rect
      x={offset.left} y={offset.top} width={offset.width} height={offset.height}
      fill="none" stroke="#000000" strokeWidth={0.75} pointerEvents="none"
    />
  );
}

function niceStep(rawStep) {
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(rawStep))));
  const norm = rawStep / mag;
  let niceNorm;
  if (norm < 1.5) niceNorm = 1;
  else if (norm < 3) niceNorm = 2;
  else if (norm < 7) niceNorm = 5;
  else niceNorm = 10;
  return niceNorm * mag;
}

// Generates clean, round-number tick positions (multiples of 1/2/5/10-ish) within [min, max],
// instead of dividing the range into evenly-spaced but arbitrary decimals.
function niceTicks(min, max, count = 7) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const rawStep = (max - min) / Math.max(1, count - 1);
  if (!Number.isFinite(rawStep) || rawStep <= 0) return [min, max];
  const step = niceStep(rawStep);
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let t = start; t <= max + step * 1e-9; t += step) {
    ticks.push(Number(t.toFixed(10)));
  }
  return ticks.length >= 2 ? ticks : [min, max];
}

function getYPixelRange(wrapperEl) {
  if (!wrapperEl) return null;
  const line = wrapperEl.querySelector('.recharts-yAxis line.recharts-cartesian-axis-line');
  if (line) {
    const y1 = parseFloat(line.getAttribute('y1'));
    const y2 = parseFloat(line.getAttribute('y2'));
    if (!Number.isNaN(y1) && !Number.isNaN(y2) && y1 !== y2) {
      return { top: Math.min(y1, y2), bottom: Math.max(y1, y2) };
    }
  }
  const h = wrapperEl.clientHeight || 384;
  return { top: 15, bottom: Math.max(60, h - 90) };
}

function makeYScale(pixelRange, domain) {
  if (!pixelRange || !domain) return null;
  const { top, bottom } = pixelRange;
  const [yMin, yMax] = domain;
  if (bottom === top || yMax === yMin) return null;
  return {
    pxToVal: (px) => yMax - ((px - top) / (bottom - top)) * (yMax - yMin),
  };
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function sanitizeSvgAttrs(el) {
  if (el.attributes) {
    for (let i = 0; i < el.attributes.length; i++) {
      const attr = el.attributes[i];
      if (typeof attr.value === 'string' && /NaN|Infinity/.test(attr.value)) {
        el.setAttribute(attr.name, attr.name === 'opacity' || attr.name === 'fill-opacity' || attr.name === 'stroke-opacity' ? '1' : '0');
      }
    }
  }
  for (let i = 0; i < el.childNodes.length; i++) {
    if (el.childNodes[i].nodeType === 1) sanitizeSvgAttrs(el.childNodes[i]);
  }
}

function buildExportSvg(originalSvg, legendItems, fallbackWidth, fallbackHeight) {
  const ns = 'http://www.w3.org/2000/svg';
  let width = Number(originalSvg.getAttribute('width'));
  let heightAttr = Number(originalSvg.getAttribute('height'));
  if (!Number.isFinite(width) || width <= 0) width = fallbackWidth || 600;
  if (!Number.isFinite(heightAttr) || heightAttr <= 0) heightAttr = fallbackHeight || 300;

  // The live chart reserves extra vertical space for its on-screen HTML legend, which isn't
  // part of the cloned SVG — use the actual x-axis position instead of the raw container height
  // so we don't leave a big blank gap before our own drawn legend.
  let plotBottom = heightAttr;
  const axisLine = originalSvg.querySelector('.recharts-yAxis line.recharts-cartesian-axis-line');
  if (axisLine) {
    const y1 = parseFloat(axisLine.getAttribute('y1'));
    const y2 = parseFloat(axisLine.getAttribute('y2'));
    if (Number.isFinite(y1) && Number.isFinite(y2)) plotBottom = Math.max(y1, y2);
  }
  const contentBottom = Math.min(heightAttr, plotBottom + 42); // room for x-axis tick labels + axis title

  const legendRowH = 22;
  const legendH = legendItems.length ? legendRowH + 14 : 0;
  const totalH = contentBottom + legendH;

  const newSvg = document.createElementNS(ns, 'svg');
  newSvg.setAttribute('xmlns', ns);
  newSvg.setAttribute('width', width);
  newSvg.setAttribute('height', totalH);
  newSvg.setAttribute('viewBox', `0 0 ${width} ${totalH}`);
  newSvg.setAttribute('font-family', 'Inter, Arial, Helvetica, sans-serif');
  newSvg.style.fontFamily = 'Inter, Arial, Helvetica, sans-serif';

  const bg = document.createElementNS(ns, 'rect');
  bg.setAttribute('x', 0);
  bg.setAttribute('y', 0);
  bg.setAttribute('width', width);
  bg.setAttribute('height', totalH);
  bg.setAttribute('fill', '#ffffff');
  newSvg.appendChild(bg);

  const cloned = originalSvg.cloneNode(true);
  const g = document.createElementNS(ns, 'g');
  while (cloned.firstChild) g.appendChild(cloned.firstChild);
  newSvg.appendChild(g);

  if (legendItems.length) {
    let x = 10;
    const y = contentBottom + 22;
    legendItems.forEach((item) => {
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', x);
      rect.setAttribute('y', y - 9);
      rect.setAttribute('width', 10);
      rect.setAttribute('height', 10);
      rect.setAttribute('fill', item.color);
      newSvg.appendChild(rect);

      const text = document.createElementNS(ns, 'text');
      text.setAttribute('x', x + 14);
      text.setAttribute('y', y);
      text.setAttribute('font-size', '11');
      text.setAttribute('font-family', 'Inter, sans-serif');
      text.setAttribute('fill', '#334155');
      text.textContent = item.name;
      newSvg.appendChild(text);

      x += 24 + item.name.length * 6;
    });
  }

  sanitizeSvgAttrs(newSvg);

  return { svgEl: newSvg, width, height: totalH };
}

function exportChart(wrapRef, name, legendItems, format, onError) {
  const originalSvg = wrapRef.current ? wrapRef.current.querySelector('svg') : null;
  if (!originalSvg) {
    onError(`Couldn't find a chart to export for "${name}".`);
    return;
  }
  const fallbackWidth = wrapRef.current.clientWidth || 600;
  const fallbackHeight = wrapRef.current.clientHeight || 300;
  const { svgEl, width, height } = buildExportSvg(originalSvg, legendItems, fallbackWidth, fallbackHeight);
  const serializer = new XMLSerializer();
  const svgStr = serializer.serializeToString(svgEl);

  if (format === 'svg') {
    const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    downloadBlob(svgBlob, `${name}.svg`);
    return;
  }

  let dataUrl;
  try {
    const base64 = btoa(unescape(encodeURIComponent(svgStr)));
    dataUrl = `data:image/svg+xml;base64,${base64}`;
  } catch (err) {
    onError(`Couldn't render "${name}" as PNG. Try SVG export instead.`);
    return;
  }

  const img = new Image();
  img.onload = () => {
    const scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    canvas.toBlob((blob) => {
      if (blob) {
        downloadBlob(blob, `${name}.png`);
        return;
      }
      try {
        const pngDataUrl = canvas.toDataURL('image/png');
        const byteStr = atob(pngDataUrl.split(',')[1]);
        const arr = new Uint8Array(byteStr.length);
        for (let i = 0; i < byteStr.length; i++) arr[i] = byteStr.charCodeAt(i);
        downloadBlob(new Blob([arr], { type: 'image/png' }), `${name}.png`);
      } catch (err) {
        onError(`Couldn't render "${name}" as PNG. Try SVG export instead.`);
      }
    }, 'image/png');
  };
  img.onerror = () => {
    onError(`Couldn't render "${name}" as PNG. Try SVG export instead.`);
  };
  img.src = dataUrl;
}

// ---------- CSV export helpers ----------

function interpolateSeries(xs, ys, xGrid) {
  const out = new Array(xGrid.length).fill('');
  if (!xs || !xs.length) return out;
  let j = 0;
  for (let i = 0; i < xGrid.length; i++) {
    const xq = xGrid[i];
    if (xq < xs[0] || xq > xs[xs.length - 1]) continue;
    while (j < xs.length - 2 && xs[j + 1] < xq) j++;
    const x0 = xs[j], x1 = xs[j + 1], y0 = ys[j], y1 = ys[j + 1];
    out[i] = x1 === x0 ? y0 : y0 + ((xq - x0) / (x1 - x0)) * (y1 - y0);
  }
  return out;
}

function buildLinGrid(min, max, count) {
  if (count < 2 || max <= min) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + i * step);
}

function roundSig(v, sig = 6) {
  if (v === 0 || !isFinite(v)) return v;
  const mag = Math.ceil(Math.log10(Math.abs(v)));
  const factor = Math.pow(10, sig - mag);
  return Math.round(v * factor) / factor;
}

function exportCsvGrid(header, xGrid, seriesList, filename) {
  const columns = seriesList.map((s) => interpolateSeries(s.xs, s.ys, xGrid));
  const rows = [header];
  for (let i = 0; i < xGrid.length; i++) {
    const row = [roundSig(xGrid[i])];
    columns.forEach((col) => row.push(typeof col[i] === 'number' ? roundSig(col[i]) : ''));
    rows.push(row);
  }
  const csv = Papa.unparse(rows);
  downloadBlob(new Blob([csv], { type: 'text/csv;charset=utf-8' }), filename);
}

// ---------- UI ----------

export default function THzAnalyzer() {
  const [datasets, setDatasets] = useState([]);
  const [errors, setErrors] = useState([]);
  const fileInputRef = useRef(null);
  const fftCacheRef = useRef(new Map()); // dataset id -> { time, amplitude, settingsKey, result } — avoids recomputing FFT when only name/color/width/visibility changes
  const timeChartWrapRef = useRef(null);
  const freqChartWrapRef = useRef(null);

  const [windowType, setWindowType] = useState('hann');
  const [zeroPadFactor, setZeroPadFactor] = useState(4);
  const [timeUnit, setTimeUnit] = useState('ps');

  const [noiseRegion, setNoiseRegion] = useState('end');
  const [noiseFraction, setNoiseFraction] = useState(0.2);

  const [bandwidthMode, setBandwidthMode] = useState('peak');
  const [marginDB, setMarginDB] = useState(10);

  const [displayMode, setDisplayMode] = useState('absolute');
  const [showWaterVapor, setShowWaterVapor] = useState(false);
  const [sortKey, setSortKey] = useState(null); // 'peakToPeak' | 'peakFreq' | 'bwWidth' | 'noiseFloorDB' | 'snrDB'
  const [sortDir, setSortDir] = useState('desc'); // 'asc' | 'desc'
  const sessionInputRef = useRef(null);

  // --- zoom / pan state (per chart) ---
  const [timeDomain, setTimeDomain] = useState(null); // null = auto (full range)
  const [freqDomain, setFreqDomain] = useState(null);
  const [timeYDomain, setTimeYDomain] = useState(null);
  const [freqYDomain, setFreqYDomain] = useState(null);
  const [timeMode, setTimeMode] = useState('zoom'); // 'zoom' | 'pan' | 'snapshot'
  const [snapshots, setSnapshots] = useState([]); // [{ id, time, entries: [{id,name,color,value}] }]
  const [freqMode, setFreqMode] = useState('zoom');
  const [timeSel, setTimeSel] = useState({ x1: null, x2: null, y1: null, y2: null });
  const [freqSel, setFreqSel] = useState({ x1: null, x2: null, y1: null, y2: null });
  const panRef = useRef({ dragging: false, startX: 0, startDomain: null, chart: null });
  const timeYScaleRef = useRef(null);
  const freqYScaleRef = useRef(null);
  const emptySel = { x1: null, x2: null, y1: null, y2: null };

  const timeFullDomain = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    datasets.forEach((d) => {
      if (!d.visible) return;
      for (let i = 0; i < d.time.length; i++) {
        if (d.time[i] < lo) lo = d.time[i];
        if (d.time[i] > hi) hi = d.time[i];
      }
    });
    return isFinite(lo) ? [lo, hi] : [0, 1];
  }, [datasets]);
  const freqFullDomain = DEFAULT_FREQ_DOMAIN;

  const resetTimeView = () => { setTimeDomain(null); setTimeYDomain(null); setTimeSel(emptySel); };
  const resetFreqView = () => { setFreqDomain(null); setFreqYDomain(null); setFreqSel(emptySel); };

  const takeSnapshot = (xVal) => {
    if (xVal == null || !Number.isFinite(xVal)) return;
    const entries = visible.map((d) => {
      const val = interpolateSeries(d.time, d.amplitude, [xVal])[0];
      return { id: d.id, name: d.name, color: d.color, value: typeof val === 'number' ? val : null };
    });
    if (!entries.length) return;
    setSnapshots((prev) => [...prev, { id: `snap_${Date.now()}_${Math.random().toString(36).slice(2)}`, time: xVal, entries }]);
  };

  const handleMouseDown = (e, chart) => {
    if (!e) return;
    if (chart === 'time') {
      if (timeMode === 'zoom') {
        const activeYDomain = timeYDomain || timeYFullDomain;
        timeYScaleRef.current = makeYScale(getYPixelRange(timeChartWrapRef.current), activeYDomain);
        const yVal = timeYScaleRef.current ? timeYScaleRef.current.pxToVal(e.chartY) : null;
        setTimeSel({ x1: e.activeLabel, x2: e.activeLabel, y1: yVal, y2: yVal });
      } else if (timeMode === 'snapshot') {
        takeSnapshot(e.activeLabel);
      } else {
        panRef.current = { dragging: true, startX: e.chartX, startDomain: timeDomain || timeFullDomain, chart: 'time' };
      }
    } else {
      if (freqMode === 'zoom') {
        const activeYDomain = freqYDomain || freqYFullDomain;
        freqYScaleRef.current = makeYScale(getYPixelRange(freqChartWrapRef.current), activeYDomain);
        const yVal = freqYScaleRef.current ? freqYScaleRef.current.pxToVal(e.chartY) : null;
        setFreqSel({ x1: e.activeLabel, x2: e.activeLabel, y1: yVal, y2: yVal });
      } else {
        panRef.current = { dragging: true, startX: e.chartX, startDomain: freqDomain || freqFullDomain, chart: 'freq' };
      }
    }
  };

  const handleMouseMove = (e, chart) => {
    if (!e) return;
    if (chart === 'time') {
      if (timeMode === 'zoom' && timeSel.x1 != null) {
        const yScale = timeYScaleRef.current;
        const yVal = yScale ? yScale.pxToVal(e.chartY) : timeSel.y2;
        setTimeSel((sel) => ({ ...sel, x2: e.activeLabel, y2: yVal }));
      } else if (timeMode === 'pan' && panRef.current.dragging && panRef.current.chart === 'time') {
        const wrapper = timeChartWrapRef.current;
        const plotWidth = wrapper ? Math.max(50, wrapper.clientWidth - 85) : 400;
        const [d0, d1] = panRef.current.startDomain;
        const span = d1 - d0;
        const deltaPx = e.chartX - panRef.current.startX;
        const deltaData = -(deltaPx / plotWidth) * span;
        setTimeDomain([d0 + deltaData, d1 + deltaData]);
      }
    } else {
      if (freqMode === 'zoom' && freqSel.x1 != null) {
        const yScale = freqYScaleRef.current;
        const yVal = yScale ? yScale.pxToVal(e.chartY) : freqSel.y2;
        setFreqSel((sel) => ({ ...sel, x2: e.activeLabel, y2: yVal }));
      } else if (freqMode === 'pan' && panRef.current.dragging && panRef.current.chart === 'freq') {
        const wrapper = freqChartWrapRef.current;
        const plotWidth = wrapper ? Math.max(50, wrapper.clientWidth - 85) : 400;
        const [d0, d1] = panRef.current.startDomain;
        const span = d1 - d0;
        const deltaPx = e.chartX - panRef.current.startX;
        const deltaData = -(deltaPx / plotWidth) * span;
        setFreqDomain([Math.max(0, d0 + deltaData), d1 + deltaData]);
      }
    }
  };

  const handleMouseUp = (chart) => {
    if (chart === 'time') {
      if (timeMode === 'zoom') {
        const { x1, x2, y1, y2 } = timeSel;
        if (x1 != null && x2 != null && x1 !== x2) setTimeDomain([Math.min(x1, x2), Math.max(x1, x2)]);
        if (y1 != null && y2 != null && y1 !== y2) setTimeYDomain([Math.min(y1, y2), Math.max(y1, y2)]);
        setTimeSel(emptySel);
      } else {
        panRef.current = { dragging: false, startX: 0, startDomain: null, chart: null };
      }
    } else {
      if (freqMode === 'zoom') {
        const { x1, x2, y1, y2 } = freqSel;
        if (x1 != null && x2 != null && x1 !== x2) setFreqDomain([Math.min(x1, x2), Math.max(x1, x2)]);
        if (y1 != null && y2 != null && y1 !== y2) setFreqYDomain([Math.min(y1, y2), Math.max(y1, y2)]);
        setFreqSel(emptySel);
      } else {
        panRef.current = { dragging: false, startX: 0, startDomain: null, chart: null };
      }
    }
  };

  const handleMouseLeave = (chart) => {
    if (chart === 'time') setTimeSel(emptySel);
    else setFreqSel(emptySel);
    if (panRef.current.chart === chart) panRef.current = { dragging: false, startX: 0, startDomain: null, chart: null };
  };

  const processingOpts = useMemo(() => ({ windowType, zeroPadFactor, timeUnit }), [windowType, zeroPadFactor, timeUnit]);

  const addError = (msg) => setErrors((prev) => [...prev, msg]);
  const dismissError = (i) => setErrors((prev) => prev.filter((_, idx) => idx !== i));

  const handleFiles = useCallback((fileList) => {
    Array.from(fileList).forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const { time, amplitude } = parseFileText(String(e.target.result));
        if (time.length < 8) {
          addError(`Could not read a two-column time/amplitude trace from "${file.name}".`);
          return;
        }
        setDatasets((prev) => [...prev, {
          id: `${file.name}_${Date.now()}_${Math.random().toString(36).slice(2)}`,
          name: file.name.replace(/\.(csv|txt)$/i, ''),
          color: COLORS[prev.length % COLORS.length],
          visible: true,
          width: 1.4,
          time, amplitude,
        }]);
      };
      reader.onerror = () => addError(`Failed to read "${file.name}".`);
      reader.readAsText(file);
    });
  }, []);

  const loadSamples = () => setDatasets((prev) => [...prev, ...makeSampleSet()]);
  const clearAll = () => { fftCacheRef.current.clear(); setDatasets([]); };
  const removeDataset = (id) => { fftCacheRef.current.delete(id); setDatasets((prev) => prev.filter((d) => d.id !== id)); };

  const [dragIndex, setDragIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);

  const handleDragStart = (e, index) => {
    setDragIndex(index);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(index));
  };
  const handleDragOverRow = (e, index) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (dragOverIndex !== index) setDragOverIndex(index);
  };
  const handleDropRow = (e, index) => {
    e.preventDefault();
    if (dragIndex === null || dragIndex === index) {
      setDragIndex(null);
      setDragOverIndex(null);
      return;
    }
    setDatasets((prev) => {
      const arr = [...prev];
      const [moved] = arr.splice(dragIndex, 1);
      arr.splice(index, 0, moved);
      return arr;
    });
    setDragIndex(null);
    setDragOverIndex(null);
  };
  const handleDragEnd = () => {
    setDragIndex(null);
    setDragOverIndex(null);
  };
  const updateDataset = (id, patch) => setDatasets((prev) => prev.map((d) => (d.id === id ? { ...d, ...patch } : d)));

  const handleModeChange = (mode) => {
    setBandwidthMode(mode);
    setMarginDB(mode === 'peak' ? 10 : 6);
  };

  const exportTdsCsv = () => {
    if (datasets.length === 0) { addError('No datasets loaded to export.'); return; }
    let lo = Infinity, hi = -Infinity, minDt = Infinity;
    datasets.forEach((d) => {
      for (let i = 0; i < d.time.length; i++) {
        if (d.time[i] < lo) lo = d.time[i];
        if (d.time[i] > hi) hi = d.time[i];
      }
      for (let i = 1; i < d.time.length; i++) {
        const dt = Math.abs(d.time[i] - d.time[i - 1]);
        if (dt > 0 && dt < minDt) minDt = dt;
      }
    });
    if (!isFinite(lo) || !isFinite(minDt)) { addError('No valid time-domain data to export.'); return; }
    let count = Math.round((hi - lo) / minDt) + 1;
    count = Math.min(Math.max(count, 2), 20000);
    const xGrid = buildLinGrid(lo, hi, count);
    const header = [`Time (${timeUnit})`, ...datasets.map((d) => d.name)];
    const seriesList = datasets.map((d) => ({ xs: d.time, ys: d.amplitude }));
    exportCsvGrid(header, xGrid, seriesList, 'thz_tds_data.csv');
  };

  const exportFftCsv = () => {
    if (datasets.length === 0) { addError('No datasets loaded to export.'); return; }
    const [fLo, fHi] = freqDomain || DEFAULT_FREQ_DOMAIN;
    const xGrid = buildLinGrid(fLo, fHi, 2000);
    const seriesList = datasets.map((d) => {
      const { freqs, mags } = computeFFT(d.time, d.amplitude, processingOpts);
      return { xs: freqs, ys: toDB(mags) };
    });
    const header = ['Frequency (THz)', ...datasets.map((d) => `${d.name} (dB)`)];
    exportCsvGrid(header, xGrid, seriesList, 'thz_fft_data.csv');
  };

  const saveSession = () => {
    if (datasets.length === 0) { addError('No datasets loaded to save.'); return; }
    const session = {
      version: 1,
      datasets: datasets.map((d) => ({
        name: d.name, color: d.color, visible: d.visible, width: d.width, time: d.time, amplitude: d.amplitude,
      })),
      settings: {
        windowType, zeroPadFactor, timeUnit, noiseRegion, noiseFraction, bandwidthMode, marginDB, displayMode, showWaterVapor,
        timeDomain, timeYDomain, freqDomain, freqYDomain,
      },
    };
    const json = JSON.stringify(session);
    downloadBlob(new Blob([json], { type: 'application/json' }), 'thz_session.json');
  };

  const loadSession = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const session = JSON.parse(String(e.target.result));
        if (!session || !Array.isArray(session.datasets)) throw new Error('bad format');
        const restored = session.datasets.map((d, i) => ({
          id: `${d.name || 'dataset'}_${Date.now()}_${i}_${Math.random().toString(36).slice(2)}`,
          name: d.name || `Dataset ${i + 1}`,
          color: d.color || COLORS[i % COLORS.length],
          visible: d.visible !== false,
          width: d.width || 1.4,
          time: Array.isArray(d.time) ? d.time : [],
          amplitude: Array.isArray(d.amplitude) ? d.amplitude : [],
        }));
        fftCacheRef.current.clear();
        setDatasets(restored);
        const s = session.settings || {};
        if (s.windowType) setWindowType(s.windowType);
        if (s.zeroPadFactor) setZeroPadFactor(s.zeroPadFactor);
        if (s.timeUnit) setTimeUnit(s.timeUnit);
        if (s.noiseRegion) setNoiseRegion(s.noiseRegion);
        if (typeof s.noiseFraction === 'number') setNoiseFraction(s.noiseFraction);
        if (s.bandwidthMode) setBandwidthMode(s.bandwidthMode);
        if (typeof s.marginDB === 'number') setMarginDB(s.marginDB);
        if (s.displayMode) setDisplayMode(s.displayMode);
        if (typeof s.showWaterVapor === 'boolean') setShowWaterVapor(s.showWaterVapor);
        setTimeDomain(Array.isArray(s.timeDomain) ? s.timeDomain : null);
        setTimeYDomain(Array.isArray(s.timeYDomain) ? s.timeYDomain : null);
        setFreqDomain(Array.isArray(s.freqDomain) ? s.freqDomain : null);
        setFreqYDomain(Array.isArray(s.freqYDomain) ? s.freqYDomain : null);
        setTimeSel(emptySel);
        setFreqSel(emptySel);
      } catch (err) {
        addError(`Couldn't load "${file.name}" — not a valid session file.`);
      }
    };
    reader.onerror = () => addError(`Failed to read "${file.name}".`);
    reader.readAsText(file);
  };

  const processed = useMemo(() => {
    const freqLoHi = freqDomain || DEFAULT_FREQ_DOMAIN;
    const settingsKey = JSON.stringify([processingOpts, noiseRegion, noiseFraction, bandwidthMode, marginDB, freqLoHi[1], displayMode]);

    return datasets.map((d) => {
      const cached = fftCacheRef.current.get(d.id);
      if (cached && cached.time === d.time && cached.amplitude === d.amplitude && cached.settingsKey === settingsKey) {
        return { ...d, ...cached.result };
      }

      const { freqs, mags } = computeFFT(d.time, d.amplitude, processingOpts);
      const magsDB = toDB(mags);
      const peakIndex = findPeakIndex(mags);
      const peakDB = magsDB[peakIndex];
      const peakFreq = freqs[peakIndex];

      const noiseFloorDB = computeNoiseFloorDB(d.time, d.amplitude, noiseRegion, noiseFraction, processingOpts);
      const snrDB = peakDB - noiseFloorDB;

      let ampMin = d.amplitude[0];
      let ampMax = d.amplitude[0];
      for (let i = 1; i < d.amplitude.length; i++) {
        if (d.amplitude[i] < ampMin) ampMin = d.amplitude[i];
        if (d.amplitude[i] > ampMax) ampMax = d.amplitude[i];
      }
      const peakToPeak = ampMax - ampMin;

      const thresholdDB = bandwidthMode === 'peak' ? peakDB - marginDB : noiseFloorDB + marginDB;
      const bw = computeBandwidth(freqs, magsDB, peakIndex, thresholdDB);

      const strideTime = Math.max(1, Math.floor(d.time.length / 1500));
      const timeChartData = [];
      for (let i = 0; i < d.time.length; i += strideTime) {
        timeChartData.push({ x: d.time[i], y: d.amplitude[i] });
      }

      const freqPlotMax = freqLoHi[1];
      let cutoff = freqs.findIndex((f) => f > freqPlotMax);
      if (cutoff === -1) cutoff = freqs.length;
      const strideFreq = Math.max(1, Math.floor(cutoff / 1500));
      const freqChartData = [];
      for (let i = 0; i < cutoff; i += strideFreq) {
        const val = displayMode === 'normalized' ? magsDB[i] - peakDB : magsDB[i];
        freqChartData.push({ x: freqs[i], y: val });
      }

      const result = { peakFreq, peakDB, noiseFloorDB, snrDB, bw, peakToPeak, timeChartData, freqChartData };
      fftCacheRef.current.set(d.id, { time: d.time, amplitude: d.amplitude, settingsKey, result });
      return { ...d, ...result };
    });
  }, [datasets, processingOpts, noiseRegion, noiseFraction, bandwidthMode, marginDB, freqDomain, displayMode]);

  const visible = processed.filter((d) => d.visible);
  const legendItems = visible.map((d) => ({ name: d.name, color: d.color }));
  const fmt = (v, digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) : '—');
  const roundDisp = (v) => (Number.isFinite(v) ? Number(v.toPrecision(6)) : v);

  const snapshotColumns = useMemo(() => {
    const seen = new Map();
    snapshots.forEach((s) => s.entries.forEach((en) => {
      if (!seen.has(en.id)) seen.set(en.id, { id: en.id, name: en.name, color: en.color });
    }));
    return Array.from(seen.values());
  }, [snapshots]);

  const clearSnapshots = () => setSnapshots([]);
  const removeSnapshot = (id) => setSnapshots((prev) => prev.filter((s) => s.id !== id));

  const sortValueOf = (d, key) => {
    if (key === 'peakToPeak') return d.peakToPeak;
    if (key === 'peakFreq') return d.peakFreq;
    if (key === 'bwWidth') return d.bw.width;
    if (key === 'noiseFloorDB') return d.noiseFloorDB;
    if (key === 'snrDB') return d.snrDB;
    return null;
  };

  const sortedProcessed = useMemo(() => {
    if (!sortKey) return processed;
    const copy = [...processed];
    copy.sort((a, b) => {
      const va = sortValueOf(a, sortKey);
      const vb = sortValueOf(b, sortKey);
      if (!Number.isFinite(va) && !Number.isFinite(vb)) return 0;
      if (!Number.isFinite(va)) return 1;
      if (!Number.isFinite(vb)) return -1;
      return sortDir === 'asc' ? va - vb : vb - va;
    });
    return copy;
  }, [processed, sortKey, sortDir]);

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const sortArrow = (key) => (sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');

  const percentileOf = (sortedVals, p) => {
    if (!sortedVals.length) return NaN;
    const idx = Math.min(sortedVals.length - 1, Math.max(0, Math.round(p * (sortedVals.length - 1))));
    return sortedVals[idx];
  };

  const timeYFullDomain = useMemo(() => {
    let lo = Infinity, hi = -Infinity;
    visible.forEach((d) => {
      d.timeChartData.forEach((p) => {
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      });
    });
    if (!isFinite(lo)) return [-1, 1];
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }, [visible]);

  const freqYFullDomain = useMemo(() => {
    const vals = [];
    visible.forEach((d) => d.freqChartData.forEach((p) => vals.push(p.y)));
    if (!vals.length) return [-40, 0];
    vals.sort((a, b) => a - b);
    // dB spectra can dip to a hard floor (near-zero magnitude bins) that isn't representative —
    // use a robust low percentile instead of the raw minimum so a few outlier dips don't compress the whole view.
    const lo = percentileOf(vals, 0.03);
    const hi = vals[vals.length - 1];
    const pad = (hi - lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }, [visible]);

  const timeXTicks = useMemo(() => {
    const [lo, hi] = timeDomain || timeFullDomain;
    return niceTicks(lo, hi);
  }, [timeDomain, timeFullDomain]);
  const timeYTicks = useMemo(() => {
    const [lo, hi] = timeYDomain || timeYFullDomain;
    return niceTicks(lo, hi);
  }, [timeYDomain, timeYFullDomain]);
  const freqXTicks = useMemo(() => {
    const [lo, hi] = freqDomain || DEFAULT_FREQ_DOMAIN;
    return niceTicks(lo, hi);
  }, [freqDomain]);
  const freqYTicks = useMemo(() => {
    const [lo, hi] = freqYDomain || freqYFullDomain;
    return niceTicks(lo, hi);
  }, [freqYDomain, freqYFullDomain]);

  return (
    <div className="min-h-full w-full bg-white text-slate-800" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
      <div className="border-b border-slate-400 px-6 py-4">
        <p className="text-2xl font-bold tracking-tight text-teal-700 mb-1">NIP - THz Team</p>
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight text-slate-900">THz Waveform &amp; Spectrum Bench</h1>
          <span className="text-xs text-slate-600 font-mono">time-domain · FFT · bandwidth · SNR</span>
        </div>
      </div>

      {errors.length > 0 && (
        <div className="px-6 pt-4 space-y-2">
          {errors.map((msg, i) => (
            <div key={i} className="flex items-center justify-between rounded border border-red-400 bg-red-50 px-3 py-2 text-sm text-red-700">
              <span>{msg}</span>
              <button onClick={() => dismissError(i)} className="text-red-700 hover:text-red-900"><X size={14} /></button>
            </div>
          ))}
        </div>
      )}

      <div className="flex flex-row gap-4 p-6 items-start">
        {/* Left: controls + metrics */}
        <div className="w-80 flex-shrink-0 space-y-4">
          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3">
            <div className="flex gap-2">
              <button
                onClick={() => fileInputRef.current.click()}
                className="flex-1 flex items-center justify-center gap-2 rounded bg-teal-50 border border-teal-400 text-teal-700 text-sm py-2 hover:bg-teal-100 transition"
              >
                <Upload size={14} /> Upload
              </button>
              <button
                onClick={loadSamples}
                className="flex-1 flex items-center justify-center gap-2 rounded bg-white border border-slate-500 text-slate-900 text-sm py-2 hover:bg-slate-100 transition"
              >
                <Sparkles size={14} /> Sample data
              </button>
            </div>
            <input
              ref={fileInputRef} type="file" multiple accept=".csv,.txt" className="hidden"
              onChange={(e) => { handleFiles(e.target.files); e.target.value = null; }}
            />

            <div className="mt-3 space-y-1.5 max-h-96 overflow-y-auto pr-1">
              {datasets.length === 0 && (
                <p className="text-xs text-slate-600 py-2">No datasets loaded. Upload a two-column time/amplitude .csv or .txt file, or load sample data to try the tool.</p>
              )}
              {datasets.map((d, index) => (
                <div
                  key={d.id}
                  onDragOver={(e) => handleDragOverRow(e, index)}
                  onDrop={(e) => handleDropRow(e, index)}
                  className={`rounded bg-white border px-2 py-1.5 space-y-1 transition ${dragOverIndex === index && dragIndex !== null && dragIndex !== index ? 'border-teal-500 border-2' : 'border-slate-400'} ${dragIndex === index ? 'opacity-40' : ''}`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      draggable
                      onDragStart={(e) => handleDragStart(e, index)}
                      onDragEnd={handleDragEnd}
                      title="Drag to reorder"
                      className="cursor-grab active:cursor-grabbing text-slate-400 hover:text-slate-700 flex-shrink-0"
                    >
                      <GripVertical size={13} />
                    </span>
                    <input
                      type="color"
                      value={/^#[0-9a-fA-F]{6}$/.test(d.color) ? d.color : '#000000'}
                      onChange={(e) => updateDataset(d.id, { color: e.target.value })}
                      className="w-5 h-5 rounded border border-slate-400 p-0 bg-transparent cursor-pointer flex-shrink-0"
                      title="Pick line color"
                    />
                    <div className="relative flex-1 min-w-0">
                      <input
                        value={d.name}
                        onChange={(e) => updateDataset(d.id, { name: e.target.value })}
                        title="Click to rename"
                        className="w-full bg-white border border-slate-300 hover:border-slate-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-400/40 rounded pl-1.5 pr-5 py-0.5 text-xs text-slate-800"
                      />
                      <Pencil size={10} className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-400" />
                    </div>
                    <button onClick={() => updateDataset(d.id, { visible: !d.visible })} className="text-slate-600 hover:text-slate-900">
                      {d.visible ? <Eye size={13} /> : <EyeOff size={13} />}
                    </button>
                    <button onClick={() => removeDataset(d.id)} className="text-slate-600 hover:text-red-700">
                      <Trash2 size={13} />
                    </button>
                  </div>
                  <div className="flex items-center gap-2 pl-9 text-xs text-slate-600">
                    <span>Hex</span>
                    <input
                      value={d.color}
                      onChange={(e) => updateDataset(d.id, { color: e.target.value })}
                      onBlur={(e) => {
                        const v = e.target.value.trim();
                        if (!/^#[0-9a-fA-F]{6}$/.test(v)) updateDataset(d.id, { color: /^#[0-9a-fA-F]{6}$/.test(d.color) ? d.color : '#0d9488' });
                      }}
                      spellCheck={false}
                      className="w-20 bg-white border border-slate-400 rounded px-1 py-0.5 text-slate-800 font-mono uppercase"
                    />
                    <span>Width</span>
                    <input
                      type="number" min={0.5} max={6} step={0.5}
                      value={d.width ?? 1.4}
                      onChange={(e) => updateDataset(d.id, { width: Math.max(0.5, Number(e.target.value) || 1.4) })}
                      className="w-14 bg-white border border-slate-400 rounded px-1 py-0.5 text-slate-800"
                    />
                  </div>
                </div>
              ))}
            </div>
            {datasets.length > 0 && (
              <button onClick={clearAll} className="mt-2 text-xs text-slate-600 hover:text-slate-900 underline underline-offset-2">
                Clear all
              </button>
            )}
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Processing</p>
            <div className="space-y-2 text-xs">
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Time unit</span>
                <select value={timeUnit} onChange={(e) => setTimeUnit(e.target.value)} className="bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800">
                  <option value="fs">fs</option>
                  <option value="ps">ps</option>
                  <option value="ns">ns</option>
                  <option value="s">s</option>
                </select>
              </label>
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Window</span>
                <select value={windowType} onChange={(e) => setWindowType(e.target.value)} className="bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800">
                  <option value="none">None</option>
                  <option value="hann">Hann</option>
                  <option value="hamming">Hamming</option>
                  <option value="blackman">Blackman</option>
                </select>
              </label>
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Zero-pad</span>
                <select value={zeroPadFactor} onChange={(e) => setZeroPadFactor(Number(e.target.value))} className="bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800">
                  <option value={1}>1×</option>
                  <option value={2}>2×</option>
                  <option value={4}>4×</option>
                  <option value={8}>8×</option>
                </select>
              </label>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Noise floor</p>
            <div className="space-y-2 text-xs">
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Region</span>
                <select value={noiseRegion} onChange={(e) => setNoiseRegion(e.target.value)} className="bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800">
                  <option value="start">Start of trace</option>
                  <option value="end">End of trace</option>
                </select>
              </label>
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Window size</span>
                <span className="text-slate-900 font-mono">{Math.round(noiseFraction * 100)}%</span>
              </label>
              <input type="range" min={5} max={45} value={noiseFraction * 100} onChange={(e) => setNoiseFraction(Number(e.target.value) / 100)} className="w-full accent-teal-600" />
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Bandwidth</p>
            <div className="space-y-2 text-xs">
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Definition</span>
                <select value={bandwidthMode} onChange={(e) => handleModeChange(e.target.value)} className="bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800">
                  <option value="peak">Relative to peak</option>
                  <option value="noise">Above noise floor</option>
                </select>
              </label>
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">{bandwidthMode === 'peak' ? 'dB below peak' : 'dB above floor'}</span>
                <input
                  type="number" value={marginDB} onChange={(e) => setMarginDB(Number(e.target.value))}
                  className="w-16 bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800 text-right"
                />
              </label>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Spectrum display</p>
            <div className="space-y-2 text-xs">
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Scale</span>
                <select value={displayMode} onChange={(e) => setDisplayMode(e.target.value)} className="bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800">
                  <option value="absolute">Absolute (dB)</option>
                  <option value="normalized">Normalized to own peak</option>
                </select>
              </label>
              <label className="flex items-center gap-2 pt-1">
                <input type="checkbox" checked={showWaterVapor} onChange={(e) => setShowWaterVapor(e.target.checked)} className="accent-teal-600" />
                <span className="text-slate-900">Show water-vapor absorption lines</span>
              </label>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">TDS axis range</p>
            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <label className="space-y-1">
                  <span className="text-slate-900 block">X min ({timeUnit})</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((timeDomain || timeFullDomain)[0])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = timeDomain || timeFullDomain; setTimeDomain([v, cur[1]]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-slate-900 block">X max ({timeUnit})</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((timeDomain || timeFullDomain)[1])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = timeDomain || timeFullDomain; setTimeDomain([cur[0], v]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-slate-900 block">Y min (a.u.)</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((timeYDomain || timeYFullDomain)[0])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = timeYDomain || timeYFullDomain; setTimeYDomain([v, cur[1]]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-slate-900 block">Y max (a.u.)</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((timeYDomain || timeYFullDomain)[1])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = timeYDomain || timeYFullDomain; setTimeYDomain([cur[0], v]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
              </div>
              <button onClick={resetTimeView} className="text-slate-600 hover:text-teal-800 underline underline-offset-2">Reset to auto</button>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">FFT axis range</p>
            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <label className="space-y-1">
                  <span className="text-slate-900 block">X min (THz)</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((freqDomain || DEFAULT_FREQ_DOMAIN)[0])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = freqDomain || DEFAULT_FREQ_DOMAIN; setFreqDomain([v, cur[1]]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-slate-900 block">X max (THz)</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((freqDomain || DEFAULT_FREQ_DOMAIN)[1])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = freqDomain || DEFAULT_FREQ_DOMAIN; setFreqDomain([cur[0], v]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-slate-900 block">Y min (dB)</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((freqYDomain || freqYFullDomain)[0])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = freqYDomain || freqYFullDomain; setFreqYDomain([v, cur[1]]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-slate-900 block">Y max (dB)</span>
                  <input
                    type="number" step="any"
                    value={roundDisp((freqYDomain || freqYFullDomain)[1])}
                    onChange={(e) => { const v = Number(e.target.value); const cur = freqYDomain || freqYFullDomain; setFreqYDomain([cur[0], v]); }}
                    className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                  />
                </label>
              </div>
              <button onClick={resetFreqView} className="text-slate-600 hover:text-teal-800 underline underline-offset-2">Reset to auto</button>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-2">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Export data</p>
            <button
              onClick={exportTdsCsv}
              className="w-full flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-teal-400 hover:bg-teal-50 transition"
            >
              <Download size={12} /> TDS data (.csv)
            </button>
            <button
              onClick={exportFftCsv}
              className="w-full flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-teal-400 hover:bg-teal-50 transition"
            >
              <Download size={12} /> FFT data (.csv)
            </button>
          </div>

          <div className="rounded-lg border border-slate-400 bg-slate-50 p-3 space-y-2">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Session</p>
            <div className="flex gap-2">
              <button
                onClick={saveSession}
                className="flex-1 flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-teal-400 hover:bg-teal-50 transition"
              >
                <Download size={12} /> Save
              </button>
              <button
                onClick={() => sessionInputRef.current.click()}
                className="flex-1 flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-teal-400 hover:bg-teal-50 transition"
              >
                <Upload size={12} /> Load
              </button>
            </div>
            <input
              ref={sessionInputRef} type="file" accept=".json" className="hidden"
              onChange={(e) => { if (e.target.files[0]) loadSession(e.target.files[0]); e.target.value = null; }}
            />
          </div>
        </div>

        {/* Right: plots */}
        <div className="flex-1 min-w-0 space-y-4">
          <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Time domain</p>
              <div className="flex gap-1.5">
                <button
                  onClick={() => setTimeMode('zoom')}
                  title="Drag to zoom into a region"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${timeMode === 'zoom' ? 'bg-teal-100 border-teal-500 text-teal-900' : 'text-slate-800 border-slate-400 hover:border-teal-400 hover:bg-teal-50'}`}
                >
                  <ZoomIn size={12} /> Zoom
                </button>
                <button
                  onClick={() => setTimeMode('pan')}
                  title="Drag to shift the view"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${timeMode === 'pan' ? 'bg-teal-100 border-teal-500 text-teal-900' : 'text-slate-800 border-slate-400 hover:border-teal-400 hover:bg-teal-50'}`}
                >
                  <Move size={12} /> Pan
                </button>
                <button
                  onClick={() => setTimeMode('snapshot')}
                  title="Click points on the plot to record y-values below"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${timeMode === 'snapshot' ? 'bg-teal-100 border-teal-500 text-teal-900' : 'text-slate-800 border-slate-400 hover:border-teal-400 hover:bg-teal-50'}`}
                >
                  <Camera size={12} /> Snapshot
                </button>
                <button
                  onClick={resetTimeView}
                  title="Reset to full view"
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-teal-900 border border-slate-400 rounded px-2 py-1 hover:border-teal-400 hover:bg-teal-50 transition"
                >
                  <RotateCcw size={12} /> Reset
                </button>
                <span className="w-px bg-slate-300 mx-0.5" />
                <button
                  onClick={() => exportChart(timeChartWrapRef, 'thz_time_domain', legendItems, 'png', addError)}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-teal-900 border border-slate-400 rounded px-2 py-1 hover:border-teal-400 hover:bg-teal-50 transition"
                >
                  <Download size={12} /> PNG
                </button>
                <button
                  onClick={() => exportChart(timeChartWrapRef, 'thz_time_domain', legendItems, 'svg', addError)}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-teal-900 border border-slate-400 rounded px-2 py-1 hover:border-teal-400 hover:bg-teal-50 transition"
                >
                  <Download size={12} /> SVG
                </button>
              </div>
            </div>
            <div className="h-96 select-none" ref={timeChartWrapRef} onMouseDown={(e) => e.preventDefault()} style={{ cursor: timeMode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  margin={{ top: 5, right: 15, bottom: 40, left: 0 }}
                  onMouseDown={(e) => handleMouseDown(e, 'time')}
                  onMouseMove={(e) => handleMouseMove(e, 'time')}
                  onMouseUp={() => handleMouseUp('time')}
                  onMouseLeave={() => handleMouseLeave('time')}
                  onDoubleClick={resetTimeView}
                >
                  <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                  <XAxis dataKey="x" type="number" domain={timeDomain || timeFullDomain} ticks={timeXTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }}
                    label={{ value: `Time (${timeUnit})`, position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                  <YAxis domain={timeYDomain || timeYFullDomain} ticks={timeYTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }} width={72}
                    tickFormatter={(v) => (v === 0 ? '0.00e+0' : v.toExponential(2))}
                    label={{ value: 'E-field (a.u.)', angle: -90, position: 'insideLeft', fill: '#334155', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #94a3b8', fontSize: 12 }} labelStyle={{ color: '#1e293b' }} />
                  <Legend verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: 11, paddingTop: 20 }} />
                  <Customized component={ChartBorder} />
                  {visible.map((d) => (
                    <Line key={d.id} data={d.timeChartData} dataKey="y" name={d.name} stroke={d.color} dot={false} isAnimationActive={false} strokeWidth={d.width || 1.4} />
                  ))}
                  {timeMode === 'zoom' && timeSel.x1 != null && timeSel.x2 != null && (
                    <ReferenceArea x1={timeSel.x1} x2={timeSel.x2} y1={timeSel.y1} y2={timeSel.y2} strokeOpacity={0.4} stroke="#0d9488" fill="#0d9488" fillOpacity={0.15} />
                  )}
                  {snapshots.map((s) => (
                    <ReferenceLine key={s.id} x={s.time} stroke="#0f766e" strokeDasharray="2 2" strokeWidth={1} ifOverflow="extendDomain" />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {timeMode === 'snapshot' && (
            <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm overflow-x-auto">
              <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Snapshot points ({snapshots.length})</p>
                <div className="flex gap-1.5">
                  <button
                    onClick={clearSnapshots}
                    disabled={!snapshots.length}
                    className="flex items-center gap-1 text-xs text-slate-800 hover:text-red-700 border border-slate-400 rounded px-2 py-1 hover:border-red-400 hover:bg-red-50 transition disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <Trash2 size={12} /> Clear
                  </button>
                </div>
              </div>
              {snapshots.length === 0 ? (
                <p className="text-xs text-slate-600">Click anywhere on the time-domain plot above to record the y-value of every visible dataset at that x-position.</p>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-slate-600 border-b border-slate-400">
                      <th className="text-right font-normal py-2 pr-4">Time ({timeUnit})</th>
                      {snapshotColumns.map((c) => (
                        <th key={c.id} className="text-right font-normal py-2 pr-4">
                          <span className="inline-flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: c.color }} />
                            {c.name}
                          </span>
                        </th>
                      ))}
                      <th className="w-6" />
                    </tr>
                  </thead>
                  <tbody>
                    {snapshots.map((s) => (
                      <tr key={s.id} className="border-b border-slate-300">
                        <td className="text-right pr-4 font-mono">{fmt(s.time, 4)}</td>
                        {snapshotColumns.map((c) => {
                          const en = s.entries.find((e) => e.id === c.id);
                          return (
                            <td key={c.id} className="text-right pr-4 font-mono">
                              {en && en.value != null ? fmt(en.value, 6) : '—'}
                            </td>
                          );
                        })}
                        <td className="text-right">
                          <button onClick={() => removeSnapshot(s.id)} className="text-slate-400 hover:text-red-600">
                            <X size={12} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Frequency domain (FFT)</p>
              <div className="flex gap-1.5">
                <button
                  onClick={() => setFreqMode('zoom')}
                  title="Drag to zoom into a region"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${freqMode === 'zoom' ? 'bg-teal-100 border-teal-500 text-teal-900' : 'text-slate-800 border-slate-400 hover:border-teal-400 hover:bg-teal-50'}`}
                >
                  <ZoomIn size={12} /> Zoom
                </button>
                <button
                  onClick={() => setFreqMode('pan')}
                  title="Drag to shift the view"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${freqMode === 'pan' ? 'bg-teal-100 border-teal-500 text-teal-900' : 'text-slate-800 border-slate-400 hover:border-teal-400 hover:bg-teal-50'}`}
                >
                  <Move size={12} /> Pan
                </button>
                <button
                  onClick={resetFreqView}
                  title="Reset to full view"
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-teal-900 border border-slate-400 rounded px-2 py-1 hover:border-teal-400 hover:bg-teal-50 transition"
                >
                  <RotateCcw size={12} /> Reset
                </button>
                <span className="w-px bg-slate-300 mx-0.5" />
                <button
                  onClick={() => exportChart(freqChartWrapRef, 'thz_frequency_domain', legendItems, 'png', addError)}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-teal-900 border border-slate-400 rounded px-2 py-1 hover:border-teal-400 hover:bg-teal-50 transition"
                >
                  <Download size={12} /> PNG
                </button>
                <button
                  onClick={() => exportChart(freqChartWrapRef, 'thz_frequency_domain', legendItems, 'svg', addError)}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-teal-900 border border-slate-400 rounded px-2 py-1 hover:border-teal-400 hover:bg-teal-50 transition"
                >
                  <Download size={12} /> SVG
                </button>
              </div>
            </div>
            <div className="h-96 select-none" ref={freqChartWrapRef} onMouseDown={(e) => e.preventDefault()} style={{ cursor: freqMode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  margin={{ top: 5, right: 15, bottom: 40, left: 0 }}
                  onMouseDown={(e) => handleMouseDown(e, 'freq')}
                  onMouseMove={(e) => handleMouseMove(e, 'freq')}
                  onMouseUp={() => handleMouseUp('freq')}
                  onMouseLeave={() => handleMouseLeave('freq')}
                  onDoubleClick={resetFreqView}
                >
                  <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                  <XAxis dataKey="x" type="number" domain={freqDomain || DEFAULT_FREQ_DOMAIN} ticks={freqXTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }}
                    label={{ value: 'Frequency (THz)', position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                  <YAxis domain={freqYDomain || freqYFullDomain} ticks={freqYTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }} width={56}
                    tickFormatter={(v) => v.toFixed(2)}
                    label={{ value: displayMode === 'normalized' ? 'dB (rel. peak)' : 'dB (a.u.)', angle: -90, position: 'insideLeft', fill: '#334155', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #94a3b8', fontSize: 12 }} labelStyle={{ color: '#1e293b' }} />
                  <Legend verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: 11, paddingTop: 20 }} />
                  <Customized component={ChartBorder} />
                  {visible.map((d) => (
                    <Line key={d.id} data={d.freqChartData} dataKey="y" name={d.name} stroke={d.color} dot={false} isAnimationActive={false} strokeWidth={d.width || 1.4} />
                  ))}
                  {freqMode === 'zoom' && freqSel.x1 != null && freqSel.x2 != null && (
                    <ReferenceArea x1={freqSel.x1} x2={freqSel.x2} y1={freqSel.y1} y2={freqSel.y2} strokeOpacity={0.4} stroke="#0d9488" fill="#0d9488" fillOpacity={0.15} />
                  )}
                  {showWaterVapor && WATER_VAPOR_LINES.filter((f) => f >= (freqDomain || DEFAULT_FREQ_DOMAIN)[0] && f <= (freqDomain || DEFAULT_FREQ_DOMAIN)[1]).map((f) => (
                    <ReferenceLine key={f} x={f} stroke="#94a3b8" strokeDasharray="2 3" strokeWidth={1} ifOverflow="extendDomain" />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm overflow-x-auto">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono mb-3">Metrics</p>
            {processed.length === 0 ? (
              <p className="text-xs text-slate-600">Load datasets to see peak frequency, bandwidth, noise floor and SNR here.</p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-600 border-b border-slate-400">
                    <th className="text-left font-normal py-2 pr-4">Dataset</th>
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-teal-800" onClick={() => toggleSort('peakToPeak')}>Peak-to-peak (a.u.){sortArrow('peakToPeak')}</th>
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-teal-800" onClick={() => toggleSort('peakFreq')}>Peak (THz){sortArrow('peakFreq')}</th>
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-teal-800" onClick={() => toggleSort('bwWidth')}>Bandwidth{sortArrow('bwWidth')}</th>
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-teal-800" onClick={() => toggleSort('noiseFloorDB')}>Noise floor (dB){sortArrow('noiseFloorDB')}</th>
                    <th className="text-right font-normal py-2 cursor-pointer select-none hover:text-teal-800" onClick={() => toggleSort('snrDB')}>SNR / DR (dB){sortArrow('snrDB')}</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedProcessed.map((d) => (
                    <tr key={d.id} className={`border-b border-slate-300 ${d.visible ? '' : 'opacity-40'}`}>
                      <td className="py-2.5 pr-4 flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: d.color }} />
                        {d.name}
                      </td>
                      <td className="text-right pr-4 font-mono">{fmt(d.peakToPeak, 4)}</td>
                      <td className="text-right pr-4 font-mono">{fmt(d.peakFreq)}</td>
                      <td className="text-right pr-4 font-mono">{fmt(d.bw.lo)}–{fmt(d.bw.hi)} ({fmt(d.bw.width)})</td>
                      <td className="text-right pr-4 font-mono">{fmt(d.noiseFloorDB, 1)}</td>
                      <td className="text-right font-mono">{fmt(d.snrDB, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>

      <div className="border-t border-slate-400 px-6 py-3">
        <p className="text-[11px] text-slate-500 font-mono">
          Developed by{' '}
          <a
            href="https://github.com/vpjuguilon"
            target="_blank"
            rel="noopener noreferrer"
            className="text-teal-700 hover:text-teal-900 underline underline-offset-2"
          >
            vpjuguilon
          </a>
          {' '}· 2026
        </p>
      </div>
    </div>
  );
}
