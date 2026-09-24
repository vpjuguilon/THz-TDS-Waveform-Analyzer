import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import logoUrl from './assets/logo.png';
import Papa from 'papaparse';
import * as math from 'mathjs';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea, ReferenceLine, Customized, ComposedChart, Scatter,
} from 'recharts';
import { Upload, Trash2, Eye, EyeOff, Sparkles, X, Download, ZoomIn, Move, RotateCcw, Pencil, Camera, GripVertical, Send, Check } from 'lucide-react';

const COLORS = ['#0d9488', '#d97706', '#db2777', '#4f46e5', '#65a30d', '#ea580c', '#0284c7', '#dc2626'];

// Default frequency-domain view range (THz) used before the user sets an explicit range or zooms.
const DEFAULT_FREQ_DOMAIN = [0, 6];

// Common strong atmospheric water-vapor absorption lines in the THz range (THz), commonly cited in THz-TDS work.
const WATER_VAPOR_LINES = [0.557, 0.752, 0.988, 1.097, 1.113, 1.163, 1.208, 1.229, 1.412, 1.602, 1.669, 1.718, 1.796, 1.867];

// Marker shapes available for the power-dependence scatter plot (native recharts Scatter shapes).
const MARKER_TYPES = ['circle', 'square', 'diamond', 'triangle', 'star', 'cross', 'wye'];

// Spreadsheet-style letter labels for snapshot points: a, b, ..., z, aa, ab, ...
function indexToLetters(i) {
  let s = '';
  let n = i;
  do {
    s = String.fromCharCode(97 + (n % 26)) + s;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return s;
}

// Fits amplitude(P) = Amax * P / (P + Psat) via damped Gauss-Newton (Levenberg-Marquardt style).
// Only 2 free parameters, so a hand-rolled solver is simple and fast enough for this data size.
//
// The fit runs on data normalized to order 1 (x / max(x), y / max(y)) and the parameters are
// rescaled afterwards. This makes the solver scale-invariant: its internal thresholds (singular
// determinant, convergence) are absolute numbers, which previously caused the solver to abort on
// its first iteration whenever the data were far from order 1 — e.g. small lock-in amplitudes
// (~1e-4) plotted against power in mW — silently returning the initial guess (Psat = median power)
// instead of a real fit. Whether it failed depended on the axis units, which is why the fit
// appeared to work in fluence mode but not in power mode.
function fitSaturationCurve(powers, amps) {
  const raw = powers.map((p, i) => ({ p, a: amps[i] })).filter((pt) => Number.isFinite(pt.p) && Number.isFinite(pt.a) && pt.p > 0);
  if (raw.length < 2) return null;

  const xScale = Math.max(...raw.map((pt) => pt.p));
  const yScale = Math.max(...raw.map((pt) => Math.abs(pt.a))) || 1;
  if (!(xScale > 0) || !(yScale > 0)) return null;
  const pts = raw.map((pt) => ({ p: pt.p / xScale, a: pt.a / yScale }));

  let Amax = Math.max(...pts.map((pt) => pt.a)) * 1.3 || 1;
  const sortedP = [...pts.map((pt) => pt.p)].sort((a, b) => a - b);
  let Psat = sortedP[Math.floor(sortedP.length / 2)] || 1;
  if (!(Psat > 0)) Psat = 1;

  let lambda = 1e-3;
  let prevSSE = Infinity;

  for (let iter = 0; iter < 500; iter++) {
    let JTJ00 = 0, JTJ01 = 0, JTJ11 = 0, JTr0 = 0, JTr1 = 0, sse = 0;
    for (const { p, a } of pts) {
      const denom = p + Psat;
      const model = (Amax * p) / denom;
      const r = model - a;
      const dAmax = p / denom;
      const dPsat = -(Amax * p) / (denom * denom);
      JTJ00 += dAmax * dAmax;
      JTJ01 += dAmax * dPsat;
      JTJ11 += dPsat * dPsat;
      JTr0 += dAmax * r;
      JTr1 += dPsat * r;
      sse += r * r;
    }
    const a00 = JTJ00 * (1 + lambda);
    const a11 = JTJ11 * (1 + lambda);
    const det = a00 * a11 - JTJ01 * JTJ01;
    // Relative singularity test (scale-free), not an absolute cutoff.
    if (!(Math.abs(det) > 1e-14 * Math.abs(a00 * a11))) { lambda *= 3; if (lambda > 1e8) break; continue; }
    const deltaAmax = (-JTr0 * a11 + JTr1 * JTJ01) / det;
    const deltaPsat = (-a00 * JTr1 + JTJ01 * JTr0) / det;
    const nextAmax = Amax + deltaAmax;
    const nextPsat = Psat + deltaPsat > 0 ? Psat + deltaPsat : Psat * 0.5;

    let nextSSE = 0;
    for (const { p, a } of pts) {
      const model = (nextAmax * p) / (p + nextPsat);
      nextSSE += (model - a) * (model - a);
    }
    if (Number.isFinite(nextSSE) && nextSSE < sse) {
      Amax = nextAmax;
      Psat = nextPsat;
      lambda = Math.max(lambda * 0.6, 1e-8);
      // Relative convergence: stop once an accepted step improves SSE by < 1e-12 of its value.
      if (Math.abs(prevSSE - nextSSE) <= 1e-12 * Math.max(nextSSE, 1e-30)) { prevSSE = nextSSE; break; }
      prevSSE = nextSSE;
    } else {
      lambda *= 3;
      if (lambda > 1e8) break;
    }
  }

  if (!Number.isFinite(Amax) || !Number.isFinite(Psat) || Psat <= 0) return null;
  return { Amax: Amax * yScale, Psat: Psat * xScale };
}

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
  const re = new Array(half);
  const im = new Array(half);
  for (let k = 0; k < half; k++) {
    freqs[k] = k / (paddedLen * dt_ps); // THz, since dt_ps is in picoseconds
    const c = spectrum[k];
    mags[k] = Math.hypot(c.re, c.im);
    re[k] = c.re;
    im[k] = c.im;
  }
  return { freqs, mags, re, im };
}

// Converts magnitudes to dB using a floor set RELATIVE to this spectrum's own peak
// (~ -300 dB below it) rather than an absolute constant. An absolute clamp produced a
// fixed -240 dB artifact whenever real magnitudes approached it, which then distorted
// axis auto-scaling downstream.
function toDB(mags) {
  let peak = 0;
  for (let i = 0; i < mags.length; i++) if (mags[i] > peak) peak = mags[i];
  const floor = peak > 0 ? peak * 1e-15 : Number.MIN_VALUE;
  return mags.map((m) => 20 * Math.log10(Math.max(m, floor)));
}

function findPeakIndex(mags) {
  let idx = 0;
  for (let i = 1; i < mags.length; i++) if (mags[i] > mags[idx]) idx = i;
  return idx;
}

// Refines the peak location to sub-bin precision by fitting a parabola through the peak
// bin and its two neighbours (done in dB, where a spectral peak is closer to quadratic).
// A raw single-bin argmax jitters between adjacent bins on noisy broadband spectra, which
// makes peak frequency unstable when comparing datasets.
function refinePeakFreq(freqs, magsDB, peakIndex) {
  if (peakIndex <= 0 || peakIndex >= magsDB.length - 1) return freqs[peakIndex];
  const yL = magsDB[peakIndex - 1];
  const y0 = magsDB[peakIndex];
  const yR = magsDB[peakIndex + 1];
  const denom = yL - 2 * y0 + yR;
  if (!Number.isFinite(denom) || Math.abs(denom) < 1e-12) return freqs[peakIndex];
  let delta = (0.5 * (yL - yR)) / denom;
  if (!Number.isFinite(delta) || Math.abs(delta) > 1) return freqs[peakIndex];
  const df = freqs[peakIndex + 1] - freqs[peakIndex];
  return freqs[peakIndex] + delta * df;
}

function interpCrossing(f1, m1, f2, m2, target) {
  if (m2 === m1) return f1;
  const t = (target - m1) / (m2 - m1);
  return f1 + t * (f2 - f1);
}

// Walks outward from the peak to find where the spectrum drops below `thresholdDB`.
// A crossing only counts if the spectrum STAYS below the threshold for `persist`
// consecutive bins — otherwise a single noisy bin, or a narrow absorption notch (e.g. a
// water-vapour line), would prematurely truncate the reported bandwidth.
function computeBandwidth(freqs, magsDB, peakIndex, thresholdDB, persist = 5) {
  const staysBelow = (start, step) => {
    for (let k = 0; k < persist; k++) {
      const j = start + k * step;
      if (j < 0 || j >= magsDB.length) return true; // ran off the end: treat as a real crossing
      if (magsDB[j] >= thresholdDB) return false;
    }
    return true;
  };

  let lo = freqs[0];
  let hi = freqs[freqs.length - 1];

  for (let i = peakIndex; i > 0; i--) {
    if (magsDB[i] >= thresholdDB && magsDB[i - 1] < thresholdDB && staysBelow(i - 1, -1)) {
      lo = interpCrossing(freqs[i - 1], magsDB[i - 1], freqs[i], magsDB[i], thresholdDB);
      break;
    }
  }
  for (let i = peakIndex; i < magsDB.length - 1; i++) {
    if (magsDB[i] >= thresholdDB && magsDB[i + 1] < thresholdDB && staysBelow(i + 1, 1)) {
      hi = interpCrossing(freqs[i], magsDB[i], freqs[i + 1], magsDB[i + 1], thresholdDB);
      break;
    }
  }
  return { lo, hi, width: Math.max(0, hi - lo) };
}

// Estimates the noise spectral floor from a signal-free stretch of the trace.
//
// Two corrections matter here, and both previously biased the floor LOW (inflating the
// reported SNR/dynamic range by roughly 9 dB combined):
//   1. An unnormalized DFT's magnitude scales as sqrt(N), so a short segment produces a
//      smaller magnitude than the same noise would over the full trace. Since the peak is
//      measured from the full trace, the segment must be scaled by sqrt(N_full/N_seg) to
//      be comparable.
//   2. Averaging in dB (mean of logs) sits below the true level because log is concave.
//      Averaging power in linear units and converting once at the end is unbiased.
function computeNoiseFloorDB(time, amplitude, region, fraction, opts) {
  const N = time.length;
  const n = Math.max(8, Math.floor(N * fraction));
  const segTime = region === 'end' ? time.slice(N - n) : time.slice(0, n);
  const segAmp = region === 'end' ? amplitude.slice(N - n) : amplitude.slice(0, n);
  const { mags } = computeFFT(segTime, segAmp, opts);
  if (!mags.length) return -Infinity;

  const lengthScale = Math.sqrt(N / n);
  let sumSq = 0;
  for (let i = 0; i < mags.length; i++) {
    const m = mags[i] * lengthScale;
    sumSq += m * m;
  }
  const rms = Math.sqrt(sumSq / mags.length);
  return 20 * Math.log10(Math.max(rms, Number.MIN_VALUE));
}

// ---------- sample data ----------

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
// A number input for controlled numeric ranges (axis min/max etc.) that keeps its own draft
// text while the user is typing. Binding a number input directly to a parsed Number value
// causes intermediate states like "-" or "5." to get wiped on every keystroke's re-render,
// making it impossible to type a negative number. This commits to the parent only once the
// draft parses to a real number, and re-syncs from the parent when it changes externally
// (e.g. a drag-zoom or Reset), without clobbering an in-progress edit.
// Circular header logo. Expects the image at src/assets/logo.png (imported above, so Vite
// inlines it into the bundle and the single-file build keeps working).
// `rounded-full` crops it to a circle and `object-cover` prevents distortion if the source
// image is not perfectly square, so it needs no pre-cropping.
function TeamLogo({ sizeClass = 'w-20 h-20' }) {
  return (
    <img
      src={logoUrl}
      alt="NIP - THz Team"
      className={`${sizeClass} flex-shrink-0 rounded-full object-cover`}
    />
  );
}

function NumberRangeField({ value, onCommit, className }) {
  const [text, setText] = useState(() => (Number.isFinite(value) ? String(value) : ''));

  useEffect(() => {
    const parsed = Number(text);
    if (!(Number.isFinite(parsed) && Math.abs(parsed - value) < 1e-9)) {
      setText(Number.isFinite(value) ? String(value) : '');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const handleChange = (e) => {
    const v = e.target.value;
    setText(v);
    if (v === '' || v === '-') return; // let the user keep typing
    const parsed = Number(v);
    if (Number.isFinite(parsed)) onCommit(parsed);
  };

  return <input type="number" step="any" value={text} onChange={handleChange} className={className} />;
}

function ChartBorder({ offset }) {
  if (!offset) return null;
  return (
    <rect
      x={offset.left} y={offset.top} width={offset.width} height={offset.height}
      fill="none" stroke="#000000" strokeWidth={0.75} pointerEvents="none"
    />
  );
}

// Returns `d` only if it is a usable [min, max] pair. Guards against a domain that got
// set to NaN (e.g. from a click where the chart reported no cursor position), which would
// otherwise blank out the axis ticks and the range input fields until a manual reset.
// Note: an inverted range (min > max) is deliberately allowed through — that is a valid
// intermediate state while someone is typing into the min/max fields, and rejecting it
// here would snap their partially-entered value back.
// Formats tooltip values so they match their axis rather than dumping raw float precision
// (e.g. -0.00399021127298511). `mode` mirrors the corresponding axis tickFormatter.
function fmtTip(v, mode) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return v;
  if (mode === 'db') return `${v.toFixed(2)} dB`;
  if (mode === 'sci') return v === 0 ? '0.000e+0' : v.toExponential(3);
  return Math.abs(v) >= 0.01 || v === 0 ? v.toFixed(4) : v.toExponential(3);
}

function fmtTipLabel(l, unit, digits = 3) {
  const n = Number(l);
  return Number.isFinite(n) ? `${n.toFixed(digits)} ${unit}` : l;
}

function validDomain(d) {
  if (!Array.isArray(d) || d.length !== 2) return null;
  if (!Number.isFinite(d[0]) || !Number.isFinite(d[1])) return null;
  return d;
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

// Lets the person pick a folder and file name via the browser's native Save As dialog
// (Chrome/Edge). Falls back to a plain auto-download (browser's default downloads folder,
// fixed name) on browsers that don't support the File System Access API, e.g. Firefox/Safari.
async function saveFile(blob, suggestedName, description, mimeType, extensions) {
  if (typeof window !== 'undefined' && window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName,
        types: [{ description, accept: { [mimeType]: extensions } }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    } catch (err) {
      if (err && err.name === 'AbortError') return false; // user cancelled the dialog — not an error
      // Fall through to plain download if the picker itself failed for some other reason.
    }
  }
  downloadBlob(blob, suggestedName);
  return true;
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
  const PAD = 18; // breathing room between the plot border and the image edge
  const innerW = width;
  const innerH = contentBottom + legendH;
  const totalW = innerW + PAD * 2;
  const totalH = innerH + PAD * 2;

  const newSvg = document.createElementNS(ns, 'svg');
  newSvg.setAttribute('xmlns', ns);
  newSvg.setAttribute('width', totalW);
  newSvg.setAttribute('height', totalH);
  newSvg.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);
  newSvg.setAttribute('font-family', 'Inter, Arial, Helvetica, sans-serif');
  newSvg.style.fontFamily = 'Inter, Arial, Helvetica, sans-serif';

  const bg = document.createElementNS(ns, 'rect');
  bg.setAttribute('x', 0);
  bg.setAttribute('y', 0);
  bg.setAttribute('width', totalW);
  bg.setAttribute('height', totalH);
  bg.setAttribute('fill', '#ffffff');
  newSvg.appendChild(bg);

  const cloned = originalSvg.cloneNode(true);
  // Strip leftover hover/tooltip artifacts (e.g. the cursor guide line, active-point highlight)
  // that shouldn't appear in a static export if the mouse happened to be over the chart.
  cloned.querySelectorAll('.recharts-tooltip-cursor, .recharts-active-dot').forEach((el) => el.remove());
  const g = document.createElementNS(ns, 'g');
  g.setAttribute('transform', `translate(${PAD}, ${PAD})`);
  while (cloned.firstChild) g.appendChild(cloned.firstChild);
  newSvg.appendChild(g);

  if (legendItems.length) {
    let x = 10 + PAD;
    const y = contentBottom + 22 + PAD;
    legendItems.forEach((item) => {
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', x);
      rect.setAttribute('y', y - 9);
      rect.setAttribute('width', 10);
      rect.setAttribute('height', 10);
      rect.setAttribute('fill', item.color);
      newSvg.appendChild(rect);

      const parts = item.parts || [{ text: item.name }];
      const text = document.createElementNS(ns, 'text');
      text.setAttribute('x', x + 14);
      text.setAttribute('y', y);
      text.setAttribute('font-size', '11');
      text.setAttribute('font-family', 'Inter, sans-serif');
      text.setAttribute('fill', '#334155');
      parts.forEach((part) => {
        const tspan = document.createElementNS(ns, 'tspan');
        if (part.dy != null) tspan.setAttribute('dy', part.dy);
        if (part.fontSize != null) tspan.setAttribute('font-size', part.fontSize);
        tspan.textContent = part.text;
        text.appendChild(tspan);
      });
      newSvg.appendChild(text);

      const totalLen = parts.reduce((sum, p) => sum + p.text.length, 0);
      x += 24 + totalLen * 6;
    });
  }

  sanitizeSvgAttrs(newSvg);

  return { svgEl: newSvg, width: totalW, height: totalH };
}

async function exportChart(wrapRef, name, legendItems, format, onError, dims) {
  const el = wrapRef.current;
  if (!el) {
    onError(`Couldn't find a chart to export for "${name}".`);
    return;
  }

  const resizing = dims && dims.width > 0 && dims.height > 0;
  let restoreWidth = null;
  let restoreHeight = null;
  if (resizing) {
    restoreWidth = el.style.width;
    restoreHeight = el.style.height;
    el.style.width = `${dims.width}px`;
    el.style.height = `${dims.height}px`;
    // Give ResponsiveContainer's resize observer (and recharts' own re-render) time to catch up
    // to the new size before we read the SVG — otherwise we'd capture the old layout.
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await new Promise((resolve) => setTimeout(resolve, 60));
  }

  try {
    const originalSvg = el.querySelector('svg');
    if (!originalSvg) {
      onError(`Couldn't find a chart to export for "${name}".`);
      return;
    }
    const fallbackWidth = resizing ? dims.width : (el.clientWidth || 600);
    const fallbackHeight = resizing ? dims.height : (el.clientHeight || 300);
    const { svgEl, width, height } = buildExportSvg(originalSvg, legendItems, fallbackWidth, fallbackHeight);
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgEl);

    if (format === 'svg') {
      const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
      saveFile(svgBlob, `${name}.svg`, 'SVG image', 'image/svg+xml', ['.svg']);
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

    await new Promise((resolve) => {
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
            saveFile(blob, `${name}.png`, 'PNG image', 'image/png', ['.png']);
            resolve();
            return;
          }
          try {
            const pngDataUrl = canvas.toDataURL('image/png');
            const byteStr = atob(pngDataUrl.split(',')[1]);
            const arr = new Uint8Array(byteStr.length);
            for (let i = 0; i < byteStr.length; i++) arr[i] = byteStr.charCodeAt(i);
            saveFile(new Blob([arr], { type: 'image/png' }), `${name}.png`, 'PNG image', 'image/png', ['.png']);
          } catch (err) {
            onError(`Couldn't render "${name}" as PNG. Try SVG export instead.`);
          }
          resolve();
        }, 'image/png');
      };
      img.onerror = () => {
        onError(`Couldn't render "${name}" as PNG. Try SVG export instead.`);
        resolve();
      };
      img.src = dataUrl;
    });
  } finally {
    if (resizing) {
      el.style.width = restoreWidth;
      el.style.height = restoreHeight;
    }
  }
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

// ---------- Tab 3: complex FFT-domain convolution/deconvolution helpers ----------

function complexMultiply(aRe, aIm, bRe, bIm) {
  return { re: aRe * bRe - aIm * bIm, im: aRe * bIm + aIm * bRe };
}

function complexDivide(aRe, aIm, bRe, bIm) {
  const denom = bRe * bRe + bIm * bIm;
  if (denom < 1e-30) return { re: 0, im: 0 };
  return { re: (aRe * bRe + aIm * bIm) / denom, im: (aIm * bRe - aRe * bIm) / denom };
}

function unwrapPhase(phase) {
  const out = phase.slice();
  for (let i = 1; i < out.length; i++) {
    let diff = out[i] - out[i - 1];
    while (diff > Math.PI) { out[i] -= 2 * Math.PI; diff = out[i] - out[i - 1]; }
    while (diff < -Math.PI) { out[i] += 2 * Math.PI; diff = out[i] - out[i - 1]; }
  }
  return out;
}

// Finds contiguous [start,end] frequency spans where `flags[i]` is true, for shading
// the frequency range where deconvolution is numerically unreliable.
function findTrueSpans(freqs, flags) {
  const spans = [];
  let start = null;
  for (let i = 0; i < flags.length; i++) {
    if (flags[i] && start === null) start = freqs[i];
    if (!flags[i] && start !== null) { spans.push([start, freqs[i - 1]]); start = null; }
  }
  if (start !== null) spans.push([start, freqs[freqs.length - 1]]);
  return spans;
}

// Resolves a card's source id to either a Tab 1 dataset (computing its FFT fresh) or an
// earlier Tab 3 card's already-computed complex spectrum.
function resolveConvSource(id, datasetsList, processingOptsLocal, resolvedSoFar, rawFftCache) {
  if (!id) return null;
  const ds = datasetsList.find((d) => d.id === id);
  if (ds) {
    const settingsKey = JSON.stringify(processingOptsLocal);
    const cached = rawFftCache.get(id);
    if (cached && cached.time === ds.time && cached.amplitude === ds.amplitude && cached.settingsKey === settingsKey) {
      return { freqs: cached.freqs, re: cached.re, im: cached.im, mags: cached.mags, name: ds.name };
    }
    const { freqs, re, im, mags } = computeFFT(ds.time, ds.amplitude, processingOptsLocal);
    rawFftCache.set(id, { time: ds.time, amplitude: ds.amplitude, settingsKey, freqs, re, im, mags });
    return { freqs, re, im, mags, name: ds.name };
  }
  const card = resolvedSoFar.find((r) => r.id === id);
  if (card && card.freqs) return { freqs: card.freqs, re: card.re, im: card.im, mags: card.mags, name: card.name };
  return null;
}

// Robust dB y-domain (ignores the bottom ~3% of values), same fix used for Tab 1's FFT axis
// to avoid the numerical floor-clamp artifact skewing the auto-range.
function computeDbYDomain(magsDB) {
  if (!magsDB || !magsDB.length) return [-40, 0];
  const vals = magsDB.slice().sort((a, b) => a - b);
  const lo = vals[Math.max(0, Math.floor(0.03 * (vals.length - 1)))];
  const hi = vals[vals.length - 1];
  const pad = (hi - lo) * 0.1 || 1;
  return [lo - pad, hi + pad];
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
  saveFile(new Blob([csv], { type: 'text/csv;charset=utf-8' }), filename, 'CSV file', 'text/csv', ['.csv']);
}

// ---------- Tab 4: time-domain waveform arithmetic helpers ----------

const ARITH_OPS = [
  { value: 'none', label: 'As is (A)', symbol: '' },
  { value: 'flip', label: 'Flip (−A)', symbol: '−' },
  { value: 'add', label: 'Add (A + B)', symbol: '+' },
  { value: 'subtract', label: 'Subtract (A − B)', symbol: '−' },
];
const arithNeedsB = (op) => op === 'add' || op === 'subtract';
// Muted colors for the dashed input-waveform overlays (A, B), so the result line stays dominant.
const INPUT_COLORS = ['#64748b', '#b45309'];

// Returns the trace with time ascending (interpolation below assumes it).
function ascendingTrace(time, amplitude) {
  if (time.length < 2 || time[time.length - 1] >= time[0]) return { time, amplitude };
  return { time: time.slice().reverse(), amplitude: amplitude.slice().reverse() };
}

// Applies a time-domain operation to waveform A (and B for add/subtract).
// For add/subtract, B is linearly interpolated onto A's time samples and the result is limited
// to the time range where both traces exist, so traces with different step sizes or scan
// windows can still be combined. `notes` records any resampling/trimming so the UI can say so.
function combineWaveforms(op, a, b) {
  const A = ascendingTrace(a.time, a.amplitude);
  if (op === 'none') return { time: A.time, amplitude: A.amplitude.slice(), notes: [] };
  if (op === 'flip') return { time: A.time, amplitude: A.amplitude.map((v) => -v), notes: [] };
  if (!b) return { error: 'Select Dataset B.' };

  const B = ascendingTrace(b.time, b.amplitude);
  const sign = op === 'subtract' ? -1 : 1;
  const notes = [];

  const sameGrid = A.time.length === B.time.length && A.time.every((t, i) => {
    const tol = 1e-9 * Math.max(Math.abs(t), Math.abs(A.time[A.time.length - 1] - A.time[0]), 1e-30);
    return Math.abs(t - B.time[i]) <= tol;
  });
  if (sameGrid) {
    return { time: A.time, amplitude: A.amplitude.map((v, i) => v + sign * B.amplitude[i]), bOnGrid: B.amplitude, notes };
  }

  const bOnA = interpolateSeries(B.time, B.amplitude, A.time);
  const time = [];
  const amplitude = [];
  const bOnGrid = [];
  for (let i = 0; i < A.time.length; i++) {
    if (typeof bOnA[i] !== 'number') continue;
    time.push(A.time[i]);
    amplitude.push(A.amplitude[i] + sign * bOnA[i]);
    bOnGrid.push(bOnA[i]);
  }
  if (time.length < 2) return { error: 'Datasets A and B do not overlap in time.' };
  notes.push("B was resampled onto A's time points (linear interpolation).");
  if (time.length < A.time.length) {
    notes.push(`Result limited to the overlapping time range ${roundSig(time[0], 4)} to ${roundSig(time[time.length - 1], 4)}.`);
  }
  return { time, amplitude, bOnGrid, notes };
}

function downsampleXY(xs, ys, maxPoints = 1500) {
  const stride = Math.max(1, Math.floor(xs.length / maxPoints));
  const out = [];
  for (let i = 0; i < xs.length; i += stride) out.push({ x: xs[i], y: ys[i] });
  return out;
}

function paddedRange(seriesList) {
  let lo = Infinity, hi = -Infinity;
  seriesList.forEach((ys) => ys.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; }));
  if (!Number.isFinite(lo)) return [-1, 1];
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
  return [lo - pad, hi + pad];
}

// ---------- UI ----------

export default function THzAnalyzer() {
  const [activeTab, setActiveTab] = useState('tds-fft'); // 'tds-fft' | 'power-dep'

  const [datasets, setDatasets] = useState([]);
  const [errors, setErrors] = useState([]);
  const fileInputRef = useRef(null);
  const fftCacheRef = useRef(new Map()); // dataset id -> { time, amplitude, settingsKey, result } — avoids recomputing FFT when only name/color/width/visibility changes
  const timeChartWrapRef = useRef(null);
  const freqChartWrapRef = useRef(null);

  const [windowType, setWindowType] = useState('none');
  const [zeroPadFactor, setZeroPadFactor] = useState(1);
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
  const [sessionName, setSessionName] = useState('');
  const [sessionStatus, setSessionStatus] = useState(null); // 'saved' | 'loaded' | null

  // --- export resolution/aspect-ratio dialog ---
  const [exportDialog, setExportDialog] = useState(null); // { wrapRef, name, legendItems, format } | null
  const [exportW, setExportW] = useState(1200);
  const [exportH, setExportH] = useState(700);

  const openExportDialog = (wrapRef, name, legendItems, format) => {
    const el = wrapRef.current;
    setExportW(el && el.clientWidth ? Math.round(el.clientWidth) : 1200);
    setExportH(el && el.clientHeight ? Math.round(el.clientHeight) : 700);
    setExportDialog({ wrapRef, name, legendItems, format });
  };

  const confirmExport = () => {
    if (!exportDialog) return;
    const { wrapRef, name, legendItems, format } = exportDialog;
    const w = Math.max(200, Math.min(4000, Math.round(Number(exportW)) || 1200));
    const h = Math.max(200, Math.min(4000, Math.round(Number(exportH)) || 700));
    exportChart(wrapRef, name, legendItems, format, addError, { width: w, height: h });
    setExportDialog(null);
  };

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

  // --- Tab 2: power dependence ---
  const makeEmptyRows = (n) => Array.from({ length: n }, () => ({ power: '', min: '', max: '' }));
  const makePowerDataset = (idx, rowCount) => ({
    id: `pd_${Date.now()}_${idx}_${Math.random().toString(36).slice(2)}`,
    name: `Dataset ${idx + 1}`,
    color: COLORS[idx % COLORS.length],
    marker: MARKER_TYPES[idx % MARKER_TYPES.length],
    rows: makeEmptyRows(rowCount),
  });

  const [numPowers, setNumPowers] = useState(5);
  const [numPowerDatasets, setNumPowerDatasets] = useState(1);
  const [powerDatasets, setPowerDatasets] = useState(() => [makePowerDataset(0, 5)]);
  const powerChartWrapRef = useRef(null);

  const [powerXUnit, setPowerXUnit] = useState('mW'); // 'mW' | 'fluence'
  const [showPowerFit, setShowPowerFit] = useState(true); // show/hide saturation-fit curves and Psat labels
  const [laserRepRate, setLaserRepRate] = useState(80); // MHz
  const [laserPulseDuration, setLaserPulseDuration] = useState(100); // fs (not used in fluence itself — reserved for a future intensity mode)
  const [laserSpotDiameter, setLaserSpotDiameter] = useState(2); // um

  // Fluence [mJ/cm^2] = 400 * P[mW] / (pi * f_rep[MHz] * d[um]^2)
  const convertPowerX = (mw) => {
    if (powerXUnit !== 'fluence') return mw;
    const rep = Number(laserRepRate), d = Number(laserSpotDiameter);
    if (!Number.isFinite(mw) || !Number.isFinite(rep) || !Number.isFinite(d) || rep <= 0 || d <= 0) return NaN;
    return (400 * mw) / (Math.PI * rep * d * d);
  };

  const handleNumPowersChange = (n) => {
    const count = Math.max(1, Math.min(200, Math.round(Number(n)) || 1));
    setNumPowers(count);
    setPowerDatasets((prev) => prev.map((ds) => {
      const rows = ds.rows.slice(0, count);
      while (rows.length < count) rows.push({ power: '', min: '', max: '' });
      return { ...ds, rows };
    }));
  };

  const handleNumPowerDatasetsChange = (n) => {
    const count = Math.max(1, Math.min(12, Math.round(Number(n)) || 1));
    setNumPowerDatasets(count);
    setPowerDatasets((prev) => {
      const arr = prev.slice(0, count);
      while (arr.length < count) arr.push(makePowerDataset(arr.length, numPowers));
      return arr;
    });
  };

  const updatePowerDataset = (id, patch) => setPowerDatasets((prev) => prev.map((ds) => (ds.id === id ? { ...ds, ...patch } : ds)));
  const updatePowerRow = (dsId, rowIndex, field, value) => {
    setPowerDatasets((prev) => prev.map((ds) => {
      if (ds.id !== dsId) return ds;
      const rows = ds.rows.map((r, i) => (i === rowIndex ? { ...r, [field]: value } : r));
      return { ...ds, rows };
    }));
  };

  const clearPowerDatasetValues = (dsId) => {
    setPowerDatasets((prev) => prev.map((ds) => (ds.id === dsId ? { ...ds, rows: makeEmptyRows(ds.rows.length) } : ds)));
  };

  const POWER_TABLE_FIELDS = ['power', 'min', 'max'];

  // Lets a copied block of spreadsheet cells (multiple rows and/or columns, tab/newline-separated)
  // be pasted starting at one cell and fill outward across rows/columns automatically.
  const handlePowerTablePaste = (e, dsId, rowIndex, field) => {
    const text = e.clipboardData ? e.clipboardData.getData('text') : '';
    if (!text.includes('\n') && !text.includes('\t')) return; // single value — let normal paste happen
    e.preventDefault();
    const lines = text.replace(/\r/g, '').split('\n');
    while (lines.length && lines[lines.length - 1] === '') lines.pop();
    const grid = lines.map((line) => line.split('\t'));
    const startFieldIdx = POWER_TABLE_FIELDS.indexOf(field);

    setPowerDatasets((prev) => prev.map((ds) => {
      if (ds.id !== dsId) return ds;
      const rows = ds.rows.map((r) => ({ ...r }));
      grid.forEach((lineCells, rOffset) => {
        const targetRow = rowIndex + rOffset;
        if (targetRow >= rows.length) return;
        lineCells.forEach((cellVal, cOffset) => {
          const fIdx = startFieldIdx + cOffset;
          if (fIdx >= POWER_TABLE_FIELDS.length) return;
          rows[targetRow][POWER_TABLE_FIELDS[fIdx]] = cellVal.trim();
        });
      });
      return { ...ds, rows };
    }));
  };

  const exportPowerDependenceCsv = () => {
    if (!powerDatasets.length) { addError('No datasets to export.'); return; }
    const rep = Number(laserRepRate), dia = Number(laserSpotDiameter);
    const toFluence = (mw) => {
      if (!Number.isFinite(mw) || !Number.isFinite(rep) || !Number.isFinite(dia) || rep <= 0 || dia <= 0) return NaN;
      return (400 * mw) / (Math.PI * rep * dia * dia);
    };
    const header = ['Power (mW)', 'Fluence (mJ/cm2)', ...powerDatasets.map((d) => `${d.name} (peak-to-peak)`)];
    const rows = [];
    for (let i = 0; i < numPowers; i++) {
      const refRow = powerDatasets[0] && powerDatasets[0].rows[i];
      const mw = refRow ? Number(refRow.power) : NaN;
      const fl = toFluence(mw);
      const row = [
        Number.isFinite(mw) ? roundSig(mw) : (refRow ? refRow.power : ''),
        Number.isFinite(fl) ? roundSig(fl) : '',
      ];
      powerDatasets.forEach((ds) => {
        const r = ds.rows[i];
        const min = r ? Number(r.min) : NaN;
        const max = r ? Number(r.max) : NaN;
        row.push(Number.isFinite(min) && Number.isFinite(max) ? roundSig(max - min) : '');
      });
      rows.push(row);
    }
    const csv = Papa.unparse([header, ...rows]);
    saveFile(new Blob([csv], { type: 'text/csv;charset=utf-8' }), 'thz_power_dependence.csv', 'CSV file', 'text/csv', ['.csv']);
  };

  // --- Tab 3: FFT convolution / deconvolution ---
  const [convCards, setConvCards] = useState([]); // [{ id, name, operation, sourceAId, sourceBId, shadeThresholdDB }]
  const convChartRefsRef = useRef({});
  const convRawFftCacheRef = useRef(new Map()); // dataset id -> { time, amplitude, settingsKey, freqs, re, im, mags }
  const getConvChartRef = (id) => {
    if (!convChartRefsRef.current[id]) convChartRefsRef.current[id] = { current: null };
    return convChartRefsRef.current[id];
  };

  // Per-card zoom/pan view state, kept separate from convCards itself so panning/zooming
  // never retriggers the (expensive) FFT + convolution recompute for any card.
  const [convXDomains, setConvXDomains] = useState({}); // { [cardId]: [min,max] | undefined }
  const [convYDomains, setConvYDomains] = useState({});
  const [convModes, setConvModes] = useState({}); // { [cardId]: 'zoom' | 'pan' }
  const [convSels, setConvSels] = useState({}); // { [cardId]: {x1,x2,y1,y2} }
  const convPanRefsRef = useRef({});
  const convYScaleRefsRef = useRef({});
  const emptyConvSel = { x1: null, x2: null, y1: null, y2: null };

  const getConvMode = (id) => convModes[id] || 'zoom';
  const getConvSel = (id) => convSels[id] || emptyConvSel;
  const getConvPanRef = (id) => {
    if (!convPanRefsRef.current[id]) convPanRefsRef.current[id] = { dragging: false, startX: 0, startDomain: null };
    return convPanRefsRef.current[id];
  };

  const resetConvView = (id) => {
    setConvXDomains((prev) => ({ ...prev, [id]: undefined }));
    setConvYDomains((prev) => ({ ...prev, [id]: undefined }));
    setConvSels((prev) => ({ ...prev, [id]: emptyConvSel }));
  };

  const handleConvMouseDown = (e, id, xDomainEff, yDomainEff) => {
    if (!e) return;
    if (getConvMode(id) === 'zoom') {
      convYScaleRefsRef.current[id] = makeYScale(getYPixelRange(getConvChartRef(id).current), yDomainEff);
      const yVal = convYScaleRefsRef.current[id] ? convYScaleRefsRef.current[id].pxToVal(e.chartY) : null;
      setConvSels((prev) => ({ ...prev, [id]: { x1: e.activeLabel, x2: e.activeLabel, y1: yVal, y2: yVal } }));
    } else {
      const panRef = getConvPanRef(id);
      panRef.dragging = true;
      panRef.startX = e.chartX;
      panRef.startDomain = xDomainEff;
    }
  };

  const handleConvMouseMove = (e, id) => {
    if (!e) return;
    const sel = getConvSel(id);
    if (getConvMode(id) === 'zoom' && sel.x1 != null) {
      const yScale = convYScaleRefsRef.current[id];
      const yVal = yScale ? yScale.pxToVal(e.chartY) : sel.y2;
      setConvSels((prev) => ({ ...prev, [id]: { ...sel, x2: e.activeLabel, y2: yVal } }));
    } else if (getConvMode(id) === 'pan' && getConvPanRef(id).dragging) {
      const wrapper = getConvChartRef(id).current;
      const plotWidth = wrapper ? Math.max(50, wrapper.clientWidth - 85) : 400;
      const panRef = getConvPanRef(id);
      const [d0, d1] = panRef.startDomain;
      const span = d1 - d0;
      const deltaPx = e.chartX - panRef.startX;
      const deltaData = -(deltaPx / plotWidth) * span;
      setConvXDomains((prev) => ({ ...prev, [id]: [d0 + deltaData, d1 + deltaData] }));
    }
  };

  const handleConvMouseUp = (id) => {
    if (getConvMode(id) === 'zoom') {
      const { x1, x2, y1, y2 } = getConvSel(id);
      if (Number.isFinite(x1) && Number.isFinite(x2) && x1 !== x2) setConvXDomains((prev) => ({ ...prev, [id]: [Math.min(x1, x2), Math.max(x1, x2)] }));
      if (Number.isFinite(y1) && Number.isFinite(y2) && y1 !== y2) setConvYDomains((prev) => ({ ...prev, [id]: [Math.min(y1, y2), Math.max(y1, y2)] }));
      setConvSels((prev) => ({ ...prev, [id]: emptyConvSel }));
    } else {
      getConvPanRef(id).dragging = false;
    }
  };

  const handleConvMouseLeave = (id) => {
    setConvSels((prev) => ({ ...prev, [id]: emptyConvSel }));
    getConvPanRef(id).dragging = false;
  };

  const addConvCard = () => {
    setConvCards((prev) => [...prev, {
      id: `conv_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      name: `Result ${prev.length + 1}`,
      operation: 'deconvolve',
      sourceAId: '',
      sourceBId: '',
      shadeThresholdDB: 20,
      color: COLORS[prev.length % COLORS.length],
    }]);
  };
  const removeConvCard = (id) => {
    delete convChartRefsRef.current[id];
    delete convPanRefsRef.current[id];
    delete convYScaleRefsRef.current[id];
    setConvXDomains((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setConvYDomains((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setConvModes((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setConvSels((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setConvCards((prev) => prev.filter((c) => c.id !== id));
  };
  const updateConvCard = (id, patch) => setConvCards((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));

  const sourceOptionsForCard = (cardIndex) => [
    ...datasets.map((d) => ({ id: d.id, name: d.name, group: 'Loaded data' })),
    ...convCards.slice(0, cardIndex).map((c) => ({ id: c.id, name: c.name, group: 'Previous results' })),
  ];

  const exportConvCsv = (card) => {
    if (!card || !card.freqs) { addError('Nothing to export for this card yet.'); return; }
    const rows = [['Frequency (THz)', 'Magnitude (dB)', 'Phase (rad)']];
    for (let i = 0; i < card.freqs.length; i++) {
      rows.push([roundSig(card.freqs[i]), roundSig(card.magsDB[i]), roundSig(card.phase[i])]);
    }
    const csv = Papa.unparse(rows);
    const safeName = (card.name || 'result').replace(/[^a-z0-9_-]+/gi, '_');
    saveFile(new Blob([csv], { type: 'text/csv;charset=utf-8' }), `${safeName}.csv`, 'CSV file', 'text/csv', ['.csv']);
  };

  const compareChartRef = useRef(null);
  const [compareSelectedIds, setCompareSelectedIds] = useState([]);
  const [compareMode, setCompareMode] = useState('zoom'); // 'zoom' | 'pan'
  const [compareXDomain, setCompareXDomain] = useState(null);
  const [compareYDomainState, setCompareYDomainState] = useState(null);
  const [compareSel, setCompareSel] = useState({ x1: null, x2: null, y1: null, y2: null });
  const comparePanRef = useRef({ dragging: false, startX: 0, startDomain: null });
  const compareYScaleRef = useRef(null);
  const toggleCompareSelection = (id) => setCompareSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

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
        const activeYDomain = validDomain(timeYDomain) || timeYFullDomain;
        timeYScaleRef.current = makeYScale(getYPixelRange(timeChartWrapRef.current), activeYDomain);
        const yVal = timeYScaleRef.current ? timeYScaleRef.current.pxToVal(e.chartY) : null;
        setTimeSel({ x1: e.activeLabel, x2: e.activeLabel, y1: yVal, y2: yVal });
      } else if (timeMode === 'snapshot') {
        takeSnapshot(e.activeLabel);
      } else {
        panRef.current = { dragging: true, startX: e.chartX, startDomain: validDomain(timeDomain) || timeFullDomain, chart: 'time' };
      }
    } else {
      if (freqMode === 'zoom') {
        const activeYDomain = validDomain(freqYDomain) || freqYFullDomain;
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
        if (Number.isFinite(x1) && Number.isFinite(x2) && x1 !== x2) setTimeDomain([Math.min(x1, x2), Math.max(x1, x2)]);
        if (Number.isFinite(y1) && Number.isFinite(y2) && y1 !== y2) setTimeYDomain([Math.min(y1, y2), Math.max(y1, y2)]);
        setTimeSel(emptySel);
      } else {
        panRef.current = { dragging: false, startX: 0, startDomain: null, chart: null };
      }
    } else {
      if (freqMode === 'zoom') {
        const { x1, x2, y1, y2 } = freqSel;
        if (Number.isFinite(x1) && Number.isFinite(x2) && x1 !== x2) setFreqDomain([Math.min(x1, x2), Math.max(x1, x2)]);
        if (Number.isFinite(y1) && Number.isFinite(y2) && y1 !== y2) setFreqYDomain([Math.min(y1, y2), Math.max(y1, y2)]);
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

  const convResults = useMemo(() => {
    const resolved = [];
    convCards.forEach((card) => {
      if (!card.sourceAId || !card.sourceBId) {
        resolved.push({ ...card, freqs: null, error: 'Select both Dataset A and Dataset B.' });
        return;
      }
      const sourceA = resolveConvSource(card.sourceAId, datasets, processingOpts, resolved, convRawFftCacheRef.current);
      const sourceB = resolveConvSource(card.sourceBId, datasets, processingOpts, resolved, convRawFftCacheRef.current);
      if (!sourceA || !sourceB) {
        resolved.push({ ...card, freqs: null, error: 'One or both source datasets are missing — reselect.' });
        return;
      }
      const freqHi = Math.min(sourceA.freqs[sourceA.freqs.length - 1], sourceB.freqs[sourceB.freqs.length - 1]);
      if (!(freqHi > 0)) {
        resolved.push({ ...card, freqs: null, error: 'No overlapping frequency range between the two sources.' });
        return;
      }

      const grid = buildLinGrid(0, freqHi, 2000);
      const aRe = interpolateSeries(sourceA.freqs, sourceA.re, grid).map((v) => (typeof v === 'number' ? v : 0));
      const aIm = interpolateSeries(sourceA.freqs, sourceA.im, grid).map((v) => (typeof v === 'number' ? v : 0));
      const bRe = interpolateSeries(sourceB.freqs, sourceB.re, grid).map((v) => (typeof v === 'number' ? v : 0));
      const bIm = interpolateSeries(sourceB.freqs, sourceB.im, grid).map((v) => (typeof v === 'number' ? v : 0));

      const resRe = new Array(grid.length);
      const resIm = new Array(grid.length);
      const resMag = new Array(grid.length);
      const rawPhase = new Array(grid.length);
      for (let i = 0; i < grid.length; i++) {
        const c = card.operation === 'convolve'
          ? complexMultiply(aRe[i], aIm[i], bRe[i], bIm[i])
          : complexDivide(aRe[i], aIm[i], bRe[i], bIm[i]);
        resRe[i] = c.re;
        resIm[i] = c.im;
        resMag[i] = Math.hypot(c.re, c.im);
        rawPhase[i] = Math.atan2(c.im, c.re);
      }
      const magsDB = toDB(resMag);
      const phase = unwrapPhase(rawPhase);

      const aMag = new Array(grid.length);
      const bMag = new Array(grid.length);
      for (let i = 0; i < grid.length; i++) {
        aMag[i] = Math.hypot(aRe[i], aIm[i]);
        bMag[i] = Math.hypot(bRe[i], bIm[i]);
      }
      const peakAMag = Math.max(...aMag) || 1;
      const peakBMag = Math.max(...bMag) || 1;
      // Bins where either operand is numerically negligible relative to its own peak (e.g. the
      // DC bin, which is always suppressed toward zero by mean-removal before the FFT) produce
      // meaningless floor-clamp artifacts regardless of operation — exclude these from y-scaling.
      const degenerateFlags = grid.map((_, i) => aMag[i] < peakAMag * 1e-6 || bMag[i] < peakBMag * 1e-6);

      let unreliableSpans = [];
      let excludeFlags = degenerateFlags;
      if (card.operation === 'deconvolve') {
        const bDB = toDB(bMag);
        const peakBDB = Math.max(...bDB);
        const threshold = peakBDB - (Number(card.shadeThresholdDB) || 20);
        const weakFlags = bDB.map((v) => v < threshold);
        unreliableSpans = findTrueSpans(grid, weakFlags);
        excludeFlags = degenerateFlags.map((f, i) => f || weakFlags[i]);
      }
      // The y-axis should scale to the trustworthy part of the spectrum only — degenerate/weak
      // bins can dominate a simple min/max or percentile trim if a large fraction of the
      // spectrum is affected (not just a rare few outlier bins).
      const reliableOnly = magsDB.filter((_, i) => !excludeFlags[i]);
      const scaleMagsDB = reliableOnly.length > 5 ? reliableOnly : magsDB;

      const stride = Math.max(1, Math.floor(grid.length / 1500));
      const chartData = [];
      for (let i = 0; i < grid.length; i += stride) chartData.push({ x: grid[i], y: magsDB[i] });

      resolved.push({
        ...card,
        freqs: grid, re: resRe, im: resIm, mags: resMag, magsDB, scaleMagsDB, phase,
        chartData, unreliableSpans,
        sourceAName: sourceA.name, sourceBName: sourceB.name,
        error: null,
      });
    });
    return resolved;
  }, [convCards, datasets, processingOpts]);

  const convColorFor = (id) => {
    const idx = convResults.findIndex((r) => r.id === id);
    const card = idx !== -1 ? convResults[idx] : null;
    if (card && /^#[0-9a-fA-F]{6}$/.test(card.color)) return card.color;
    return COLORS[Math.max(0, idx) % COLORS.length];
  };
  const compareEntries = convResults.filter((r) => compareSelectedIds.includes(r.id) && !r.error);

  const exportCompareCsv = () => {
    if (!compareEntries.length) { addError('Select at least one result to export.'); return; }
    const maxFreq = Math.max(...compareEntries.map((e) => e.freqs[e.freqs.length - 1]));
    const grid = buildLinGrid(0, maxFreq, 2000);
    const header = ['Frequency (THz)', ...compareEntries.map((e) => `${e.name} (dB)`)];
    const columns = compareEntries.map((e) => interpolateSeries(e.freqs, e.magsDB, grid));
    const rows = [header];
    for (let i = 0; i < grid.length; i++) {
      const row = [roundSig(grid[i])];
      columns.forEach((col) => row.push(typeof col[i] === 'number' ? roundSig(col[i]) : ''));
      rows.push(row);
    }
    const csv = Papa.unparse(rows);
    saveFile(new Blob([csv], { type: 'text/csv;charset=utf-8' }), 'thz_conv_compare.csv', 'CSV file', 'text/csv', ['.csv']);
  };

  const compareXTop = compareEntries.length ? Math.max(...compareEntries.map((e) => e.freqs[e.freqs.length - 1])) : 6;
  const autoCompareYDomain = compareEntries.length ? computeDbYDomain(compareEntries.flatMap((e) => e.scaleMagsDB || e.magsDB)) : [-40, 0];
  const compareXDomainEffective = validDomain(compareXDomain) || [0, compareXTop];
  const compareYDomainEffective = validDomain(compareYDomainState) || autoCompareYDomain;

  const resetCompareView = () => {
    setCompareXDomain(null);
    setCompareYDomainState(null);
    setCompareSel({ x1: null, x2: null, y1: null, y2: null });
  };

  const handleCompareMouseDown = (e) => {
    if (!e) return;
    if (compareMode === 'zoom') {
      compareYScaleRef.current = makeYScale(getYPixelRange(compareChartRef.current), compareYDomainEffective);
      const yVal = compareYScaleRef.current ? compareYScaleRef.current.pxToVal(e.chartY) : null;
      setCompareSel({ x1: e.activeLabel, x2: e.activeLabel, y1: yVal, y2: yVal });
    } else {
      comparePanRef.current = { dragging: true, startX: e.chartX, startDomain: compareXDomainEffective };
    }
  };

  const handleCompareMouseMove = (e) => {
    if (!e) return;
    if (compareMode === 'zoom' && compareSel.x1 != null) {
      const yScale = compareYScaleRef.current;
      const yVal = yScale ? yScale.pxToVal(e.chartY) : compareSel.y2;
      setCompareSel((sel) => ({ ...sel, x2: e.activeLabel, y2: yVal }));
    } else if (compareMode === 'pan' && comparePanRef.current.dragging) {
      const wrapper = compareChartRef.current;
      const plotWidth = wrapper ? Math.max(50, wrapper.clientWidth - 85) : 400;
      const [d0, d1] = comparePanRef.current.startDomain;
      const span = d1 - d0;
      const deltaPx = e.chartX - comparePanRef.current.startX;
      const deltaData = -(deltaPx / plotWidth) * span;
      setCompareXDomain([d0 + deltaData, d1 + deltaData]);
    }
  };

  const handleCompareMouseUp = () => {
    if (compareMode === 'zoom') {
      const { x1, x2, y1, y2 } = compareSel;
      if (Number.isFinite(x1) && Number.isFinite(x2) && x1 !== x2) setCompareXDomain([Math.min(x1, x2), Math.max(x1, x2)]);
      if (Number.isFinite(y1) && Number.isFinite(y2) && y1 !== y2) setCompareYDomainState([Math.min(y1, y2), Math.max(y1, y2)]);
      setCompareSel({ x1: null, x2: null, y1: null, y2: null });
    } else {
      comparePanRef.current = { dragging: false, startX: 0, startDomain: null };
    }
  };

  const handleCompareMouseLeave = () => {
    setCompareSel({ x1: null, x2: null, y1: null, y2: null });
    comparePanRef.current.dragging = false;
  };

  // --- Tab 4: time-domain waveform arithmetic ---
  // Cards reuse Tab 3's id-keyed zoom/pan machinery (convXDomains, handleConvMouse*, etc.),
  // which works for any chart id; Tab 4 ids are prefixed 'arith_' so they never collide.
  const [arithCards, setArithCards] = useState([]); // [{ id, name, operation, sourceAId, sourceBId, color, showInputs }]
  const [arithCompareIds, setArithCompareIds] = useState([]);
  const [arithSentIds, setArithSentIds] = useState([]); // brief "Sent" confirmation on the Send-to-Tab-1 button
  const ARITH_COMPARE_ID = 'arith__compare';

  const addArithCard = () => {
    setArithCards((prev) => [...prev, {
      id: `arith_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      name: `Waveform ${prev.length + 1}`,
      operation: 'subtract',
      sourceAId: '',
      sourceBId: '',
      color: COLORS[prev.length % COLORS.length],
      showInputs: true,
    }]);
  };
  const removeArithCard = (id) => {
    delete convChartRefsRef.current[id];
    delete convPanRefsRef.current[id];
    delete convYScaleRefsRef.current[id];
    setConvXDomains((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setConvYDomains((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setConvModes((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setConvSels((prev) => { const next = { ...prev }; delete next[id]; return next; });
    setArithCompareIds((prev) => prev.filter((x) => x !== id));
    setArithCards((prev) => prev.filter((c) => c.id !== id));
  };
  const updateArithCard = (id, patch) => setArithCards((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));

  const arithSourceOptions = (cardIndex) => [
    ...datasets.map((d) => ({ id: d.id, name: d.name, group: 'Loaded data' })),
    ...arithCards.slice(0, cardIndex).map((c) => ({ id: c.id, name: c.name, group: 'Previous results' })),
  ];

  const arithResults = useMemo(() => {
    const resolved = [];
    const resolveSource = (id) => {
      if (!id) return null;
      const ds = datasets.find((d) => d.id === id);
      if (ds) return { time: ds.time, amplitude: ds.amplitude, name: ds.name };
      const prev = resolved.find((r) => r.id === id);
      if (prev && prev.time) return { time: prev.time, amplitude: prev.amplitude, name: prev.name };
      return null;
    };
    arithCards.forEach((card) => {
      const needsB = arithNeedsB(card.operation);
      if (!card.sourceAId || (needsB && !card.sourceBId)) {
        resolved.push({ ...card, time: null, error: needsB ? 'Select both Dataset A and Dataset B.' : 'Select Dataset A.' });
        return;
      }
      const a = resolveSource(card.sourceAId);
      const b = needsB ? resolveSource(card.sourceBId) : null;
      if (!a || (needsB && !b)) {
        resolved.push({ ...card, time: null, error: 'A source dataset is missing or has an error — reselect.' });
        return;
      }
      const out = combineWaveforms(card.operation, a, b);
      if (out.error) { resolved.push({ ...card, time: null, error: out.error }); return; }

      const aAsc = ascendingTrace(a.time, a.amplitude);
      const inputSeries = [{ key: 'A', name: `A: ${a.name}`, xs: aAsc.time, ys: aAsc.amplitude }];
      if (needsB) inputSeries.push({ key: 'B', name: `B: ${b.name}`, xs: out.time, ys: out.bOnGrid });
      resolved.push({
        ...card,
        time: out.time, amplitude: out.amplitude, notes: out.notes,
        chartData: downsampleXY(out.time, out.amplitude),
        inputs: inputSeries.map((sr) => ({ ...sr, chartData: downsampleXY(sr.xs, sr.ys) })),
        sourceAName: a.name, sourceBName: b ? b.name : null,
        error: null,
      });
    });
    return resolved;
  }, [arithCards, datasets]);

  const arithColorFor = (id) => {
    const idx = arithResults.findIndex((r) => r.id === id);
    const card = idx !== -1 ? arithResults[idx] : null;
    if (card && /^#[0-9a-fA-F]{6}$/.test(card.color)) return card.color;
    return COLORS[Math.max(0, idx) % COLORS.length];
  };
  const arithFormula = (r) => {
    if (r.operation === 'none') return r.sourceAName;
    if (r.operation === 'flip') return `−(${r.sourceAName})`;
    return `${r.sourceAName} ${r.operation === 'add' ? '+' : '−'} ${r.sourceBName}`;
  };

  const exportArithCsv = (r) => {
    if (!r || !r.time) { addError('Nothing to export for this card yet.'); return; }
    const header = [`Time (${timeUnit})`, `${r.name}`];
    const extra = [];
    if (r.operation !== 'none') {
      header.push(`A: ${r.sourceAName}`);
      extra.push(interpolateSeries(r.inputs[0].xs, r.inputs[0].ys, r.time));
    }
    if (arithNeedsB(r.operation)) { header.push(`B: ${r.sourceBName} (on result time grid)`); extra.push(r.inputs[1].ys); }
    const rows = [header];
    for (let i = 0; i < r.time.length; i++) {
      const row = [roundSig(r.time[i]), roundSig(r.amplitude[i])];
      extra.forEach((col) => row.push(typeof col[i] === 'number' ? roundSig(col[i]) : ''));
      rows.push(row);
    }
    const safeName = (r.name || 'waveform').replace(/[^a-z0-9_-]+/gi, '_');
    saveFile(new Blob([Papa.unparse(rows)], { type: 'text/csv;charset=utf-8' }), `${safeName}.csv`, 'CSV file', 'text/csv', ['.csv']);
  };

  // Copies a result into Tab 1 as a regular dataset, so it gets an FFT, metrics, snapshots, etc.
  // It is a snapshot copy: later edits to the card do not update the sent dataset.
  const sendArithToTab1 = (r) => {
    if (!r || !r.time) return;
    const color = arithColorFor(r.id);
    setDatasets((prev) => {
      const base = r.name || 'Waveform';
      let name = base;
      for (let n = 2; prev.some((d) => d.name === name); n++) name = `${base} (${n})`;
      return [...prev, {
        id: `arithds_${Date.now()}_${Math.random().toString(36).slice(2)}`,
        name, color, visible: true, width: 1.4,
        time: r.time.slice(), amplitude: r.amplitude.slice(),
      }];
    });
    setArithSentIds((prev) => [...prev, r.id]);
    setTimeout(() => setArithSentIds((prev) => prev.filter((x) => x !== r.id)), 1800);
  };

  const arithCompareEntries = arithResults.filter((r) => arithCompareIds.includes(r.id) && !r.error);
  const toggleArithCompare = (id) => setArithCompareIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const exportArithCompareCsv = () => {
    if (!arithCompareEntries.length) { addError('Select at least one result to export.'); return; }
    const lo = Math.min(...arithCompareEntries.map((e) => e.time[0]));
    const hi = Math.max(...arithCompareEntries.map((e) => e.time[e.time.length - 1]));
    const count = Math.min(20000, Math.max(...arithCompareEntries.map((e) => e.time.length)));
    exportCsvGrid(
      [`Time (${timeUnit})`, ...arithCompareEntries.map((e) => e.name)],
      buildLinGrid(lo, hi, count),
      arithCompareEntries.map((e) => ({ xs: e.time, ys: e.amplitude })),
      'thz_waveform_compare.csv',
    );
  };

  const addError = (msg) => setErrors((prev) => [...prev, msg]);
  const dismissError = (i) => setErrors((prev) => prev.filter((_, idx) => idx !== i));

  const handleFiles = useCallback((fileList) => {
    Array.from(fileList).forEach((file) => {
      if (/\.json$/i.test(file.name)) {
        addError(`"${file.name}" looks like a session file — use the "Load" button under Session instead of Upload.`);
        return;
      }
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
    const [fLo, fHi] = validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN;
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
    const name = sessionName.trim() || 'thz_session';
    const session = {
      version: 2,
      name,
      datasets: datasets.map((d) => ({
        name: d.name, color: d.color, visible: d.visible, width: d.width, time: d.time, amplitude: d.amplitude,
      })),
      settings: {
        windowType, zeroPadFactor, timeUnit, noiseRegion, noiseFraction, bandwidthMode, marginDB, displayMode, showWaterVapor,
        timeDomain, timeYDomain, freqDomain, freqYDomain,
      },
      snapshots,
      powerDependence: {
        numPowers, numPowerDatasets,
        datasets: powerDatasets.map((ds) => ({ name: ds.name, color: ds.color, marker: ds.marker, rows: ds.rows })),
        powerXUnit, showPowerFit, laserRepRate, laserPulseDuration, laserSpotDiameter,
      },
      convolution: convCards.map((c) => {
        const resolveRef = (id) => {
          const dsIdx = datasets.findIndex((d) => d.id === id);
          if (dsIdx !== -1) return { kind: 'dataset', index: dsIdx };
          const cardIdx = convCards.findIndex((cc) => cc.id === id);
          if (cardIdx !== -1) return { kind: 'card', index: cardIdx };
          return null;
        };
        return {
          name: c.name, operation: c.operation, shadeThresholdDB: c.shadeThresholdDB, color: c.color,
          sourceA: resolveRef(c.sourceAId), sourceB: resolveRef(c.sourceBId),
        };
      }),
      arithmetic: arithCards.map((c) => {
        const resolveRef = (id) => {
          const dsIdx = datasets.findIndex((d) => d.id === id);
          if (dsIdx !== -1) return { kind: 'dataset', index: dsIdx };
          const cardIdx = arithCards.findIndex((cc) => cc.id === id);
          if (cardIdx !== -1) return { kind: 'card', index: cardIdx };
          return null;
        };
        return {
          name: c.name, operation: c.operation, color: c.color, showInputs: c.showInputs !== false,
          sourceA: resolveRef(c.sourceAId), sourceB: resolveRef(c.sourceBId),
        };
      }),
    };
    const json = JSON.stringify(session);
    saveFile(new Blob([json], { type: 'application/json' }), `${name}.json`, 'JSON session file', 'application/json', ['.json']);
    setSessionName(name);
    setSessionStatus('saved');
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

        if (Array.isArray(session.snapshots)) setSnapshots(session.snapshots);

        const pd = session.powerDependence;
        if (pd && Array.isArray(pd.datasets) && pd.datasets.length) {
          const rowCount = pd.numPowers || (pd.datasets[0].rows ? pd.datasets[0].rows.length : 5);
          setNumPowers(rowCount);
          setNumPowerDatasets(pd.numPowerDatasets || pd.datasets.length);
          setPowerDatasets(pd.datasets.map((ds, i) => ({
            id: `pd_${Date.now()}_${i}_${Math.random().toString(36).slice(2)}`,
            name: ds.name || `Dataset ${i + 1}`,
            color: ds.color || COLORS[i % COLORS.length],
            marker: MARKER_TYPES.includes(ds.marker) ? ds.marker : MARKER_TYPES[i % MARKER_TYPES.length],
            rows: Array.isArray(ds.rows) ? ds.rows : makeEmptyRows(rowCount),
          })));
          if (pd.powerXUnit === 'fluence' || pd.powerXUnit === 'mW') setPowerXUnit(pd.powerXUnit);
          setShowPowerFit(pd.showPowerFit !== false); // older sessions lack the field → default to shown
          if (typeof pd.laserRepRate === 'number') setLaserRepRate(pd.laserRepRate);
          if (typeof pd.laserPulseDuration === 'number') setLaserPulseDuration(pd.laserPulseDuration);
          if (typeof pd.laserSpotDiameter === 'number') setLaserSpotDiameter(pd.laserSpotDiameter);
        }

        const conv = session.convolution;
        if (Array.isArray(conv)) {
          const newCardIds = conv.map((_, i) => `conv_${Date.now()}_${i}_${Math.random().toString(36).slice(2)}`);
          const resolveRef = (ref) => {
            if (!ref) return '';
            if (ref.kind === 'dataset') return (restored[ref.index] && restored[ref.index].id) || '';
            if (ref.kind === 'card') return newCardIds[ref.index] || '';
            return '';
          };
          setConvCards(conv.map((c, i) => ({
            id: newCardIds[i],
            name: c.name || `Result ${i + 1}`,
            operation: c.operation === 'convolve' ? 'convolve' : 'deconvolve',
            sourceAId: resolveRef(c.sourceA),
            sourceBId: resolveRef(c.sourceB),
            shadeThresholdDB: typeof c.shadeThresholdDB === 'number' ? c.shadeThresholdDB : 20,
            color: /^#[0-9a-fA-F]{6}$/.test(c.color) ? c.color : COLORS[i % COLORS.length],
          })));
        }

        // Tab 4 cards (absent in older sessions → start empty).
        const arith = Array.isArray(session.arithmetic) ? session.arithmetic : [];
        const newArithIds = arith.map((_, i) => `arith_${Date.now()}_${i}_${Math.random().toString(36).slice(2)}`);
        const resolveArithRef = (ref) => {
          if (!ref) return '';
          if (ref.kind === 'dataset') return (restored[ref.index] && restored[ref.index].id) || '';
          if (ref.kind === 'card') return newArithIds[ref.index] || '';
          return '';
        };
        setArithCompareIds([]);
        setArithCards(arith.map((c, i) => ({
          id: newArithIds[i],
          name: c.name || `Waveform ${i + 1}`,
          operation: ARITH_OPS.some((o) => o.value === c.operation) ? c.operation : 'none',
          sourceAId: resolveArithRef(c.sourceA),
          sourceBId: resolveArithRef(c.sourceB),
          color: /^#[0-9a-fA-F]{6}$/.test(c.color) ? c.color : COLORS[i % COLORS.length],
          showInputs: c.showInputs !== false,
        })));

        const derivedName = (typeof session.name === 'string' && session.name.trim())
          || file.name.replace(/\.json$/i, '');
        setSessionName(derivedName);
        setSessionStatus('loaded');
      } catch (err) {
        addError(`Couldn't load "${file.name}" — not a valid session file.`);
      }
    };
    reader.onerror = () => addError(`Failed to read "${file.name}".`);
    reader.readAsText(file);
  };

  const processed = useMemo(() => {
    const freqLoHi = validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN;
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
      const peakFreq = refinePeakFreq(freqs, magsDB, peakIndex);

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

  const powerAxisTop = useMemo(() => {
    let maxX = 0;
    powerDatasets.forEach((ds) => ds.rows.forEach((r) => {
      const x = convertPowerX(Number(r.power));
      if (Number.isFinite(x) && x > maxX) maxX = x;
    }));
    if (!(maxX > 0)) return powerXUnit === 'fluence' ? 10 : 80;
    return maxX * 1.15;
  }, [powerDatasets, powerXUnit, laserRepRate, laserSpotDiameter]);

  const powerPlotData = useMemo(() => {
    return powerDatasets.map((ds) => {
      const points = ds.rows
        .map((r) => ({ power: Number(r.power), min: Number(r.min), max: Number(r.max) }))
        .filter((r) => Number.isFinite(r.power) && Number.isFinite(r.min) && Number.isFinite(r.max))
        .map((r) => ({ x: convertPowerX(r.power), y: r.max - r.min }))
        .filter((p) => Number.isFinite(p.x));
      const fit = fitSaturationCurve(points.map((p) => p.x), points.map((p) => p.y));
      let fitLine = [];
      if (fit) {
        const steps = 60;
        fitLine = Array.from({ length: steps + 1 }, (_, i) => {
          const p = (powerAxisTop * i) / steps;
          return { x: p, y: (fit.Amax * p) / (p + fit.Psat) };
        });
      }
      return { id: ds.id, name: ds.name, color: ds.color, marker: ds.marker, points, fit, fitLine };
    });
  }, [powerDatasets, powerAxisTop, powerXUnit, laserRepRate, laserSpotDiameter]);

  const powerXTicks = useMemo(() => niceTicks(0, powerAxisTop), [powerAxisTop]);

  const buildPowerLegendItems = () => (powerPlotData || []).map((pd) => {
    const symbol = powerXUnit === 'fluence' ? 'F' : 'P';
    const unit = powerXUnit === 'fluence' ? 'mJ/cm2' : 'mW';
    return {
      color: pd.color,
      parts: !showPowerFit
        ? [{ text: pd.name }]
        : pd.fit
        ? [{ text: `${pd.name} (${symbol}` }, { text: 'sat', dy: 3, fontSize: 8 }, { text: ` = ${roundDisp(pd.fit.Psat)} ${unit})`, dy: -3 }]
        : [{ text: `${pd.name} (fit unavailable)` }],
    };
  });

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
    const [lo, hi] = validDomain(timeDomain) || timeFullDomain;
    return niceTicks(lo, hi);
  }, [timeDomain, timeFullDomain]);
  const timeYTicks = useMemo(() => {
    const [lo, hi] = validDomain(timeYDomain) || timeYFullDomain;
    return niceTicks(lo, hi);
  }, [timeYDomain, timeYFullDomain]);
  const freqXTicks = useMemo(() => {
    const [lo, hi] = validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN;
    return niceTicks(lo, hi);
  }, [freqDomain]);
  const freqYTicks = useMemo(() => {
    const [lo, hi] = validDomain(freqYDomain) || freqYFullDomain;
    return niceTicks(lo, hi);
  }, [freqYDomain, freqYFullDomain]);

  return (
    <div className="min-h-full w-full bg-[#f7f6f3] text-slate-800" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>
      <div className="border-b border-slate-400 px-6 py-4">
        <div className="flex items-center gap-4">
          <TeamLogo />
          <div>
            <p className="text-2xl font-bold tracking-tight text-slate-700 mb-1">NIP - THz Team</p>
            <div className="flex items-baseline gap-3 flex-wrap">
              <h1 className="text-lg font-semibold tracking-tight text-slate-900">THz Waveform &amp; Spectrum Analyzer</h1>
              <span className="text-xs text-slate-600 font-mono">time-domain · FFT · bandwidth · SNR</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex gap-1 px-6 pt-3 border-b border-slate-400 bg-[#efede8]">
        <button
          onClick={() => setActiveTab('tds-fft')}
          className={`text-sm px-4 py-2 rounded-t border border-b-0 -mb-px transition ${activeTab === 'tds-fft' ? 'bg-white border-slate-400 text-slate-900 font-medium' : 'border-transparent text-slate-600 hover:text-slate-900'}`}
        >
          TDS &amp; FFT
        </button>
        <button
          onClick={() => setActiveTab('power-dep')}
          className={`text-sm px-4 py-2 rounded-t border border-b-0 -mb-px transition ${activeTab === 'power-dep' ? 'bg-white border-slate-400 text-slate-900 font-medium' : 'border-transparent text-slate-600 hover:text-slate-900'}`}
        >
          Power Fit
        </button>
        <button
          onClick={() => setActiveTab('convolution')}
          className={`text-sm px-4 py-2 rounded-t border border-b-0 -mb-px transition ${activeTab === 'convolution' ? 'bg-white border-slate-400 text-slate-900 font-medium' : 'border-transparent text-slate-600 hover:text-slate-900'}`}
        >
          Conv / Deconv
        </button>
        <button
          onClick={() => setActiveTab('arithmetic')}
          className={`text-sm px-4 py-2 rounded-t border border-b-0 -mb-px transition ${activeTab === 'arithmetic' ? 'bg-white border-slate-400 text-slate-900 font-medium' : 'border-transparent text-slate-600 hover:text-slate-900'}`}
        >
          Waveform Math
        </button>
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

      {activeTab === 'tds-fft' && (
      <div className="flex flex-row gap-4 p-6 items-start">
        {/* Left: controls + metrics */}
        <div className="w-80 flex-shrink-0 space-y-4">
          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3">
            <div className="flex gap-2">
              <button
                onClick={() => fileInputRef.current.click()}
                className="flex-1 flex items-center justify-center gap-2 rounded bg-slate-100 border border-slate-500 text-slate-700 text-sm py-2 hover:bg-slate-200 transition"
              >
                <Upload size={14} /> Upload
              </button>
            </div>
            <input
              ref={fileInputRef} type="file" multiple accept=".csv,.txt" className="hidden"
              onChange={(e) => { handleFiles(e.target.files); e.target.value = null; }}
            />

            <div className="mt-3 space-y-1.5 max-h-96 overflow-y-auto pr-1">
              {datasets.length === 0 && (
                <p className="text-xs text-slate-600 py-2">No datasets loaded. Upload a two-column time/amplitude .csv, .txt or .dat file to get started.</p>
              )}
              {datasets.map((d, index) => (
                <div
                  key={d.id}
                  onDragOver={(e) => handleDragOverRow(e, index)}
                  onDrop={(e) => handleDropRow(e, index)}
                  className={`rounded bg-white border px-2 py-1.5 space-y-1 transition ${dragOverIndex === index && dragIndex !== null && dragIndex !== index ? 'border-slate-600 border-2' : 'border-slate-400'} ${dragIndex === index ? 'opacity-40' : ''}`}
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
                        className="w-full bg-white border border-slate-300 hover:border-slate-500 focus:border-slate-600 focus:outline-none focus:ring-1 focus:ring-slate-500/40 rounded pl-1.5 pr-5 py-0.5 text-xs text-slate-800"
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

          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-3">
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

          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-3">
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
              <input type="range" min={5} max={45} value={noiseFraction * 100} onChange={(e) => setNoiseFraction(Number(e.target.value) / 100)} className="w-full accent-slate-700" />
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-3">
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

          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-3">
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
                <input type="checkbox" checked={showWaterVapor} onChange={(e) => setShowWaterVapor(e.target.checked)} className="accent-slate-700" />
                <span className="text-slate-900">Show water-vapor absorption lines</span>
              </label>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-2">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Export data</p>
            <button
              onClick={exportTdsCsv}
              className="w-full flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-slate-500 hover:bg-slate-100 transition"
            >
              <Download size={12} /> TDS data (.csv)
            </button>
            <button
              onClick={exportFftCsv}
              className="w-full flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-slate-500 hover:bg-slate-100 transition"
            >
              <Download size={12} /> FFT data (.csv)
            </button>
          </div>

          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-2">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Session</p>
            <label className="block space-y-1">
              <span className="text-xs text-slate-900">Session name</span>
              <input
                value={sessionName}
                onChange={(e) => { setSessionName(e.target.value); setSessionStatus(null); }}
                placeholder="thz_session"
                className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-xs text-slate-800"
              />
            </label>
            <div className="flex gap-2">
              <button
                onClick={saveSession}
                className="flex-1 flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-slate-500 hover:bg-slate-100 transition"
              >
                <Download size={12} /> Save
              </button>
              <button
                onClick={() => sessionInputRef.current.click()}
                className="flex-1 flex items-center justify-center gap-2 rounded bg-white border border-slate-400 text-slate-800 text-xs py-1.5 hover:border-slate-500 hover:bg-slate-100 transition"
              >
                <Upload size={12} /> Load
              </button>
            </div>
            {sessionStatus && sessionName && (
              <p className="text-xs text-slate-900">
                {sessionStatus === 'saved' ? 'Saved as ' : 'Loaded '}<span className="font-semibold">{sessionName}</span>
              </p>
            )}
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
              <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Time domain (TDS)</p>
              <div className="flex gap-1.5">
                <button
                  onClick={() => setTimeMode('zoom')}
                  title="Drag to zoom into a region"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${timeMode === 'zoom' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                >
                  <ZoomIn size={12} /> Zoom
                </button>
                <button
                  onClick={() => setTimeMode('pan')}
                  title="Drag to shift the view"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${timeMode === 'pan' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                >
                  <Move size={12} /> Pan
                </button>
                <button
                  onClick={() => setTimeMode('snapshot')}
                  title="Click points on the plot to record y-values below"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${timeMode === 'snapshot' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                >
                  <Camera size={12} /> Snapshot
                </button>
                <button
                  onClick={resetTimeView}
                  title="Reset to full view"
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <RotateCcw size={12} /> Reset
                </button>
                <span className="w-px bg-slate-300 mx-0.5" />
                <button
                  onClick={() => openExportDialog(timeChartWrapRef, 'thz_time_domain', legendItems, 'png')}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <Download size={12} /> PNG
                </button>
                <button
                  onClick={() => openExportDialog(timeChartWrapRef, 'thz_time_domain', legendItems, 'svg')}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <Download size={12} /> SVG
                </button>
              </div>
            </div>
            <div className="h-96 select-none" ref={timeChartWrapRef} onMouseDown={(e) => e.preventDefault()} style={{ cursor: timeMode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  margin={{ top: 24, right: 15, bottom: 40, left: 0 }}
                  onMouseDown={(e) => handleMouseDown(e, 'time')}
                  onMouseMove={(e) => handleMouseMove(e, 'time')}
                  onMouseUp={() => handleMouseUp('time')}
                  onMouseLeave={() => handleMouseLeave('time')}
                  onDoubleClick={resetTimeView}
                >
                  <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                  <XAxis dataKey="x" type="number" domain={validDomain(timeDomain) || timeFullDomain} ticks={timeXTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }}
                    label={{ value: `Time (${timeUnit})`, position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                  <YAxis domain={validDomain(timeYDomain) || timeYFullDomain} ticks={timeYTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }} width={72}
                    tickFormatter={(v) => (v === 0 ? '0.00e+0' : v.toExponential(2))}
                    label={{ value: 'E-field (a.u.)', angle: -90, position: 'insideLeft', fill: '#334155', fontSize: 11 }} />
                  <Tooltip cursor={false} contentStyle={{ background: 'rgba(255, 255, 255, 0.80)', border: '1px solid rgba(148, 163, 184, 0.85)', fontSize: 12, backdropFilter: 'blur(1.5px)' }} labelStyle={{ color: '#1e293b' }} formatter={(v) => fmtTip(v, 'sci')} labelFormatter={(l) => fmtTipLabel(l, timeUnit)} />
                  <Legend verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: 11, paddingTop: 20 }} />
                  <Customized component={ChartBorder} />
                  {visible.map((d) => (
                    <Line key={d.id} data={d.timeChartData} dataKey="y" name={d.name} stroke={d.color} dot={false} isAnimationActive={false} strokeWidth={d.width || 1.4} />
                  ))}
                  {timeMode === 'zoom' && timeSel.x1 != null && timeSel.x2 != null && (
                    <ReferenceArea x1={timeSel.x1} x2={timeSel.x2} y1={timeSel.y1} y2={timeSel.y2} strokeOpacity={0.4} stroke="#334155" fill="#334155" fillOpacity={0.15} />
                  )}
                  {snapshots.map((s, i) => (
                    <ReferenceLine
                      key={s.id} x={s.time} stroke="#334155" strokeDasharray="2 2" strokeWidth={1} ifOverflow="extendDomain"
                      label={{ value: indexToLetters(i), position: 'top', fill: '#334155', fontSize: 12, fontWeight: 700 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
              <label className="space-y-1">
                <span className="text-slate-900 block">X min ({timeUnit})</span>
                <NumberRangeField
                  value={roundDisp((validDomain(timeDomain) || timeFullDomain)[0])}
                  onCommit={(v) => setTimeDomain([v, (validDomain(timeDomain) || timeFullDomain)[1]])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">X max ({timeUnit})</span>
                <NumberRangeField
                  value={roundDisp((validDomain(timeDomain) || timeFullDomain)[1])}
                  onCommit={(v) => setTimeDomain([(validDomain(timeDomain) || timeFullDomain)[0], v])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">Y min (a.u.)</span>
                <NumberRangeField
                  value={roundDisp((validDomain(timeYDomain) || timeYFullDomain)[0])}
                  onCommit={(v) => setTimeYDomain([v, (validDomain(timeYDomain) || timeYFullDomain)[1]])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">Y max (a.u.)</span>
                <NumberRangeField
                  value={roundDisp((validDomain(timeYDomain) || timeYFullDomain)[1])}
                  onCommit={(v) => setTimeYDomain([(validDomain(timeYDomain) || timeYFullDomain)[0], v])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
            </div>
            <button onClick={resetTimeView} className="mt-2 text-xs text-slate-600 hover:text-slate-900 underline underline-offset-2">Reset to auto</button>
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
                      <th className="text-left font-normal py-2 pr-4">Label</th>
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
                    {snapshots.map((s, i) => (
                      <tr key={s.id} className="border-b border-slate-300">
                        <td className="text-left pr-4 font-mono font-semibold text-slate-900">{indexToLetters(i)}</td>
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
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${freqMode === 'zoom' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                >
                  <ZoomIn size={12} /> Zoom
                </button>
                <button
                  onClick={() => setFreqMode('pan')}
                  title="Drag to shift the view"
                  className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${freqMode === 'pan' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                >
                  <Move size={12} /> Pan
                </button>
                <button
                  onClick={resetFreqView}
                  title="Reset to full view"
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <RotateCcw size={12} /> Reset
                </button>
                <span className="w-px bg-slate-300 mx-0.5" />
                <button
                  onClick={() => openExportDialog(freqChartWrapRef, 'thz_frequency_domain', legendItems, 'png')}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <Download size={12} /> PNG
                </button>
                <button
                  onClick={() => openExportDialog(freqChartWrapRef, 'thz_frequency_domain', legendItems, 'svg')}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <Download size={12} /> SVG
                </button>
              </div>
            </div>
            <div className="h-96 select-none" ref={freqChartWrapRef} onMouseDown={(e) => e.preventDefault()} style={{ cursor: freqMode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  margin={{ top: 24, right: 15, bottom: 40, left: 0 }}
                  onMouseDown={(e) => handleMouseDown(e, 'freq')}
                  onMouseMove={(e) => handleMouseMove(e, 'freq')}
                  onMouseUp={() => handleMouseUp('freq')}
                  onMouseLeave={() => handleMouseLeave('freq')}
                  onDoubleClick={resetFreqView}
                >
                  <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                  <XAxis dataKey="x" type="number" domain={validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN} ticks={freqXTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }}
                    label={{ value: 'Frequency (THz)', position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                  <YAxis domain={validDomain(freqYDomain) || freqYFullDomain} ticks={freqYTicks} allowDataOverflow stroke="#334155" tick={{ fontSize: 11 }} width={56}
                    tickFormatter={(v) => v.toFixed(2)}
                    label={{ value: displayMode === 'normalized' ? 'dB (rel. peak)' : 'dB (a.u.)', angle: -90, position: 'insideLeft', fill: '#334155', fontSize: 11 }} />
                  <Tooltip cursor={false} contentStyle={{ background: 'rgba(255, 255, 255, 0.80)', border: '1px solid rgba(148, 163, 184, 0.85)', fontSize: 12, backdropFilter: 'blur(1.5px)' }} labelStyle={{ color: '#1e293b' }} formatter={(v) => fmtTip(v, 'db')} labelFormatter={(l) => fmtTipLabel(l, 'THz')} />
                  <Legend verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: 11, paddingTop: 20 }} />
                  <Customized component={ChartBorder} />
                  {visible.map((d) => (
                    <Line key={d.id} data={d.freqChartData} dataKey="y" name={d.name} stroke={d.color} dot={false} isAnimationActive={false} strokeWidth={d.width || 1.4} />
                  ))}
                  {freqMode === 'zoom' && freqSel.x1 != null && freqSel.x2 != null && (
                    <ReferenceArea x1={freqSel.x1} x2={freqSel.x2} y1={freqSel.y1} y2={freqSel.y2} strokeOpacity={0.4} stroke="#334155" fill="#334155" fillOpacity={0.15} />
                  )}
                  {showWaterVapor && WATER_VAPOR_LINES.filter((f) => f >= (validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN)[0] && f <= (validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN)[1]).map((f) => (
                    <ReferenceLine key={f} x={f} stroke="#94a3b8" strokeDasharray="2 3" strokeWidth={1} ifOverflow="extendDomain" />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
              <label className="space-y-1">
                <span className="text-slate-900 block">X min (THz)</span>
                <NumberRangeField
                  value={roundDisp((validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN)[0])}
                  onCommit={(v) => setFreqDomain([v, (validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN)[1]])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">X max (THz)</span>
                <NumberRangeField
                  value={roundDisp((validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN)[1])}
                  onCommit={(v) => setFreqDomain([(validDomain(freqDomain) || DEFAULT_FREQ_DOMAIN)[0], v])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">Y min (dB)</span>
                <NumberRangeField
                  value={roundDisp((validDomain(freqYDomain) || freqYFullDomain)[0])}
                  onCommit={(v) => setFreqYDomain([v, (validDomain(freqYDomain) || freqYFullDomain)[1]])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">Y max (dB)</span>
                <NumberRangeField
                  value={roundDisp((validDomain(freqYDomain) || freqYFullDomain)[1])}
                  onCommit={(v) => setFreqYDomain([(validDomain(freqYDomain) || freqYFullDomain)[0], v])}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
            </div>
            <button onClick={resetFreqView} className="mt-2 text-xs text-slate-600 hover:text-slate-900 underline underline-offset-2">Reset to auto</button>
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
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-slate-900" onClick={() => toggleSort('peakToPeak')}>Peak-to-peak (a.u.){sortArrow('peakToPeak')}</th>
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-slate-900" onClick={() => toggleSort('peakFreq')}>Peak (THz){sortArrow('peakFreq')}</th>
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-slate-900" onClick={() => toggleSort('bwWidth')}>Bandwidth{sortArrow('bwWidth')}</th>
                    <th className="text-right font-normal py-2 pr-4 cursor-pointer select-none hover:text-slate-900" onClick={() => toggleSort('noiseFloorDB')}>Noise floor (dB){sortArrow('noiseFloorDB')}</th>
                    <th className="text-right font-normal py-2 cursor-pointer select-none hover:text-slate-900" onClick={() => toggleSort('snrDB')}>SNR / DR (dB){sortArrow('snrDB')}</th>
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
      )}

      {activeTab === 'power-dep' && (
      <div className="flex flex-row gap-4 p-6 items-start">
        {/* Left: dataset settings */}
        <div className="w-80 flex-shrink-0 space-y-4">
          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">X-axis units</p>
            <div className="flex gap-4 text-xs">
              <label className="flex items-center gap-1.5 text-slate-900">
                <input type="radio" name="powerXUnit" checked={powerXUnit === 'mW'} onChange={() => setPowerXUnit('mW')} className="accent-slate-700" />
                Power (mW)
              </label>
              <label className="flex items-center gap-1.5 text-slate-900">
                <input type="radio" name="powerXUnit" checked={powerXUnit === 'fluence'} onChange={() => setPowerXUnit('fluence')} className="accent-slate-700" />
                Fluence (mJ/cm²)
              </label>
            </div>
            <label className="flex items-center gap-1.5 text-xs text-slate-900 pt-1 border-t border-slate-300">
              <input type="checkbox" checked={showPowerFit} onChange={(e) => setShowPowerFit(e.target.checked)} className="accent-slate-700" />
              Show saturation fit (A = A<sub>max</sub>·P / (P + P<sub>sat</sub>))
            </label>
            <p className="text-xs text-slate-600">Laser specs (used for the fluence conversion):</p>
            <div className="grid grid-cols-1 gap-2 text-xs">
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Repetition rate (MHz)</span>
                <input
                  type="number" min={0} step="any" value={laserRepRate}
                  onChange={(e) => setLaserRepRate(e.target.value)}
                  className="w-20 bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Pulse duration (fs)</span>
                <input
                  type="number" min={0} step="any" value={laserPulseDuration}
                  onChange={(e) => setLaserPulseDuration(e.target.value)}
                  className="w-20 bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="flex items-center justify-between gap-2">
                <span className="text-slate-900">Spot diameter (µm)</span>
                <input
                  type="number" min={0} step="any" value={laserSpotDiameter}
                  onChange={(e) => setLaserSpotDiameter(e.target.value)}
                  className="w-20 bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
            </div>
          </div>

          <div className="rounded-lg border border-slate-400 bg-[#efede8] p-3 space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Dataset settings</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <label className="space-y-1">
                <span className="text-slate-900 block">Number of powers</span>
                <input
                  type="number" min={1} max={200} value={numPowers}
                  onChange={(e) => handleNumPowersChange(e.target.value)}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">Number of datasets</span>
                <input
                  type="number" min={1} max={12} value={numPowerDatasets}
                  onChange={(e) => handleNumPowerDatasetsChange(e.target.value)}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
            </div>

            <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
              {powerDatasets.map((ds) => (
                <div key={ds.id} className="rounded bg-white border border-slate-400 px-2 py-1.5 space-y-1">
                  <div className="flex items-center gap-2">
                    <input
                      type="color"
                      value={/^#[0-9a-fA-F]{6}$/.test(ds.color) ? ds.color : '#000000'}
                      onChange={(e) => updatePowerDataset(ds.id, { color: e.target.value })}
                      className="w-5 h-5 rounded border border-slate-400 p-0 bg-transparent cursor-pointer flex-shrink-0"
                      title="Pick color"
                    />
                    <input
                      value={ds.name}
                      onChange={(e) => updatePowerDataset(ds.id, { name: e.target.value })}
                      className="flex-1 min-w-0 bg-white border border-slate-300 rounded px-1.5 py-0.5 text-xs text-slate-800"
                    />
                  </div>
                  <div className="flex items-center gap-2 pl-7 text-xs text-slate-600">
                    <span>Hex</span>
                    <input
                      value={ds.color}
                      onChange={(e) => updatePowerDataset(ds.id, { color: e.target.value })}
                      onBlur={(e) => { if (!/^#[0-9a-fA-F]{6}$/.test(e.target.value.trim())) updatePowerDataset(ds.id, { color: /^#[0-9a-fA-F]{6}$/.test(ds.color) ? ds.color : '#0d9488' }); }}
                      spellCheck={false}
                      className="w-20 bg-white border border-slate-400 rounded px-1 py-0.5 text-slate-800 font-mono uppercase"
                    />
                    <span>Marker</span>
                    <select
                      value={ds.marker}
                      onChange={(e) => updatePowerDataset(ds.id, { marker: e.target.value })}
                      className="flex-1 bg-white border border-slate-400 rounded px-1 py-0.5 text-slate-800 capitalize"
                    >
                      {MARKER_TYPES.map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: data tables + scatter plot */}
        <div className="flex-1 min-w-0 space-y-4">
          <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono mb-2">Snapshot reference</p>
            {snapshots.length === 0 ? (
              <p className="text-xs text-slate-600">No snapshots yet. Use Snapshot mode on the TDS &amp; FFT tab's time-domain plot to capture values here for copying.</p>
            ) : (
              <div className="overflow-x-auto max-h-64 overflow-y-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-600 border-b border-slate-400">
                      <th className="text-left font-normal py-1 pr-2">Label</th>
                      <th className="text-right font-normal py-1 pr-2">Time ({timeUnit})</th>
                      {snapshotColumns.map((c) => (
                        <th key={c.id} className="text-right font-normal py-1 pr-2">
                          <span className="inline-flex items-center gap-1">
                            <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: c.color }} />
                            {c.name}
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {snapshots.map((s, i) => (
                      <tr key={s.id} className="border-b border-slate-300">
                        <td className="text-left pr-2 font-mono font-semibold text-slate-900">{indexToLetters(i)}</td>
                        <td className="text-right pr-2 font-mono">{fmt(s.time, 4)}</td>
                        {snapshotColumns.map((c) => {
                          const en = s.entries.find((e) => e.id === c.id);
                          return <td key={c.id} className="text-right pr-2 font-mono">{en && en.value != null ? fmt(en.value, 6) : '—'}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between">
            <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Data tables</p>
            <button
              onClick={exportPowerDependenceCsv}
              className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition bg-white"
            >
              <Download size={12} /> Export CSV
            </button>
          </div>

          {powerDatasets.map((ds) => (
            <div key={ds.id} className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm overflow-x-auto">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full inline-block flex-shrink-0" style={{ backgroundColor: ds.color }} />
                  <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">{ds.name}</p>
                  <span className="text-[10px] text-slate-500 capitalize">({ds.marker} marker)</span>
                </div>
                <button
                  onClick={() => clearPowerDatasetValues(ds.id)}
                  className="flex items-center gap-1 text-xs text-slate-600 hover:text-red-700 border border-slate-400 rounded px-2 py-1 hover:border-red-400 hover:bg-red-50 transition"
                >
                  <Trash2 size={12} /> Clear
                </button>
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-600 border-b border-slate-400">
                    <th className="text-right font-normal py-1.5 pr-4">Power (mW)</th>
                    <th className="text-right font-normal py-1.5 pr-4">Min amplitude</th>
                    <th className="text-right font-normal py-1.5 pr-4">Max amplitude</th>
                    <th className="text-right font-normal py-1.5">Peak-to-peak</th>
                  </tr>
                </thead>
                <tbody>
                  {ds.rows.map((row, i) => {
                    const min = Number(row.min), max = Number(row.max);
                    const p2p = Number.isFinite(min) && Number.isFinite(max) ? max - min : null;
                    return (
                      <tr key={i} className="border-b border-slate-200">
                        <td className="text-right pr-4 py-1">
                          <input
                            value={row.power}
                            onChange={(e) => updatePowerRow(ds.id, i, 'power', e.target.value)}
                            onPaste={(e) => handlePowerTablePaste(e, ds.id, i, 'power')}
                            className="w-full text-right bg-white border border-slate-300 rounded px-1.5 py-0.5 font-mono text-xs"
                            placeholder="0.0"
                          />
                        </td>
                        <td className="text-right pr-4 py-1">
                          <input
                            value={row.min}
                            onChange={(e) => updatePowerRow(ds.id, i, 'min', e.target.value)}
                            onPaste={(e) => handlePowerTablePaste(e, ds.id, i, 'min')}
                            className="w-full text-right bg-white border border-slate-300 rounded px-1.5 py-0.5 font-mono text-xs"
                            placeholder="insert value"
                          />
                        </td>
                        <td className="text-right pr-4 py-1">
                          <input
                            value={row.max}
                            onChange={(e) => updatePowerRow(ds.id, i, 'max', e.target.value)}
                            onPaste={(e) => handlePowerTablePaste(e, ds.id, i, 'max')}
                            className="w-full text-right bg-white border border-slate-300 rounded px-1.5 py-0.5 font-mono text-xs"
                            placeholder="insert value"
                          />
                        </td>
                        <td className="text-right font-mono text-xs text-slate-700">{p2p != null ? p2p.toPrecision(6) : '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ))}

          <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Power dependence plot</p>
              <div className="flex gap-1.5">
                <button
                  onClick={() => openExportDialog(powerChartWrapRef, 'thz_power_dependence', buildPowerLegendItems(), 'png')}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <Download size={12} /> PNG
                </button>
                <button
                  onClick={() => openExportDialog(powerChartWrapRef, 'thz_power_dependence', buildPowerLegendItems(), 'svg')}
                  className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                >
                  <Download size={12} /> SVG
                </button>
              </div>
            </div>

            <>
                <div className="h-96" ref={powerChartWrapRef}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart margin={{ top: 15, right: 25, bottom: 40, left: 20 }}>
                      <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                      <XAxis dataKey="x" type="number" domain={[0, powerXTicks.length ? powerXTicks[powerXTicks.length - 1] : 'auto']} ticks={powerXTicks} tickFormatter={(v) => (powerXUnit === 'fluence' ? v.toFixed(2) : Math.round(v))} stroke="#334155" tick={{ fontSize: 11 }}
                        label={{ value: powerXUnit === 'fluence' ? 'Fluence (mJ/cm²)' : 'Power (mW)', position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                      <YAxis dataKey="y" type="number" stroke="#334155" tick={{ fontSize: 11 }}
                        label={{ value: 'Peak-to-peak amplitude (a.u.)', angle: -90, position: 'center', dx: -35, fill: '#334155', fontSize: 11 }} />
                      <Tooltip cursor={false} contentStyle={{ background: 'rgba(255, 255, 255, 0.80)', border: '1px solid rgba(148, 163, 184, 0.85)', fontSize: 12, backdropFilter: 'blur(1.5px)' }} labelStyle={{ color: '#1e293b' }} formatter={(v) => fmtTip(v, 'auto')} labelFormatter={(l) => fmtTipLabel(l, powerXUnit === 'fluence' ? 'mJ/cm²' : 'mW', 2)} />
                      <Customized component={ChartBorder} />
                      {powerPlotData && powerPlotData.map((pd) => (
                        <Scatter key={`${pd.id}-pts`} data={pd.points} fill={pd.color} shape={pd.marker} name={pd.name} line={false} isAnimationActive={false} />
                      ))}
                      {showPowerFit && powerPlotData && powerPlotData.filter((pd) => pd.fit).map((pd) => (
                        <Line key={`${pd.id}-fit`} data={pd.fitLine} dataKey="y" stroke={pd.color} strokeDasharray="5 4" strokeWidth={1.5} dot={false} isAnimationActive={false} legendType="none" />
                      ))}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex flex-wrap gap-x-5 gap-y-2 justify-center pt-3 text-xs text-slate-800">
                  {powerPlotData && powerPlotData.map((pd) => (
                    <span key={pd.id} className="inline-flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: pd.color }} />
                      {pd.name}
                      {showPowerFit && <span className="text-slate-500">
                        {pd.fit ? <>({powerXUnit === 'fluence' ? 'F' : 'P'}<sub>sat</sub> = {roundDisp(pd.fit.Psat)} {powerXUnit === 'fluence' ? 'mJ/cm²' : 'mW'})</> : '(fit unavailable)'}
                      </span>}
                    </span>
                  ))}
                </div>
              </>
          </div>
        </div>
      </div>
      )}

      {activeTab === 'convolution' && (
      <div className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">FFT convolution / deconvolution</p>
          <button
            onClick={addConvCard}
            className="flex items-center gap-1.5 text-xs bg-slate-200 border border-slate-600 text-slate-900 rounded px-3 py-1.5 hover:bg-slate-300 transition"
          >
            <Sparkles size={12} /> Add new plot
          </button>
        </div>

        {convResults.length === 0 && (
          <p className="text-xs text-slate-600">No plots yet. Click "Add new plot," then pick two datasets (loaded data, or an earlier result from this tab) and an operation — convolution multiplies their complex FFT spectra bin-by-bin, deconvolution divides them.</p>
        )}

        {convResults.map((card, index) => {
          const color = /^#[0-9a-fA-F]{6}$/.test(card.color) ? card.color : COLORS[index % COLORS.length];
          const autoXTop = card.freqs ? card.freqs[card.freqs.length - 1] : 6;
          const autoYDomain = card.scaleMagsDB ? computeDbYDomain(card.scaleMagsDB) : [-40, 0];
          const xDomainEff = validDomain(convXDomains[card.id]) || [0, autoXTop];
          const yDomainEff = validDomain(convYDomains[card.id]) || autoYDomain;
          const xTicks = niceTicks(xDomainEff[0], xDomainEff[1]);
          const yTicks = niceTicks(yDomainEff[0], yDomainEff[1]);
          const sel = getConvSel(card.id);
          const mode = getConvMode(card.id);
          const opts = sourceOptionsForCard(index);
          const groupsA = {};
          opts.forEach((o) => { (groupsA[o.group] = groupsA[o.group] || []).push(o); });

          return (
            <div key={card.id} className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <input
                    value={card.name}
                    onChange={(e) => updateConvCard(card.id, { name: e.target.value })}
                    className="bg-white border border-slate-300 rounded px-2 py-1 text-sm font-semibold text-slate-900 w-48"
                  />
                  <select
                    value={card.operation}
                    onChange={(e) => updateConvCard(card.id, { operation: e.target.value })}
                    className="bg-white border border-slate-400 rounded px-2 py-1 text-xs text-slate-800"
                  >
                    <option value="convolve">Convolution (multiply)</option>
                    <option value="deconvolve">Deconvolution (divide)</option>
                  </select>
                  <input
                    type="color"
                    value={color}
                    onChange={(e) => updateConvCard(card.id, { color: e.target.value })}
                    className="w-7 h-7 rounded border border-slate-400 p-0 bg-transparent cursor-pointer flex-shrink-0"
                    title="Pick line color"
                  />
                  <input
                    value={card.color || color}
                    onChange={(e) => updateConvCard(card.id, { color: e.target.value })}
                    onBlur={(e) => { if (!/^#[0-9a-fA-F]{6}$/.test(e.target.value.trim())) updateConvCard(card.id, { color }); }}
                    spellCheck={false}
                    className="w-20 bg-white border border-slate-400 rounded px-1.5 py-1 text-xs text-slate-800 font-mono uppercase"
                  />
                </div>
                <button onClick={() => removeConvCard(card.id)} className="text-slate-400 hover:text-red-600">
                  <Trash2 size={14} />
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3 text-xs">
                <label className="space-y-1">
                  <span className="text-slate-900 block">Dataset A {card.operation === 'deconvolve' ? '(numerator)' : ''}</span>
                  <select
                    value={card.sourceAId}
                    onChange={(e) => updateConvCard(card.id, { sourceAId: e.target.value })}
                    className="w-full bg-white border border-slate-400 rounded px-2 py-1.5 text-slate-800"
                  >
                    <option value="">— select —</option>
                    {Object.entries(groupsA).map(([g, items]) => (
                      <optgroup key={g} label={g}>
                        {items.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
                      </optgroup>
                    ))}
                  </select>
                </label>
                <label className="space-y-1">
                  <span className="text-slate-900 block">Dataset B {card.operation === 'deconvolve' ? '(denominator / reference)' : ''}</span>
                  <select
                    value={card.sourceBId}
                    onChange={(e) => updateConvCard(card.id, { sourceBId: e.target.value })}
                    className="w-full bg-white border border-slate-400 rounded px-2 py-1.5 text-slate-800"
                  >
                    <option value="">— select —</option>
                    {Object.entries(groupsA).map(([g, items]) => (
                      <optgroup key={g} label={g}>
                        {items.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
                      </optgroup>
                    ))}
                  </select>
                </label>
              </div>

              {card.operation === 'deconvolve' && (
                <label className="flex items-center gap-2 text-xs mb-3">
                  <span className="text-slate-900">Shade threshold (dB below Dataset B's peak)</span>
                  <input
                    type="number" value={card.shadeThresholdDB}
                    onChange={(e) => updateConvCard(card.id, { shadeThresholdDB: e.target.value })}
                    className="w-16 bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                  />
                  <span className="text-slate-500">— shaded region marks where B is too weak for reliable division</span>
                </label>
              )}

              {card.error ? (
                <p className="text-xs text-amber-800 bg-amber-50 border border-amber-300 rounded px-3 py-2">{card.error}</p>
              ) : (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-2 text-xs">
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X min (THz)</span>
                      <NumberRangeField
                        value={roundDisp(xDomainEff[0])}
                        onCommit={(v) => setConvXDomains((prev) => ({ ...prev, [card.id]: [v, xDomainEff[1]] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X max (THz)</span>
                      <NumberRangeField
                        value={roundDisp(xDomainEff[1])}
                        onCommit={(v) => setConvXDomains((prev) => ({ ...prev, [card.id]: [xDomainEff[0], v] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y min (dB)</span>
                      <NumberRangeField
                        value={roundDisp(yDomainEff[0])}
                        onCommit={(v) => setConvYDomains((prev) => ({ ...prev, [card.id]: [v, yDomainEff[1]] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y max (dB)</span>
                      <NumberRangeField
                        value={roundDisp(yDomainEff[1])}
                        onCommit={(v) => setConvYDomains((prev) => ({ ...prev, [card.id]: [yDomainEff[0], v] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                  </div>
                  <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                    <p className="text-xs text-slate-500">{card.sourceAName} {card.operation === 'convolve' ? '×' : '÷'} {card.sourceBName}</p>
                    <div className="flex gap-1.5">
                      <button
                        onClick={() => setConvModes((prev) => ({ ...prev, [card.id]: 'zoom' }))}
                        title="Drag to zoom into a region"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${mode === 'zoom' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <ZoomIn size={12} /> Zoom
                      </button>
                      <button
                        onClick={() => setConvModes((prev) => ({ ...prev, [card.id]: 'pan' }))}
                        title="Drag to shift the view"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${mode === 'pan' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <Move size={12} /> Pan
                      </button>
                      <button
                        onClick={() => resetConvView(card.id)}
                        title="Reset to full view"
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <RotateCcw size={12} /> Reset
                      </button>
                      <span className="w-px bg-slate-300 mx-0.5" />
                      <button
                        onClick={() => openExportDialog(getConvChartRef(card.id), (card.name || 'result').replace(/[^a-z0-9_-]+/gi, '_'), [{ name: card.name, color }], 'png')}
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <Download size={12} /> PNG
                      </button>
                      <button
                        onClick={() => openExportDialog(getConvChartRef(card.id), (card.name || 'result').replace(/[^a-z0-9_-]+/gi, '_'), [{ name: card.name, color }], 'svg')}
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <Download size={12} /> SVG
                      </button>
                      <button
                        onClick={() => exportConvCsv(card)}
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <Download size={12} /> CSV
                      </button>
                    </div>
                  </div>
                  <div
                    className="h-80 select-none" ref={getConvChartRef(card.id)} onMouseDown={(e) => e.preventDefault()}
                    style={{ cursor: mode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        margin={{ top: 15, right: 15, bottom: 40, left: 10 }}
                        onMouseDown={(e) => handleConvMouseDown(e, card.id, xDomainEff, yDomainEff)}
                        onMouseMove={(e) => handleConvMouseMove(e, card.id)}
                        onMouseUp={() => handleConvMouseUp(card.id)}
                        onMouseLeave={() => handleConvMouseLeave(card.id)}
                        onDoubleClick={() => resetConvView(card.id)}
                      >
                        <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" domain={xDomainEff} allowDataOverflow ticks={xTicks} tickFormatter={(v) => v.toFixed(2)} stroke="#334155" tick={{ fontSize: 11 }}
                          label={{ value: 'Frequency (THz)', position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                        <YAxis domain={yDomainEff} allowDataOverflow ticks={yTicks} tickFormatter={(v) => v.toFixed(2)} stroke="#334155" tick={{ fontSize: 11 }}
                          label={{ value: 'Magnitude (dB)', angle: -90, position: 'center', dx: -32, fill: '#334155', fontSize: 11 }} />
                        <Tooltip cursor={false} contentStyle={{ background: 'rgba(255, 255, 255, 0.80)', border: '1px solid rgba(148, 163, 184, 0.85)', fontSize: 12, backdropFilter: 'blur(1.5px)' }} labelStyle={{ color: '#1e293b' }} formatter={(v) => fmtTip(v, 'db')} labelFormatter={(l) => fmtTipLabel(l, 'THz')} />
                        <Customized component={ChartBorder} />
                        {card.unreliableSpans && card.unreliableSpans.map((span, i) => (
                          <ReferenceArea key={i} x1={span[0]} x2={span[1]} fill="#94a3b8" fillOpacity={0.18} stroke="none" ifOverflow="extendDomain" />
                        ))}
                        {mode === 'zoom' && sel.x1 != null && sel.x2 != null && (
                          <ReferenceArea x1={sel.x1} x2={sel.x2} y1={sel.y1} y2={sel.y2} strokeOpacity={0.4} stroke="#334155" fill="#334155" fillOpacity={0.15} />
                        )}
                        <Line data={card.chartData} dataKey="y" stroke={color} dot={false} isAnimationActive={false} strokeWidth={1.4} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </div>
          );
        })}

        <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
          <p className="text-xs uppercase tracking-wide text-slate-600 font-mono mb-2">Compare results</p>
          {convResults.filter((r) => !r.error).length === 0 ? (
            <p className="text-xs text-slate-600">Add and configure at least one plot above to compare results here.</p>
          ) : (
            <>
              <div className="flex flex-wrap gap-3 mb-3 text-xs">
                {convResults.filter((r) => !r.error).map((r) => (
                  <label key={r.id} className="flex items-center gap-1.5">
                    <input type="checkbox" checked={compareSelectedIds.includes(r.id)} onChange={() => toggleCompareSelection(r.id)} className="accent-slate-700" />
                    <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: convColorFor(r.id) }} />
                    {r.name}
                  </label>
                ))}
              </div>

              {compareEntries.length === 0 ? (
                <p className="text-xs text-slate-600">Select one or more results above to overlay them here.</p>
              ) : (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-2 text-xs">
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X min (THz)</span>
                      <NumberRangeField
                        value={roundDisp(compareXDomainEffective[0])}
                        onCommit={(v) => setCompareXDomain([v, compareXDomainEffective[1]])}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X max (THz)</span>
                      <NumberRangeField
                        value={roundDisp(compareXDomainEffective[1])}
                        onCommit={(v) => setCompareXDomain([compareXDomainEffective[0], v])}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y min (dB)</span>
                      <NumberRangeField
                        value={roundDisp(compareYDomainEffective[0])}
                        onCommit={(v) => setCompareYDomainState([v, compareYDomainEffective[1]])}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y max (dB)</span>
                      <NumberRangeField
                        value={roundDisp(compareYDomainEffective[1])}
                        onCommit={(v) => setCompareYDomainState([compareYDomainEffective[0], v])}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                  </div>
                  <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                    <button onClick={resetCompareView} className="text-xs text-slate-600 hover:text-slate-900 underline underline-offset-2">
                      Reset to auto
                    </button>
                    <div className="flex gap-1.5">
                      <button
                        onClick={() => setCompareMode('zoom')}
                        title="Drag to zoom into a region"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${compareMode === 'zoom' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <ZoomIn size={12} /> Zoom
                      </button>
                      <button
                        onClick={() => setCompareMode('pan')}
                        title="Drag to shift the view"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${compareMode === 'pan' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <Move size={12} /> Pan
                      </button>
                      <button
                        onClick={resetCompareView}
                        title="Reset to full view"
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <RotateCcw size={12} /> Reset
                      </button>
                      <span className="w-px bg-slate-300 mx-0.5" />
                      <button
                        onClick={() => openExportDialog(compareChartRef, 'thz_conv_compare', compareEntries.map((e) => ({ name: e.name, color: convColorFor(e.id) })), 'png')}
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <Download size={12} /> PNG
                      </button>
                      <button
                        onClick={() => openExportDialog(compareChartRef, 'thz_conv_compare', compareEntries.map((e) => ({ name: e.name, color: convColorFor(e.id) })), 'svg')}
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <Download size={12} /> SVG
                      </button>
                      <button
                        onClick={exportCompareCsv}
                        className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition"
                      >
                        <Download size={12} /> CSV
                      </button>
                    </div>
                  </div>
                  <div
                    className="h-96 select-none" ref={compareChartRef} onMouseDown={(e) => e.preventDefault()}
                    style={{ cursor: compareMode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        margin={{ top: 15, right: 15, bottom: 40, left: 10 }}
                        onMouseDown={handleCompareMouseDown}
                        onMouseMove={handleCompareMouseMove}
                        onMouseUp={handleCompareMouseUp}
                        onMouseLeave={handleCompareMouseLeave}
                        onDoubleClick={resetCompareView}
                      >
                        <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" domain={compareXDomainEffective} allowDataOverflow ticks={niceTicks(compareXDomainEffective[0], compareXDomainEffective[1])} tickFormatter={(v) => v.toFixed(2)} stroke="#334155" tick={{ fontSize: 11 }}
                          label={{ value: 'Frequency (THz)', position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                        <YAxis domain={compareYDomainEffective} allowDataOverflow ticks={niceTicks(compareYDomainEffective[0], compareYDomainEffective[1])} tickFormatter={(v) => v.toFixed(2)} stroke="#334155" tick={{ fontSize: 11 }}
                          label={{ value: 'Magnitude (dB)', angle: -90, position: 'center', dx: -32, fill: '#334155', fontSize: 11 }} />
                        <Tooltip cursor={false} contentStyle={{ background: 'rgba(255, 255, 255, 0.80)', border: '1px solid rgba(148, 163, 184, 0.85)', fontSize: 12, backdropFilter: 'blur(1.5px)' }} labelStyle={{ color: '#1e293b' }} formatter={(v) => fmtTip(v, 'db')} labelFormatter={(l) => fmtTipLabel(l, 'THz')} />
                        <Legend verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: 11, paddingTop: 20 }} />
                        <Customized component={ChartBorder} />
                        {compareMode === 'zoom' && compareSel.x1 != null && compareSel.x2 != null && (
                          <ReferenceArea x1={compareSel.x1} x2={compareSel.x2} y1={compareSel.y1} y2={compareSel.y2} strokeOpacity={0.4} stroke="#334155" fill="#334155" fillOpacity={0.15} />
                        )}
                        {compareEntries.map((e) => (
                          <Line key={e.id} data={e.chartData} dataKey="y" name={e.name} stroke={convColorFor(e.id)} dot={false} isAnimationActive={false} strokeWidth={1.4} />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
      )}

      {activeTab === 'arithmetic' && (
      <div className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-xs uppercase tracking-wide text-slate-600 font-mono">Time-domain waveform math</p>
          <button
            onClick={addArithCard}
            className="flex items-center gap-1.5 text-xs bg-slate-200 border border-slate-600 text-slate-900 rounded px-3 py-1.5 hover:bg-slate-300 transition"
          >
            <Sparkles size={12} /> Add new plot
          </button>
        </div>

        {arithResults.length === 0 && (
          <p className="text-xs text-slate-600">No plots yet. Click "Add new plot," then pick waveforms (loaded data from TDS &amp; FFT, or an earlier result from this tab) and an operation: show one as is, flip it (×−1), or add/subtract two waveforms point by point in the time domain.</p>
        )}

        {arithResults.map((card, index) => {
          const color = /^#[0-9a-fA-F]{6}$/.test(card.color) ? card.color : COLORS[index % COLORS.length];
          const needsB = arithNeedsB(card.operation);
          const showIn = card.showInputs && card.operation !== 'none' && !card.error;
          const autoX = card.time ? [card.time[0], card.time[card.time.length - 1]] : [0, 1];
          const autoY = card.time ? paddedRange([card.amplitude, ...(showIn ? card.inputs.map((i) => i.ys) : [])]) : [-1, 1];
          const xDomainEff = validDomain(convXDomains[card.id]) || autoX;
          const yDomainEff = validDomain(convYDomains[card.id]) || autoY;
          const sel = getConvSel(card.id);
          const mode = getConvMode(card.id);
          const opts = arithSourceOptions(index);
          const groups = {};
          opts.forEach((o) => { (groups[o.group] = groups[o.group] || []).push(o); });
          const legendItems = [{ name: card.name, color }, ...(showIn ? card.inputs.map((inp, k) => ({ name: inp.name, color: INPUT_COLORS[k] })) : [])];
          const sourceSelect = (value, key) => (
            <select
              value={value}
              onChange={(e) => updateArithCard(card.id, { [key]: e.target.value })}
              className="w-full bg-white border border-slate-400 rounded px-2 py-1.5 text-slate-800"
            >
              <option value="">— select —</option>
              {Object.entries(groups).map(([g, items]) => (
                <optgroup key={g} label={g}>
                  {items.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
                </optgroup>
              ))}
            </select>
          );

          return (
            <div key={card.id} className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
              <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <input
                    value={card.name}
                    onChange={(e) => updateArithCard(card.id, { name: e.target.value })}
                    className="bg-white border border-slate-300 rounded px-2 py-1 text-sm font-semibold text-slate-900 w-48"
                  />
                  <select
                    value={card.operation}
                    onChange={(e) => updateArithCard(card.id, { operation: e.target.value })}
                    className="bg-white border border-slate-400 rounded px-2 py-1 text-xs text-slate-800"
                  >
                    {ARITH_OPS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </select>
                  <input
                    type="color"
                    value={color}
                    onChange={(e) => updateArithCard(card.id, { color: e.target.value })}
                    className="w-7 h-7 rounded border border-slate-400 p-0 bg-transparent cursor-pointer flex-shrink-0"
                    title="Pick line color"
                  />
                  <input
                    value={card.color || color}
                    onChange={(e) => updateArithCard(card.id, { color: e.target.value })}
                    onBlur={(e) => { if (!/^#[0-9a-fA-F]{6}$/.test(e.target.value.trim())) updateArithCard(card.id, { color }); }}
                    spellCheck={false}
                    className="w-20 bg-white border border-slate-400 rounded px-1.5 py-1 text-xs text-slate-800 font-mono uppercase"
                  />
                </div>
                <button onClick={() => removeArithCard(card.id)} className="text-slate-400 hover:text-red-600">
                  <Trash2 size={14} />
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3 text-xs">
                <label className="space-y-1">
                  <span className="text-slate-900 block">Dataset A</span>
                  {sourceSelect(card.sourceAId, 'sourceAId')}
                </label>
                {needsB && (
                  <label className="space-y-1">
                    <span className="text-slate-900 block">Dataset B {card.operation === 'subtract' ? '(subtracted from A)' : ''}</span>
                    {sourceSelect(card.sourceBId, 'sourceBId')}
                  </label>
                )}
              </div>

              {card.operation !== 'none' && (
                <label className="flex items-center gap-1.5 text-xs text-slate-900 mb-3">
                  <input type="checkbox" checked={card.showInputs !== false} onChange={(e) => updateArithCard(card.id, { showInputs: e.target.checked })} className="accent-slate-700" />
                  Show input waveforms (dashed) for comparison
                </label>
              )}

              {card.error ? (
                <p className="text-xs text-amber-800 bg-amber-50 border border-amber-300 rounded px-3 py-2">{card.error}</p>
              ) : (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-2 text-xs">
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X min ({timeUnit})</span>
                      <NumberRangeField
                        value={roundDisp(xDomainEff[0])}
                        onCommit={(v) => setConvXDomains((prev) => ({ ...prev, [card.id]: [v, xDomainEff[1]] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X max ({timeUnit})</span>
                      <NumberRangeField
                        value={roundDisp(xDomainEff[1])}
                        onCommit={(v) => setConvXDomains((prev) => ({ ...prev, [card.id]: [xDomainEff[0], v] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y min (a.u.)</span>
                      <NumberRangeField
                        value={roundDisp(yDomainEff[0])}
                        onCommit={(v) => setConvYDomains((prev) => ({ ...prev, [card.id]: [v, yDomainEff[1]] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y max (a.u.)</span>
                      <NumberRangeField
                        value={roundDisp(yDomainEff[1])}
                        onCommit={(v) => setConvYDomains((prev) => ({ ...prev, [card.id]: [yDomainEff[0], v] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                  </div>
                  <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                    <p className="text-xs text-slate-500">Result = {arithFormula(card)}</p>
                    <div className="flex gap-1.5 flex-wrap">
                      <button
                        onClick={() => setConvModes((prev) => ({ ...prev, [card.id]: 'zoom' }))}
                        title="Drag to zoom into a region"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${mode === 'zoom' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <ZoomIn size={12} /> Zoom
                      </button>
                      <button
                        onClick={() => setConvModes((prev) => ({ ...prev, [card.id]: 'pan' }))}
                        title="Drag to shift the view"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${mode === 'pan' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <Move size={12} /> Pan
                      </button>
                      <button onClick={() => resetConvView(card.id)} title="Reset to full view" className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <RotateCcw size={12} /> Reset
                      </button>
                      <span className="w-px bg-slate-300 mx-0.5" />
                      <button onClick={() => openExportDialog(getConvChartRef(card.id), (card.name || 'waveform').replace(/[^a-z0-9_-]+/gi, '_'), legendItems, 'png')} className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <Download size={12} /> PNG
                      </button>
                      <button onClick={() => openExportDialog(getConvChartRef(card.id), (card.name || 'waveform').replace(/[^a-z0-9_-]+/gi, '_'), legendItems, 'svg')} className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <Download size={12} /> SVG
                      </button>
                      <button onClick={() => exportArithCsv(card)} className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <Download size={12} /> CSV
                      </button>
                      <button onClick={() => sendArithToTab1(card)} title="Copy this result into TDS &amp; FFT as a new dataset (FFT, metrics, snapshots)" className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        {arithSentIds.includes(card.id) ? <><Check size={12} /> Sent</> : <><Send size={12} /> Send to TDS &amp; FFT</>}
                      </button>
                    </div>
                  </div>
                  {card.notes && card.notes.length > 0 && (
                    <p className="text-xs text-slate-600 bg-slate-50 border border-slate-300 rounded px-3 py-1.5 mb-2">{card.notes.join(' ')}</p>
                  )}
                  <div
                    className="h-80 select-none" ref={getConvChartRef(card.id)} onMouseDown={(e) => e.preventDefault()}
                    style={{ cursor: mode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        margin={{ top: 15, right: 15, bottom: 40, left: 20 }}
                        onMouseDown={(e) => handleConvMouseDown(e, card.id, xDomainEff, yDomainEff)}
                        onMouseMove={(e) => handleConvMouseMove(e, card.id)}
                        onMouseUp={() => handleConvMouseUp(card.id)}
                        onMouseLeave={() => handleConvMouseLeave(card.id)}
                        onDoubleClick={() => resetConvView(card.id)}
                      >
                        <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" domain={xDomainEff} allowDataOverflow ticks={niceTicks(xDomainEff[0], xDomainEff[1])} stroke="#334155" tick={{ fontSize: 11 }}
                          label={{ value: `Time (${timeUnit})`, position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                        <YAxis domain={yDomainEff} allowDataOverflow ticks={niceTicks(yDomainEff[0], yDomainEff[1])} stroke="#334155" tick={{ fontSize: 11 }} width={72}
                          tickFormatter={(v) => (v === 0 ? '0.00e+0' : v.toExponential(2))}
                          label={{ value: 'E-field (a.u.)', angle: -90, position: 'insideLeft', fill: '#334155', fontSize: 11 }} />
                        <Tooltip cursor={false} contentStyle={{ background: 'rgba(255, 255, 255, 0.80)', border: '1px solid rgba(148, 163, 184, 0.85)', fontSize: 12, backdropFilter: 'blur(1.5px)' }} labelStyle={{ color: '#1e293b' }} formatter={(v) => fmtTip(v, 'sci')} labelFormatter={(l) => fmtTipLabel(l, timeUnit)} />
                        <Customized component={ChartBorder} />
                        <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={1} />
                        {mode === 'zoom' && sel.x1 != null && sel.x2 != null && (
                          <ReferenceArea x1={sel.x1} x2={sel.x2} y1={sel.y1} y2={sel.y2} strokeOpacity={0.4} stroke="#334155" fill="#334155" fillOpacity={0.15} />
                        )}
                        {showIn && card.inputs.map((inp, k) => (
                          <Line key={inp.key} data={inp.chartData} dataKey="y" name={inp.name} stroke={INPUT_COLORS[k]} strokeDasharray={k === 0 ? '4 3' : '1.5 2.5'} dot={false} isAnimationActive={false} strokeWidth={1.1} />
                        ))}
                        <Line data={card.chartData} dataKey="y" name={card.name} stroke={color} dot={false} isAnimationActive={false} strokeWidth={1.6} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  {showIn && (
                    <div className="flex flex-wrap gap-x-5 gap-y-1 justify-center pt-2 text-xs text-slate-800">
                      {legendItems.map((li, k) => (
                        <span key={k} className="inline-flex items-center gap-1.5">
                          <span className="w-4 inline-block border-t-2" style={{ borderColor: li.color, borderStyle: k === 0 ? 'solid' : 'dashed' }} />
                          {li.name}
                        </span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}

        <div className="rounded-lg border border-slate-400 bg-white p-4 shadow-sm">
          <p className="text-xs uppercase tracking-wide text-slate-600 font-mono mb-2">Compare results</p>
          {arithResults.filter((r) => !r.error).length === 0 ? (
            <p className="text-xs text-slate-600">Add and configure at least one plot above to compare results here.</p>
          ) : (() => {
            const cmpAutoX = arithCompareEntries.length
              ? [Math.min(...arithCompareEntries.map((e) => e.time[0])), Math.max(...arithCompareEntries.map((e) => e.time[e.time.length - 1]))]
              : [0, 1];
            const cmpAutoY = arithCompareEntries.length ? paddedRange(arithCompareEntries.map((e) => e.amplitude)) : [-1, 1];
            const cmpX = validDomain(convXDomains[ARITH_COMPARE_ID]) || cmpAutoX;
            const cmpY = validDomain(convYDomains[ARITH_COMPARE_ID]) || cmpAutoY;
            const cmpMode = getConvMode(ARITH_COMPARE_ID);
            const cmpSel = getConvSel(ARITH_COMPARE_ID);
            const cmpLegend = arithCompareEntries.map((e) => ({ name: e.name, color: arithColorFor(e.id) }));
            return (
              <>
                <div className="flex flex-wrap gap-3 mb-3 text-xs">
                  {arithResults.filter((r) => !r.error).map((r) => (
                    <label key={r.id} className="flex items-center gap-1.5">
                      <input type="checkbox" checked={arithCompareIds.includes(r.id)} onChange={() => toggleArithCompare(r.id)} className="accent-slate-700" />
                      <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: arithColorFor(r.id) }} />
                      {r.name}
                    </label>
                  ))}
                </div>
                {arithCompareEntries.length === 0 ? (
                  <p className="text-xs text-slate-600">Select one or more results above to overlay them here.</p>
                ) : (
                  <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-2 text-xs">
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X min ({timeUnit})</span>
                      <NumberRangeField
                        value={roundDisp(cmpX[0])}
                        onCommit={(v) => setConvXDomains((prev) => ({ ...prev, [ARITH_COMPARE_ID]: [v, cmpX[1]] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">X max ({timeUnit})</span>
                      <NumberRangeField
                        value={roundDisp(cmpX[1])}
                        onCommit={(v) => setConvXDomains((prev) => ({ ...prev, [ARITH_COMPARE_ID]: [cmpX[0], v] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y min (a.u.)</span>
                      <NumberRangeField
                        value={roundDisp(cmpY[0])}
                        onCommit={(v) => setConvYDomains((prev) => ({ ...prev, [ARITH_COMPARE_ID]: [v, cmpY[1]] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                    <label className="space-y-1">
                      <span className="text-slate-900 block">Y max (a.u.)</span>
                      <NumberRangeField
                        value={roundDisp(cmpY[1])}
                        onCommit={(v) => setConvYDomains((prev) => ({ ...prev, [ARITH_COMPARE_ID]: [cmpY[0], v] }))}
                        className="w-full bg-white border border-slate-400 rounded px-1.5 py-1 text-slate-800"
                      />
                    </label>
                  </div>
                  <div className="flex items-center justify-end mb-2 flex-wrap gap-2">
                    <div className="flex gap-1.5 flex-wrap">
                      <button
                        onClick={() => setConvModes((prev) => ({ ...prev, [ARITH_COMPARE_ID]: 'zoom' }))}
                        title="Drag to zoom into a region"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${cmpMode === 'zoom' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <ZoomIn size={12} /> Zoom
                      </button>
                      <button
                        onClick={() => setConvModes((prev) => ({ ...prev, [ARITH_COMPARE_ID]: 'pan' }))}
                        title="Drag to shift the view"
                        className={`flex items-center gap-1 text-xs border rounded px-2 py-1 transition ${cmpMode === 'pan' ? 'bg-slate-200 border-slate-600 text-slate-900' : 'text-slate-800 border-slate-400 hover:border-slate-500 hover:bg-slate-100'}`}
                      >
                        <Move size={12} /> Pan
                      </button>
                      <button onClick={() => resetConvView(ARITH_COMPARE_ID)} title="Reset to full view" className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <RotateCcw size={12} /> Reset
                      </button>
                      <span className="w-px bg-slate-300 mx-0.5" />
                      <button onClick={() => openExportDialog(getConvChartRef(ARITH_COMPARE_ID), 'thz_waveform_compare', cmpLegend, 'png')} className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <Download size={12} /> PNG
                      </button>
                      <button onClick={() => openExportDialog(getConvChartRef(ARITH_COMPARE_ID), 'thz_waveform_compare', cmpLegend, 'svg')} className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <Download size={12} /> SVG
                      </button>
                      <button onClick={exportArithCompareCsv} className="flex items-center gap-1 text-xs text-slate-800 hover:text-slate-900 border border-slate-400 rounded px-2 py-1 hover:border-slate-500 hover:bg-slate-100 transition">
                        <Download size={12} /> CSV
                      </button>
                    </div>
                  </div>
                  <div
                    className="h-96 select-none" ref={getConvChartRef(ARITH_COMPARE_ID)} onMouseDown={(e) => e.preventDefault()}
                    style={{ cursor: cmpMode === 'pan' ? 'grab' : 'crosshair', userSelect: 'none', WebkitUserSelect: 'none', MozUserSelect: 'none' }}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        margin={{ top: 15, right: 15, bottom: 40, left: 20 }}
                        onMouseDown={(e) => handleConvMouseDown(e, ARITH_COMPARE_ID, cmpX, cmpY)}
                        onMouseMove={(e) => handleConvMouseMove(e, ARITH_COMPARE_ID)}
                        onMouseUp={() => handleConvMouseUp(ARITH_COMPARE_ID)}
                        onMouseLeave={() => handleConvMouseLeave(ARITH_COMPARE_ID)}
                        onDoubleClick={() => resetConvView(ARITH_COMPARE_ID)}
                      >
                        <CartesianGrid stroke="#cbd5e1" strokeDasharray="3 3" />
                        <XAxis dataKey="x" type="number" domain={cmpX} allowDataOverflow ticks={niceTicks(cmpX[0], cmpX[1])} stroke="#334155" tick={{ fontSize: 11 }}
                          label={{ value: `Time (${timeUnit})`, position: 'insideBottom', offset: -5, fill: '#334155', fontSize: 11 }} />
                        <YAxis domain={cmpY} allowDataOverflow ticks={niceTicks(cmpY[0], cmpY[1])} stroke="#334155" tick={{ fontSize: 11 }} width={72}
                          tickFormatter={(v) => (v === 0 ? '0.00e+0' : v.toExponential(2))}
                          label={{ value: 'E-field (a.u.)', angle: -90, position: 'insideLeft', fill: '#334155', fontSize: 11 }} />
                        <Tooltip cursor={false} contentStyle={{ background: 'rgba(255, 255, 255, 0.80)', border: '1px solid rgba(148, 163, 184, 0.85)', fontSize: 12, backdropFilter: 'blur(1.5px)' }} labelStyle={{ color: '#1e293b' }} formatter={(v) => fmtTip(v, 'sci')} labelFormatter={(l) => fmtTipLabel(l, timeUnit)} />
                        <Legend verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: 11, paddingTop: 20 }} />
                        <Customized component={ChartBorder} />
                        <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={1} />
                        {cmpMode === 'zoom' && cmpSel.x1 != null && cmpSel.x2 != null && (
                          <ReferenceArea x1={cmpSel.x1} x2={cmpSel.x2} y1={cmpSel.y1} y2={cmpSel.y2} strokeOpacity={0.4} stroke="#334155" fill="#334155" fillOpacity={0.15} />
                        )}
                        {arithCompareEntries.map((e) => (
                          <Line key={e.id} data={e.chartData} dataKey="y" name={e.name} stroke={arithColorFor(e.id)} dot={false} isAnimationActive={false} strokeWidth={1.4} />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  </>
                )}
              </>
            );
          })()}
        </div>
      </div>
      )}

      <div className="border-t border-slate-400 px-6 py-3">
        <p className="text-[11px] text-slate-500 font-mono">
          Developed by{' '}
          <a
            href="https://vpjuguilon.github.io"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-700 hover:text-slate-900 underline underline-offset-2"
          >
            vpjuguilon
          </a>
          {' '}· 2026
        </p>
      </div>

      {exportDialog && (
        <div className="fixed inset-0 bg-slate-900/40 flex items-center justify-center z-50 p-4" onClick={() => setExportDialog(null)}>
          <div className="bg-white rounded-lg border border-slate-400 shadow-lg p-5 w-80" onClick={(e) => e.stopPropagation()}>
            <p className="text-sm font-semibold text-slate-900 mb-3">Export {exportDialog.format.toUpperCase()}</p>
            <div className="grid grid-cols-2 gap-2 text-xs mb-3">
              <label className="space-y-1">
                <span className="text-slate-900 block">Width (px)</span>
                <input
                  type="number" min={200} max={4000} value={exportW}
                  onChange={(e) => setExportW(e.target.value)}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
              <label className="space-y-1">
                <span className="text-slate-900 block">Height (px)</span>
                <input
                  type="number" min={200} max={4000} value={exportH}
                  onChange={(e) => setExportH(e.target.value)}
                  className="w-full bg-white border border-slate-500 rounded px-1.5 py-1 text-slate-800"
                />
              </label>
            </div>
            <div className="flex flex-wrap gap-1.5 mb-4">
              <button onClick={() => { setExportW(1200); setExportH(700); }} className="text-xs border border-slate-400 rounded px-2 py-1 text-slate-700 hover:bg-slate-100">Wide 1200×700</button>
              <button onClick={() => { setExportW(1000); setExportH(1000); }} className="text-xs border border-slate-400 rounded px-2 py-1 text-slate-700 hover:bg-slate-100">Square 1000×1000</button>
              <button onClick={() => { setExportW(1600); setExportH(900); }} className="text-xs border border-slate-400 rounded px-2 py-1 text-slate-700 hover:bg-slate-100">Wide 1600×900</button>
            </div>
            <div className="flex justify-end gap-2">
              <button onClick={() => setExportDialog(null)} className="text-xs px-3 py-1.5 rounded border border-slate-400 text-slate-700 hover:bg-slate-100">Cancel</button>
              <button onClick={confirmExport} className="text-xs px-3 py-1.5 rounded bg-slate-700 text-white hover:bg-slate-700">Export</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
