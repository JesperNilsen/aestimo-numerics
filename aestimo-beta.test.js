/*
 * aestimo-beta.test.js — headless invariant battery.
 *
 * Run: node aestimo-beta.test.js
 * Prints PASS/FAIL per invariant; exits non-zero on any failure.
 *
 * APPEND DISCIPLINE: each future layer adds its own banner + checks ABOVE the
 * footer. The footer (exit logic) must remain the last thing in this file.
 *
 * Status: Layer 0 invariants present. Golden-number locks land in Session 3
 * (once vasicekFromPeers exists).
 */

'use strict';

const { regress, beta, rollingBeta, blume, vasicek, vasicekFromPeers, selfTest } = require('./aestimo-beta.js');

// --- harness ---------------------------------------------------------------
const EPS = 1e-9; // tolerance for exact-relationship invariants
let failures = 0;

function approx(a, b, eps = EPS) {
  if (Number.isNaN(b)) return Number.isNaN(a); // expecting NaN
  return Math.abs(a - b) <= eps;
}
function check(name, cond) {
  console.log((cond ? 'PASS' : 'FAIL') + '  ' + name);
  if (!cond) failures++;
}

// ===========================================================================
// Layer 0 — OLS core
// ===========================================================================

// y = x exactly → β=1, α=0, R²=1, SE=0
(function () {
  const x = [1, 2, 3, 4, 5];
  const r = regress(x, x);
  check('L0 identity: beta=1', approx(r.beta, 1));
  check('L0 identity: alpha=0', approx(r.alpha, 0));
  check('L0 identity: r2=1', approx(r.r2, 1));
  check('L0 identity: se=0', approx(r.se, 0));
})();

// y = a + b·x exactly → β=b, α=a, R²=1
(function () {
  const x = [0, 1, 2, 3, 4, 5, 6];
  const a = 2.5, b = -1.3;
  const r = regress(x, x.map(v => a + b * v));
  check('L0 affine: beta=b', approx(r.beta, b));
  check('L0 affine: alpha=a', approx(r.alpha, a));
  check('L0 affine: r2=1', approx(r.r2, 1));
})();

// y = k·x → β=k
(function () {
  const x = [1, 3, 5, 7, 9, 11];
  const k = 4.2;
  check('L0 proportional: beta=k', approx(regress(x, x.map(v => k * v)).beta, k));
})();

// scaling all y by c → β and α scale by c
(function () {
  const x = [1, 2, 4, 8, 16];
  const y = [3, 5, 4, 9, 7];
  const base = regress(x, y);
  const c = 3.7;
  const s = regress(x, y.map(v => c * v));
  check('L0 scale-y: beta scales', approx(s.beta, c * base.beta));
  check('L0 scale-y: alpha scales', approx(s.alpha, c * base.alpha));
})();

// adding constant d to all y → α shifts by d, β unchanged
(function () {
  const x = [1, 2, 4, 8, 16];
  const y = [3, 5, 4, 9, 7];
  const base = regress(x, y);
  const d = 11.2;
  const sh = regress(x, y.map(v => v + d));
  check('L0 shift-y: beta unchanged', approx(sh.beta, base.beta));
  check('L0 shift-y: alpha += d', approx(sh.alpha, base.alpha + d));
})();

// permutation invariance: shuffling paired (x,y) leaves β, α unchanged
(function () {
  const x = [1, 2, 4, 8, 16];
  const y = [3, 5, 4, 9, 7];
  const base = regress(x, y);
  const idx = [3, 0, 4, 1, 2];
  const xp = idx.map(i => x[i]);
  const yp = idx.map(i => y[i]);
  const p = regress(xp, yp);
  check('L0 permutation: beta invariant', approx(p.beta, base.beta));
  check('L0 permutation: alpha invariant', approx(p.alpha, base.alpha));
})();

// degenerate handling
(function () {
  const flat = regress([2, 2, 2, 2], [1, 2, 3, 4]);
  check('L0 flat-x: beta NaN', Number.isNaN(flat.beta));

  const two = regress([1, 2], [2, 4]);
  check('L0 n=2: se NaN', Number.isNaN(two.se));
  check('L0 n=2: beta defined', approx(two.beta, 2));

  const one = regress([5], [9]);
  check('L0 n=1: beta NaN', Number.isNaN(one.beta));
  check('L0 n=1: se NaN', Number.isNaN(one.se));
})();

// ===========================================================================
// Layer 1 — Beta on excess returns
// ===========================================================================

// riskFree = 0 reproduces the plain regression slope
(function () {
  const m = [1, 2, 4, 8, 16];
  const a = [3, 5, 4, 9, 7];
  check('L1 rf=0 implicit ≡ regress slope', approx(beta(m, a).beta, regress(m, a).beta));
  check('L1 rf=0 explicit ≡ regress slope', approx(beta(m, a, { riskFree: 0 }).beta, regress(m, a).beta));
})();

// constant riskFree leaves beta unchanged (scalar and array forms)
(function () {
  const m = [1, 2, 4, 8, 16];
  const a = [3, 5, 4, 9, 7];
  const base = beta(m, a).beta;
  const c = 0.013;
  check('L1 const rf (scalar): beta unchanged', approx(beta(m, a, { riskFree: c }).beta, base));
  check('L1 const rf (array): beta unchanged', approx(beta(m, a, { riskFree: m.map(() => c) }).beta, base));
})();

// returns the FULL regression object — .se present and matching for Vasicek
(function () {
  const m = [1, 2, 4, 8, 16];
  const a = [3, 5, 4, 9, 7];
  const r = beta(m, a);
  const direct = regress(m, a);
  check('L1 full object: se matches regress', approx(r.se, direct.se));
  check('L1 full object: se finite & > 0', Number.isFinite(r.se) && r.se > 0);
  check('L1 full object: n carried', r.n === 5);
})();

// ===========================================================================
// Layer 2 — Rolling-window beta
// ===========================================================================

// window = full length → final slot equals full-sample beta, rest NaN-padded
(function () {
  const m = [1, 2, 4, 8, 16, 7, 3];
  const a = [3, 5, 4, 9, 7, 6, 2];
  const rb = rollingBeta(m, a, m.length);
  check('L2 window=len: length = n', rb.length === m.length);
  check('L2 window=len: last = full-sample beta', approx(rb[rb.length - 1], beta(m, a).beta));
  let pad = true;
  for (let i = 0; i < rb.length - 1; i++) if (!Number.isNaN(rb[i])) pad = false;
  check('L2 window=len: first n−1 are NaN', pad);
})();

// clean linear y = k·x → every window returns exactly k
(function () {
  const m = [1, 2, 3, 4, 5, 6, 7, 8];
  const k = 2.4;
  const a = m.map(v => k * v);
  const w = 4;
  const rb = rollingBeta(m, a, w);
  let allK = true;
  for (let i = w - 1; i < rb.length; i++) if (!approx(rb[i], k)) allK = false;
  check('L2 clean linear: every window beta = k', allK);
})();

// generic window: length = n, first w−1 NaN, remaining finite
(function () {
  const m = [1, 2, 4, 8, 16, 7, 3, 11, 5];
  const a = [3, 5, 4, 9, 7, 6, 2, 8, 1];
  const w = 4;
  const rb = rollingBeta(m, a, w);
  check('L2 padding: length = n', rb.length === m.length);
  let padOK = true;
  for (let i = 0; i < w - 1; i++) if (!Number.isNaN(rb[i])) padOK = false;
  check('L2 padding: first w−1 are NaN', padOK);
  let restOK = true;
  for (let i = w - 1; i < rb.length; i++) if (!Number.isFinite(rb[i])) restOK = false;
  check('L2 padding: remaining entries finite', restOK);
})();

// constant array riskFree threads through the windowed slice path unchanged
(function () {
  const m = [1, 2, 4, 8, 16, 7, 3, 11, 5];
  const a = [3, 5, 4, 9, 7, 6, 2, 8, 1];
  const w = 5;
  const base = rollingBeta(m, a, w);
  const withRf = rollingBeta(m, a, w, { riskFree: m.map(() => 0.02) });
  let same = true;
  for (let i = 0; i < base.length; i++) {
    const bothNaN = Number.isNaN(base[i]) && Number.isNaN(withRf[i]);
    if (!bothNaN && !approx(withRf[i], base[i])) same = false;
  }
  check('L2 const rf array: betas unchanged vs rf=0', same);
})();

// ===========================================================================
// Layer 3 — Blume shrinkage
// ===========================================================================
(function () {
  // strict pull toward the prior (1) for raw betas on both sides of 1
  for (const rb of [0.4, 0.7, 1.3, 1.8, 2.5]) {
    check('L3 blume pulls toward 1 (raw=' + rb + ')', Math.abs(blume(rb) - 1) < Math.abs(rb - 1));
  }
  // default 0.67/0.33 convention honored
  check('L3 blume default w=0.67', approx(blume(1.5), 0.67 * 1.5 + 0.33 * 1));
  // custom w and prior honored
  check('L3 blume custom w,prior', approx(blume(1.5, { w: 0.5, prior: 0.9 }), 0.5 * 1.5 + 0.5 * 0.9));
  // boundaries: w=1 ⇒ raw, w=0 ⇒ prior
  check('L3 blume w=1 ≡ raw', approx(blume(1.5, { w: 1 }), 1.5));
  check('L3 blume w=0 ≡ prior', approx(blume(1.5, { w: 0, prior: 0.8 }), 0.8));
})();

// ===========================================================================
// Layer 4 — Vasicek shrinkage
// ===========================================================================
(function () {
  const rb = 1.4, pm = 1.0;
  // priorVar → ∞ ⇒ no prior information ⇒ result → raw
  check('L4 priorVar→∞ ⇒ raw', approx(vasicek(rb, 0.2, { priorMean: pm, priorVar: 1e12 }), rb, 1e-6));
  // priorVar → 0 ⇒ dogmatic prior ⇒ result → priorMean
  check('L4 priorVar→0 ⇒ priorMean', approx(vasicek(rb, 0.2, { priorMean: pm, priorVar: 1e-12 }), pm, 1e-6));
  // seBeta = 0 ⇒ perfect estimate, full precision ⇒ result = raw
  check('L4 se=0 ⇒ raw', approx(vasicek(rb, 0, { priorMean: pm, priorVar: 0.05 }), rb));
  // monotone: larger se ⇒ result nearer the prior (distance to prior non-increasing)
  let prev = Infinity, mono = true;
  for (const se of [0.05, 0.1, 0.2, 0.4, 0.8]) {
    const d = Math.abs(vasicek(rb, se, { priorMean: pm, priorVar: 0.05 }) - pm);
    if (d > prev) mono = false;
    prev = d;
  }
  check('L4 monotone: larger se ⇒ nearer prior', mono);
  // exact formula pin: σ² = priorVar ⇒ w = 1/2 ⇒ midpoint of raw and prior
  check('L4 σ²=priorVar ⇒ w=0.5 midpoint',
    approx(vasicek(rb, Math.sqrt(0.05), { priorMean: pm, priorVar: 0.05 }), 0.5 * rb + 0.5 * pm));
  // missing priorVar is a usage error
  let threw = false;
  try { vasicek(rb, 0.2, { priorMean: pm }); } catch (e) { threw = true; }
  check('L4 missing priorVar throws', threw);
})();

// ===========================================================================
// Layer 4b — Peer-derived prior
// ===========================================================================
(function () {
  // peers chosen so mean=1.0 and POPULATION variance=0.02 by hand;
  // a sample-variance (÷n−1) bug would move priorVar to ~0.0267 and fail this.
  const peers = [0.8, 1.0, 1.2, 1.0];
  const rb = 1.4, se = 0.1;
  check('L4b peers derive exact mean/var (pop)',
    approx(vasicekFromPeers(rb, se, peers), vasicek(rb, se, { priorMean: 1.0, priorVar: 0.02 })));
})();

// ===========================================================================
// Golden locks — fixed synthetic dataset, captured on first green build.
// Any change that moves these is a BREAKING change requiring explicit
// acknowledgement (backward-compatibility guard). Exact equality: the IEEE-754
// op sequence over these fixed arrays is deterministic across platforms.
// ===========================================================================
(function () {
  const MARKET = [0.012, -0.008, 0.021, 0.005, -0.015, 0.018, -0.003, 0.009, 0.014, -0.011, 0.007, 0.022];
  const PERTURB = [0.001, -0.002, 0.0015, -0.001, 0.0008, -0.0012, 0.0005, -0.0007, 0.0011, -0.0009, 0.0006, -0.0013];
  const ASSET = MARKET.map((m, i) => 1.3 * m + PERTURB[i]);
  const PEERS = [1.10, 0.95, 1.25, 1.40, 0.88, 1.05];

  // Locked literals (shortest round-trip; ~12+ significant figures):
  const GOLDEN_RAW_BETA     = 1.3134640210375332;
  const GOLDEN_SE_BETA      = 0.028953252872596542;
  const GOLDEN_VASICEK_BETA = 1.3079680450883593;

  const fit = beta(MARKET, ASSET);
  const vB = vasicekFromPeers(fit.beta, fit.se, PEERS);

  check('GOLDEN raw beta locked',     fit.beta === GOLDEN_RAW_BETA);
  check('GOLDEN se(beta) locked',     fit.se   === GOLDEN_SE_BETA);
  check('GOLDEN vasicek beta locked', vB       === GOLDEN_VASICEK_BETA);
})();

// ===========================================================================
// Layer 5 — Self-test seal
// ===========================================================================
(function () {
  // The module's in-process integrity check must agree with the full battery.
  check('L5 selfTest() === true', selfTest() === true);

  // API-surface guard: exactly the spec's public functions, nothing more.
  // The "stop at the defined surface" discipline, made executable — any future
  // accidental export fails here.
  const surface = Object.keys(require('./aestimo-beta.js')).sort();
  const expected = ['beta', 'blume', 'regress', 'rollingBeta', 'selfTest', 'vasicek', 'vasicekFromPeers'];
  check('L5 API surface == spec (no extras)',
    surface.length === expected.length && surface.every((k, i) => k === expected[i]));
})();

// === footer (keep last) ====================================================
console.log('');
if (failures > 0) {
  console.log(failures + ' FAILED');
  process.exit(1);
}
console.log('ALL PASS');
