/*
 * aestimo-beta.js — OLS beta estimator with shrinkage.
 *
 * Offline, pure, dependency-free. Inputs are arrays of periodic returns
 * supplied by the caller. UMD-wrapped: CommonJS under Node, and
 * globalThis.AestimoBeta in the browser — so it composes into portfolio the
 * same way solve/chol/random/kelly do. DOM-agnostic.
 *
 * Comments state the MECHANISM, not the syntax — the file is meant to read
 * as a derivation.
 *
 * Build status: COMPLETE. Layers 0–5 (OLS core, beta, rolling, Blume, Vasicek,
 * peer prior, self-test) shipped; golden locks set; public surface frozen.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.AestimoBeta = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
'use strict';

// ---------------------------------------------------------------------------
// Internal stats helpers (exported only when a public layer needs them).
// ---------------------------------------------------------------------------

// Arithmetic mean. Kept separate so the deviation-form sums below read cleanly.
function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

// Population variance in deviation form: Σ(aᵢ−ā)²/n.
// STANCE: ÷n, not ÷(n−1) — the peer set is the prior universe itself, not a
// sample drawn from a larger population, so there is no lost degree of freedom.
// Deviation form, never E[a²]−E[a]² (that form invites catastrophic cancellation).
function variance(a) {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    s += d * d;
  }
  return s / a.length;
}

// ---------------------------------------------------------------------------
// Layer 0 — OLS simple regression core.
// ---------------------------------------------------------------------------

/**
 * regress(x, y) -> { n, beta, alpha, r2, se }
 *
 * The least-squares line y ≈ alpha + beta·x, solved via the normal equations
 * in deviation form. Deviation form is deliberate: computing variance as
 * E[x²] − E[x]² invites catastrophic cancellation; Σ(xᵢ−x̄)² does not.
 */
function regress(x, y) {
  const n = x.length;
  if (n !== y.length) throw new Error('regress: x and y length mismatch');

  // A line is unidentified below two points — return NaN throughout rather
  // than emit a meaningless slope.
  if (n < 2) return { n, beta: NaN, alpha: NaN, r2: NaN, se: NaN };

  const xbar = mean(x);
  const ybar = mean(y);

  // Sxx = Σ(xᵢ−x̄)²   Sxy = Σ(xᵢ−x̄)(yᵢ−ȳ)   Stot = SStot = Σ(yᵢ−ȳ)²
  let Sxx = 0, Sxy = 0, Stot = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - xbar;
    const dy = y[i] - ybar;
    Sxx += dx * dx;
    Sxy += dx * dy;
    Stot += dy * dy;
  }

  // Normal equations: β minimizes Σeᵢ²; α anchors the line at (x̄, ȳ).
  // Flat x ⇒ slope unidentified ⇒ NaN by stance (not Inf, not a throw).
  const beta = Sxx === 0 ? NaN : Sxy / Sxx;
  const alpha = ybar - beta * xbar; // NaN propagates from beta when x is flat.

  // Realized residuals → SSres. R² = 1 − SSres/SStot is the share of y's
  // variance the line explains.
  let SSres = 0;
  for (let i = 0; i < n; i++) {
    const e = y[i] - (alpha + beta * x[i]);
    SSres += e * e;
  }
  // SStot = 0 ⇒ y is constant ⇒ no variance to explain ⇒ R² undefined.
  const r2 = Stot === 0 ? NaN : 1 - SSres / Stot;

  // s² = SSres/(n−2): residual variance on n−2 d.o.f. (α and β each consume
  // one). SE(β) = √(s²/Sxx). With n < 3 there are zero d.o.f. ⇒ SE undefined.
  const s2 = SSres / (n - 2);
  const se = n < 3 ? NaN : Math.sqrt(s2 / Sxx);

  return { n, beta, alpha, r2, se };
}

// ---------------------------------------------------------------------------
// Layer 1 — Beta on excess returns.
// ---------------------------------------------------------------------------

/**
 * beta(market, asset, { riskFree = 0 }) -> full regression object.
 *
 * Beta is the slope of the ASSET's excess returns regressed on the MARKET's
 * excess returns: assetEx ≈ alpha + beta·marketEx. The full object is
 * returned (not just the number) so the pipeline stays one call — Vasicek
 * downstream needs `.se`, the standard error of the slope.
 *
 * STANCE: riskFree = 0 is the approximation valid when the risk-free rate is
 * ~constant over the window. Subtracting any constant from both series leaves
 * the deviations — and therefore beta — unchanged, so a flat risk-free only
 * shifts the intercept. That is documented here, not hidden.
 */
function beta(market, asset, { riskFree = 0 } = {}) {
  const n = market.length;
  if (n !== asset.length) throw new Error('beta: market and asset length mismatch');
  const rf = Array.isArray(riskFree) ? riskFree : null; // array ⇒ per-period
  if (rf && rf.length !== n) throw new Error('beta: riskFree array length mismatch');

  const mEx = new Array(n);
  const aEx = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = rf ? rf[i] : riskFree;
    mEx[i] = market[i] - r;
    aEx[i] = asset[i] - r;
  }
  return regress(mEx, aEx); // x = market excess, y = asset excess
}

// ---------------------------------------------------------------------------
// Layer 2 — Rolling-window beta.
// ---------------------------------------------------------------------------

/**
 * rollingBeta(market, asset, window, opts) -> Array(n)
 *
 * Each entry i (i ≥ window−1) is the beta over the trailing `window`
 * observations ending at i. The first window−1 slots lack enough history and
 * are NaN by construction. window === series length reproduces the
 * full-sample beta exactly in the final slot.
 */
function rollingBeta(market, asset, window, opts = {}) {
  const n = market.length;
  if (n !== asset.length) throw new Error('rollingBeta: market and asset length mismatch');
  if (!Number.isInteger(window) || window < 2) {
    throw new Error('rollingBeta: window must be an integer >= 2');
  }
  const rf = opts.riskFree;
  const rfIsArr = Array.isArray(rf);
  if (rfIsArr && rf.length !== n) {
    throw new Error('rollingBeta: riskFree array length mismatch');
  }

  const out = new Array(n).fill(NaN); // insufficient-history slots stay NaN
  for (let i = window - 1; i < n; i++) {
    const lo = i - window + 1;        // trailing window is [lo, i]
    const winOpts = rfIsArr
      ? { riskFree: rf.slice(lo, i + 1) }
      : (rf === undefined ? {} : { riskFree: rf });
    out[i] = beta(market.slice(lo, i + 1), asset.slice(lo, i + 1), winOpts).beta;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Layer 3 — Blume shrinkage.
// ---------------------------------------------------------------------------

/**
 * blume(rawBeta, { w = 0.67, prior = 1 }) -> w·rawBeta + (1−w)·prior
 *
 * STANCE: raw beta mean-reverts across estimation periods, so pull it toward
 * the market beta of 1. The 0.67/0.33 split is the conventional
 * Bloomberg-adjusted coefficient — a CONVENTION, parameterized here, never
 * implied to have been estimated from the data in front of us.
 */
function blume(rawBeta, { w = 0.67, prior = 1 } = {}) {
  return w * rawBeta + (1 - w) * prior;
}

// ---------------------------------------------------------------------------
// Layer 4 — Vasicek shrinkage (the reason this tool exists).
// ---------------------------------------------------------------------------

/**
 * vasicek(rawBeta, seBeta, { priorMean = 1, priorVar }) -> shrunk beta
 *
 * Bayesian precision weighting. The posterior mean blends estimate and prior
 * in proportion to their PRECISIONS (inverse variances):
 *   w = precision(estimate) / [precision(estimate) + precision(prior)]
 *     = (1/σ²) / [(1/σ²) + (1/priorVar)]
 *     = priorVar / (priorVar + σ²),   where σ² = seBeta².
 *
 * STANCE — this is the whole methodological edge: shrinkage intensity scales
 * with ESTIMATION NOISE, not a fixed constant. A noisy beta (large SE) is
 * pulled hard toward the prior; a precise one is left nearly untouched. That
 * is what Vasicek buys over Blume's flat 2/3.
 *
 * priorVar is required — there is no sensible default for the prior's
 * dispersion (unlike priorMean = 1), so its absence is a usage error.
 */
function vasicek(rawBeta, seBeta, { priorMean = 1, priorVar } = {}) {
  if (priorVar === undefined) throw new Error('vasicek: priorVar is required');
  const sigma2 = seBeta * seBeta;           // estimation variance of the raw beta
  const w = priorVar / (priorVar + sigma2); // weight on the raw estimate
  return w * rawBeta + (1 - w) * priorMean;
}

// ---------------------------------------------------------------------------
// Layer 4b — Peer-derived prior. Turns the estimator from a toy into the
// real thing: the prior is the cross-section of comparable names, not a 1.
// ---------------------------------------------------------------------------

/**
 * vasicekFromPeers(rawBeta, seBeta, peerBetas) -> shrunk beta
 * Prior mean and variance are the mean and (population) variance of the peer
 * betas; the raw estimate is then shrunk toward that empirical prior.
 */
function vasicekFromPeers(rawBeta, seBeta, peerBetas) {
  const priorMean = mean(peerBetas);
  const priorVar = variance(peerBetas);
  return vasicek(rawBeta, seBeta, { priorMean, priorVar });
}

// ---------------------------------------------------------------------------
// Layer 5 — Self-test. A runtime integrity check that ships INSIDE the module
// (so a consumer can verify behavior without the dev battery). It embeds its
// own curated, high-signal invariants — including the golden locks, the
// strongest single integrity signal — and returns a boolean. Silent by
// contract: the test file is the verbose diagnostic; this is the go/no-go.
// Mirrors the appraiser's-console `self-test` pattern.
// ---------------------------------------------------------------------------
function selfTest() {
  const EPS = 1e-9;
  const near = (a, b, eps = EPS) => (Number.isNaN(b) ? Number.isNaN(a) : Math.abs(a - b) <= eps);
  let ok = true;
  const T = (cond) => { if (!cond) ok = false; };

  // OLS exacts + degenerate stance
  {
    const x = [1, 2, 3, 4, 5];
    const r = regress(x, x);
    T(near(r.beta, 1) && near(r.alpha, 0) && near(r.r2, 1));
    const a = 2.5, b = -1.3, xa = [0, 1, 2, 3, 4, 5, 6];
    const ra = regress(xa, xa.map(v => a + b * v));
    T(near(ra.beta, b) && near(ra.alpha, a));
    T(Number.isNaN(regress([2, 2, 2, 2], [1, 2, 3, 4]).beta)); // flat x ⇒ NaN
  }
  // Beta / rolling
  {
    const m = [1, 2, 4, 8, 16], a = [3, 5, 4, 9, 7];
    T(near(beta(m, a).beta, regress(m, a).beta));               // rf=0 ≡ plain slope
    const rb = rollingBeta(m, a, m.length);
    T(near(rb[rb.length - 1], beta(m, a).beta));                // window=len ≡ full sample
  }
  // Blume / Vasicek
  {
    T(Math.abs(blume(1.8) - 1) < Math.abs(1.8 - 1));            // pulls toward prior
    const rb = 1.4, pm = 1.0;
    T(near(vasicek(rb, 0.2, { priorMean: pm, priorVar: 1e12 }), rb, 1e-6));   // ∞ ⇒ raw
    T(near(vasicek(rb, 0.2, { priorMean: pm, priorVar: 1e-12 }), pm, 1e-6));  // 0 ⇒ prior
    T(near(vasicek(rb, 0, { priorMean: pm, priorVar: 0.05 }), rb));           // se=0 ⇒ raw
    T(near(vasicek(rb, Math.sqrt(0.05), { priorMean: pm, priorVar: 0.05 }),
           0.5 * rb + 0.5 * pm));                                             // σ²=priorVar ⇒ midpoint
    T(near(vasicekFromPeers(1.4, 0.1, [0.8, 1.0, 1.2, 1.0]),
           vasicek(1.4, 0.1, { priorMean: 1.0, priorVar: 0.02 })));          // peer prior exact
  }
  // Golden locks (exact equality — IEEE-754 deterministic on this op sequence)
  {
    const MARKET = [0.012, -0.008, 0.021, 0.005, -0.015, 0.018, -0.003, 0.009, 0.014, -0.011, 0.007, 0.022];
    const PERTURB = [0.001, -0.002, 0.0015, -0.001, 0.0008, -0.0012, 0.0005, -0.0007, 0.0011, -0.0009, 0.0006, -0.0013];
    const ASSET = MARKET.map((mm, i) => 1.3 * mm + PERTURB[i]);
    const PEERS = [1.10, 0.95, 1.25, 1.40, 0.88, 1.05];
    const fit = beta(MARKET, ASSET);
    const vB = vasicekFromPeers(fit.beta, fit.se, PEERS);
    T(fit.beta === 1.3134640210375332);
    T(fit.se === 0.028953252872596542);
    T(vB === 1.3079680450883593);
  }
  return ok;
}

return { regress, beta, rollingBeta, blume, vasicek, vasicekFromPeers, selfTest };
});
