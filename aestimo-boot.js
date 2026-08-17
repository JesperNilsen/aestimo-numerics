/* ===========================================================================
 * aestimo-boot.js  —  Bootstrap resampling inference for return statistics
 * ===========================================================================
 *
 *  Puts an honest confidence interval around every point estimate the other
 *  Aestimo engines produce — a bootstrap CI on a beta, an IRR, a Sharpe.
 *
 *  THE ONE IDEA. The bootstrap is the *plug-in principle* made computational.
 *  We want the sampling distribution of a statistic θ = t(F) under the unknown
 *  data-generating law F. We cannot draw fresh samples from F, but the empirical
 *  CDF F̂ₙ (mass 1/n on each observed point) is the nonparametric maximum-
 *  likelihood estimate of F. So we substitute F̂ₙ for F and draw from it — and
 *  drawing from F̂ₙ is exactly sampling the observed values with replacement.
 *  Every layer below is a consequence of that single substitution.
 *
 *  QUANTILE CONVENTION (pinned, load-bearing). Linear interpolation between
 *  order statistics — numpy default / R type 7. BCa endpoint lookups and the
 *  golden numbers depend on this exact convention. Stated once, here.
 *
 *  STD CONVENTION. Sample standard deviation uses the (n−1) divisor throughout
 *  (Bessel). The one place the (÷n) plug-in variance appears is the bias demo in
 *  the test battery, and it is named there explicitly.
 *
 *  DEPENDENCY. Seeded PRNG `makeRNG(seed)` from aestimo-random.js. All
 *  randomness flows through it; there is no Math.random anywhere in this file.
 *  Φ and Φ⁻¹ (needed by BCa) are carried internally (erf-based normCDF, Acklam
 *  normInv) so the tool is self-contained; if the random module ever exposes a
 *  probit layer, these can be sourced from it instead.
 * ------------------------------------------------------------------------- */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    // Node: pull the seeded generator from the sibling module.
    module.exports = factory(require('./aestimo-random.js'));
  } else {
    // Browser: the random module must have assigned globalThis.AestimoRandom
    // *before* this script runs (splice order guarantees it).
    root.AestimoBoot = factory(root.AestimoRandom);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (AestimoRandom) {
  'use strict';

  // --- the single dependency seam -----------------------------------------
  // If your real module returns something other than a bare `() => [0,1)`
  // function from makeRNG, adapt ONLY this line.
  //
  // MIGRATION (2026-07-06): the real aestimo-random.js (xoshiro128**) returns a
  // generator OBJECT with .next() -> [0,1), not a bare function. This bridge —
  // the single seam the stand-in's header sanctions — wraps it to the bare
  // `() => [0,1)` contract boot consumes. Swapping the sfc32 stand-in for the
  // real stream re-locked boot's three stream-dependent goldens (see the .test.js
  // header + reports/boot-relock.md); every analytic invariant is unchanged.
  const _mkRandom = AestimoRandom.makeRNG;
  const makeRNG = function (seed) {
    const gen = _mkRandom(seed);
    return (typeof gen === 'function') ? gen : function () { return gen.next(); };
  };

  /* =======================================================================
   * Φ and Φ⁻¹  — standard normal CDF and quantile
   * -----------------------------------------------------------------------
   * BCa maps interval probabilities through the normal law twice (bias
   * correction z₀ lives in Φ⁻¹ space; adjusted endpoints come back through Φ).
   * =====================================================================*/

  // Horner helpers. polevl: full polynomial (N+1 coeffs). p1evl: monic, implicit
  // leading 1 (N coeffs). Both descending powers, Cephes convention.
  function polevl(x, c) { let y = c[0]; for (let i = 1; i < c.length; i++) y = y * x + c[i]; return y; }
  function p1evl(x, c) { let y = x + c[0]; for (let i = 1; i < c.length; i++) y = y * x + c[i]; return y; }

  // erf / erfc — Cephes rational-Chebyshev port, accurate to ~1e-15 in double.
  // A numerics tool that locks 12-significant-figure goldens cannot ship the
  // usual 1e-7 A&S approximation: it would make the BCa reduction identity
  // (z₀=a=0 ⇒ percentile endpoints) hold only to ~1e-7, and would smear the
  // second-order BCa endpoints against the true normal. Machine-precision Φ is
  // the correct primitive here.
  const _erfT = [9.60497373987051638749e0, 9.00260197203842689217e1,
                 2.23200534594684319226e3, 7.00332514112805075473e3, 5.55923013010394962768e4];
  const _erfU = [3.35617141647503099647e1, 5.21357949780152679795e2,
                 4.59432382970980127987e3, 2.26290000613890934246e4, 4.92673942608635921086e4];
  const _erfP = [2.46196981473530512524e-10, 5.64189564831068821977e-1, 7.46321056442269912687e0,
                 4.86371970985681366614e1, 1.96520832956077098242e2, 5.26445194995477358631e2,
                 9.34528527171957607540e2, 1.02755188689515710272e3, 5.57535335369399327526e2];
  const _erfQ = [1.32281951154744992508e1, 8.67072140885989742329e1, 3.54937778887819891062e2,
                 9.75708501743205489753e2, 1.82390916687909736289e3, 2.24633760818710981792e3,
                 1.65666309194161350182e3, 5.57535340817727675546e2];
  const _erfR = [5.64189583547755073984e-1, 1.27536670759978104416e0, 5.01905042251180477414e0,
                 6.16021097993053585195e0, 7.40974269950448939160e0, 2.97886665372100240670e0];
  const _erfS = [2.26052863220117276590e0, 9.39603524938001434673e0, 1.20489539808096656605e1,
                 1.70814450747565897222e1, 9.60896809063285878198e0, 3.36907645100081516050e0];

  function erf(x) {
    if (Math.abs(x) > 1) return 1 - erfc(x);
    const z = x * x;
    return x * polevl(z, _erfT) / p1evl(z, _erfU);
  }
  function erfc(a) {
    const x = Math.abs(a);
    if (x < 1) return 1 - erf(a);
    let z = Math.exp(-a * a);
    let p, q;
    if (x < 8) { p = polevl(x, _erfP); q = p1evl(x, _erfQ); }
    else       { p = polevl(x, _erfR); q = p1evl(x, _erfS); }
    let y = (z * p) / q;
    if (a < 0) y = 2 - y;
    return y;
  }

  const _INV_SQRT_2PI = 0.3989422804014327;
  function normPDF(x) { return _INV_SQRT_2PI * Math.exp(-0.5 * x * x); }
  // Φ(x) = ½·erfc(−x/√2). Machine-precision via the Cephes erfc above.
  function normCDF(x) { return 0.5 * erfc(-x / Math.SQRT2); }

  // Φ⁻¹: Acklam's rational approximation (~1.15e-9 relative) followed by ONE
  // Newton step against the machine-precision Φ. Newton is quadratic, so a 1e-9
  // seed collapses to ~1e-18 (double-precision floor) — Φ∘Φ⁻¹ then round-trips
  // at machine epsilon, which is what lets the reduction identity lock tight.
  function normInv(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
               3.754408661907416e+00];
    const plow = 0.02425, phigh = 1 - plow;
    let x, q, r;
    if (p < plow) {
      q = Math.sqrt(-2 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
          ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= phigh) {
      q = p - 0.5; r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
          (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    const e = normCDF(x) - p;         // residual
    const u = e / normPDF(x);         // Newton correction
    return x - u;
  }

  /* =======================================================================
   * LAYER 0 — Empirical primitives
   * -----------------------------------------------------------------------
   * Drawing indices uniformly and reading data[idx] IS drawing from F̂ₙ. This
   * is the plug-in step; nothing else in the file is more fundamental.
   * =====================================================================*/

  // One nonparametric bootstrap resample: n draws with replacement.
  // Index map `floor(rng()*n)` is deliberate — the moving-block engine reuses
  // exactly this map at ℓ=1, which is what makes the ℓ=1 reduction bit-exact.
  function resample(data, rng) {
    const n = data.length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = data[(rng() * n) | 0];
    return out;
  }

  // Quantile of an already-sorted array under the pinned (type 7) convention.
  function quantile(sorted, p) {
    const n = sorted.length;
    if (n === 0) return NaN;
    if (n === 1) return sorted[0];
    if (p <= 0) return sorted[0];
    if (p >= 1) return sorted[n - 1];
    const h = (n - 1) * p;
    const lo = Math.floor(h);
    const frac = h - lo;
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
  }

  // Empirical CDF evaluated at x: fraction of observations ≤ x.
  function ecdf(sorted, x) {
    // binary search for the count of values ≤ x
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] <= x) lo = mid + 1; else hi = mid; }
    return lo / sorted.length;
  }

  const sortedCopy = (a) => a.slice().sort((x, y) => x - y);

  /* =======================================================================
   * LAYER 1 — Statistic library.  Each is a plain stat(sample) -> number.
   * The bootstrap engine is statistic-agnostic; this is just a convenient set,
   * chosen so the set spans the honest range of skew: `mean` is symmetric and
   * unbiased (percentile ≈ BCa), `maxDrawdown` is the most skewed and biased
   * (percentile is silently wrong, BCa earns its keep).
   * =====================================================================*/

  function mean(s) {
    let m = 0;
    for (let i = 0; i < s.length; i++) m += s[i];
    return m / s.length;
  }

  // Sample std, (n−1) divisor, deviation-form accumulation for numerical hygiene.
  function std(s) {
    const n = s.length;
    if (n < 2) return NaN;
    const m = mean(s);
    let ss = 0;
    for (let i = 0; i < n; i++) { const d = s[i] - m; ss += d * d; }
    return Math.sqrt(ss / (n - 1));
  }

  // Sharpe ratio. rf is a PER-PERIOD risk-free rate subtracted from each return
  // (constant, so it shifts the mean but not the std). Annualized by √periods
  // only when periodsPerYear is supplied — the √-of-time rule, which assumes iid
  // returns; that assumption is exactly what the block bootstrap later relaxes
  // when it puts an interval around this number.
  function sharpe(s, opts) {
    opts = opts || {};
    const rf = opts.rf || 0;
    const sd = std(s);
    if (!(sd > 0)) return NaN;
    let sr = (mean(s) - rf) / sd;
    if (opts.periodsPerYear) sr *= Math.sqrt(opts.periodsPerYear);
    return sr;
  }

  // Sample skewness g1 = m3 / m2^{3/2} with biased central moments (÷n). This is
  // the population-analogue estimator; it is itself skewed in small samples,
  // which is part of why we bootstrap rather than trust a normal approximation.
  function skewness(s) {
    const n = s.length;
    const m = mean(s);
    let m2 = 0, m3 = 0;
    for (let i = 0; i < n; i++) { const d = s[i] - m; m2 += d * d; m3 += d * d * d; }
    m2 /= n; m3 /= n;
    if (!(m2 > 0)) return NaN;
    return m3 / Math.pow(m2, 1.5);
  }

  // Maximum drawdown on the COMPOUNDED wealth path W_t = Π(1+r_i), W_0 = 1.
  // Returned as a positive magnitude: the largest peak-to-trough fractional
  // decline. This is the most skewed and most median-biased statistic in the
  // set — bounded below by 0, heavy right tail — so it is the sharpest test of
  // why the naive percentile interval misleads.
  function maxDrawdown(s) {
    let wealth = 1, peak = 1, maxDD = 0;
    for (let i = 0; i < s.length; i++) {
      wealth *= (1 + s[i]);
      if (wealth > peak) peak = wealth;
      const dd = (peak - wealth) / peak;
      if (dd > maxDD) maxDD = dd;
    }
    return maxDD;
  }

  const STATS = { mean, std, sharpe, skewness, maxDrawdown };

  /* =======================================================================
   * LAYER 2 — iid bootstrap engine (the core)
   * -----------------------------------------------------------------------
   * The SD of the replicate distribution estimates the statistic's standard
   * error; the mean of the replicates minus the plug-in estimate estimates its
   * bias. Both are Monte-Carlo approximations to the *ideal* bootstrap
   * (B → ∞ resamples of F̂ₙ), which is itself the plug-in approximation to truth.
   * =====================================================================*/

  function bootstrap(data, stat, opts) {
    opts = opts || {};
    const B = opts.B || 10000;
    const rng = makeRNG(opts.seed);
    const estimate = stat(data);
    const replicates = new Array(B);
    for (let b = 0; b < B; b++) replicates[b] = stat(resample(data, rng));
    const se = std(replicates);
    const bias = mean(replicates) - estimate;
    return { estimate, replicates, se, bias, B };
  }

  /* =======================================================================
   * LAYER 3 — CI ladder (crude → better)
   * -----------------------------------------------------------------------
   * Confidence 1−α. `replicates` need not be pre-sorted; the helpers sort as
   * needed. Each rung makes one fewer assumption than the naive normal rung.
   * =====================================================================*/

  // Normal-theory interval. Assumes the sampling distribution is symmetric and
  // Gaussian — the crudest rung, blind to skew and bias alike.
  function ciNormal(estimate, se, alpha) {
    const z = normInv(1 - alpha / 2);
    return { lo: estimate - z * se, hi: estimate + z * se, method: 'normal' };
  }

  // Basic (pivotal) interval. Treats θ̂ − θ as a pivot and reflects the bootstrap
  // distribution about the estimate. Corrects bias in the *location* sense but
  // can produce endpoints outside a parameter's natural range.
  function ciBasic(estimate, replicates, alpha) {
    const s = sortedCopy(replicates);
    const qLo = quantile(s, alpha / 2);
    const qHi = quantile(s, 1 - alpha / 2);
    return { lo: 2 * estimate - qHi, hi: 2 * estimate - qLo, method: 'basic' };
  }

  // Percentile interval. The empirical quantiles of the replicates. Simple and
  // transformation-respecting (monotone reparametrisations map through cleanly)
  // — BUT it silently inherits any median-bias or skew in the estimator. This is
  // the naive default BCa exists to fix.
  function ciPercentile(replicates, alpha) {
    const s = sortedCopy(replicates);
    return { lo: quantile(s, alpha / 2), hi: quantile(s, 1 - alpha / 2), method: 'percentile' };
  }

  /* =======================================================================
   * LAYER 4 — BCa (the centerpiece, the moat)
   * -----------------------------------------------------------------------
   * Bias-Corrected and accelerated. Two corrections to the percentile method:
   *
   *   z₀ (bias correction): where does the observed estimate sit inside its own
   *       bootstrap distribution? z₀ = Φ⁻¹( #{θ*_b < estimate} / B ). If the
   *       estimator is median-unbiased, half the replicates fall below it and
   *       z₀ = 0 — the correction vanishes.
   *
   *   a (acceleration): the standardized skewness of the jackknife influence
   *       values. It lets the effective standard error vary with θ, which is
   *       precisely the asymmetry the percentile method ignores.
   *
   * BCa is second-order accurate and transformation-respecting. Reporting a
   * percentile interval for a skewed statistic (Sharpe, drawdown) is the same
   * failure mode as Excel's IRR returning one root without flagging multiplicity:
   * a plausible number that misleads. This layer is the point of the tool.
   * =====================================================================*/

  // Bias-correction constant from the replicate distribution.
  function biasCorrection(replicates, estimate, B) {
    let below = 0;
    for (let b = 0; b < B; b++) if (replicates[b] < estimate) below++;
    let p = below / B;
    // Guard the degenerate tails (all/none below) so z₀ stays finite; a full
    // 0 or 1 would send Φ⁻¹ to ±∞ and collapse the interval. Nudge by half a
    // replicate's worth of mass — the conventional continuity fix.
    if (p <= 0) p = 0.5 / B;
    else if (p >= 1) p = 1 - 0.5 / B;
    return normInv(p);
  }

  // Acceleration from the leave-one-out jackknife.
  // a = Σ(θ_(·) − θ_(i))³ / (6·[Σ(θ_(·) − θ_(i))²]^{3/2}),  θ_(·) = mean of θ_(i).
  function acceleration(data, stat) {
    const n = data.length;
    const theta = new Array(n);
    for (let i = 0; i < n; i++) {
      // data with observation i deleted
      const jk = new Array(n - 1);
      let k = 0;
      for (let j = 0; j < n; j++) if (j !== i) jk[k++] = data[j];
      theta[i] = stat(jk);
    }
    const tDot = mean(theta);
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const diff = tDot - theta[i];
      num += diff * diff * diff;
      den += diff * diff;
    }
    if (den === 0) return 0; // flat influence -> no acceleration
    return num / (6 * Math.pow(den, 1.5));
  }

  // Pure endpoint map — factored out so the reduction identity (z₀=0,a=0 ⇒
  // percentile endpoints) is unit-testable exactly, with no Monte-Carlo noise.
  // Returns the two adjusted probabilities [α₁, α₂] to look up in the replicates.
  function bcaAlphas(z0, a, alpha) {
    const za = normInv(alpha / 2);
    const zb = normInv(1 - alpha / 2);
    const adjust = (z) => normCDF(z0 + (z0 + z) / (1 - a * (z0 + z)));
    return [adjust(za), adjust(zb)];
  }

  function ciBCa(data, stat, replicates, estimate, opts) {
    opts = opts || {};
    const alpha = opts.alpha != null ? opts.alpha : 0.05;
    const B = replicates.length;
    const z0 = biasCorrection(replicates, estimate, B);
    const a = acceleration(data, stat);
    const [a1, a2] = bcaAlphas(z0, a, alpha);
    const s = sortedCopy(replicates);
    return {
      lo: quantile(s, a1), hi: quantile(s, a2),
      z0, a, alpha1: a1, alpha2: a2, method: 'bca'
    };
  }

  // Convenience: run the iid bootstrap and return all four intervals at once.
  function bootstrapCI(data, stat, opts) {
    opts = opts || {};
    const alpha = opts.alpha != null ? opts.alpha : 0.05;
    const boot = bootstrap(data, stat, opts);
    return {
      ...boot,
      alpha,
      normal: ciNormal(boot.estimate, boot.se, alpha),
      basic: ciBasic(boot.estimate, boot.replicates, alpha),
      percentile: ciPercentile(boot.replicates, alpha),
      bca: ciBCa(data, stat, boot.replicates, boot.estimate, { alpha })
    };
  }

  /* =======================================================================
   * LAYER 5 — Block bootstraps for dependent data (the domain centerpiece)
   * -----------------------------------------------------------------------
   * Returns are NOT iid: they carry autocorrelation and volatility clustering.
   * Resampling individual observations shatters that temporal structure and
   * UNDERSTATES the sampling variance of any statistic that depends on it — most
   * consequentially the Sharpe ratio and drawdown, whose whole point is the
   * path. The block methods resample contiguous runs, preserving local
   * dependence. The choice between iid and block must be explicit, never a
   * silent default. (Automatic block-length selection à la Politis–White is out
   * of scope — ℓ and p are your knobs.)
   * =====================================================================*/

  // Moving-block (Künsch) bootstrap. Overlapping blocks of fixed length ℓ; draw
  // ⌈n/ℓ⌉ block starts uniformly from the n−ℓ+1 admissible positions, concatenate,
  // truncate to n. At ℓ=1 the start map `floor(rng()*(n-ℓ+1))` becomes
  // `floor(rng()*n)` — identical rng consumption and identical index map as the
  // iid `resample`, so the ℓ=1 reduction is BIT-EXACT, not merely distributional.
  function movingBlockResample(data, rng, ell) {
    const n = data.length;
    const nStarts = n - ell + 1;
    const out = new Array(n);
    let filled = 0;
    while (filled < n) {
      const start = (rng() * nStarts) | 0;
      for (let j = 0; j < ell && filled < n; j++) out[filled++] = data[start + j];
    }
    return out;
  }

  function blockBootstrap(data, stat, opts) {
    opts = opts || {};
    const B = opts.B || 10000;
    const ell = opts.blockLength || 1;
    const rng = makeRNG(opts.seed);
    const estimate = stat(data);
    const replicates = new Array(B);
    for (let b = 0; b < B; b++) replicates[b] = stat(movingBlockResample(data, rng, ell));
    const se = std(replicates);
    const bias = mean(replicates) - estimate;
    return { estimate, replicates, se, bias, B, blockLength: ell };
  }

  // Stationary (Politis–Romano) bootstrap. Block lengths are geometric with
  // continuation probability 1−p (mean length 1/p); wrapping is circular so the
  // resampled series is strictly stationary. At p=1 every "block" is length 1,
  // reducing to iid resampling (asserted distributionally in the battery — the
  // extra continuation coin-flips make it non-bit-identical to `resample`).
  function stationaryBlockResample(data, rng, p) {
    const n = data.length;
    const out = new Array(n);
    let i = (rng() * n) | 0;
    for (let t = 0; t < n; t++) {
      out[t] = data[i];
      if (rng() < p) i = (rng() * n) | 0;   // start a fresh block
      else i = (i + 1) % n;                  // extend the current block (circular)
    }
    return out;
  }

  function stationaryBootstrap(data, stat, opts) {
    opts = opts || {};
    const B = opts.B || 10000;
    const p = opts.p != null ? opts.p : 0.1;
    const rng = makeRNG(opts.seed);
    const estimate = stat(data);
    const replicates = new Array(B);
    for (let b = 0; b < B; b++) replicates[b] = stat(stationaryBlockResample(data, rng, p));
    const se = std(replicates);
    const bias = mean(replicates) - estimate;
    return { estimate, replicates, se, bias, B, p };
  }

  /* =======================================================================
   * LAYER 6 — self-test (mirrors the console's self-test; full battery lives
   * in aestimo-boot.test.js). Returns boolean; throws nothing.
   * =====================================================================*/
  function selfTest() {
    try {
      const seed = 424242;
      // (a) SE of the mean ≈ s/√n
      const d = [];
      { const rng = makeRNG(7); for (let i = 0; i < 60; i++) d.push(normInv(rng()) * 0.04 + 0.01); }
      const b = bootstrap(d, mean, { B: 20000, seed });
      const analytic = std(d) / Math.sqrt(d.length);
      if (Math.abs(b.se - analytic) / analytic > 0.06) return false;
      // (b) BCa reduction identity: z0=0, a=0 -> percentile alphas
      const [r1, r2] = bcaAlphas(0, 0, 0.05);
      if (Math.abs(r1 - 0.025) > 1e-9 || Math.abs(r2 - 0.975) > 1e-9) return false;
      // (c) moving-block ℓ=1 is bit-identical to iid
      const rngA = makeRNG(99), rngB = makeRNG(99);
      const ra = resample(d, rngA), rb = movingBlockResample(d, rngB, 1);
      for (let i = 0; i < ra.length; i++) if (ra[i] !== rb[i]) return false;
      return true;
    } catch (_) { return false; }
  }

  return {
    // primitives
    resample, quantile, ecdf, sortedCopy,
    // stats
    mean, std, sharpe, skewness, maxDrawdown, STATS,
    // normal law
    normCDF, normInv,
    // iid engine + CIs
    bootstrap, ciNormal, ciBasic, ciPercentile, ciBCa, bootstrapCI,
    // BCa internals (exposed for exact unit tests)
    biasCorrection, acceleration, bcaAlphas,
    // block engines
    blockBootstrap, stationaryBootstrap, movingBlockResample, stationaryBlockResample,
    // self-test
    selfTest,
    VERSION: '1.0.0'
  };
});
