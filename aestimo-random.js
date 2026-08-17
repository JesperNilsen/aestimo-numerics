/*
 * Aestimo N — Seeded PRNG & Gaussian sampling engine.
 *
 * Backs the DCF Workstation's Monte Carlo and Gaussian copula. Owns the
 * sampling engine rather than renting Math.random, which is neither
 * seedable nor reproducible — unusable for a simulation whose golden
 * numbers must reproduce bit-for-bit across runs and machines.
 *
 * Pure, dependency-free, deterministic. No hidden module-level state:
 * every function that consumes randomness takes an explicit rng argument.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.AestimoRandom = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------------
  // Layer 0 — Seed expansion (splitmix32)
  //
  // A single 32-bit seed carries little entropy. Feeding it directly into
  // xoshiro128**'s 128-bit state would leave most of that state at zero —
  // a near-degenerate initial condition that produces a visibly poor
  // stream for the first many outputs. splitmix32 is a fast, well-mixed
  // generator in its own right; running it four times over the seed
  // diffuses one integer into four independent-looking uint32 words.
  // Seed the generator through this diffusion step — never hand-set the
  // raw xoshiro state directly.
  // ---------------------------------------------------------------------
  function splitmix32(seed) {
    let s = seed >>> 0;
    return function next() {
      // Weyl sequence increment (golden-ratio constant, odd, full period
      // over 2^32) walks s through every residue before it repeats.
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      // Two rounds of xor-shift/multiply avalanche each input bit across
      // the whole 32-bit output — the standard splitmix32 finalizer.
      z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
      z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
      z = (z ^ (z >>> 15)) >>> 0;
      return z >>> 0;
    };
  }

  // ---------------------------------------------------------------------
  // Layer 1 — Core PRNG (xoshiro128**)
  //
  // Reproducibility is the entire point: identical seed -> identical
  // stream, run to run, machine to machine. Period ~2^128 - 1 with good
  // equidistribution — adequate for simulation. This is NOT cryptographic
  // (the state is trivially invertible from a handful of outputs) and
  // must never be used where unpredictability matters, only where
  // reproducible statistical coverage does.
  // ---------------------------------------------------------------------
  function rotl(x, k) {
    return ((x << k) | (x >>> (32 - k))) >>> 0;
  }

  function makeRNG(seed) {
    const gen = splitmix32(seed >>> 0);
    let s0 = gen();
    let s1 = gen();
    let s2 = gen();
    let s3 = gen();

    function nextU32() {
      // "**" scramble: rotl(s1*5, 7)*9 decorrelates the output from the
      // linear xor-shift update below, which is what gives xoshiro its
      // statistical quality despite a cheap update step.
      const result = (Math.imul(rotl(Math.imul(s1, 5) >>> 0, 7), 9)) >>> 0;

      const t = (s1 << 9) >>> 0;

      s2 = (s2 ^ s0) >>> 0;
      s3 = (s3 ^ s1) >>> 0;
      s1 = (s1 ^ s2) >>> 0;
      s0 = (s0 ^ s3) >>> 0;

      s2 = (s2 ^ t) >>> 0;
      s3 = rotl(s3, 11);

      return result >>> 0;
    }

    function next() {
      // Divide by 2^32 to map the uint32 output into [0, 1).
      return nextU32() / 4294967296;
    }

    function clone() {
      const c = makeRNG(0);
      c._setState(s0, s1, s2, s3);
      return c;
    }

    function _setState(a, b, c, d) {
      s0 = a >>> 0; s1 = b >>> 0; s2 = c >>> 0; s3 = d >>> 0;
    }

    return {
      nextU32,
      next,
      clone,
      _setState,
      get state() { return [s0, s1, s2, s3]; }
    };
  }

  // ---------------------------------------------------------------------
  // Layer 2 — Sample moments
  //
  // Convention: sampleVariance uses the unbiased (n-1) denominator —
  // Bessel's correction. All central moments are computed in deviation
  // form (subtract the mean first, then accumulate powers of the
  // deviation) rather than via E[x^2] - E[x]^2, which catastrophically
  // cancels for samples whose mean is far from zero relative to their
  // spread. Same rule as the beta module.
  // ---------------------------------------------------------------------
  function sampleMean(xs) {
    let sum = 0;
    for (let i = 0; i < xs.length; i++) sum += xs[i];
    return sum / xs.length;
  }

  function centralMoment(xs, mean, power) {
    let sum = 0;
    for (let i = 0; i < xs.length; i++) {
      sum += Math.pow(xs[i] - mean, power);
    }
    return sum / xs.length;
  }

  function sampleVariance(xs) {
    const n = xs.length;
    if (n < 2) throw new Error('sampleVariance requires n >= 2');
    const mean = sampleMean(xs);
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const d = xs[i] - mean;
      sumSq += d * d;
    }
    return sumSq / (n - 1);
  }

  function sampleStd(xs) {
    return Math.sqrt(sampleVariance(xs));
  }

  function sampleSkewness(xs) {
    // Population-moment skewness g1 = m3 / m2^1.5 — the biased estimator,
    // NOT the small-sample-corrected Fisher-Pearson G1. Deliberate: this
    // module's job is characterizing 1e6-path Monte Carlo output, where the
    // O(1/n) bias is ~1e-6 and irrelevant. If ever called on a small set
    // (e.g. a 20-branch scenario engine output), prefer a bias-corrected
    // estimator upstream — at n=20 the correction is material.
    const mean = sampleMean(xs);
    const m2 = centralMoment(xs, mean, 2);
    const m3 = centralMoment(xs, mean, 3);
    return m3 / Math.pow(m2, 1.5);
  }

  function sampleKurtosis(xs) {
    // Excess kurtosis: m4/m2^2 - 3, so a standard normal reads ~0. Same
    // biased population-moment convention and same small-n caveat as
    // sampleSkewness above.
    const mean = sampleMean(xs);
    const m2 = centralMoment(xs, mean, 2);
    const m4 = centralMoment(xs, mean, 4);
    return m4 / (m2 * m2) - 3;
  }

  // ---------------------------------------------------------------------
  // Layer 3 — Box-Muller
  //
  // Mechanism: the standard 2-D Gaussian density exp(-(x^2+y^2)/2)/(2*pi)
  // is rotationally symmetric, so it factors cleanly in polar coordinates
  // into an independent radius and angle. Changing variables from
  // Cartesian (x, y) to polar (R, Theta) introduces a Jacobian of R; that
  // extra factor of R turns exp(-r^2/2) into exactly the density of
  // R^2 ~ Exponential(1/2) — i.e. R = sqrt(-2 ln U) for U ~ Uniform(0,1) —
  // while Theta comes out Uniform(0, 2*pi) with no r-dependence at all.
  // Draw (R, Theta) from those two easy 1-D distributions, convert back to
  // Cartesian, and the result is jointly N(0, I_2). Two mechanisms below
  // sample the same distribution two different ways.
  // ---------------------------------------------------------------------
  function boxMullerTrig(u1, u2) {
    // ln(0) = -Infinity, so a uniform draw of exactly 0 must never reach
    // Math.log. Nudge into (0, 1] rather than reject-and-redraw, since the
    // caller already owns the two uniforms here.
    if (u1 === 0) u1 = Number.MIN_VALUE;
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    return [r * Math.cos(theta), r * Math.sin(theta)];
  }

  function boxMullerPolar(rng) {
    // Marsaglia polar method: sample (u, v) uniformly on the unit disk by
    // rejection from the enclosing square, rather than computing an angle
    // via sin/cos directly. s = u^2 + v^2 is then Uniform(0,1) on the
    // disk's radius-squared, so it substitutes for the "u1" of the trig
    // form without ever calling a trig function — at the cost of
    // discarding the ~1 - pi/4 ~ 21.5% of draws that fall outside the
    // disk.
    let u, v, s;
    do {
      u = 2 * rng.next() - 1;
      v = 2 * rng.next() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const factor = Math.sqrt((-2 * Math.log(s)) / s);
    return [u * factor, v * factor];
  }

  function makeNormal(rng, options) {
    const method = (options && options.method) || 'trig';
    if (method !== 'trig' && method !== 'polar') {
      throw new Error('makeNormal: method must be "trig" or "polar", got ' + method);
    }
    let spare = null;

    function next() {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      let z0, z1;
      if (method === 'trig') {
        const u1 = rng.next();
        const u2 = rng.next();
        [z0, z1] = boxMullerTrig(u1, u2);
      } else {
        [z0, z1] = boxMullerPolar(rng);
      }
      // Box-Muller always produces a pair; buffering z1 rather than
      // discarding it means every uniform draw pulled from rng
      // contributes to the normal stream — no wasted variates.
      spare = z1;
      return z0;
    }

    return { next };
  }

  // ---------------------------------------------------------------------
  // Layer 4 — Scaled draws
  // ---------------------------------------------------------------------
  function normalSamples(rng, n, options) {
    const mean = (options && options.mean !== undefined) ? options.mean : 0;
    const sd = (options && options.sd !== undefined) ? options.sd : 1;
    const method = (options && options.method) || 'trig';
    const normal = makeNormal(rng, { method });
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      // Location-scale transform: X ~ N(mean, sd^2) from a standard
      // normal Z via X = mean + sd*Z.
      out[i] = mean + sd * normal.next();
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // Layer 6b (optional) — Inverse-CDF / probit generator
  //
  // Included because the Gaussian copula needs it: the inverse-CDF
  // transform Phi^-1(U) is monotonic (strictly increasing) in U, so it
  // preserves rank order between the uniform draw and the resulting
  // normal. That rank-preservation is exactly what the copula's
  // dependence structure relies on — correlated uniforms stay correlated
  // in the same rank order after the transform. Box-Muller has no such
  // property: which of the two uniforms maps to which output normal is
  // an artifact of the polar-to-Cartesian conversion, not a rank-ordered
  // function of a single uniform. Fine for plain iid sampling; wrong
  // if the draw is meant to carry a specific rank into a copula.
  //
  // Acklam's rational approximation of Phi^-1 gives ~1.15e-9 relative
  // error on its own; a single Halley step (below) polishes it to machine
  // precision. Acklam is the seed, not the final answer.
  // ---------------------------------------------------------------------
  const ACKLAM_A = [
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
  ];
  const ACKLAM_B = [
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01
  ];
  const ACKLAM_C = [
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
  ];
  const ACKLAM_D = [
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00
  ];
  const ACKLAM_P_LOW = 0.02425;
  const ACKLAM_P_HIGH = 1 - ACKLAM_P_LOW;

  function acklam(p) {
    let q, r;
    if (p < ACKLAM_P_LOW) {
      // Lower tail: work in terms of q = sqrt(-2 ln p), a substitution
      // that linearizes the far-tail behaviour of Phi^-1 for the rational
      // approximation.
      q = Math.sqrt(-2 * Math.log(p));
      return (((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q + ACKLAM_C[3]) * q + ACKLAM_C[4]) * q + ACKLAM_C[5]) /
             ((((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q + ACKLAM_D[3]) * q + 1);
    } else if (p <= ACKLAM_P_HIGH) {
      // Central region: rational approximation directly in q = p - 0.5.
      q = p - 0.5;
      r = q * q;
      return (((((ACKLAM_A[0] * r + ACKLAM_A[1]) * r + ACKLAM_A[2]) * r + ACKLAM_A[3]) * r + ACKLAM_A[4]) * r + ACKLAM_A[5]) * q /
             (((((ACKLAM_B[0] * r + ACKLAM_B[1]) * r + ACKLAM_B[2]) * r + ACKLAM_B[3]) * r + ACKLAM_B[4]) * r + 1);
    } else {
      // Upper tail: mirror image of the lower tail via 1 - p, negated.
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q + ACKLAM_C[3]) * q + ACKLAM_C[4]) * q + ACKLAM_C[5]) /
              ((((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q + ACKLAM_D[3]) * q + 1);
    }
  }

  // ---------------------------------------------------------------------
  // Cody's rational Chebyshev approximation of erf/erfc (double precision,
  // agrees with the system libm to ~1e-16 relative — externally validated
  // against C's erfc across a grid spanning the tails). Needed for the
  // Halley refinement of invNormCDF, which requires Phi to full precision.
  // Three regions: |x|<=0.46875 (erf, A/B); 0.46875<|x|<=4 (erfc, C/D);
  // |x|>4 (erfc asymptotic, P/Q). The floor(y*16)/16 split evaluates
  // exp(-x^2) as exp(-ysq^2)*exp(-del) so the dominant exponent is exact.
  // ---------------------------------------------------------------------
  const CODY_A = [3.16112374387056560e0, 1.13864154151050156e2, 3.77485237685302021e2, 3.20937758913846947e3, 1.85777706184603153e-1];
  const CODY_B = [2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3, 2.84423683343917062e3];
  const CODY_C = [5.64188496988670089e-1, 8.88314979438837594e0, 6.61191906371416295e1, 2.98635138197400131e2, 8.81952221241769090e2, 1.71204761263407058e3, 2.05107837782607147e3, 1.23033935479799725e3, 2.15311535474403846e-8];
  const CODY_D = [1.57449261107098347e1, 1.17693950891312499e2, 5.37181101862009858e2, 1.62138957456669019e3, 3.29079923573345963e3, 4.36261909014324716e3, 3.43936767414372164e3, 1.23033935480374942e3];
  const CODY_P = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2];
  const CODY_Q = [2.56852019228982242e0, 1.87295284992346047e0, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3];
  const CODY_SQRPI = 5.6418958354775628695e-1; // 1/sqrt(pi)
  const CODY_THRESH = 0.46875;

  function calerf(x, jint) {
    // jint = 0 -> erf, 1 -> erfc.
    const y = Math.abs(x);
    let result;
    if (y <= CODY_THRESH) {
      let ysq = 0;
      if (y > 1.11e-16) ysq = y * y;
      let xnum = CODY_A[4] * ysq;
      let xden = ysq;
      for (let i = 0; i < 3; i++) {
        xnum = (xnum + CODY_A[i]) * ysq;
        xden = (xden + CODY_B[i]) * ysq;
      }
      result = x * (xnum + CODY_A[3]) / (xden + CODY_B[3]);
      if (jint !== 0) result = 1 - result;
      return result;
    } else if (y <= 4.0) {
      let xnum = CODY_C[8] * y;
      let xden = y;
      for (let i = 0; i < 7; i++) {
        xnum = (xnum + CODY_C[i]) * y;
        xden = (xden + CODY_D[i]) * y;
      }
      result = (xnum + CODY_C[7]) / (xden + CODY_D[7]);
      const ysq = Math.floor(y * 16) / 16;
      const del = (y - ysq) * (y + ysq);
      result = Math.exp(-ysq * ysq) * Math.exp(-del) * result;
    } else {
      result = 0;
      if (y < 26.543) {
        const ysqinv = 1 / (y * y);
        let xnum = CODY_P[5] * ysqinv;
        let xden = ysqinv;
        for (let i = 0; i < 4; i++) {
          xnum = (xnum + CODY_P[i]) * ysqinv;
          xden = (xden + CODY_Q[i]) * ysqinv;
        }
        result = ysqinv * (xnum + CODY_P[4]) / (xden + CODY_Q[4]);
        result = (CODY_SQRPI - result) / y;
        const ysq = Math.floor(y * 16) / 16;
        const del = (y - ysq) * (y + ysq);
        result = Math.exp(-ysq * ysq) * Math.exp(-del) * result;
      }
    }
    // Common tail assembly for the two outer regions.
    if (jint === 0) {
      result = 1 - result;
      if (x < 0) result = -result;
    } else {
      if (x < 0) result = 2 - result;
    }
    return result;
  }

  function erf(x) { return calerf(x, 0); }
  function erfc(x) { return calerf(x, 1); }

  const INV_SQRT2PI_DENOM = Math.sqrt(2 * Math.PI);

  function invNormCDF(p) {
    if (!(p > 0 && p < 1)) {
      throw new Error('invNormCDF: p must be in (0, 1), got ' + p);
    }
    // Seed with Acklam (~1e-9), then take one Halley step. Halley converges
    // cubically, so a single iteration lifts the seed to machine precision.
    // The step for solving Phi(x) = p, using phi (the density) and its
    // derivative -x*phi:
    //   e = Phi(x) - p
    //   u = e / phi(x)           = e * sqrt(2pi) * exp(x^2/2)
    //   x <- x - u / (1 + x*u/2)
    let x = acklam(p);
    // Residual computed piecewise to avoid subtractive cancellation: for
    // p > 0.5 both Phi(x) and p sit near 1, so form the identical quantity
    // as (1 - p) - Phi(-x) where both terms are small. Phi(-x) = erfc(x/sqrt2)/2.
    // (Note: for p within ~1e-8 of 1, the input double has already lost the
    // tail information carried by 1-p; accuracy there is input-representation
    // limited, not algorithmic. Deep upper-tail work should pass the tail
    // probability directly via the symmetry invNormCDF(p) = -invNormCDF(1-p).)
    let e;
    if (p <= 0.5) {
      e = 0.5 * erfc(-x / Math.SQRT2) - p;
    } else {
      e = (1 - p) - 0.5 * erfc(x / Math.SQRT2);
    }
    const u = e * INV_SQRT2PI_DENOM * Math.exp(x * x / 2);
    x = x - u / (1 + x * u / 2);
    return x;
  }

  function probitNormal(rng) {
    let u = rng.next();
    // invNormCDF requires p in the open interval (0, 1); a uniform draw
    // of exactly 0 must be nudged rather than passed through.
    if (u === 0) u = Number.MIN_VALUE;
    return invNormCDF(u);
  }

  // ---------------------------------------------------------------------
  // Layer 6c (optional) — Correlated normals via Cholesky
  //
  // The other half of the copula engine. If Sigma = L * L^T (L lower
  // triangular) and z is a vector of iid standard normals, then L*z has
  // covariance L * Cov(z) * L^T = L * I * L^T = Sigma. Multiplying by the
  // Cholesky factor is exactly the linear map that imposes a target
  // covariance structure onto independent draws.
  // ---------------------------------------------------------------------
  function cholesky(sigma) {
    const n = sigma.length;
    const L = [];
    for (let i = 0; i < n; i++) L.push(new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = sigma[i][j];
        for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
        if (i === j) {
          // Non-positive diagonal => Sigma is not positive definite (up to
          // floating point). Reject loudly rather than emit NaN from
          // sqrt(negative) or silently jitter the matrix. This is a
          // deliberate legible-failure stance, same as the IRR-multiplicity
          // flag: a non-PSD correlation matrix almost always means a
          // mis-specified input upstream (e.g. two effectively double-counted
          // drivers), and that belongs fixed at the source, not papered over
          // with an epsilon nudge that hides the modelling error.
          if (sum <= 0) {
            throw new Error('cholesky: matrix is not positive definite (failed at diagonal index ' + i + ')');
          }
          L[i][j] = Math.sqrt(sum);
        } else {
          L[i][j] = sum / L[j][j];
        }
      }
    }
    return L;
  }

  function correlatedNormals(rng, n, sigma) {
    const d = sigma.length;
    const L = cholesky(sigma);
    const normal = makeNormal(rng);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const z = new Array(d);
      for (let k = 0; k < d; k++) z[k] = normal.next();
      const x = new Array(d).fill(0);
      for (let r = 0; r < d; r++) {
        let s = 0;
        for (let c = 0; c <= r; c++) s += L[r][c] * z[c];
        x[r] = s;
      }
      out[i] = x;
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // Layer 5 — Self-test
  // ---------------------------------------------------------------------
  function selfTest() {
    try {
      const rng = makeRNG(42);
      const rng2 = makeRNG(42);
      // Reproducibility
      for (let i = 0; i < 100; i++) {
        if (rng.nextU32() !== rng2.nextU32()) return false;
      }
      // Range + rough moments
      const rng3 = makeRNG(7);
      const us = new Array(10000);
      for (let i = 0; i < us.length; i++) {
        const u = rng3.next();
        if (!(u >= 0 && u < 1)) return false;
        us[i] = u;
      }
      const uMean = sampleMean(us);
      if (Math.abs(uMean - 0.5) > 0.05) return false;

      // Normal moments
      const rng4 = makeRNG(123);
      const zs = normalSamples(rng4, 20000);
      if (Math.abs(sampleMean(zs)) > 0.1) return false;
      if (Math.abs(sampleStd(zs) - 1) > 0.1) return false;

      // invNormCDF round trip
      if (Math.abs(invNormCDF(0.5)) > 1e-9) return false;

      // Cholesky reconstruction
      const sigma = [[1, 0.5], [0.5, 1]];
      const L = cholesky(sigma);
      const recon = [
        [L[0][0] * L[0][0] + 0 * 0, L[0][0] * L[1][0] + L[0][1] * L[1][1]],
        [L[1][0] * L[0][0] + L[1][1] * 0, L[1][0] * L[1][0] + L[1][1] * L[1][1]]
      ];
      if (Math.abs(recon[0][0] - sigma[0][0]) > 1e-9) return false;

      return true;
    } catch (e) {
      return false;
    }
  }

  // ---------------------------------------------------------------
  // Invariant battery — the canonical check list + golden-stream lock +
  // external C-reference cross-checks, shipped WITH the engine so every
  // consumer (Node runner, in-page bench) asserts the same guarantees
  // against the same source — chol's pattern; nothing to drift.
  //
  // Phase 2 normalization note: moved here from the page script of
  // aestimo-random.html (runBattery), where the list lived at migration.
  // Every seed, reference value, golden stream, and tolerance is unchanged;
  // the checks were restructured into self-contained closures (identical
  // seeds ⇒ identical asserted values) so each can pass/fail independently.
  // ---------------------------------------------------------------
  function battery() {
    const approx = (a, b, t) => Math.abs(a - b) <= t;
    const CHECKS = [];
    const add = (name, fn) => CHECKS.push([name, fn]);

    add("identical seeds → identical stream", () => {
      const a = makeRNG(42), b = makeRNG(42);
      for (let i = 0; i < 5000; i++) if (a.nextU32() !== b.nextU32()) return false;
      return true;
    });
    add("clone() continues parent stream", () => {
      const par = makeRNG(999);
      for (let c = 0; c < 137; c++) par.nextU32();
      const ch = par.clone();
      for (let d = 0; d < 1000; d++) if (par.nextU32() !== ch.nextU32()) return false;
      return true;
    });
    add("golden stream lock · seed 42", () => {
      const gold = [660444221, 3652823732, 77672526, 910233633, 2297337756, 3786072677, 3123505064, 1891482476, 2460634111, 3466307039];
      const g = makeRNG(42);
      for (let e = 0; e < gold.length; e++) if (g.nextU32() !== gold[e]) return false;
      return true;
    });
    add("splitmix32 matches C reference", () => {
      const sm = splitmix32(42), smRef = [551831576, 144025891, 322543647, 3034809370];
      for (let s = 0; s < 4; s++) if (sm() !== smRef[s]) return false;
      return true;
    });
    add("xoshiro128** core matches C reference", () => {
      const core = makeRNG(0); core._setState(1, 2, 3, 4);
      const coreRef = [11520, 0, 5927040, 70819200, 2031721883, 1637235492, 1287239034, 3734860849, 3729100597, 4258142804];
      for (let q = 0; q < coreRef.length; q++) if (core.nextU32() !== coreRef[q]) return false;
      return true;
    });
    add("every next() in [0,1)", () => {
      const ur = makeRNG(7);
      for (let u = 0; u < 200000; u++) { const v = ur.next(); if (!(v >= 0 && v < 1)) return false; }
      return true;
    });
    add("uniform mean ≈ 0.5", () => {
      const ur = makeRNG(7), us = new Array(200000);
      for (let u = 0; u < us.length; u++) us[u] = ur.next();
      return approx(sampleMean(us), 0.5, 0.01);
    });
    ["trig", "polar"].forEach((m) => {
      add("normal(" + m + ") mean ≈ 0", () => {
        const zs = normalSamples(makeRNG(2026), 200000, { method: m });
        return approx(sampleMean(zs), 0, 0.02);
      });
      add("normal(" + m + ") sd ≈ 1", () => {
        const zs = normalSamples(makeRNG(2026), 200000, { method: m });
        return approx(sampleStd(zs), 1, 0.02);
      });
      add("normal(" + m + ") excess kurtosis ≈ 0", () => {
        const zs = normalSamples(makeRNG(2026), 200000, { method: m });
        return approx(sampleKurtosis(zs), 0, 0.08);
      });
    });
    add("erfc matches libm ground truth", () => {
      const grid = [[-1, 1.842700792949715], [1.7, 0.016209541409225436], [3, 2.2090496998585438e-05], [4.5, 1.9661604415428873e-10]];
      return grid.every((row) => Math.abs((erfc(row[0]) - row[1]) / row[1]) <= 1e-13);
    });
    add("invNormCDF core region at machine precision", () => {
      const grid = [[-2, 0.022750131948179219], [-1, 0.15865525393145707], [1, 0.84134474606854293], [3, 0.9986501019683699]];
      return grid.every((row) => Math.abs(invNormCDF(row[1]) - row[0]) <= 1e-12);
    });
    add("invNormCDF(0.5) = 0", () => approx(invNormCDF(0.5), 0, 1e-15));
    add("cholesky rejects non-positive-definite Σ", () => {
      try { cholesky([[1, 2], [2, 1]]); } catch (er) { return true; }
      return false;
    });
    add("correlatedNormals correlation ≈ target", () => {
      const cd = correlatedNormals(makeRNG(4242), 200000, [[1, 0.6], [0.6, 2]]);
      let m0 = 0, m1 = 0;
      cd.forEach((p) => { m0 += p[0]; m1 += p[1]; }); m0 /= cd.length; m1 /= cd.length;
      let s01 = 0, s00 = 0, s11 = 0;
      cd.forEach((p) => { s01 += (p[0] - m0) * (p[1] - m1); s00 += (p[0] - m0) * (p[0] - m0); s11 += (p[1] - m1) * (p[1] - m1); });
      return approx(s01 / Math.sqrt(s00 * s11), 0.6 / Math.sqrt(2), 0.015);
    });
    add("selfTest() returns true", () => selfTest() === true);

    return CHECKS.map(([name, fn]) => {
      let pass = false, detail = '';
      try { pass = fn(); } catch (e) { detail = (e && e.message) ? e.message : String(e); }
      return { name, pass, detail };
    });
  }

  return {
    splitmix32,
    makeRNG,
    sampleMean,
    sampleVariance,
    sampleStd,
    sampleSkewness,
    sampleKurtosis,
    boxMullerTrig,
    boxMullerPolar,
    makeNormal,
    normalSamples,
    erf,
    erfc,
    invNormCDF,
    probitNormal,
    cholesky,
    correlatedNormals,
    selfTest,
    battery
  };
}));

