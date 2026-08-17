/*
 * aestimo-kelly.js — log-optimal position sizing.
 * ===========================================================================
 * The capstone of the Aestimo suite. It maximizes E[log(1 + f·X)] — equivalently
 * the long-run geometric growth rate of wealth — and it ties the suite together
 * by consuming the primitives already built rather than reinventing them:
 *   - single-asset Kelly is a 1-D root-find  → solve      (aestimo-solve.js)
 *   - the multi-asset vector is an SPD solve  → solveSPD   (aestimo-chol.js)
 *
 * WHY LOG (this is the whole foundation, derived once):
 * Wealth compounds MULTIPLICATIVELY. Bet fraction f of capital each period on a
 * return random variable X; after n periods
 *      W_n = W_0 · Π (1 + f·Xᵢ)   ⇒   log W_n = log W_0 + Σ log(1 + f·Xᵢ).
 * Divide by n: the per-period log-growth is the SAMPLE MEAN of log(1 + f·Xᵢ).
 * By the law of large numbers that mean → E[log(1 + f·X)] almost surely. So the
 * long-run growth RATE of wealth is E[log(1 + f·X)], and maximizing it is not a
 * utility axiom — it is the a.s.-optimal choice for the compounded outcome. This
 * is *why* the log appears; CRRA/utility framings are downstream of it.
 *
 * The objective g(f) = E[log(1 + f·X)] is STRICTLY CONCAVE in f (its second
 * derivative is −E[X²/(1+fX)²] < 0). A strictly concave function has a monotone-
 * decreasing derivative, hence a UNIQUE stationary point = the maximizer. And
 *      g'(0) = E[X]  (the "edge").
 * You bet (f* > 0) iff the mean return is positive; a non-positive edge means the
 * growth-optimal fraction is ≤ 0 and we clamp to 0 rather than dress a short as a
 * "Kelly bet."
 *
 * CONVENTION. A return random variable is given either as discrete outcomes
 *   outcomes = [{ p, b }, ...]   (b = net return per unit staked; b<0 is a loss;
 *                                 the p's should sum to 1)
 * or as a sample array of realized per-period returns
 *   returns  = [r1, r2, ...].
 * For the multi-asset layer, `mu` is the excess-return vector and `Sigma` the SPD
 * covariance (row-major arrays of arrays; vectors are flat arrays), `rf` the
 * risk-free rate. Fractions f are expressed per unit of current wealth.
 * ===========================================================================
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./aestimo-solve.js'), require('./aestimo-chol.js'));
  } else {
    root.AestimoKelly = factory(root.AestimoSolve, root.AestimoChol);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (Solve, Chol) {
  'use strict';

  var solve = Solve.solve;
  var solveSPD = Chol.solveSPD;

  // Root-finds are driven to ~machine precision (well past the 1e-10 the solver
  // defaults to) so the located fraction is stable to <1e-12 across any correct
  // safeguarded root-finder — the golden locks below depend on that stability.
  var SOLVE_OPTS = { tol: 1e-14, maxIter: 200 };

  // ===========================================================================
  // Layer 0 — Growth-rate primitives (the objective everything maximizes)
  // ===========================================================================

  // g(f) = Σ pᵢ · log(1 + f·bᵢ). The expected log-growth per period.
  function logGrowth(f, outcomes) {
    var s = 0;
    for (var i = 0; i < outcomes.length; i++) {
      var o = outcomes[i];
      s += o.p * Math.log(1 + f * o.b);
    }
    return s;
  }

  // Sample analog: (1/n) Σ log(1 + f·rᵢ). The empirical growth rate at fraction f.
  function logGrowthSample(f, returns) {
    var s = 0, n = returns.length;
    for (var i = 0; i < n; i++) s += Math.log(1 + f * returns[i]);
    return s / n;
  }

  // g'(f) = Σ pᵢ·bᵢ / (1 + f·bᵢ). Concave objective ⇒ this is monotone decreasing
  // ⇒ its unique root is the maximizer. Note g'(0) = Σ pᵢbᵢ = E[X], the edge.
  function logGrowthDeriv(f, outcomes) {
    var s = 0;
    for (var i = 0; i < outcomes.length; i++) {
      var o = outcomes[i];
      s += o.p * o.b / (1 + f * o.b);
    }
    return s;
  }
  function logGrowthSampleDeriv(f, returns) {
    var s = 0, n = returns.length;
    for (var i = 0; i < n; i++) s += returns[i] / (1 + f * returns[i]);
    return s / n;
  }

  // g''(f) = −Σ pᵢ·bᵢ²/(1+f·bᵢ)² < 0 (strict concavity, made explicit). Supplied
  // to `solve` as the analytic derivative of the objective's derivative so the
  // safeguarded hybrid can take Newton steps inside its bracket.
  function logGrowthDeriv2(f, outcomes) {
    var s = 0;
    for (var i = 0; i < outcomes.length; i++) {
      var o = outcomes[i], d = 1 + f * o.b;
      s -= o.p * o.b * o.b / (d * d);
    }
    return s;
  }
  function logGrowthSampleDeriv2(f, returns) {
    var s = 0, n = returns.length;
    for (var i = 0; i < n; i++) { var d = 1 + f * returns[i]; s -= returns[i] * returns[i] / (d * d); }
    return s / n;
  }

  // Realized ex-post growth rate of a fully-invested unit: mean(log(1 + rᵢ)).
  // The empirical counterpart of the objective — what you MEASURE after the fact.
  function geomGrowthRate(returns) {
    var s = 0, n = returns.length;
    for (var i = 0; i < n; i++) s += Math.log(1 + returns[i]);
    return s / n;
  }

  // Largest f keeping 1 + f·bᵢ > 0 for EVERY outcome. A loss bᵢ<0 requires
  // f < 1/(−bᵢ); the binding one is the biggest loss. Beyond this bound the log
  // diverges to −∞ and wealth goes non-positive — it is the solver's hard ceiling.
  // ∞ when there is no losing outcome (a bet with no downside has no interior max).
  function boundFromBs(bs) {
    var worst = 0; // = max(−b) over losing b
    for (var i = 0; i < bs.length; i++) if (bs[i] < 0 && -bs[i] > worst) worst = -bs[i];
    return worst > 0 ? 1 / worst : Infinity;
  }
  function noBankruptcyBound(outcomes) {
    var bs = new Array(outcomes.length);
    for (var i = 0; i < outcomes.length; i++) bs[i] = outcomes[i].b;
    return boundFromBs(bs);
  }

  // Internal: root-find g'(f)=0 on [0, bound) given evaluators. Returns the
  // structured Kelly result with the negative-edge clamp and the unbounded flag.
  function optimize(edge, bound, deriv, deriv2, growthAt, opts) {
    // Non-positive edge: the unconstrained optimum is f ≤ 0. Clamp to 0, flag it.
    if (edge <= 0) return { f: 0, growth: 0, clampedToZero: true };
    // No downside: g'(f) > 0 for all f, no finite maximizer. Flag rather than loop.
    if (!Number.isFinite(bound)) return { f: Infinity, growth: Infinity, clampedToZero: false, unbounded: true };
    // Bracket [0, bound·(1−ε)]: g'(0)=edge>0, and as f→bound⁻ the biggest-loss term
    // → −∞, so the upper end is strongly negative — a guaranteed sign change.
    var hi = bound * (1 - 1e-9);
    // Caller opts (tol/maxIter) override the tight defaults; goldens use the defaults.
    var so = opts ? Object.assign({}, SOLVE_OPTS, opts) : SOLVE_OPTS;
    var res = solve(deriv, { bracket: [0, hi], fprime: deriv2 }, so);
    var f = res.root;
    return { f: f, growth: growthAt(f), clampedToZero: false };
  }

  // ===========================================================================
  // Layer 1 — Discrete Kelly (the "toy" done rigorously)
  // ===========================================================================

  // INPUT VALIDATION (hardened 2026-08-01, external review). Finance inputs
  // arrive from UIs and imports, not proofs: probabilities outside [0,1],
  // zero odds, non-finite samples. The engines reject these LEGIBLY —
  // scalar functions return NaN, result-object functions return
  // { invalid: true, reason } with f: NaN — never a plausible wrong number.
  function badProb(p) { return !(typeof p === 'number' && isFinite(p) && p >= 0 && p <= 1); }

  // Closed form for a simple bet: win net b per unit with prob p, else lose the
  // stake. f* = p − (1−p)/b = (b·p − q)/b, "edge over odds." Canonical: p=.6,b=1 → .2.
  // Requires 0 < p < 1 and b > 0; anything else returns NaN (a degenerate
  // "bet" with certain outcome or non-positive odds has no Kelly fraction).
  function kellyBinary(bet) {
    var p = bet.p, b = bet.b, q = 1 - p;
    if (badProb(p) || p === 0 || p === 1 || !(typeof b === 'number' && isFinite(b) && b > 0)) return NaN;
    return (b * p - q) / b;
  }

  // General discrete case: maximize logGrowth by root-finding logGrowthDeriv = 0
  // on [0, noBankruptcyBound) via the safeguarded solver (the exact use case it
  // was built for). The binary closed form is precisely this on two outcomes.
  function kellyDiscrete(outcomes, opts) {
    // Reject malformed distributions: empty, non-finite entries, probabilities
    // outside [0,1], or probabilities that do not sum to 1 (within 1e-9 — a
    // distribution that leaks mass silently rescales every downstream number).
    if (!Array.isArray(outcomes) || outcomes.length === 0) {
      return { f: NaN, growth: NaN, invalid: true, reason: 'outcomes must be a non-empty array' };
    }
    var psum = 0;
    for (var vi = 0; vi < outcomes.length; vi++) {
      var o = outcomes[vi];
      if (!o || badProb(o.p) || !(typeof o.b === 'number' && isFinite(o.b))) {
        return { f: NaN, growth: NaN, invalid: true, reason: 'outcome ' + vi + ': need p in [0,1] and finite b' };
      }
      psum += o.p;
    }
    if (Math.abs(psum - 1) > 1e-9) {
      return { f: NaN, growth: NaN, invalid: true, reason: 'probabilities sum to ' + psum + ', not 1' };
    }
    var edge = logGrowthDeriv(0, outcomes); // = E[X]
    var bound = noBankruptcyBound(outcomes);
    return optimize(
      edge, bound,
      function (f) { return logGrowthDeriv(f, outcomes); },
      function (f) { return logGrowthDeriv2(f, outcomes); },
      function (f) { return logGrowth(f, outcomes); },
      opts
    );
  }

  // ===========================================================================
  // Layer 2 — Continuous / empirical Kelly (the general single-asset case)
  // ===========================================================================

  // EXACT log-optimal fraction on the empirical distribution: root-find
  // (1/n)Σ rᵢ/(1+f·rᵢ) = 0. Nonparametric, rank-preserving, assumes no distribution.
  function kellySample(returns, opts) {
    // Reject empty or non-finite samples. Returns ≤ −1 stay LEGAL — a −100%
    // outcome is a real instrument (total loss), and losses beyond −100%
    // (levered/margin positions) are handled correctly by the bankruptcy
    // bound, which shrinks below 1 accordingly.
    if (!Array.isArray(returns) || returns.length === 0) {
      return { f: NaN, growth: NaN, invalid: true, reason: 'returns must be a non-empty array' };
    }
    for (var vi = 0; vi < returns.length; vi++) {
      if (!(typeof returns[vi] === 'number' && isFinite(returns[vi]))) {
        return { f: NaN, growth: NaN, invalid: true, reason: 'returns[' + vi + '] is not a finite number' };
      }
    }
    var edge = logGrowthSampleDeriv(0, returns); // = sample mean
    var bound = boundFromBs(returns);
    return optimize(
      edge, bound,
      function (f) { return logGrowthSampleDeriv(f, returns); },
      function (f) { return logGrowthSampleDeriv2(f, returns); },
      function (f) { return logGrowthSample(f, returns); },
      opts
    );
  }

  // The classic approximation f* = μ/σ², with growthApprox = ½(μ/σ)² = ½·Sharpe².
  // DERIVATION: Taylor log(1+fX) ≈ fX − ½f²X² about 1; take E[·] with E[X]=μ,
  // E[X²]≈σ² in the small-edge limit ⇒ g(f) ≈ μf − ½σ²f². Maximize: g'=μ−σ²f=0 ⇒
  // f*=μ/σ². Then g(f*) = μ²/σ² − ½μ²/σ² = ½(μ/σ)². The max growth rate is HALF the
  // squared Sharpe ratio — the bridge from Kelly to the Sharpe machinery elsewhere.
  //
  // STANCE — this overbets fat tails. It is a second-order expansion; it holds in
  // the small-edge limit and BREAKS exactly where a left tail lives, because the
  // true log objective penalizes large adverse moves heavily and the quadratic
  // does not. For a fat-tailed / negatively-skewed sample the EXACT kellySample
  // fraction is strictly smaller than μ/σ². Compute both; the gap is the warning.
  function kellyGaussian(params) {
    var mu = params.mu, sigma = params.sigma;
    // σ ≤ 0 is not a distribution; μ must be finite. Reject legibly rather
    // than return Infinity/NaN dressed as a position size.
    if (!(typeof mu === 'number' && isFinite(mu)) || !(typeof sigma === 'number' && isFinite(sigma) && sigma > 0)) {
      return { f: NaN, growthApprox: NaN, sharpe: NaN, invalid: true, reason: 'need finite mu and sigma > 0' };
    }
    var sharpe = mu / sigma;
    return { f: mu / (sigma * sigma), growthApprox: 0.5 * sharpe * sharpe, sharpe: sharpe };
  }

  // ===========================================================================
  // Layer 3 — Fractional Kelly & drawdown (the risk-management centerpiece)
  // ===========================================================================
  // Under the Gaussian approximation, betting fraction c of full Kelly (position
  // c·f*) has, per period:
  //   growth rate  m(c) = μf − ½σ²f²  with f=c·μ/σ²  =  ½·Sharpe²·(2c − c²)
  //   log-variance v(c) = f²σ²         with f=c·μ/σ²  =  c²·Sharpe²
  // So growth is a downward parabola in c (peak at c=1) while VARIANCE grows like
  // c² — you buy the last slivers of growth with quadratically more risk.

  // Fraction of full-Kelly growth retained at fraction c: g(c·f*)/g(f*) = 2c − c².
  // (½Sharpe²·(2c−c²) over ½Sharpe²·(2·1−1²).) 1 at c=1; 0.75 at c=½; 0 at c=2.
  function growthRatio(c) { return 2 * c - c * c; }

  // Probability wealth EVER falls to fraction α (0<α<1) of its current value while
  // betting fraction c of full Kelly:  drawdownProb = α^(2/c − 1).
  // DERIVATION: log-wealth is a Brownian motion with drift m(c) and variance v(c).
  // For such a process the probability of ever hitting a barrier a = log(1/α) below
  // the start is exp(−2m·a/v) = α^(2m/v). And
  //   2m/v = 2·[½Sharpe²(2c−c²)] / [c²Sharpe²] = (2c−c²)/c² = 2/c − 1.
  // At full Kelly (c=1): α^1 = α — a 50% chance of ever halving. At half (c=½):
  // α^3 = 0.125 for α=½. The Sharpe cancels: drawdown risk depends only on c.
  function drawdownProb(alpha, c) { return Math.pow(alpha, 2 / c - 1); }

  // Bundle: the actual fraction c·f_full plus the two headline read-outs. The
  // engine NEVER hands back a bare full-Kelly f as a "recommendation" — that is the
  // same silent-danger failure as an optimizer returning a monstrous levered vector
  // with no fragility flag. You get the trade, not just the number.
  function fractionalKelly(f_full, c) {
    return {
      f: c * f_full,
      c: c,
      growthRatio: growthRatio(c),
      drawdownProb: function (alpha) { return drawdownProb(alpha, c); }
    };
  }

  // ===========================================================================
  // Layer 4 — Multi-asset / simultaneous Kelly (the capstone connection)
  // ===========================================================================
  // Across correlated simultaneous bets the Gaussian-approx log-optimal vector is
  //      f* = Σ⁻¹(μ − rf·𝟙)
  // solved as Σ·f = (μ − rf·𝟙) via solveSPD — never by forming Σ⁻¹.
  //
  // STANCE — full-Kelly betting IS the tangency portfolio. Σ⁻¹(μ − rf·𝟙) is exactly
  // the (unnormalized) maximum-Sharpe direction of mean-variance analysis. Kelly and
  // Markowitz are one optimization in two languages — growth-rate maximization and
  // mean-variance efficiency coincide at this vector — with the gross sum Σfᵢ setting
  // leverage. For diagonal Σ (uncorrelated bets) it collapses to the independent
  // per-asset Kelly fractions f*ᵢ = (μᵢ − rf)/σᵢ².
  function kellyVector(mu, Sigma, rf) {
    rf = rf || 0;
    var n = mu.length, rhs = new Array(n);
    for (var i = 0; i < n; i++) rhs[i] = mu[i] - rf;
    var res = solveSPD(Sigma, rhs);
    // Propagate chol's structured failure faithfully: its field is `failedAt`
    // (the failing leading minor), not `index` — the old mapping always
    // yielded undefined and threw away the pivot information.
    if (!res.ok) return { ok: false, f: null, reason: res.reason, failedAt: res.failedAt };
    // Leverage semantics (fixed 2026-08-01, external review): GROSS leverage
    // is the sum of |exposures| — signed summing let long/short legs cancel,
    // understating the balance-sheet footprint. The signed sum is still the
    // economically meaningful NET exposure, so both are returned.
    var f = res.x, gross = 0, net = 0;
    for (i = 0; i < n; i++) { gross += Math.abs(f[i]); net += f[i]; }
    return { ok: true, f: f, grossLeverage: gross, netExposure: net };
  }

  // ===========================================================================
  // Layer 5 — Invariant battery + self-test (single source of truth)
  // ===========================================================================
  // The battery is defined ONCE here and consumed by both selfTest() and the Node
  // runner (aestimo-kelly.test.js) — no diverging copies, the discipline the
  // Cholesky consolidation established.

  function approx(a, b, tol) { return Math.abs(a - b) <= tol; }

  // Golden regression locks — captured from the first fully-green build, hardcoded.
  // Any future change that moves them is a breaking change requiring explicit
  // acknowledgement. See the header of the .test.js for the fixtures.
  var GOLD = {
    binary: 0.454782608695652,          // kellyBinary({p:0.62, b:2.3})
    sampleFatTail: 0.262507206188192,   // exact kellySample on FAT_TAIL — the μ/σ² divergence case
    growthRatioHalf: 0.75,              // growthRatio(0.5)
    drawdownHalfHalf: 0.125,            // drawdownProb(0.5, 0.5)
    vec: [1.354545454545454, 0.450000000000000, 0.263636363636364] // kellyVector on {MU3, SIG3, RF3}
  };

  // Fixed fixtures shared by the golden locks.
  var FAT_TAIL = [0.04, 0.05, 0.03, 0.06, 0.04, 0.05, 0.03, 0.05, 0.04, -0.35];
  var MU3 = [0.08, 0.06, 0.05];
  var SIG3 = [[0.0400, 0.0100, 0.0050], [0.0100, 0.0500, 0.0150], [0.0050, 0.0150, 0.0625]];
  var RF3 = 0.02;

  function battery() {
    var T = [], TOL = 1e-10;
    function check(name, pass, detail) { T.push({ name: name, pass: !!pass, detail: detail || '' }); }

    // ---- Layer 0 / discrete ----
    var oc = [{ p: 0.5, b: 2 }, { p: 0.5, b: -1 }];
    check('g\'(0) = E[X] (edge)', approx(logGrowthDeriv(0, oc), 0.5 * 2 + 0.5 * -1, TOL), 'edge=0.5');
    // Concavity: second difference of g over a sampled f-range ≤ 0.
    var conc = true, prev2 = null, prev1 = null;
    for (var f = 0.0; f <= 0.9; f += 0.05) {
      var gv = logGrowth(f, oc);
      if (prev2 !== null) { if ((gv - 2 * prev1 + prev2) > 1e-9) conc = false; }
      prev2 = prev1; prev1 = gv;
    }
    check('logGrowth concave (2nd diff ≤ 0)', conc);
    check('noBankruptcyBound = 1/max(−b)', approx(noBankruptcyBound(oc), 1.0, TOL), 'bound=1');
    check('bound = ∞ when no losing outcome', noBankruptcyBound([{ p: 1, b: 0.1 }]) === Infinity);

    // "= 0.2 exactly" holds mathematically, but 0.6 and 0.4 are not representable
    // in IEEE-754, so 0.6 − 0.4 = 0.19999999999999998. Assert to representation
    // tolerance (error ~3e-17) rather than ship an exact-equality landmine that
    // fails on a correct implementation — the same FP-honesty call as the par-bond
    // identity in aestimo-solve.
    check('kellyBinary({.6,1}) = 0.2 (to ULP)', approx(kellyBinary({ p: 0.6, b: 1 }), 0.2, 1e-15),
      'Δ=' + (kellyBinary({ p: 0.6, b: 1 }) - 0.2).toExponential(2));
    // General binary f* = p − q/b matches kellyDiscrete on the same two outcomes.
    var bp = 0.6, bb = 1.5;
    var closed = bp - (1 - bp) / bb;
    var viaGen = kellyDiscrete([{ p: bp, b: bb }, { p: 1 - bp, b: -1 }]).f;
    check('binary closed form = kellyDiscrete(2 outcomes)', approx(closed, viaGen, 1e-10), 'Δ=' + (closed - viaGen).toExponential(2));
    // Negative edge ⇒ clamp to 0 with flag.
    var neg = kellyDiscrete([{ p: 0.4, b: 1 }, { p: 0.6, b: -1 }]); // E[X] = -0.2
    check('negative edge ⇒ f=0, clampedToZero', neg.f === 0 && neg.clampedToZero === true);
    // No-bankruptcy honoured: 1 + f*·bᵢ > 0 for every outcome at the returned f*.
    var kd = kellyDiscrete(oc), okB = true;
    for (var i = 0; i < oc.length; i++) if (1 + kd.f * oc[i].b <= 0) okB = false;
    check('solver never crosses bankruptcy bound', okB, 'f*=' + kd.f.toFixed(6));

    // ---- Layer 2: continuous / approximation ----
    // Convergence: on a symmetric (zero-skew) sample, kellySample → μ/σ² as μ→0.
    var shape = [-2, -1, -1, 0, 0, 0, 1, 1, 2]; // symmetric, mean 0
    function meanVar(arr) {
      var m = 0, n = arr.length; for (var k = 0; k < n; k++) m += arr[k]; m /= n;
      var v = 0; for (k = 0; k < n; k++) v += (arr[k] - m) * (arr[k] - m); v /= n; return { m: m, v: v };
    }
    // Amplitude 0.1 keeps a genuine downside (a negative return) at every mu, so
    // kellySample stays finite and the convergence to μ/σ² is the real effect —
    // not an artifact of an all-positive sample (which would be unbounded Kelly).
    var relErr = [];
    [0.1, 0.01, 0.001].forEach(function (mu) {
      var r = shape.map(function (z) { return mu + 0.1 * z; });
      var mv = meanVar(r);
      var exact = kellySample(r).f;
      var approxF = mv.m / mv.v; // = μ/σ² on this sample
      relErr.push(Math.abs(exact - approxF) / Math.abs(approxF));
    });
    check('kellySample → μ/σ² as edge→0 (rel err ↓)', relErr[0] > relErr[1] && relErr[1] > relErr[2],
      'relErr=' + relErr.map(function (e) { return e.toExponential(1); }).join(', '));

    // Tail penalty: on the fat-tailed series, EXACT kellySample < Gaussian μ/σ².
    var mvF = meanVar(FAT_TAIL);
    var exactFat = kellySample(FAT_TAIL).f;
    var gaussFat = kellyGaussian({ mu: mvF.m, sigma: Math.sqrt(mvF.v) }).f;
    check('fat tail: kellySample < μ/σ² (overbet trap)', exactFat < gaussFat,
      'exact=' + exactFat.toFixed(4) + ' vs μ/σ²=' + gaussFat.toFixed(4));

    // growthApprox = ½·Sharpe².
    var kg = kellyGaussian({ mu: 0.08, sigma: 0.2 });
    check('kellyGaussian.growthApprox = ½·Sharpe²', approx(kg.growthApprox, 0.5 * Math.pow(0.08 / 0.2, 2), TOL));

    // ---- Layer 3: fractional / drawdown (headline) ----
    check('growthRatio(1) = 1', approx(growthRatio(1), 1, TOL));
    check('growthRatio(0.5) = 0.75', approx(growthRatio(0.5), 0.75, TOL));
    check('growthRatio(2) = 0', approx(growthRatio(2), 0, TOL));
    // Maximized at c=1.
    var maxAt1 = growthRatio(1) >= growthRatio(0.9) && growthRatio(1) >= growthRatio(1.1);
    check('growthRatio maximized at c=1', maxAt1);
    [0.3, 0.5, 0.8].forEach(function (a) {
      check('drawdownProb(α,1) = α  (α=' + a + ')', approx(drawdownProb(a, 1), a, TOL));
    });
    check('drawdownProb(0.5,0.5) = 0.125', approx(drawdownProb(0.5, 0.5), 0.125, TOL));
    // Vol scales linearly in c: recover per-period log-std from the TWO independent
    // Layer-3 formulas and confirm std(c)/c is constant. std = sqrt(2·m/exponent),
    // m = ½Sharpe²·growthRatio(c), exponent = 2/c−1 (from drawdownProb via any α).
    var sharpe = 0.5, a0 = 0.5, ratios = [];
    [0.25, 0.5, 0.75, 1.0].forEach(function (c) {
      var m = 0.5 * sharpe * sharpe * growthRatio(c);
      var expo = Math.log(drawdownProb(a0, c)) / Math.log(a0); // = 2/c − 1
      var v = 2 * m / expo;
      ratios.push(Math.sqrt(v) / c); // should be constant = sharpe
    });
    var volLinear = ratios.every(function (x) { return approx(x, sharpe, 1e-9); });
    check('fractional vol scales linearly in c', volLinear, 'std/c=' + ratios[0].toFixed(4));

    // ---- Layer 4: multi-asset ----
    // Diagonal Σ ⇒ per-asset independent Kelly (μᵢ−rf)/σᵢ².
    var diag = [[0.04, 0, 0], [0, 0.09, 0], [0, 0, 0.16]];
    var muD = [0.05, 0.06, 0.07], rfD = 0.01;
    var kv = kellyVector(muD, diag, rfD);
    var diagOK = kv.ok &&
      approx(kv.f[0], (muD[0] - rfD) / 0.04, 1e-10) &&
      approx(kv.f[1], (muD[1] - rfD) / 0.09, 1e-10) &&
      approx(kv.f[2], (muD[2] - rfD) / 0.16, 1e-10);
    check('diagonal Σ ⇒ per-asset Kelly (μ−rf)/σ²', diagOK);
    // Tangency: f solves Σf = μ−rf𝟙 (residual ≈ 0) ⇒ f = Σ⁻¹(μ−rf𝟙), the max-Sharpe
    // direction. (Cross-check to MVO `tangency` weights when that module lands.)
    var kvT = kellyVector(MU3, SIG3, RF3);
    var resid = Chol.matVec(SIG3, kvT.f), maxres = 0;
    for (i = 0; i < 3; i++) maxres = Math.max(maxres, Math.abs(resid[i] - (MU3[i] - RF3)));
    check('kellyVector solves Σf = μ−rf·𝟙 (tangency)', maxres < 1e-10, 'maxResid=' + maxres.toExponential(2));
    // Scaling excess returns by k scales the vector by k (linear solve, rf=0).
    var k = 3.0;
    var base = kellyVector([0.08, 0.06, 0.05], SIG3, 0);
    var scaled = kellyVector([0.24, 0.18, 0.15], SIG3, 0);
    var scaleOK = base.ok && scaled.ok &&
      approx(scaled.f[0], k * base.f[0], 1e-9) &&
      approx(scaled.f[1], k * base.f[1], 1e-9) &&
      approx(scaled.f[2], k * base.f[2], 1e-9);
    check('scaling excess returns by k scales f by k', scaleOK);
    // Legible failure on a non-SPD "covariance".
    var bad = kellyVector([0.1, 0.1], [[1, 2], [2, 1]], 0);
    check('non-SPD Σ ⇒ ok:false (no silent NaN)', bad.ok === false);
    // ---- input validation (hardened 2026-08-01) ----
    check('kellyBinary rejects p∉(0,1) and b≤0 with NaN',
      isNaN(kellyBinary({ p: 1.2, b: 1 })) && isNaN(kellyBinary({ p: 0.5, b: 0 })) &&
      isNaN(kellyBinary({ p: 0, b: 1 })) && isNaN(kellyBinary({ p: 0.5, b: -2 })));
    var badSum = kellyDiscrete([{ p: 0.5, b: 1 }, { p: 0.3, b: -1 }]); // Σp = 0.8
    check('kellyDiscrete rejects Σp ≠ 1 (invalid + reason, f = NaN)',
      badSum.invalid === true && isNaN(badSum.f) && /sum/.test(badSum.reason));
    var badRet = kellySample([0.02, NaN, 0.01]);
    check('kellySample rejects non-finite returns (invalid + reason)',
      badRet.invalid === true && isNaN(badRet.f));
    var badSig = kellyGaussian({ mu: 0.05, sigma: 0 });
    check('kellyGaussian rejects sigma ≤ 0 (invalid, no Infinity position)',
      badSig.invalid === true && isNaN(badSig.f));
    // Legal edge case pinned: a −100% outcome is a valid instrument; the
    // bankruptcy bound handles it (bound = 1) and the solver stays inside.
    var wipeout = kellySample([0.3, 0.3, 0.3, -1]);
    check('kellySample accepts a −100% return (bound handles it)',
      !wipeout.invalid && isFinite(wipeout.f) && wipeout.f >= 0 && wipeout.f < 1);

    // Gross vs net semantics: with one asset below rf (short leg), gross
    // leverage must sum |f| while net exposure sums f — a signed "gross"
    // would let the legs cancel. Diagonal Σ makes the legs exact:
    // f = [(0.05−0.02)/0.04, (0.01−0.02)/0.04] = [0.75, −0.25].
    var ls = kellyVector([0.05, 0.01], [[0.04, 0], [0, 0.04]], 0.02);
    check('grossLeverage = Σ|f|, netExposure = Σf (long/short)',
      ls.ok && approx(ls.grossLeverage, 1.0, 1e-12) && approx(ls.netExposure, 0.5, 1e-12),
      'gross=' + ls.grossLeverage.toFixed(4) + ' net=' + ls.netExposure.toFixed(4));

    // ---- Golden regression locks ----
    check('GOLD kellyBinary', approx(kellyBinary({ p: 0.62, b: 2.3 }), GOLD.binary, 1e-12),
      kellyBinary({ p: 0.62, b: 2.3 }).toFixed(15));
    check('GOLD kellySample (fat tail)', approx(kellySample(FAT_TAIL).f, GOLD.sampleFatTail, 1e-12),
      kellySample(FAT_TAIL).f.toFixed(15));
    check('GOLD growthRatio(0.5)', approx(growthRatio(0.5), GOLD.growthRatioHalf, 1e-12));
    check('GOLD drawdownProb(0.5,0.5)', approx(drawdownProb(0.5, 0.5), GOLD.drawdownHalfHalf, 1e-12));
    var gv3 = kellyVector(MU3, SIG3, RF3);
    check('GOLD kellyVector (3-asset)',
      approx(gv3.f[0], GOLD.vec[0], 1e-12) && approx(gv3.f[1], GOLD.vec[1], 1e-12) && approx(gv3.f[2], GOLD.vec[2], 1e-12),
      '[' + gv3.f.map(function (x) { return x.toFixed(12); }).join(', ') + ']');

    return T;
  }

  function selfTest() {
    var t = battery();
    for (var i = 0; i < t.length; i++) if (!t[i].pass) return false;
    return true;
  }

  return {
    // Layer 0
    logGrowth: logGrowth, logGrowthSample: logGrowthSample,
    logGrowthDeriv: logGrowthDeriv, logGrowthSampleDeriv: logGrowthSampleDeriv,
    geomGrowthRate: geomGrowthRate, noBankruptcyBound: noBankruptcyBound,
    // Layer 1
    kellyBinary: kellyBinary, kellyDiscrete: kellyDiscrete,
    // Layer 2
    kellySample: kellySample, kellyGaussian: kellyGaussian,
    // Layer 3
    growthRatio: growthRatio, drawdownProb: drawdownProb, fractionalKelly: fractionalKelly,
    // Layer 4
    kellyVector: kellyVector,
    // Layer 5
    battery: battery, selfTest: selfTest
  };
});
