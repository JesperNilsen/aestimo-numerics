'use strict';

/*
 * aestimo-solve.js — robust scalar root-finding and two finance applications.
 *
 * ONE numerical primitive: find x such that f(x) = 0, safely. Built up in
 * layers, each gated by the battery in aestimo-solve.test.js. Pure, offline,
 * deterministic, DOM-agnostic, zero dependencies. Slots into the Aestimo suite
 * beside aestimo-console.js / aestimo-beta.js and backs the console's existing
 * YTM / IRR / bond commands with one solver, not three ad-hoc ones.
 *
 * Every solver returns a STRUCTURED result:
 *     { root, iters, converged, residual }
 * so a failure to converge is legible — never a plausible-looking wrong number
 * dressed up as an answer. residual is |f(root)|.
 */

// Shared defaults. `tol` governs both the bracket width and the residual;
// `maxIter` caps every loop so divergence can stall but never hang.
const DEFAULTS = { tol: 1e-10, maxIter: 100 };

// "Tangent is effectively flat" threshold. Newton divides by f'(x); as f'→0
// the tangent's x-intercept flies to infinity, so we refuse the step rather
// than emit Inf/NaN and call it convergence.
const FLAT = 1e-14;

/* ===========================================================================
 * Layer 0 — Bisection. Unconditionally robust, linearly slow.
 *
 * Idea (Intermediate Value Theorem): if f is continuous and f(a), f(b) carry
 * opposite signs, a root lies strictly between them. Halve the interval, keep
 * the half that still straddles the sign change. Each step buys exactly one
 * bit of precision — the error halves every iteration (linear convergence).
 * It cannot fail given a valid bracket; that guarantee is its entire reason
 * for living inside the hybrid below.
 * ========================================================================= */
function bisect(f, a, b, opts = {}) {
  const { tol, maxIter } = { ...DEFAULTS, ...opts };
  let fa = f(a);
  const fb = f(b);

  // No sign change ⇒ the IVT promises nothing. Refuse to invent a root.
  if (!(fa * fb < 0)) {
    return { root: NaN, iters: 0, converged: false, residual: NaN };
  }

  let lo = a;
  let hi = b;
  let mid = NaN;
  let fmid = NaN;
  let iters = 0;

  while (iters < maxIter) {
    iters++;
    mid = 0.5 * (lo + hi);
    fmid = f(mid);
    // |mid − true root| ≤ half the current width, so this is a real bound.
    if (0.5 * (hi - lo) < tol || Math.abs(fmid) < tol) {
      return { root: mid, iters, converged: true, residual: Math.abs(fmid) };
    }
    // Keep the sub-interval that still carries the sign change.
    if (fa * fmid < 0) {
      hi = mid;
    } else {
      lo = mid;
      fa = fmid; // lo moved, so the left-endpoint value moves with it.
    }
  }
  return { root: mid, iters, converged: false, residual: Math.abs(fmid) };
}

/* ===========================================================================
 * Layer 1 — Newton-Raphson. Fast where it works, fragile where it doesn't.
 *
 * Idea (local linearization): near xₙ, replace f by its tangent line and take
 * the tangent's x-intercept as the next guess:
 *     x_{n+1} = xₙ − f(xₙ) / f'(xₙ).
 * Near a SIMPLE root the error squares each step (quadratic convergence). But
 * the method is unguarded: a flat derivative, a poor seed, or an inflection
 * between guess and root sends it to infinity or into a cycle. Exposed for
 * study and explicitly labelled fragile; the production default (Layer 3)
 * never trusts it alone.
 * ========================================================================= */
function newton(f, fprime, x0, opts = {}) {
  const { tol, maxIter } = { ...DEFAULTS, ...opts };
  let x = x0;
  let iters = 0;

  while (iters < maxIter) {
    iters++;
    const fx = f(x);
    if (Math.abs(fx) < tol) {
      return { root: x, iters, converged: true, residual: Math.abs(fx) };
    }
    const dfx = fprime(x);
    // Near-flat tangent ⇒ intercept undefined/enormous. Bail, don't explode.
    if (!Number.isFinite(dfx) || Math.abs(dfx) < FLAT) {
      return { root: x, iters, converged: false, residual: Math.abs(fx) };
    }
    const step = fx / dfx;
    const xNext = x - step;
    if (!Number.isFinite(xNext)) {
      // A non-finite iterate means we have left the rails entirely.
      return { root: x, iters, converged: false, residual: Math.abs(fx) };
    }
    if (Math.abs(step) < tol) {
      const fNext = f(xNext);
      return { root: xNext, iters, converged: true, residual: Math.abs(fNext) };
    }
    x = xNext;
  }
  return { root: x, iters, converged: false, residual: Math.abs(f(x)) };
}

/* ===========================================================================
 * Layer 2 — Secant. Newton without the calculus.
 *
 * Idea: approximate f'(xₙ) by the slope through the last two iterates,
 *     f'(xₙ) ≈ (f(xₙ) − f(xₙ₋₁)) / (xₙ − xₙ₋₁),
 * then take the same tangent-intercept step. No analytic derivative needed.
 * Convergence order is the golden ratio φ ≈ 1.618 — superlinear, sitting
 * between bisection and Newton. It inherits Newton's fragility: a flat secant
 * stalls it, and without a bracket it can wander.
 * ========================================================================= */
function secant(f, x0, x1, opts = {}) {
  const { tol, maxIter } = { ...DEFAULTS, ...opts };
  let xPrev = x0;
  let xCur = x1;
  let fPrev = f(xPrev);
  let fCur = f(xCur);
  let iters = 0;

  while (iters < maxIter) {
    iters++;
    if (Math.abs(fCur) < tol) {
      return { root: xCur, iters, converged: true, residual: Math.abs(fCur) };
    }
    const denom = fCur - fPrev;
    // Flat two-point slope ⇒ no usable step.
    if (!Number.isFinite(denom) || Math.abs(denom) < FLAT) {
      return { root: xCur, iters, converged: false, residual: Math.abs(fCur) };
    }
    const step = (fCur * (xCur - xPrev)) / denom;
    const xNext = xCur - step;
    if (!Number.isFinite(xNext)) {
      return { root: xCur, iters, converged: false, residual: Math.abs(fCur) };
    }
    xPrev = xCur;
    fPrev = fCur;
    xCur = xNext;
    fCur = f(xCur);
    if (Math.abs(step) < tol) {
      return { root: xCur, iters, converged: true, residual: Math.abs(fCur) };
    }
  }
  return { root: xCur, iters, converged: false, residual: Math.abs(fCur) };
}

/* ===========================================================================
 * Layer 3 — Safeguarded hybrid. The whole reason this module exists.
 *
 * Newton (or secant) for speed, bisection for the guarantee. We always hold a
 * valid bracket [xl, xh] with f(xl) < 0 < f(xh). Each step we PROPOSE a fast
 * move and ACCEPT it only if it (a) lands inside the current bracket and
 * (b) is shrinking the interval fast enough to beat plain halving; otherwise
 * we take a bisection step. Either way the new point updates the bracket. This
 * is the rtsafe / Brent philosophy: the fast method can never throw us out of
 * the region known to contain the root. The default solver — the one the
 * console's YTM/IRR/bond commands call — is THIS, never bare Newton.
 *
 * Two acceptance tests (both order-agnostic, so a decreasing f with xl > xh
 * numerically is fine):
 *   - "out of range": the Newton intercept p = rts − f/f' would leave the
 *     bracket. Since ((rts−xh)f' − f)·((rts−xl)f' − f) = f'²·(p−xh)(p−xl),
 *     the product is > 0 exactly when p sits outside [xl, xh].
 *   - "too slow": |2f| > |dxOld·f'| means the proposed step is not even
 *     halving the residual's reach, so bisect instead.
 * ========================================================================= */
function solve(f, spec = {}, opts = {}) {
  const { tol, maxIter } = { ...DEFAULTS, ...opts };
  const { bracket, x0, fprime } = spec;

  if (!bracket || bracket.length !== 2) {
    return { root: NaN, iters: 0, converged: false, residual: NaN };
  }
  const [a, b] = bracket;
  const fa = f(a);
  const fb = f(b);
  if (!(fa * fb < 0)) {
    // No opposite signs ⇒ no bracket ⇒ no safeguard. Refuse.
    return { root: NaN, iters: 0, converged: false, residual: NaN };
  }

  // Orient so f(xl) < 0 < f(xh). Note: for a DECREASING f this puts the
  // numerically larger endpoint in xl — handled because every comparison
  // below uses Math.abs() or the order-agnostic product test.
  let xl;
  let xh;
  if (fa < 0) { xl = a; xh = b; } else { xl = b; xh = a; }

  const inBracket = (x) => x >= Math.min(a, b) && x <= Math.max(a, b);
  let rts = (typeof x0 === 'number' && inBracket(x0)) ? x0 : 0.5 * (a + b);

  let dxOld = Math.abs(b - a); // "stepsize before last"
  let dx = dxOld;
  let fr = f(rts);

  // Secant fallback needs a prior point; seed it from the opposite endpoint.
  let xPrev = (rts === a) ? b : a;
  let fPrev = (rts === a) ? fb : fa;

  let iters = 0;
  while (iters < maxIter) {
    iters++;

    // Derivative: analytic if supplied, else the secant two-point slope.
    let dfr;
    if (typeof fprime === 'function') {
      dfr = fprime(rts);
    } else {
      const d = rts - xPrev;
      dfr = (Math.abs(d) < FLAT) ? NaN : (fr - fPrev) / d;
    }

    const fastUnusable = !Number.isFinite(dfr) || dfr === 0;
    const outOfRange = !fastUnusable &&
      (((rts - xh) * dfr - fr) * ((rts - xl) * dfr - fr) > 0);
    const tooSlow = !fastUnusable &&
      (Math.abs(2 * fr) > Math.abs(dxOld * dfr));

    if (fastUnusable || outOfRange || tooSlow) {
      // Bisection step — guaranteed progress, half the interval.
      dxOld = dx;
      dx = 0.5 * (xh - xl);
      xPrev = rts; fPrev = fr;
      rts = xl + dx;
    } else {
      // Newton / secant step — x_{n+1} = xₙ − f/f'.
      dxOld = dx;
      dx = fr / dfr;
      xPrev = rts; fPrev = fr;
      rts = rts - dx;
    }

    // Convergence on step size or residual. HARDENED (2026-08-01, external
    // review): a small STEP alone is not proof of convergence — a near-flat
    // stretch can shrink the Newton step while the iterate is still far from
    // the root. Accept the step-size criterion only when corroborated by a
    // small residual OR a collapsed bracket (bisection's own x-guarantee:
    // the root provably lies within |xh−xl|). Otherwise keep iterating —
    // the tooSlow test forces bisection steps that shrink the bracket, so
    // termination is unchanged and maxIter still caps the loop.
    if (Math.abs(dx) < tol) {
      const fEnd = f(rts);
      if (Math.abs(fEnd) < tol || Math.abs(xh - xl) < tol) {
        return { root: rts, iters, converged: true, residual: Math.abs(fEnd) };
      }
    }
    fr = f(rts);
    if (Math.abs(fr) < tol) {
      return { root: rts, iters, converged: true, residual: Math.abs(fr) };
    }

    // Maintain the bracket from the sign of the new point.
    if (fr < 0) { xl = rts; } else { xh = rts; }
  }
  return { root: rts, iters, converged: false, residual: Math.abs(fr) };
}

/* ===========================================================================
 * Layer 4 — Bond pricing and yield-to-maturity.
 *
 * Forward map: present value of a level-coupon bond at per-period yield y,
 *     PV(y) = Σ_{t=1}^{N} coupon/(1+y)^t  +  face/(1+y)^N.
 * Every term falls monotonically as y rises, so PV is strictly DECREASING in
 * y: exactly one yield reproduces a given price, and bracketing it is trivial.
 * ========================================================================= */
function bondPrice(y, { face, coupon, periods }) {
  const g = 1 + y;
  let pv = 0;
  for (let t = 1; t <= periods; t++) {
    pv += coupon / Math.pow(g, t);
  }
  pv += face / Math.pow(g, periods);
  return pv;
}

/*
 * Analytic price sensitivity dPV/dy:
 *     dPV/dy = −Σ_{t=1}^{N} t·CF_t/(1+y)^{t+1},   CF_t = coupon (+ face at t=N).
 * Read its shape: this equals −1/(1+y) times Σ t·CF_t/(1+y)^t, the cashflow-
 * weighted average maturity (the Macaulay-duration numerator). Up to the
 * −1/(1+y) factor and a /PV normalization it IS the bond's duration — the
 * first-order price/yield slope. Handing this to `solve` turns the YTM search
 * into a safeguarded Newton riding a TRUE derivative, not an estimated one.
 */
function bondPriceDeriv(y, { face, coupon, periods }) {
  const g = 1 + y;
  let d = 0;
  for (let t = 1; t <= periods; t++) {
    d -= (t * coupon) / Math.pow(g, t + 1);
  }
  d -= (periods * face) / Math.pow(g, periods + 1);
  return d;
}

function ytm({ price, face, coupon, periods }, opts = {}) {
  const { frequency = 1, annualize = 'bey', tol, maxIter } =
    { ...DEFAULTS, ...opts };

  const f = (y) => bondPrice(y, { face, coupon, periods }) - price;
  const fp = (y) => bondPriceDeriv(y, { face, coupon, periods });

  // Bracket the per-period yield. PV decreases in y, so f(low y) > 0 and
  // f(high y) < 0. Start just above −1 (where PV blows up) and push the upper
  // end out until the price is undershot.
  const lo = -0.9999;
  let hi = 1.0;
  let guard = 0;
  while (f(hi) > 0 && guard < 200) { hi *= 2; guard++; }

  const res = solve(f, { bracket: [lo, hi], fprime: fp }, { tol, maxIter });
  const periodicYield = res.root;

  // Annualization — stated explicitly, never implicit:
  //   'bey'       bond-equivalent yield = periodic × frequency (simple).
  //   'effective' compounded            = (1+periodic)^frequency − 1.
  const annualYield = annualize === 'effective'
    ? Math.pow(1 + periodicYield, frequency) - 1
    : periodicYield * frequency;

  return {
    periodicYield,
    annualYield,
    annualization: annualize, // label which convention produced annualYield.
    converged: res.converged,
    residual: res.residual,
  };
}

/* ===========================================================================
 * Layer 5 — NPV and IRR.
 *
 * NPV discounts a dated cashflow sequence back to t=0:
 *     NPV(r) = Σ_{t=0}^{n} CF_t/(1+r)^t,   CF_0 the period-0 flow.
 * IRR is the rate that zeroes NPV. The catch is UNIQUENESS: by Descartes' rule
 * of signs the count of positive real roots is bounded by the number of sign
 * changes in the cashflow sequence. One sign change ⇒ a single IRR (the
 * conventional invest-then-harvest profile). Several sign changes ⇒ possibly
 * several IRRs, at which point "the" IRR is a category error. We DETECT that
 * rather than silently returning whichever root a seed happens to land nearest
 * (which is exactly what Excel's IRR does).
 * ========================================================================= */
function npv(rate, cashflows) {
  const g = 1 + rate;
  let v = 0;
  for (let t = 0; t < cashflows.length; t++) {
    v += cashflows[t] / Math.pow(g, t);
  }
  return v;
}

function npvDeriv(rate, cashflows) {
  // d(NPV)/dr = Σ_{t≥1} −t·CF_t/(1+r)^{t+1}. The t=0 term is rate-independent.
  const g = 1 + rate;
  let d = 0;
  for (let t = 1; t < cashflows.length; t++) {
    d -= (t * cashflows[t]) / Math.pow(g, t + 1);
  }
  return d;
}

function countSignChanges(seq) {
  let changes = 0;
  let prev = 0;
  for (const x of seq) {
    if (x === 0) continue; // zeros do not break or make a sign change.
    const s = Math.sign(x);
    if (prev !== 0 && s !== prev) changes++;
    prev = s;
  }
  return changes;
}

function irr(cashflows, opts = {}) {
  const { tol, maxIter } = { ...DEFAULTS, ...opts };
  const signChanges = countSignChanges(cashflows);

  if (signChanges > 1) {
    // Non-conventional: there may be several IRRs. Enumerate, report the
    // first, and flag non-uniqueness instead of pretending there is one.
    const roots = irrAll(cashflows, {});
    return {
      rate: roots.length ? roots[0] : NaN,
      converged: roots.length > 0,
      unique: false,
      signChanges,
      roots,
      warning: 'IRR is non-unique for non-conventional cashflows; roots enumerated in `roots`.',
    };
  }

  const f = (r) => npv(r, cashflows);
  const fp = (r) => npvDeriv(r, cashflows);

  // Conventional bracket on r ∈ (−1, ∞). Just above −1 the near-term flows
  // blow up; far out NPV → CF_0. A single sign change straddles zero here.
  const lo = -0.999999;
  let hi = 1.0;
  let guard = 0;
  while (f(lo) * f(hi) > 0 && guard < 200) { hi *= 2; guard++; }

  const res = solve(f, { bracket: [lo, hi], fprime: fp }, { tol, maxIter });
  return {
    rate: res.root,
    converged: res.converged,
    unique: true,
    signChanges,
    residual: res.residual,
  };
}

// SCOPE OF THE ENUMERATION (stated so nobody mistakes it for a proof): this
// is a grid scan over [lo, hi] (defaults −99%..+1000%) polished by the
// safeguarded solver. It finds every root the grid BRACKETS. It can miss:
// roots outside [lo, hi] (widen via opts — the range is configurable),
// even-multiplicity/tangent roots (NPV touches zero without a sign change),
// and root PAIRS closer together than `step`. Descartes' bound in irr()
// caps how many roots can exist; when irrAll returns fewer than that bound,
// treat the enumeration as a floor, not a census.
function irrAll(cashflows, opts = {}) {
  const { lo = -0.99, hi = 10, step = 0.001, tol, maxIter } =
    { ...DEFAULTS, ...opts };
  const f = (r) => npv(r, cashflows);
  const fp = (r) => npvDeriv(r, cashflows);
  const roots = [];
  const pushRoot = (x) => {
    // De-dupe roots two adjacent brackets (or a node + a crossing) both find.
    if (Number.isFinite(x) && !roots.some((y) => Math.abs(y - x) < 1e-7)) roots.push(x);
  };

  // Scan the grid for sign changes of NPV; each strict crossing brackets a
  // distinct root, which we polish with the safeguarded solver. A node landing
  // *exactly* on a root makes NPV evaluate to 0 — the product test sees a zero
  // (not a sign change) on both sides and would skip the root, so we detect
  // that node directly. This matters precisely when IRRs are clean rationals.
  let rPrev = lo;
  let fPrev = f(rPrev);
  if (fPrev === 0) pushRoot(rPrev);
  for (let r = lo + step; r <= hi + 1e-12; r += step) {
    const fVal = f(r);
    if (Number.isFinite(fPrev) && Number.isFinite(fVal)) {
      if (fPrev * fVal < 0) {
        const res = solve(f, { bracket: [rPrev, r], fprime: fp }, { tol, maxIter });
        if (res.converged) pushRoot(res.root);
      } else if (fVal === 0) {
        pushRoot(r); // grid node coincides with a root.
      }
    }
    rPrev = r;
    fPrev = fVal;
  }
  return roots;
}

/* ===========================================================================
 * Layer 6 — selfTest. The engine's own conscience: a compact in-process run
 * of the load-bearing invariants, returning a single boolean. Mirrors the
 * console's `self-test`. The exhaustive battery lives in the .test.js file.
 * ========================================================================= */
function selfTest() {
  const near = (a, b, eps = 1e-8) => Number.isFinite(a) && Math.abs(a - b) < eps;
  const c = [];

  // Solver core
  c.push(near(solve((x) => x - 3, { bracket: [0, 10] }).root, 3));
  c.push(near(solve((x) => x * x - 2, { bracket: [0, 2] }).root, Math.SQRT2));
  c.push(bisect((x) => x * x + 1, -1, 1).converged === false);
  c.push(newton((x) => x * x - 2, (x) => 2 * x, 0).converged === false);

  // YTM round-trip
  const par = ytm({ price: 100, face: 100, coupon: 5, periods: 10 });
  c.push(near(par.periodicYield, 0.05));
  c.push(near(bondPrice(par.periodicYield, { face: 100, coupon: 5, periods: 10 }), 100, 1e-6));

  // IRR
  c.push(near(irr([-100, 110]).rate, 0.1));
  c.push(near(irr([-100, 0, 121]).rate, 0.1));
  c.push(irr([-100, 230, -132]).unique === false);
  c.push(irrAll([-100, 230, -132]).length === 2);

  return c.every(Boolean);
}

/* ===========================================================================
 * Layer 6b — XIRR (OPTIONAL, separable). IRR for irregularly-dated cashflows.
 * Same solver; only the exponent changes — time is measured in actual/365 year
 * fractions from the first date instead of integer periods:
 *     NPV(r) = Σ_i CF_i / (1+r)^{(d_i − d_0)/365}.
 * Delete this block and Layers 0–6 are untouched. Present only because the
 * console roadmap's `xirr` should ride THIS engine, not a second ad-hoc one.
 * ========================================================================= */
function yearFractions(dates) {
  const toDate = (d) => (d instanceof Date ? d : new Date(d));
  const d0 = toDate(dates[0]);
  const MS_PER_DAY = 86400000;
  return dates.map((d) => (toDate(d) - d0) / MS_PER_DAY / 365);
}

function xnpv(rate, cashflows, dates) {
  const ts = yearFractions(dates);
  const g = 1 + rate;
  let v = 0;
  for (let i = 0; i < cashflows.length; i++) {
    v += cashflows[i] / Math.pow(g, ts[i]);
  }
  return v;
}

function xnpvDeriv(rate, cashflows, dates) {
  const ts = yearFractions(dates);
  const g = 1 + rate;
  let d = 0;
  for (let i = 0; i < cashflows.length; i++) {
    if (ts[i] === 0) continue;
    d -= (ts[i] * cashflows[i]) / Math.pow(g, ts[i] + 1);
  }
  return d;
}

function xirr(cashflows, dates, opts = {}) {
  const { tol, maxIter } = { ...DEFAULTS, ...opts };
  const f = (r) => xnpv(r, cashflows, dates);
  const fp = (r) => xnpvDeriv(r, cashflows, dates);
  const signChanges = countSignChanges(cashflows);
  const lo = -0.999999;
  let hi = 1.0;
  let guard = 0;
  while (f(lo) * f(hi) > 0 && guard < 200) { hi *= 2; guard++; }
  const res = solve(f, { bracket: [lo, hi], fprime: fp }, { tol, maxIter });
  return {
    rate: res.root,
    converged: res.converged,
    unique: signChanges <= 1,
    signChanges,
    residual: res.residual,
  };
}

// Public API — one object, exposed to whichever environment loads the module.
// The engine itself touches no DOM and no I/O; it only hands this surface to
// its host. Node's `require` (the battery) takes the CommonJS branch; a browser
// `<script src>` (the test bench) takes the global branch. Same source, no fork.
const API = {
  // Layer 0–3 — solver core
  bisect, newton, secant, solve,
  // Layer 4 — bonds
  bondPrice, bondPriceDeriv, ytm,
  // Layer 5 — NPV / IRR
  npv, npvDeriv, countSignChanges, irr, irrAll,
  // Layer 6 — self-test
  selfTest,
  // Layer 6b — optional, separable
  yearFractions, xnpv, xnpvDeriv, xirr,
  // constants (exposed for the battery)
  DEFAULTS, FLAT,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;            // Node / CommonJS — aestimo-solve.test.js.
} else if (typeof globalThis !== 'undefined') {
  globalThis.AestimoSolve = API;   // Browser — aestimo-test.html.
}
