/*
 * aestimo-chol.js — SPD covariance-factorization primitive.
 *
 * Cholesky decomposition, the triangular solves it enables, an SPD
 * solve/inverse built on top, the correlated-draw transform used by the
 * Gaussian copula, and LDLᵀ as the graceful-degradation path for
 * positive-*semi*-definite (rank-deficient) targets.
 *
 * Convention: matrices are row-major arrays of arrays, A[row][col].
 * Vectors are flat arrays.
 *
 * Zero dependencies. No RNG lives here — correlate() is pure linear
 * algebra; the PRNG module supplies the iid draws it consumes. Deployed
 * as a single source of truth behind two consumers: the DCF Workstation's
 * Gaussian copula, and the forthcoming Markowitz optimizer's SPD solve
 * (which replaces any explicit Σ⁻¹ with a solve against Σ).
 *
 * UMD wrapper — runs under Node (module.exports) and in the browser
 * (globalThis.AestimoChol), same pattern as aestimo-solve.js.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.AestimoChol = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ============================================================
  // Layer 0 — matrix utilities
  // ============================================================

  // A matrix is square iff every row has the same length as the row count.
  function isSquare(A) {
    if (!Array.isArray(A) || A.length === 0) return false;
    var n = A.length;
    for (var i = 0; i < n; i++) {
      if (!Array.isArray(A[i]) || A[i].length !== n) return false;
    }
    return true;
  }

  // Symmetry (A = Aᵀ) is checked entrywise against a tolerance, since a Σ
  // built by summing outer products — (1/N) Σ rᵢrᵢᵀ — is symmetric by
  // construction yet accumulates its (i,j) and (j,i) sums in different
  // orders, so the two entries differ in their last bits.
  //
  // The tolerance MUST be scale-relative. A fixed absolute floor (e.g.
  // 1e-12) is not scale-invariant: at a data scale of 1e8 a single ULP of
  // rounding (~1e-8 in absolute terms, ~1e-15 relative) already exceeds it,
  // and a genuinely symmetric covariance is wrongly rejected. We use the
  // standard mixed form |a−b| ≤ tol·(1 + max(|a|,|b|)): the constant term
  // is an absolute floor for entries near zero, the max term scales the
  // allowance with the magnitude of the entries being compared, so the test
  // measures asymmetry in ULPs rather than in absolute units.
  function isSymmetric(A, tol) {
    if (tol === undefined) tol = 1e-12;
    if (!isSquare(A)) return false;
    var n = A.length;
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        var a = A[i][j], b = A[j][i];
        var scale = Math.max(Math.abs(a), Math.abs(b));
        if (Math.abs(a - b) > tol * (1 + scale)) return false;
      }
    }
    return true;
  }

  function transpose(A) {
    var n = A.length, m = A[0].length;
    var T = new Array(m);
    for (var j = 0; j < m; j++) {
      T[j] = new Array(n);
      for (var i = 0; i < n; i++) T[j][i] = A[i][j];
    }
    return T;
  }

  // Matrix-vector product. Each output entry is a single dot product
  // accumulated into one running sum — no intermediate arrays per entry.
  function matVec(A, x) {
    var n = A.length;
    var out = new Array(n);
    for (var i = 0; i < n; i++) {
      var row = A[i];
      var sum = 0;
      for (var k = 0; k < row.length; k++) sum += row[k] * x[k];
      out[i] = sum;
    }
    return out;
  }

  // Matrix-matrix product, needed for reconstruction tests (L·Lᵀ ≈ A) and
  // for the invSPD round-trip check. Standard triple loop, running-sum
  // accumulation per output entry.
  function matMul(A, B) {
    var n = A.length, k = B.length, m = B[0].length;
    var C = new Array(n);
    for (var i = 0; i < n; i++) {
      C[i] = new Array(m);
      for (var j = 0; j < m; j++) {
        var sum = 0;
        for (var t = 0; t < k; t++) sum += A[i][t] * B[t][j];
        C[i][j] = sum;
      }
    }
    return C;
  }

  function zerosVec(n) {
    var v = new Array(n);
    for (var i = 0; i < n; i++) v[i] = 0;
    return v;
  }

  function zerosMat(n) {
    var M = new Array(n);
    for (var i = 0; i < n; i++) M[i] = zerosVec(n);
    return M;
  }

  // ============================================================
  // Layer 1 — Cholesky: the point of the tool
  // ============================================================
  //
  // We want L, lower-triangular, with L·Lᵀ = A. Equate entry (i,j) of
  // L·Lᵀ — which is Σ_{k=0}^{j} L[i][k]·L[j][k] — to A[i][j], and notice
  // that when j < i every term on the right except the last (k=j) involves
  // only entries of L already computed in earlier rows/columns. Solving
  // the last term for the one new unknown gives:
  //
  //   L[i][j] = ( A[i][j] − Σ_{k<j} L[i][k]·L[j][k] ) / L[j][j]     (j < i)
  //   L[i][i] = √( A[i][i] − Σ_{k<i} L[i][k]² )                     (j = i)
  //
  // Processing row by row, left to right within each row (Cholesky–
  // Banachiewicz order), every term needed on the right has already been
  // computed by the time it's used.
  //
  // Why a non-positive diagonal pivot means "not positive-definite": the
  // quantity under the square root at step i is exactly the Schur
  // complement of the leading (i×i) principal submatrix — algebraically,
  // it is det(A_{0..i}) / det(A_{0..i-1}) for the leading minors. A matrix
  // is SPD iff *every* leading principal minor is strictly positive
  // (Sylvester's criterion). So the moment a pivot is ≤ 0, the leading
  // i×i block has a non-positive determinant ratio, which is equivalent
  // to that block failing to be positive-definite — and a matrix cannot
  // be SPD if any leading submatrix isn't. Cholesky succeeding is not
  // merely *compatible with* positive-definiteness, it IS the test.
  function cholesky(A, opts) {
    var tol = (opts && opts.tol !== undefined) ? opts.tol : 1e-12;

    if (!isSquare(A)) {
      return { ok: false, reason: 'not square' };
    }
    if (!isSymmetric(A, tol)) {
      return { ok: false, reason: 'not symmetric' };
    }

    var n = A.length;
    var L = zerosMat(n);

    // Pivot scale: the largest diagonal magnitude of A. A pivot is judged
    // non-positive relative to this, not against an absolute floor — the
    // same scale-invariance argument as isSymmetric. Without it, a valid
    // SPD matrix scaled up (data-scale covariance) is fine but a valid one
    // scaled *down* would spuriously trip, and "how close to zero counts as
    // singular" would depend on the units of the returns rather than on the
    // conditioning of the matrix.
    var pivotScale = 0;
    for (var d = 0; d < n; d++) pivotScale = Math.max(pivotScale, Math.abs(A[d][d]));
    var pivotTol = tol * (1 + pivotScale);

    for (var i = 0; i < n; i++) {
      for (var j = 0; j <= i; j++) {
        var sum = A[i][j];
        for (var k = 0; k < j; k++) sum -= L[i][k] * L[j][k];

        if (j === i) {
          if (sum <= pivotTol) {
            // Not positive-definite: the i-th leading minor is
            // non-positive (relative to matrix scale). Legible structured
            // failure, never NaN.
            return { ok: false, reason: 'not positive definite', failedAt: i };
          }
          L[i][i] = Math.sqrt(sum);
        } else {
          L[i][j] = sum / L[j][j];
        }
      }
    }

    return { ok: true, L: L };
  }

  // ============================================================
  // Layer 2 — triangular solves
  // ============================================================
  //
  // A triangular system unwinds one unknown at a time: in L·y = b with L
  // lower-triangular, row i involves y[0..i] but only y[i] is unknown once
  // y[0..i-1] are known, so solving top-to-bottom peels off exactly one
  // new unknown per row — O(n²) total, no elimination needed. Back
  // substitution is the mirror image, bottom-to-top, for an upper-
  // triangular system.

  // Solves L·y = b for y, L lower-triangular.
  function forwardSub(L, b) {
    var n = L.length;
    var y = new Array(n);
    for (var i = 0; i < n; i++) {
      var sum = b[i];
      for (var k = 0; k < i; k++) sum -= L[i][k] * y[k];
      y[i] = sum / L[i][i];
    }
    return y;
  }

  // Solves U·x = y for x, U upper-triangular.
  function backSub(U, y) {
    var n = U.length;
    var x = new Array(n);
    for (var i = n - 1; i >= 0; i--) {
      var sum = y[i];
      for (var k = i + 1; k < n; k++) sum -= U[i][k] * x[k];
      x[i] = sum / U[i][i];
    }
    return x;
  }

  // ============================================================
  // Layer 3 — SPD solve & inverse
  // ============================================================
  //
  // Solving A·x = b for SPD A is factor-then-substitute: A = L·Lᵀ, so
  // A·x = b becomes L·(Lᵀ·x) = b. Let y = Lᵀ·x; forward-solve L·y = b for
  // y, then back-solve Lᵀ·x = y for x. Two O(n²) triangular solves replace
  // one O(n³) elimination on every subsequent right-hand side once L is
  // cached — the standard justification for factoring once, solving many.

  // Primary interface. The buy-side-relevant quantity is Σ⁻¹μ, which is a
  // *solve*, not a matrix you form explicitly — solveSPD encodes that.
  function solveSPD(A, b) {
    var chol = cholesky(A);
    if (!chol.ok) return chol; // propagate structured failure

    var L = chol.L;
    var LT = transpose(L);
    var y = forwardSub(L, b);
    var x = backSub(LT, y);
    return { ok: true, x: x };
  }

  // Forming an explicit inverse is numerically inferior to solving against
  // specific right-hand sides and is usually unnecessary; invSPD is
  // exposed only for the occasions it's genuinely needed (e.g. reporting
  // a full covariance of estimated parameters). Factor once, then solve
  // A·X = I column by column — cheaper than n independent solveSPD calls
  // since the factorization is shared.
  function invSPD(A) {
    var chol = cholesky(A);
    if (!chol.ok) return chol;

    var L = chol.L;
    var LT = transpose(L);
    var n = A.length;
    var cols = new Array(n);

    for (var j = 0; j < n; j++) {
      var e = zerosVec(n);
      e[j] = 1;
      var y = forwardSub(L, e);
      cols[j] = backSub(LT, y);
    }

    // cols[j] is column j of X; assemble row-major X from it.
    var X = new Array(n);
    for (var i = 0; i < n; i++) {
      X[i] = new Array(n);
      for (var jj = 0; jj < n; jj++) X[i][jj] = cols[jj][i];
    }

    return { ok: true, X: X };
  }

  // ============================================================
  // Layer 4 — correlated-draw transform
  // ============================================================
  //
  // If z is a vector of iid standard-normal draws, Cov(z) = I. For any
  // matrix L, Cov(L·z) = L·Cov(z)·Lᵀ = L·I·Lᵀ = L·Lᵀ. So if L is the
  // Cholesky factor of a target covariance Σ (Σ = L·Lᵀ by construction),
  // then correlate(L, z) = L·z is a draw with exactly covariance Σ. This
  // identity is the entire basis of the Gaussian copula: factor the
  // target correlation/covariance once, then every iid normal vector
  // pushed through L·(·) inherits the target's correlation structure.
  // This module supplies L and the transform; it never generates z.

  function correlate(L, z) {
    return matVec(L, z);
  }

  // Covariance → correlation: strip the marginal scale (standard
  // deviations) off the diagonal, leaving pure co-movement.
  //   R[i][j] = Σ[i][j] / (sd[i]·sd[j]),   sd[i] = √Σ[i][i]
  function covToCorr(Sigma) {
    var n = Sigma.length;
    var sd = new Array(n);
    for (var i = 0; i < n; i++) sd[i] = Math.sqrt(Sigma[i][i]);

    var R = new Array(n);
    for (var i2 = 0; i2 < n; i2++) {
      R[i2] = new Array(n);
      for (var j = 0; j < n; j++) {
        R[i2][j] = Sigma[i2][j] / (sd[i2] * sd[j]);
      }
    }
    return R;
  }

  // Correlation + marginal scales → covariance (inverse of covToCorr).
  //   Σ[i][j] = R[i][j]·sd[i]·sd[j]
  function corrToCov(R, sd) {
    var n = R.length;
    var Sigma = new Array(n);
    for (var i = 0; i < n; i++) {
      Sigma[i] = new Array(n);
      for (var j = 0; j < n; j++) {
        Sigma[i][j] = R[i][j] * sd[i] * sd[j];
      }
    }
    return Sigma;
  }

  // ============================================================
  // Layer 5 — LDLᵀ: the semi-definite boundary
  // ============================================================
  //
  // Real correlation matrices built from data are frequently positive
  // *semi*-definite rather than strictly definite — e.g. one asset is an
  // exact linear combination of others, or n assets are estimated from
  // fewer than n independent return periods. Textbook Cholesky hits √0
  // (or, with noise, √(tiny negative)) at that pivot and reports failure.
  // LDLᵀ avoids the square root entirely and degrades gracefully: it
  // factors A = L·D·Lᵀ with L unit-lower-triangular (1s on the diagonal)
  // and D diagonal, and a zero eigenvalue simply shows up as a zero entry
  // in D rather than aborting the factorization.
  //
  // Derivation mirrors Cholesky, but pulls the diagonal scaling out into
  // D instead of into L: equating entry (i,j) of L·D·Lᵀ (j < i) to A[i][j]
  // gives Σ_{k≤j} L[i][k]·D[k]·L[j][k] = A[i][j]; since L[j][j] = 1, the
  // one new unknown L[i][j] solves out as:
  //
  //   D[i]    = A[i][i] − Σ_{k<i} L[i][k]²·D[k]
  //   L[i][j] = ( A[i][j] − Σ_{k<j} L[i][k]·L[j][k]·D[k] ) / D[j]   (j<i)
  //
  // Relationship to Cholesky for strictly SPD input: L_chol = L·D^{1/2}
  // (scale each column j of L by √D[j]) — LDLᵀ is the more general
  // factorization; Cholesky is the special case with the square root
  // folded back into L.
  //
  // Handling D[j] ≈ 0: if a pivot collapses to (numerically) zero, then
  // for a genuinely PSD matrix the corresponding column contributes
  // nothing to any later entry — the numerator that would divide by D[j]
  // is itself ≈0 for consistent PSD input. We clamp D[j] to exactly 0 and
  // set the corresponding L entries to 0 rather than dividing, which is
  // the well-defined continuation of the limit as the pivot → 0.
  function ldlt(A, opts) {
    var tol = (opts && opts.tol !== undefined) ? opts.tol : 1e-12;

    if (!isSquare(A)) return { ok: false, reason: 'not square' };
    if (!isSymmetric(A, tol)) return { ok: false, reason: 'not symmetric' };

    var n = A.length;
    var L = zerosMat(n);
    var D = zerosVec(n);
    for (var d = 0; d < n; d++) L[d][d] = 1;

    // Scale-relative zero clamp, same argument as cholesky's pivotTol: a
    // pivot counts as zero relative to the matrix's diagonal magnitude, not
    // against an absolute floor.
    var pivotScale = 0;
    for (var d2 = 0; d2 < n; d2++) pivotScale = Math.max(pivotScale, Math.abs(A[d2][d2]));
    var pivotTol = tol * (1 + pivotScale);

    for (var i = 0; i < n; i++) {
      var dsum = A[i][i];
      for (var k = 0; k < i; k++) dsum -= L[i][k] * L[i][k] * D[k];
      D[i] = Math.abs(dsum) <= pivotTol ? 0 : dsum;

      for (var j = i + 1; j < n; j++) {
        if (D[i] === 0) {
          // Degenerate column: well-defined continuation, see derivation.
          L[j][i] = 0;
          continue;
        }
        var lsum = A[j][i];
        for (var k2 = 0; k2 < i; k2++) lsum -= L[j][k2] * L[i][k2] * D[k2];
        L[j][i] = lsum / D[i];
      }
    }

    // HARDENED (2026-08-01, external review): LDLᵀ happily factors INDEFINITE
    // symmetric matrices (negative pivots land in D as-is — that is correct
    // LDLᵀ behavior), but this module advertises it as the positive-SEMI-
    // definite degradation path. A caller using it as a PSD fallback must be
    // able to tell the two apart without re-scanning D, so the verdict ships
    // in the result: psd is true iff every pivot is ≥ 0.
    var psd = true;
    for (var p = 0; p < n; p++) if (D[p] < 0) { psd = false; break; }
    return { ok: true, L: L, D: D, psd: psd };
  }

  // ============================================================
  // Layer 6 — self-test & invariant battery
  // ============================================================
  //
  // The battery is the single source of truth for the module's executable
  // invariants — including the golden regression locks. It ships *with* the
  // engine so every consumer asserts the same guarantees against the same
  // source: the node runner (aestimo-chol.test.js) is a thin PASS/FAIL
  // printer over battery(), and the in-browser bench renders the same array
  // to its rail. No copy of the checks lives anywhere else; there is nothing
  // to drift.
  //
  // battery() returns [{ name, pass, detail }]. selfTest() collapses that to
  // a boolean, mirroring the console's `self-test` command (Layer 6 as
  // originally specified).
  function approxEqualMat(A, B, tol) {
    if (A.length !== B.length) return false;
    for (var i = 0; i < A.length; i++) {
      if (A[i].length !== B[i].length) return false;
      for (var j = 0; j < A[i].length; j++) {
        if (Math.abs(A[i][j] - B[i][j]) > tol) return false;
      }
    }
    return true;
  }
  function approxEqualVec(a, b, tol) {
    if (a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > tol) return false;
    }
    return true;
  }
  function eye(m) {
    var I = new Array(m);
    for (var i = 0; i < m; i++) {
      I[i] = new Array(m);
      for (var j = 0; j < m; j++) I[i][j] = (i === j ? 1 : 0);
    }
    return I;
  }

  var EXACT = 1e-12;
  var LOOSE = 1e-9;

  // The canonical check list. Each entry: [name, () => boolean].
  var CHECKS = [
    ['cholesky reconstruction L·Lᵀ ≈ A (SPD)', function () {
      var A = [[6, 2, 1], [2, 5, 2], [1, 2, 4]];
      var r = cholesky(A);
      return r.ok && approxEqualMat(matMul(r.L, transpose(r.L)), A, LOOSE);
    }],
    ['cholesky L lower-triangular, diag > 0', function () {
      var r = cholesky([[6, 2, 1], [2, 5, 2], [1, 2, 4]]);
      if (!r.ok) return false;
      for (var i = 0; i < 3; i++) {
        if (r.L[i][i] <= 0) return false;
        for (var j = i + 1; j < 3; j++) if (r.L[i][j] !== 0) return false;
      }
      return true;
    }],
    ['cholesky(I) = I', function () {
      var r = cholesky(eye(3));
      return r.ok && approxEqualMat(r.L, eye(3), EXACT);
    }],
    ['cholesky(diag(4,9,16)) = diag(2,3,4)', function () {
      var r = cholesky([[4, 0, 0], [0, 9, 0], [0, 0, 16]]);
      return r.ok && approxEqualMat(r.L, [[2, 0, 0], [0, 3, 0], [0, 0, 4]], EXACT);
    }],
    ['golden L [[4,2],[2,3]] → [[2,0],[1,√2]]', function () {
      var r = cholesky([[4, 2], [2, 3]]);
      return r.ok && approxEqualMat(r.L, [[2, 0], [1, Math.SQRT2]], LOOSE);
    }],
    ['non-PD [[1,2],[2,1]] → ok:false, failedAt:1, no NaN', function () {
      var r = cholesky([[1, 2], [2, 1]]);
      return r.ok === false && r.failedAt === 1 && !r.hasOwnProperty('L');
    }],
    ['non-symmetric flagged, not factored', function () {
      var r = cholesky([[1, 2], [3, 4]]);
      return r.ok === false && r.reason === 'not symmetric';
    }],
    ['PSD-singular [[1,1],[1,1]] → rank-deficient', function () {
      return cholesky([[1, 1], [1, 1]]).ok === false;
    }],
    // Regression lock for the scale-relative symmetry/pivot bug fix: a
    // data-scale covariance with one-ULP summation asymmetry must factor.
    ['scale-relative symmetry accepts data-scale Σ (bugfix)', function () {
      var A = [[4e8, 2e8, 1e8], [2e8 + 3e-7, 5e8, 2e8], [1e8, 2e8 - 4e-7, 4e8]];
      var r = cholesky(A);
      return r.ok === true;
    }],
    ['forwardSub L·y=b round trip', function () {
      var L = [[2, 0, 0], [1, 3, 0], [4, 1, 5]], b = [4, 10, 27];
      return approxEqualVec(matVec(L, forwardSub(L, b)), b, LOOSE);
    }],
    ['backSub U·x=y round trip', function () {
      var U = [[2, 1, 4], [0, 3, 1], [0, 0, 5]], y = [7, 10, 5];
      return approxEqualVec(matVec(U, backSub(U, y)), y, LOOSE);
    }],
    ['solveSPD residual A·x ≈ b', function () {
      var A = [[6, 2, 1], [2, 5, 2], [1, 2, 4]], b = [1, 2, 3];
      var r = solveSPD(A, b);
      return r.ok && approxEqualVec(matVec(A, r.x), b, LOOSE);
    }],
    ['solveSPD(I, b) = b', function () {
      var b = [3, -1, 7];
      var r = solveSPD(eye(3), b);
      return r.ok && approxEqualVec(r.x, b, EXACT);
    }],
    ['solveSPD(A, e_j) reproduces invSPD column j', function () {
      var A = [[6, 2, 1], [2, 5, 2], [1, 2, 4]];
      var inv = invSPD(A);
      if (!inv.ok) return false;
      for (var j = 0; j < 3; j++) {
        var e = [0, 0, 0]; e[j] = 1;
        var sol = solveSPD(A, e);
        if (!sol.ok) return false;
        for (var i = 0; i < 3; i++) if (Math.abs(sol.x[i] - inv.X[i][j]) > LOOSE) return false;
      }
      return true;
    }],
    ['invSPD round-trip A⁻¹·A ≈ I', function () {
      var A = [[6, 2, 1], [2, 5, 2], [1, 2, 4]];
      var inv = invSPD(A);
      return inv.ok && approxEqualMat(matMul(inv.X, A), eye(3), LOOSE);
    }],
    ['correlate(I, z) = z', function () {
      var z = [0.5, -1.2, 2.3];
      return approxEqualVec(correlate(eye(3), z), z, EXACT);
    }],
    ['covToCorr / corrToCov round trip', function () {
      var S = [[4, 2], [2, 9]];
      var R = covToCorr(S);
      return approxEqualMat(corrToCov(R, [2, 3]), S, LOOSE) &&
             Math.abs(R[0][0] - 1) < EXACT && Math.abs(R[1][1] - 1) < EXACT;
    }],
    ['correlate: sample cov of many draws ≈ L·Lᵀ', function () {
      // Deterministic inline generator; the engine proper stays RNG-free.
      var seed = 12345;
      function rnd() {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      }
      function randn() { var u1 = Math.max(rnd(), 1e-12), u2 = rnd(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
      var Sigma = [[4, 2], [2, 9]];
      var chol = cholesky(Sigma);
      if (!chol.ok) return false;
      var L = chol.L, N = 60000;
      var sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
      for (var i = 0; i < N; i++) {
        var w = correlate(L, [randn(), randn()]);
        sx += w[0]; sy += w[1]; sxx += w[0] * w[0]; sxy += w[0] * w[1]; syy += w[1] * w[1];
      }
      var mx = sx / N, my = sy / N;
      var cxx = sxx / N - mx * mx, cxy = sxy / N - mx * my, cyy = syy / N - my * my;
      var t = 0.15;
      return Math.abs(cxx - 4) < t && Math.abs(cxy - 2) < t && Math.abs(cyy - 9) < t;
    }],
    ['ldlt L·D·Lᵀ ≈ A, L unit-lower, D diagonal', function () {
      var A = [[6, 2, 1], [2, 5, 2], [1, 2, 4]];
      var r = ldlt(A);
      if (!r.ok) return false;
      for (var i = 0; i < 3; i++) {
        if (r.L[i][i] !== 1) return false;
        for (var j = i + 1; j < 3; j++) if (r.L[i][j] !== 0) return false;
      }
      var D = eye(3); for (var k = 0; k < 3; k++) D[k][k] = r.D[k];
      return approxEqualMat(matMul(matMul(r.L, D), transpose(r.L)), A, LOOSE);
    }],
    ['ldlt PSD-singular → D = diag(1, 0)', function () {
      var r = ldlt([[1, 1], [1, 1]]);
      return r.ok && Math.abs(r.D[0] - 1) < EXACT && Math.abs(r.D[1]) < EXACT;
    }],
    ['ldlt psd flag: PSD ⇒ true, indefinite ⇒ false (negative pivot exposed)', function () {
      var psdCase = ldlt([[1, 1], [1, 1]]);          // eigenvalues {2, 0}
      var indef = ldlt([[1, 2], [2, 1]]);            // eigenvalues {3, −1} ⇒ D[1] = −3
      return psdCase.ok && psdCase.psd === true &&
             indef.ok && indef.psd === false && indef.D[1] < 0;
    }],
    ['ldlt: L·√D equals Cholesky factor (SPD)', function () {
      var A = [[6, 2, 1], [2, 5, 2], [1, 2, 4]];
      var l = ldlt(A), c = cholesky(A);
      if (!l.ok || !c.ok) return false;
      var s = [];
      for (var i = 0; i < 3; i++) { s.push([]); for (var j = 0; j < 3; j++) s[i].push(l.L[i][j] * Math.sqrt(l.D[j])); }
      return approxEqualMat(s, c.L, LOOSE);
    }],
    // ---- Golden number regression locks (first fully-green build) ----
    ['GOLDEN L of [[6,2,1],[2,5,2],[1,2,4]]', function () {
      var r = cholesky([[6, 2, 1], [2, 5, 2], [1, 2, 4]]);
      var g = [
        [2.44948974278, 0, 0],
        [0.816496580928, 2.08166599947, 0],
        [0.408248290464, 0.800640769025, 1.78670302297]
      ];
      return r.ok && approxEqualMat(r.L, g, 1e-10);
    }],
    ['GOLDEN x of A·x=b, b=[1,2,3]', function () {
      var r = solveSPD([[6, 2, 1], [2, 5, 2], [1, 2, 4]], [1, 2, 3]);
      var g = [0.0120481927711, 0.120481927711, 0.686746987952];
      return r.ok && approxEqualVec(r.x, g, 1e-10);
    }]
  ];

  function battery() {
    var results = new Array(CHECKS.length);
    for (var i = 0; i < CHECKS.length; i++) {
      var name = CHECKS[i][0], pass = false, detail = '';
      try {
        pass = CHECKS[i][1]();
      } catch (e) {
        pass = false;
        detail = (e && e.message) ? e.message : String(e);
      }
      results[i] = { name: name, pass: pass, detail: detail };
    }
    return results;
  }

  function selfTest() {
    var r = battery();
    for (var i = 0; i < r.length; i++) if (!r[i].pass) return false;
    return true;
  }

  return {
    // Layer 0
    isSquare: isSquare,
    isSymmetric: isSymmetric,
    transpose: transpose,
    matVec: matVec,
    matMul: matMul,
    // Layer 1
    cholesky: cholesky,
    // Layer 2
    forwardSub: forwardSub,
    backSub: backSub,
    // Layer 3
    solveSPD: solveSPD,
    invSPD: invSPD,
    // Layer 4
    correlate: correlate,
    covToCorr: covToCorr,
    corrToCov: corrToCov,
    // Layer 5
    ldlt: ldlt,
    // Layer 6
    battery: battery,
    selfTest: selfTest
  };
});
