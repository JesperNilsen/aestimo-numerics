/*
 * aestimo-random.test.js — thin PASS/FAIL printer over the engine's canonical
 * battery(). No checks are defined here; the invariant list, golden-stream
 * lock, and C-reference cross-checks live in aestimo-random.js — one source
 * of truth, nothing to drift. (Phase 2 normalization: the checks this file
 * used to carry were moved into the engine, matching chol's pattern.)
 */
'use strict';

var R = require('./aestimo-random.js');

function run() {
  return { module: 'random', results: R.battery() };
}

module.exports = { run: run };

if (require.main === module) {
  var r = run().results;
  var pass = 0;
  r.forEach(function (x) {
    if (x.pass) pass++;
    console.log((x.pass ? 'PASS' : 'FAIL') + '  ' + x.name + (x.detail ? '  · ' + x.detail : ''));
  });
  console.log(pass + '/' + r.length + ' passed');
  process.exit(pass === r.length ? 0 : 1);
}
