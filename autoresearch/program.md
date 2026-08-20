# Research agenda: low-percentile band (0-25%), pure ML from scratch

## Context: why a narrower scope
A prior campaign trained one model across the FULL percentile range (max-min
through sum-rate). It found a qualitative regime shift around the 50%->100%
transition: optimal policies are egalitarian/scheduling-like at low percentiles
but concentrated/greedy near sum-rate, and a single model represented both
poorly. The percentile range has been split into four bands (0-25%, 25-50%,
50-75%, 75-100%); **this campaign is band 1 (0-25%)** -- deliberately matching
the scope the original paper (Khan & Adve, arXiv:2403.16344 Part I) actually
studied: cell-edge / max-min fairness, not the full spectrum through sum-rate.

## Goal
Derive, FROM SCRATCH, the right learning framework and architecture for a
single model that maps (channel realization, percentile-within-band) -> transmit
powers, for ANY network size K in 1..10 users/cell and ANY percentile within
the 0-25% band. Required API:

    powers = model(A, Kq)      # A: [batch, K, B, B]  ->  powers: [batch, K, B]

Maximize `HELDOUT_SCORE`: the mean over a pinned **17-cell** (K, percentile) grid
of (model SLqP / full-power SLqP), each cell on its own pinned drops. 1.000 =
the trivial full-power floor. Physics is byte-identical to the prior campaign's
certified evaluator; only the percentile scope is new.

## FROM SCRATCH -- what that means here
- **No inheritance from ANY prior campaign.** Do not read, load, imitate, or
  distill from any earlier checkpoints -- not the full-range campaign's, not
  (once they exist) any other band's. This campaign re-derives the approach.
- **All learning frameworks are open**: unsupervised/direct-objective,
  supervised (labels this campaign generates itself -- `qft_reference.py` solves
  any (K, Kq) instance in well under a second for cells in this band; cache
  labels if you build a distillation pipeline, and note the one-time cost in
  your hypothesis), self-supervised/auxiliary tasks, deep RL, curricula,
  ensembles-in-training.
- **All architectures are open**: MLP, CNN, GNN/message-passing, transformer/
  attention, DeepSets, recurrent -- the SAME weights must serve every K, so
  size-generalizing structure (equivariance, weight sharing) is effectively
  mandatory, and Kq must enter as a conditioning input the model actually uses.

## THE INFERENCE CONTRACT (enforced by the evaluator -- violations abort the run)
1. Total forward time over the whole grid (17 cells x 250 drops) <= **10.0 s**.
   QFT needs many minutes on a grid this size; the budget forces a large,
   deliberate speedup and structurally rules out per-instance optimization.
2. Parameters identical before/after evaluation -- no test-time fitting
   (tripwired).
3. Banned inside `forward()` even if fast: gradient steps; restarts or candidate
   sets selected by computed utility; any loop whose acceptance/output depends
   on evaluating the objective on candidate powers. (SINR-like *features* of the
   input are fine.) Learned unrolled optimizers that evaluate objective
   gradients at test time remain out of scope; argue for an exception in a
   hypothesis and wait for the director before implementing.
All restrictions apply at INFERENCE only; training may use anything.

## Reference bars (ratio to full power, sample-matched on the pinned grid drops)
The baseline `equivariant_mlp_baseline` already scores **~1.19** on a short
initial run (it escapes the full-power floor immediately in this band, unlike
the full-range campaign's baseline, which stayed pinned at exactly 1.000 for
its first several dozen experiments). **Full power = 1.000** by definition.

**QFT reference = 1.485** (mean over the 17 cells; 30 sample-matched drops/cell,
10 QFT iterations -- independently verified converged for this band's hardest
cell, K=10, extended to 60 iterations with <0.1% drift on every column; this
band never touches the sum-rate degenerate objective that required special
handling in the full-range campaign). Per-cell:

| K (users) | min | p10 | p25 |
|---|---|---|---|
| 1 (7) | 1.12 | -- | 1.10 |
| 2 (14) | 1.29 | 1.21 | 1.15 |
| 4 (28) | 1.54 | 1.44 | 1.28 |
| 6 (42) | 1.73 | 1.55 | 1.33 |
| 8 (56) | 2.12 | 1.66 | 1.40 |
| 10 (70) | 2.07 | 1.80 | 1.44 |

Read the structure: QFT's edge over full power grows toward max-min fairness
(min column) and large K -- up to x2.12 at K=8 -- and is more modest but still
real at p25 (x1.10-x1.44). Unlike the full-range campaign's sum-rate column,
every one of these numbers is independently verified converged; trust this
table.

## Why this is hard -- the traps that still apply
1. **Full-power / saturation trap.** Locally, more power raises each user's own
   rate, so naive gradient training can drive every power toward `P_T` and a
   `sigmoid` head can saturate there. The baseline's escape from this trap on
   its FIRST run suggests the narrower, more homogeneous band is a materially
   easier landscape than the full range -- but don't assume the trap is gone;
   watch for it as the agent tries larger architectures.
2. **All-off collapse.** A muted user has rate 0; enough zeros crush the
   smallest-Kq objective. Aggressive muting backfires, especially near min
   (Kq=1, where a single mistake is catastrophic).
3. Exact top-k passes gradient to few users per sample (as few as 1 at min).
   Annealed/softened selection may still help even within this easier band.

## Pre-launch audit (performed before this campaign was released)
A 27-check invariant audit was run over the evaluator, the baseline, and the
harness. It found and fixed one REAL bug worth knowing about, because it is the
kind of thing that is easy to reintroduce:

- **Train/eval Kq convention mismatch.** The grid builds its percentile counts
  with `ceil(frac * K*B)`; the baseline's training sampler used `round(...)`.
  Consequence: at K=2, 6 and 10 the `p25` cell sat just past the sampler's
  reach and received **zero** training mass -- three of the seventeen graded
  cells were evaluated at a Kq the model had never trained on. Fixed by making
  the sampler call the evaluator's own `kq_of()`; every graded cell now gets
  3-57% of the mass. **If you change the Kq sampling scheme, re-check this
  invariant** -- `_band_kq_max()` must equal the largest graded Kq for each K.

Also verified: SLqP monotone in Kq; SLqP decreases as all powers scale down
(this system is noise-significant, median desired SNR ~-4 dB -- it is NOT
scale-invariant, so do not assume uniform power scaling is free); model API
returns correct shapes and stays in [0, P_T] at off-grid K in {3,7}; identical
code reproduces identical scores; and QFT convergence at 10 iterations for this
band's hardest cells (verified to 60 iterations, <0.1% drift).

## Rules
- **Edit only `train.py`.** `prepare.py` is hashed and verified every iteration.
- Print `FAMILY <tag>` and `HELDOUT_SCORE <float>`.
- Respect `STEPS`/`BATCH`/`POOL` env budgets (~1-2 min/run).
- Model classes module-level and no-arg constructible.
- One hypothesis per experiment, briefly stated.

## Protocol -- lessons from two prior campaigns
The full-range campaign over-explored early (17 architectures, one experiment
each, before any depth-tuning) and separately over-narrowed its training
distribution to exactly the graded grid points late (which inflated its
headline score while quietly destroying generalization to untested settings --
confirmed by an off-grid check). Both failure modes are avoidable here:

- **Breadth first, but capped.** Implement AT MOST 6 genuinely different
  (framework x architecture) families before settling into depth-tuning one.
  New families get a 5-iteration grace window -- use it for real tuning, not
  just to justify trying a 7th idea.
- **Train broadly WITHIN the band.** The provided baseline already samples Kq
  continuously across the full 0-25% range during training (not just the 3
  graded points) -- keep this property. If you narrow the training
  distribution for efficiency, verify off-grid generalization within the band
  (e.g., percentile fractions like 5%, 15%, 20% never in the graded grid, and
  K in {3,5,7,9} never in `KS_TEST`) before trusting a resulting score jump.
- Check `log.csv` / `leaderboard.csv` before repeating an idea. `final_eval.py`
  re-scores the winner on fresh channels and reports ms/drop vs QFT.
