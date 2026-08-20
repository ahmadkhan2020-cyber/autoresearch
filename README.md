# Agentic AI Autoresearch for Cell-Edge Power Control

Code, evaluator and campaign log for:

> A. A. Khan, A. Bin Sediq, S. Azadegi Naeini and R. S. Adve, "Agentic AI
> Autoresearch for Cell-Edge Power Control: Radically Redefining the
> Researcher's Role."

An AI coding agent was given an immutable evaluator and a research charter, and
authority over the architecture, input representation, output parameterization,
loss function and task-sampling law of a learned power-control model. Over
roughly eighty unattended experiments spanning six architecture families and
twenty-six hours, it closed **94% of the gap** between its own first working
architecture and a converged fractional-programming reference on
sum-least-percentile-rate power control — a problem that is non-convex,
non-smooth and strongly NP-hard away from its max-min vertex.

The repository has two halves, and you probably want one of them:

### → [`verify/`](verify/) — check the claim

One command reproduces the paper's held-out number.

```bash
pip install -r requirements.txt
cd verify && python verify.py --qft
```

(Add `verify/train.py` and, optionally, `verify/last_model.pt` first — see
[`MANIFEST.md`](MANIFEST.md).)

```
HELDOUT_SCORE : 1.477473      [paper: 1.4775]
QFT_SCORE     : 1.4850        [paper: 1.4850]
ratio         : 99.5%
```

### → [`autoresearch/`](autoresearch/) — run your own campaign

The protocol, the charter, and the seed script. Start with
[`autoresearch/PROTOCOL.md`](autoresearch/PROTOCOL.md).

---

## The evaluator is the whole argument

`verify/prepare.py` is **immutable**. The agent could import it and never edit
it, and its hash was verified every iteration. That is what makes the campaign's
numbers mean anything: a self-improving system that can edit its own test will
eventually make the test easier instead of the model better.

Confirm the judge you were shipped is the one the campaign was scored against:

```bash
sha256sum verify/prepare.py
```

```
SHA-256: 7d43c742c4d8fd76f304240562018b7c913d423c5d43255a06ea0b2c2c0af95a
```

The held-out set is **not shipped as data**. `prepare.py` regenerates it from
pinned seeds (`TEST` seeds 5000–5010), disjoint from the training stream
(20,000,000+) and the pool (1000–1520). There is no curated file to distrust.

## The result in one table

| | Score | Notes |
|---|---|---|
| Full-power floor | 1.0000 | the trivial baseline the metric is normalised against |
| First working architecture (exp 1) | 1.3687 | equivariant cell-coordinated MPNN |
| **Champion (exp 81)** | **1.4775** | one fixed feed-forward pass |
| QFT reference | 1.4850 | converged, sample-matched on identical pinned drops |

An independent reproduction on separate hardware recomputed the QFT bar from
scratch at **1.485645** and confirmed the exactness result against the CVXPY
solver; see [`verify/VERIFICATION.md`](verify/VERIFICATION.md).

At the minimum percentile the model's output is the exact max-min optimum
**for every value of its trained weights** — an algebraic identity, not a
learned property. See Theorem 1 in the paper; the mechanism is the cut clamp
followed by the balancing recursion.

## Environment

Bit-exact reproduction depends on the torch version and thread count. See
[`requirements.txt`](requirements.txt) and record what you verified under. Small
deviations are expected across environments; the campaign's own noise band was
±0.0005.

## Citing

```bibtex
@inproceedings{khan2026autoresearch,
  author    = {Khan, Ahmad Ali and Bin Sediq, Akram and
               Azadegi Naeini, Sara and Adve, Raviraj S.},
  title     = {Agentic {AI} Autoresearch for Cell-Edge Power Control:
               Radically Redefining the Researcher's Role},
  booktitle = {IEEE Global Communications Conference (GLOBECOM) Workshops},
  year      = {2026}
}
```

## License

GNU GPLv3 (General Public License)
