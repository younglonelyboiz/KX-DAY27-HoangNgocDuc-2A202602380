# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**

- **Subtle distribution shifts & AI infrastructure drift** (`distribution_shift`, `embedding_drift`, `feature_skew`, `corpus_staleness`):
  - Subtle faults sit inside the 2.0σ–2.8σ boundary rather than obvious >3σ excursions.
  - Relying solely on static 3σ baseline thresholds misses these subtle faults (reducing TPR). Conversely, loosening static thresholds naively causes false positives on clean tail-variance (FPR penalty is severe at −0.3).
  - Effective detection required combining multi-signal deviations (e.g., joint z-score thresholds on both volume and mean amount) and calibrating drift boundaries just above clean stream maximums.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

- With 300 credits spent for full single-pass coverage against the 320 private budget (`cost_overage = 0.0`), the 1-tool-per-event strategy maximized TPR safely without budget penalty.
- On another pass, I would use `ctx.state` to build online rolling statistics (Welford's algorithm) to dynamically calibrate drift boundaries over the stream lifetime, enabling tighter sensitivity without static threshold fragility.
