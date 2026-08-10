# Literature Comparison (RQ5)

RQ5: *"How does the performance of the Random Forest model developed in
this study compare with results reported in previous studies using the
Cleveland Heart Disease dataset?"*

Citations below match Chapter One/Two of the thesis
(`docs/Heart_Disease_RF_1-2.docx`) exactly.

## Comparison table

| Study | Approach | Reported accuracy | Evaluation method |
|---|---|---|---|
| Singh et al. (2017) | Random Forest, Cleveland dataset (303 instances) | **85.81%** | 10-fold cross-validation |
| Gavhane et al. (2018) | Multiple ML approaches incl. Multilayer Perceptron | Not reported as a single figure in the cited discussion | Not specified in source |
| Katarya & Srinivas (2020) | Multiple ML models; RF "among the consistently strong performers" | Not reported as a single figure in the cited discussion | Not specified in source |
| **This study** | Random Forest, Cleveland dataset (303 instances), tuned via cross-validated grid search | **83.5% ± 3.9%** (95% CI: see below) | **5-outer/5-inner nested cross-validation** |

This study's number is read live from `output/metrics.json` →
`nested_cv_summary.accuracy` / `bootstrap_ci.accuracy` at generation time
(threshold = 0.5, raw pipeline — see `docs/model_card.md` for why that's
the right number to compare against prior "accuracy" figures, and never
the high-sensitivity operating-threshold number).

The wider literature context noted in Chapter Two: *"Random Forest
accuracy on the Cleveland dataset or its close variants typically falls
in the range of 80% to 90%, depending on pre processing choices, cross
validation strategy, and the specific train and test split used, with
reported accuracies as high as the mid 90s in studies that combine Random
Forest with additional feature selection or optimization techniques."*
This study's honest estimate sits within that range, toward its lower
end.

## The methodology caveat (the actual point of RQ5)

Comparing a single point figure against Singh et al.'s 85.81% invites the
wrong question — "why is this study's number lower?" — when the more
important question is **what kind of number each figure is**:

- Singh et al. (2017) report a single accuracy figure from 10-fold
  cross-validation. This is a large step up from a single 80/20 split,
  but it is still a **point estimate with no reported uncertainty
  interval** — there is no way to know from the published figure alone
  whether 85.81% is a stable estimate or one favorable resampling away
  from being meaningfully different.
- This study deliberately goes one level further: **nested** cross-validation
  (an outer loop for the performance estimate, an inner loop — entirely
  separate from the outer loop's test data — for hyperparameter search)
  plus a **95% bootstrap confidence interval** on the pooled out-of-fold
  predictions. The result is not a single number but a number with an
  honest error bar (see `docs/model_card.md` and the H1₁ hypothesis test
  in `output/metrics.json`).
- An earlier version of this project's own pipeline produced an **88.5%**
  single-split accuracy figure — higher than Singh et al.'s 85.81% — before
  this methodology was tightened. That number is superseded and no longer
  reported anywhere in this repository (see git history), precisely
  *because* a single split is exactly the kind of optimistic, unstable
  estimate this discussion is warning about. It is mentioned here, not in
  spite of being an inconvenient data point, but because it is the clearest
  illustration available of why the comparison in this table has to be read
  carefully rather than as a simple ranking.

**Honest conclusion for RQ5**: this study's nested-CV accuracy (≈83.5%,
with a 95% CI whose lower bound sits just under 80% — see the H1₁
hypothesis test) is broadly consistent with, if slightly below, Singh et
al.'s (2017) 85.81% single-cross-validation figure, and sits within the
80-90% range the wider literature reports for Random Forest on this
dataset. The apparent gap is more plausibly explained by measurement
methodology (nested CV with a reported interval vs. a single reported
point estimate) than by a meaningful difference in modelling quality —
which is itself the more interesting and defensible finding to report at
viva than a bare number comparison would be.
