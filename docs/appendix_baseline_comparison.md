# Appendix: Supplementary Baseline Comparison

**Not part of the thesis experimental scope.** Thesis §1.7 (Scope of the
Study) states this study "is experimentally limited to the Random Forest
algorithm" and that Logistic Regression, SVM, kNN, Naive Bayes, Decision
Trees, and ANN "are considered in the literature review but are not
experimentally implemented or compared as part of this study." The
comparison below (`src/analysis/benchmark.py`) does experimentally implement and
compare Logistic Regression and a Decision Tree against Random Forest, so
it is documented here, separately from the thesis's main results, rather
than folded into `docs/report.md` or `docs/model_card.md`.

It is kept, not deleted, because it is a legitimate and useful piece of
supplementary analysis — see `DECISION REQUIRED` in the repository's
commit history / PR description for the open question of whether a future
revision of the thesis scope paragraph should incorporate it properly.

## What was run

`src/analysis/benchmark.py` runs Random Forest, Logistic Regression, and a single
Decision Tree under an identical nested cross-validation protocol (same
outer/inner folds, same seed, same scoring) via
`src/core/evaluation.py::nested_cv_compare`, so any difference between them reflects
the algorithms, not an uneven evaluation.

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 83.5% ± 3.9% | 81.4% ± 4.7% | 0.908 ± 0.027 |
| Logistic Regression | 83.8% ± 3.4% | 81.8% ± 4.1% | 0.909 ± 0.017 |
| Decision Tree | 78.5% ± 6.2% | 76.5% ± 6.5% | 0.792 ± 0.063 |

(mean ± std across outer CV folds; see `output/model_comparison.json` for
exact per-fold figures and best hyperparameters)

Logistic Regression is statistically indistinguishable from Random
Forest on every metric here, with lower fold-to-fold variance and
substantially more interpretability (a coefficient per feature vs. an
ensemble of trees). A single Decision Tree underperforms both, as
expected. At n=303 with 13 largely monotonic clinical predictors, Random
Forest's added flexibility does not translate into a measurable
performance advantage over a linear model on this dataset.

## Why this is presented as supplementary rather than a study finding

The thesis's single-algorithm scope (§1.7) is a deliberate methodological
boundary, not an oversight — it keeps the study's contribution focused on
a complete, rigorously-evaluated pipeline for one algorithm rather than a
shallower sweep across several. Reporting this comparison as a thesis
*finding* would contradict that stated scope. Reporting it here, labelled
clearly as supplementary and outside the stated scope, preserves both: the
thesis's scope commitment stays intact, and the analysis remains available
to anyone extending this work (see the recommendation in
`docs/model_card.md`).
