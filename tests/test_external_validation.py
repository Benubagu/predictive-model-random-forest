import config
from external_validation import load_cohort
from preprocessing import FEATURE_NAMES


def test_external_cohorts_load_with_matching_schema():
    for name in config.EXTERNAL_COHORTS:
        X, y = load_cohort(name)
        assert list(X.columns) == FEATURE_NAMES
        assert set(y.unique()) <= {0, 1}
        assert len(X) == len(y) > 0


def test_external_cohort_row_counts_match_known_uci_sizes():
    counts = {name: len(load_cohort(name)[1]) for name in config.EXTERNAL_COHORTS}
    assert counts == {"hungarian": 294, "switzerland": 123, "va": 200}
