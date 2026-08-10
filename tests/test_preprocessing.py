import pandas as pd
import pytest

from preprocessing import (
    EXPECTED_COLUMNS, EXPECTED_ROWS, FEATURE_NAMES, clean_data, validate_raw_data,
)


def test_raw_data_shape(raw_df):
    assert raw_df.shape == (EXPECTED_ROWS, len(EXPECTED_COLUMNS))


def test_validate_raw_data_accepts_real_data(raw_df):
    validate_raw_data(raw_df)  # should not raise


def test_validate_raw_data_rejects_wrong_row_count(raw_df):
    truncated = raw_df.iloc[:-1]
    with pytest.raises(ValueError, match="rows"):
        validate_raw_data(truncated)


def test_validate_raw_data_rejects_out_of_range_value(raw_df):
    corrupted = raw_df.copy()
    corrupted.loc[corrupted.index[0], "cp"] = 9  # cp must be in {1, 2, 3, 4}
    with pytest.raises(ValueError, match="cp"):
        validate_raw_data(corrupted)


def test_validate_raw_data_rejects_missing_column(raw_df):
    dropped = raw_df.drop(columns=["thal"])
    with pytest.raises(ValueError, match="thal"):
        validate_raw_data(dropped)


def test_clean_data_target_is_binary(clean_df):
    assert set(clean_df["target"].unique()) <= {0, 1}
    assert (clean_df["target"] == (clean_df["diagnosis"] > 0).astype(int)).all()


def test_clean_data_leaves_missing_values_unimputed(clean_df):
    # ca/thal carry '?' missing markers in the raw Cleveland CSV; if this
    # count ever hits 0, something upstream started imputing before the
    # split again, or the source CSV changed.
    assert clean_df[FEATURE_NAMES].isna().sum().sum() > 0


def test_get_features_and_target_shapes(features_target):
    X, y = features_target
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == len(y) == EXPECTED_ROWS


def test_clean_data_treats_implausible_zero_chol_and_trestbps_as_missing(raw_df):
    # Switzerland/VA cohorts encode missing cholesterol as a literal 0
    # (physiologically impossible) rather than '?'. Simulate that here
    # since Cleveland itself has no such rows.
    corrupted = raw_df.copy()
    corrupted.loc[corrupted.index[0], "chol"] = 0
    corrupted.loc[corrupted.index[1], "trestbps"] = 0
    clean = clean_data(corrupted)
    assert pd.isna(clean.loc[clean.index[0], "chol"])
    assert pd.isna(clean.loc[clean.index[1], "trestbps"])


def test_clean_data_leaves_genuine_zero_values_alone(raw_df):
    # fbs/exang/sex are legitimately 0/1 -- clean_data's zero->NaN rule is
    # scoped to chol/trestbps only and must not touch these.
    clean = clean_data(raw_df)
    assert (clean["fbs"] == 0).sum() > 0
    assert (clean["sex"] == 0).sum() > 0
