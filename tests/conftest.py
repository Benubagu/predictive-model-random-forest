import pytest

from config import DATA_PATH
from preprocessing import clean_data, get_features_and_target, load_raw_data


@pytest.fixture(scope="session")
def raw_df():
    return load_raw_data(DATA_PATH)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def features_target(clean_df):
    return get_features_and_target(clean_df)
