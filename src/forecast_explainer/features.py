"""Day-ahead features for hourly load.

Every load-derived feature uses data at least 24 hours old, so a forecast for any hour of
tomorrow can be made with today's data. Names are kept readable because they end up in
explanations shown to business users.
"""

import holidays
import pandas as pd

MIN_LAG = 24
LAGS = [24, 25, 48, 72, 168, 336]  # hours
SAME_HOUR_WEEKS = 4  # average of the same hour over the last 4 weeks

# In this dataset most public holidays move load by only +-10% vs the same day last week, but
# Christmas drops it ~40% every year (2012-2014), and New Year's Day is similarly low.
MAJOR_HOLIDAYS = ["12-25", "01-01"]

# Fixed category sets so a single row (e.g. the one hour being explained) encodes the same
# way as the training data.
CATEGORICAL_FEATURES = {"hour": range(24), "day_of_week": range(7), "month": range(1, 13)}


def make_features(load: pd.Series) -> pd.DataFrame:
    """Return a feature frame aligned with `load`'s index (rows with incomplete history are NaN)."""
    X = pd.DataFrame(index=load.index)

    for lag in LAGS:
        X[f"load_lag_{lag}h"] = load.shift(lag)

    known = load.shift(MIN_LAG)
    X["load_mean_24h"] = known.rolling(24).mean()
    X["load_std_24h"] = known.rolling(24).std()
    X["load_mean_7d"] = known.rolling(168).mean()
    X["load_same_hour_mean_4w"] = sum(load.shift(168 * w) for w in range(1, SAME_HOUR_WEEKS + 1)) / SAME_HOUR_WEEKS

    cal = calendar_features(load.index)
    # A holiday inside a lag window makes that lag look unusually low; flag it so the model
    # (and the explanation) can tell "last week was Christmas" apart from a real downturn.
    for lag in (24, 168):
        X[f"lag_{lag}h_was_holiday"] = cal["is_holiday"].shift(lag)
        X[f"lag_{lag}h_was_major_holiday"] = cal["is_major_holiday"].shift(lag)

    X = X.join(cal)
    return encode_categoricals(X)


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col, categories in CATEGORICAL_FEATURES.items():
        X[col] = pd.Categorical(X[col], categories=list(categories))
    return X


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    pt_holidays = holidays.Portugal(years=range(index.year.min(), index.year.max() + 1))
    holiday_dates = pd.DatetimeIndex(list(pt_holidays.keys()))
    dates = index.normalize()
    day_of_week = index.dayofweek

    cal = pd.DataFrame(index=index)
    cal["hour"] = index.hour
    cal["day_of_week"] = day_of_week
    cal["month"] = index.month
    cal["is_weekend"] = (day_of_week >= 5).astype(int)
    cal["is_holiday"] = dates.isin(holiday_dates).astype(int)
    cal["is_major_holiday"] = index.strftime("%m-%d").isin(MAJOR_HOLIDAYS).astype(int)
    # Portuguese "pontes": a Monday before a Tuesday holiday or a Friday after a Thursday one.
    next_day_holiday = (dates + pd.Timedelta(days=1)).isin(holiday_dates)
    prev_day_holiday = (dates - pd.Timedelta(days=1)).isin(holiday_dates)
    cal["is_bridge_day"] = (((day_of_week == 0) & next_day_holiday) | ((day_of_week == 4) & prev_day_holiday)).astype(int)
    return cal
