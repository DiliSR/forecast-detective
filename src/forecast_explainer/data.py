"""Load the UCI ElectricityLoadDiagrams20112014 dataset and turn it into hourly load series.

Raw format: 15-minute kW readings for 370 Portuguese clients, ';'-separated with decimal
commas. Each timestamp marks the *end* of its 15-minute interval.
"""

import urllib.request
import zipfile

import pandas as pd

from forecast_explainer import DATA_DIR

RAW_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
RAW_ZIP = DATA_DIR / "raw" / "electricityloaddiagrams20112014.zip"
RAW_TXT = DATA_DIR / "raw" / "LD2011_2014.txt"
HOURLY_PARQUET = DATA_DIR / "processed" / "hourly_kwh.parquet"

# Clients must have data over this whole window to be part of the "total" series;
# many clients only start reporting in 2012-2014, which would create level shifts.
FULL_COVERAGE_START = "2012-01-01"
FULL_COVERAGE_END = "2014-12-01"


def download_raw() -> None:
    """Fetch the zip from UCI (ucimlrepo can't import dataset 321) and extract the txt."""
    if RAW_TXT.exists():
        return
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_ZIP.exists():
        urllib.request.urlretrieve(RAW_URL, RAW_ZIP)
    with zipfile.ZipFile(RAW_ZIP) as zf:
        zf.extract("LD2011_2014.txt", RAW_TXT.parent)


def load_15min() -> pd.DataFrame:
    download_raw()
    return pd.read_csv(RAW_TXT, sep=";", decimal=",", index_col=0, parse_dates=True)


def to_hourly_kwh(df_15min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15-minute kW readings into hourly kWh, labelled by the hour's start."""
    # Repair first: the bad rows straddle two hourly buckets once shifted.
    # Shift so 00:15..01:00 (end-labelled) falls into the 00:00 bucket; kW * 0.25h = kWh.
    return fix_dst(df_15min).shift(freq="-15min").resample("h").sum() / 4


def fix_dst(df_15min: pd.DataFrame) -> pd.DataFrame:
    """Repair the dataset's daylight-saving artefacts in the raw 15-minute readings.

    The meters log on local clock time with a fixed 96 rows per day. On the last Sunday of March
    (clocks jump 01:00 -> 02:00) the rows labelled 01:00, 01:15, 01:30 and 01:45 are ~zero for
    every client; we interpolate them from 00:45 and 02:00. On the last Sunday of October (01:00-02:00
    happens twice) the same four rows hold two hours of load; we halve them.
    """
    df = df_15min.copy()
    for year in sorted(set(df.index.year)):
        for month in (3, 10):
            start = _last_sunday(year, month) + pd.Timedelta(hours=1)
            bad = pd.date_range(start, periods=4, freq="15min")
            if not bad.isin(df.index).all():
                continue
            if month == 3:
                window = slice(start - pd.Timedelta(minutes=15), start + pd.Timedelta(hours=1))
                df.loc[bad] = float("nan")
                df.loc[window] = df.loc[window].interpolate(method="time")
            else:
                df.loc[bad] = df.loc[bad] / 2
    return df


def _last_sunday(year: int, month: int) -> pd.Timestamp:
    month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return month_end - pd.Timedelta(days=(month_end.dayofweek + 1) % 7)


def load_hourly(refresh: bool = False) -> pd.DataFrame:
    """Hourly kWh per client for 2011-2014, cached as parquet."""
    if HOURLY_PARQUET.exists() and not refresh:
        return pd.read_parquet(HOURLY_PARQUET)
    hourly = to_hourly_kwh(load_15min())
    # The last raw reading (2015-01-01 00:00) spills into a partial 2014-12-31 23:00 hour only.
    hourly = hourly.loc[: "2014-12-31 23:00"]
    HOURLY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(HOURLY_PARQUET)
    return hourly


def full_coverage_clients(hourly: pd.DataFrame) -> list[str]:
    active = hourly > 0
    first_reading = active.idxmax()
    last_reading = active[::-1].idxmax()
    keep = (first_reading <= FULL_COVERAGE_START) & (last_reading >= FULL_COVERAGE_END)
    return keep[keep].index.tolist()


def load_series(client: str = "total", start: str = FULL_COVERAGE_START) -> pd.Series:
    """Hourly load to forecast: one client (e.g. 'MT_320') or 'total' over full-coverage clients."""
    hourly = load_hourly()
    if client == "total":
        series = hourly[full_coverage_clients(hourly)].sum(axis=1)
    else:
        series = hourly[client]
    series = series.loc[start:].asfreq("h")
    series.name = "load_kwh"
    return series
