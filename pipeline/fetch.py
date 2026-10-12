"""Open-Meteo client with an on-disk, per-year cache.

Weather: ERA5 family via the historical archive API (hourly, local time).
Air quality: CAMS via the air-quality API (Europe ~2013+, global ~2023+).

Requests are split per calendar year so runs are resumable and only the
current (incomplete) year is re-downloaded, once per day.
"""
import datetime as dt
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from config import AQ_VARS, CACHE, DEW_VARS, WEATHER_VARS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


class DailyLimit(Exception):
    pass


def _get(url, params, retries=4, wait_daily=True):
    """GET with retries. On 429 waits out the quota window (minute/hour/day)
    instead of failing, so long grid runs can be left unattended. With
    wait_daily=False a spent daily quota raises DailyLimit instead."""
    q = url + "?" + urllib.parse.urlencode(params)
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(q, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code != 429:
                raise RuntimeError(f"{e.code} {body[:300]}") from e
            if ("Daily" in body or "Hourly" in body) and not wait_daily:
                raise DailyLimit(body)
            wait = 3600 if "Daily" in body else 600 if "Hourly" in body else 60
            print(f"    429: {body[:100]} (sleeping {wait}s)", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError):
            attempt += 1
            if attempt >= retries:
                raise RuntimeError(f"failed: {q}")
            time.sleep(5 * attempt)


def _cache_path(kind, lat, lon, year):
    today = dt.date.today()
    tag = f"{year}" if year < today.year else f"{year}_{today.isoformat()}"
    return CACHE / kind / f"{lat:.3f}_{lon:.3f}" / f"{tag}.json.gz"


def _year_range(year):
    today = dt.date.today()
    end = min(dt.date(year, 12, 31), today - dt.timedelta(days=1))
    return dt.date(year, 1, 1).isoformat(), end.isoformat()


def fetch_year(kind, lat, lon, year, wait_daily=True):
    """Return the raw Open-Meteo JSON for one location-year (cached)."""
    path = _cache_path(kind, lat, lon, year)
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    start, end = _year_range(year)
    url, hourly = {"weather": (ARCHIVE_URL, WEATHER_VARS), "dew": (ARCHIVE_URL, DEW_VARS), "aq": (AQ_URL, AQ_VARS)}[kind]
    data = _get(url, {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(hourly), "timezone": "auto",
    }, wait_daily=wait_daily)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop stale snapshots of the current year.
    for old in path.parent.glob(f"{year}_*.json.gz"):
        old.unlink()
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    return data


def fetch_batch(kind, points, start, end):
    """Uncached multi-location request; returns one hourly dict per point."""
    url, hourly = (ARCHIVE_URL, WEATHER_VARS) if kind == "weather" else (AQ_URL, AQ_VARS)
    data = _get(url, {
        "latitude": ",".join(f"{p[0]:.3f}" for p in points),
        "longitude": ",".join(f"{p[1]:.3f}" for p in points),
        "start_date": start, "end_date": end,
        "hourly": ",".join(hourly), "timezone": "auto",
    })
    if isinstance(data, dict):
        data = [data]
    return [_to_arrays(d) for d in data]


def _to_arrays(d):
    h = d["hourly"]
    out = {"time": np.array(h["time"], dtype="datetime64[h]")}
    for k, v in h.items():
        if k != "time":
            out[k] = np.array([np.nan if x is None else x for x in v], dtype=float)
    out["meta"] = {k: d[k] for k in ("latitude", "longitude", "elevation", "utc_offset_seconds", "timezone") if k in d}
    return out


def load_hourly(kind, lat, lon, years, wait_daily=True):
    """Concatenate years into numpy arrays: dict(time=datetime64[h], var=float array)."""
    times, cols, meta = [], {}, None
    for y in years:
        d = fetch_year(kind, lat, lon, y, wait_daily)
        meta = meta or {k: d[k] for k in ("latitude", "longitude", "elevation", "utc_offset_seconds", "timezone") if k in d}
        h = d["hourly"]
        times.append(np.array(h["time"], dtype="datetime64[h]"))
        for k, v in h.items():
            if k != "time":
                cols.setdefault(k, []).append(np.array([np.nan if x is None else x for x in v], dtype=float))
    out = {"time": np.concatenate(times)}
    out.update({k: np.concatenate(v) for k, v in cols.items()})
    out["meta"] = meta
    return out
