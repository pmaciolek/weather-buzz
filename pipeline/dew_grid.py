"""Hourly 2 m dew point for the map points (and fallback for the places).

  python3.13 dew_grid.py fetch      # ERA5 yearly global fields at 1°, sampled at the points (resumable)
  python3.13 dew_grid.py fetch-ts   # alternative: per point from the ERA5 time-series service (~10 h)
  python3.13 dew_grid.py assemble   # per-point series -> cache/grid/dew/<lat>_<lon>.npz

The map's temperature, rain and wind came from the per-point ERA5 time-series
service (~1 min per point). Dew point was added later; fetching it the same way
would take another ~11 h, so it comes from gridded ERA5 instead: one request per
year for the whole land band at 1°, bilinearly sampled at each point and then
discarded. Dew point varies smoothly, so 1° is close enough for the score.
"""
import datetime as dt
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config as C
from build_grid import LAT_MAX, LAT_MIN, WX_DIR, _f, _i16

DIR = C.CACHE / "grid" / "dew"
MONTHS_DIR = DIR / "months"
TMP = C.CACHE / "grid" / "tmp"


def points():
    """Map points plus the places, as an (n, 2) lat/lon array, in a fixed order."""
    pts = sorted(tuple(map(float, p.stem.split("_"))) for p in WX_DIR.glob("*.npz"))
    pts += [(lat, lon) for _, lat, lon in C.CITIES]
    return np.array(pts)


def months():
    today = dt.date.today()
    last = (today.replace(day=1) - dt.timedelta(days=7)).replace(day=1)   # ERA5 runs ~5 days behind
    out, y, m = [], C.BASELINE_YEARS[0], 1
    while (y, m) <= (last.year, last.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_year(client, y, ms, pts):
    """One request per year (queue waits dominate, so fewer, larger requests finish
    sooner); the field is sampled at the points and stored per month."""
    import netCDF4
    todo = [m for m in ms if not (MONTHS_DIR / f"{y}-{m:02d}.npz").exists()]
    if not todo:
        return
    TMP.mkdir(parents=True, exist_ok=True)
    nc = TMP / f"d2m_{y}.nc"
    client.retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": ["2m_dewpoint_temperature"],
        "year": [str(y)], "month": [f"{m:02d}" for m in todo], "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)], "data_format": "netcdf",
        "download_format": "unarchived", "area": [LAT_MAX + 2, -180, LAT_MIN - 2, 180], "grid": [1.0, 1.0]}, str(nc))
    f = netCDF4.Dataset(nc)
    tv = f.variables["valid_time"]
    t = np.array(netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False), dtype="datetime64[h]")
    lat = np.asarray(f.variables["latitude"][:]); lon = np.asarray(f.variables["longitude"][:])
    flip = lat[0] > lat[-1]
    lat = lat[::-1] if flip else lat
    lon = np.where(lon >= 180, lon - 360, lon)
    o = np.argsort(lon); lon = lon[o]
    lonw = np.concatenate([lon, [lon[0] + 360]])
    q = np.column_stack([pts[:, 0], np.where(pts[:, 1] < lon[0], pts[:, 1] + 360, pts[:, 1])])
    month = t.astype("datetime64[M]").astype(int) % 12 + 1
    MONTHS_DIR.mkdir(parents=True, exist_ok=True)
    for m in todo:
        sel = np.where(month == m)[0]
        x = np.asarray(f.variables["d2m"][sel[0]:sel[-1] + 1], dtype=np.float32) - 273.15   # (time, lat, lon)
        x = x[:, ::-1] if flip else x
        x = x[:, :, o]
        x = np.concatenate([x, x[:, :, :1]], axis=2)
        vals = RegularGridInterpolator((lat, lonw), np.moveaxis(x, 0, -1))(q).T   # (time, points)
        out = MONTHS_DIR / f"{y}-{m:02d}.npz"
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, t=t[sel].astype(np.int64).astype(np.int32), d=_i16(vals, 10), pts=pts)
        tmp.replace(out)
    f.close(); nc.unlink()


def fetch(workers=3):
    import cdsapi
    pts = points()
    by_year = {}
    for y, m in months():
        if not (MONTHS_DIR / f"{y}-{m:02d}.npz").exists():
            by_year.setdefault(y, []).append(m)
    print(f"{len(pts)} points, {sum(map(len, by_year.values()))} months in {len(by_year)} yearly requests", flush=True)
    local, lock, done, t0 = threading.local(), threading.Lock(), [0], time.time()

    def job(y):
        if not hasattr(local, "c"):
            local.c = cdsapi.Client(quiet=True, progress=False)
        attempt = 0
        while attempt < 4:
            try:
                fetch_year(local.c, y, by_year[y], pts)
                break
            except Exception as e:
                msg = " ".join(str(e).split())
                if "temporarily limited" in msg:
                    time.sleep(60)   # queue full: wait for a slot, not a failure
                    continue
                attempt += 1
                print(f"  {y}: {type(e).__name__}: {msg[:300]}", flush=True)
                time.sleep(30 * attempt)
        with lock:
            done[0] += 1
            print(f"  {done[0]}/{len(by_year)} {y}  {(time.time() - t0) / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, sorted(by_year)))


def _ts_point(client, lat, lon):
    """Dew point for one point from the ERA5 time-series service (native 0.25° cell)."""
    import csv, io, zipfile
    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / f"d2m_{lat:.2f}_{lon:.2f}.zip"
    client.retrieve("reanalysis-era5-single-levels-timeseries", {
        "variable": ["2m_dewpoint_temperature"], "location": {"latitude": lat, "longitude": lon},
        "date": [f"{C.BASELINE_YEARS[0]}-01-01/{C.BASELINE_YEARS[-1]}-12-31"], "data_format": "csv"}, str(tmp))
    with zipfile.ZipFile(tmp) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), "utf-8")))
    tmp.unlink()
    t = np.array([r["valid_time"].replace(" ", "T") for r in rows], dtype="datetime64[h]")
    d = np.array([float(r["d2m"]) - 273.15 if r["d2m"] else np.nan for r in rows])
    DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DIR / f"{lat:.3f}_{lon:.3f}.npz", t=t.astype(np.int64).astype(np.int32), d=_i16(d, 10), src="ts")


def fetch_ts(workers=5):
    """Per-point alternative to fetch(), used when the gridded ERA5 queue is blocked."""
    import cdsapi
    pts = [tuple(p) for p in points()[:-len(C.CITIES)]]   # places get dew point from Open-Meteo
    todo = [p for p in pts if not (DIR / f"{p[0]:.3f}_{p[1]:.3f}.npz").exists()]
    print(f"{len(pts)} points, {len(todo)} to fetch", flush=True)
    local, lock, done, t0 = threading.local(), threading.Lock(), [0], time.time()

    def job(p):
        if not hasattr(local, "c"):
            local.c = cdsapi.Client(quiet=True, progress=False)
        attempt = 0
        while attempt < 3:
            try:
                _ts_point(local.c, *p)
                break
            except Exception as e:
                msg = " ".join(str(e).split())
                if "temporarily limited" in msg or "Maximum number queued requests" in msg:
                    time.sleep(45)   # queue full: wait for a slot, not a failure
                    continue
                attempt += 1
                print(f"  {p}: {type(e).__name__}: {msg[:300]}", flush=True)
                time.sleep(30 * attempt)
        with lock:
            done[0] += 1
            if done[0] % 20 == 0:
                rate = done[0] / (time.time() - t0) * 3600
                print(f"  {done[0]}/{len(todo)}  {rate:.0f} pts/h  eta {(len(todo) - done[0]) / rate:.1f} h", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, todo))


def assemble():
    files = sorted(MONTHS_DIR.glob("*.npz"))
    zs = [np.load(p) for p in files]
    pts = zs[-1]["pts"]
    # Months fetched before new points were added hold fewer columns; match by coordinates.
    t = np.concatenate([z["t"] for z in zs])
    cols = []
    for z in zs:
        key = {tuple(p): i for i, p in enumerate(z["pts"])}
        idx = np.array([key.get(tuple(p), -1) for p in pts])
        d = z["d"][:, np.maximum(idx, 0)]
        d[:, idx < 0] = -32768
        cols.append(d)
    D = np.concatenate(cols)
    DIR.mkdir(parents=True, exist_ok=True)
    for j, (lat, lon) in enumerate(pts):
        if (DIR / f"{lat:.3f}_{lon:.3f}.npz").exists() and "src" in np.load(DIR / f"{lat:.3f}_{lon:.3f}.npz"):
            continue   # keep the native time-series value
        np.savez_compressed(DIR / f"{lat:.3f}_{lon:.3f}.npz", t=t, d=D[:, j])
    print(f"assembled {len(pts)} points, {len(files)} months ({files[0].stem} .. {files[-1].stem})")


def load_dew(lat, lon):
    """(time UTC datetime64[h], dew point °C) for a point, or None."""
    p = DIR / f"{lat:.3f}_{lon:.3f}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["t"].astype("datetime64[h]"), _f(z["d"], 10)


def align(t_target, t_src, v_src):
    """Values of a series at the given hours (NaN where missing)."""
    out = np.full(len(t_target), np.nan)
    i = np.searchsorted(t_src, t_target)
    ok = (i < len(t_src)) & (t_src[np.minimum(i, len(t_src) - 1)] == t_target)
    out[ok] = v_src[i[ok]]
    return out


if __name__ == "__main__":
    {"fetch": fetch, "fetch-ts": fetch_ts, "assemble": assemble}[sys.argv[1]]()
