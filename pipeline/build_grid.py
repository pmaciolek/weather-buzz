"""Global land grid: fetch raw hourly data (resumable), then score + interpolate.

  python3.13 build_grid.py fetch-wx [--step 2] [--workers 8] [--limit N]   # ERA5 via Copernicus CDS
  python3.13 build_grid.py fetch-aq [--step 4] [--limit N]                 # CAMS via Open-Meteo
  python3.13 build_grid.py render

Weather: ERA5 hourly 2016-2025 per point from the CDS "ERA5 hourly time-series"
service (one request per point, ~1 min each, run in parallel).
Air quality: CAMS global only exists from late 2022 and only Open-Meteo serves
it per point, so it is sampled on a coarser grid and the monthly probability of
a poor-AQ day is interpolated onto the weather points as an expected penalty.
Raw series are stored per point so scoring rules can change without
re-downloading. The key cities (cities.json) are blended in as extra samples.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from global_land_mask import globe
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

import config as C
from fetch import fetch_batch
from score import daily_scores, monthly_climatology

WX_DIR = C.CACHE / "grid" / "wx"
AQ_DIR = C.CACHE / "grid" / "aq"
LEGACY_DIR = C.CACHE / "grid" / "raw"
CLIM_CACHE = C.CACHE / "grid" / "clim_cache.json"
TMP = C.CACHE / "grid" / "tmp"
AQ_RANGE = ("2024-01-01", "2025-12-31")
CDS_DATASET = "reanalysis-era5-single-levels-timeseries"
LAT_MIN, LAT_MAX = -56, 72
RASTER_RES = 1 / 6          # display raster (~18 km); the samples themselves are WX_STEP apart
WX_STEP = 2.0
CHANGE_RANGE = 0.05      # projected change in day score mapped onto the full diverging ramp


def grid_points(step):
    """Land points on a regular grid, snapped to the ERA5 0.25° grid. Coastal
    cells are moved onto the nearest land sample so they aren't scored over sea."""
    pts = []
    sub = np.linspace(-step / 2, step / 2, 5)
    for la in np.arange(LAT_MIN + step / 2, LAT_MAX, step):
        for lo in np.arange(-180 + step / 2, 180, step):
            LA, LO = np.meshgrid(la + sub, lo + sub)
            land = globe.is_land(np.clip(LA, -89.9, 89.9), ((LO + 180) % 360) - 180)
            if land.mean() >= 0.25:
                cand = np.argwhere(land)
                c = tuple(cand[np.argmin(np.abs(cand - 2).sum(axis=1))])
                snap = lambda x: round(round(float(x) * 4) / 4, 2)
                pts.append((snap(LA[c]), snap(((LO[c] + 180) % 360) - 180)))
    pts = list(dict.fromkeys(pts))
    # Shuffled so a partial run already covers the whole globe sparsely.
    np.random.default_rng(0).shuffle(pts)
    return pts


def _path(d, lat, lon):
    return d / f"{lat:.2f}_{lon:.2f}.npz"


def _i16(x, scale):
    return np.where(np.isnan(x), -32768, np.round(x * scale)).astype(np.int16)


def _f(x, scale):
    return np.where(x == -32768, np.nan, x / scale)


# --- weather: Copernicus CDS ---------------------------------------------------

def _cds_point(client, lat, lon, out=None, end=None):
    """Download one point's hourly ERA5 series (2016 → end) from the CDS and store it."""
    TMP.mkdir(parents=True, exist_ok=True)
    tmp = TMP / f"{lat:.2f}_{lon:.2f}.zip"
    client.retrieve(CDS_DATASET, {
        "variable": ["2m_temperature", "total_precipitation",
                     "10m_u_component_of_wind", "10m_v_component_of_wind"],
        "location": {"latitude": lat, "longitude": lon},
        "date": [f"{C.BASELINE_YEARS[0]}-01-01/{end or f'{C.BASELINE_YEARS[-1]}-12-31'}"],
        "data_format": "csv",
    }, str(tmp))
    with zipfile.ZipFile(tmp) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), "utf-8")))
    tmp.unlink()
    t = np.array([r["valid_time"].replace(" ", "T") for r in rows], dtype="datetime64[h]")
    col = lambda k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
    T = col("t2m") - 273.15
    P = np.maximum(col("tp"), 0) * 1000                   # m -> mm (hourly accumulation)
    W = np.hypot(col("u10"), col("v10")) * 3.6            # m/s -> km/h
    np.savez_compressed(out or _path(WX_DIR, lat, lon), t=t.astype(np.int64).astype(np.int32),
                        T=_i16(T, 10), P=_i16(P, 100), W=_i16(W, 10),
                        cell=np.array([float(rows[0]["latitude"]), float(rows[0]["longitude"])]))


def fetch_wx(step, workers, limit):
    import cdsapi
    pts = grid_points(step)
    WX_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in pts if not _path(WX_DIR, *p).exists()]
    todo = missing[:limit]
    print(f"{len(pts)} land points at {step}°, {len(pts) - len(missing)} cached, fetching {len(todo)} "
          f"with {workers} workers", flush=True)
    local = threading.local()
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def job(p):
        if not hasattr(local, "client"):
            local.client = cdsapi.Client(quiet=True, progress=False)
        attempt = 0
        while attempt < 3:
            try:
                _cds_point(local.client, *p)
                break
            except Exception as e:
                msg = " ".join(str(e).split())
                if "temporarily limited" in msg:
                    time.sleep(45)  # dataset queue full: wait our turn, doesn't count as a failure
                    continue
                attempt += 1        # other errors: retry a few times, then skip (a rerun picks it up)
                print(f"    {p}: {type(e).__name__}: {msg[:400]}", flush=True)
                time.sleep(30 * attempt)
        with lock:
            done[0] += 1
            n = done[0]
            if n % 10 == 0:
                rate = n / (time.time() - t0) * 3600
                print(f"  {n}/{len(todo)}  {rate:.0f} pts/h  eta {(len(todo) - n) / rate:.1f} h", flush=True)
            if n % 200 == 0:
                render()

    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(job, p) for p in todo]):
            f.result()
    render()


def load_wx(path, lon, lat=None):
    """Hourly series for a point; with lat, also the gridded ERA5 dew point if fetched."""
    from dew_grid import align, load_dew
    z = np.load(path)
    # ERA5 is UTC; shift to local *solar* time (nearest whole hour). The riding
    # window is about daylight, so solar time is the right clock here.
    off = int(round(lon / 15)) * 3600
    t_utc = z["t"].astype("datetime64[h]")
    out = {"time": t_utc + np.timedelta64(off // 3600, "h"), "temperature_2m": _f(z["T"], 10),
           "precipitation": _f(z["P"], 100), "wind_speed_10m": _f(z["W"], 10), "meta": {"utc_offset_seconds": off}}
    dew = load_dew(lat, lon) if lat is not None else None
    if dew is not None:
        out["dew_point_2m"] = align(t_utc, *dew)
    return out


# --- air quality: Open-Meteo (CAMS) --------------------------------------------

def _save_aq(lat, lon, t, aqi):
    np.savez_compressed(_path(AQ_DIR, lat, lon), t=t.astype("datetime64[h]").astype(np.int64).astype(np.int32),
                        aqi=_i16(aqi, 1))


def migrate_legacy_aq():
    """Earlier Open-Meteo grid runs stored AQ alongside weather; reuse it."""
    if not LEGACY_DIR.exists():
        return
    AQ_DIR.mkdir(parents=True, exist_ok=True)
    for p in LEGACY_DIR.glob("*.npz"):
        lat, lon = map(float, p.stem.split("_"))
        if not _path(AQ_DIR, lat, lon).exists():
            z = np.load(p)
            _save_aq(lat, lon, z["aq_t"].astype("datetime64[h]"), _f(z["aqi"], 1))


def fetch_aq(step, limit):
    migrate_legacy_aq()
    have = [tuple(map(float, p.stem.split("_"))) for p in AQ_DIR.glob("*.npz")]
    tree = cKDTree(np.array(have)) if have else None
    pts = grid_points(step)
    # Skip grid points already covered by a migrated sample within half a cell.
    todo = [p for p in pts if tree is None or tree.query(p)[0] > step / 2][:limit]
    print(f"{len(pts)} AQ points at {step}°, {len(have)} cached, fetching {len(todo)}", flush=True)
    for i in range(0, len(todo), 10):
        batch = todo[i:i + 10]
        for (lat, lon), a in zip(batch, fetch_batch("aq", batch, *AQ_RANGE)):
            _save_aq(lat, lon, a["time"], a["european_aqi"])
        print(f"  {i + len(batch)}/{len(todo)}", flush=True)


def aq_probability(path):
    """Per calendar month: share of days with daytime (09-19) mean EAQI >= veto."""
    z = np.load(path)
    t = z["t"].astype("datetime64[h]"); a = _f(z["aqi"], 1)
    h = (t - t.astype("datetime64[D]")).astype(int)
    sel = (h >= 9) & (h < 19) & ~np.isnan(a)
    days, idx = np.unique(t[sel].astype("datetime64[D]"), return_inverse=True)
    s = np.bincount(idx, a[sel]); c = np.bincount(idx)
    ok = c >= 6
    veto = (s[ok] / c[ok]) >= C.AQI_VETO
    month = days[ok].astype("datetime64[M]").astype(int) % 12
    return np.array([veto[month == m].mean() if (month == m).any() else np.nan for m in range(12)])


# --- scoring + raster ----------------------------------------------------------

def _atomic_write(path, text):
    # The fetcher and a manual render can run at once; never leave a half-written file.
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _rules_hash(extra=""):
    h = hashlib.sha1(extra.encode())
    for f in ("score.py", "config.py", "projection.py"):
        h.update((C.ROOT / "pipeline" / f).read_bytes())
    return h.hexdigest()[:12]


def score_points():
    import projection
    from dew_grid import DIR as DEW_DIR
    models = projection.models_available()
    # Recompute when the rules, the projection models or the dew-point coverage change.
    rh = _rules_hash(",".join(models) + f"|dew{len(list(DEW_DIR.glob('*.npz'))) if DEW_DIR.exists() else 0}")
    cache = json.loads(CLIM_CACHE.read_text()) if CLIM_CACHE.exists() else {}
    cache = {k: v for k, v in cache.items() if v["h"] == rh}
    paths = sorted(WX_DIR.glob("*.npz"))
    ll = np.array([list(map(float, p.stem.split("_"))) for p in paths])
    deltas = projection.deltas_at(ll[:, 0], ll[:, 1]) if models and len(ll) else None
    rows = []
    for i, path in enumerate(paths):
        lat, lon = ll[i]
        if path.stem not in cache:
            wx = load_wx(path, lon, lat)
            clim = monthly_climatology(daily_scores(wx, None, lat, lon), C.BASELINE_YEARS)
            fut = None
            if deltas is not None:
                fut = monthly_climatology(daily_scores(projection.apply_delta(wx, deltas[0][i], deltas[1][i]), None, lat, lon),
                                          C.BASELINE_YEARS)
            cache[path.stem] = {"h": rh, "clim": clim, "fut": fut}
        rows.append({"key": [lat, lon], "clim": cache[path.stem]["clim"], "fut": cache[path.stem]["fut"]})
        if i % 500 == 499:
            print(f"  scored {i + 1}/{len(paths)}", flush=True)
    _atomic_write(CLIM_CACHE, json.dumps(cache, separators=(",", ":")))

    # AQ penalty from the (coarser) AQ samples, inverse-distance within ~6°.
    aq_files = sorted(AQ_DIR.glob("*.npz")) if AQ_DIR.exists() else []
    if aq_files and rows:
        aq_ll = np.array([list(map(float, p.stem.split("_"))) for p in aq_files])
        aq_p = np.array([aq_probability(p) for p in aq_files])
        tree = cKDTree(_xyz(aq_ll[:, 0], aq_ll[:, 1]))
        wl = np.array([r["key"] for r in rows])
        dist, idx = tree.query(_xyz(wl[:, 0], wl[:, 1]), k=min(4, len(aq_files)))
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
        w = np.where(dist < np.radians(6), 1 / np.maximum(dist, 1e-6) ** 2, 0)
        for i, r in enumerate(rows):
            if w[i].sum() == 0:
                continue
            p = np.nansum(w[i][:, None] * aq_p[idx[i]], axis=0) / w[i].sum()
            for key in ("clim", "fut"):
                if r[key] is None:
                    continue
                r[key] = [dict(cm, aq_veto_frac=round(float(p[m]), 4), score=round(cm["score"] * (1 - float(p[m])), 4))
                          for m, cm in enumerate(r[key])]

    cities = C.WEB_DATA / "cities.json"
    if cities.exists():
        for c in json.loads(cities.read_text())["cities"]:
            rows.append({"key": [c["lat"], c["lon"]], "clim": c["climatology"], "fut": c.get("climatology_future"), "city": c["name"]})
    return rows


def render():
    rows = score_points()
    n_grid = sum(1 for r in rows if "city" not in r)
    lat = np.array([r["key"][0] for r in rows]); lon = np.array([r["key"][1] for r in rows])
    print(f"rendering {n_grid} grid points + {len(rows) - n_grid} cities", flush=True)

    rlat = np.arange(LAT_MAX - RASTER_RES / 2, LAT_MIN, -RASTER_RES)
    rlon = np.arange(-180 + RASTER_RES / 2, 180, RASTER_RES)
    LO, LA = np.meshgrid(rlon, rlat)
    # Values are computed for every cell (sea too); the page clips the raster to the
    # vector coastline, so coasts stay sharp at any zoom.
    land = np.ones(LA.shape, dtype=bool)
    qlon, qlat = LO[land], LA[land]

    # Layers: present and projected score / ideal share, and the projected change in each.
    keys = ("score", "ideal_frac", "score_f", "ideal_frac_f", "change", "change_i")
    fut = lambda r: r["fut"] or r["clim"]
    vals = np.array([[m["score"] for m in r["clim"]] + [m["ideal_frac"] for m in r["clim"]]
                     + [m["score"] for m in fut(r)] + [m["ideal_frac"] for m in fut(r)]
                     + [f["score"] - m["score"] for m, f in zip(r["clim"], fut(r))]
                     + [f["ideal_frac"] - m["ideal_frac"] for m, f in zip(r["clim"], fut(r))] for r in rows], dtype=float)  # (points, 72)

    # Linear interpolation over a Delaunay mesh in lon/lat gives smooth gradients at
    # fine raster resolution (IDW makes bull's-eyes). Samples near the antimeridian are
    # copied across it so the mesh wraps around the globe.
    wrap_e, wrap_w = lon > 150, lon < -150
    mlon = np.concatenate([lon, lon[wrap_e] - 360, lon[wrap_w] + 360])
    mlat = np.concatenate([lat, lat[wrap_e], lat[wrap_w]])
    mval = np.concatenate([vals, vals[wrap_e], vals[wrap_w]])
    V = LinearNDInterpolator(np.column_stack([mlon, mlat]), mval)(qlon, qlat)

    # Cells far from any sample (not fetched yet) stay "no data"; cells inside the
    # radius but outside the mesh fall back to inverse-distance weighting.
    tree = cKDTree(_xyz(lat, lon))
    dist, idx = tree.query(_xyz(qlat, qlon), k=min(6, len(rows)))
    near = dist[:, 0] <= np.radians(WX_STEP * 1.75)
    miss = near & np.isnan(V).any(axis=1)
    if miss.any():
        w = 1 / np.maximum(dist[miss], 1e-6) ** 2
        V[miss] = (w[:, :, None] * vals[idx[miss]]).sum(axis=1) / w.sum(axis=1)[:, None]

    # 13 images per layer: January..December, then the year-round mean of the twelve.
    layers = {k: [] for k in keys}
    for j in range(len(keys) * 13):
        key, mo = keys[j // 13], j % 13
        cols = V[:, (j // 13) * 12:(j // 13 + 1) * 12]
        x = cols.mean(axis=1) if mo == 12 else cols[:, mo]
        # 0..250 grey levels: shares directly; the change is centred on 125 over ±CHANGE_RANGE.
        v = 125 + x / CHANGE_RANGE * 125 if key.startswith("change") else x * 250
        img = np.full(LA.shape, 255, dtype=np.uint8)
        # Even levels only: 126 steps is finer than the 7-colour ramp can show, and the PNGs shrink.
        img[land] = np.where(near, np.clip(np.round(np.nan_to_num(v) / 2) * 2, 0, 250), 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img, "L").save(buf, "PNG", optimize=True)
        layers[key].append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())

    # Compact point summaries for the tooltips: shares as whole percent; the projected
    # block leaves out wind and air quality, which the projection doesn't change.
    pc = lambda xs: [None if x is None else int(round(x * 100)) for x in xs]
    def summary(clim, fut=False):
        out = {"s": pc([m["score"] for m in clim]), "i": pc([m["ideal_frac"] for m in clim]),
               "md": pc([m["medium_frac"] for m in clim]), "lo": pc([m["low_frac"] for m in clim]),
               "t": [round(m["t_max"], 1) for m in clim], "hx": [m.get("hx") for m in clim],
               "rv": pc([m["rain_veto_frac"] for m in clim]), "hv": pc([m.get("humid_frac") for m in clim])}
        if not fut:
            out.update(wv=pc([m["wind_veto_frac"] for m in clim]), av=pc([m["aq_veto_frac"] for m in clim]))
        return out
    points = [{"lat": r["key"][0], "lon": r["key"][1], **({"city": r["city"]} if "city" in r else {}), **summary(r["clim"]),
               **({"f": summary(r["fut"], True)} if r["fut"] else {})} for r in rows if "city" not in r]
    out = {"res": RASTER_RES, "lat_max": LAT_MAX, "lat_min": LAT_MIN, "pct": True, "w": len(rlon), "h": len(rlat),
           "scale": 250, "layers": layers, "points": points, "step": WX_STEP, "n_target": len(grid_points(WX_STEP)),
           "n_grid": n_grid, "n_aq": len(list(AQ_DIR.glob("*.npz"))) if AQ_DIR.exists() else 0,
           "baseline": [C.BASELINE_YEARS[0], C.BASELINE_YEARS[-1]], "aq_years": list(AQ_RANGE),
           "change_range": CHANGE_RANGE, "n_dew": sum(1 for r in rows if "city" not in r and r["clim"][0].get("hx") is not None), "projection": list(C.PROJ_YEARS), "models": __import__("projection").models_available()}
    _atomic_write(C.WEB_DATA / "grid.json", json.dumps(out, separators=(",", ":")))
    print("wrote", C.WEB_DATA / "grid.json", f"{(C.WEB_DATA / 'grid.json').stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch-wx", "fetch-aq", "render"])
    ap.add_argument("--step", type=float)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=10 ** 9)
    a = ap.parse_args()
    if a.cmd == "fetch-wx":
        fetch_wx(a.step or WX_STEP, a.workers, a.limit)
    elif a.cmd == "fetch-aq":
        fetch_aq(a.step or 4.0, a.limit)
    else:
        render()
    sys.exit(0)
