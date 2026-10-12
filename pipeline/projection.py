"""Climate change deltas for the "next 10 years" view (CMIP6, SSP2-4.5).

  python3.13 projection.py fetch     # monthly tas + pr per model from the Copernicus CDS
  python3.13 projection.py summary   # print the deltas at the places

Delta change with pattern scaling. A single model run's change over one decade
is mostly year-to-year noise, so for each model:
  1. its global mean warming between the two decades (dG) comes from a straight
     line fitted to its annual global mean temperature, 2015-2050;
  2. each cell and calendar month is regressed on that global mean over the whole
     2015-2100 run, giving the local change per degree of global warming;
  3. local change = pattern x dG, smoothed lightly across neighbouring months.
Temperature changes are added in °C; precipitation is applied as a ratio
(clamped to PROJ_RAIN_RATIO). The mean over the models is applied to the observed
hours (apply_delta): dew point keeps the hour's relative humidity; wind and air
quality are unchanged.
"""
import glob
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config as C

DIR = C.CACHE / "cmip6"
VARS = {"tas": "near_surface_air_temperature", "pr": "precipitation"}
SHIFT = C.PROJ_YEARS[0] - C.BASELINE_YEARS[0]   # 10 years between the two period centres


def _nc(model, var):
    hits = glob.glob(str(DIR / model / f"{var}_*.nc"))
    return hits[0] if hits else None


def fetch_one(model, var):
    import cdsapi
    if _nc(model, var):
        return f"{model} {var}: cached"
    out = DIR / model
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{var}.zip"
    while True:
        try:
            cdsapi.Client(quiet=True, progress=False).retrieve("projections-cmip6", {
                "temporal_resolution": "monthly", "experiment": C.PROJ_SCENARIO,
                "variable": VARS[var], "model": model,
                "year": [str(y) for y in range(C.PROJ_FIT_YEARS[0], C.PROJ_FIT_YEARS[1] + 1)],
                "month": [f"{m:02d}" for m in range(1, 13)]}, str(tmp))
            break
        except Exception as e:
            msg = " ".join(str(e).split())
            if "temporarily limited" not in msg:
                return f"{model} {var}: FAILED {msg[:200]}"
            time.sleep(45)   # the CDS caps queued requests per dataset; wait for a slot
    with zipfile.ZipFile(tmp) as z:
        for n in z.namelist():
            if n.endswith(".nc"):
                z.extract(n, out)
    tmp.unlink()
    return f"{model} {var}: ok"


def fetch():
    jobs = [(m, v) for m in C.PROJ_MODELS for v in VARS]
    with ThreadPoolExecutor(3) as ex:
        for msg in ex.map(lambda mv: fetch_one(*mv), jobs):
            print(msg, flush=True)


def _read(model, var):
    """Monthly cube (year, month, lat, lon); some models ship one file per year."""
    import netCDF4
    year, month, xs = [], [], []
    for path in sorted(glob.glob(str(DIR / model / f"{var}_*.nc"))):
        f = netCDF4.Dataset(path)
        tv = f.variables["time"]
        t = netCDF4.num2date(tv[:], tv.units, tv.calendar)
        year += [d.year for d in t]; month += [d.month for d in t]
        xs.append(np.asarray(f.variables[var][:], dtype=float))
        lat, lon = np.asarray(f.variables["lat"][:]), np.asarray(f.variables["lon"][:])
        f.close()
    year, month, x = np.array(year), np.array(month), np.concatenate(xs)
    years = np.arange(year.min(), year.max() + 1)
    cube = np.full((len(years), 12) + x.shape[1:], np.nan)
    cube[year - years[0], month - 1] = x
    return lat, lon, years, cube


def _smooth_months(a):
    """Light circular smoothing across calendar months (1-2-1)."""
    return 0.25 * np.roll(a, 1, axis=0) + 0.5 * a + 0.25 * np.roll(a, -1, axis=0)


def _fields(model):
    """Per-model change fields: (lat, lon, dT[12, nlat, nlon], ratio[12, nlat, nlon]), cached."""
    cache = DIR / model / "deltas.npz"
    if cache.exists():
        z = np.load(cache)
        return z["lat"], z["lon"], z["dT"], z["ratio"]
    lat, lon, years, tas = _read(model, "tas")
    w = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))
    G = np.nansum(np.nanmean(tas, axis=1) * w, axis=(1, 2)) / w.sum()   # annual global mean
    full = np.isfinite(tas).all(axis=(1, 2, 3))
    near = full & (years <= 2050)
    dG = np.polyfit(years[near], G[near], 1)[0] * SHIFT
    g = G[full] - G[full].mean()
    out = {}
    for var in VARS:
        _, _, yv, x = _read(model, var) if var != "tas" else (lat, lon, years, tas)
        x = x[np.isin(yv, years[full])]
        xa = x - x.mean(axis=0)
        k = np.einsum("y,ymij->mij", g, xa) / (g @ g)                  # change per °C of global warming
        if var == "tas":
            out[var] = _smooth_months(k * dG)
        else:
            gb = np.mean([G[years == y][0] for y in C.BASELINE_YEARS]) - G[full].mean()
            base = x.mean(axis=0) + k * gb                                   # fitted 2016-2025 mean
            ratio = np.where(base > 1e-7, 1 + k * dG / np.maximum(base, 1e-12), 1.0)
            out[var] = _smooth_months(ratio)
    np.savez_compressed(cache, lat=lat, lon=lon, dT=out["tas"], ratio=out["pr"], dG=dG)
    return lat, lon, out["tas"], out["pr"]


def models_available():
    return [m for m in C.PROJ_MODELS if _nc(m, "tas") and _nc(m, "pr")]


def deltas_at(lats, lons):
    """Multi-model mean changes at points: dT (n, 12) in °C and rain ratio (n, 12)."""
    lats, lons = np.atleast_1d(lats).astype(float), np.atleast_1d(lons).astype(float) % 360
    dts, rts = [], []
    for model in models_available():
        lat, lon, dT, ratio = _fields(model)
        # Wrap longitude so points between the last column and 360° interpolate.
        lonw = np.concatenate([lon, [lon[0] + 360]])
        for arr, acc in ((dT, dts), (ratio, rts)):
            a = np.concatenate([arr, arr[:, :, :1]], axis=2)
            vals = [RegularGridInterpolator((lat, lonw), a[m], bounds_error=False, fill_value=None)(np.column_stack([lats, lons]))
                    for m in range(12)]
            acc.append(np.stack(vals, axis=1))
    if not dts:
        return None
    return np.mean(dts, axis=0), np.clip(np.mean(rts, axis=0), *C.PROJ_RAIN_RATIO)


def apply_delta(wx, dT, ratio):
    """A copy of the hourly dict with the projected change applied by calendar month."""
    month = wx["time"].astype("datetime64[M]").astype(int) % 12
    T0 = wx["temperature_2m"]
    T1 = T0 + dT[month]
    out = dict(wx, temperature_2m=T1, precipitation=wx["precipitation"] * ratio[month])
    Td = wx.get("dew_point_2m")
    if Td is not None:
        # Constant relative humidity (Magnus formula): shift the vapour pressure with T.
        a, b = 17.62, 243.12
        g = a * Td / (b + Td) - a * T0 / (b + T0) + a * T1 / (b + T1)
        out["dew_point_2m"] = b * g / (a - g)
    return out


def summary():
    print("models:", ", ".join(models_available()))
    lat = [c[1] for c in C.CITIES]; lon = [c[2] for c in C.CITIES]
    dT, r = deltas_at(lat, lon)
    for (name, *_), t, q in zip(C.CITIES, dT, r):
        print(f"{name:24s} dT Jan {t[0]:+.2f} Jul {t[6]:+.2f} year {t.mean():+.2f} °C   rain Jan {q[0]:.2f} Jul {q[6]:.2f}")


if __name__ == "__main__":
    {"fetch": fetch, "summary": summary}[sys.argv[1]]()
