"""Build per-city monthly climatology + month anomalies -> web/data/cities.json.

Usage: python3.13 build_cities.py
"""
import datetime as dt
import json
import time

import numpy as np

import config as C
import projection
from dew_grid import align, load_dew
from fetch import DailyLimit, load_hourly
from score import aq_probability_by_month, daily_scores, final_scores, monthly_climatology

TODAY = dt.date.today()
ALL_YEARS = list(range(C.BASELINE_YEARS[0], TODAY.year + 1))
ANOMALY_MONTHS = 13  # current month + previous 12


def _num(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def month_metrics(d, s, mask):
    aq = d["aqi"][mask]
    return {
        "score": float(s[mask].mean()),
        "good_days": float(s[mask].sum()),
        "t_mean": float(np.nanmean(d["t_mean"][mask])),
        "t_max": float(np.nanmean(d["t_max"][mask])),
        "rain_mm": float(d["rain_day"][mask].sum()),
        "rain_days": float((d["rain_day"][mask] >= 1).sum()),
        "wind_max": float(d["wind_max"][mask].mean()),
        "hx": float(np.nanmean(d["hx_max"][mask])) if np.isfinite(d["hx_max"][mask]).any() else np.nan,
        "aqi": float(np.nanmean(aq)) if np.isfinite(aq).any() else np.nan,
    }


def anomalies(d):
    year = d["date"].astype("datetime64[Y]").astype(int) + 1970
    month = d["date"].astype("datetime64[M]").astype(int) % 12 + 1
    dom = (d["date"] - d["date"].astype("datetime64[M]")).astype(int) + 1
    s = final_scores(d, aq_probability_by_month(d))
    last_day = d["date"][d["complete"]].max()

    out = []
    cur = dt.date(TODAY.year, TODAY.month, 1)
    for i in range(ANOMALY_MONTHS - 1, -1, -1):
        y, m = cur.year, cur.month - i
        while m < 1:
            y, m = y - 1, m + 12
        sel = (year == y) & (month == m) & d["complete"]
        if not sel.any():
            continue
        max_dom = int(dom[sel].max())
        partial = (y, m) == (TODAY.year, TODAY.month) or max_dom < 28
        actual = month_metrics(d, s, sel)
        base, base_years = [], []
        for by in C.BASELINE_YEARS:
            bsel = (year == by) & (month == m) & (dom <= max_dom) & d["complete"]
            if bsel.any():
                base.append(month_metrics(d, s, bsel))
                base_years.append(by)
        row = {"year": y, "month": m, "days": int(sel.sum()), "through_day": max_dom, "partial": bool(partial), "metrics": {}}
        for k, v in actual.items():
            bv = np.array([b[k] for b in base], dtype=float)
            bv = bv[np.isfinite(bv)]
            mean = bv.mean() if len(bv) else np.nan
            sd = bv.std(ddof=1) if len(bv) > 1 else np.nan
            rank = float((bv < v).mean()) if len(bv) and np.isfinite(v) else np.nan
            row["metrics"][k] = {
                "value": _num(v), "base_mean": _num(mean), "base_sd": _num(sd),
                "anomaly": _num(v - mean), "z": _num((v - mean) / sd if sd and sd > 0 else np.nan),
                "pct_below": _num(rank), "base_n": int(len(bv)),
                "base_values": [_num(b[k]) for b in base], "base_years": base_years,
            }
        out.append(row)

    # Daily detail for the last ~45 days.
    recent = np.where(d["date"] >= np.datetime64(TODAY - dt.timedelta(days=45)))[0]
    daily = [{
        "date": str(d["date"][i]), "score": _num(s[i]), "score_wx": _num(d["score_wx"][i]),
        "t_max": _num(d["t_max"][i], 1), "t_mean": _num(d["t_mean"][i], 1),
        "rain": _num(d["rain_day"][i], 1), "rain_ride": _num(d["rain_ride"][i], 1),
        "wind_max": _num(d["wind_max"][i], 0), "aqi": _num(d["aqi"][i], 0),
        "hx": _num(d["hx_max"][i], 1), "humid": bool(d["humid"][i]),
        "ideal_h": int(d["n_ideal"][i]), "ok_h": int(d["n_ok"][i]), "low_h": int(d["n_low"][i]),
        "warm_dry_h": int(d["n_warm_dry"][i]), "wet_warm": bool(d["wet_warm"][i]),
        "complete": bool(d["complete"][i]),
    } for i in recent]
    return out, daily, str(last_day)


def nearest_grid_aq(lat, lon, max_deg=3.0):
    """Fallback air quality: the nearest CAMS sample from the map's AQ grid (2024-2025)."""
    from build_grid import AQ_DIR, _f
    best, bd = None, max_deg
    for p in AQ_DIR.glob("*.npz"):
        plat, plon = map(float, p.stem.split("_"))
        d = np.hypot(plat - lat, (plon - lon) * np.cos(np.radians(lat)))
        if d < bd:
            best, bd = p, d
    if best is None:
        return None, "none available yet"
    z = np.load(best)
    aq = {"time": z["t"].astype("datetime64[h]"), "european_aqi": _f(z["aqi"], 1), "meta": {}}
    return aq, f"nearest map sample ({best.stem.replace('_', '°, ')}°, {bd:.1f}° away)"


def load_cds_city(lat, lon):
    """Fallback weather source: the CDS ERA5 time-series, 2016 to the latest
    available day (~6 days behind). Cached per day; local solar time."""
    import cdsapi
    from build_grid import _cds_point, load_wx
    path = C.CACHE / "weather_cds" / f"{lat:.3f}_{lon:.3f}_{TODAY.isoformat()}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for old in path.parent.glob(f"{lat:.3f}_{lon:.3f}_*.npz"):
            old.unlink()
        client = cdsapi.Client(quiet=True, progress=False)
        while True:
            try:
                _cds_point(client, lat, lon, out=path, end=TODAY.isoformat())
                break
            except Exception as e:
                if "temporarily limited" not in str(e):
                    raise
                time.sleep(45)  # the map fetcher shares the CDS queue; wait for a slot
    wx = load_wx(path, lon, lat)
    cell = np.load(path)["cell"]
    wx["meta"].update(latitude=float(cell[0]), longitude=float(cell[1]), elevation=None, timezone="local solar time")
    return wx


def add_dew(wx, lat, lon):
    """Dew point for an Open-Meteo series: Open-Meteo first, else gridded ERA5 (dew_grid.py)."""
    try:
        dew = load_hourly("dew", lat, lon, ALL_YEARS, wait_daily=False)
        wx["dew_point_2m"] = align(wx["time"], dew["time"], dew["dew_point_2m"])
        return "Open-Meteo"
    except DailyLimit:
        g = load_dew(lat, lon)
        if g is None:
            return None
        # Gridded series is UTC; Open-Meteo times are local clock time.
        wx["dew_point_2m"] = align(wx["time"] - np.timedelta64(int(wx["meta"]["utc_offset_seconds"]) // 3600, "h"), *g)
        return "ERA5 at 1°"


def main():
    cities = []
    deltas = projection.deltas_at([c[1] for c in C.CITIES], [c[2] for c in C.CITIES])
    for ci, (name, lat, lon) in enumerate(C.CITIES):
        print(f"{name}: weather", flush=True)
        try:
            wx = load_hourly("weather", lat, lon, ALL_YEARS, wait_daily=False)
            source = "Open-Meteo (ERA5 + ERA5-Land)"
            dew_source = add_dew(wx, lat, lon)
        except DailyLimit:
            print(f"{name}: Open-Meteo daily limit reached, using ERA5 from the Copernicus CDS", flush=True)
            wx = load_cds_city(lat, lon)
            source = "Copernicus CDS (ERA5)"
            dew_source = "ERA5 at 1°" if "dew_point_2m" in wx else None
        print(f"{name}: air quality", flush=True)
        # CAMS global (outside Europe) only starts Aug 2022; skip empty years to save quota.
        europe = -25 <= lon <= 45 and 30 <= lat <= 72
        aq_source = "CAMS via Open-Meteo"
        try:
            aq = load_hourly("aq", lat, lon, ALL_YEARS if europe else [y for y in ALL_YEARS if y >= 2022], wait_daily=False)
        except DailyLimit:
            aq, aq_source = nearest_grid_aq(lat, lon)
            print(f"{name}: air-quality quota spent, using {aq_source}", flush=True)
        d = daily_scores(wx, aq, lat, lon)
        clim = monthly_climatology(d, C.BASELINE_YEARS)
        fut, delta = None, None
        if deltas is not None:
            dT, ratio = deltas[0][ci], deltas[1][ci]
            # Same AQ series; the projection changes temperature, dew point and rain only.
            fut = monthly_climatology(daily_scores(projection.apply_delta(wx, dT, ratio), aq, lat, lon), C.BASELINE_YEARS)
            delta = {"t": [round(float(x), 2) for x in dT], "rain": [round(float(x), 3) for x in ratio]}
        anom, daily, last = anomalies(d)
        aq_first = d["date"][~np.isnan(d["aqi"])]
        cities.append({
            "name": name, "lat": lat, "lon": lon,
            "grid_lat": wx["meta"]["latitude"], "grid_lon": wx["meta"]["longitude"], "source": source, "aq_source": aq_source, "dew_source": dew_source,
            "elevation": wx["meta"]["elevation"], "timezone": wx["meta"]["timezone"],
            "aq_since": str(aq_first[0]) if len(aq_first) else None,
            "data_through": last,
            "climatology": clim, "climatology_future": fut, "delta": delta, "anomaly": anom, "daily": daily,
        })
        print(f"  annual score {np.mean([c['score'] for c in clim]):.3f}; AQ since {cities[-1]['aq_since']}")

    out = {
        "generated": TODAY.isoformat(),
        "baseline": [C.BASELINE_YEARS[0], C.BASELINE_YEARS[-1]],
        "rules": {
            "ideal_t": C.IDEAL_T, "ok_t": [C.OK_T_LOW, C.OK_T_HIGH], "min_hours": C.MIN_GOOD_HOURS,
            "rain_max_mm": C.DAY_RAIN_MAX_MM, "wind_not_ideal": C.WIND_NOT_IDEAL,
            "wind_zero": C.WIND_ZERO, "aqi_veto": C.AQI_VETO, "low_t": C.LOW_T,
            "wet_warm_t": C.WET_WARM_T, "wet_warm_hours": C.WET_WARM_MIN_DRY_HOURS,
            "wet_warm_rain_max": C.WET_WARM_RAIN_MAX_MM,
            "scores": [C.SCORE_IDEAL, C.SCORE_MEDIUM, C.SCORE_LOW],
            "humidex": [C.HUMIDEX_IDEAL_MAX, C.HUMIDEX_MEDIUM_MAX, C.HUMIDEX_LOW_MAX],
        },
        "projection": {"years": list(C.PROJ_YEARS), "scenario": "SSP2-4.5", "models": projection.models_available(),
                       "fit_years": list(C.PROJ_FIT_YEARS)} if deltas is not None else None,
        "cities": cities,
    }
    C.WEB_DATA.mkdir(parents=True, exist_ok=True)
    (C.WEB_DATA / "cities.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("wrote", C.WEB_DATA / "cities.json")


if __name__ == "__main__":
    main()
