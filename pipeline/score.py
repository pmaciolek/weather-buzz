"""Daily cycling-weather score from hourly data.

Per riding hour (local 06-22h, sun above horizon), the hour takes the worst of:
  temperature : ideal 18-30 °C; medium 12-18 or 30-33 °C; low 4-12 °C
  wind        : <= 25 km/h no limit; 25-40 km/h one step down (ideal -> medium,
                medium -> low; a low hour stays low); > 40 km/h rules out the day
  humidex     : <= 25 no limit; 25-30 at best medium; 30-35 at best low;
                > 35 not usable (needs dew point; without it the hour isn't limited)
Per day:
  any riding hour with wind > 40 km/h          -> 0
  rain over riding window < 1 mm (dry day):
    >= 6 ideal hours                           -> 1.0
    >= 6 ideal+medium hours                    -> 0.5
    >= 6 ideal+medium+low hours                -> 0.25
  1-10 mm rain, but >= 3 dry hours above 24 °C
    with humidex <= 35                         -> 0.25 (warm, showery)
  otherwise                                    -> 0
  daytime mean European AQI >= 60 (poor)       -> 0  (applied when AQ data exists)
"""
import numpy as np

import config as C


def _sun_up(time_local, utc_offset_s, lat, lon):
    t_utc = time_local - np.timedelta64(int(utc_offset_s), "s")
    doy = (t_utc.astype("datetime64[D]") - t_utc.astype("datetime64[Y]")).astype(int) + 1
    hour_utc = (t_utc - t_utc.astype("datetime64[D]")).astype("timedelta64[m]").astype(float) / 60 + 0.5
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.radians((hour_utc + lon / 15 - 12) * 15)
    phi = np.radians(lat)
    sin_el = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return sin_el > 0


def humidex(T, Td):
    """Environment Canada humidex from air temperature and dew point (°C)."""
    e = 6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + Td)))
    return T + 0.5555 * (e - 10.0)


def daily_scores(wx, aq, lat, lon):
    """wx/aq: dicts from fetch.load_hourly (aq may be None). Returns dict of daily arrays."""
    t = wx["time"].astype("datetime64[h]")  # hour arithmetic below assumes hourly units
    hour = (t - t.astype("datetime64[D]")).astype(int)
    days, day_idx = np.unique(t.astype("datetime64[D]"), return_inverse=True)
    n = len(days)

    T, P, W = wx["temperature_2m"], wx["precipitation"], wx["wind_speed_10m"]
    ride = (hour >= C.RIDE_START) & (hour < C.RIDE_END) & _sun_up(t, wx["meta"]["utc_offset_seconds"], lat, lon)
    valid = ~np.isnan(T) & ~np.isnan(P) & ~np.isnan(W)

    Td = wx.get("dew_point_2m")
    Td = np.full_like(T, np.nan) if Td is None else Td

    t_ideal = (T >= C.IDEAL_T[0]) & (T <= C.IDEAL_T[1])
    t_ok = ((T >= C.OK_T_LOW[0]) & (T < C.OK_T_LOW[1])) | ((T > C.OK_T_HIGH[0]) & (T <= C.OK_T_HIGH[1]))
    t_low = (T >= C.LOW_T[0]) & (T < C.LOW_T[1])
    # Hour level: 3 ideal, 2 medium, 1 low, 0 unusable.
    lvl = np.select([t_ideal, t_ok, t_low], [3, 2, 1], 0)
    breezy = (W > C.WIND_NOT_IDEAL) & (W <= C.WIND_ZERO)
    lvl = np.where(breezy & (lvl > 1), lvl - 1, lvl)
    lvl = np.where(W > C.WIND_ZERO, 0, lvl)
    lvl = np.where(ride & valid, lvl, 0)
    # Humidex caps the level; a missing dew point doesn't limit the hour.
    H = humidex(T, Td)
    cap = np.select([H > C.HUMIDEX_LOW_MAX, H > C.HUMIDEX_MEDIUM_MAX, H > C.HUMIDEX_IDEAL_MAX], [0, 1, 2], 3)
    lvl_dew = np.minimum(lvl, cap)
    ideal_h, ok_h, low_h = lvl_dew == 3, lvl_dew == 2, lvl_dew == 1
    warm_dry_h = ride & valid & (T > C.WET_WARM_T) & (P < C.DRY_HOUR_MM) & (W <= C.WIND_ZERO) & ~(H > C.HUMIDEX_LOW_MAX)

    def per_day(values, fn=np.add, init=0.0):
        out = np.full(n, init, dtype=float)
        fn.at(out, day_idx, values)
        return out

    n_ideal = per_day(ideal_h.astype(float))
    n_ok = per_day(ok_h.astype(float))
    n_low = per_day(low_h.astype(float))
    n_warm_dry = per_day(warm_dry_h.astype(float))
    # The same counts without the humidex cap, to tell which days heat and humidity cost.
    n_ideal_nd = per_day((lvl == 3).astype(float))
    n_ok_nd = per_day((lvl == 2).astype(float))
    n_low_nd = per_day((lvl == 1).astype(float))
    # Daily peak humidex over the riding hours.
    has_h = ride & valid & ~np.isnan(H)
    hx_max = per_day(np.where(has_h, H, -99.0), np.maximum, -99.0)
    hx_max = np.where(per_day(has_h.astype(float)) > 0, hx_max, np.nan)
    rain = per_day(np.where(ride, np.nan_to_num(P), 0.0))
    wind_max = per_day(np.where(ride, np.nan_to_num(W), 0.0), np.maximum, 0.0)
    t_max = per_day(np.where(valid, T, -99.0), np.maximum, -99.0)
    t_sum = per_day(np.where(valid, T, 0.0))
    n_valid = per_day(valid.astype(float))
    rain_all = per_day(np.nan_to_num(P))
    complete = n_valid >= 20

    rain_veto = rain >= C.DAY_RAIN_MAX_MM
    wind_veto = wind_max > C.WIND_ZERO
    def tier(ni, no, nl):
        return np.select([ni >= C.MIN_GOOD_HOURS, ni + no >= C.MIN_GOOD_HOURS, ni + no + nl >= C.MIN_GOOD_HOURS],
                         [C.SCORE_IDEAL, C.SCORE_MEDIUM, C.SCORE_LOW], 0.0)
    dry_score = tier(n_ideal, n_ok, n_low)
    wet_warm = rain_veto & (rain <= C.WET_WARM_RAIN_MAX_MM) & (n_warm_dry >= C.WET_WARM_MIN_DRY_HOURS)
    score = np.where(rain_veto, np.where(wet_warm, C.SCORE_LOW, 0.0), dry_score)
    score = np.where(wind_veto, 0.0, score)
    # Days that would have scored higher without the humidex limit.
    humid = ~rain_veto & ~wind_veto & (tier(n_ideal_nd, n_ok_nd, n_low_nd) > dry_score)

    # Air quality: daytime (riding window) mean European AQI per day.
    aqi = np.full(n, np.nan)
    if aq is not None:
        lookup = {d: i for i, d in enumerate(days)}
        at = aq["time"].astype("datetime64[h]")
        ah = (at - at.astype("datetime64[D]")).astype(int)
        sel = (ah >= 9) & (ah < 19) & ~np.isnan(aq["european_aqi"])
        ad = at[sel].astype("datetime64[D]")
        av = aq["european_aqi"][sel]
        idx = np.array([lookup.get(d, -1) for d in ad])
        ok = idx >= 0
        s = np.zeros(n); c = np.zeros(n)
        np.add.at(s, idx[ok], av[ok]); np.add.at(c, idx[ok], 1)
        aqi = np.where(c >= 6, s / np.maximum(c, 1), np.nan)
    aq_veto = np.where(np.isnan(aqi), np.nan, (aqi >= C.AQI_VETO).astype(float))

    return {
        "date": days, "complete": complete,
        "score_wx": score, "aq_veto": aq_veto, "aqi": aqi,
        "n_ideal": n_ideal, "n_ok": n_ok, "n_low": n_low, "n_warm_dry": n_warm_dry, "wet_warm": wet_warm,
        "rain_ride": rain, "rain_day": rain_all, "wind_max": wind_max,
        "t_max": np.where(complete, t_max, np.nan),
        "t_mean": np.where(complete, t_sum / np.maximum(n_valid, 1), np.nan),
        "rain_veto": rain_veto, "wind_veto": wind_veto, "humid": humid, "hx_max": np.where(complete, hx_max, np.nan),
    }


def final_scores(d, p_aq_by_month=None):
    """Daily score including AQ. Days without AQ data get an expected penalty
    (1 - probability of a poor-AQ day in that calendar month at that location)."""
    month = (d["date"].astype("datetime64[M]").astype(int) % 12)
    s = d["score_wx"].copy()
    has = ~np.isnan(d["aq_veto"])
    s[has] *= 1 - d["aq_veto"][has]
    if p_aq_by_month is not None:
        s[~has] *= 1 - p_aq_by_month[month[~has]]
    return s


def aq_probability_by_month(d):
    """Fraction of days with AQ veto per calendar month, from days that have AQ data."""
    month = (d["date"].astype("datetime64[M]").astype(int) % 12)
    p = np.zeros(12)
    for m in range(12):
        v = d["aq_veto"][(month == m) & ~np.isnan(d["aq_veto"])]
        p[m] = v.mean() if len(v) else 0.0
    return p


def _r(x, nd):
    return round(float(x), nd)


def monthly_climatology(d, years):
    """Calendar-month means over the given years (complete days only)."""
    year = d["date"].astype("datetime64[Y]").astype(int) + 1970
    month = d["date"].astype("datetime64[M]").astype(int) % 12
    p_aq = aq_probability_by_month(d)
    s = final_scores(d, p_aq)
    base = np.isin(year, years) & d["complete"]
    out = []
    for m in range(12):
        k = base & (month == m)
        aqk = k & ~np.isnan(d["aq_veto"])
        out.append({
            "month": m + 1,
            "score": round(float(s[k].mean()), 4),
            "ideal_frac": round(float((d["score_wx"][k] == C.SCORE_IDEAL).mean()), 4),
            "medium_frac": round(float((d["score_wx"][k] == C.SCORE_MEDIUM).mean()), 4),
            "low_frac": round(float((d["score_wx"][k] == C.SCORE_LOW).mean()), 4),
            "rain_veto_frac": round(float((d["rain_veto"][k] & ~d["wet_warm"][k] & ~d["wind_veto"][k]).mean()), 4),
            "wind_veto_frac": round(float(d["wind_veto"][k].mean()), 4),
            "humid_frac": round(float(d["humid"][k].mean()), 4),
            "hx": _r(np.nanmean(d["hx_max"][k]), 1) if np.isfinite(d["hx_max"][k]).any() else None,
            "aq_veto_frac": round(float(p_aq[m]), 4),
            "aq_days": int(aqk.sum()),
            "t_max": round(float(np.nanmean(d["t_max"][k])), 2),
            "rain_mm": round(float(d["rain_day"][k].mean()), 2),
            "wind_max": round(float(d["wind_max"][k].mean()), 2),
            "aqi": round(float(np.nanmean(d["aqi"][k])), 1) if aqk.any() else None,
        })
    return out
