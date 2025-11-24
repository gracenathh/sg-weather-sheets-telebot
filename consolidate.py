import os, requests, io, csv, math, time
import datetime as dt
from sheets_utils import open_sheet, ensure_worksheet
from typing import List, Dict, Tuple
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from datetime import date, timedelta
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

SGT = dt.timezone(dt.timedelta(hours=8))
UA = os.getenv("USER_AGENT", "sg-weather-collector/1.0")
API_KEY = os.getenv("DATA_GOV_SG_API_KEY")
SHEET_ID = os.environ["SHEET_ID"]
TIME_OF_DAY_ORDER = [
    ("Late Night", 0, 6),
    ("Breakfast", 6, 10),
    ("Lunch", 10, 14),
    ("Tea", 14, 17),
    ("Dinner", 17, 21),
    ("Supper", 21, 24),
]

def open_sheet_by_id(sheet_id):
    creds = Credentials.from_service_account_file(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=["https://www.googleapis.com/auth/spreadsheets"])
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id)

def append_unique(ws, rows, key_cols):
    if not rows: 
        return

    existing_values = ws.get_all_values()
    existing = set()
    header = []

    if existing_values:
        header = existing_values[0]
        idx = {header[i]: i for i in range(len(header))}
        for r in existing_values[1:]:
            try: existing.add(tuple(r[idx[c]] for c in key_cols))
            except KeyError: pass

    if not header: 
        return

    key_positions = [header.index(c) for c in key_cols]
    to_append = []
    for row in rows:
        k = tuple(row[p] for p in key_positions)
        if k in existing: continue
        to_append.append(row); existing.add(k)

    if to_append: 
        ws.append_rows(to_append, value_input_option="USER_ENTERED")

def to_dt(tstr):
    if not tstr: return None
    try:
        if str(tstr).endswith("Z"):
            return dt.datetime.fromisoformat(str(tstr).replace("Z", "+00:00")).astimezone(SGT)
        return dt.datetime.fromisoformat(str(tstr)).astimezone(SGT)
    except Exception:
        return None

def iter_dates_inclusive(start_date_str, end_date_str=None):
    d0 = dt.date.fromisoformat(start_date_str)
    d1 = dt.date.fromisoformat(end_date_str) if end_date_str else dt.datetime.now(SGT).date()
    step = dt.timedelta(days=1) if d0 <= d1 else dt.timedelta(days=-1)
    d = d0
    while True:
        yield d
        if d == d1: break
        d += step

def _headers():
    h = {"Accept": "application/json", "User-Agent": UA}
    if API_KEY: 
        h["X-Api-Key"] = API_KEY
    return h

def fetch_data(url, raw = False, params = None): # raw is only for V1 -- https://api.data.gov.sg/v1/environment/
    params = params or {}
    requestor = requests.get(url, headers=_headers(), params = params, timeout=30)
    requestor.raise_for_status()
    json_ = requestor.json() or {}

    if raw:
        return json_

    df = json_.get("data") or {}
    return df

def forecast_value_reading(raw_data):

    # v2 shape
    if "valid_period" in raw_data and "forecasts" in raw_data:
        valid_from = raw_data["valid_period"].get("start")
        valid_to = raw_data["valid_period"].get("end")
        issued_at = raw_data.get("update_timestamp") or raw_data.get("timestamp")
        forecasts = raw_data.get("forecasts") or []

    # v1 fallback shape
    elif isinstance(raw_data.get("items"), list) and raw_data["items"]:
        item = raw_data["items"][0]
        vp   = item.get("valid_period") or {}
        valid_from = vp.get("start") or item.get("valid_from")
        valid_to = vp.get("end") or item.get("valid_to")
        issued_at = item.get("update_timestamp") or item.get("timestamp")
        forecasts = item.get("forecasts") or []
    else:
        raise RuntimeError(f"Unexpected 2hr forecast shape: {raw_data}")

    forecast_rows = []
    for f in forecasts:
        forecast_rows.append([f.get("area"), valid_from, valid_to, f.get("forecast"), issued_at])

    areas = raw_data.get("area_metadata") or []
    area_rows = []
    for a in areas:
        name = a.get("name")
        ll = a.get("label_location") or {}
        area_rows.append([name, ll.get("latitude"), ll.get("longitude")])

    return forecast_rows, area_rows

def run_2h_forecast(sh):
    # appending value + area for 2h forecast
    forecast_url = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
    forecast_json_data = fetch_data(forecast_url)
    forecast_value, forecast_area = forecast_value_reading(forecast_json_data)

    forecast_value_ws_frame = ensure_worksheet(sh, "forecasts_2h", ["area","valid_from","valid_to","forecast","issued_at"])
    append_unique(forecast_value_ws_frame, forecast_value, key_cols=["area","valid_from","valid_to"])

    forecast_area_ws_frame =  ensure_worksheet(sh, "areas", ["area","lat","lon"])
    if forecast_area:
        # refresh areas each run (your current behavior)
        forecast_area_ws_frame.clear()
        forecast_area_ws_frame.append_row(["area","lat","lon"])
        forecast_area_ws_frame.append_rows(forecast_area, value_input_option="USER_ENTERED")

def rainfall_data_reading(raw_data):
    if isinstance(raw_data, dict) and "stations" in raw_data and "readings" in raw_data:
        d = raw_data
    else: 
        d = (raw_data or {}).get("data", {}) or {}

    stations_list = d.get("stations", []) or []
    buckets = d.get("readings", []) or []
    paginationToken = d.get("paginationToken", []) or None

    # index stations
    sid_to_meta = {}
    for s in stations_list:
        sid = str(s.get("id") or s.get("stationId") or s.get("deviceId") or "")
        if not sid:
            continue
        loc = s.get("location") or {}
        sid_to_meta[sid] = [
            sid,
            s.get("name"),
            loc.get("latitude"),
            loc.get("longitude"),
        ]

    obs_rows, st_rows_map = [], {}
    for bucket in buckets:
        ts = bucket.get("timestamp")
        for rd in (bucket.get("data") or []):
            sid = str(rd.get("stationId") or "")
            if not sid or not ts: 
                continue
            meta = sid_to_meta.get(sid, [sid, None, None, None])
            val = rd.get("value")
            obs_rows.append([meta[0], meta[1], meta[2], meta[3], val, ts])
            st_rows_map[sid] = meta

    return obs_rows, list(st_rows_map.values()), paginationToken

def run_rainfall(sh, date_str = None):
    """
    Returns (rainfall_value, rainfall_area). 
    If date_str is None, uses today's date (SGT) for v2 day fetch.
    """
    if date_str != None:
        rainfall_url = f"https://api-open.data.gov.sg/v2/real-time/api/rainfall?date={date_str}"
    else:
        rainfall_url = f"https://api-open.data.gov.sg/v2/real-time/api/rainfall"

    rainfall_json_data = fetch_data(rainfall_url)
    rainfall_value, rainfall_area, rainfall_page = rainfall_data_reading(rainfall_json_data)

    rainfall_next_value = None
    rainfall_next_area = None

    while rainfall_page != None:
        new_url = f"https://api-open.data.gov.sg/v2/real-time/api/rainfall?date={date_str}&paginationToken={rainfall_page}"

        next_page_json_data = fetch_data(new_url)
        rainfall_next_value, rainfall_next_area, rainfall_next_page = rainfall_data_reading(next_page_json_data)
        for item in rainfall_next_value:
            rainfall_value.append(item)
        
        for area in rainfall_next_area:
            rainfall_area.append(area)

        rainfall_page = rainfall_next_page

        
    # rainfall_value_ws_frame = ensure_worksheet(sh, "rainfall_data", ["station_id","station_name","lat","lon","reading_value","reading_time"])
    # append_unique(rainfall_value_ws_frame,rainfall_value,key_cols=["station_id","reading_time"])

    """
    rainfall_station_ws_frame = ensure_worksheet(sh, "stations", ["station_id","station_name","lat","lon"])

    if rainfall_area:
        # refresh stations each run (matches your current behavior)
        rainfall_station_ws_frame.clear()
        rainfall_station_ws_frame.append_row(["station_id","station_name","lat","lon"])
        rainfall_station_ws_frame.append_rows(rainfall_area, value_input_option="USER_ENTERED")
    """

    return rainfall_value, rainfall_area

def parse_mets_page(data_dict):
    """
    data_dict is what fetch_data(...) returns (already the 'data' object).
    Returns: (rows, next_token) where rows = [(timestamp, station_id, value), ...]
    """
    d = data_dict or {}
    buckets = d.get("readings") or []
    token = d.get("paginationToken") or None

    rows = []
    for bucket in buckets:
        ts = bucket.get("timestamp")
        for drow in (bucket.get("data") or []):
            sid = drow.get("stationId")
            val = drow.get("value")
            if ts and sid is not None and val is not None:
                rows.append((ts, str(sid), float(val)))
    return rows, token

def fetch_all_mets_rows(base_url, date_str=None):
    """
    Loops over pagination for a single METS endpoint (e.g., temperature).
    Returns a de-duplicated list of (timestamp, station_id, value).
    """
    output_row = []

    if date_str != None:
        mets_url = f"{base_url}?date={date_str}"
    else:
        mets_url = f"{base_url}"
    
    json_data = fetch_data(mets_url)
    row_value, paginationToken = parse_mets_page(json_data)

    seen = set()
    for ts, sid, val in row_value:
        key = (ts, sid)
        if key not in seen:
            seen.add(key)
            output_row.append((ts, sid, val))

    next_value = None
    next_page = None

    while paginationToken != None:
        
        new_url = f"{base_url}?date={date_str}&paginationToken={paginationToken}"
        next_json_data = fetch_data(new_url)
        next_value, next_page = parse_mets_page(next_json_data)

        for ts, sid, val in next_value:
            key = (ts, sid)
            if key not in seen:
                seen.add(key)
                output_row.append((ts, sid, val))

        paginationToken = next_page

    return output_row

def run_mets(sh, rainfall_area, date_str=None, write_to_sheet=True):
    """
    Uses v2 METS endpoints, paginates through the whole day, joins into rows:
    [timestamp, station_id, station_name, temp_c, rh_pct, wind_ms, wind_dir_deg]
    """

    urls = {
        "temp":"https://api-open.data.gov.sg/v2/real-time/api/air-temperature",
        "humidity":"https://api-open.data.gov.sg/v2/real-time/api/relative-humidity",
        "wind_dir":"https://api-open.data.gov.sg/v2/real-time/api/wind-direction",
        "wind_speed":"https://api-open.data.gov.sg/v2/real-time/api/wind-speed"
    }

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {k: ex.submit(fetch_all_mets_rows, u, date_str) for k, u in urls.items()}
        temperature_rows = futs["temp"].result()
        humidity_rows    = futs["humidity"].result()
        wind_dir_rows    = futs["wind_dir"].result()
        wind_speed_rows  = futs["wind_speed"].result()
    """
    # Pull all pages for each metric
    temperature_rows = fetch_all_mets_rows("https://api-open.data.gov.sg/v2/real-time/api/air-temperature", date_str)
    print("3A. Pulled Temperature")
    humidity_rows = fetch_all_mets_rows("https://api-open.data.gov.sg/v2/real-time/api/relative-humidity", date_str)
    print("3B. Pulled Humidity")
    wind_dir_rows = fetch_all_mets_rows("https://api-open.data.gov.sg/v2/real-time/api/wind-direction", date_str)
    print("3C. Pulled Wind Dir")
    wind_speed_rows = fetch_all_mets_rows("https://api-open.data.gov.sg/v2/real-time/api/wind-speed", date_str)
    print("3D. Pulled Wind Speed")
    """

    # Build lookup maps
    tmap = {(ts, sid): val for ts, sid, val in temperature_rows}
    hmap = {(ts, sid): val for ts, sid, val in humidity_rows}
    spm  = {(ts, sid): val for ts, sid, val in wind_speed_rows}   # speed (m/s)
    dgm  = {(ts, sid): val for ts, sid, val in wind_dir_rows}     # direction (deg)

    # Union of all keys across the four metrics
    keys = sorted(set().union(tmap.keys(), hmap.keys(), spm.keys(), dgm.keys()))

    # Station name map from rainfall_area
    station_name_map = {str(item[0]): (item[1] if len(item) > 1 else "") for item in (rainfall_area or [])}

    # Assemble final rows
    mets_data = []
    for ts, sid in keys:
        name = station_name_map.get(sid, "")
        temp = tmap.get((ts, sid))
        hum  = hmap.get((ts, sid))
        spd  = spm.get((ts, sid))
        deg  = dgm.get((ts, sid))
        mets_data.append([
            ts,
            sid,
            name,
            "" if temp is None else temp,
            "" if hum  is None else hum,
            "" if spd  is None else spd,
            "" if deg  is None else deg,
        ])

    if write_to_sheet:
        mets_headers = ["timestamp","station_id","station_name","temp_value_celcius","humidity_value_percentage","wind_speed_ms","wind_dir_deg"]
        mets_value_ws_frame = ensure_worksheet(sh, "mets_data", mets_headers)
        # ensure header exists on a brand-new sheet
        if not mets_value_ws_frame.get_all_values():
            mets_value_ws_frame.append_row(mets_headers, value_input_option="USER_ENTERED")
        append_unique(mets_value_ws_frame, mets_data, key_cols=["timestamp", "station_id"])

    return mets_data

def rainfall_summary_processing(rainfall_df):
    sums = defaultdict(float)
    meta = {}
    sums_by_day = defaultdict(float)

    for row in rainfall_df:
        if not row or len(row) < 6:
            continue

        sid, sname, lat, lon, val, tstr = row[:6]
        t = to_dt(tstr)

        if not t:
            continue

        key = (t.date(), str(sid))

        try:
            v = float(val or 0)
        except Exception:
            v = 0.0

        sums[key] += v
        sums_by_day[t.date()] += v
        # keep last seen station meta for that day
        meta[key] = (sname, lat, lon)

    return_row = []
    for (d, sid), total in sorted(sums.items()):
        sname, lat, lon = meta.get((d, sid), ("", "", ""))
        date_ = dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=SGT).isoformat()
        return_row.append([date_, sid, sname, lat, lon, round(total, 1),])
    
    for d in sorted(sums_by_day.keys()):
        sum_mm = round(sums_by_day[d], 1)
        date_ = dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=SGT).isoformat()
        # If you prefer mean: sum_mm = round(sums_by_day[d] / max(1, sum(1 for (dd,_) in sums if dd==d)), 1)
        # Or max: sum_mm = round(max(v for (dd,_),v in sums.items() if dd==d), 1)
        return_row.append([date_,"SSS", "All Station", "", "",sum_mm,])

    return_row.sort(key=lambda r: (r[0], r[1]))

    return return_row

def normalise_24h_forecast_reading():
    url = "https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast"
    forecast_data = fetch_data(url, raw = True)
    items = (forecast_data.get("data") or {}).get("records", [])

    output_rows = []

    for it in items:
        issued_ts = it.get("timestamp") or ""
        updated_ts = it.get("updatedTimestamp") or ""
        day_date   = it.get("date") or ""  

        vp = it.get("validPeriod") or it.get("valid_period") or {}
        period_start = vp.get("start") or ""
        period_end   = vp.get("end") or ""

        general = it.get("general") or {}

        periods = it.get("periods") or []
        for p in periods:
            t = p.get("time") or p.get("timePeriod") or {}
            p_start = t.get("start") or ""
            p_end   = t.get("end") or ""
            p_text  = t.get("text") or p.get("name") or ""

            regions = p.get("regions") or p.get("regionForecasts") or {}
            for region_key, val in (regions or {}).items():
                if isinstance(val, dict):
                    f_text = val.get("forecast") or val.get("text") or ""
                    f_code = val.get("forecastCode") or val.get("code") or ""
                else:
                    f_text = str(val) if val is not None else ""
                    f_code = ""
                
                output_rows.append([ day_date, issued_ts, updated_ts, p_start or period_start, p_end or period_end, p_text, str(region_key).lower(), f_text, f_code])

    return output_rows

def run_24h_forecast(sh):
    data = normalise_24h_forecast_reading()
    ws = ensure_worksheet(sh, "forecast_24h",["date","issued_ts","updated_ts","period_start","period_end","period_text","region","forecast_text","forecast_code"])

    append_unique(ws, data, key_cols=["date","region","period_start"])

def circular_std_deg(vals):
    """Circular spread of wind direction in degrees (0..180)."""
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().values
    if len(arr) == 0:
        return np.nan

    rads = np.deg2rad(arr.astype(float))
    R = np.hypot(np.sin(rads).sum(), np.cos(rads).sum()) / len(rads)
    if R <= 0:
        return 180.0

    return np.rad2deg(np.sqrt(-2 * np.log(R))) if R < 1 else 0.0

def modal(series):
    s = [str(x).strip() for x in series if x is not None and str(x).strip() != ""]
    if not s:
        return ""
    return Counter(s).most_common(1)[0][0]

def label_from_metrics(rain_mm, temp_c, rh_pct, wind_ms):
    rain = (rain_mm or 0.0)
    if rain >= 64:  
        return "Very Heavy Rain"
    if rain >= 31:  
        return "Heavy Rain"
    if rain >= 11:  
        return "Moderate Rain"
    if rain >= 1:   
        return "Light Rain"
    if rain > 0:   
        return "Very Light Rain"

    if (rh_pct is not None and temp_c is not None) and (rh_pct >= 90 and temp_c <= 25): 
        return "Mist"

    if (wind_ms or 0) >= 8: 
        return "Windy"

    if rh_pct is not None:
        if rh_pct >= 80: 
            return "Cloudy"
        if 60 <= rh_pct < 80: 
            return "Partly Cloudy"

    if temp_c is not None and temp_c >= 33: 
        return "Fair and Warm"

    return "Fair"

def build_daily_zone_weather_from_lists(
    zone_to_stations,
    rainfall_summary_value,
    mets_data,
    target_date_iso,
    rain_agg = "sum",  # "sum" or "mean" if you ever want to switch
):
    target_d = dt.date.fromisoformat(target_date_iso)

    # -------- per-station daily rainfall & ALL (SSS) --------
    per_station = {}    # sid -> total_mm on target date
    sss_value   = None  # keep SSS if provided
    for row in (rainfall_summary_value or []):
        if not row or len(row) < 6: 
            continue
        ts, sid, *_rest, total = row[:6]
        t = to_dt(ts)
        if not t or t.date() != target_d:
            continue
        sid = str(sid).strip()
        v = float(total or 0.0)
        if sid == "SSS":
            sss_value = v
        else:
            per_station[sid] = per_station.get(sid, 0.0) + v

    # ALL ZONE rain: prefer SSS if present; else aggregate all stations (average across stations)
    if sss_value is not None:
        station_count = len(per_station)
        if station_count > 0:
            all_zone_rain = float(sss_value) / station_count
        else:
            all_zone_rain = float(sss_value)
    else:
        vals = list(per_station.values())
        if not vals:
            all_zone_rain = np.nan
        else:
            all_zone_rain = float(np.mean(vals))

    # -------- authoritative zone list so every zone appears --------
    zones_list = list((zone_to_stations or {}).keys())
    base = pd.DataFrame({"zone_name": zones_list})

    # -------- zone rainfall (aggregate across mapped station ids) --------
    rain_rows = []
    for zname in zones_list:
        sids = [str(s).strip() for s in (zone_to_stations.get(zname) or [])]
        vals = [per_station.get(s) for s in sids if s in per_station]
        vals = [v for v in vals if v is not None]
        if not vals:
            agg_val = np.nan
        else:
            agg_val = float(np.mean(vals))
        rain_rows.append([zname, agg_val])
    rain_df = pd.DataFrame(rain_rows, columns=["zone_name","rain_mm"])

    # -------- METS per zone --------
    mets_df = pd.DataFrame(mets_data or [], columns=[
        "timestamp","station_id","station_name","temp_c","rh_pct","wind_ms","wind_dir_deg"
    ])
    if not mets_df.empty:
        mets_df["station_id"] = mets_df["station_id"].astype(str).str.strip()
        for c in ["temp_c","rh_pct","wind_ms","wind_dir_deg"]:
            mets_df[c] = pd.to_numeric(mets_df[c], errors="coerce")

    met_rows = []
    for zname in zones_list:
        sids = [str(s).strip() for s in (zone_to_stations.get(zname) or [])]
        zdf = mets_df[mets_df["station_id"].isin(sids)] if (not mets_df.empty and sids) else pd.DataFrame()
        if zdf.empty:
            met_rows.append([zname, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        else:
            met_rows.append([
                zname,
                float(zdf["temp_c"].max()),
                float(zdf["temp_c"].min()),
                float(zdf["rh_pct"].mean()),
                float(zdf["wind_ms"].mean()),
                float(zdf["wind_ms"].max()),
                float(circular_std_deg(zdf["wind_dir_deg"].dropna().tolist())),
            ])
    met_df = pd.DataFrame(met_rows, columns=[
        "zone_name","max_temp","min_temp","rh_mean","wind_ms_mean","wind_ms_max","wind_dir_variability"
    ])

    # -------- merge on zone list (left joins keep every zone) --------
    merged = base.merge(rain_df, on="zone_name", how="left") \
                 .merge(met_df,  on="zone_name", how="left")

    # -------- label (works even if rain_mm is NaN) --------
    merged.insert(0, "date", target_date_iso)
    merged["weather_label"] = [
        label_from_metrics(
            rain_mm = r.get("rain_mm"),
            temp_c  = r.get("max_temp"),
            rh_pct  = r.get("rh_mean"),
            wind_ms = r.get("wind_ms_mean"),
        )
        for _, r in merged.iterrows()
    ]

    # -------- order + rounding --------
    merged = merged[[
        "date","zone_name","weather_label","rain_mm","max_temp","min_temp",
        "rh_mean","wind_ms_mean","wind_ms_max","wind_dir_variability"
    ]]
    for c in ["rain_mm","max_temp","min_temp","rh_mean","wind_dir_variability"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").round(1)
    merged["wind_ms_mean"] = pd.to_numeric(merged["wind_ms_mean"], errors="coerce").round(2)
    merged["wind_ms_max"]  = pd.to_numeric(merged["wind_ms_max"],  errors="coerce").round(2)

    # -------- ALL ZONE row --------
    if mets_df.empty:
        all_met = [np.nan]*6
    else:
        all_met = [
            float(mets_df["temp_c"].max()),
            float(mets_df["temp_c"].min()),
            float(mets_df["rh_pct"].mean()),
            float(mets_df["wind_ms"].mean()),
            float(mets_df["wind_ms"].max()),
            float(circular_std_deg(mets_df["wind_dir_deg"].dropna().tolist())),
        ]
    all_row = pd.DataFrame([{
        "date": target_date_iso,
        "zone_name": "All Zone",
        "weather_label": label_from_metrics(
            rain_mm = np.round(all_zone_rain, 1) if np.isfinite(all_zone_rain) else np.nan,
            temp_c  = all_met[0] if np.isfinite(all_met[0]) else np.nan,
            rh_pct  = all_met[2] if np.isfinite(all_met[2]) else np.nan,
            wind_ms = all_met[3] if np.isfinite(all_met[3]) else np.nan,
        ),
        "rain_mm": np.round(all_zone_rain, 1) if np.isfinite(all_zone_rain) else np.nan,
        "max_temp": np.round(all_met[0], 1) if np.isfinite(all_met[0]) else np.nan,
        "min_temp": np.round(all_met[1], 1) if np.isfinite(all_met[1]) else np.nan,
        "rh_mean":  np.round(all_met[2], 1) if np.isfinite(all_met[2]) else np.nan,
        "wind_ms_mean": np.round(all_met[3], 2) if np.isfinite(all_met[3]) else np.nan,
        "wind_ms_max":  np.round(all_met[4], 2) if np.isfinite(all_met[4]) else np.nan,
        "wind_dir_variability": np.round(all_met[5], 1) if np.isfinite(all_met[5]) else np.nan,
    }])

    out = pd.concat([merged, all_row], ignore_index=True)
    out = out.sort_values(by=["zone_name"]).reset_index(drop=True)
    return out


def write_daily_zone_weather(sh, df):
    """
    Upsert df into 'daily_zone_weather' on (date, zone_name).
    """
    headers = [ "date", "zone_name", "weather_label", "rain_mm", "max_temp", "min_temp", "rh_mean", "wind_ms_mean", "wind_ms_max", "wind_dir_variability"]
    ws = ensure_worksheet(sh, "daily_zone_weather", headers)
    data = ws.get_all_values()

    if data:
        cur_hdr = data[0]
        cur_rows = data[1:]
        cur = pd.DataFrame(cur_rows, columns=cur_hdr)
    else:
        cur = pd.DataFrame(columns=headers)

    # normalize dtypes
    if "date" not in cur.columns:
        cur = pd.DataFrame(columns=headers)

    # drop existing rows for that date (we upsert *the whole day* set)
    tgt_date = str(df.iloc[0]["date"])
    cur = cur[cur["date"] != tgt_date]

    # append new
    merged = pd.concat([cur, df], ignore_index=True)

    # sort
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.sort_values(["date","zone_name"]).reset_index(drop=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    # write back
    ws.clear()
    ws.append_row(headers, value_input_option="USER_ENTERED")
    rows = merged.astype(object).where(pd.notnull(merged), "").values.tolist()
    if rows:
        for i in range(0, len(rows), 500):
            ws.append_rows(rows[i:i+500], value_input_option="USER_ENTERED")

def tod_label(dt_obj):
    hour = dt_obj.hour
    for label, start, end in TIME_OF_DAY_ORDER:
        if start <= hour < end:
            return label
    return "Late Night"

def build_tod_daily_zone_weather_from_lists(
    zone_to_stations,
    rainfall_rows,
    mets_rows,
    target_date_iso):
    target_d = dt.date.fromisoformat(target_date_iso)
    station_to_zones = {}
    for zone, stations in (zone_to_stations or {}).items():
        if not zone:
            continue
        for sid in stations or []:
            sid_str = str(sid).strip()
            if not sid_str:
                continue
            station_to_zones.setdefault(sid_str, []).append(zone)
    zone_names = sorted((zone_to_stations or {}).keys())
    buckets = [label for label, _, _ in TIME_OF_DAY_ORDER]

    # Collect rainfall per station per bucket
    station_bucket = {}
    for row in rainfall_rows or []:
        sid = str(row[0]).strip()
        t = to_dt(row[5])
        if not sid or not t or t.date() != target_d:
            continue
        bucket = tod_label(t)
        station_bucket.setdefault((sid, bucket), []).append(float(row[4] or 0))

    # Aggregate to zone averages per bucket
    zone_bucket_rain = {}
    for (sid, bucket), vals in station_bucket.items():
        zones = station_to_zones.get(sid, [])
        if not zones:
            continue
        avg_val = float(np.mean(vals))
        for zone in zones:
            key = (zone, bucket)
            zone_bucket_rain.setdefault(key, []).append(avg_val)

    # METS metrics per zone/bucket
    zone_bucket_metrics = {}
    for row in mets_rows or []:
        ts, sid, *_rest, temp, rh, wind_ms, wind_dir = row[:7]
        t = to_dt(ts)
        sid = str(sid).strip()
        if not sid or not t or t.date() != target_d:
            continue
        bucket = tod_label(t)
        zones = station_to_zones.get(sid, [])
        for zone in zones:
            metrics = zone_bucket_metrics.setdefault((zone, bucket), {"temp": [], "rh": [], "wind": [], "wind_dir": []})
            if temp not in ("", None):
                metrics["temp"].append(float(temp))
            if rh not in ("", None):
                metrics["rh"].append(float(rh))
            if wind_ms not in ("", None):
                metrics["wind"].append(float(wind_ms))
            if wind_dir not in ("", None):
                metrics["wind_dir"].append(float(wind_dir))

    rows = []
    for bucket in buckets:
        for zone in zone_names:
            rain_vals = zone_bucket_rain.get((zone, bucket), [])
            rain_mm = float(np.mean(rain_vals)) if rain_vals else np.nan
            metrics = zone_bucket_metrics.get((zone, bucket), {})
            max_temp = float(np.max(metrics.get("temp", []))) if metrics.get("temp") else np.nan
            min_temp = float(np.min(metrics.get("temp", []))) if metrics.get("temp") else np.nan
            rh_mean = float(np.mean(metrics.get("rh", []))) if metrics.get("rh") else np.nan
            wind_mean = float(np.mean(metrics.get("wind", []))) if metrics.get("wind") else np.nan
            wind_max = float(np.max(metrics.get("wind", []))) if metrics.get("wind") else np.nan
            wind_var = float(circular_std_deg(metrics.get("wind_dir", []))) if metrics.get("wind_dir") else np.nan
            rows.append([
                target_date_iso,
                bucket,
                zone,
                label_from_metrics(rain_mm, max_temp, rh_mean, wind_mean),
                rain_mm,
                max_temp,
                min_temp,
                rh_mean,
                wind_mean,
                wind_max,
                wind_var,
            ])
        # aggregate All Zone row for this bucket
        all_rain_vals = []
        all_metrics = {"temp": [], "rh": [], "wind": [], "wind_dir": []}
        for zone in zone_names:
            all_rain_vals.extend(zone_bucket_rain.get((zone, bucket), []))
            metrics = zone_bucket_metrics.get((zone, bucket), {})
            for key in all_metrics:
                all_metrics[key].extend(metrics.get(key, []))
        rain_all = float(np.mean(all_rain_vals)) if all_rain_vals else np.nan
        max_temp = float(np.max(all_metrics["temp"])) if all_metrics["temp"] else np.nan
        min_temp = float(np.min(all_metrics["temp"])) if all_metrics["temp"] else np.nan
        rh_mean = float(np.mean(all_metrics["rh"])) if all_metrics["rh"] else np.nan
        wind_mean = float(np.mean(all_metrics["wind"])) if all_metrics["wind"] else np.nan
        wind_max = float(np.max(all_metrics["wind"])) if all_metrics["wind"] else np.nan
        wind_var = float(circular_std_deg(all_metrics["wind_dir"])) if all_metrics["wind_dir"] else np.nan
        rows.append([
            target_date_iso,
            bucket,
            "All Zone",
            label_from_metrics(rain_all, max_temp, rh_mean, wind_mean),
            rain_all,
            max_temp,
            min_temp,
            rh_mean,
            wind_mean,
            wind_max,
            wind_var,
        ])
    return pd.DataFrame(rows, columns=[
        "date","time_of_day","zone_name","weather_label","rain_mm",
        "max_temp","min_temp","rh_mean","wind_ms_mean","wind_ms_max","wind_dir_variability"
    ])

def write_zone_weather_with_tod(sh, df):
    headers = ["date","time_of_day","zone_name","weather_label","rain_mm",
               "max_temp","min_temp","rh_mean","wind_ms_mean","wind_ms_max","wind_dir_variability"]
    ws = ensure_worksheet(sh, "tod_daily_zone_weather", headers)
    data = ws.get_all_values()
    existing_keys = set()
    if data:
        header = data[0]
        idx = {name: header.index(name) for name in headers}
        for row in data[1:]:
            if len(row) < len(headers):
                continue
            key = (row[idx["date"]], row[idx["time_of_day"]], row[idx["zone_name"]])
            existing_keys.add(key)
    else:
        ws.append_row(headers, value_input_option="USER_ENTERED")

    prepared = df.astype(object).where(pd.notnull(df), "").values.tolist()
    to_append = []
    for row in prepared:
        key = (row[0], row[1], row[2])
        if key in existing_keys:
            continue
        to_append.append(row)
        existing_keys.add(key)

    if to_append:
        for i in range(0, len(to_append), 500):
            ws.append_rows(to_append[i:i+500], value_input_option="USER_ENTERED")
            
def load_zone_to_stations_from_sheet(sh, sheet_name="zone_station_map"):
    """
    Expects sheet with columns:
      zone_id | zone_name | station_id | station_name | station_lat | station_lon
    Returns: dict { zone_name (exact) : [station_id, ...] }
    """
    ws = ensure_worksheet(sh, sheet_name,
                          ["zone_id","zone_name","station_id","station_name","station_lat","station_lon"])
    vals = ws.get_all_values()
    z2s = {}
    if not vals or len(vals) < 2:
        return z2s
    hdr = vals[0]
    idx = {h: i for i, h in enumerate(hdr)}
    for row in vals[1:]:
        try:
            zname = row[idx["zone_name"]].strip()
            sid   = row[idx["station_id"]].strip()
        except Exception:
            continue
        if not zname or not sid:
            continue
        z2s.setdefault(zname, []).append(sid)
    return z2s


if __name__ == "__main__":
    # open sheet
    sh = open_sheet_by_id(SHEET_ID)

    run_24h_forecast(sh)
    run_2h_forecast(sh)

    _, rainfall_area = run_rainfall(sh)

    run_mets(sh, rainfall_area)

    tdy_date_str = dt.datetime.now(SGT).date().isoformat()
    ytd_date = dt.datetime.now(SGT).date() - dt.timedelta(days=1)
    ytd_date_str = ytd_date.isoformat()

    start_date = ytd_date_str
    end_date = tdy_date_str

    zone_to_stations = load_zone_to_stations_from_sheet(sh, sheet_name="zone_station_map")

    for n, d in enumerate(iter_dates_inclusive(start_date, end_date), start=1):
        
        ds = d.isoformat()
        rainfall_value, rainfall_area = run_rainfall(sh, ds)
        rainfall_summary_headers = ["timestamp","station_id","station_name","lat","lon","rain_total_mm"]
        rainfall_summary_ws = ensure_worksheet(sh, "rainfall_daily_combined", rainfall_summary_headers)

        if not rainfall_summary_ws.get_all_values():
            rainfall_summary_ws.append_row(rainfall_summary_headers, value_input_option="USER_ENTERED")

        rainfall_summary_value = rainfall_summary_processing(rainfall_value or [])
        append_unique(rainfall_summary_ws, rainfall_summary_value, key_cols=["timestamp","station_id"])
        mets_data = run_mets(sh, rainfall_area, ds, write_to_sheet = False)
        df = build_daily_zone_weather_from_lists(zone_to_stations, rainfall_summary_value, mets_data, ds)
        write_daily_zone_weather(sh, df)

        df_tod = build_tod_daily_zone_weather_from_lists(zone_to_stations, rainfall_value, mets_data, ds)
        write_zone_weather_with_tod(sh, df_tod)

        # Soft throttle every 5 days processed
        if n % 5 == 0:
            time.sleep(2.0)
