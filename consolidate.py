import os, requests
import datetime as dt
from sheets_utils import open_sheet, ensure_worksheet
from typing import List, Dict, Tuple
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from datetime import date, timedelta
from collections import defaultdict
import math
import time

load_dotenv()

SGT = dt.timezone(dt.timedelta(hours=8))
# BASE = os.getenv("DGS_REALTIME_BASE", "https://api-open.data.gov.sg/v2/real-time/api")
UA = os.getenv("USER_AGENT", "sg-weather-collector/1.0")
API_KEY = os.getenv("DATA_GOV_SG_API_KEY")
SHEET_ID = os.environ["SHEET_ID"]

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
    # reading_type  = "rainfall"
    buckets       = d.get("readings", []) or []

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

    return obs_rows, list(st_rows_map.values())

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
    rainfall_value, rainfall_area = rainfall_data_reading(rainfall_json_data)

    rainfall_value_ws_frame = ensure_worksheet(sh, "rainfall_data", ["station_id","station_name","lat","lon","reading_value","reading_time"])
    append_unique(rainfall_value_ws_frame,rainfall_value,key_cols=["station_id","reading_time"])

    rainfall_station_ws_frame = ensure_worksheet(sh, "stations", ["station_id","station_name","lat","lon"])

    if rainfall_area:
        # refresh stations each run (matches your current behavior)
        rainfall_station_ws_frame.clear()
        rainfall_station_ws_frame.append_row(["station_id","station_name","lat","lon"])
        rainfall_station_ws_frame.append_rows(rainfall_area, value_input_option="USER_ENTERED")

    return rainfall_value, rainfall_area


def normalise_mets_data_reading(readings):
    rows = []
    
    for bucket in readings or []:
        ts = bucket.get("timestamp")
        for d in (bucket.get("data") or []):
            sid = d.get("stationId"); val = d.get("value")
            if ts and sid is not None and val is not None:
                rows.append((ts, str(sid), float(val)))
    return rows

def run_mets(sh, rainfall_area, date_str = None):
    """
    Uses v2 METS, writes to 'mets_data' with your headers.
    Requires rainfall_area to build station_name_map.
    """
    # METS DATA
    if date_str != None:
        date_param = {"date": date_str}
    else:
        date_param = None

    temperature_url = "https://api-open.data.gov.sg/v2/real-time/api/air-temperature"
    temperature_json_data = fetch_data(temperature_url, params = date_param)
    temperature_data = normalise_mets_data_reading(temperature_json_data.get("readings"))

    humidity_url = "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity"
    humidity_json_data = fetch_data(humidity_url, params = date_param)
    humidity_data = normalise_mets_data_reading(humidity_json_data.get("readings"))

    wind_dir_url = "https://api-open.data.gov.sg/v2/real-time/api/wind-direction"
    wind_dir_json_data = fetch_data(wind_dir_url, params = date_param)
    wind_dir_data = normalise_mets_data_reading(wind_dir_json_data.get("readings"))

    wind_speed_url = "https://api-open.data.gov.sg/v2/real-time/api/wind-speed"
    wind_speed_json_data = fetch_data(wind_speed_url, params = date_param)
    wind_speed_data = normalise_mets_data_reading(wind_speed_json_data.get("readings"))

    tmap = {(ts, sid): val for ts, sid, val in temperature_data}
    hmap = {(ts, sid): val for ts, sid, val in humidity_data}
    spm  = {(ts, sid): val for ts, sid, val in wind_speed_data}  # speed (m/s)
    dgm  = {(ts, sid): val for ts, sid, val in wind_dir_data}    # dir (deg)
    keys = sorted(set(tmap) | set(hmap) | set(spm) | set(dgm))

    station_name_map = {item[0]: item[1] for item in rainfall_area if len(item) >= 2}
    mets_data = []

    for ts, sid in keys:
        name = (station_name_map or {}).get(sid, "")
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

    mets_headers = ["timestamp","station_id","station_name","temp_value_celcius", "humidity_value_percentage","wind_speed_ms","wind_dir_deg"]
    mets_value_ws_frame = ensure_worksheet(sh, "mets_data", mets_headers)
    append_unique(mets_value_ws_frame, mets_data, key_cols=["timestamp", "station_id"])

    return mets_data

def backfill_rainfall_processing(rainfall_df):
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

def backfill_rainfall(sh, rainfall_df):
    """
    Ensures a single tab with both station-level daily totals and one 'All Station' row per day.
    """
    headers = ["timestamp","station_id","station_name","lat","lon","rain_total_mm"]
    ws = ensure_worksheet(sh, "rainfall_daily_combined", headers)
    rows = backfill_rainfall_processing(rainfall_df)
    # dedupe by (timestamp, station_id) so the all-station row stays unique per day
    append_unique(ws, rows, key_cols=["timestamp","station_id"])
    
    return len(rows)


if __name__ == "__main__":
    # open sheet
    sh = open_sheet_by_id(SHEET_ID)

    """
    run_24h_forecast(sh)
    run_2h_forecast(sh)

    _, rainfall_area = run_rainfall(sh)

    run_mets(sh, rainfall_area)

    ytd_date = dt.datetime.now(SGT).date() - dt.timedelta(days=1)
    ytd_date_str = ytd_date.isoformat()

    start_date = ytd_date_str
    end_date = ytd_date_str
    """

    start_date = "2025-07-01"
    end_date = None

    for n, d in enumerate(iter_dates_inclusive(start_date, end_date), start=1):
        
        ds = d.isoformat()
        rainfall_value, rainfall_area = run_rainfall(sh, ds)
        print(ds,": pulling rainfall area")
        #wrote = backfill_rainfall(sh, rainfall_value or [])

        run_mets(sh, rainfall_area, ds)
        print("Done writing mets")

        # Soft throttle every 5 days processed
        if n % 5 == 0:
            time.sleep(2.0)


    #d = fetch_data("https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast", params={"date":"2025-11-02"}, raw= True)
    #print("keys in data:", list(d.keys())) # expect ['items', 'paginationToken'] (token may be absent)
    #print(d.get("data"))
    # print((d.get("data") or {}).get("items", []))

    # 1) 2-hour forecast (writes forecasts_2h, areas)
    #run_forecast(sh)


    # 2) rainfall for today (writes rainfall_data, stations)
    #_, rainfall_area = run_rainfall(sh, "2025-10-31")

    # 3) mets uses the station map from rainfall_area (writes mets_data)
    #run_mets(sh, rainfall_area)

    # OPTIONAL: backfill a range (uncomment if needed)
    #today = dt.datetime.now(SGT).date().isoformat()
    #last_week = (dt.datetime.now(SGT).date() - dt.timedelta(days=7)).isoformat()
    #backfill_rainfall_and_mets(sh, "2025-10-27", "2025-10-27")

    # aggregating yesterday rainfall data
    # 3. Calculate yesterday's date
    """
    ytd_date = dt.datetime.now(SGT).date() - dt.timedelta(days=1)
    ytd_date_str = ytd_date.isoformat()

    start_date = ytd_date_str
    end_date = ytd_date_str

    for n, d in enumerate(iter_dates_inclusive(start_date, end_date), start=1):
        
        ds = d.isoformat()
        rainfall_value, rainfall_area = run_rainfall(sh, ds)
        wrote = backfill_rainfall(sh, rainfall_value or [])

        # Soft throttle every 5 days processed
        if n % 5 == 0:
            time.sleep(2.0)
    



def build_ranfall_daily_all_from_station_daily(station_daily_rows):
    buckets = defaultdict(list)
    for r in station_daily_rows or []:
        if not r or len(r) < 6: 
            continue
        sid, sname, lat, lon, total_mm, tstr = r[:6]
        d = _to_dt(str(tstr))
        if not d:
            continue
        try:
            v = float(total_mm)
        except:
            continue
        buckets[d.date()].append(v)

    return_list = []
    for d, vals in sorted(buckets.items()):
        if not vals:
            continue
        mean_v = round(sum(vals)/len(vals), 1)
        max_v  = round(max(vals), 1)
        sum_v  = round(sum(vals), 1)
        return_list.append([_day_stamp(d), mean_v, max_v, sum_v])

    return return_list

def write_rainfall_daily_all(sh, station_daily_rows):
    ws = ensure_worksheet(
        sh, "rainfall_daily_all",
        ["timestamp","rain_mean_mm_all_stations","rain_max_mm_any_station","rain_sum_mm_all_stations"])

    rows = build_rainfall_daily_all_from_station_daily(station_daily_rows)
    append_unique(ws, rows, key_cols=["timestamp"])
    return len(rows)

# FOR BACKFILLING 

def _norm_v2_readings(readings):
    rows = []
    for bucket in readings or []:
        ts = bucket.get("timestamp")
        for d in (bucket.get("data") or []):
            sid = d.get("stationId"); val = d.get("value")
            if ts and sid is not None and val is not None:
                rows.append((ts, str(sid), float(val)))
    return rows

def _fetch_day_paged_full(url_base, date_str):
    out = {"readings": []}
    while True:
        url = f"{url_base}?date={date_str}"
        data = fetch_data(url)  # returns .data
        out["readings"].extend(data.get("readings") or [])
        if "stations" in data: out["stations"] = data.get("stations")
        token = data.get("paginationToken") or data.get("pagination_token")
        if not token:
            break
    return out

def _build_mets_rows_for_day(date_str, name_map):
    temp_day = _fetch_day_paged_full("https://api-open.data.gov.sg/v2/real-time/api/air-temperature", date_str)
    hum_day  = _fetch_day_paged_full("https://api-open.data.gov.sg/v2/real-time/api/relative-humidity", date_str)
    ws_day   = _fetch_day_paged_full("https://api-open.data.gov.sg/v2/real-time/api/wind-speed", date_str)
    wd_day   = _fetch_day_paged_full("https://api-open.data.gov.sg/v2/real-time/api/wind-direction", date_str)

    trows = _norm_v2_readings(temp_day.get("readings"))
    hrows = _norm_v2_readings(hum_day.get("readings"))
    srows = _norm_v2_readings(ws_day.get("readings"))
    drows = _norm_v2_readings(wd_day.get("readings"))

    tmap = {(ts, sid): v for ts, sid, v in trows}
    hmap = {(ts, sid): v for ts, sid, v in hrows}
    smap = {(ts, sid): v for ts, sid, v in srows}
    dmap = {(ts, sid): v for ts, sid, v in drows}

    keys = sorted(set(tmap) | set(hmap) | set(smap) | set(dmap))
    out = []
    for ts, sid in keys:
        name = name_map.get(sid, "")
        tv = tmap.get((ts, sid)); hv = hmap.get((ts, sid))
        sv = smap.get((ts, sid)); dv = dmap.get((ts, sid))
        out.append([ts, sid, name,
                    "" if tv is None else tv,
                    "" if hv is None else hv,
                    "" if sv is None else sv,
                    "" if dv is None else dv])
    return out

def backfill_rainfall_and_mets(sh, d0, d1):
    
    Inclusive dates, default = today..today (SGT).
    d0/d1 format: 'YYYY-MM-DD'
    
    if not d0 or not d1:
        today = dt.datetime.now(SGT).date().isoformat()
        d0 = d0 or today
        d1 = d1 or today

    d0_dt = dt.date.fromisoformat(d0)
    d1_dt = dt.date.fromisoformat(d1)

    rainfall_ws = ensure_worksheet(
        sh, "rainfall_data",
        ["station_id","station_name","lat","lon","reading_value","reading_time"]
    )
    stations_ws = ensure_worksheet(
        sh, "stations", ["station_id","station_name","lat","lon"]
    )
    mets_ws = ensure_worksheet(
        sh, "mets_data",
        ["timestamp","station_id","station_name","temp_value_celcius",
         "humidity_value_percentage","wind_speed_ms","wind_dir_deg"]
    )

    cur = d0_dt
    while cur <= d1_dt:
        ds = cur.isoformat()
        # rainfall (day, paged)
        rain_day = _fetch_day_paged_full("https://api-open.data.gov.sg/v2/real-time/api/rainfall", ds)
        rain_json_for_reader = {
            "readings": rain_day.get("readings") or [],
            "stations": rain_day.get("stations") or [],
        }
        rainfall_value, rainfall_area = rainfall_data_reading(rain_json_for_reader)

        if rainfall_value:
            append_unique(rainfall_ws, rainfall_value, key_cols=["station_id","reading_time"])

        if rainfall_area:
            append_unique(stations_ws, rainfall_area, key_cols=["station_id"])

        # build name map from that day's stations
        station_name_map = {r[0]: r[1] for r in (rainfall_area or []) if isinstance(r, (list, tuple)) and len(r) >= 2}

        mets_rows = _build_mets_rows_for_day(ds, station_name_map)
        if mets_rows:
            append_unique(mets_ws, mets_rows, key_cols=["timestamp","station_id"])

        print(f"[backfill {ds}] rain={len(rainfall_value or [])}, stations={len(rainfall_area or [])}, mets={len(mets_rows or [])}")
        cur += dt.timedelta(days=1)



forecast_url = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
forecast_json_data = fetch_data(forecast_url)
forecast_value, forecast_area = forecast_value_reading(forecast_json_data)

# open sheet
sh = open_sheet_by_id(SHEET_ID)

# appending value data
forecast_value_ws_frame = ensure_worksheet(sh, "forecasts_2h", ["area","valid_from","valid_to","forecast","issued_at"])
append_unique(forecast_value_ws_frame, forecast_value, key_cols=["area","valid_from","valid_to"])

# appending area data
forecast_area_ws_frame =  ensure_worksheet(sh, "areas", ["area","lat","lon"])
if forecast_area:
    forecast_area_ws_frame.clear()
    forecast_area_ws_frame.append_row(["area","lat","lon"])
    forecast_area_ws_frame.append_rows(forecast_area, value_input_option="USER_ENTERED")

# RAINFALL DATA
rainfall_url = "https://api-open.data.gov.sg/v2/real-time/api/rainfall?date=2025-11-02"
rainfall_json_data = fetch_data(rainfall_url)
rainfall_value, rainfall_area = rainfall_data_reading(rainfall_json_data)

rainfall_value_ws_frame = ensure_worksheet(sh, "rainfall_data", ["station_id","station_name","lat","lon","reading_value","reading_time"])
append_unique(rainfall_value_ws_frame, rainfall_value, key_cols=["station_id","reading_time"]) # , headers=obs_headers)

rainfall_station_ws_frame = ensure_worksheet(sh, "stations", ["station_id","station_name","lat","lon"])
if rainfall_area:
    rainfall_station_ws_frame.clear()
    rainfall_station_ws_frame.append_row(["station_id","station_name","lat","lon"])
    rainfall_station_ws_frame.append_rows(rainfall_area, value_input_option="USER_ENTERED")


# METS DATA

temperature_url = "https://api-open.data.gov.sg/v2/real-time/api/air-temperature"
temperature_json_data = fetch_data(temperature_url)
temperature_data = normalise_mets_data_reading(temperature_json_data.get("readings"))

humidity_url = "https://api-open.data.gov.sg/v2/real-time/api/relative-humidity"
humidity_json_data = fetch_data(humidity_url)
humidity_data = normalise_mets_data_reading(humidity_json_data.get("readings"))

wind_dir_url = "https://api-open.data.gov.sg/v2/real-time/api/wind-direction"
wind_dir_json_data = fetch_data(wind_dir_url)
wind_dir_data = normalise_mets_data_reading(wind_dir_json_data.get("readings"))

wind_speed_url = "https://api-open.data.gov.sg/v2/real-time/api/wind-speed"
wind_speed_json_data = fetch_data(wind_speed_url)
wind_speed_data = normalise_mets_data_reading(wind_speed_json_data.get("readings"))

#wind_dir = {(ts, sid): v for ts, sid, v in wind_dir_data}
#wind_speed = {(ts, sid): v for ts, sid, v in wind_speed_data}

tmap = {(ts, sid): val for ts, sid, val in temperature_data}
hmap = {(ts, sid): val for ts, sid, val in humidity_data}
spm  = {(ts, sid): val for ts, sid, val in wind_speed_data}  # speed (m/s)
dgm  = {(ts, sid): val for ts, sid, val in wind_dir_data}  # dir (deg)
keys = sorted(set(tmap) | set(hmap) | set(spm) | set(dgm))

station_name_map = {item[0]: item[1] for item in rainfall_area if len(item) >= 2}
mets_data = []

for ts, sid in keys:
    name = (station_name_map or {}).get(sid, "")
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

mets_headers = ["timestamp","station_id","station_name","temp_value_celcius", "humidity_value_percentage","wind_speed_ms","wind_dir_deg"]
mets_value_ws_frame = ensure_worksheet(sh, "mets_data", mets_headers)
append_unique(mets_value_ws_frame, mets_data, key_cols=["timestamp", "station_id"])
"""