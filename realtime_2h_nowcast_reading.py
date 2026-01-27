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

load_dotenv()

SGT = dt.timezone(dt.timedelta(hours=8))
UA = os.getenv("USER_AGENT", "sg-weather-collector/1.0")
API_KEY = os.getenv("DATA_GOV_SG_API_KEY")
SHEET_ID = os.environ["SHEET_ID"]
FORECAST_RETENTION_DAYS = 7

RAIN_LIKE_FORECASTS = {
    "Heavy Thundery Showers with Gusty Winds",
    "Heavy Thundery Showers",
    "Thundery Showers",
    "Heavy Showers",
    "Heavy Rain",
    "Moderate Rain",
    "Showers",
    "Light Showers"
    "Light Rain",
}

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

def fetch_data(url, raw = False, params = None): 
    # raw is only for V1 -- https://api.data.gov.sg/v1/environment/
    params = params or {}
    requestor = requests.get(url, headers=_headers(), params = params, timeout=30)
    requestor.raise_for_status()
    json_ = requestor.json() or {}

    if raw:
        return json_

    df = json_.get("data") or {}
    return df

def forecast_value_reading(raw_data):
    """
    Returns (forecast_rows, area_rows, pagination_token).
    Mirrors rainfall pagination handling.
    """
    if isinstance(raw_data, dict) and "items" in raw_data:
        d = raw_data
    else:
        d = (raw_data or {}).get("data", {}) or {}

    area_meta = d.get("area_metadata") or []
    items = d.get("items") or []
    pagination = d.get("paginationToken")

    area_rows = []
    seen = set()
    for a in area_meta:
        name = a.get("name")
        if not name or name in seen:
            continue
        ll = a.get("label_location") or {}
        area_rows.append([name, ll.get("latitude"), ll.get("longitude")])
        seen.add(name)

    forecast_rows = []
    for item in items:
        vp = item.get("valid_period") or {}
        valid_from = vp.get("start") or item.get("valid_from")
        valid_to = vp.get("end") or item.get("valid_to")
        issued_at = item.get("update_timestamp") or item.get("timestamp")
        for fx in item.get("forecasts") or []:
            forecast_rows.append([fx.get("area"), valid_from, valid_to, fx.get("forecast"), issued_at])

    return forecast_rows, area_rows, pagination

def load_area_to_zone_map(sh):
    ws = ensure_worksheet(sh, "zone_area_map", ["zone_id","zone_name","area","area_lat","area_lon"])
    values = ws.get_all_values()
    if len(values) < 2:
        return {}
    header = values[0]
    idx = {h: i for i, h in enumerate(header)}
    mapping = {}
    for row in values[1:]:
        try:
            area = row[idx["area"]].strip()
        except Exception:
            continue
        if not area:
            continue
        zone_name = row[idx["zone_name"]] if "zone_name" in idx else ""
        mapping[area] = zone_name.strip()
    return mapping

def get_existing_forecast_dates(sh, sheet_name="nowcasts_2h_data"):
    ws = ensure_worksheet(sh, sheet_name, ["area","valid_from","valid_to","forecast","issued_at"])
    values = ws.get_all_values()
    dates = set()
    if len(values) <= 1:
        return dates
    header = values[0]
    if "valid_from" not in header:
        return dates
    idx = header.index("valid_from")
    for row in values[1:]:
        if len(row) <= idx:
            continue
        dt_val = to_dt(row[idx])
        if not dt_val:
            continue
        dates.add(dt_val.date())
    return dates

def rewrite_sheet(ws, rows):
    ws.clear()
    if rows:
        ws.update(values=rows, range_name="A1", value_input_option="USER_ENTERED")


def upsert_forecast_rows(ws, headers, new_rows, key_cols, sort_cols=None):
    data = ws.get_all_values()
    if data:
        existing_header = data[0]
        rows = data[1:]
    else:
        ws.append_row(headers, value_input_option="USER_ENTERED")
        existing_header = headers
        rows = []

    if existing_header != headers:
        # rewrite header if mismatch
        existing_header = headers
        rows = []
        ws.clear()
        ws.append_row(headers, value_input_option="USER_ENTERED")

    key_positions = [existing_header.index(col) for col in key_cols if col in existing_header]
    if not key_positions:
        # nothing to de-duplicate, just append as-is
        combined = rows + new_rows

    else:
        new_keys = set()
        normalized_new = []
        for row in new_rows or []:
            row = list(row)
            if len(row) < len(existing_header):
                row += [""] * (len(existing_header) - len(row))
            elif len(row) > len(existing_header):
                row = row[:len(existing_header)]
            key = tuple(row[pos] for pos in key_positions)
            new_keys.add(key)
            normalized_new.append(row)

        filtered_existing = []
        for row in rows:
            row_key = tuple(row[pos] for pos in key_positions)
            if row_key in new_keys:
                continue
            filtered_existing.append(row)
        combined = filtered_existing + normalized_new

    # Keep only latest issued_at per key (area, valid_from, valid_to)
    if key_positions:
        issued_idx = existing_header.index("issued_at") if "issued_at" in existing_header else None
        best = {}
        for row in combined:
            key = tuple(row[pos] for pos in key_positions)
            if issued_idx is None:
                if key not in best:
                    best[key] = row
                continue
            issued_val = to_dt(row[issued_idx]) or row[issued_idx]
            prev = best.get(key)
            if prev is None:
                best[key] = row
            else:
                prev_issued = to_dt(prev[issued_idx]) or prev[issued_idx]
                if issued_val and prev_issued:
                    if isinstance(issued_val, dt.datetime) and isinstance(prev_issued, dt.datetime):
                        if issued_val >= prev_issued:
                            best[key] = row
                    else:
                        if str(issued_val) >= str(prev_issued):
                            best[key] = row
                else:
                    best[key] = row
        combined = list(best.values())

    rows_out = [headers] + combined
    rewrite_sheet(ws, rows_out)

def prune_sheet_by_date(ws, column_name, cutoff_date, parser):
    values = ws.get_all_values()
    if len(values) <= 1:
        return
    header = values[0]
    if column_name not in header:
        return
    idx = header.index(column_name)
    kept = [header]
    for row in values[1:]:
        if len(row) <= idx:
            continue
        parsed = parser(row[idx])
        if not parsed:
            continue
        if parsed >= cutoff_date:
            kept.append(row)
    if len(kept) == len(values):
        return
    rewrite_sheet(ws, kept)

def latest_forecast_rows(rows):
    """
    Keep only latest issued_at per (area, valid_from, valid_to).
    """
    if not rows:
        return []
    best = {}
    for row in rows:
        if len(row) < 5:
            continue
        area, valid_from, valid_to, forecast, issued_at = row
        key = (area, valid_from, valid_to)
        issued_val = to_dt(issued_at) or issued_at
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue
        prev_val = to_dt(prev[4]) or prev[4]
        if issued_val and prev_val:
            if isinstance(issued_val, dt.datetime) and isinstance(prev_val, dt.datetime):
                if issued_val >= prev_val:
                    best[key] = row
            else:
                if str(issued_val) >= str(prev_val):
                    best[key] = row
        else:
            best[key] = row
    return list(best.values())
def split_date_time(ts_str):
    dt_val = to_dt(ts_str)
    if not dt_val:
        return "", ""
    dt_val = dt_val.astimezone(SGT)
    return dt_val.date().isoformat(), dt_val.strftime("%H:%M")

def parse_valid_from_date(val):
    dt_val = to_dt(val)
    return dt_val.date() if dt_val else None

def parse_iso_date(val):
    try:
        return dt.date.fromisoformat(val.strip())
    except Exception:
        return None
    
def build_zone_outputs(rows, area_zone_map):
    rows = latest_forecast_rows(rows)
    zone_buckets = {}
    for row in rows or []:
        if len(row) < 5:
            continue
        area, valid_from, valid_to, forecast, issued_at = row
        zone_name = area_zone_map.get(area.strip()) if area else ""
        vf_date, vf_time = split_date_time(valid_from)
        vt_date, vt_time = split_date_time(valid_to)
        if not zone_name:
            continue
        bucket = zone_buckets.setdefault((zone_name, valid_from, valid_to), [])
        bucket.append({
            "forecast": forecast or "",
            "issued_at": issued_at or "",
            "vf_date": vf_date,
            "vf_time": vf_time,
            "vt_date": vt_date,
            "vt_time": vt_time,
        })

    zone_rows = []
    for (zone_name, valid_from, valid_to), entries in zone_buckets.items():
        if not entries:
            continue
        area_count = len(entries)
        rain_count = sum(
            1 for e in entries if (e["forecast"] or "").strip() in RAIN_LIKE_FORECASTS
        )
        ratio = round(rain_count / area_count, 3) if area_count else 0
        forecasts = [e["forecast"] for e in entries if e["forecast"]]
        forecast_label = ""
        if forecasts:
            counts = Counter(forecasts)
            if ratio >= 0.5:
                rain_candidates = [(label, count) for label, count in counts.items() if label.strip() in RAIN_LIKE_FORECASTS]
                if rain_candidates:
                    forecast_label = max(rain_candidates, key=lambda x: (x[1], x[0]))[0]
                else:
                    forecast_label = counts.most_common(1)[0][0]
            else:
                forecast_label = counts.most_common(1)[0][0]
        
        recommendation_label = "Review to deploy" if forecast_label in RAIN_LIKE_FORECASTS else ""

        issued_candidates = []
        for e in entries:
            ts = to_dt(e["issued_at"])
            if ts:
                issued_candidates.append((ts, e["issued_at"]))
            elif e["issued_at"]:
                issued_candidates.append((e["issued_at"], e["issued_at"]))
        if issued_candidates:
            issued_at = max(issued_candidates, key=lambda x: x[0])[1]
        else:
            issued_at = ""

        issued_date, issued_time = split_date_time(issued_at)
        
        # assume first entry covers date/time splits (all same valid window)
        ref = entries[0]
        zone_rows.append([
            zone_name,
            area_count,
            rain_count,
            ratio,
            forecast_label,
            ref["vf_date"],
            ref["vf_time"],
            ref["vt_date"],
            ref["vt_time"],
            issued_at,
            issued_date,
            issued_time,
            recommendation_label,
        ])

    return zone_rows

def drop_rows_for_date(ws, column_name, target_date, parser):
    if target_date is None:
        return
    values = ws.get_all_values()
    if len(values) <= 1:
        return
    header = values[0]
    if column_name not in header:
        return
    idx = header.index(column_name)
    kept = [header]
    for row in values[1:]:
        if len(row) <= idx:
            continue
        parsed = parser(row[idx])
        if parsed is None:
            kept.append(row)
            continue
        if parsed != target_date:
            kept.append(row)
    if len(kept) == len(values):
        return
    rewrite_sheet(ws, kept)


def run_2h_forecast(sh, date_str=None, refresh_areas=False):
    """
    Fetches and writes 2h forecasts for a given date (or latest if date_str is None).
    Set refresh_areas=True only when you want to overwrite the areas sheet.
    """
    forecast_url = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
    base_params = {}
    if date_str:
        base_params["date"] = date_str


    all_values = []
    all_areas = []
    area_seen = set()
    pagination = None

    while True:
        params = dict(base_params)
        if pagination:
            params["paginationToken"] = pagination
        forecast_json_data = fetch_data(forecast_url, params=params)
        forecast_value, forecast_area, pagination = forecast_value_reading(forecast_json_data)
        if forecast_value:
            all_values.extend(forecast_value)
        if forecast_area:
            for area in forecast_area:
                if not area or not area[0]:
                    continue
                if area[0] in area_seen:
                    continue
                area_seen.add(area[0])
                all_areas.append(area)
        if not pagination:
            break
    
    if all_values:
        target_date = None
        if date_str:
            try:
                target_date = dt.date.fromisoformat(date_str)
            except Exception:
                target_date = None
        else:
            target_date = dt.datetime.now(SGT).date()

        forecast_value_ws_frame = ensure_worksheet(sh, "nowcasts_2h_data", ["area","valid_from","valid_to","forecast","issued_at"])
        drop_rows_for_date(
            forecast_value_ws_frame,
            "valid_from",
            target_date,
            parse_valid_from_date,
        )
        upsert_forecast_rows(
            forecast_value_ws_frame,
            ["area","valid_from","valid_to","forecast","issued_at"],
            all_values,
            ["area","valid_from","valid_to"],
            sort_cols=["valid_from","area"],
        )

        cutoff = dt.datetime.now(SGT).date() - dt.timedelta(days=7)

        prune_sheet_by_date(
            forecast_value_ws_frame,
            "valid_from",
            cutoff,
            parse_valid_from_date,
        )

    # map to zones and write enriched sheets
        area_zone_map = load_area_to_zone_map(sh)
        zone_detail_rows = build_zone_outputs(all_values, area_zone_map)

        area_zone_ws = ensure_worksheet(sh, "nowcasts_2h_zone_data", 
                                        ["zone_name","area_count","rain_area_count","rain_ratio",
                                         "forecast","valid_from_date","valid_from_time","valid_to_date",
                                         "valid_to_time","issued_at", "issued_date","issued_time","recommendation"])
        drop_rows_for_date(
            area_zone_ws,
            "valid_from_date",
            target_date,
            parse_iso_date,
        )
        upsert_forecast_rows(
            area_zone_ws,
            ["zone_name","area_count","rain_area_count","rain_ratio",
             "forecast","valid_from_date","valid_from_time","valid_to_date",
             "valid_to_time","issued_at", "issued_date","issued_time","recommendation"],
            zone_detail_rows,
            ["zone_name","valid_from_date","valid_from_time","valid_to_date","valid_to_time"],
            sort_cols=["valid_from_date","valid_from_time","zone_name"],
        )
        prune_sheet_by_date(
            area_zone_ws,
            "valid_from_date",
            cutoff,
            parse_iso_date,
        )

    if refresh_areas and all_areas:
        forecast_area_ws_frame = ensure_worksheet(sh, "areas", ["area","lat","lon"])
        forecast_area_ws_frame.clear()
        forecast_area_ws_frame.append_row(["area","lat","lon"])
        forecast_area_ws_frame.append_rows(all_areas, value_input_option="USER_ENTERED")
    
def normalize_date_str(value: str, fallback: str) -> str:
    if not value:
        return fallback
    value = value.strip().lower()
    today = dt.datetime.now(SGT).date()
    if value == "today":
        return today.isoformat()
    if value == "yesterday":
        return (today - dt.timedelta(days=1)).isoformat()
    return value


if __name__ == "__main__":
    sh = open_sheet_by_id(SHEET_ID)

    today = dt.datetime.now(SGT).date()
    default_start = today - dt.timedelta(days=max(0, FORECAST_RETENTION_DAYS - 1))
    existing_dates = get_existing_forecast_dates(sh)

    dates_to_refresh = {today, today - dt.timedelta(days=1)}
    cursor = today - dt.timedelta(days=2)
    while cursor >= default_start:
        if cursor not in existing_dates:
            dates_to_refresh.add(cursor)
        cursor -= dt.timedelta(days=1)

    for n, day in enumerate(sorted(dates_to_refresh), start=1):
        ds = day.isoformat()
        print(f"Fetching 2h forecast for {ds}")
        run_2h_forecast(sh, ds)

        if n % 5 == 0:
            time.sleep(10.0)
