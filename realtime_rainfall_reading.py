import os
import datetime as dt
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from sheets_utils import ensure_worksheet
from consolidate import (
    append_unique,
    to_dt,
    iter_dates_inclusive,
    run_rainfall,
    open_sheet_by_id,
    load_zone_to_stations_from_sheet,
)

load_dotenv()

SGT = dt.timezone(dt.timedelta(hours=8))
SHEET_ID = os.environ["SHEET_ID"]

RAIN_SHEET_NAME = os.getenv("RAIN_SHEET_NAME", "rainfall_data")
RAIN_HEADERS = ["station_id", "station_name", "lat", "lon", "reading_value", "reading_time", "reading_label"]
RAIN_KEY_COLS = ["station_id", "reading_time"]

RAIN_ZONE_SHEET_NAME = os.getenv("RAIN_ZONE_SHEET_NAME", "rainfall_data_sg_zone")
RAIN_ZONE_HEADERS = ["zone_name", "reading_time", "reading_value", "reading_label"]
RAIN_ZONE_KEY_COLS = ["zone_name", "reading_time"]

ZONE_SHEET_NAME = os.getenv("ZONE_SHEET_NAME", "zone_station_map")

DEFAULT_LOOKBACK_DAYS = int(os.getenv("RAIN_INITIAL_LOOKBACK_DAYS", "1"))
POLL_MINUTES = max(1, int(os.getenv("RAIN_LOOP_MINUTES", "5")))
ENABLE_LOOP = os.getenv("RAIN_LOOP", "true").lower() not in {"0", "false", "no"}

def reading_label_from_mm(value):
    rain = float(value or 0.0)
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
    return "No Rain"

def get_last_timestamp(ws):
    values = ws.get_all_values()
    if len(values) <= 1:
        return None

    header = values[0]
    if "reading_time" not in header:
        return None

    time_idx = header.index("reading_time")
    latest_ts = None
    for row in values[1:]:
        if len(row) <= time_idx:
            continue
        ts = to_dt(row[time_idx])
        if not ts:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
    return latest_ts

def compute_fetch_window(last_ts):
    today = dt.datetime.now(SGT).date()
    if last_ts:
        start_date = last_ts.date()
    else:
        # allow a configurable initial lookback when the sheet is empty
        lookback = max(0, DEFAULT_LOOKBACK_DAYS - 1)
        start_date = today - dt.timedelta(days=lookback)

    if start_date > today:
        start_date = today

    return start_date, today


def filter_new_rows(rows, last_ts):
    if not rows:
        return []

    filtered = []
    for row in rows:
        if len(row) < 6:
            continue
        ts = to_dt(row[5])
        if not ts:
            continue
        if last_ts and ts <= last_ts:
            continue
        filtered.append(row)
    return filtered

def collect_incremental_rainfall(sh, last_ts):
    start_date, end_date = compute_fetch_window(last_ts)
    new_rows: List[List] = []

    for day in iter_dates_inclusive(start_date.isoformat(), end_date.isoformat()):
        rainfall_value, _ = run_rainfall(sh, day.isoformat())
        new_rows.extend(filter_new_rows(rainfall_value, last_ts))

    return new_rows


def build_station_to_zones_map(zone_to_stations):
    station_lookup = {}
    for zone_name, stations in (zone_to_stations or {}).items():
        if not zone_name:
            continue
        for sid in stations or []:
            sid_str = str(sid).strip()
            if not sid_str:
                continue
            station_lookup.setdefault(sid_str, []).append(zone_name)
    return station_lookup


def aggregate_zone_rainfall(rows, station_zone_lookup):
    if not rows or not station_zone_lookup:
        return []

    totals: Dict[Tuple[str, str], float] = {}
    for row in rows:
        if len(row) < 6:
            continue
        station_id = str(row[0]).strip()
        if not station_id:
            continue
        zones = station_zone_lookup.get(station_id)
        if not zones:
            continue

        reading_time = row[5]
        try:
            reading_value = float(row[4] or 0)
        except Exception:
            continue

        for zone_name in zones:
            key = (zone_name, reading_time)
            totals[key] = totals.get(key, 0.0) + reading_value

    zone_rows = []
    for (zone_name, reading_time), total in totals.items():
        zone_rows.append([zone_name, reading_time, round(total, 3), reading_label_from_mm(total)])
    return zone_rows


def add_labels_to_station_rows(rows):
    labeled = []
    for row in rows:
        if len(row) < 6:
            continue
        try:
            value = float(row[4] or 0)
        except Exception:
            value = 0.0
        label = reading_label_from_mm(value)
        labeled.append(row + [label])
    return labeled

def main():
    sh = open_sheet_by_id(SHEET_ID)
    rainfall_ws = ensure_worksheet(sh, RAIN_SHEET_NAME, RAIN_HEADERS)
    rainfall_zone_ws = ensure_worksheet(sh, RAIN_ZONE_SHEET_NAME, RAIN_ZONE_HEADERS)

    last_ts = get_last_timestamp(rainfall_ws)
    if last_ts:
        print(f"Last recorded rainfall timestamp: {last_ts.isoformat()}")
    else:
        print("No previous rainfall data found. Using initial lookback window.")

    new_rows = collect_incremental_rainfall(sh, last_ts)
    print(f"Fetched {len(new_rows)} new rainfall readings.")
    labeled_station_rows = add_labels_to_station_rows(new_rows)

    zone_to_stations = load_zone_to_stations_from_sheet(sh, sheet_name=ZONE_SHEET_NAME)
    station_zone_lookup = build_station_to_zones_map(zone_to_stations)
    zone_rows = aggregate_zone_rainfall(new_rows, station_zone_lookup)
    print(f"Derived {len(zone_rows)} zone-summed rainfall readings.")

    if labeled_station_rows:
        append_unique(rainfall_ws, labeled_station_rows, key_cols=RAIN_KEY_COLS)
        print("Rainfall sheet updated.")
    else:
        print("Nothing new to append to rainfall sheet.")

    if zone_rows:
        append_unique(rainfall_zone_ws, zone_rows, key_cols=RAIN_ZONE_KEY_COLS)
        print("Zone rainfall sheet updated.")
    else:
        print("Nothing new to append to zone rainfall sheet.")


if __name__ == "__main__":
    main()
