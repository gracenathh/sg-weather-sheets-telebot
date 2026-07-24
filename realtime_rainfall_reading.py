import os
import datetime as dt
from typing import Dict, Tuple

from dotenv import load_dotenv

from consolidate import (
    append_unique,
    to_dt,
    run_rainfall,
    open_sheet_by_id,
    load_zone_to_stations_from_sheet,
)

load_dotenv()

SGT = dt.timezone(dt.timedelta(hours=8))
SHEET_ID = os.environ["SHEET_ID"]

RAIN_SHEET_NAME = os.getenv(
    "RAIN_SHEET_NAME",
    "rainfall_data",
)

RAIN_HEADERS = [
    "station_id",
    "station_name",
    "lat",
    "lon",
    "reading_value",
    "reading_time",
    "reading_label",
]

RAIN_KEY_COLS = [
    "station_id",
    "reading_time",
]

RAIN_ZONE_SHEET_NAME = os.getenv(
    "RAIN_ZONE_SHEET_NAME",
    "rainfall_data_sg_zone",
)

# Keep the first four columns backward-compatible
# with the existing sheet.
RAIN_ZONE_HEADERS = [
    "zone_name",
    "reading_time",
    "reading_value",
    "reading_label",
    "max_rain_mm",
    "wet_station_count",
    "total_station_count",
    "wet_station_share",
    "is_raining",
    "aggregation_method",
]

RAIN_ZONE_KEY_COLS = [
    "zone_name",
    "reading_time",
]

ZONE_SHEET_NAME = os.getenv(
    "ZONE_SHEET_NAME",
    "zone_station_map",
)


def ensure_worksheet_preserving_rows(sh, title, headers):
    """
    Create the worksheet if it does not exist.

    If new columns are added to an existing worksheet,
    add the headers without deleting its existing rows.
    """
    ws = next(
        (
            worksheet
            for worksheet in sh.worksheets()
            if worksheet.title == title
        ),
        None,
    )

    if ws is None:
        ws = sh.add_worksheet(
            title=title,
            rows=1000,
            cols=max(10, len(headers)),
        )
        ws.append_row(
            headers,
            value_input_option="USER_ENTERED",
        )
        return ws

    existing_headers = ws.row_values(1)

    if not existing_headers:
        ws.append_row(
            headers,
            value_input_option="USER_ENTERED",
        )
        return ws

    if existing_headers == headers:
        return ws

    # Only allow new columns to be added at the end.
    # Stop if existing columns were renamed or reordered.
    if existing_headers != headers[: len(existing_headers)]:
        raise RuntimeError(
            f"Unexpected headers in '{title}': "
            f"{existing_headers}. Expected the existing "
            f"headers to be the start of: {headers}"
        )

    for column_number, header in enumerate(
        headers[len(existing_headers) :],
        start=len(existing_headers) + 1,
    ):
        ws.update_cell(
            1,
            column_number,
            header,
        )

    return ws


def delete_rows_in_groups(ws, row_numbers):
    """
    Delete worksheet rows from the bottom upward.

    This prevents the remaining row numbers from shifting
    before they are deleted.
    """
    if not row_numbers:
        return

    row_numbers = sorted(row_numbers)

    groups = []
    group_start = row_numbers[0]
    group_end = row_numbers[0]

    for row_number in row_numbers[1:]:
        if row_number == group_end + 1:
            group_end = row_number
            continue

        groups.append(
            (group_start, group_end)
        )

        group_start = row_number
        group_end = row_number

    groups.append(
        (group_start, group_end)
    )

    for start_row, end_row in reversed(groups):
        ws.delete_rows(
            start_row,
            end_row,
        )


def prune_to_rolling_two_days(ws):
    """
    Keep only yesterday (D-1) and today (D-day),
    based on Singapore time.

    Rows with a blank or invalid reading_time are
    left untouched to avoid accidental deletion.
    """
    values = ws.get_all_values()

    if len(values) <= 1:
        return 0

    headers = values[0]

    if "reading_time" not in headers:
        raise RuntimeError(
            f"Worksheet '{ws.title}' "
            "has no reading_time column."
        )

    time_idx = headers.index("reading_time")

    today = dt.datetime.now(SGT).date()
    yesterday = today - dt.timedelta(days=1)

    dates_to_keep = {
        yesterday,
        today,
    }

    rows_to_delete = []

    # Google Sheets rows start from 1.
    # Row 1 contains the headers.
    for sheet_row_number, row in enumerate(
        values[1:],
        start=2,
    ):
        if len(row) <= time_idx:
            continue

        reading_ts = to_dt(row[time_idx])

        if not reading_ts:
            continue

        if reading_ts.date() not in dates_to_keep:
            rows_to_delete.append(
                sheet_row_number
            )

    delete_rows_in_groups(
        ws,
        rows_to_delete,
    )

    return len(rows_to_delete)


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


def collect_latest_rainfall(sh):
    """
    Fetch only the newest realtime rainfall bucket.

    This does not use the latest timestamp in Google Sheets
    as a historical start date. Therefore, an old record in
    the Sheet cannot trigger a months-long backfill.
    """
    rainfall_value, _ = run_rainfall(sh)

    parsed_rows = []

    for row in rainfall_value or []:
        if len(row) < 6:
            continue

        reading_ts = to_dt(row[5])

        if reading_ts:
            parsed_rows.append(
                (reading_ts, row)
            )

    if not parsed_rows:
        return []

    latest_api_ts = max(
        reading_ts
        for reading_ts, _ in parsed_rows
    )

    latest_rows = [
        row
        for reading_ts, row in parsed_rows
        if reading_ts == latest_api_ts
    ]

    return latest_rows


def build_station_to_zones_map(zone_to_stations):
    station_lookup = {}

    for zone_name, stations in (
        zone_to_stations or {}
    ).items():
        if not zone_name:
            continue

        for station_id in stations or []:
            station_id_string = str(
                station_id
            ).strip()

            if not station_id_string:
                continue

            station_lookup.setdefault(
                station_id_string,
                set(),
            ).add(zone_name)

    return {
        station_id: sorted(zones)
        for station_id, zones
        in station_lookup.items()
    }


def aggregate_zone_rainfall(
    rows,
    station_zone_lookup,
):
    if not rows or not station_zone_lookup:
        return []

    # Keep one value per station for each
    # zone and reading timestamp.
    readings_by_zone: Dict[
        Tuple[str, str],
        Dict[str, float],
    ] = {}

    for row in rows:
        if len(row) < 6:
            continue

        station_id = str(row[0]).strip()

        if not station_id:
            continue

        zones = station_zone_lookup.get(
            station_id
        )

        if not zones:
            continue

        reading_time = row[5]

        try:
            reading_value = float(
                row[4] or 0
            )
        except Exception:
            continue

        for zone_name in zones:
            key = (
                zone_name,
                reading_time,
            )

            readings_by_zone.setdefault(
                key,
                {},
            )[station_id] = reading_value

    zone_rows = []

    for (
        zone_name,
        reading_time,
    ), station_readings in readings_by_zone.items():

        values = list(
            station_readings.values()
        )

        if not values:
            continue

        avg_rain_mm = (
            sum(values) / len(values)
        )

        max_rain_mm = max(values)

        wet_station_count = sum(
            value > 0
            for value in values
        )

        total_station_count = len(values)

        wet_station_share = (
            wet_station_count
            / total_station_count
        )

        is_raining = max_rain_mm > 0

        zone_rows.append(
            [
                zone_name,
                reading_time,
                round(avg_rain_mm, 3),
                reading_label_from_mm(
                    avg_rain_mm
                ),
                round(max_rain_mm, 3),
                wet_station_count,
                total_station_count,
                round(wet_station_share, 3),
                is_raining,
                "mean",
            ]
        )

    zone_rows.sort(
        key=lambda row: (
            str(row[1]),
            row[0],
        )
    )

    return zone_rows


def add_labels_to_station_rows(rows):
    labeled_rows = []

    for row in rows:
        if len(row) < 6:
            continue

        try:
            value = float(
                row[4] or 0
            )
        except Exception:
            value = 0.0

        label = reading_label_from_mm(
            value
        )

        labeled_rows.append(
            row + [label]
        )

    return labeled_rows


def main():
    sh = open_sheet_by_id(
        SHEET_ID
    )

    rainfall_ws = (
        ensure_worksheet_preserving_rows(
            sh,
            RAIN_SHEET_NAME,
            RAIN_HEADERS,
        )
    )

    rainfall_zone_ws = (
        ensure_worksheet_preserving_rows(
            sh,
            RAIN_ZONE_SHEET_NAME,
            RAIN_ZONE_HEADERS,
        )
    )

    # Delete D-2 and older records from both tabs.
    deleted_station_rows = (
        prune_to_rolling_two_days(
            rainfall_ws
        )
    )

    deleted_zone_rows = (
        prune_to_rolling_two_days(
            rainfall_zone_ws
        )
    )

    print(
        "Rolling two-day cleanup removed "
        f"{deleted_station_rows} station rows "
        f"and {deleted_zone_rows} zone rows."
    )

    # Fetch only the newest current rainfall reading.
    new_rows = collect_latest_rainfall(
        sh
    )

    print(
        f"Fetched {len(new_rows)} readings "
        "from the latest realtime bucket."
    )

    labeled_station_rows = (
        add_labels_to_station_rows(
            new_rows
        )
    )

    zone_to_stations = (
        load_zone_to_stations_from_sheet(
            sh,
            sheet_name=ZONE_SHEET_NAME,
        )
    )

    station_zone_lookup = (
        build_station_to_zones_map(
            zone_to_stations
        )
    )

    zone_rows = aggregate_zone_rainfall(
        new_rows,
        station_zone_lookup,
    )

    print(
        f"Derived {len(zone_rows)} "
        "zone-average rainfall readings."
    )

    # append_unique prevents the same
    # station/timestamp from being inserted twice.
    if labeled_station_rows:
        append_unique(
            rainfall_ws,
            labeled_station_rows,
            key_cols=RAIN_KEY_COLS,
        )
        print(
            "Rainfall sheet updated."
        )
    else:
        print(
            "Nothing new to append "
            "to rainfall sheet."
        )

    # Prevent duplicate zone/timestamp rows.
    if zone_rows:
        append_unique(
            rainfall_zone_ws,
            zone_rows,
            key_cols=RAIN_ZONE_KEY_COLS,
        )
        print(
            "Zone rainfall sheet updated."
        )
    else:
        print(
            "Nothing new to append "
            "to zone rainfall sheet."
        )


if __name__ == "__main__":
    main()
