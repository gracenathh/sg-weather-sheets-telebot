import os
import datetime as dt
from typing import Dict, Tuple

from dotenv import load_dotenv

from consolidate import (
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
    "tod_label",
    "cumulative_tod_avg_mm",
    "cumulative_all_day_avg_mm",
]

RAIN_ZONE_KEY_COLS = [
    "zone_name",
    "reading_time",
]

ZONE_SHEET_NAME = os.getenv(
    "ZONE_SHEET_NAME",
    "zone_station_map",
)


# ============================================================
# MODE SWITCH
# ============================================================
#
# True:
# - Fetch yesterday + today
# - Recalculate yesterday + today
# - Replace yesterday + today in both Sheets
#
# False:
# - Fetch and recalculate today only
# - Preserve yesterday's existing rows
# - Replace today's rows
#
# Use True once for checking yesterday's values.
# Change to False after the backfill is verified.
#
BACKFILL_YESTERDAY = False


# (label, starting hour >=, ending hour <)
TIME_OF_DAY_ORDER = [
    ("Late Night (00:00 - 05:59)", 0, 6),
    ("Breakfast (06:00 - 09:59)", 6, 10),
    ("Lunch (10:00 - 13:59)", 10, 14),
    ("Teabreak (14:00 - 16:59)", 14, 17),
    ("Dinner (17:00 - 20:59)", 17, 21),
    ("Supper (21:00 - 23:59)", 21, 24),
]


def ensure_worksheet_preserving_rows(
    sh,
    title,
    headers,
):
    """
    Create the worksheet if it does not exist.

    If it exists, safely add new trailing columns without
    deleting any historical rows.
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

    # Expand the physical grid before writing K1, L1, M1.
    if ws.col_count < len(headers):
        ws.resize(
            cols=len(headers),
        )

    existing_headers = ws.row_values(1)

    if not existing_headers:
        ws.append_row(
            headers,
            value_input_option="USER_ENTERED",
        )
        return ws

    if existing_headers == headers:
        return ws

    # Allow only new columns added at the end.
    if (
        existing_headers
        != headers[:len(existing_headers)]
    ):
        raise RuntimeError(
            f"Unexpected headers in '{title}': "
            f"{existing_headers}. "
            f"Expected the existing headers to be "
            f"the start of: {headers}"
        )

    for column_number, header in enumerate(
        headers[len(existing_headers):],
        start=len(existing_headers) + 1,
    ):
        ws.update_cell(
            1,
            column_number,
            header,
        )

    return ws


def delete_rows_in_groups(
    ws,
    row_numbers,
):
    """
    Delete non-adjacent worksheet rows safely.

    Delete from the bottom upwards so row numbers do not
    shift before they are processed.
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
    Keep only yesterday and today, based on Singapore time.

    Blank or unparseable timestamps are preserved.
    """
    values = ws.get_all_values()

    if len(values) <= 1:
        return 0

    headers = values[0]

    if "reading_time" not in headers:
        raise RuntimeError(
            f"Worksheet '{ws.title}' has no "
            f"reading_time column."
        )

    time_idx = headers.index("reading_time")

    today = dt.datetime.now(SGT).date()
    yesterday = today - dt.timedelta(days=1)

    rows_to_delete = []

    # Row 1 is the header.
    for sheet_row_number, row in enumerate(
        values[1:],
        start=2,
    ):
        if len(row) <= time_idx:
            continue

        reading_ts = to_dt(
            row[time_idx]
        )

        if not reading_ts:
            continue

        if reading_ts.date() not in {
            yesterday,
            today,
        }:
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


def collect_rainfall_for_dates(
    sh,
    target_dates,
):
    """
    Fetch every available 5-minute rainfall reading for
    every requested date.

    Even if GitHub runs every 10 minutes, fetching the full
    date retrieves:

    :00, :05, :10, :15 ... :55

    Duplicate station/timestamp combinations are removed.
    """
    rows_by_key = {}

    for target_date in sorted(
        set(target_dates)
    ):
        rainfall_value, _ = run_rainfall(
            sh,
            target_date.isoformat(),
        )

        for row in rainfall_value or []:
            if len(row) < 6:
                continue

            station_id = str(
                row[0]
            ).strip()

            reading_ts = to_dt(
                row[5]
            )

            if not station_id:
                continue

            if not reading_ts:
                continue

            if reading_ts.date() != target_date:
                continue

            rows_by_key[
                (
                    station_id,
                    reading_ts,
                )
            ] = row

    return [
        row
        for (_, _), row in sorted(
            rows_by_key.items(),
            key=lambda item: (
                item[0][1],
                item[0][0],
            ),
        )
    ]


def build_station_to_zones_map(
    zone_to_stations,
):
    """
    Reverse:

    zone -> stations

    into:

    station -> zones
    """
    station_lookup = {}

    for zone_name, stations in (
        zone_to_stations or {}
    ).items():
        if not zone_name:
            continue

        for station_id in stations or []:
            station_id = str(
                station_id
            ).strip()

            if not station_id:
                continue

            station_lookup.setdefault(
                station_id,
                set(),
            ).add(zone_name)

    return {
        station_id: sorted(zones)
        for station_id, zones
        in station_lookup.items()
    }


def get_time_of_day_label(
    reading_ts,
):
    """
    Assign a TOD label using the Singapore hour.
    """
    for (
        tod_label,
        start_hour,
        end_hour,
    ) in TIME_OF_DAY_ORDER:
        if (
            start_hour
            <= reading_ts.hour
            < end_hour
        ):
            return tod_label

    return "Unknown"


def aggregate_zone_rainfall(
    rows,
    station_zone_lookup,
):
    """
    Create one zone row for every five-minute timestamp.

    reading_value:
        Average rainfall across reporting stations for
        the latest five-minute interval.

    cumulative_tod_avg_mm:
        Sum each station's readings within the current TOD,
        then average those station totals within the zone.

    cumulative_all_day_avg_mm:
        Sum each station's readings since 00:00,
        then average those station totals within the zone.
    """
    if not rows:
        return []

    if not station_zone_lookup:
        return []

    readings_by_zone: Dict[
        Tuple[str, dt.datetime],
        Dict[str, float],
    ] = {}

    original_reading_times = {}

    for row in rows:
        if len(row) < 6:
            continue

        station_id = str(
            row[0]
        ).strip()

        if not station_id:
            continue

        zones = station_zone_lookup.get(
            station_id
        )

        if not zones:
            continue

        reading_ts = to_dt(
            row[5]
        )

        if not reading_ts:
            continue

        try:
            reading_value = float(
                row[4] or 0
            )
        except Exception:
            continue

        for zone_name in zones:
            key = (
                zone_name,
                reading_ts,
            )

            readings_by_zone.setdefault(
                key,
                {},
            )[station_id] = reading_value

            original_reading_times[key] = row[5]

    zone_rows = []

    # Key:
    # (zone, date, TOD)
    #
    # Value:
    # {station_id: cumulative rainfall}
    cumulative_tod_station_totals = {}

    # Key:
    # (zone, date)
    #
    # Value:
    # {station_id: cumulative rainfall}
    cumulative_day_station_totals = {}

    # Must be chronological so cumulative values build
    # correctly before the output is later sorted descending.
    for (
        zone_name,
        reading_ts,
    ), station_readings in sorted(
        readings_by_zone.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    ):
        values = list(
            station_readings.values()
        )

        if not values:
            continue

        tod_label = get_time_of_day_label(
            reading_ts
        )

        tod_key = (
            zone_name,
            reading_ts.date(),
            tod_label,
        )

        day_key = (
            zone_name,
            reading_ts.date(),
        )

        tod_station_totals = (
            cumulative_tod_station_totals.setdefault(
                tod_key,
                {},
            )
        )

        day_station_totals = (
            cumulative_day_station_totals.setdefault(
                day_key,
                {},
            )
        )

        # First sum rainfall separately for each station.
        for (
            station_id,
            reading_value,
        ) in station_readings.items():
            tod_station_totals[station_id] = (
                tod_station_totals.get(
                    station_id,
                    0.0,
                )
                + reading_value
            )

            day_station_totals[station_id] = (
                day_station_totals.get(
                    station_id,
                    0.0,
                )
                + reading_value
            )

        # Current five-minute zone average.
        avg_rain_mm = (
            sum(values)
            / len(values)
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

        # Average cumulative station totals within the TOD.
        cumulative_tod_avg_mm = (
            sum(
                tod_station_totals.values()
            )
            / len(tod_station_totals)
        )

        # Average cumulative station totals since 00:00.
        cumulative_all_day_avg_mm = (
            sum(
                day_station_totals.values()
            )
            / len(day_station_totals)
        )

        reading_time = original_reading_times[
            (
                zone_name,
                reading_ts,
            )
        ]

        zone_rows.append([
            zone_name,
            reading_time,
            round(
                avg_rain_mm,
                3,
            ),
            reading_label_from_mm(
                avg_rain_mm
            ),
            round(
                max_rain_mm,
                3,
            ),
            wet_station_count,
            total_station_count,
            round(
                wet_station_share,
                3,
            ),
            is_raining,
            "mean",
            tod_label,
            round(
                cumulative_tod_avg_mm,
                3,
            ),
            round(
                cumulative_all_day_avg_mm,
                3,
            ),
        ])

    return zone_rows


def add_labels_to_station_rows(rows):
    """
    Add a rainfall label to each station reading.
    """
    labeled = []

    for row in rows:
        if len(row) < 6:
            continue

        try:
            value = float(
                row[4] or 0
            )
        except Exception:
            value = 0.0

        labeled.append(
            row
            + [
                reading_label_from_mm(
                    value
                )
            ]
        )

    return labeled


def replace_dates_sort_and_write(
    ws,
    headers,
    new_rows,
    key_cols,
    replace_dates,
):
    """
    Preserve dates that are not being refreshed.

    Replace the requested dates with newly calculated rows,
    remove duplicate keys, sort newest-first, then rewrite
    the worksheet.
    """
    existing_values = ws.get_all_values()

    existing_headers = (
        existing_values[0]
        if existing_values
        else headers
    )

    if existing_headers != headers:
        raise RuntimeError(
            f"Unexpected headers in '{ws.title}': "
            f"{existing_headers}. "
            f"Expected: {headers}"
        )

    time_idx = headers.index(
        "reading_time"
    )

    key_positions = [
        headers.index(column)
        for column in key_cols
    ]

    replace_dates = set(
        replace_dates
    )

    kept_rows = []

    for row in existing_values[1:]:
        # Make every existing row exactly the same length
        # as the current schema.
        normalized = list(
            row[:len(headers)]
        )

        normalized.extend(
            [""] * (
                len(headers)
                - len(normalized)
            )
        )

        reading_ts = to_dt(
            normalized[time_idx]
        )

        # Remove existing rows for dates that are being
        # recalculated.
        if (
            reading_ts
            and reading_ts.date()
            in replace_dates
        ):
            continue

        kept_rows.append(
            normalized
        )

    rows_by_key = {}

    # Existing preserved rows are inserted first.
    # Refreshed rows are inserted afterwards, so new rows
    # win if the same key appears twice.
    for row in (
        kept_rows
        + list(new_rows)
    ):
        normalized = list(
            row[:len(headers)]
        )

        normalized.extend(
            [""] * (
                len(headers)
                - len(normalized)
            )
        )

        key = tuple(
            str(
                normalized[position]
            )
            for position in key_positions
        )

        rows_by_key[key] = normalized

    merged_rows = list(
        rows_by_key.values()
    )

    def descending_time_sort_key(row):
        reading_ts = to_dt(
            row[time_idx]
        )

        unix_time = (
            reading_ts.timestamp()
            if reading_ts
            else float("-inf")
        )

        # Negative timestamp means newest first.
        # Station ID or zone name is ascending when the
        # timestamps are equal.
        return (
            -unix_time,
            str(
                row[key_positions[0]]
            ),
        )

    merged_rows.sort(
        key=descending_time_sort_key
    )

    if ws.col_count < len(headers):
        ws.resize(
            cols=len(headers),
        )

    # Rewrite the worksheet after merge/deduplication/sort.
    # Rewrite the worksheet after merge/deduplication/sort.
    ws.clear()
    
    # Include the header in the first bulk write.
    rows_to_write = [headers,] + merged_rows
    
    # 5,000 rows per request keeps the number of write
    # requests comfortably below Google's per-minute quota.
    WRITE_BATCH_SIZE = 5000
    
    for start in range(
        0,
        len(rows_to_write),
        WRITE_BATCH_SIZE,
    ):
        ws.append_rows(
            rows_to_write[
                start:start + WRITE_BATCH_SIZE
            ],
            value_input_option="USER_ENTERED",
        )
    
    return len(merged_rows)

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

    # Keep only yesterday and today.
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
        f"{deleted_station_rows} station rows and "
        f"{deleted_zone_rows} zone rows."
    )

    today = dt.datetime.now(
        SGT
    ).date()

    yesterday = (
        today
        - dt.timedelta(days=1)
    )

    # Normal mode:
    # Refresh today only.
    fetch_dates = {
        today,
    }

    replace_dates = {
        today,
    }

    # Backfill mode:
    # Refresh yesterday and today.
    if BACKFILL_YESTERDAY:
        fetch_dates.add(
            yesterday
        )

        replace_dates.add(
            yesterday
        )

    # Retrieve all available five-minute readings for the
    # selected dates.
    new_rows = collect_rainfall_for_dates(
        sh,
        fetch_dates,
    )

    timestamps = {
        to_dt(row[5])
        for row in new_rows
        if (
            len(row) >= 6
            and to_dt(row[5])
        )
    }

    print(
        f"Fetched {len(new_rows)} station readings "
        f"across {len(timestamps)} five-minute buckets "
        f"for "
        f"{', '.join(sorted(day.isoformat() for day in fetch_dates))}."
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
        f"zone-average rainfall rows."
    )

    # Replace requested dates and sort newest-first.
    station_row_count = (
        replace_dates_sort_and_write(
            rainfall_ws,
            RAIN_HEADERS,
            labeled_station_rows,
            RAIN_KEY_COLS,
            replace_dates,
        )
    )

    zone_row_count = (
        replace_dates_sort_and_write(
            rainfall_zone_ws,
            RAIN_ZONE_HEADERS,
            zone_rows,
            RAIN_ZONE_KEY_COLS,
            replace_dates,
        )
    )

    print(
        "Sheets rewritten newest-first: "
        f"{station_row_count} station rows and "
        f"{zone_row_count} zone rows."
    )


if __name__ == "__main__":
    main()
