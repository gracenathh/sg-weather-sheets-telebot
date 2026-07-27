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
    Create a worksheet, or safely add new trailing
    headers to an existing worksheet.

    The existing rainfall history will not be cleared
    when new columns are added.
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

    # Only allow new columns to be added to the end.
    # This prevents existing data from becoming
    # misaligned if older columns were renamed or moved.
    if existing_headers != headers[:len(existing_headers)]:
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

    Rows are grouped and deleted from the bottom upward
    so earlier row numbers do not shift.
    """
    if not row_numbers:
        return

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
    Keep only yesterday and today based on
    Singapore time.

    Blank or unparseable timestamps are not deleted.
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

    # Google Sheets uses row 1 for headers.
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


def collect_today_rainfall(sh):
    """
    Fetch every available five-minute rainfall reading
    for the current Singapore date.

    Even if GitHub runs every 10 minutes, this retrieves
    every available API bucket:

    :00, :05, :10, :15 ... :55

    append_unique() later prevents previously stored
    station/timestamp combinations from being duplicated.
    """
    today = dt.datetime.now(SGT).date()

    rainfall_value, _ = run_rainfall(
        sh,
        today.isoformat(),
    )

    rows_by_key = {}

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

        if reading_ts.date() != today:
            continue

        # Keep only one row per station and
        # five-minute timestamp.
        rows_by_key[
            (station_id, reading_ts)
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
    Reverse the zone-to-station mapping into:

    station_id -> list of zones
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
    Return the TOD label corresponding to the
    reading's Singapore hour.
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
    Calculate zone rainfall for every five-minute
    timestamp.

    reading_value:
        Average of the station readings for the
        latest five-minute timestamp.

    cumulative_tod_avg_mm:
        Sum each station's readings since the start
        of the current TOD, then average the station
        totals within the zone.

    cumulative_all_day_avg_mm:
        Sum each station's readings since 00:00,
        then average the station totals within
        the zone.
    """
    if not rows:
        return []

    if not station_zone_lookup:
        return []

    # Structure:
    #
    # (zone, timestamp)
    #     -> {
    #          station_id: reading_value
    #        }
    #
    # Using one value per station prevents duplicate
    # API rows or duplicate mappings from affecting
    # the calculation.
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

    # Running station totals for each TOD.
    #
    # Key:
    # (zone, date, TOD label)
    #
    # Value:
    # {station_id: cumulative rainfall}
    cumulative_tod_station_totals = {}

    # Running station totals for the whole day.
    #
    # Key:
    # (zone, date)
    #
    # Value:
    # {station_id: cumulative rainfall}
    cumulative_day_station_totals = {}

    # Sorting by timestamp is important because the
    # cumulative totals must be built chronologically.
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

        # Sum rainfall separately for every station.
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

        # Latest five-minute zone average.
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

        # Sum each station within the current TOD,
        # then average those cumulative station totals.
        cumulative_tod_avg_mm = (
            sum(
                tod_station_totals.values()
            )
            / len(tod_station_totals)
        )

        # Sum each station since 00:00,
        # then average those cumulative station totals.
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

    zone_rows.sort(
        key=lambda row: (
            str(row[1]),
            row[0],
        )
    )

    return zone_rows


def add_labels_to_station_rows(rows):
    """
    Add a rainfall severity label to each individual
    station reading.
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

        label = reading_label_from_mm(
            value
        )

        labeled.append(
            row + [label]
        )

    return labeled


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

    # Retain only yesterday and today.
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

    # Retrieve every available five-minute bucket
    # for today.
    new_rows = collect_today_rainfall(
        sh
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
        f"across {len(timestamps)} five-minute "
        f"buckets for today."
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
        f"zone-average rainfall readings."
    )

    # append_unique prevents the previously captured
    # station/timestamp rows from being inserted again.
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
            "Nothing new to append to "
            "rainfall sheet."
        )

    # One unique row per zone and timestamp.
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
            "Nothing new to append to "
            "zone rainfall sheet."
        )


if __name__ == "__main__":
    main()
