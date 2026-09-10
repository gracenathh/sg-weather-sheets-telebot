"""Maintain a readable, rolling 24-hour forecast worksheet.

The worksheet stores one row per forecast vintage and six-hour period, with
South/North/East/West shown side by side.

On regular runs:
- Yesterday and today are requested from the API.
- If today's dated endpoint is not yet available (404), the latest available
  forecast is fetched instead.
- Existing history is preserved.
- Rows are deduplicated using issued_ts + period_start.
- If the same forecast vintage is revised, the row with the latest updated_ts
  is kept.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import os
import time
from typing import Iterable

import requests
from dotenv import load_dotenv

from sheets_utils import ensure_worksheet


load_dotenv()

SGT = dt.timezone(dt.timedelta(hours=8))

URL = (
    "https://api-open.data.gov.sg/v2/real-time/api/"
    "twenty-four-hr-forecast"
)

SHEET_NAME = os.getenv(
    "FORECAST_24H_SHEET_NAME",
    "forecast_24h",
)

COMMS_LOG_SHEET = os.getenv(
    "FORECAST_24H_COMMS_LOG_SHEET",
    "forecast_24h_comms_log",
)

REGIONS = (
    "south",
    "north",
    "east",
    "west",
)

HEADERS = [
    "date",
    "issued_ts",
    "updated_ts",
    "forecast_window",
    "campaign_use",
    "period_start",
    "period_end",
    "period_text",
    "south_forecast",
    "north_forecast",
    "east_forecast",
    "west_forecast",
    "thunder_regions",
    "campaign_comms",
    "south_code",
    "north_code",
    "east_code",
    "west_code",
]

COMMS_LOG_HEADERS = [
    "notification_key",
    "campaign",
    "campaign_date",
    "forecast_issued_ts",
    "forecast_updated_ts",
    "message",
    "telegram_chat_id",
    "telegram_message_id",
    "sent_at",
]


# ============================================================
# DATETIME HELPERS
# ============================================================


def parse_datetime(value):
    if not value:
        return None

    try:
        parsed = dt.datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=SGT
        )

    return parsed.astimezone(SGT)


def subtract_calendar_months(
    day: dt.date,
    months: int,
) -> dt.date:
    month_index = (
        day.year * 12
        + day.month
        - 1
        - months
    )

    year, zero_based_month = divmod(
        month_index,
        12,
    )

    month = zero_based_month + 1

    return dt.date(
        year,
        month,
        min(
            day.day,
            calendar.monthrange(
                year,
                month,
            )[1],
        ),
    )


def first_day_months_ago(
    day: dt.date,
    months: int,
) -> dt.date:
    shifted = subtract_calendar_months(
        day.replace(day=1),
        months,
    )

    return shifted.replace(day=1)


def dates_inclusive(
    start: dt.date,
    end: dt.date,
) -> Iterable[dt.date]:
    current = start

    while current <= end:
        yield current
        current += dt.timedelta(days=1)


# ============================================================
# API HELPERS
# ============================================================


def request_headers():
    headers = {
        "Accept": "application/json",
        "User-Agent": os.getenv(
            "USER_AGENT",
            "sg-24h-forecast-holder/1.0",
        ),
    }

    api_key = os.getenv(
        "DATA_GOV_SG_API_KEY"
    )

    if api_key:
        headers["X-Api-Key"] = api_key

    return headers


def validate_payload(payload):
    if payload.get("code") not in (
        None,
        0,
    ):
        raise RuntimeError(
            f"API code "
            f"{payload.get('code')}: "
            f"{payload.get('errorMsg')}"
        )

    return payload


def fetch_latest():
    """
    Fetch the latest forecast available from the API.

    This intentionally does NOT include a date parameter.
    """

    response = requests.get(
        URL,
        headers=request_headers(),
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    return validate_payload(payload)


def fetch_day(
    day: dt.date,
    fallback_to_latest: bool = False,
):
    """
    Fetch forecast data for one requested date.

    If fallback_to_latest=True and the dated endpoint returns
    404, fetch the API's latest available forecast instead.

    This is intended for today, because today's dated endpoint
    may not yet exist even though the API already has a latest
    forecast available.
    """

    last_error = None

    for attempt in range(5):
        try:
            response = requests.get(
                URL,
                params={
                    "date": day.isoformat()
                },
                headers=request_headers(),
                timeout=60,
            )

            # ------------------------------------------------
            # Today's dated record may not exist yet.
            #
            # Instead of failing:
            #     ?date=2026-09-10 -> 404
            #
            # fetch:
            #     /twenty-four-hr-forecast
            #
            # and use whatever latest forecast NEA returns.
            # ------------------------------------------------
            if (
                response.status_code == 404
                and fallback_to_latest
            ):
                print(
                    f"24h forecast: dated record "
                    f"{day} returned 404; "
                    f"fetching latest available "
                    f"forecast instead"
                )

                payload = fetch_latest()

                return payload

            response.raise_for_status()

            payload = response.json()

            return validate_payload(
                payload
            )

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc

            if attempt < 4:
                sleep_seconds = min(
                    20,
                    2**attempt,
                )

                print(
                    f"24h forecast: attempt "
                    f"{attempt + 1} failed for "
                    f"{day}: {exc}; "
                    f"retrying in "
                    f"{sleep_seconds}s"
                )

                time.sleep(
                    sleep_seconds
                )

    raise RuntimeError(
        f"Unable to fetch {day}: "
        f"{last_error}"
    )


# ============================================================
# FORECAST PARSING
# ============================================================


def normalized_window(
    start_value,
    end_value,
):
    start = parse_datetime(
        start_value
    )

    end = parse_datetime(
        end_value
    )

    if not start or not end:
        return ""

    return (
        f"{start:%H:%M}-"
        f"{end:%H:%M}"
    )


def campaign_use(window):
    return {
        "06:00-12:00": (
            "lunch_block_1"
        ),
        "12:00-18:00": (
            "lunch_block_2"
        ),
        "18:00-00:00": (
            "dinner"
        ),
    }.get(
        window,
        "",
    )


def is_thunder_family(value):
    return (
        "thundery showers"
        in str(value or "").lower()
    )


def communication_messages(
    kind,
    regions,
    campaign_date=None,
):
    if not regions:
        return ""

    names = [
        region.title()
        for region in regions
    ]

    if len(names) == 1:
        display_regions = names[0]

    elif len(names) == 2:
        display_regions = (
            f"{names[0]} and "
            f"{names[1]}"
        )

    else:
        display_regions = (
            f"{', '.join(names[:-1])} "
            f"and {names[-1]}"
        )

    area_word = (
        "area"
        if len(regions) == 1
        else "areas"
    )

    date_text = (
        f" ({campaign_date:%d/%m})"
        if campaign_date
        else ""
    )

    if kind == "lunch":
        return (
            f"Rain is forecast around "
            f"lunchtime tomorrow"
            f"{date_text} in the "
            f"{display_regions} "
            f"{area_word}. "
            f"Consider bringing your "
            f"rain jacket if you plan "
            f"to ride during lunch period!"
        )

    if kind == "dinner":
        return (
            f"Rain is forecast tonight"
            f"{date_text} in the "
            f"{display_regions} "
            f"{area_word}. "
            f"Consider bringing your "
            f"rain jacket if you plan "
            f"to ride during dinner period!"
        )

    return ""


def flatten_payload(payload):
    """
    Convert API records into worksheet rows.

    IMPORTANT:
    Dates and timestamps come directly from the returned
    API record. We do NOT overwrite them with the date
    that was requested.

    This matters when:
        today's dated request -> 404
        fallback -> latest forecast from yesterday

    That forecast should remain labelled according to
    its actual API timestamps.
    """

    output = []

    records = (
        (payload.get("data") or {})
        .get("records")
        or []
    )

    for record in records:
        issued = (
            record.get("timestamp")
            or ""
        )

        updated = (
            record.get(
                "updatedTimestamp"
            )
            or issued
        )

        record_date = (
            record.get("date")
            or ""
        )

        prepared_periods = []

        for period in (
            record.get("periods")
            or []
        ):
            period_time = (
                period.get("timePeriod")
                or period.get("time")
                or {}
            )

            start = (
                period_time.get("start")
                or ""
            )

            end = (
                period_time.get("end")
                or ""
            )

            window = normalized_window(
                start,
                end,
            )

            regional = (
                period.get("regions")
                or period.get(
                    "regionForecasts"
                )
                or {}
            )

            values = {}
            thunder = []

            for region in REGIONS:
                item = (
                    regional.get(region)
                    or {}
                )

                if isinstance(
                    item,
                    dict,
                ):
                    text = (
                        item.get("text")
                        or item.get(
                            "forecast"
                        )
                        or ""
                    )

                    code = (
                        item.get("code")
                        or item.get(
                            "forecastCode"
                        )
                        or ""
                    )

                else:
                    text = str(
                        item or ""
                    )

                    code = ""

                values[
                    f"{region}_forecast"
                ] = text

                values[
                    f"{region}_code"
                ] = code

                if is_thunder_family(
                    text
                ):
                    thunder.append(
                        region
                    )

            parsed_start = (
                parse_datetime(start)
            )

            period_date = (
                parsed_start
                .date()
                .isoformat()
                if parsed_start
                else record_date
            )

            prepared_periods.append(
                {
                    "record_date": (
                        record_date
                    ),
                    "issued": issued,
                    "updated": updated,
                    "start": start,
                    "end": end,
                    "period_text": (
                        period_time.get(
                            "text"
                        )
                        or period.get(
                            "name"
                        )
                        or ""
                    ),
                    "period_date": (
                        period_date
                    ),
                    "window": window,
                    "values": values,
                    "thunder": thunder,
                }
            )

        thunder_by_period = {
            (
                item["period_date"],
                item["window"],
            ): set(
                item["thunder"]
            )
            for item
            in prepared_periods
        }

        for item in prepared_periods:
            communication = ""

            if (
                item["window"]
                == "12:00-18:00"
            ):
                morning_thunder = (
                    thunder_by_period.get(
                        (
                            item[
                                "period_date"
                            ],
                            "06:00-12:00",
                        ),
                        set(),
                    )
                )

                lunch_regions = [
                    region
                    for region in REGIONS
                    if (
                        region
                        in morning_thunder
                        and region
                        in item["thunder"]
                    )
                ]

                try:
                    campaign_date = (
                        dt.date.fromisoformat(
                            item[
                                "period_date"
                            ]
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    campaign_date = None

                communication = (
                    communication_messages(
                        "lunch",
                        lunch_regions,
                        campaign_date,
                    )
                )

            elif (
                item["window"]
                == "18:00-00:00"
            ):
                dinner_regions = [
                    region
                    for region in REGIONS
                    if region
                    in item["thunder"]
                ]

                try:
                    campaign_date = (
                        dt.date.fromisoformat(
                            item[
                                "period_date"
                            ]
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    campaign_date = None

                communication = (
                    communication_messages(
                        "dinner",
                        dinner_regions,
                        campaign_date,
                    )
                )

            output.append(
                [
                    item["record_date"],
                    item["issued"],
                    item["updated"],
                    item["window"],
                    campaign_use(
                        item["window"]
                    ),
                    item["start"],
                    item["end"],
                    item["period_text"],
                    *(
                        item["values"][
                            f"{region}_forecast"
                        ]
                        for region in REGIONS
                    ),
                    "|".join(
                        item["thunder"]
                    ),
                    communication,
                    *(
                        item["values"][
                            f"{region}_code"
                        ]
                        for region in REGIONS
                    ),
                ]
            )

    return output


# ============================================================
# SHEET ROW HELPERS
# ============================================================


def row_dicts(values):
    if not values:
        return []

    header = values[0]

    return [
        dict(
            zip(
                header,
                row,
            )
        )
        for row in values[1:]
        if row
    ]


def latest_issued_date(values):
    latest = None

    for row in row_dicts(
        values
    ):
        issued = parse_datetime(
            row.get(
                "issued_ts"
            )
        )

        if (
            issued
            and (
                latest is None
                or issued > latest
            )
        ):
            latest = issued

    return (
        latest.date()
        if latest
        else None
    )


def retain_recent_rows(
    rows,
    cutoff_date,
):
    """
    Keep only rows whose issued timestamp falls within
    the retention period.
    """

    retained = []

    issued_idx = HEADERS.index(
        "issued_ts"
    )

    for row in rows:
        if len(row) <= issued_idx:
            continue

        issued = parse_datetime(
            row[issued_idx]
        )

        if (
            issued
            and issued.date()
            >= cutoff_date
        ):
            retained.append(
                row
            )

    return retained


def deduplicate(rows):
    """
    Ensure worksheet rows are unique.

    Unique forecast-period key:
        issued_ts + period_start

    Why:
    - One forecast issue contains multiple forecast periods.
    - period_start separates those periods.
    - Fetching "latest" multiple times may return exactly the
      same forecast again.

    If the same issued_ts + period_start appears multiple times,
    keep whichever version has the newest updated_ts.
    """

    by_key = {}

    issued_idx = HEADERS.index(
        "issued_ts"
    )

    start_idx = HEADERS.index(
        "period_start"
    )

    updated_idx = HEADERS.index(
        "updated_ts"
    )

    for row in rows:
        # Pad shorter existing rows just in case.
        row = list(row) + (
            [""] * (
                len(HEADERS)
                - len(row)
            )
        )

        issued_value = (
            row[issued_idx]
        )

        start_value = (
            row[start_idx]
        )

        # Ignore malformed rows that cannot have
        # a meaningful unique key.
        if (
            not issued_value
            or not start_value
        ):
            continue

        key = (
            issued_value,
            start_value,
        )

        current = (
            by_key.get(key)
        )

        if current is None:
            by_key[key] = row
            continue

        old_updated = (
            parse_datetime(
                current[
                    updated_idx
                ]
            )
            or dt.datetime.min.replace(
                tzinfo=SGT
            )
        )

        new_updated = (
            parse_datetime(
                row[
                    updated_idx
                ]
            )
            or dt.datetime.min.replace(
                tzinfo=SGT
            )
        )

        if (
            new_updated
            >= old_updated
        ):
            by_key[key] = row

    return sorted(
        by_key.values(),
        key=lambda row: (
            parse_datetime(
                row[issued_idx]
            )
            or dt.datetime.min.replace(
                tzinfo=SGT
            ),
            parse_datetime(
                row[start_idx]
            )
            or dt.datetime.min.replace(
                tzinfo=SGT
            ),
        ),
    )


# ============================================================
# COMMUNICATION LOGIC
# ============================================================


def keep_only_cutoff_communications(
    rows,
):
    """
    Populate campaign_comms only on the latest forecast
    available at each campaign cutoff.
    """

    comms_idx = HEADERS.index(
        "campaign_comms"
    )

    window_idx = HEADERS.index(
        "forecast_window"
    )

    start_idx = HEADERS.index(
        "period_start"
    )

    issued_idx = HEADERS.index(
        "issued_ts"
    )

    updated_idx = HEADERS.index(
        "updated_ts"
    )

    result = [
        list(row)
        for row in rows
    ]

    for row in result:
        row[comms_idx] = ""

    candidates = {}

    rows_by_issue_period = {}

    for index, row in enumerate(
        rows
    ):
        window = (
            row[window_idx]
        )

        period_start = (
            parse_datetime(
                row[start_idx]
            )
        )

        available = (
            parse_datetime(
                row[updated_idx]
            )
            or parse_datetime(
                row[issued_idx]
            )
        )

        if (
            not period_start
            or not available
        ):
            continue

        target_date = (
            period_start.date()
        )

        rows_by_issue_period[
            (
                row[issued_idx],
                target_date,
                window,
            )
        ] = row

        if (
            window
            == "12:00-18:00"
        ):
            kind = "lunch"

            cutoff = (
                dt.datetime.combine(
                    (
                        target_date
                        - dt.timedelta(
                            days=1
                        )
                    ),
                    dt.time(
                        23,
                        30,
                    ),
                    SGT,
                )
            )

        elif (
            window
            == "18:00-00:00"
        ):
            kind = "dinner"

            cutoff = (
                dt.datetime.combine(
                    target_date,
                    dt.time(
                        5,
                        30,
                    ),
                    SGT,
                )
            )

        else:
            continue

        if available > cutoff:
            continue

        key = (
            target_date,
            kind,
        )

        current = (
            candidates.get(
                key
            )
        )

        if (
            current is None
            or available
            > current[0]
        ):
            candidates[key] = (
                available,
                index,
            )

    thunder_idx = HEADERS.index(
        "thunder_regions"
    )

    for (
        target_date,
        kind,
    ), (
        _,
        index,
    ) in candidates.items():

        selected = (
            rows[index]
        )

        if kind == "lunch":
            morning = (
                rows_by_issue_period.get(
                    (
                        selected[
                            issued_idx
                        ],
                        target_date,
                        "06:00-12:00",
                    )
                )
            )

            morning_regions = set(
                str(
                    morning[
                        thunder_idx
                    ]
                    if morning
                    else ""
                ).split("|")
            )

            afternoon_regions = set(
                str(
                    selected[
                        thunder_idx
                    ]
                ).split("|")
            )

            regions = [
                region
                for region in REGIONS
                if (
                    region
                    in morning_regions
                    and region
                    in afternoon_regions
                )
            ]

        else:
            selected_regions = set(
                str(
                    selected[
                        thunder_idx
                    ]
                ).split("|")
            )

            regions = [
                region
                for region in REGIONS
                if region
                in selected_regions
            ]

        result[index][
            comms_idx
        ] = communication_messages(
            kind,
            regions,
            target_date,
        )

    return result


def migrate_legacy_rows(values):
    """
    Accept rows only when the existing worksheet has
    the expected schema.
    """

    if (
        not values
        or values[0]
        != HEADERS
    ):
        return []

    return [
        row
        + [""] * (
            len(HEADERS)
            - len(row)
        )
        for row in values[1:]
        if row
    ]


# ============================================================
# FORECAST REFRESH
# ============================================================


def run_24h_forecast_holder(
    sh,
    today=None,
):
    today = (
        today
        or dt.datetime.now(
            SGT
        ).date()
    )

    cutoff_date = (
        subtract_calendar_months(
            today,
            3,
        )
    )

    try:
        ws = sh.worksheet(
            SHEET_NAME
        )

        original_values = (
            ws.get_all_values()
        )

    except Exception:
        ws = ensure_worksheet(
            sh,
            SHEET_NAME,
            HEADERS,
        )

        original_values = (
            ws.get_all_values()
        )

    existing_rows = (
        migrate_legacy_rows(
            original_values
        )
    )

    # --------------------------------------------------------
    # FIRST RUN
    #
    # Start from the first day two calendar months ago.
    #
    # Example:
    # 10 Sep 2026 -> 1 Jul 2026
    #
    # NORMAL RUN
    #
    # Request yesterday + today.
    # --------------------------------------------------------

    start_date = (
        (
            today
            - dt.timedelta(
                days=1
            )
        )
        if existing_rows
        else first_day_months_ago(
            today,
            2,
        )
    )

    start_date = max(
        start_date,
        cutoff_date,
    )

    fetched_rows = []

    failures = []

    # --------------------------------------------------------
    # REQUEST EACH DATE
    # --------------------------------------------------------

    for day in dates_inclusive(
        start_date,
        today,
    ):
        try:
            payload = fetch_day(
                day,
                fallback_to_latest=(
                    day == today
                ),
            )

            day_rows = (
                flatten_payload(
                    payload
                )
            )

            fetched_rows.extend(
                day_rows
            )

            print(
                f"forecast_24h: "
                f"requested {day}; "
                f"received "
                f"{len(day_rows)} rows"
            )

            # Useful diagnostic when fallback returns
            # a record belonging to another API date.
            returned_dates = sorted(
                {
                    str(
                        row[
                            HEADERS.index(
                                "date"
                            )
                        ]
                    )
                    for row
                    in day_rows
                    if row
                }
            )

            if returned_dates:
                print(
                    f"forecast_24h: "
                    f"requested {day}; "
                    f"API record date(s): "
                    f"{', '.join(returned_dates)}"
                )

        except RuntimeError as exc:
            failures.append(
                str(exc)
            )

    # --------------------------------------------------------
    # REAL API FAILURE:
    #
    # Abort BEFORE touching Google Sheets.
    #
    # A today's-date 404 does not reach here because it
    # falls back to fetch_latest().
    # --------------------------------------------------------

    if failures:
        raise RuntimeError(
            f"24h forecast refresh "
            f"aborted without changing "
            f"the sheet; "
            f"{len(failures)} date(s) "
            f"failed. "
            f"First error: "
            f"{failures[0]}"
        )

    # --------------------------------------------------------
    # DO NOT DELETE YESTERDAY/TODAY FIRST.
    #
    # Instead:
    #
    # existing history
    #       +
    # newly fetched data
    #       ↓
    # retain L3M
    #       ↓
    # deduplicate
    #
    # This is particularly important when today's dated
    # endpoint returns 404 and the fallback returns a
    # forecast vintage already present in the sheet.
    # --------------------------------------------------------

    combined = (
        retain_recent_rows(
            (
                existing_rows
                + fetched_rows
            ),
            cutoff_date,
        )
    )

    before_dedup = len(
        combined
    )

    combined = deduplicate(
        combined
    )

    after_dedup = len(
        combined
    )

    duplicates_removed = (
        before_dedup
        - after_dedup
    )

    print(
        f"forecast_24h: "
        f"deduplication removed "
        f"{duplicates_removed} "
        f"duplicate/replaced rows"
    )

    combined = (
        keep_only_cutoff_communications(
            combined
        )
    )

    # --------------------------------------------------------
    # Only now rewrite the worksheet.
    # --------------------------------------------------------

    ws.clear()

    ws.append_rows(
        [
            HEADERS,
            *combined,
        ],
        value_input_option=(
            "USER_ENTERED"
        ),
    )

    print(
        f"forecast_24h: "
        f"requested "
        f"{start_date} to {today}; "
        f"stored "
        f"{len(combined)} unique rows "
        f"from "
        f"{cutoff_date} onward"
    )

    return combined


# ============================================================
# CAMPAIGN HELPERS
# ============================================================


def target_date_for(
    campaign,
    now,
):
    if campaign == "dinner":
        return now.date()

    # Lunch communications are for tomorrow when
    # running the night-before job.
    return (
        now.date()
        + dt.timedelta(
            days=1
        )
        if now.hour >= 12
        else now.date()
    )


def decision_cutoff(
    campaign,
    target_date,
):
    if campaign == "lunch":
        return dt.datetime.combine(
            (
                target_date
                - dt.timedelta(
                    days=1
                )
            ),
            dt.time(
                23,
                30,
            ),
            SGT,
        )

    return dt.datetime.combine(
        target_date,
        dt.time(
            5,
            30,
        ),
        SGT,
    )


def select_latest_campaign_row(
    rows,
    campaign,
    target_date,
):
    expected_use = (
        "lunch_block_2"
        if campaign == "lunch"
        else "dinner"
    )

    cutoff = decision_cutoff(
        campaign,
        target_date,
    )

    candidates = []

    for row in rows:
        record = dict(
            zip(
                HEADERS,
                row,
            )
        )

        if (
            str(
                record.get(
                    "campaign_use",
                    "",
                )
            ).strip()
            != expected_use
        ):
            continue

        period_start = (
            parse_datetime(
                record.get(
                    "period_start"
                )
            )
        )

        available = (
            parse_datetime(
                record.get(
                    "updated_ts"
                )
            )
            or parse_datetime(
                record.get(
                    "issued_ts"
                )
            )
        )

        if (
            not period_start
            or not available
        ):
            continue

        if (
            period_start.date()
            == target_date
            and available
            <= cutoff
        ):
            candidates.append(
                (
                    available,
                    record,
                )
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item[0],
    )[1]


# ============================================================
# TELEGRAM
# ============================================================


def send_telegram(
    token,
    chat_id,
    message,
):
    response = requests.post(
        (
            f"https://api.telegram.org/"
            f"bot{token}/sendMessage"
        ),
        json={
            "chat_id": chat_id,
            "text": message,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = (
        response.json()
    )

    if not payload.get("ok"):
        raise RuntimeError(
            f"Telegram rejected "
            f"the message: "
            f"{payload}"
        )

    return (
        payload["result"]
        .get(
            "message_id",
            "",
        )
    )


def send_campaign_comms(
    sh,
    rows,
    campaign,
    dry_run=False,
    now=None,
    target_date=None,
):
    now = (
        now
        or dt.datetime.now(
            SGT
        )
    )

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if (
        not dry_run
        and (
            not token
            or not chat_id
        )
    ):
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID "
            "are required"
        )

    target_date = (
        target_date
        or target_date_for(
            campaign,
            now,
        )
    )

    selected = (
        select_latest_campaign_row(
            rows,
            campaign,
            target_date,
        )
    )

    if selected is None:
        print(
            f"{campaign} "
            f"{target_date}: "
            f"no eligible forecast "
            f"row; nothing sent"
        )

        return "no_forecast"

    message = str(
        selected.get(
            "campaign_comms",
            "",
        )
    ).strip()

    if not message:
        print(
            f"{campaign} "
            f"{target_date}: "
            f"latest forecast "
            f"has no comms; "
            f"nothing sent"
        )

        return "no_comms"

    notification_key = (
        f"{campaign}:"
        f"{target_date.isoformat()}"
    )

    log_ws = ensure_worksheet(
        sh,
        COMMS_LOG_SHEET,
        COMMS_LOG_HEADERS,
    )

    log_values = (
        log_ws.get_all_values()
    )

    existing_keys = {
        row[0]
        for row in log_values[1:]
        if row
    }

    if (
        notification_key
        in existing_keys
    ):
        print(
            f"{notification_key}: "
            f"already sent; "
            f"nothing sent"
        )

        return "duplicate"

    if dry_run:
        print(
            f"DRY RUN "
            f"{notification_key}: "
            f"{message}"
        )

        return "dry_run"

    message_id = (
        send_telegram(
            token,
            chat_id,
            message,
        )
    )

    log_ws.append_row(
        [
            notification_key,
            campaign,
            target_date.isoformat(),
            selected.get(
                "issued_ts",
                "",
            ),
            selected.get(
                "updated_ts",
                "",
            ),
            message,
            chat_id,
            message_id,
            now.isoformat(),
        ],
        value_input_option=(
            "USER_ENTERED"
        ),
    )

    print(
        f"{notification_key}: "
        f"sent Telegram message "
        f"{message_id}"
    )

    return "sent"


# ============================================================
# CLI
# ============================================================


def main():
    parser = (
        argparse.ArgumentParser(
            description=__doc__
        )
    )

    parser.add_argument(
        "--sheet-id",
        default=os.getenv(
            "SHEET_ID"
        ),
    )

    parser.add_argument(
        "--notify",
        action="store_true",
    )

    parser.add_argument(
        "--campaign",
        choices=(
            "lunch",
            "dinner",
        ),
    )

    parser.add_argument(
        "--target-date",
        type=dt.date.fromisoformat,
        help=(
            "Override the campaign "
            "date (YYYY-MM-DD), "
            "mainly for testing"
        ),
    )

    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help=(
            "Use rows already in "
            "forecast_24h instead of "
            "calling the NEA API"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if not args.sheet_id:
        raise SystemExit(
            "SHEET_ID is required"
        )

    if (
        args.notify
        and not args.campaign
    ):
        raise SystemExit(
            "--campaign is required "
            "when --notify is used"
        )

    from consolidate import (
        open_sheet_by_id,
    )

    sh = open_sheet_by_id(
        args.sheet_id
    )

    if args.skip_refresh:
        if not args.notify:
            raise SystemExit(
                "--skip-refresh is "
                "only valid with "
                "--notify"
            )

        values = (
            sh.worksheet(
                SHEET_NAME
            ).get_all_values()
        )

        if (
            not values
            or values[0]
            != HEADERS
        ):
            raise RuntimeError(
                f"{SHEET_NAME} "
                f"does not have the "
                f"expected columns"
            )

        rows = [
            row
            + [""] * (
                len(HEADERS)
                - len(row)
            )
            for row in values[1:]
            if row
        ]

    else:
        rows = (
            run_24h_forecast_holder(
                sh
            )
        )

    if args.notify:
        send_campaign_comms(
            sh,
            rows,
            args.campaign,
            dry_run=(
                args.dry_run
            ),
            target_date=(
                args.target_date
            ),
        )


if __name__ == "__main__":
    main()
