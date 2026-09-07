import os, gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", 
          "https://www.googleapis.com/auth/drive.readonly"]

def open_sheet(sheet_name: str):
    creds_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc.open(sheet_name)

def ensure_worksheet(sh, title, headers):
    ws = None
    for w in sh.worksheets():
        if w.title == title:
            ws = w
            break
    if ws is None:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(10, len(headers)))
        if headers:
            ws.append_row(headers, value_input_option="USER_ENTERED")
    else:
        if headers:
            existing = ws.row_values(1)
            if existing != headers:
                ws.clear()
                ws.append_row(headers, value_input_option="USER_ENTERED")
    return ws

def prune_older_than(ws: gspread.Worksheet, time_col: str = "reading_time", max_age_hours: int = 24):
    """
    Keep only rows whose <time_col> is within the last `max_age_hours` hours.
    Assumes `time_col` is in the header row, and timestamps are ISO8601 strings.
    """
    values = ws.get_all_values()
    if not values:
        return

    header = values[0]
    if time_col not in header:
        # nothing to do if the time column isn't present
        return

    time_idx = header.index(time_col)
    # Compare timestamps using Singapore time instead of UTC.
    cutoff = datetime.now(SG_TZ) - timedelta(hours=max_age_hours)

    new_rows = [header]  # keep header

    for row in values[1:]:
        if len(row) <= time_idx:
            continue
        ts_str = row[time_idx].strip()
        if not ts_str:
            continue

        try:
            # adjust this parsing if your format is different
            # e.g. "2025-11-02T21:45:00+08:00" or "...Z"
            ts = datetime.fromisoformat(
                ts_str.replace("Z", "+00:00")
            )
            # ensure it's aware; if naive, assume Singapore time
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=SG_TZ)
            ts = ts.astimezone(SG_TZ)
        except Exception:
            # if can't parse timestamp, skip this row
            continue

        if ts >= cutoff:
            new_rows.append(row)

    # Rewrite sheet with only the recent rows
    ws.clear()
    if new_rows:
        ws.append_rows(new_rows, value_input_option="USER_ENTERED")
