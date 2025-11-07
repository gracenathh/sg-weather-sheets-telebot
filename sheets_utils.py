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
