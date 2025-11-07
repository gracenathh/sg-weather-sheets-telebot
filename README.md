# SG Weather → Google Sheets + Telegram Bot

Fetches and stores real-time Singapore weather data from [data.gov.sg](https://data.gov.sg) APIs into Google Sheets.  
Includes optional Telegram + OpenAI summaries (coming soon).

---

## 🧠 Overview

All data goes into a single Google Sheet, automatically created or updated by the script:

- `rainfall_data`, `rainfall_daily_combined`
- `forecast_24h`
- (optional) `mets_data`, `forecast_2h`, etc.
- `stations` and other mapping tabs

You can schedule this to run continuously on **GitHub Actions**, even when your laptop is off.

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/gracenathh/sg-weather-sheets-telebot.git
cd sg-weather-sheets-telebot
