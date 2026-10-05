import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "XOM":  "ExxonMobil",
    "TSLA": "Tesla",
    "NEE":  "NextEra Energy",
}

START_DATE = "2019-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# ─────────────────────────────────────────────
# 2. ESG / EMISSIONS DATA
#    Source: CDP & company sustainability reports
#    Scope 1+2 emissions in metric tons CO2e (2022)
# ─────────────────────────────────────────────

esg_data = pd.DataFrame({
    "Ticker":          ["AAPL",  "MSFT",      "XOM",       "TSLA",  "NEE"],
    "Company":         ["Apple", "Microsoft", "ExxonMobil","Tesla", "NextEra Energy"],
    "Emissions_MtCO2": [22.6,    13.5,         112000,      97.5,    6800],
    "ESG_Score":       [72,       88,            32,          45,      78],
    "Renewable_Pct":   [100,      100,            2,           70,      67],
})

# ─────────────────────────────────────────────
# 3. FETCH STOCK DATA
# ─────────────────────────────────────────────

print("Fetching stock data from Yahoo Finance...")
price_data = {}

for ticker in COMPANIES:
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    df = df[["Close"]].rename(columns={"Close": ticker})
    price_data[ticker] = df

prices = pd.concat(price_data.values(), axis=1)
prices.index = pd.to_datetime(prices.index)

if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.droplevel(1)

print(f"Loaded {prices.shape[0]} trading days x {prices.shape[1]} tickers\n")

# ─────────────────────────────────────────────
# 4. CALCULATE 5-YEAR RETURN
# ─────────────────────────────────────────────

returns = {}
for ticker in COMPANIES:
    series = prices[ticker].dropna()
    pct_return = ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
    returns[ticker] = round(float(pct_return), 2)

esg_data["5yr_Return_Pct"] = esg_data["Ticker"].map(returns)

print("5-Year Returns:")
print(esg_data[["Company", "5yr_Return_Pct"]].to_string(index=False))
print()

# ─────────────────────────────────────────────
# 5. NORMALIZE PRICES TO BASE 100
# ─────────────────────────────────────────────

normalized = prices.div(prices.iloc[0]) * 100

# ─────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────

COLORS = {
    "AAPL": "#555555",
    "MSFT": "#00a2ed",
    "XOM":  "#e63946",
    "TSLA": "#ef233c",
    "NEE":  "#2a9d8f",
}

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Corporate Sustainability vs. Stock Performance (5-Year Analysis)",
             fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1: Normalized price history (full width)
ax1 = fig.add_subplot(gs[0, :])
for ticker, name in COMPANIES.items():
    ax1.plot(normalized.index, normalized[ticker],
             label=f"{name} ({ticker})", color=COLORS[ticker], linewidth=2)
ax1.axhline(100, color="gray", linestyle="--", linewidth=0.8, label="Baseline (100)")
ax1.set_title("Normalized Stock Price (Base = 100 at Start)", fontsize=13)
ax1.set_xlabel("Date")
ax1.set_ylabel("Indexed Price")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Emissions vs. 5-Year Return scatter
ax2 = fig.add_subplot(gs[1, 0])
log_emissions = np.log10(esg_data["Emissions_MtCO2"])
ax2.scatter(log_emissions, esg_data["5yr_Return_Pct"],
            c=[COLORS[t] for t in esg_data["Ticker"]], s=120, zorder=5)
for _, row in esg_data.iterrows():
    ax2.annotate(row["Company"],
                 (np.log10(row["Emissions_MtCO2"]), row["5yr_Return_Pct"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
ax2.set_title("Emissions vs. 5-Year Stock Return", fontsize=12)
ax2.set_xlabel("Log10(Scope 1+2 Emissions, tCO2e)")
ax2.set_ylabel("5-Year Return (%)")
ax2.grid(True, alpha=0.3)

# Panel 3: Bar chart with ESG score labels
ax3 = fig.add_subplot(gs[1, 1])
bars = ax3.bar(esg_data["Company"], esg_data["5yr_Return_Pct"],
               color=[COLORS[t] for t in esg_data["Ticker"]],
               edgecolor="white", linewidth=0.5)
for bar, score in zip(bars, esg_data["ESG_Score"]):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 5,
             f"ESG: {score}", ha="center", va="bottom", fontsize=8, color="dimgray")
ax3.set_title("5-Year Return by Company\n(ESG Score labeled)", fontsize=12)
ax3.set_xlabel("Company")
ax3.set_ylabel("5-Year Return (%)")
ax3.tick_params(axis="x", rotation=20)
ax3.grid(True, axis="y", alpha=0.3)

plt.savefig("sustainability_analysis.png", dpi=150, bbox_inches="tight")
print("Chart saved to sustainability_analysis.png")
plt.show()

# ─────────────────────────────────────────────
# 7. SUMMARY TABLE
# ─────────────────────────────────────────────

print("\nFull Summary Table:")
summary = esg_data[["Company", "Emissions_MtCO2", "ESG_Score", "Renewable_Pct", "5yr_Return_Pct"]].copy()
summary.columns = ["Company", "Emissions (tCO2e)", "ESG Score", "Renewable %", "5yr Return %"]
print(summary.to_string(index=False))
