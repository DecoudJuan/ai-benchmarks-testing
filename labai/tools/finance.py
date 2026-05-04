"""
Finance tools for agent evaluation — backed by Yahoo Finance (yfinance).

Real-time data: prices fetched live, no API key required.
Financial ratios come from yfinance .info (TTM / most recent filing).

Registered tools:
    get_stock_price        — live price + daily change
    get_financial_ratios   — P/E, P/B, ROE, debt/equity, dividend yield
    calculate_return       — return calculation given buy price and shares
    compare_companies      — side-by-side price + ratios for multiple tickers
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import yfinance as yf

from labai.core.base import BaseTool
from labai.core.registry import Registry

_executor = ThreadPoolExecutor(max_workers=6)


def _run_sync(fn):
    """Run a synchronous yfinance call in a thread to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, fn)


def _fetch_ticker_data(ticker: str) -> dict:
    """Fetch price history + info for a ticker. Returns combined dict."""
    t    = yf.Ticker(ticker)
    info = t.info or {}
    hist = t.history(period="5d")

    current_price = None
    prev_close    = None

    if not hist.empty:
        current_price = float(hist["Close"].iloc[-1])
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
        else:
            prev_close = info.get("previousClose") or current_price
    else:
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close    = info.get("previousClose") or current_price

    change_pct = 0.0
    if current_price and prev_close and prev_close != 0:
        change_pct = ((current_price - prev_close) / prev_close) * 100

    return {
        "price":        current_price,
        "prev_close":   prev_close,
        "change_pct":   change_pct,
        "name":         info.get("shortName") or info.get("longName") or ticker,
        "pe":           info.get("trailingPE"),
        "forward_pe":   info.get("forwardPE"),
        "pb":           info.get("priceToBook"),
        "roe":          (info.get("returnOnEquity") or 0) * 100,
        "debt_equity":  (info.get("debtToEquity") or 0) / 100,
        "div_yield":    _calc_div_yield(info, current_price),
        "market_cap":   info.get("marketCap"),
        "sector":       info.get("sector", ""),
        "currency":     info.get("currency", "USD"),
    }


def _calc_div_yield(info: dict, current_price: float | None) -> float:
    """Calculate dividend yield % reliably from dividendRate/price or dividendYield."""
    div_rate = info.get("dividendRate") or 0
    if div_rate and current_price:
        return (div_rate / current_price) * 100
    raw = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0
    # yfinance sometimes returns as decimal (0.007), sometimes as pct (0.73)
    return raw * 100 if raw < 1 else raw


def _fmt_price(price, currency="USD") -> str:
    if price is None:
        return "N/A"
    sym = "$" if currency == "USD" else f"{currency} "
    return f"{sym}{price:,.2f}"

def _fmt_ratio(val, suffix="x", decimals=1) -> str:
    if val is None or val == 0:
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


# ── Tool implementations ───────────────────────────────────────────────────────

@Registry.tool("get_stock_price")
class StockPriceTool(BaseTool):
    """Returns the live stock price and daily change for any ticker symbol."""

    name        = "get_stock_price"
    description = (
        "Get the current (live) stock price and daily percentage change for any "
        "publicly traded ticker symbol (e.g. AAPL, MSFT, GOOGL, TSLA, NVDA, JPM, "
        "BRK-B, AMZN, META, NFLX, or any other valid ticker)."
    )

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'BRK-B').",
                        },
                    },
                    "required": ["ticker"],
                },
            },
        }

    async def execute(self, ticker: str, **_: Any) -> str:
        ticker = ticker.upper().strip().replace(".", "-")
        try:
            data = await _run_sync(lambda: _fetch_ticker_data(ticker))
        except Exception as exc:
            return f"Error fetching data for '{ticker}': {exc}"

        if data["price"] is None:
            return f"No price data available for ticker '{ticker}'. Verify the symbol is correct."

        direction = "+" if data["change_pct"] >= 0 else ""
        currency  = data["currency"]
        cap_str   = ""
        if data["market_cap"]:
            cap_b = data["market_cap"] / 1e9
            cap_str = f"\n  Market cap    : ${cap_b:,.1f}B"

        return (
            f"{data['name']} ({ticker})\n"
            f"  Current price : {_fmt_price(data['price'], currency)}\n"
            f"  Daily change  : {direction}{data['change_pct']:.2f}%"
            f"{cap_str}"
            + (f"\n  Sector        : {data['sector']}" if data["sector"] else "")
        )


@Registry.tool("get_financial_ratios")
class FinancialRatiosTool(BaseTool):
    """Returns key financial ratios (P/E, P/B, ROE, debt/equity, dividend yield) via Yahoo Finance."""

    name        = "get_financial_ratios"
    description = (
        "Get key financial ratios for any publicly traded stock: trailing P/E, forward P/E, "
        "P/B, ROE (%), debt-to-equity, and dividend yield (%). "
        "Data sourced live from Yahoo Finance."
    )

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol.",
                        },
                    },
                    "required": ["ticker"],
                },
            },
        }

    async def execute(self, ticker: str, **_: Any) -> str:
        ticker = ticker.upper().strip().replace(".", "-")
        try:
            data = await _run_sync(lambda: _fetch_ticker_data(ticker))
        except Exception as exc:
            return f"Error fetching ratios for '{ticker}': {exc}"

        if data["price"] is None:
            return f"No data available for ticker '{ticker}'. Verify the symbol is correct."

        return (
            f"Financial ratios for {data['name']} ({ticker}):\n"
            f"  Trailing P/E     : {_fmt_ratio(data['pe'])}\n"
            f"  Forward P/E      : {_fmt_ratio(data['forward_pe'])}\n"
            f"  P/B ratio        : {_fmt_ratio(data['pb'])}\n"
            f"  ROE              : {_fmt_ratio(data['roe'], '%')}\n"
            f"  Debt / Equity    : {_fmt_ratio(data['debt_equity'], '', 2)}\n"
            f"  Dividend yield   : {_fmt_ratio(data['div_yield'], '%', 2)}"
        )


@Registry.tool("calculate_return")
class CalculateReturnTool(BaseTool):
    """Calculates total and annualized return on a stock investment."""

    name        = "calculate_return"
    description = (
        "Calculate the total return (dollars and percentage) and optionally the "
        "annualized return for a stock investment given buy price, current price, "
        "number of shares, and optional holding period in years."
    )

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "buy_price": {
                            "type": "number",
                            "description": "Price per share at purchase.",
                        },
                        "current_price": {
                            "type": "number",
                            "description": "Current price per share.",
                        },
                        "shares": {
                            "type": "number",
                            "description": "Number of shares held.",
                        },
                        "years_held": {
                            "type": "number",
                            "description": "Years the investment has been held (optional, for annualized return).",
                        },
                    },
                    "required": ["buy_price", "current_price", "shares"],
                },
            },
        }

    async def execute(
        self,
        buy_price:     float,
        current_price: float,
        shares:        float,
        years_held:    float | None = None,
        **_: Any,
    ) -> str:
        cost          = buy_price * shares
        current_value = current_price * shares
        gain          = current_value - cost
        pct_return    = (gain / cost) * 100 if cost else 0.0

        lines = [
            "Investment analysis:",
            f"  Shares bought    : {shares:.4g}",
            f"  Cost basis       : ${cost:,.2f}",
            f"  Current value    : ${current_value:,.2f}",
            f"  Gain / Loss      : ${gain:+,.2f}",
            f"  Total return     : {pct_return:+.2f}%",
        ]

        if years_held and years_held > 0:
            annualized = ((current_value / cost) ** (1 / years_held) - 1) * 100
            lines.append(f"  Annualized return: {annualized:+.2f}% per year")

        return "\n".join(lines)


@Registry.tool("compare_companies")
class CompareCompaniesTool(BaseTool):
    """Side-by-side comparison of live price and key ratios for multiple tickers."""

    name        = "compare_companies"
    description = (
        "Compare two or more companies side-by-side on live stock price, P/E, P/B, "
        "ROE, debt-to-equity, and dividend yield. Works with any valid ticker symbols."
    )

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 2-5 ticker symbols to compare.",
                        },
                    },
                    "required": ["tickers"],
                },
            },
        }

    async def execute(self, tickers: list[str], **_: Any) -> str:
        cleaned = [t.upper().strip().replace(".", "-") for t in tickers]

        # Fetch all tickers concurrently
        async def fetch_one(ticker: str):
            try:
                return ticker, await _run_sync(lambda: _fetch_ticker_data(ticker))
            except Exception as exc:
                return ticker, {"error": str(exc)}

        results_raw = await asyncio.gather(*[fetch_one(t) for t in cleaned])

        rows = []
        for ticker, data in results_raw:
            if "error" in data:
                rows.append(f"  {ticker}: error — {data['error']}")
                continue
            if data["price"] is None:
                rows.append(f"  {ticker}: no data available")
                continue

            direction = "+" if data["change_pct"] >= 0 else ""
            rows.append(
                f"  {ticker} ({data['name']}):\n"
                f"    Price      : {_fmt_price(data['price'], data['currency'])} "
                f"({direction}{data['change_pct']:.2f}%)\n"
                f"    P/E (TTM)  : {_fmt_ratio(data['pe'])}\n"
                f"    P/B        : {_fmt_ratio(data['pb'])}\n"
                f"    ROE        : {_fmt_ratio(data['roe'], '%')}\n"
                f"    D/E        : {_fmt_ratio(data['debt_equity'], '', 2)}\n"
                f"    Div. Yield : {_fmt_ratio(data['div_yield'], '%', 2)}"
            )

        return "Company comparison (live data):\n\n" + "\n\n".join(rows)
