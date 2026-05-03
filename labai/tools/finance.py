"""
Finance tools for agent evaluation.

These tools simulate real finance data APIs with mock data.
Replace the mock data dicts with actual API calls (Yahoo Finance, Alpha Vantage,
Bloomberg, etc.) when moving to production.

Registered tools:
    get_stock_price        — current price + daily change
    get_financial_ratios   — P/E, P/B, ROE, debt/equity, dividend yield
    calculate_return       — return calculation given buy price and shares
    compare_companies      — side-by-side price + ratios for multiple tickers
"""

from __future__ import annotations

from typing import Any

from labai.core.base import BaseTool
from labai.core.registry import Registry

# ── Mock data ──────────────────────────────────────────────────────────────────
# In production: replace with real API calls.

_PRICES: dict[str, dict] = {
    "AAPL":  {"price": 178.72,  "change_pct":  0.83,  "name": "Apple Inc."},
    "MSFT":  {"price": 378.85,  "change_pct":  1.12,  "name": "Microsoft Corp."},
    "GOOGL": {"price": 163.45,  "change_pct": -0.34,  "name": "Alphabet Inc."},
    "AMZN":  {"price": 178.25,  "change_pct":  2.15,  "name": "Amazon.com Inc."},
    "TSLA":  {"price": 242.10,  "change_pct": -1.87,  "name": "Tesla Inc."},
    "NVDA":  {"price": 875.40,  "change_pct":  3.21,  "name": "NVIDIA Corp."},
    "META":  {"price": 473.28,  "change_pct":  0.55,  "name": "Meta Platforms"},
    "NFLX":  {"price": 608.12,  "change_pct": -0.72,  "name": "Netflix Inc."},
    "JPM":   {"price": 193.47,  "change_pct":  0.41,  "name": "JPMorgan Chase"},
    "BRK.B": {"price": 356.20,  "change_pct":  0.18,  "name": "Berkshire Hathaway B"},
}

_RATIOS: dict[str, dict] = {
    "AAPL":  {"pe": 28.5,  "pb": 8.9,  "roe": 160.1, "debt_equity": 1.76, "dividend_yield": 0.55},
    "MSFT":  {"pe": 35.2,  "pb": 12.8, "roe":  43.1, "debt_equity": 0.32, "dividend_yield": 0.73},
    "GOOGL": {"pe": 25.1,  "pb":  6.4, "roe":  27.3, "debt_equity": 0.07, "dividend_yield": 0.00},
    "AMZN":  {"pe": 21.3,  "pb":  8.2, "roe":  20.5, "debt_equity": 0.58, "dividend_yield": 0.00},
    "TSLA":  {"pe": 65.3,  "pb": 12.3, "roe":  19.4, "debt_equity": 0.18, "dividend_yield": 0.00},
    "NVDA":  {"pe": 72.1,  "pb": 35.6, "roe":  91.5, "debt_equity": 0.41, "dividend_yield": 0.04},
    "META":  {"pe": 23.4,  "pb": 7.1,  "roe":  36.2, "debt_equity": 0.13, "dividend_yield": 0.43},
    "NFLX":  {"pe": 44.8,  "pb": 14.2, "roe":  29.3, "debt_equity": 1.45, "dividend_yield": 0.00},
    "JPM":   {"pe": 11.2,  "pb":  1.9, "roe":  17.1, "debt_equity": 1.23, "dividend_yield": 2.24},
    "BRK.B": {"pe": 22.1,  "pb":  1.5, "roe":   7.8, "debt_equity": 0.27, "dividend_yield": 0.00},
}


# ── Tool implementations ───────────────────────────────────────────────────────

@Registry.tool("get_stock_price")
class StockPriceTool(BaseTool):
    """Returns the current stock price and daily change for a ticker symbol."""

    name        = "get_stock_price"
    description = (
        "Get the current stock price and daily percentage change for a given ticker symbol. "
        "Supports: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, JPM, BRK.B"
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
                            "description": "Stock ticker symbol (e.g. 'AAPL', 'MSFT').",
                        },
                    },
                    "required": ["ticker"],
                },
            },
        }

    async def execute(self, ticker: str, **_: Any) -> str:
        ticker = ticker.upper().strip()
        data   = _PRICES.get(ticker)
        if not data:
            return (
                f"Error: ticker '{ticker}' not found. "
                f"Available: {', '.join(sorted(_PRICES))}"
            )
        direction = "+" if data["change_pct"] >= 0 else ""
        return (
            f"{data['name']} ({ticker})\n"
            f"  Current price : ${data['price']:.2f}\n"
            f"  Daily change  : {direction}{data['change_pct']:.2f}%"
        )


@Registry.tool("get_financial_ratios")
class FinancialRatiosTool(BaseTool):
    """Returns key financial ratios (P/E, P/B, ROE, debt/equity, dividend yield)."""

    name        = "get_financial_ratios"
    description = (
        "Get key financial ratios for a stock: P/E, P/B, ROE (%), debt-to-equity, "
        "and dividend yield (%). "
        "Supports: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, JPM, BRK.B"
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
        ticker = ticker.upper().strip()
        data   = _RATIOS.get(ticker)
        if not data:
            return (
                f"Error: ticker '{ticker}' not found. "
                f"Available: {', '.join(sorted(_RATIOS))}"
            )
        return (
            f"Financial ratios for {ticker}:\n"
            f"  P/E ratio        : {data['pe']:.1f}x\n"
            f"  P/B ratio        : {data['pb']:.1f}x\n"
            f"  ROE              : {data['roe']:.1f}%\n"
            f"  Debt / Equity    : {data['debt_equity']:.2f}\n"
            f"  Dividend yield   : {data['dividend_yield']:.2f}%"
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
            f"Investment analysis:",
            f"  Shares bought   : {shares:.4g}",
            f"  Cost basis      : ${cost:,.2f}",
            f"  Current value   : ${current_value:,.2f}",
            f"  Gain / Loss     : ${gain:+,.2f}",
            f"  Total return    : {pct_return:+.2f}%",
        ]

        if years_held and years_held > 0:
            annualized = ((current_value / cost) ** (1 / years_held) - 1) * 100
            lines.append(f"  Annualized return: {annualized:+.2f}% per year")

        return "\n".join(lines)


@Registry.tool("compare_companies")
class CompareCompaniesTool(BaseTool):
    """Side-by-side comparison of price and key ratios for multiple tickers."""

    name        = "compare_companies"
    description = (
        "Compare two or more companies side-by-side on current stock price, P/E, P/B, "
        "ROE, debt-to-equity, and dividend yield. "
        "Supports: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, JPM, BRK.B"
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
        results = []
        for raw in tickers:
            t = raw.upper().strip()
            price_data = _PRICES.get(t)
            ratio_data = _RATIOS.get(t)
            if not price_data or not ratio_data:
                results.append(f"  {t}: not found")
                continue
            results.append(
                f"  {t} ({price_data['name']}):\n"
                f"    Price     : ${price_data['price']:.2f} "
                f"({'+' if price_data['change_pct'] >= 0 else ''}{price_data['change_pct']:.2f}%)\n"
                f"    P/E       : {ratio_data['pe']:.1f}x\n"
                f"    P/B       : {ratio_data['pb']:.1f}x\n"
                f"    ROE       : {ratio_data['roe']:.1f}%\n"
                f"    D/E       : {ratio_data['debt_equity']:.2f}\n"
                f"    Div. Yield: {ratio_data['dividend_yield']:.2f}%"
            )

        return "Company comparison:\n" + "\n\n".join(results)
