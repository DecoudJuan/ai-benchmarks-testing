"""
FinanceDataset — live-generated finance Q&A items backed by Yahoo Finance.

Questions and expected answers are built from real market data fetched at
load time, so they always match what the agent tools will return.

Each question requires the agent to call at least one finance tool
(get_stock_price, get_financial_ratios, calculate_return, compare_companies).

Registered as: "finance_qa"
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import NamedTuple

import yfinance as yf

from labai.core.base import BaseDataset
from labai.core.registry import Registry
from labai.core.types import EvalItem

# Tickers used for generating questions
_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "BRK-B"]


class _TickerData(NamedTuple):
    ticker:       str
    name:         str
    price:        float
    change_pct:   float
    price_1y:     float | None   # price ~1 trading year ago
    price_2y:     float | None   # price ~2 trading years ago
    pe:           float | None
    forward_pe:   float | None
    pb:           float | None
    roe:          float | None   # percent
    debt_equity:  float | None   # ratio
    div_yield:    float | None   # percent
    market_cap:   float | None


def _fetch_ticker(ticker: str) -> _TickerData | None:
    """Fetch all data needed for question generation. Returns None on failure."""
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="2y")

        if hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])
        prev  = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        change_pct = ((price - prev) / prev) * 100 if prev else 0.0

        price_1y = float(hist["Close"].iloc[-252]) if len(hist) >= 252 else None
        price_2y = float(hist["Close"].iloc[-504]) if len(hist) >= 504 else None

        roe_raw = info.get("returnOnEquity")
        roe     = roe_raw * 100 if roe_raw is not None else None

        de_raw = info.get("debtToEquity")
        debt_equity = de_raw / 100 if de_raw is not None else None

        div_rate = info.get("dividendRate") or 0
        if div_rate and price:
            div_yield = (div_rate / price) * 100
        else:
            raw_dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0
            div_yield = raw_dy * 100 if raw_dy < 1 else raw_dy
        div_yield = div_yield or None

        return _TickerData(
            ticker      = ticker,
            name        = info.get("shortName") or info.get("longName") or ticker,
            price       = price,
            change_pct  = change_pct,
            price_1y    = price_1y,
            price_2y    = price_2y,
            pe          = info.get("trailingPE"),
            forward_pe  = info.get("forwardPE"),
            pb          = info.get("priceToBook"),
            roe         = roe,
            debt_equity = debt_equity,
            div_yield   = div_yield,
            market_cap  = info.get("marketCap"),
        )
    except Exception:
        return None


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"

def _pct(v: float, decimals: int = 2) -> str:
    return f"{v:+.{decimals}f}%"


def _build_items(data: dict[str, _TickerData]) -> list[EvalItem]:
    items: list[EvalItem] = []

    def add(id_: str, q: str, expected: str, cat: str, diff: str):
        items.append(EvalItem(
            id       = id_,
            input    = q,
            expected = expected,
            metadata = {"category": cat, "difficulty": diff},
        ))

    # ── Stock price — easy ─────────────────────────────────────────────────────

    for ticker in ["AAPL", "TSLA", "NVDA", "AMZN"]:
        d = data.get(ticker)
        if not d:
            continue
        direction = "up" if d.change_pct >= 0 else "down"
        add(
            f"price_{ticker.lower()}",
            f"What is the current stock price of {d.name} ({ticker})? "
            f"Is it up or down today?",
            f"{d.name} ({ticker}) is currently trading at ${_fmt(d.price)}, "
            f"{direction} {abs(d.change_pct):.2f}% today.",
            "stock_price", "easy",
        )

    # Which is more expensive per share
    aapl, msft = data.get("AAPL"), data.get("MSFT")
    if aapl and msft:
        more_exp = "MSFT" if msft.price > aapl.price else "AAPL"
        more_name = msft.name if msft.price > aapl.price else aapl.name
        add(
            "price_aapl_vs_msft",
            "Which stock has a higher share price right now — Apple (AAPL) or Microsoft (MSFT)?",
            f"{more_name} ({more_exp}) has the higher share price "
            f"(${_fmt(msft.price)} vs ${_fmt(aapl.price)}).",
            "stock_price", "easy",
        )

    # ── Valuation — easy/medium ────────────────────────────────────────────────

    for ticker in ["MSFT", "GOOGL", "NVDA"]:
        d = data.get(ticker)
        if not d or d.pe is None:
            continue
        add(
            f"pe_{ticker.lower()}",
            f"What is the current trailing P/E ratio of {d.name} ({ticker})?",
            f"{d.name}'s trailing P/E ratio is approximately {d.pe:.1f}x.",
            "valuation", "easy",
        )

    # Compare P/E
    aapl, amzn = data.get("AAPL"), data.get("AMZN")
    if aapl and amzn and aapl.pe and amzn.pe:
        cheaper = "AMZN" if amzn.pe < aapl.pe else "AAPL"
        cheaper_name = amzn.name if amzn.pe < aapl.pe else aapl.name
        add(
            "pe_aapl_vs_amzn",
            "Compare the P/E ratios of Apple (AAPL) and Amazon (AMZN). "
            "Which appears cheaper relative to earnings?",
            f"Amazon's P/E is {amzn.pe:.1f}x and Apple's is {aapl.pe:.1f}x. "
            f"{cheaper_name} ({cheaper}) appears cheaper relative to earnings.",
            "valuation", "medium",
        )

    # P/B ratio question
    tsla = data.get("TSLA")
    if tsla and tsla.pb:
        add(
            "pb_tsla",
            f"What is Tesla's (TSLA) Price-to-Book ratio, and what does it say about investor expectations?",
            f"Tesla's P/B ratio is approximately {tsla.pb:.1f}x, meaning investors value "
            f"the stock at {tsla.pb:.1f} times its book value, reflecting strong growth expectations.",
            "valuation", "medium",
        )

    # ── Returns — medium/hard ──────────────────────────────────────────────────

    # 1-year return for AAPL
    if aapl and aapl.price_1y:
        shares   = 100
        cost     = aapl.price_1y * shares
        value    = aapl.price * shares
        gain     = value - cost
        ret_pct  = (gain / cost) * 100
        add(
            "return_aapl_1y",
            f"I bought 100 shares of Apple (AAPL) one year ago at ${_fmt(aapl.price_1y)} per share. "
            f"What is my total return in dollars and as a percentage at the current price?",
            f"Cost basis: ${cost:,.2f}. Current value: ${value:,.2f}. "
            f"Gain/Loss: ${gain:+,.2f} ({ret_pct:+.2f}%).",
            "returns", "medium",
        )

    # 2-year annualized return for MSFT
    if msft and msft.price_2y:
        cost2    = msft.price_2y * 50
        value2   = msft.price * 50
        gain2    = value2 - cost2
        ret2     = (gain2 / cost2) * 100
        ann2     = ((msft.price / msft.price_2y) ** 0.5 - 1) * 100
        add(
            "return_msft_2y_annualized",
            f"I bought 50 shares of Microsoft (MSFT) two years ago at ${_fmt(msft.price_2y)} "
            f"per share. What is my total return and annualized return?",
            f"Cost basis: ${cost2:,.2f}. Current value: ${value2:,.2f}. "
            f"Total return: {ret2:+.2f}%. Annualized return: approximately {ann2:+.2f}% per year.",
            "returns", "hard",
        )

    # Investment value question
    nvda = data.get("NVDA")
    if nvda and nvda.price_1y:
        invest   = 5000.0
        shares_b = invest / nvda.price_1y
        value_b  = shares_b * nvda.price
        ret_b    = ((value_b - invest) / invest) * 100
        add(
            "return_nvda_investment",
            f"If I invested $5,000 in NVIDIA (NVDA) one year ago when the price was "
            f"${_fmt(nvda.price_1y)}, how many shares did I buy and what is my investment "
            f"worth today?",
            f"Shares purchased: {shares_b:.2f}. Current value: ${value_b:,.2f}. "
            f"Return: {ret_b:+.2f}%.",
            "returns", "medium",
        )

    # ── Fundamentals — easy/medium ─────────────────────────────────────────────

    for ticker in ["JPM", "META"]:
        d = data.get(ticker)
        if not d or d.roe is None:
            continue
        add(
            f"roe_{ticker.lower()}",
            f"What is the Return on Equity (ROE) of {d.name} ({ticker})?",
            f"{d.name}'s ROE is approximately {d.roe:.1f}%.",
            "fundamentals", "easy",
        )

    # Dividend yield
    jpm = data.get("JPM")
    if jpm and jpm.div_yield:
        add(
            "div_yield_jpm",
            "What is JPMorgan Chase's (JPM) dividend yield?",
            f"JPMorgan Chase's dividend yield is approximately {jpm.div_yield:.2f}%.",
            "fundamentals", "easy",
        )

    # Debt/equity
    meta = data.get("META")
    if meta and meta.debt_equity is not None:
        add(
            "de_meta",
            "What is Meta Platforms' (META) debt-to-equity ratio? What does it indicate?",
            f"Meta's debt-to-equity ratio is approximately {meta.debt_equity:.2f}, "
            f"indicating {'low' if meta.debt_equity < 0.5 else 'moderate'} financial leverage.",
            "fundamentals", "medium",
        )

    # ── Comparison — medium/hard ───────────────────────────────────────────────

    # ROE comparison 3 companies
    triplet = [data.get(t) for t in ["AAPL", "MSFT", "GOOGL"]]
    triplet = [d for d in triplet if d and d.roe is not None]
    if len(triplet) >= 2:
        best = max(triplet, key=lambda d: d.roe)
        roe_strs = ", ".join(f"{d.ticker}: {d.roe:.1f}%" for d in sorted(triplet, key=lambda d: d.roe, reverse=True))
        add(
            "roe_aapl_msft_googl",
            "Which company has a higher Return on Equity (ROE): Apple (AAPL), Microsoft (MSFT), "
            "or Google (GOOGL)? Rank them.",
            f"{best.name} ({best.ticker}) leads in ROE. Ranking: {roe_strs}.",
            "comparison", "medium",
        )

    # Full comparison AAPL vs MSFT
    if aapl and msft and aapl.pe and msft.pe and aapl.pb and msft.pb:
        cheaper_pe   = "AAPL" if aapl.pe < msft.pe else "MSFT"
        cheaper_pb   = "AAPL" if aapl.pb < msft.pb else "MSFT"
        add(
            "compare_aapl_msft_valuation",
            "Compare Apple (AAPL) and Microsoft (MSFT) on P/E ratio, P/B ratio, and current price. "
            "Which looks more attractive from a value-investing perspective?",
            f"AAPL: price ${_fmt(aapl.price)}, P/E {aapl.pe:.1f}x, P/B {aapl.pb:.1f}x. "
            f"MSFT: price ${_fmt(msft.price)}, P/E {msft.pe:.1f}x, P/B {msft.pb:.1f}x. "
            f"{cheaper_pe} has the lower P/E and {cheaper_pb} has the lower P/B, "
            f"making {'AAPL' if cheaper_pe == cheaper_pb == 'AAPL' else 'MSFT' if cheaper_pe == cheaper_pb == 'MSFT' else 'neither clearly'} "
            f"the better value on both metrics.",
            "comparison", "hard",
        )

    # D/E comparison
    t_data = [data.get(t) for t in ["TSLA", "AMZN"]]
    t_data = [d for d in t_data if d and d.debt_equity is not None]
    if len(t_data) == 2:
        more_lev = max(t_data, key=lambda d: d.debt_equity)
        add(
            "compare_tsla_amzn_leverage",
            "Compare the debt-to-equity ratios of Tesla (TSLA) and Amazon (AMZN). "
            "Which carries more financial leverage?",
            f"Tesla D/E: {t_data[0].debt_equity:.2f}, Amazon D/E: {t_data[1].debt_equity:.2f}. "
            f"{more_lev.name} ({more_lev.ticker}) carries more leverage.",
            "comparison", "hard",
        )

    # NVDA vs META vs GOOGL growth comparison
    growth = [data.get(t) for t in ["NVDA", "META", "GOOGL"]]
    growth = [d for d in growth if d and d.pe and d.roe]
    if len(growth) >= 2:
        best_roe = max(growth, key=lambda d: d.roe)
        summary = "; ".join(f"{d.ticker} P/E {d.pe:.1f}x ROE {d.roe:.1f}%" for d in growth)
        add(
            "compare_nvda_meta_googl_growth",
            "Compare NVIDIA (NVDA), Meta (META), and Google (GOOGL) on P/E ratio and ROE. "
            "Which appears most attractive for a growth investor?",
            f"{summary}. {best_roe.name} ({best_roe.ticker}) has the highest ROE at {best_roe.roe:.1f}%.",
            "comparison", "hard",
        )

    # ── Multi-step — hard ──────────────────────────────────────────────────────

    # Implied EPS from P/E and price
    if tsla and tsla.pe:
        eps = tsla.price / tsla.pe
        add(
            "eps_tsla_implied",
            f"Tesla (TSLA) is currently trading at ${_fmt(tsla.price)} with a trailing P/E "
            f"of {tsla.pe:.1f}. What is the implied Earnings Per Share (EPS)?",
            f"EPS = Price / P/E = ${_fmt(tsla.price)} / {tsla.pe:.1f} = approximately ${eps:.2f} per share.",
            "multi_step", "hard",
        )

    # Target price for 15% return
    if msft:
        target = msft.price * 1.15
        add(
            "target_price_msft",
            f"If I want to achieve a 15% return on my Microsoft (MSFT) investment "
            f"starting at the current price, what target price do I need?",
            f"Current MSFT price: ${_fmt(msft.price)}. "
            f"Target = ${_fmt(msft.price)} × 1.15 = ${_fmt(target)}.",
            "multi_step", "hard",
        )

    # Shares buyable + upside scenario
    if aapl:
        invest_amt  = 10_000.0
        shares_buy  = invest_amt / aapl.price
        value_up    = shares_buy * (aapl.price * 1.20)
        add(
            "aapl_10k_scenario",
            f"With $10,000 to invest in Apple (AAPL) at the current price, how many shares "
            f"can I buy, and what would my portfolio be worth if the stock rises 20%?",
            f"At ${_fmt(aapl.price)}/share, $10,000 buys {shares_buy:.2f} shares. "
            f"At +20% (${_fmt(aapl.price * 1.20)}), portfolio value = ${value_up:,.2f}.",
            "multi_step", "hard",
        )

    # Value vs growth: pick best between NVDA (growth) and JPM (value)
    if nvda and jpm and nvda.pe and jpm.pe:
        add(
            "value_vs_growth_nvda_jpm",
            "I'm deciding between NVIDIA (NVDA) as a growth play and JPMorgan (JPM) as a value play. "
            "Using current P/E ratio and stock price, make the case for each and conclude which "
            "is more appropriate for a risk-averse investor.",
            f"NVDA: ${_fmt(nvda.price)}, P/E {nvda.pe:.1f}x (high growth, expensive valuation). "
            f"JPM: ${_fmt(jpm.price)}, P/E {jpm.pe:.1f}x (value, lower multiple). "
            f"A risk-averse investor would prefer JPM for its lower P/E and dividend income.",
            "multi_step", "hard",
        )

    return items


@Registry.dataset("finance_qa")
class FinanceDataset(BaseDataset):
    """
    Finance Q&A dataset — questions and expected answers generated from live
    Yahoo Finance data at load time.

    Categories: stock_price, valuation, returns, comparison, fundamentals, multi_step
    Difficulties: easy | medium | hard

    Data is fetched once per load() call and cached for the session.
    """

    name     = "finance_qa"
    domain   = "finance"
    language = "en"

    _cache: dict[str, _TickerData] | None = None

    def load(
        self,
        subjects:  list[str] | None = None,
        n_samples: int | None       = None,
        seed:      int              = 42,
    ) -> list[EvalItem]:
        """
        Load finance evaluation items with live expected answers.

        Args:
            subjects:  Optional category keywords to filter
                       (e.g. ['valuation', 'returns']).
            n_samples: Max items. Sampled evenly across categories when limited.
            seed:      Random seed for reproducibility.
        """
        if FinanceDataset._cache is None:
            FinanceDataset._cache = self._fetch_data()

        data = FinanceDataset._cache

        if not data:
            raise RuntimeError(
                "Failed to fetch market data from Yahoo Finance. "
                "Check your internet connection."
            )

        all_items = _build_items(data)

        if subjects:
            keywords = [s.lower() for s in subjects]
            all_items = [
                item for item in all_items
                if any(kw in item.metadata.get("category", "").lower() for kw in keywords)
            ]

        if not all_items:
            return []

        if n_samples is not None and n_samples < len(all_items):
            all_items = _sample_evenly(all_items, n_samples, key="category", seed=seed)

        return all_items

    @staticmethod
    def _fetch_data() -> dict[str, _TickerData]:
        """Fetch live market data for all tickers. Silently skips failures."""
        print("  [finance_qa] Fetching live market data from Yahoo Finance...")
        result = {}
        for ticker in _TICKERS:
            d = _fetch_ticker(ticker)
            if d:
                result[ticker] = d
                print(f"    {ticker}: ${d.price:,.2f}")
            else:
                print(f"    {ticker}: failed — skipping")
        print(f"  [finance_qa] Loaded data for {len(result)}/{len(_TICKERS)} tickers\n")
        return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sample_evenly(
    items: list[EvalItem],
    n:     int,
    key:   str = "category",
    seed:  int = 42,
) -> list[EvalItem]:
    """Sample n items evenly across metadata[key] groups."""
    rng    = random.Random(seed)
    groups: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        groups[item.metadata.get(key, "unknown")].append(item)

    per_group = math.ceil(n / max(len(groups), 1))
    sampled: list[EvalItem] = []
    for grp in groups.values():
        rng.shuffle(grp)
        sampled.extend(grp[:per_group])

    rng.shuffle(sampled)
    return sampled[:n]
