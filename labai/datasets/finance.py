"""
FinanceDataset — curated finance Q&A items requiring tool use.

Each question requires the agent to call at least one finance tool
(stock price, financial ratios, return calculation, or company comparison)
to arrive at the correct answer.

Registered as: "finance_qa"
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from labai.core.base import BaseDataset
from labai.core.registry import Registry
from labai.core.types import EvalItem

# ── Mock question bank ─────────────────────────────────────────────────────────
# Each entry: (id, input, expected_answer, category, difficulty)
# expected is a short string the judge uses as ground truth.

_QUESTIONS: list[tuple[str, str, str, str, str]] = [

    # ── Valuation ─────────────────────────────────────────────────────────────
    ("fin_001",
     "What is the current P/E ratio of Apple (AAPL)?",
     "Apple's P/E ratio is approximately 28.5",
     "valuation", "easy"),

    ("fin_002",
     "What is the current P/E ratio of Microsoft (MSFT)?",
     "Microsoft's P/E ratio is approximately 35.2",
     "valuation", "easy"),

    ("fin_003",
     "Compare the P/E ratios of Apple (AAPL) and Google (GOOGL). Which is more expensive relative to earnings?",
     "Google (GOOGL) has a higher P/E ratio (~25.1) than Apple (~28.5); however both are in a similar range. "
     "The agent should compare the two values correctly.",
     "valuation", "medium"),

    ("fin_004",
     "What is Tesla's (TSLA) Price-to-Book ratio, and what does it indicate about investor sentiment?",
     "Tesla's P/B ratio is approximately 12.3, indicating investors price the stock at 12x book value, "
     "reflecting high growth expectations.",
     "valuation", "medium"),

    # ── Stock prices ──────────────────────────────────────────────────────────
    ("fin_005",
     "What is the current stock price of Amazon (AMZN)?",
     "Amazon's current stock price is approximately $178.25",
     "stock_price", "easy"),

    ("fin_006",
     "What is the current stock price of Tesla (TSLA)?",
     "Tesla's current stock price is approximately $242.10",
     "stock_price", "easy"),

    ("fin_007",
     "Which is more expensive per share right now — Apple (AAPL) or Microsoft (MSFT)?",
     "Microsoft (MSFT) is more expensive per share at approximately $378.85 vs Apple at $178.72",
     "stock_price", "easy"),

    # ── Returns ───────────────────────────────────────────────────────────────
    ("fin_008",
     "If I bought 100 shares of Apple (AAPL) at $150 per share and the current price is $178.72, "
     "what is my total return in dollars and as a percentage?",
     "Total return: $2,872 (19.15%)",
     "returns", "medium"),

    ("fin_009",
     "If I invested $10,000 in Amazon (AMZN) when the price was $120 per share, how many shares did "
     "I buy and what is my investment worth at the current price of $178.25?",
     "Bought 83 shares (rounded down). Current value: approximately $14,794.75. Return: ~47.9%",
     "returns", "medium"),

    ("fin_010",
     "Calculate the annualized return for an investment in Microsoft (MSFT) bought 2 years ago at $280 "
     "and now worth $378.85.",
     "Total return: 35.3%. Annualized: approximately 16.3% per year.",
     "returns", "hard"),

    # ── Comparison ────────────────────────────────────────────────────────────
    ("fin_011",
     "Compare Apple (AAPL) and Microsoft (MSFT) on P/E ratio, P/B ratio, and current price. "
     "Which company looks more attractive from a value investing perspective?",
     "Microsoft has a higher P/E (35.2 vs 28.5) and P/B (12.8 vs 8.9), making Apple relatively "
     "cheaper on both multiples. From a pure value perspective, Apple appears more attractively priced.",
     "comparison", "hard"),

    ("fin_012",
     "Compare the financial ratios of Tesla (TSLA) and Amazon (AMZN). Which has higher debt-to-equity?",
     "Tesla has a debt-to-equity of 0.18 vs Amazon at 0.58. Amazon carries significantly more leverage.",
     "comparison", "hard"),

    ("fin_013",
     "Which company has a better return on equity (ROE): Apple (AAPL), Microsoft (MSFT), or Google (GOOGL)?",
     "Apple has the highest ROE at 160.1%, followed by Microsoft at 43.1%, then Google at 27.3%.",
     "comparison", "medium"),

    # ── Ratios / fundamentals ─────────────────────────────────────────────────
    ("fin_014",
     "What is Apple's (AAPL) debt-to-equity ratio and what does it suggest about their financial leverage?",
     "Apple's debt-to-equity is approximately 1.76, meaning they use more debt than equity to finance assets, "
     "but this is manageable given their strong cash flows.",
     "fundamentals", "medium"),

    ("fin_015",
     "What is the dividend yield for Microsoft (MSFT)?",
     "Microsoft's dividend yield is approximately 0.73%.",
     "fundamentals", "easy"),

    ("fin_016",
     "What is Google's (GOOGL) current revenue and net income margin?",
     "Google's revenue is approximately $307.4 billion with a net income margin of about 24%.",
     "fundamentals", "medium"),

    # ── Multi-step reasoning ──────────────────────────────────────────────────
    ("fin_017",
     "I have $50,000 to invest. Should I buy Apple (AAPL) or Amazon (AMZN) based on current valuations? "
     "Use P/E ratio and recent price as your main factors.",
     "Based on P/E ratios, Amazon (~21.3) appears cheaper relative to earnings than Apple (~28.5). "
     "An agent should compare both metrics and recommend Amazon as the better value play, though Apple "
     "has stronger dividend history.",
     "multi_step", "hard"),

    ("fin_018",
     "Tesla's stock is at $242.10. If its P/E ratio is 65.3, what is the implied earnings per share (EPS)?",
     "EPS = Price / P/E = 242.10 / 65.3 = approximately $3.71 per share.",
     "multi_step", "hard"),

    ("fin_019",
     "If I want to achieve a 20% annual return on a $25,000 investment in Microsoft (MSFT), "
     "what target price do I need to reach in one year?",
     "Target price = current price * 1.20 = $378.85 * 1.20 = approximately $454.62.",
     "multi_step", "hard"),

    ("fin_020",
     "What percentage of Apple's (AAPL) market cap would represent a $10,000 investment given "
     "the current stock price and assuming 15.2 billion shares outstanding?",
     "Market cap = $178.72 * 15.2B = ~$2.716 trillion. "
     "$10,000 / $2,716,544,000,000 = approximately 0.000000368% of market cap.",
     "multi_step", "hard"),
]


@Registry.dataset("finance_qa")
class FinanceDataset(BaseDataset):
    """
    Finance Q&A dataset requiring tool use (stock prices, ratios, returns).

    20 curated questions across 5 categories:
        - valuation    (P/E, P/B)
        - stock_price
        - returns      (total and annualized)
        - comparison   (multi-metric)
        - fundamentals (ratios, dividends)
        - multi_step   (derived calculations)

    Difficulties: easy | medium | hard
    """

    name     = "finance_qa"
    domain   = "finance"
    language = "en"

    def load(
        self,
        subjects:  list[str] | None = None,
        n_samples: int | None       = None,
        seed:      int              = 42,
    ) -> list[EvalItem]:
        """
        Load finance evaluation items.

        Args:
            subjects:  Optional list of category keywords to filter
                       (e.g. ['valuation', 'returns']).
            n_samples: Max items. Sampled evenly across categories when limited.
            seed:      Random seed for reproducibility.

        Returns:
            List of EvalItem.
        """
        rows = _QUESTIONS

        # Filter by subject/category if requested
        if subjects:
            keywords = [s.lower() for s in subjects]
            rows = [
                r for r in rows
                if any(kw in r[3].lower() for kw in keywords)
            ]

        if not rows:
            return []

        # Convert to EvalItem first
        items = [
            EvalItem(
                id       = row[0],
                input    = row[1],
                expected = row[2],
                metadata = {"category": row[3], "difficulty": row[4]},
            )
            for row in rows
        ]

        # Even sampling across categories
        if n_samples is not None and n_samples < len(items):
            items = _sample_evenly(items, n_samples, key="category", seed=seed)

        return items


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sample_evenly(
    items:    list[EvalItem],
    n:        int,
    key:      str = "category",
    seed:     int = 42,
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
