"""
Keep It Ross — Runner Hunter (paper research scanner)
----------------------------------------------------
Scans a broad universe of active, tradable US stocks using Alpaca market data.
This version is ALERT/RESEARCH ONLY: it does not submit orders.

Signals are heuristic, not predictive guarantees. Short interest is not included
because reliable short-interest/float data usually requires a separate data source.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

APP_NAME = "Keep It Ross — Runner Hunter"
OUT_DIR = Path(os.getenv("RUNNER_HUNTER_OUT", "output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "scanner.log"
SIGNALS_FILE = OUT_DIR / "signals.csv"

# Tweak through environment variables if needed.
MIN_PRICE = float(os.getenv("MIN_PRICE", "2.00"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "50.00"))
MIN_AVG_DOLLAR_VOLUME = float(os.getenv("MIN_AVG_DOLLAR_VOLUME", "2000000"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))  # 0 means scan every eligible symbol
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "80"))  # A/A+ only
FEED_NAME = os.getenv("ALPACA_DATA_FEED", "iex").lower()
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "150"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(APP_NAME)


@dataclass
class Candidate:
    symbol: str
    score: int
    setup: str
    close: float
    change_1d_pct: float
    avg_dollar_volume_20d: float
    volume_ratio: float
    range_position: float
    base_contraction: float
    higher_lows: bool
    near_20d_high: bool
    relative_strength_20d_pct: float
    entry_reference: float
    stop_reference: float
    target_reference: float
    notes: str


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def make_clients():
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY. "
            "Add your Alpaca PAPER account API keys as environment variables."
        )
    trading = TradingClient(key, secret, paper=True)
    data = StockHistoricalDataClient(key, secret)
    return trading, data


def get_universe(trading: TradingClient) -> list[str]:
    """Get active, tradable US stocks from the connected Alpaca account's asset list."""
    req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = trading.get_all_assets(req)
    symbols = []
    for asset in assets:
        symbol = getattr(asset, "symbol", None)
        if not symbol or not getattr(asset, "tradable", False):
            continue
        exchange = str(getattr(asset, "exchange", "") or "").upper()
        if exchange not in {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}:
            continue
        # Skip warrants/units/rights and other symbols that commonly create noisy data.
        if any(ch in symbol for ch in [".", "/", "$", "^"]):
            continue
        if len(symbol) > 5:
            continue
        symbols.append(symbol)
    symbols = sorted(set(symbols))
    # Default is the full eligible Alpaca asset universe. Set MAX_SYMBOLS to a
    # positive integer only for a smaller diagnostic run; capped runs are not market-wide.
    if MAX_SYMBOLS > 0 and len(symbols) > MAX_SYMBOLS:
        stride = len(symbols) / MAX_SYMBOLS
        symbols = [symbols[int(i * stride)] for i in range(MAX_SYMBOLS)]
        log.warning("CAPPED TEST RUN: scanning %d sampled symbols, not the whole eligible universe", len(symbols))
    log.info("Universe prepared: %d active tradable symbols (cap=%s)", len(symbols), MAX_SYMBOLS or "all")
    return symbols


def get_feed():
    return DataFeed.SIP if FEED_NAME == "sip" else DataFeed.IEX


def get_bars(data_client: StockHistoricalDataClient, symbols: list[str]) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=get_feed(),
        limit=10000,
    )
    response = data_client.get_stock_bars(request)
    df = response.df
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()
    # Normalize field names from Alpaca's MultiIndex DataFrame.
    df.columns = [str(c).lower() for c in df.columns]
    if "symbol" not in df.columns:
        return pd.DataFrame()
    return df.sort_values(["symbol", "timestamp"])


def score_symbol(symbol: str, frame: pd.DataFrame, spy_returns: dict[int, float]) -> Candidate | None:
    frame = frame.dropna(subset=["close", "volume"]).copy()
    if len(frame) < 60:
        return None
    frame = frame.sort_values("timestamp")
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    last = float(close.iloc[-1])
    if not (MIN_PRICE <= last <= MAX_PRICE):
        return None

    dollar_vol = close * volume
    adv20 = float(dollar_vol.tail(20).mean())
    if adv20 < MIN_AVG_DOLLAR_VOLUME:
        return None

    # Do not use today's partial bar for final end-of-day results when market is open.
    # Intraday partial data can still create provisional signals; rerun after close for confirmation.
    vol20 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
    volume_ratio = float(volume.iloc[-1] / vol20) if vol20 > 0 else 0.0

    ret1 = (last / float(close.iloc[-2]) - 1) * 100 if len(close) >= 2 else 0.0
    ret20 = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
    spy20 = spy_returns.get(20, 0.0)
    relative_strength = ret20 - spy20

    # Contraction: compare average high-low range in recent 10 sessions with prior 10.
    ranges = ((high - low) / close.replace(0, pd.NA)).dropna()
    contraction = 1.0
    if len(ranges) >= 25:
        old = float(ranges.iloc[-25:-15].mean())
        recent = float(ranges.iloc[-10:].mean())
        contraction = recent / old if old > 0 else 1.0

    # Higher lows: 3 latest weekly-ish low points should trend upward.
    weekly_lows = [
        float(low.iloc[-offset-5:-offset].min())
        for offset in (25, 20, 15, 10, 5)
        if len(low) >= offset + 5
    ]
    higher_lows = len(weekly_lows) >= 4 and all(
        weekly_lows[i] <= weekly_lows[i + 1] for i in range(len(weekly_lows) - 1)
    )

    high20 = float(high.tail(20).max())
    near_high_pct = (last / high20 - 1) * 100 if high20 else -99
    near_20d_high = near_high_pct >= -4.0
    range_low = float(low.tail(20).min())
    range_position = (last - range_low) / (high20 - range_low) if high20 > range_low else 0.0

    # 0–100 scoring. This ranks watch candidates; it does not prove a breakout.
    score = 0
    notes = []
    if contraction <= 0.75:
        score += 20; notes.append("range contraction")
    elif contraction <= 0.90:
        score += 12; notes.append("some contraction")
    if volume_ratio <= 0.75:
        score += 10; notes.append("volume dry-up")
    if higher_lows:
        score += 15; notes.append("higher lows")
    if near_20d_high:
        score += 15; notes.append("near 20-day high")
    if range_position >= 0.75:
        score += 10; notes.append("upper-range pressure")
    if relative_strength >= 5:
        score += 15; notes.append("relative strength")
    elif relative_strength >= 0:
        score += 8; notes.append("holding vs SPY")
    if volume_ratio >= 1.5 and ret1 > 0:
        score += 15; notes.append("ignition volume")
    elif volume_ratio >= 1.2 and ret1 > 0:
        score += 8; notes.append("volume expansion")
    if 0 <= ret1 <= 5:
        score += 5; notes.append("orderly daily move")
    if ret1 > 12:
        score -= 15; notes.append("already extended")
    if last < 5:
        score -= 5; notes.append("low-priced / higher risk")

    if score < MIN_SCORE:
        return None

    ignition = volume_ratio >= 1.5 and ret1 > 0 and near_20d_high
    setup = "IGNITION" if ignition else "COIL / PRE-IGNITION"
    entry = last
    stop = round(entry * 0.98, 4)
    target = round(entry * 1.10, 4)
    return Candidate(
        symbol=symbol, score=max(0, min(100, score)), setup=setup, close=round(last, 4),
        change_1d_pct=round(ret1, 2), avg_dollar_volume_20d=round(adv20, 0),
        volume_ratio=round(volume_ratio, 2), range_position=round(range_position, 3),
        base_contraction=round(contraction, 3), higher_lows=bool(higher_lows),
        near_20d_high=bool(near_20d_high), relative_strength_20d_pct=round(relative_strength, 2),
        entry_reference=round(entry, 4), stop_reference=stop, target_reference=target,
        notes=", ".join(notes),
    )


def main():
    trading, data = make_clients()
    symbols = get_universe(trading)
    if "SPY" not in symbols:
        symbols.append("SPY")

    all_frames = []
    for batch_no, batch in enumerate(chunks(symbols, BATCH_SIZE), start=1):
        try:
            bars = get_bars(data, batch)
            if not bars.empty:
                all_frames.append(bars)
            log.info("Fetched batch %d: %d symbols", batch_no, len(batch))
        except Exception as exc:
            log.warning("Batch %d failed: %s", batch_no, exc)
        # Keep request pace polite; exact rate limits depend on account/data plan.
        time.sleep(0.25)

    if not all_frames:
        raise RuntimeError("No bars returned. Check Alpaca keys, market-data permissions, and feed setting.")
    market = pd.concat(all_frames, ignore_index=True)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    grouped = {sym: grp for sym, grp in market.groupby("symbol")}

    spy_returns = {}
    if "SPY" in grouped and len(grouped["SPY"]) >= 21:
        spy_close = grouped["SPY"].sort_values("timestamp")["close"].astype(float)
        spy_returns[20] = (float(spy_close.iloc[-1]) / float(spy_close.iloc[-21]) - 1) * 100

    candidates = []
    for symbol, frame in grouped.items():
        if symbol == "SPY":
            continue
        try:
            candidate = score_symbol(symbol, frame, spy_returns)
            if candidate:
                candidates.append(candidate)
        except Exception as exc:
            log.debug("Skipping %s: %s", symbol, exc)

    candidates.sort(key=lambda x: (x.score, x.volume_ratio, x.relative_strength_20d_pct), reverse=True)
    stamp = datetime.now(timezone.utc).isoformat()
    rows = [asdict(c) | {"scan_time_utc": stamp} for c in candidates]
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=list(Candidate.__annotations__.keys()) + ["scan_time_utc"])
    out.to_csv(SIGNALS_FILE, index=False)

    log.info("Scan complete. A/A+ candidates: %d. Output: %s", len(candidates), SIGNALS_FILE)
    print("\n" + "=" * 72)
    print(APP_NAME)
    print(f"Scan time UTC: {stamp}")
    print(f"Universe cap: {MAX_SYMBOLS} | Data feed: {FEED_NAME.upper()}")
    print(f"A/A+ candidates (score >= {MIN_SCORE}): {len(candidates)}")
    print("=" * 72)
    if not candidates:
        print("NO TRADE — no candidates met the strict score threshold.")
    else:
        for c in candidates[:25]:
            print(
                f"{c.symbol:6} score={c.score:3} {c.setup:16} close=${c.close:<8} "
                f"vol={c.volume_ratio:>4}x  RS20={c.relative_strength_20d_pct:>6}%  "
                f"stop=${c.stop_reference} target=${c.target_reference}"
            )
            print(f"       {c.notes}")
    print(f"\nFull results saved to: {SIGNALS_FILE}")
    print("No orders were placed.")


if __name__ == "__main__":
    main()
