"""
APEX PROP — GhoulHQ Prop Firm Division
=======================================
Multi-agent trading system targeting +10% per 30-day challenge period.

Agents (orchestrated via state flags in Gist):
  Leader     — runs this main loop, delegates via state["agent_directives"]
  Supervisor — monitors per-asset health, can halt/tweak per-symbol params
  Backtester — writes optimized params to state["optimized_params"] each run

Tokens  : HYPEUSDT, BTCUSDT, VIRTUALUSDT, ZECUSDT
Timeframe: 5-minute (env: TIMEFRAME=5)
Goal    : +10% / 30 days, max DD 10%, daily DD 5%

Strategy (EMA-200 crossover + RSI filter + ATR-based SL/TP):
  Entry : EMA-200 cross + RSI not overbought/oversold
  SL    : 1× ATR below/above entry (dynamic, tighter than fixed %)
  TP1   : 2× ATR → close 50%
  TP2   : 4× ATR → close 30% of remaining
  TP3   : 6× ATR → close all remaining
  Trail : 1.5× ATR from swing high/low after TP1
  Size  : Kelly-adjusted, max 20% equity per position
"""

import os, json, time, hmac, hashlib, logging, math, requests, numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("apex_prop.log"),
    ]
)
log = logging.getLogger("apex_prop")

# ─── Config ───────────────────────────────────────────────────────────────────
API_KEY       = os.environ.get("BYBIT_API_KEY", "")
API_SECRET    = os.environ.get("API_SECRET", "")
TESTNET       = os.environ.get("BYBIT_TESTNET", "true").lower() == "true"
GIST_TOKEN    = os.environ.get("GIST_TOKEN", "")
GIST_ID       = os.environ.get("GIST_ID", "")
TIMEFRAME     = os.environ.get("TIMEFRAME", "5")           # 5-min candles
TOKENS        = os.environ.get("TOKENS", "HYPEUSDT,BTCUSDT,VIRTUALUSDT,ZECUSDT").split(",")
CATEGORY      = "linear"
DRY_RUN       = os.environ.get("DRY_RUN", "false").lower() == "true"

BASE_URL = "https://api-testnet.bybit.com" if TESTNET else "https://api.bybit.com"

# ── Strategy defaults (can be overridden by Backtester via optimized_params) ──
EMA_LENGTH    = int(os.environ.get("EMA_LENGTH",    "200"))
RSI_LENGTH    = int(os.environ.get("RSI_LENGTH",    "14"))
RSI_OB        = float(os.environ.get("RSI_OB",      "70"))   # overbought
RSI_OS        = float(os.environ.get("RSI_OS",      "30"))   # oversold
ATR_LENGTH    = int(os.environ.get("ATR_LENGTH",    "14"))
ATR_SL_MULT   = float(os.environ.get("ATR_SL_MULT", "1.0"))
ATR_TP1_MULT  = float(os.environ.get("ATR_TP1_MULT","2.0"))
ATR_TP2_MULT  = float(os.environ.get("ATR_TP2_MULT","4.0"))
ATR_TP3_MULT  = float(os.environ.get("ATR_TP3_MULT","6.0"))
ATR_TRAIL     = float(os.environ.get("ATR_TRAIL",   "1.5"))
TP1_CLOSE_PCT = float(os.environ.get("TP1_CLOSE_PCT","50")) / 100
TP2_CLOSE_PCT = float(os.environ.get("TP2_CLOSE_PCT","30")) / 100
POS_SIZE_PCT  = float(os.environ.get("POSITION_SIZE_PCT","20")) / 100
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "4"))    # one per token
MAX_PYRAMID   = int(os.environ.get("MAX_PYRAMID",   "3"))
LEVERAGE      = int(os.environ.get("LEVERAGE",      "1"))
MAX_DD_PCT    = float(os.environ.get("MAX_DD_PCT",  "10")) / 100
DAILY_DD_PCT  = float(os.environ.get("DAILY_DD_PCT","5"))  / 100

# Monthly target for prop challenge
MONTHLY_TARGET_PCT = float(os.environ.get("MONTHLY_TARGET_PCT", "10")) / 100


# ─── Bybit Client ─────────────────────────────────────────────────────────────
class BybitClient:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key     = api_key
        self.api_secret  = api_secret
        self.base_url    = base_url
        self.recv_window = "5000"

    def _sign(self, params_str):
        ts = str(int(time.time() * 1000))
        payload = ts + self.api_key + self.recv_window + params_str
        sig = hmac.new(self.api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return sig, ts

    def _headers(self, params_str):
        sig, ts = self._sign(params_str)
        return {
            "X-BAPI-API-KEY":      self.api_key,
            "X-BAPI-SIGN":         sig,
            "X-BAPI-TIMESTAMP":    ts,
            "X-BAPI-RECV-WINDOW":  self.recv_window,
            "Content-Type":        "application/json",
        }

    def get(self, path, params=None):
        params = params or {}
        qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        r = requests.get(f"{self.base_url}{path}", params=params,
                         headers=self._headers(qs), timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("retCode") != 0:
            raise RuntimeError(f"Bybit {d['retCode']}: {d.get('retMsg')}")
        return d["result"]

    def post(self, path, body):
        if DRY_RUN:
            log.info(f"[DRY RUN] POST {path} {body}")
            return {"orderId": "DRY_RUN"}
        s = json.dumps(body)
        r = requests.post(f"{self.base_url}{path}", data=s,
                          headers=self._headers(s), timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("retCode") != 0:
            raise RuntimeError(f"Bybit {d['retCode']}: {d.get('retMsg')}")
        return d["result"]

    def get_klines(self, symbol, interval, limit=220):
        r = self.get("/v5/market/kline", {
            "category": CATEGORY, "symbol": symbol,
            "interval": interval, "limit": limit,
        })
        return list(reversed(r["list"]))  # oldest first

    def get_ticker(self, symbol):
        return self.get("/v5/market/tickers", {
            "category": CATEGORY, "symbol": symbol,
        })["list"][0]

    def get_wallet_balance(self):
        r = self.get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        coins = r["list"][0]["coin"]
        usdt = next((c for c in coins if c["coin"] == "USDT"), {})
        return {
            "equity":         float(usdt.get("equity", 0)),
            "available":      float(usdt.get("availableToWithdraw", 0)),
            "unrealised_pnl": float(usdt.get("unrealisedPnl", 0)),
            "wallet_balance": float(usdt.get("walletBalance", 0)),
        }

    def get_positions(self, symbol=""):
        p = {"category": CATEGORY, "settleCoin": "USDT"}
        if symbol: p["symbol"] = symbol
        r = self.get("/v5/position/list", p)
        return [x for x in r["list"] if float(x.get("size", 0)) > 0]

    def get_instrument_info(self, symbol):
        return self.get("/v5/market/instruments-info", {
            "category": CATEGORY, "symbol": symbol,
        })["list"][0]

    def set_leverage(self, symbol, leverage):
        try:
            self.post("/v5/position/set-leverage", {
                "category": CATEGORY, "symbol": symbol,
                "buyLeverage": str(leverage), "sellLeverage": str(leverage),
            })
        except Exception as e:
            log.warning(f"Leverage set failed {symbol}: {e}")

    def place_market_order(self, symbol, side, qty):
        return self.post("/v5/order/create", {
            "category": CATEGORY, "symbol": symbol,
            "side": side, "orderType": "Market",
            "qty": str(round(qty, 6)), "timeInForce": "IOC",
        })

    def close_position_pct(self, symbol, side, position_qty, close_pct):
        qty = round(position_qty * close_pct, 6)
        close_side = "Sell" if side == "Buy" else "Buy"
        return self.post("/v5/order/create", {
            "category": CATEGORY, "symbol": symbol,
            "side": close_side, "orderType": "Market",
            "qty": str(qty), "timeInForce": "IOC", "reduceOnly": True,
        })


# ─── Technical Indicators ─────────────────────────────────────────────────────
def calc_ema(values: list, length: int) -> list:
    vals = [float(v) for v in values]
    k = 2 / (length + 1)
    ema = [vals[0]]
    for v in vals[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema

def calc_rsi(closes: list, length: int = 14) -> list:
    closes = [float(c) for c in closes]
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    if len(gains) < length:
        return [50.0] * len(closes)
    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length
    rsi_vals = []
    for i in range(length, len(gains)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi_vals.append(100 - 100 / (1 + rs))
    padding = [50.0] * (len(closes) - len(rsi_vals))
    return padding + rsi_vals

def calc_atr(highs: list, lows: list, closes: list, length: int = 14) -> list:
    highs  = [float(h) for h in highs]
    lows   = [float(l) for l in lows]
    closes = [float(c) for c in closes]
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i]  - closes[i-1]))
        trs.append(tr)
    if len(trs) < length:
        return [trs[0] if trs else 0] * len(closes)
    atr = [sum(trs[:length]) / length]
    for tr in trs[length:]:
        atr.append((atr[-1] * (length - 1) + tr) / length)
    padding = [atr[0]] * (len(closes) - len(atr))
    return padding + atr


# ─── Backtesting Engine ───────────────────────────────────────────────────────
class BacktestEngine:
    """
    Supervisor/Backtester agent — runs in same process.
    Tests strategy on stored kline history and returns performance metrics.
    """

    def run(self, klines: list, params: dict) -> dict:
        """
        klines: list of [ts, open, high, low, close, volume]
        params: strategy parameter dict
        Returns: {roi, win_rate, max_dd, sharpe, total_trades, profit_factor}
        """
        closes = [float(k[4]) for k in klines]
        highs  = [float(k[2]) for k in klines]
        lows   = [float(k[3]) for k in klines]

        ema_len  = params.get("ema_length", EMA_LENGTH)
        rsi_len  = params.get("rsi_length", RSI_LENGTH)
        atr_len  = params.get("atr_length", ATR_LENGTH)
        sl_m     = params.get("atr_sl_mult",  ATR_SL_MULT)
        tp1_m    = params.get("atr_tp1_mult", ATR_TP1_MULT)
        tp2_m    = params.get("atr_tp2_mult", ATR_TP2_MULT)
        tp3_m    = params.get("atr_tp3_mult", ATR_TP3_MULT)
        trail_m  = params.get("atr_trail",    ATR_TRAIL)
        tp1_cls  = params.get("tp1_close_pct", TP1_CLOSE_PCT)
        tp2_cls  = params.get("tp2_close_pct", TP2_CLOSE_PCT)
        rsi_ob   = params.get("rsi_ob", RSI_OB)
        rsi_os   = params.get("rsi_os", RSI_OS)

        need = max(ema_len, rsi_len, atr_len) + 10
        if len(closes) < need:
            return {"error": f"Not enough bars: {len(closes)} < {need}"}

        ema  = calc_ema(closes, ema_len)
        rsi  = calc_rsi(closes, rsi_len)
        atr  = calc_atr(highs, lows, closes, atr_len)

        equity    = 1000.0  # normalised starting equity
        peak_eq   = equity
        trades    = []
        equity_curve = [equity]
        position  = None  # {side, entry, sl, tp1, tp2, tp3, trail_ref, qty, remaining, tp1_done, tp2_done}

        for i in range(need, len(closes)):
            c, h, l = closes[i], highs[i], lows[i]
            pc       = closes[i-1]
            cur_ema  = ema[i]; prev_ema = ema[i-1]
            cur_rsi  = rsi[i]
            cur_atr  = atr[i]

            if position:
                side = position["side"]
                remaining = position["remaining"]

                if side == "Buy":
                    # Update trail ref
                    position["trail_ref"] = max(position["trail_ref"], h)
                    trail_sl = position["trail_ref"] * (1 - trail_m * cur_atr / c) if position["tp1_done"] else None

                    # SL hit
                    if not position["tp1_done"] and l <= position["sl"]:
                        pnl = (position["sl"] - position["entry"]) * remaining
                        equity += pnl
                        trades.append({"pnl": pnl, "reason": "sl"})
                        position = None; equity_curve.append(equity); continue

                    # Trailing SL
                    if position["tp1_done"] and trail_sl and l <= trail_sl:
                        pnl = (trail_sl - position["entry"]) * remaining
                        equity += pnl
                        trades.append({"pnl": pnl, "reason": "trail"})
                        position = None; equity_curve.append(equity); continue

                    # TP1
                    if not position["tp1_done"] and h >= position["tp1"]:
                        closed = remaining * tp1_cls
                        pnl = (position["tp1"] - position["entry"]) * closed
                        equity += pnl
                        position["remaining"] -= closed
                        position["tp1_done"]   = True
                        position["sl"]         = position["entry"]  # move SL to BE

                    # TP2
                    if position["tp1_done"] and not position["tp2_done"] and h >= position["tp2"]:
                        closed = position["remaining"] * tp2_cls
                        pnl = (position["tp2"] - position["entry"]) * closed
                        equity += pnl
                        position["remaining"] -= closed
                        position["tp2_done"]   = True

                    # TP3
                    if position["tp2_done"] and h >= position["tp3"]:
                        pnl = (position["tp3"] - position["entry"]) * position["remaining"]
                        equity += pnl
                        trades.append({"pnl": pnl, "reason": "tp3"})
                        position = None; equity_curve.append(equity); continue

                else:  # Short
                    position["trail_ref"] = min(position["trail_ref"], l)
                    trail_sl = position["trail_ref"] * (1 + trail_m * cur_atr / c) if position["tp1_done"] else None

                    if not position["tp1_done"] and h >= position["sl"]:
                        pnl = (position["entry"] - position["sl"]) * remaining
                        equity += pnl
                        trades.append({"pnl": pnl, "reason": "sl"})
                        position = None; equity_curve.append(equity); continue

                    if position["tp1_done"] and trail_sl and h >= trail_sl:
                        pnl = (position["entry"] - trail_sl) * remaining
                        equity += pnl
                        trades.append({"pnl": pnl, "reason": "trail"})
                        position = None; equity_curve.append(equity); continue

                    if not position["tp1_done"] and l <= position["tp1"]:
                        closed = remaining * tp1_cls
                        pnl = (position["entry"] - position["tp1"]) * closed
                        equity += pnl
                        position["remaining"] -= closed
                        position["tp1_done"]   = True
                        position["sl"]         = position["entry"]

                    if position["tp1_done"] and not position["tp2_done"] and l <= position["tp2"]:
                        closed = position["remaining"] * tp2_cls
                        pnl = (position["entry"] - position["tp2"]) * closed
                        equity += pnl
                        position["remaining"] -= closed
                        position["tp2_done"]   = True

                    if position["tp2_done"] and l <= position["tp3"]:
                        pnl = (position["entry"] - position["tp3"]) * position["remaining"]
                        equity += pnl
                        trades.append({"pnl": pnl, "reason": "tp3"})
                        position = None; equity_curve.append(equity); continue

                equity_curve.append(equity)
                peak_eq = max(peak_eq, equity)
                continue

            # ── Entry signals ──────────────────────────────────────
            long_signal  = pc <= prev_ema and c > cur_ema and cur_rsi < rsi_ob
            short_signal = pc >= prev_ema and c < cur_ema and cur_rsi > rsi_os

            sl_dist  = cur_atr * sl_m
            qty      = 0.20  # 20% of normalised equity unit

            if long_signal:
                position = {
                    "side": "Buy", "entry": c, "remaining": qty,
                    "sl":   c - sl_dist,
                    "tp1":  c + cur_atr * tp1_m,
                    "tp2":  c + cur_atr * tp2_m,
                    "tp3":  c + cur_atr * tp3_m,
                    "trail_ref": h,
                    "tp1_done": False, "tp2_done": False,
                }
            elif short_signal:
                position = {
                    "side": "Sell", "entry": c, "remaining": qty,
                    "sl":   c + sl_dist,
                    "tp1":  c - cur_atr * tp1_m,
                    "tp2":  c - cur_atr * tp2_m,
                    "tp3":  c - cur_atr * tp3_m,
                    "trail_ref": l,
                    "tp1_done": False, "tp2_done": False,
                }

            equity_curve.append(equity)
            peak_eq = max(peak_eq, equity)

        # ── Metrics ──────────────────────────────────────────────
        if not trades:
            return {"roi": 0, "win_rate": 0, "max_dd": 0, "sharpe": 0,
                    "total_trades": 0, "profit_factor": 0}

        total_pnl  = sum(t["pnl"] for t in trades)
        wins       = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses     = [t["pnl"] for t in trades if t["pnl"] <= 0]
        win_rate   = len(wins) / len(trades) * 100
        gross_win  = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 1
        pf         = gross_win / gross_loss

        # Sharpe (daily returns approximation)
        rets = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve))]
        if len(rets) > 1:
            avg_r = sum(rets) / len(rets)
            std_r = (sum((r - avg_r)**2 for r in rets) / len(rets)) ** 0.5
            sharpe = (avg_r / std_r * (288 ** 0.5)) if std_r > 0 else 0  # 288 5min bars/day
        else:
            sharpe = 0

        # Max drawdown
        peak = equity_curve[0]; max_dd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd   = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        roi = (equity_curve[-1] - 1000) / 1000 * 100

        return {
            "roi":           round(roi, 2),
            "win_rate":      round(win_rate, 1),
            "max_dd":        round(max_dd * 100, 2),
            "sharpe":        round(sharpe, 2),
            "total_trades":  len(trades),
            "profit_factor": round(pf, 2),
        }

    def optimize(self, klines: list, base_params: dict) -> dict:
        """
        Grid search over key parameters. Returns best params found.
        Backtester agent role: find settings that maximise Sharpe while
        keeping max_dd < 8% and roi >= target pace for +10%/month.
        """
        log.info("[BACKTESTER] Running parameter optimisation...")
        best_params = base_params.copy()
        best_score  = -999

        # Search space
        ema_lengths = [100, 150, 200]
        sl_mults    = [0.8, 1.0, 1.2]
        tp1_mults   = [1.5, 2.0, 2.5]
        rsi_obs     = [65, 70, 75]

        total_combos = len(ema_lengths) * len(sl_mults) * len(tp1_mults) * len(rsi_obs)
        log.info(f"[BACKTESTER] Testing {total_combos} parameter combinations...")

        for ema_l in ema_lengths:
            for sl_m in sl_mults:
                for tp1_m in tp1_mults:
                    for rsi_ob in rsi_obs:
                        params = base_params.copy()
                        params.update({
                            "ema_length":    ema_l,
                            "atr_sl_mult":   sl_m,
                            "atr_tp1_mult":  tp1_m,
                            "atr_tp2_mult":  tp1_m * 2,
                            "atr_tp3_mult":  tp1_m * 3,
                            "rsi_ob":        rsi_ob,
                            "rsi_os":        100 - rsi_ob,
                        })
                        result = self.run(klines, params)
                        if "error" in result:
                            continue
                        # Score = Sharpe * profit_factor, penalise high DD
                        dd_pen = max(0, result["max_dd"] - 6) * 2
                        score  = result["sharpe"] * result["profit_factor"] - dd_pen
                        if score > best_score and result["max_dd"] < 9:
                            best_score  = score
                            best_params = params.copy()
                            best_params["_score"]  = round(score, 3)
                            best_params["_result"] = result

        log.info(f"[BACKTESTER] Best score: {best_score:.3f}")
        if "_result" in best_params:
            r = best_params["_result"]
            log.info(f"[BACKTESTER] Best result: ROI={r['roi']}% | WR={r['win_rate']}% | "
                     f"Sharpe={r['sharpe']} | DD={r['max_dd']}% | PF={r['profit_factor']}")
        return best_params


# ─── Position State ───────────────────────────────────────────────────────────
@dataclass
class PositionState:
    symbol:           str
    side:             str
    entry_price:      float
    qty:              float
    remaining_qty:    float
    entry_time:       str
    sl_price_val:     float = 0.0
    tp1_price_val:    float = 0.0
    tp2_price_val:    float = 0.0
    tp3_price_val:    float = 0.0
    tp1_reached:      bool  = False
    tp2_reached:      bool  = False
    tp3_reached:      bool  = False
    high_since_entry: float = 0.0
    low_since_entry:  float = float("inf")
    pyramid_level:    int   = 0
    partial_exits:    list  = field(default_factory=list)

    def update_extremes(self, high, low):
        self.high_since_entry = max(self.high_since_entry, high)
        self.low_since_entry  = min(self.low_since_entry, low)

    def sl(self):   return self.sl_price_val
    def tp1(self):  return self.tp1_price_val
    def tp2(self):  return self.tp2_price_val
    def tp3(self):  return self.tp3_price_val

    def trailing_stop(self, atr: float) -> float:
        if self.side == "Buy":
            ts = self.high_since_entry - ATR_TRAIL * atr
            return max(self.entry_price, ts)
        else:
            ts = self.low_since_entry + ATR_TRAIL * atr
            return min(self.entry_price, ts)

    def unrealised_pnl(self, price: float) -> float:
        if self.side == "Buy":
            return (price - self.entry_price) * self.remaining_qty
        return (self.entry_price - price) * self.remaining_qty


# ─── Gist State ───────────────────────────────────────────────────────────────
class GistState:
    FILENAME = "apex_prop_state.json"

    def __init__(self, token, gist_id):
        self.token   = token
        self.gist_id = gist_id
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

    def load(self) -> dict:
        if not self.gist_id:
            log.warning("No GIST_ID — using in-memory state")
            return self._default()
        try:
            r = requests.get(f"https://api.github.com/gists/{self.gist_id}",
                             headers=self.headers, timeout=10)
            r.raise_for_status()
            return json.loads(r.json()["files"][self.FILENAME]["content"])
        except Exception as e:
            log.error(f"State load failed: {e}")
            return self._default()

    def save(self, state: dict):
        if not self.gist_id or not self.token:
            log.warning("No Gist credentials — state not persisted")
            return
        try:
            requests.patch(
                f"https://api.github.com/gists/{self.gist_id}",
                headers=self.headers,
                json={"files": {self.FILENAME: {"content": json.dumps(state, indent=2)}}},
                timeout=10,
            )
            log.info("State saved to Gist")
        except Exception as e:
            log.error(f"State save failed: {e}")

    def _default(self) -> dict:
        return {
            "positions":         {},
            "trades":            [],
            "equity_curve":      [],
            "initial_equity":    None,
            "peak_equity":       None,
            "daily_start":       None,
            "daily_start_eq":    None,
            "challenge_start":   None,
            "challenge_start_eq":None,
            "last_run":          None,
            "run_count":         0,
            # Agent communication
            "agent_directives":  {},   # Leader → other agents
            "supervisor_flags":  {},   # Supervisor per-symbol flags: halt/tweak
            "optimized_params":  {},   # Backtester → live params per symbol
            "backtest_results":  {},   # Latest backtest per symbol
            "supervisor_report": {},   # Latest supervisor health report
        }


# ─── Supervisor Agent ─────────────────────────────────────────────────────────
class SupervisorAgent:
    """
    Monitors per-asset performance. Decisions:
      - HALT: if token is losing consistently or too close to daily DD
      - TWEAK: recommend param adjustments written to agent_directives
      - PROMOTE: if token is performing well, allow slightly larger sizing
    Goal: protect the +10%/month pace, never blow the prop firm account.
    """

    def evaluate(self, symbol: str, state: dict, equity: float) -> dict:
        trades = [t for t in state["trades"] if t["symbol"] == symbol]
        flags  = state.get("supervisor_flags", {}).get(symbol, {})

        report = {
            "symbol":        symbol,
            "ts":            datetime.now(timezone.utc).isoformat(),
            "action":        "OK",
            "reason":        "",
            "trade_count":   len(trades),
        }

        if len(trades) < 3:
            report["reason"] = "Insufficient trade history"
            return report

        recent = trades[-10:]  # last 10 trades
        wins   = sum(1 for t in recent if t["pnl"] > 0)
        losses = len(recent) - wins
        wr     = wins / len(recent)
        total_pnl = sum(t["pnl"] for t in recent)

        report["recent_wr"]  = round(wr * 100, 1)
        report["recent_pnl"] = round(total_pnl, 2)

        # Halt condition: win rate < 30% on last 10 trades
        if wr < 0.30:
            report["action"] = "HALT"
            report["reason"] = f"WR {wr*100:.0f}% on last {len(recent)} trades — halting {symbol}"
            log.warning(f"[SUPERVISOR] HALTING {symbol}: {report['reason']}")
            state.setdefault("supervisor_flags", {})[symbol] = {"halted": True, "ts": report["ts"]}
            return report

        # Tweak condition: win rate < 45% — suggest tighter SL
        if wr < 0.45 and total_pnl < 0:
            report["action"] = "TWEAK"
            report["reason"] = f"WR {wr*100:.0f}%, PnL ${total_pnl:.2f} — suggest tighter SL"
            state.setdefault("agent_directives", {})[symbol] = {
                "adjust_sl_mult": max(0.7, ATR_SL_MULT - 0.2),
                "reason": report["reason"],
                "ts": report["ts"],
            }
            log.info(f"[SUPERVISOR] TWEAKING {symbol}: {report['reason']}")
            return report

        # Resume halted symbol if improved
        if flags.get("halted") and wr > 0.55:
            report["action"] = "RESUME"
            report["reason"] = f"WR recovered to {wr*100:.0f}% — resuming {symbol}"
            state.setdefault("supervisor_flags", {})[symbol] = {"halted": False}
            log.info(f"[SUPERVISOR] RESUMING {symbol}: {report['reason']}")
            return report

        # Promote: strong performance → slightly larger size
        if wr > 0.65 and total_pnl > 0:
            report["action"] = "PROMOTE"
            report["reason"] = f"WR {wr*100:.0f}% strong — allowing up to 22% size"
            state.setdefault("agent_directives", {})[symbol] = {
                "pos_size_override": 0.22,
                "reason": report["reason"],
                "ts": report["ts"],
            }

        report["reason"] = report.get("reason") or "All clear"
        return report

    def check_monthly_pace(self, state: dict, equity: float) -> dict:
        """Check if we're on pace for +10% this month."""
        init = state.get("challenge_start_eq") or state.get("initial_equity") or equity
        now  = datetime.now(timezone.utc)
        cstart = state.get("challenge_start")

        if cstart:
            start_dt = datetime.fromisoformat(cstart.replace("Z", "+00:00"))
            days_in  = (now - start_dt).days
        else:
            days_in = 0
            state["challenge_start"]    = now.isoformat()
            state["challenge_start_eq"] = equity

        if days_in == 0:
            return {"days_in": 0, "current_pct": 0, "target_pct": 0, "on_pace": True}

        current_pct  = (equity - init) / init * 100
        target_pct   = (days_in / 30) * 10          # linear +10% over 30 days
        on_pace      = current_pct >= target_pct * 0.8  # within 80% of daily pace

        report = {
            "days_in":     days_in,
            "current_pct": round(current_pct, 2),
            "target_pct":  round(target_pct, 2),
            "on_pace":     on_pace,
            "equity":      round(equity, 2),
        }
        log.info(f"[SUPERVISOR] Monthly pace: {current_pct:.2f}% / target {target_pct:.2f}% | "
                 f"{'ON PACE ✓' if on_pace else 'BEHIND — need to catch up'}")
        return report


# ─── Leader Agent ─────────────────────────────────────────────────────────────
class LeaderAgent:
    """
    Orchestrates the run:
      1. Checks global risk (DD limits)
      2. Runs Supervisor on each symbol
      3. Triggers Backtester every N runs
      4. Applies optimised params from state
      5. Delegates per-symbol processing
    """

    BACKTEST_EVERY = 12   # run optimisation every 12 cycles (~1 hour on 5min)

    def __init__(self, client: BybitClient, gist: GistState):
        self.client     = client
        self.gist       = gist
        self.state      = gist.load()
        self.supervisor = SupervisorAgent()
        self.backtester = BacktestEngine()
        self.now        = datetime.now(timezone.utc).isoformat()

    def run(self):
        log.info("=" * 60)
        log.info(f"APEX PROP | GhoulHQ Prop Firm Division | {'TESTNET' if TESTNET else 'LIVE'}")
        log.info(f"Tokens: {TOKENS} | TF: {TIMEFRAME}m | DRY_RUN: {DRY_RUN}")
        log.info("=" * 60)

        # ── Wallet ────────────────────────────────────────────
        try:
            wallet  = self.client.get_wallet_balance()
            equity  = wallet["equity"]
        except Exception as e:
            log.error(f"[LEADER] Cannot reach Bybit: {e}")
            return

        log.info(f"[LEADER] Equity: ${equity:,.2f} | Unrealised: ${wallet['unrealised_pnl']:,.2f}")

        # ── Initialise ────────────────────────────────────────
        if self.state["initial_equity"] is None:
            self.state["initial_equity"] = equity
            self.state["peak_equity"]    = equity
            self.state["challenge_start"]    = self.now
            self.state["challenge_start_eq"] = equity
            log.info(f"[LEADER] Challenge started — target: +${equity * MONTHLY_TARGET_PCT:,.2f} in 30 days")

        self.state["peak_equity"] = max(self.state["peak_equity"], equity)
        peak = self.state["peak_equity"]

        # ── Risk checks ───────────────────────────────────────
        drawdown = (peak - equity) / peak if peak > 0 else 0
        if drawdown >= MAX_DD_PCT:
            log.error(f"[LEADER] MAX DRAWDOWN {drawdown*100:.2f}% ≥ {MAX_DD_PCT*100:.0f}% — ALL POSITIONS HALTED")
            self._record_equity(equity)
            self.gist.save(self.state)
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state["daily_start"] != today:
            self.state["daily_start"]    = today
            self.state["daily_start_eq"] = equity

        daily_dd = (self.state["daily_start_eq"] - equity) / self.state["daily_start_eq"] \
                   if self.state["daily_start_eq"] else 0
        if daily_dd >= DAILY_DD_PCT:
            log.warning(f"[LEADER] Daily DD {daily_dd*100:.2f}% — exits only")
            self._process_exits_only(equity)
            self._record_equity(equity)
            self.gist.save(self.state)
            return

        # ── Monthly pace check (Supervisor) ───────────────────
        pace = self.supervisor.check_monthly_pace(self.state, equity)
        self.state.setdefault("supervisor_report", {})["monthly_pace"] = pace

        # ── Supervisor: per-symbol health ─────────────────────
        log.info("[LEADER] Delegating to Supervisor for asset health check...")
        for symbol in TOKENS:
            report = self.supervisor.evaluate(symbol, self.state, equity)
            self.state["supervisor_report"][symbol] = report

        # ── Backtester: optimise every N runs ─────────────────
        run_count = self.state.get("run_count", 0)
        if run_count % self.BACKTEST_EVERY == 0:
            log.info("[LEADER] Triggering Backtester optimisation cycle...")
            for symbol in TOKENS:
                try:
                    klines = self.client.get_klines(symbol, TIMEFRAME, limit=500)
                    base   = self._get_params(symbol)
                    opt    = self.backtester.optimize(klines, base)
                    self.state["optimized_params"][symbol] = opt
                    if "_result" in opt:
                        self.state["backtest_results"][symbol] = opt["_result"]
                        log.info(f"[BACKTESTER] {symbol}: {opt['_result']}")
                except Exception as e:
                    log.error(f"[BACKTESTER] Failed for {symbol}: {e}")

        # ── Process each token ────────────────────────────────
        log.info("[LEADER] Delegating trade execution to Trader agent...")
        trader = TraderAgent(self.client, self.state)
        for symbol in TOKENS:
            try:
                # Skip if Supervisor halted this symbol
                flags = self.state.get("supervisor_flags", {}).get(symbol, {})
                if flags.get("halted"):
                    log.info(f"[LEADER] {symbol} HALTED by Supervisor — skipping")
                    continue
                trader.process(symbol, equity)
            except Exception as e:
                log.error(f"[LEADER] Error on {symbol}: {e}")

        # ── Wrap up ───────────────────────────────────────────
        self._record_equity(equity)
        self.state["last_run"]  = self.now
        self.state["run_count"] = run_count + 1
        self.gist.save(self.state)
        log.info(f"[LEADER] Run #{run_count + 1} complete | Trades: {len(self.state['trades'])}")

    def _get_params(self, symbol: str) -> dict:
        """Merge optimised params for symbol with defaults."""
        opt = self.state.get("optimized_params", {}).get(symbol, {})
        return {
            "ema_length":    opt.get("ema_length",    EMA_LENGTH),
            "rsi_length":    opt.get("rsi_length",    RSI_LENGTH),
            "atr_length":    opt.get("atr_length",    ATR_LENGTH),
            "atr_sl_mult":   opt.get("atr_sl_mult",   ATR_SL_MULT),
            "atr_tp1_mult":  opt.get("atr_tp1_mult",  ATR_TP1_MULT),
            "atr_tp2_mult":  opt.get("atr_tp2_mult",  ATR_TP2_MULT),
            "atr_tp3_mult":  opt.get("atr_tp3_mult",  ATR_TP3_MULT),
            "atr_trail":     opt.get("atr_trail",     ATR_TRAIL),
            "tp1_close_pct": opt.get("tp1_close_pct", TP1_CLOSE_PCT),
            "tp2_close_pct": opt.get("tp2_close_pct", TP2_CLOSE_PCT),
            "rsi_ob":        opt.get("rsi_ob",        RSI_OB),
            "rsi_os":        opt.get("rsi_os",        RSI_OS),
        }

    def _record_equity(self, equity: float):
        self.state["equity_curve"].append({"ts": self.now, "equity": round(equity, 4)})
        self.state["equity_curve"] = self.state["equity_curve"][-2000:]

    def _process_exits_only(self, equity: float):
        trader = TraderAgent(self.client, self.state)
        for symbol, pos_data in list(self.state["positions"].items()):
            try:
                ps = PositionState(**pos_data)
                ticker = self.client.get_ticker(symbol)
                price  = float(ticker["lastPrice"])
                high   = float(ticker.get("highPrice24h", price))
                low    = float(ticker.get("lowPrice24h",  price))
                ps.update_extremes(high, low)
                klines = self.client.get_klines(symbol, TIMEFRAME, limit=ATR_LENGTH + 5)
                atr    = calc_atr(
                    [float(k[2]) for k in klines],
                    [float(k[3]) for k in klines],
                    [float(k[4]) for k in klines],
                    ATR_LENGTH
                )[-1]
                live = self.client.get_positions(symbol)
                if live:
                    trader._manage_exits(symbol, ps, price, high, low, atr)
                    self.state["positions"][symbol] = asdict(ps)
            except Exception as e:
                log.error(f"Exit-only failed {symbol}: {e}")


# ─── Trader Agent ─────────────────────────────────────────────────────────────
class TraderAgent:
    """
    Handles live entries and exits for each symbol.
    Uses ATR-based SL/TP. Applies Supervisor size overrides.
    """

    def __init__(self, client: BybitClient, state: dict):
        self.client = client
        self.state  = state
        self.now    = datetime.now(timezone.utc).isoformat()

    def process(self, symbol: str, equity: float):
        log.info(f"\n── {symbol} ──")
        klines = self.client.get_klines(symbol, TIMEFRAME, limit=max(EMA_LENGTH, 220) + 10)
        if len(klines) < EMA_LENGTH + 10:
            log.warning(f"{symbol}: Not enough bars ({len(klines)})")
            return

        closes = [float(k[4]) for k in klines]
        highs  = [float(k[2]) for k in klines]
        lows   = [float(k[3]) for k in klines]

        ema = calc_ema(closes, EMA_LENGTH)
        rsi = calc_rsi(closes, RSI_LENGTH)
        atr = calc_atr(highs, lows, closes, ATR_LENGTH)

        c, h, l  = closes[-1], highs[-1], lows[-1]
        pc       = closes[-2]
        cur_ema  = ema[-1]; prev_ema = ema[-2]
        cur_rsi  = rsi[-1]
        cur_atr  = atr[-1]

        log.info(f"  Close={c:.4f} | EMA-{EMA_LENGTH}={cur_ema:.4f} | RSI={cur_rsi:.1f} | ATR={cur_atr:.4f}")

        # ── Manage existing position ──────────────────────────
        pos_data = self.state["positions"].get(symbol)
        if pos_data:
            ps = PositionState(**pos_data)
            ps.update_extremes(h, l)
            live = self.client.get_positions(symbol)
            if not live:
                log.info(f"{symbol}: Closed externally — clearing")
                self._log_trade(ps, c, "external_close")
                del self.state["positions"][symbol]
            else:
                ps.remaining_qty = float(live[0]["size"])
                self._manage_exits(symbol, ps, c, h, l, cur_atr)
                self.state["positions"][symbol] = asdict(ps)
            return

        # ── Entry signals ─────────────────────────────────────
        long_signal  = pc <= prev_ema and c > cur_ema and cur_rsi < RSI_OB
        short_signal = pc >= prev_ema and c < cur_ema and cur_rsi > RSI_OS

        if len(self.state["positions"]) >= MAX_POSITIONS:
            log.info(f"Max positions reached ({MAX_POSITIONS})")
            return

        if long_signal:
            log.info(f"  LONG SIGNAL ↑ — EMA cross + RSI {cur_rsi:.0f}")
            self._open(symbol, "Buy", c, h, l, cur_atr, equity)
        elif short_signal:
            log.info(f"  SHORT SIGNAL ↓ — EMA cross + RSI {cur_rsi:.0f}")
            self._open(symbol, "Sell", c, h, l, cur_atr, equity)
        else:
            log.info(f"  No signal")

    def _open(self, symbol: str, side: str, price: float, high: float,
              low: float, atr: float, equity: float):
        # Check for Supervisor size override
        directives   = self.state.get("agent_directives", {}).get(symbol, {})
        pos_size_pct = directives.get("pos_size_override", POS_SIZE_PCT)
        sl_mult      = directives.get("adjust_sl_mult", ATR_SL_MULT)

        sl_dist = atr * sl_mult
        sl_p    = price - sl_dist if side == "Buy" else price + sl_dist
        tp1_p   = price + atr * ATR_TP1_MULT if side == "Buy" else price - atr * ATR_TP1_MULT
        tp2_p   = price + atr * ATR_TP2_MULT if side == "Buy" else price - atr * ATR_TP2_MULT
        tp3_p   = price + atr * ATR_TP3_MULT if side == "Buy" else price - atr * ATR_TP3_MULT

        # Kelly-adjusted sizing (simplified: edge / odds)
        edge   = max(0, (ATR_TP1_MULT - 1) / ATR_SL_MULT)  # rough R multiple
        kelly  = edge / (ATR_TP1_MULT / ATR_SL_MULT) if edge > 0 else pos_size_pct
        size_pct = min(pos_size_pct, max(0.05, kelly * 0.5))  # half-Kelly cap

        size_usdt = equity * size_pct
        try:
            info     = self.client.get_instrument_info(symbol)
            min_qty  = float(info["lotSizeFilter"]["minOrderQty"])
            qty_step = float(info["lotSizeFilter"]["qtyStep"])
        except Exception:
            min_qty, qty_step = 0.001, 0.001

        qty = max(min_qty, round(round(size_usdt / price / qty_step) * qty_step, 6))
        log.info(f"  Opening {side} {symbol}: qty={qty} @ ~{price:.4f} | SL={sl_p:.4f} | TP1={tp1_p:.4f}")

        self.client.set_leverage(symbol, LEVERAGE)
        try:
            self.client.place_market_order(symbol, side, qty)
        except Exception as e:
            log.error(f"  Order failed {symbol}: {e}")
            return

        ps = PositionState(
            symbol=symbol, side=side, entry_price=price,
            qty=qty, remaining_qty=qty, entry_time=self.now,
            sl_price_val=sl_p, tp1_price_val=tp1_p,
            tp2_price_val=tp2_p, tp3_price_val=tp3_p,
            high_since_entry=high, low_since_entry=low,
        )
        self.state["positions"][symbol] = asdict(ps)

    def _manage_exits(self, symbol: str, ps: PositionState, price: float,
                      high: float, low: float, atr: float):
        if ps.side == "Buy":
            trail = ps.trailing_stop(atr)

            if not ps.tp1_reached and low <= ps.sl():
                log.warning(f"  {symbol}: SL hit @ {ps.sl():.4f}")
                self._close_all(symbol, ps, ps.sl(), "stop_loss"); return

            if ps.tp1_reached and low <= trail:
                log.info(f"  {symbol}: Trailing stop @ {trail:.4f}")
                self._close_all(symbol, ps, trail, "trailing_stop"); return

            if not ps.tp1_reached and high >= ps.tp1():
                log.info(f"  {symbol}: TP1 @ {ps.tp1():.4f} — closing {TP1_CLOSE_PCT*100:.0f}%")
                try:
                    self.client.close_position_pct(symbol, ps.side, ps.remaining_qty, TP1_CLOSE_PCT)
                    closed = ps.remaining_qty * TP1_CLOSE_PCT
                    pnl = (ps.tp1() - ps.entry_price) * closed
                    ps.partial_exits.append({"reason": "TP1", "price": ps.tp1(), "qty": closed, "pnl": pnl})
                    ps.remaining_qty -= closed
                    ps.tp1_reached    = True
                    ps.sl_price_val   = ps.entry_price  # move to breakeven
                    log.info(f"  TP1: +${pnl:.2f}")
                except Exception as e:
                    log.error(f"  TP1 close failed: {e}")

            if ps.tp1_reached and not ps.tp2_reached and high >= ps.tp2():
                log.info(f"  {symbol}: TP2 @ {ps.tp2():.4f} — closing {TP2_CLOSE_PCT*100:.0f}%")
                try:
                    self.client.close_position_pct(symbol, ps.side, ps.remaining_qty, TP2_CLOSE_PCT)
                    closed = ps.remaining_qty * TP2_CLOSE_PCT
                    pnl = (ps.tp2() - ps.entry_price) * closed
                    ps.partial_exits.append({"reason": "TP2", "price": ps.tp2(), "qty": closed, "pnl": pnl})
                    ps.remaining_qty -= closed
                    ps.tp2_reached    = True
                    log.info(f"  TP2: +${pnl:.2f}")
                except Exception as e:
                    log.error(f"  TP2 close failed: {e}")

            if ps.tp2_reached and high >= ps.tp3():
                log.info(f"  {symbol}: TP3 @ {ps.tp3():.4f} — closing ALL")
                self._close_all(symbol, ps, ps.tp3(), "tp3"); return

        else:  # Short — mirror logic
            trail = ps.trailing_stop(atr)

            if not ps.tp1_reached and high >= ps.sl():
                log.warning(f"  {symbol}: SL hit @ {ps.sl():.4f}")
                self._close_all(symbol, ps, ps.sl(), "stop_loss"); return

            if ps.tp1_reached and high >= trail:
                log.info(f"  {symbol}: Trailing stop @ {trail:.4f}")
                self._close_all(symbol, ps, trail, "trailing_stop"); return

            if not ps.tp1_reached and low <= ps.tp1():
                try:
                    self.client.close_position_pct(symbol, ps.side, ps.remaining_qty, TP1_CLOSE_PCT)
                    closed = ps.remaining_qty * TP1_CLOSE_PCT
                    pnl = (ps.entry_price - ps.tp1()) * closed
                    ps.partial_exits.append({"reason": "TP1", "price": ps.tp1(), "qty": closed, "pnl": pnl})
                    ps.remaining_qty -= closed
                    ps.tp1_reached    = True
                    ps.sl_price_val   = ps.entry_price
                    log.info(f"  TP1: +${pnl:.2f}")
                except Exception as e:
                    log.error(f"  TP1 close failed: {e}")

            if ps.tp1_reached and not ps.tp2_reached and low <= ps.tp2():
                try:
                    self.client.close_position_pct(symbol, ps.side, ps.remaining_qty, TP2_CLOSE_PCT)
                    closed = ps.remaining_qty * TP2_CLOSE_PCT
                    pnl = (ps.entry_price - ps.tp2()) * closed
                    ps.partial_exits.append({"reason": "TP2", "price": ps.tp2(), "qty": closed, "pnl": pnl})
                    ps.remaining_qty -= closed
                    ps.tp2_reached    = True
                    log.info(f"  TP2: +${pnl:.2f}")
                except Exception as e:
                    log.error(f"  TP2 close failed: {e}")

            if ps.tp2_reached and low <= ps.tp3():
                log.info(f"  {symbol}: TP3 @ {ps.tp3():.4f} — closing ALL")
                self._close_all(symbol, ps, ps.tp3(), "tp3"); return

        log.info(f"  {symbol}: Holding | SL={ps.sl():.4f} trail={ps.trailing_stop(atr):.4f}")

    def _close_all(self, symbol: str, ps: PositionState, price: float, reason: str):
        try:
            self.client.close_position_pct(symbol, ps.side, ps.remaining_qty, 1.0)
        except Exception as e:
            log.error(f"  Close all failed {symbol}: {e}")
        self._log_trade(ps, price, reason)
        if symbol in self.state["positions"]:
            del self.state["positions"][symbol]

    def _log_trade(self, ps: PositionState, exit_price: float, reason: str):
        if ps.side == "Buy":
            pnl = (exit_price - ps.entry_price) * ps.remaining_qty
        else:
            pnl = (ps.entry_price - exit_price) * ps.remaining_qty
        partial_pnl = sum(e.get("pnl", 0) for e in ps.partial_exits)
        total_pnl   = pnl + partial_pnl
        self.state["trades"].append({
            "symbol":      ps.symbol, "side": ps.side,
            "entry_price": ps.entry_price, "exit_price": exit_price,
            "qty":         ps.qty, "entry_time": ps.entry_time,
            "exit_time":   self.now, "reason": reason,
            "pnl":         round(total_pnl, 4),
            "partial_exits": ps.partial_exits,
        })
        sign = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
        log.info(f"  CLOSED {ps.symbol} {ps.side} | {reason} | {sign}")


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        log.error("BYBIT_API_KEY and API_SECRET required")
        exit(1)
    client = BybitClient(API_KEY, API_SECRET, BASE_URL)
    gist   = GistState(GIST_TOKEN, GIST_ID)
    leader = LeaderAgent(client, gist)
    leader.run()
