import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil import tz
from scipy.stats import norm


# ---------------------------
# Data models
# ---------------------------

@dataclass
class MaxDrawdownResult:
    max_drawdown_pct: float
    peak_date: Optional[pd.Timestamp]
    trough_date: Optional[pd.Timestamp]
    series: pd.Series


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_price_history(symbol: str, period_years: int = 10) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * period_years)
    hist = yf.download(
        tickers=symbol,
        start=start.date(),
        end=end.date(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.rename(columns=str.title)
    hist = hist.dropna(subset=["Close"])  # safety
    hist.index = pd.to_datetime(hist.index).tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    return hist


def compute_max_drawdown(close: pd.Series) -> MaxDrawdownResult:
    if close.empty:
        return MaxDrawdownResult(max_drawdown_pct=float("nan"), peak_date=None, trough_date=None, series=pd.Series(dtype=float))

    running_max = close.cummax()
    drawdown = (close / running_max) - 1.0
    trough_idx = drawdown.idxmin()
    max_dd = drawdown.loc[trough_idx]
    peak_idx = close.loc[:trough_idx].idxmax()
    return MaxDrawdownResult(max_drawdown_pct=float(max_dd), peak_date=peak_idx, trough_date=trough_idx, series=drawdown)


@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_option_expirations(symbol: str) -> List[str]:
    try:
        tk = yf.Ticker(symbol)
        return list(tk.options or [])
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=10 * 60)
def fetch_put_chain(symbol: str, expiration: str) -> pd.DataFrame:
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiration)
        puts = chain.puts.copy()
        # Normalize column names
        if "impliedVolatility" in puts.columns:
            puts["impliedVolatility"] = puts["impliedVolatility"].astype(float)
        return puts
    except Exception:
        return pd.DataFrame()


def estimate_spot_price(symbol: str, hist: Optional[pd.DataFrame]) -> Optional[float]:
    try:
        tk = yf.Ticker(symbol)
        fast = getattr(tk, "fast_info", {}) or {}
        last = fast.get("lastPrice") if isinstance(fast, dict) else None
        if last is not None and np.isfinite(last):
            return float(last)
    except Exception:
        pass
    if hist is not None and not hist.empty:
        return float(hist["Close"].iloc[-1])
    return None


def bs_d1_d2(spot: float, strike: float, t_years: float, iv: float, r: float = 0.0, q: float = 0.0) -> Tuple[float, float]:
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return float("nan"), float("nan")
    vsqrt = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t_years) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def put_assignment_probability(spot: float, strike: float, t_years: float, iv: float, r: float = 0.0, q: float = 0.0) -> float:
    _, d2 = bs_d1_d2(spot, strike, t_years, iv, r, q)
    if not np.isfinite(d2):
        return float("nan")
    # P_put_ITM = P(S_T < K) = N(-d2)
    return float(norm.cdf(-d2))


def touch_probability_approx(spot: float, strike: float, t_years: float, iv: float, r: float = 0.0, q: float = 0.0) -> float:
    p_itm = put_assignment_probability(spot, strike, t_years, iv, r, q)
    if not np.isfinite(p_itm):
        return float("nan")
    # Simple approximation: Probability of touching ≈ 2 × P(ITM at expiration)
    return float(min(1.0, max(0.0, 2.0 * p_itm)))


def compute_delta_put(spot: float, strike: float, t_years: float, iv: float, r: float = 0.0, q: float = 0.0) -> float:
    d1, _ = bs_d1_d2(spot, strike, t_years, iv, r, q)
    if not np.isfinite(d1):
        return float("nan")
    # Put delta under Black–Scholes with dividend yield q
    delta = math.exp(-q * t_years) * (norm.cdf(d1) - 1.0)
    return float(delta)


def mid_price(row: pd.Series) -> Optional[float]:
    bid = row.get("bid")
    ask = row.get("ask")
    last = row.get("lastPrice")
    prices = [p for p in [bid, ask, last] if p is not None and np.isfinite(p) and p > 0]
    if bid is not None and ask is not None and np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return float((bid + ask) / 2.0)
    if last is not None and np.isfinite(last) and last > 0:
        return float(last)
    if prices:
        return float(np.median(prices))
    return None


def analyze_puts(
    symbol: str,
    spot: float,
    expirations: List[str],
    dte_min: int,
    dte_max: int,
    target_delta_abs_min: float,
    target_delta_abs_max: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)
    rows: List[dict] = []

    for exp_str in expirations:
        try:
            expiry = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            # Some tickers may return non-standard formats, skip them
            continue
        dte = (expiry - now_utc).days
        if dte < dte_min or dte > dte_max:
            continue

        puts = fetch_put_chain(symbol, exp_str)
        if puts.empty:
            continue

        for _, r in puts.iterrows():
            strike = float(r.get("strike"))
            iv = float(r.get("impliedVolatility", np.nan))
            mprice = mid_price(r)
            if not np.isfinite(iv) or iv <= 0 or mprice is None:
                continue
            t_years = max(1.0 / 365.0, dte / 365.0)
            delta_put = compute_delta_put(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
            if not np.isfinite(delta_put):
                continue
            delta_abs = abs(delta_put)
            if delta_abs < target_delta_abs_min or delta_abs > target_delta_abs_max:
                continue

            premium = float(mprice)
            premium_contract = premium * 100.0
            breakeven = strike - premium
            annualized_on_cash = (premium / strike) * (365.0 / max(1.0, dte))
            annualized_on_breakeven = (premium / max(1e-9, breakeven)) * (365.0 / max(1.0, dte)) if breakeven > 0 else float("nan")
            p_assign = put_assignment_probability(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
            p_touch = touch_probability_approx(spot, strike, t_years, iv, risk_free_rate, dividend_yield)

            rows.append(
                {
                    "expiration": exp_str,
                    "dte": dte,
                    "strike": strike,
                    "mid": premium,
                    "premium_contract": premium_contract,
                    "iv": iv,
                    "delta_put": delta_put,
                    "delta_abs": delta_abs,
                    "breakeven": breakeven,
                    "yield_ann_cash": annualized_on_cash,
                    "yield_ann_breakeven": annualized_on_breakeven,
                    "p_assign": p_assign,
                    "p_touch": p_touch,
                    "bid": r.get("bid"),
                    "ask": r.get("ask"),
                    "last": r.get("lastPrice"),
                    "volume": r.get("volume"),
                    "openInterest": r.get("openInterest"),
                    "inTheMoney": r.get("inTheMoney"),
                    "contractSymbol": r.get("contractSymbol"),
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.sort_values(by=["yield_ann_cash"], ascending=False, inplace=True)
    return df.reset_index(drop=True)


def format_percentage(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.2f}%"


def format_currency(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"$ {x:,.2f}"


def plot_price_and_drawdown(hist: pd.DataFrame, drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100.0, mode="lines", name="Drawdown %", yaxis="y2"))
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Close Price", side="left"),
        yaxis2=dict(title="Drawdown %", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="美股最大回撤与卖出看跌期权分析", layout="wide")
    st.title("美股最大回撤与卖出看跌期权（Sell Put）分析")

    with st.sidebar:
        st.header("参数")
        symbol = st.text_input("股票代码（如 AAPL, MSFT, SPY）", value="AAPL").upper().strip()
        years = st.slider("历史回看年数", min_value=3, max_value=25, value=10, step=1)
        rf = st.number_input("无风险利率 r（年化）", min_value=0.0, max_value=0.20, value=0.045, step=0.005, format="%.3f")
        q = st.number_input("股息率 q（年化，近似）", min_value=0.0, max_value=0.10, value=0.0, step=0.005, format="%.3f")
        dte_range = st.slider("到期天数范围（DTE）", min_value=1, max_value=365, value=(7, 45), step=1)
        delta_abs_range = st.slider("目标 |Delta| 范围（卖出看跌）", min_value=0.01, max_value=0.95, value=(0.15, 0.35), step=0.01)
        st.caption("注：Delta 为看跌期权的绝对值筛选区间")

    # Price history and MDD
    hist = fetch_price_history(symbol, years)
    if hist.empty:
        st.error("未能获取历史数据，请检查股票代码或稍后重试。")
        return

    mdd = compute_max_drawdown(hist["Close"]) 
    spot = estimate_spot_price(symbol, hist)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("现价", format_currency(spot) if spot is not None else "—")
    with col2:
        st.metric("最大回撤", format_percentage(mdd.max_drawdown_pct))
    with col3:
        st.metric("峰值日期", mdd.peak_date.strftime("%Y-%m-%d") if mdd.peak_date else "—")
    with col4:
        st.metric("谷值日期", mdd.trough_date.strftime("%Y-%m-%d") if mdd.trough_date else "—")

    st.plotly_chart(plot_price_and_drawdown(hist, mdd.series), use_container_width=True)

    # Options analysis
    st.subheader("卖出看跌期权（CSP）收益与风险估算")
    exps = fetch_option_expirations(symbol)
    if not exps:
        st.warning("该标的暂无可用期权到期日或数据获取失败。")
        return

    dte_min, dte_max = dte_range
    df = analyze_puts(
        symbol=symbol,
        spot=spot if spot is not None else float(hist["Close"].iloc[-1]),
        expirations=exps,
        dte_min=int(dte_min),
        dte_max=int(dte_max),
        target_delta_abs_min=float(delta_abs_range[0]),
        target_delta_abs_max=float(delta_abs_range[1]),
        risk_free_rate=float(rf),
        dividend_yield=float(q),
    )

    if df.empty:
        st.info("按当前筛选条件未找到合适的期权合约，可调整 DTE 或 |Delta| 范围。")
        return

    display = df.copy()
    display["yield_ann_cash"] = display["yield_ann_cash"].apply(format_percentage)
    display["yield_ann_breakeven"] = display["yield_ann_breakeven"].apply(format_percentage)
    display["p_assign"] = display["p_assign"].apply(format_percentage)
    display["p_touch"] = display["p_touch"].apply(format_percentage)
    display["mid"] = display["mid"].apply(format_currency)
    display["premium_contract"] = display["premium_contract"].apply(format_currency)
    display["breakeven"] = display["breakeven"].apply(format_currency)
    display["iv"] = display["iv"].apply(lambda v: f"{v*100:.2f}%" if np.isfinite(v) else "—")
    display["delta_put"] = display["delta_put"].apply(lambda v: f"{v:.3f}" if np.isfinite(v) else "—")

    st.dataframe(
        display[
            [
                "expiration",
                "dte",
                "strike",
                "mid",
                "premium_contract",
                "iv",
                "delta_put",
                "breakeven",
                "yield_ann_cash",
                "yield_ann_breakeven",
                "p_assign",
                "p_touch",
                "volume",
                "openInterest",
                "contractSymbol",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    top_n = st.slider("显示前 N 个候选（按年化收益率排序）", 1, 50, 10)
    st.write("推荐候选：")
    st.dataframe(display.head(top_n), use_container_width=True, hide_index=True)

    st.caption(
        "风险提示：本工具基于历史数据与简化模型进行估计，不构成任何投资建议。期权具有高风险，请谨慎评估。"
    )


if __name__ == "__main__":
    main()

