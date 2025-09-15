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
import pytz


# ---------------------------
# Trading hours utilities
# ---------------------------

def is_market_open() -> bool:
    """检查当前是否在交易时段"""
    now_est = datetime.now(pytz.timezone('US/Eastern'))
    weekday = now_est.weekday()  # 0=Monday, 6=Sunday
    
    # 检查是否是工作日
    if weekday >= 5:  # Saturday or Sunday
        return False
    
    # 检查是否在交易时段 (9:30 AM - 4:00 PM EST)
    market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_est <= market_close

def get_last_trading_day() -> datetime:
    """获取上个交易日"""
    now_est = datetime.now(pytz.timezone('US/Eastern'))
    
    # 如果是周一，返回上周五
    if now_est.weekday() == 0:  # Monday
        days_back = 3
    # 如果是周末，返回上周五
    elif now_est.weekday() >= 5:  # Saturday or Sunday
        days_back = now_est.weekday() - 4
    else:
        days_back = 1
    
    return now_est - timedelta(days=days_back)

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
    
    # 处理多级列名：如果有多级列名，只保留第一级（列名）
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    
    # 重命名列名为标题格式
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
    
    # 找到导致最大回撤的峰值：在回撤低点之前，且价格等于running_max的最后一个点
    peak_idx = close.loc[:trough_idx][close.loc[:trough_idx] == running_max.loc[trough_idx]].index[-1]
    
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
        
        # 如果不在交易时段，尝试使用历史数据
        if not is_market_open():
            # 获取上个交易日的数据
            last_trading_day = get_last_trading_day()
            try:
                # 使用历史数据获取期权链
                hist_data = tk.history(start=last_trading_day.date(), end=(last_trading_day + timedelta(days=1)).date())
                if not hist_data.empty:
                    # 使用历史数据，但期权链可能不完整
                    chain = tk.option_chain(expiration)
                    puts = chain.puts.copy()
                    if "impliedVolatility" in puts.columns:
                        puts["impliedVolatility"] = puts["impliedVolatility"].astype(float)
                    return puts
            except Exception:
                pass
        
        # 正常获取当前期权链
        chain = tk.option_chain(expiration)
        puts = chain.puts.copy()
        # Normalize column names
        if "impliedVolatility" in puts.columns:
            puts["impliedVolatility"] = puts["impliedVolatility"].astype(float)
        return puts
    except Exception:
        return pd.DataFrame()


def calculate_historical_volatility(symbol: str, days: int = 30) -> float:
    """计算历史波动率作为IV的替代"""
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=f"{days}d")
        
        if hist.empty:
            return 0.25  # 默认波动率25%
        
        # 计算日收益率
        returns = hist['Close'].pct_change().dropna()
        
        # 计算年化波动率
        volatility = returns.std() * np.sqrt(252)
        
        return float(volatility)
    except:
        return 0.25  # 默认波动率25%

def estimate_spot_price(symbol: str, hist: Optional[pd.DataFrame]) -> Optional[float]:
    try:
        tk = yf.Ticker(symbol)
        
        # 如果不在交易时段，优先使用历史数据
        if not is_market_open():
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        
        # 尝试获取实时价格
        fast = getattr(tk, "fast_info", {}) or {}
        last = fast.get("lastPrice") if isinstance(fast, dict) else None
        if last is not None and np.isfinite(last):
            return float(last)
    except Exception:
        pass
    
    # 回退到历史数据
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

def call_assignment_probability(spot: float, strike: float, t_years: float, iv: float, r: float = 0.0, q: float = 0.0) -> float:
    """计算卖出看涨期权的被指派概率（到期时股价高于执行价的概率）"""
    _, d2 = bs_d1_d2(spot, strike, t_years, iv, r, q)
    if not np.isfinite(d2):
        return float("nan")
    # P_call_ITM = P(S_T > K) = 1 - N(-d2) = N(d2)
    return float(norm.cdf(d2))


def touch_probability_approx(spot: float, strike: float, t_years: float, iv: float, r: float = 0.0, q: float = 0.0) -> float:
    p_itm = put_assignment_probability(spot, strike, t_years, iv, r, q)
    if not np.isfinite(p_itm):
        return float("nan")
    # Simple approximation: Probability of touching ≈ 2 × P(ITM at expiration)
    return float(min(1.0, max(0.0, 2.0 * p_itm)))

def touch_probability_approx_call(spot: float, strike: float, t_years: float, iv: float, r: float = 0.0, q: float = 0.0) -> float:
    """计算看涨期权的触碰概率近似值"""
    p_itm = call_assignment_probability(spot, strike, t_years, iv, r, q)
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


def analyze_calls(
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
    """分析卖出看涨期权（Covered Call）"""
    now_utc = datetime.now(timezone.utc)
    rows: List[dict] = []
    
    for exp_str in expirations:
        # 处理时区问题：确保exp也有时区信息
        if exp_str.endswith("Z"):
            exp = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
        else:
            # 如果没有时区信息，假设为UTC
            exp = datetime.fromisoformat(exp_str).replace(tzinfo=timezone.utc)
        dte = (exp - now_utc).days
        if dte < dte_min or dte > dte_max:
            continue
            
        try:
            chain = yf.Ticker(symbol).option_chain(exp_str)
            calls = chain.calls
            
            if calls.empty:
                continue
                
            t_years = dte / 365.0
            
            for _, row in calls.iterrows():
                strike = row["strike"]
                bid = row["bid"]
                ask = row["ask"]
                
                # 检查bid和ask是否有效
                if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0:
                    continue
                    
                mid = (bid + ask) / 2
                iv = row["impliedVolatility"]
                volume = row["volume"]
                open_interest = row["openInterest"]
                
                # 处理成交量为NaN的情况
                if pd.isna(volume):
                    volume = 0
                
                if mid <= 0 or not np.isfinite(iv) or iv <= 0:
                    continue
                
                # 计算Delta（看涨期权）
                d1, d2 = bs_d1_d2(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
                if not np.isfinite(d1):
                    continue
                    
                delta_call = norm.cdf(d1)
                delta_abs = abs(delta_call)
                
                if delta_abs < target_delta_abs_min or delta_abs > target_delta_abs_max:
                    continue
                
                # 计算年化收益率（基于持仓成本）
                yield_ann_cash = (mid / spot) * (365 / dte)
                yield_ann_breakeven = (mid / strike) * (365 / dte)
                
                # 计算被指派概率
                p_assign = call_assignment_probability(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
                
                # 计算触碰概率（近似）
                p_touch = touch_probability_approx_call(spot, strike, t_years, iv, risk_free_rate, dividend_yield)
                
                rows.append({
                    "expiration": exp_str,
                    "dte": dte,
                    "strike": strike,
                    "mid": mid,
                    "premium_contract": mid * 100,
                    "iv": iv,
                    "delta_call": delta_call,
                    "breakeven": strike + mid,  # 看涨期权的盈亏平衡点
                    "yield_ann_cash": yield_ann_cash,
                    "yield_ann_breakeven": yield_ann_breakeven,
                    "p_assign": p_assign,
                    "p_touch": p_touch,
                    "volume": volume,
                    "openInterest": open_interest,
                    "contractSymbol": row["contractSymbol"],
                })
                
        except Exception as e:
            continue
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    return df.sort_values("yield_ann_cash", ascending=False)

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
            original_iv = float(r.get("impliedVolatility", np.nan))
            
            # 如果IV数据有问题（太小或太大），使用历史波动率
            if not np.isfinite(original_iv) or original_iv <= 0.01 or original_iv > 5.0:
                iv = calculate_historical_volatility(symbol, 30)
            else:
                iv = original_iv
                
            mprice = mid_price(r)
            if mprice is None:
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


def get_yield_priority_recommendations(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    基于收益优先策略获取推荐期权
    筛选条件：
    1. 年化收益率 > 15%
    2. 被指派概率 < 40% (与强烈推荐页面保持一致)
    3. 成交量 > 50 (与强烈推荐页面保持一致)
    4. 按年化收益率排序
    """
    if df.empty:
        return pd.DataFrame()
    
    # 筛选条件
    filtered = df[
        (df['yield_ann_cash'] > 0.15) &  # 年化收益率 > 15%
        (df['p_assign'] < 0.40) &        # 被指派概率 < 40% (调整阈值)
        (df['volume'] > 50)              # 成交量 > 50 (调整阈值)
    ].copy()
    
    if filtered.empty:
        # 如果严格筛选没有结果，放宽条件
        filtered = df[
            (df['yield_ann_cash'] > 0.10) &  # 年化收益率 > 10%
            (df['p_assign'] < 0.50) &        # 被指派概率 < 50%
            (df['volume'] > 20)              # 成交量 > 20
        ].copy()
    
    # 按年化收益率降序排序
    filtered = filtered.sort_values('yield_ann_cash', ascending=False)
    
    return filtered.head(top_n)


def display_recommendation_card(recommendation: pd.Series, symbol: str) -> None:
    """显示单个推荐期权的卡片"""
    with st.container():
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"**{symbol} {recommendation['strike']:.0f}P**")
            st.caption(f"到期: {recommendation['expiration']}")
            st.caption(f"DTE: {recommendation['dte']}天")
        
        with col2:
            st.metric(
                "年化收益率", 
                f"{recommendation['yield_ann_cash']*100:.1f}%",
                help="基于现金担保金额的年化收益率"
            )
            st.caption(f"权利金: ${recommendation['mid']:.2f}")
        
        with col3:
            st.metric(
                "被指派概率", 
                f"{recommendation['p_assign']*100:.1f}%",
                help="到期时股价低于执行价的概率"
            )
            st.caption(f"盈亏平衡: ${recommendation['breakeven']:.2f}")
        
        with col4:
            st.metric(
                "Delta", 
                f"{recommendation['delta_put']:.3f}",
                help="期权价格对股价变化的敏感度"
            )
            st.caption(f"成交量: {recommendation['volume']}")


def show_about_page():
    """显示About页面"""
    st.title("📚 关于期权策略分析")
    
    st.markdown("""
    ## 🎯 应用简介
    
    本应用基于Streamlit构建，专门用于分析美股卖出看跌期权（Cash-Secured Put, CSP）策略。
    通过量化分析历史数据和期权链信息，为用户提供科学的期权投资决策支持。
    """)
    
    st.markdown("""
    ## 📊 核心指标计算逻辑
    
    ### 1. 年化收益率计算
    
    #### 现金担保年化收益率
    ```
    年化收益率 = (权利金 / 执行价) × (365 / 到期天数)
    ```
    
    **参数说明：**
    - **权利金**：卖出期权获得的收入
    - **执行价**：期权执行价格（现金担保金额）
    - **到期天数**：距离期权到期的时间
    
    **示例：**
    - 执行价：$100
    - 权利金：$2
    - 到期天数：30天
    - 年化收益率 = ($2 / $100) × (365 / 30) = 24.33%
    
    #### 盈亏平衡年化收益率
    ```
    盈亏平衡年化收益率 = (权利金 / 盈亏平衡点) × (365 / 到期天数)
    盈亏平衡点 = 执行价 - 权利金
    ```
    
    ### 2. 被指派概率计算
    
    #### Black-Scholes模型
    基于Black-Scholes期权定价模型计算被指派概率：
    
    ```
    P(Assignment) = P(S_T < K) = N(-d2)
    ```
    
    其中：
    ```
    d2 = d1 - σ√T
    d1 = [ln(S/K) + (r - q + 0.5σ²)T] / (σ√T)
    ```
    
    **参数说明：**
    - **S**：当前股价
    - **K**：执行价
    - **T**：到期时间（年）
    - **σ**：隐含波动率
    - **r**：无风险利率
    - **q**：股息率
    - **N()**：标准正态分布累积分布函数
    
    #### 影响因素
    1. **执行价与现价关系**：价外期权被指派概率较低
    2. **时间衰减**：到期时间越长，被指派概率越高
    3. **波动率**：隐含波动率越高，被指派概率越高
    4. **利率和股息**：利率越高，被指派概率越高
    
    ### 3. 触碰概率计算
    
    ```
    触碰概率 ≈ 2 × P(ITM at expiration)
    ```
    
    这是一个近似公式，用于估算期权在到期前被触及的概率。
    """)
    
    st.markdown("""
    ## ⚠️ 风险提示
    
    1. **模型假设**：基于Black-Scholes模型，假设股价服从对数正态分布
    2. **市场现实**：实际市场可能存在跳跃、波动率微笑等现象
    3. **仅供参考**：计算结果仅用于研究和教育目的，不构成投资建议
    4. **期权风险**：期权交易具有高风险，可能导致重大损失
    5. **充分理解**：请在充分理解风险的前提下进行投资决策
    """)
    
    st.markdown("""
    ## 🔧 技术实现
    
    - **数据源**：yfinance API获取实时股票和期权数据
    - **计算引擎**：Python + NumPy + SciPy进行数值计算
    - **可视化**：Plotly进行交互式图表展示
    - **部署平台**：Streamlit Cloud
    """)

def get_nasdaq100_stocks():
    """获取纳斯达克100指数成分股"""
    # 纳斯达克100主要成分股（简化列表）
    nasdaq100_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST", "PEP",
        "NFLX", "ADBE", "CSCO", "INTC", "AMD", "QCOM", "TXN", "CMCSA", "HON", "AMAT",
        "INTU", "BKNG", "ISRG", "VRTX", "GILD", "ADP", "REGN", "PYPL", "SBUX", "MDLZ",
        "FISV", "ATVI", "CSX", "CHTR", "WBA", "ILMN", "AMGN", "BIIB", "EXC", "EA",
        "LRCX", "KLAC", "MRNA", "CTAS", "NXPI", "SNPS", "CDNS", "ORLY", "IDXX", "DXCM"
    ]
    return nasdaq100_stocks

def analyze_nasdaq100_recommendations():
    """分析纳斯达克100成分股，找出强烈推荐的期权"""
    st.subheader("🔥 强烈推荐买入")
    
    # 显示交易时段状态
    if is_market_open():
        st.success("✅ 当前在交易时段，使用实时数据")
    else:
        st.warning("⚠️ 当前不在交易时段，使用上个交易日数据")
        last_trading_day = get_last_trading_day()
        st.caption(f"数据来源: {last_trading_day.strftime('%Y-%m-%d')} (上个交易日)")
    
    st.markdown("基于纳斯达克100成分股分析，筛选年化收益率>25%且被指派概率<40%的期权")
    
    # 添加筛选参数
    with st.sidebar:
        st.header("筛选参数")
        dte_range = st.slider("到期天数范围（DTE）", min_value=1, max_value=365, value=(1, 45), step=1)
        max_stocks = st.slider("分析股票数量", min_value=5, max_value=50, value=20, step=5)
        st.caption("分析更多股票会需要更长时间")
    
    # 获取纳斯达克100股票列表
    stocks = get_nasdaq100_stocks()
    
    # 存储推荐结果
    recommendations = []
    
    # 显示进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    dte_min, dte_max = dte_range
    
    for i, symbol in enumerate(stocks[:max_stocks]):  # 使用用户设置的数量
        status_text.text(f"正在分析 {symbol}...")
        progress_bar.progress((i + 1) / max_stocks)
        
        try:
            # 获取期权数据
            exps = fetch_option_expirations(symbol)
            if not exps:
                continue
                
            # 获取历史数据
            hist = fetch_price_history(symbol, 2)
            if hist.empty:
                continue
                
            spot = estimate_spot_price(symbol, hist)
            if spot is None:
                continue
            
            # 分析期权
            df = analyze_puts(
                symbol=symbol,
                spot=spot,
                expirations=exps,
                dte_min=dte_min,
                dte_max=dte_max,
                target_delta_abs_min=0.0,  # 移除Delta下限，允许所有价外期权
                target_delta_abs_max=0.99,
                risk_free_rate=0.045,
                dividend_yield=0.0,
            )
            
            if df.empty:
                continue
            
            # 筛选符合条件的期权
            filtered = df[
                (df['yield_ann_cash'] > 0.25) &  # 年化收益率 > 25%
                (df['p_assign'] < 0.40) &        # 被指派概率 < 40% (调整阈值)
                (df['volume'] > 50)              # 成交量 > 50
            ]
            
            if not filtered.empty:
                # 取最佳推荐
                best = filtered.sort_values('yield_ann_cash', ascending=False).iloc[0]
                recommendations.append({
                    'symbol': symbol,
                    'spot': spot,
                    'strike': best['strike'],
                    'expiration': best['expiration'],
                    'dte': best['dte'],
                    'yield_ann': best['yield_ann_cash'],
                    'p_assign': best['p_assign'],
                    'premium': best['mid'],
                    'breakeven': best['breakeven'],
                    'delta': best['delta_put'],
                    'volume': best['volume']
                })
                
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # 显示推荐结果
    if recommendations:
        # 按年化收益率排序
        recommendations.sort(key=lambda x: x['yield_ann'], reverse=True)
        
        st.success(f"找到 {len(recommendations)} 个强烈推荐期权！")
        
        for i, rec in enumerate(recommendations, 1):  # 显示所有推荐
            with st.container():
                # 创建可点击的股票代码按钮
                col_title, col_button = st.columns([3, 1])
                with col_title:
                    st.markdown(f"### 🎯 推荐 #{i}: {rec['symbol']} {rec['strike']:.0f}P")
                with col_button:
                    if st.button(f"📊 分析 {rec['symbol']}", key=f"analyze_{rec['symbol']}_{i}"):
                        # 设置session state来传递参数
                        st.session_state['selected_symbol'] = rec['symbol']
                        st.session_state['switch_to_home'] = True
                        st.rerun()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "年化收益率", 
                        f"{rec['yield_ann']*100:.1f}%",
                        help="基于现金担保金额的年化收益率"
                    )
                    st.caption(f"现价: ${rec['spot']:.2f}")
                
                with col2:
                    st.metric(
                        "被指派概率", 
                        f"{rec['p_assign']*100:.1f}%",
                        help="到期被要求买入股票的概率"
                    )
                    st.caption(f"执行价: ${rec['strike']:.2f}")
                
                with col3:
                    st.metric(
                        "权利金", 
                        f"${rec['premium']:.2f}",
                        help="每份期权的收入"
                    )
                    st.caption(f"盈亏平衡: ${rec['breakeven']:.2f}")
                
                with col4:
                    st.metric(
                        "到期时间", 
                        f"{rec['dte']}天",
                        help="距离期权到期的时间"
                    )
                    st.caption(f"到期日: {rec['expiration']}")
                
                st.markdown("---")
    else:
        st.warning("当前市场条件下未找到符合条件的强烈推荐期权。")
        st.info("建议：可以适当放宽筛选条件或稍后重试。")

def show_sell_call_page():
    """显示Sell Call页面"""
    st.title("📈 Sell Call 策略分析")
    st.markdown("基于您已有持仓的股票，分析卖出看涨期权（Covered Call）的收益和风险")
    
    with st.sidebar:
        st.header("持仓信息")
        symbol = st.text_input("股票代码", value="AAPL").upper().strip()
        cost_basis = st.number_input("持仓成本价 ($)", min_value=0.01, value=150.0, step=0.01, format="%.2f")
        shares = st.number_input("持股数量", min_value=1, value=100, step=1)
        
        st.header("分析参数")
        years = st.slider("历史回看年数", min_value=1, max_value=10, value=2, step=1)
        rf = st.number_input("无风险利率 r（年化）", min_value=0.0, max_value=0.20, value=0.045, step=0.005, format="%.3f")
        q = st.number_input("股息率 q（年化）", min_value=0.0, max_value=0.10, value=0.0, step=0.005, format="%.3f")
        dte_range = st.slider("到期天数范围（DTE）", min_value=1, max_value=365, value=(1, 45), step=1)
        delta_abs_range = st.slider("目标 |Delta| 范围（卖出看涨）", min_value=0.01, max_value=0.95, value=(0.05, 0.95), step=0.01)
        st.caption("注：Delta 为看涨期权的绝对值筛选区间")
    
    # 获取当前股价
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            st.error(f"无法获取 {symbol} 的历史数据")
            return
        current_price = hist["Close"].iloc[-1]
        if pd.isna(current_price) or current_price <= 0:
            st.error(f"获取到的 {symbol} 股价无效: {current_price}")
            return
        st.success(f"当前 {symbol} 股价: ${current_price:.2f}")
        
        # 计算持仓盈亏
        total_cost = cost_basis * shares
        current_value = current_price * shares
        unrealized_pnl = current_value - total_cost
        pnl_pct = (unrealized_pnl / total_cost) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("持仓成本", f"${total_cost:,.2f}")
        with col2:
            st.metric("当前价值", f"${current_value:,.2f}")
        with col3:
            st.metric("未实现盈亏", f"${unrealized_pnl:,.2f}", delta=f"{pnl_pct:.1f}%")
        with col4:
            st.metric("每股成本", f"${cost_basis:.2f}")
            
    except Exception as e:
        st.error(f"无法获取 {symbol} 的股价信息: {str(e)}")
        return
    
    # 获取期权数据
    st.subheader("📊 Sell Call 期权分析")
    exps = fetch_option_expirations(symbol)
    if not exps:
        st.warning("该标的暂无可用期权到期日或数据获取失败。")
        return
    
    dte_min, dte_max = dte_range
    df = analyze_calls(
        symbol=symbol,
        spot=current_price,
        expirations=exps,
        dte_min=int(dte_min),
        dte_max=int(dte_max),
        target_delta_abs_min=float(delta_abs_range[0]),
        target_delta_abs_max=float(delta_abs_range[1]),
        risk_free_rate=float(rf),
        dividend_yield=float(q),
    )
    
    if df.empty:
        st.info("按当前筛选条件未找到合适的看涨期权合约，可调整 DTE 或 |Delta| 范围。")
        return
    
    # 计算基于持仓成本的年化收益率
    df['yield_ann_cost_basis'] = (df['mid'] / cost_basis) * (365 / df['dte'])
    
    # 筛选推荐期权
    st.subheader("🎯 推荐 Sell Call 期权")
    
    # 筛选条件：年化收益率 > 15%，被指派概率 < 30%，成交量 > 50
    strict_filter = (df['yield_ann_cost_basis'] > 0.15) & (df['p_assign'] < 0.30) & (df['volume'] > 50)
    loose_filter = (df['yield_ann_cost_basis'] > 0.10) & (df['p_assign'] < 0.40) & (df['volume'] > 20)
    
    if strict_filter.any():
        recommendations = df[strict_filter].sort_values('yield_ann_cost_basis', ascending=False)
        st.success(f"找到 {len(recommendations)} 个符合严格条件的推荐期权！")
        st.markdown("**筛选条件**: 年化收益率 > 15%，被指派概率 < 30%，成交量 > 50")
    elif loose_filter.any():
        recommendations = df[loose_filter].sort_values('yield_ann_cost_basis', ascending=False)
        st.warning(f"严格条件下未找到推荐，放宽条件后找到 {len(recommendations)} 个推荐期权")
        st.markdown("**筛选条件**: 年化收益率 > 10%，被指派概率 < 40%，成交量 > 20")
    else:
        st.info("当前筛选条件下未找到符合条件的推荐期权，建议调整参数或查看下方完整列表。")
        recommendations = df.head(5)  # 显示前5个作为参考
    
    # 显示推荐期权
    if not recommendations.empty:
        for idx, (_, rec) in enumerate(recommendations.head(3).iterrows(), 1):
            st.markdown(f"### 推荐 #{idx}: {symbol} {rec['strike']:.0f}C")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "年化收益率", 
                    f"{rec['yield_ann_cost_basis']*100:.1f}%",
                    help="基于持仓成本的年化收益率"
                )
                st.caption(f"现价: ${current_price:.2f}")
            
            with col2:
                st.metric(
                    "被指派概率", 
                    f"{rec['p_assign']*100:.1f}%",
                    help="到期被要求卖出股票的概率"
                )
                st.caption(f"执行价: ${rec['strike']:.2f}")
            
            with col3:
                st.metric(
                    "权利金", 
                    f"${rec['mid']:.2f}",
                    help="每份期权的收入"
                )
                st.caption(f"盈亏平衡: ${rec['breakeven']:.2f}")
            
            with col4:
                st.metric(
                    "到期时间", 
                    f"{rec['dte']}天",
                    help="距离期权到期的时间"
                )
                st.caption(f"到期日: {rec['expiration']}")
            
            # 计算总收益
            total_premium = rec['mid'] * shares
            st.info(f"**总权利金收入**: ${total_premium:.2f} (${shares} 股 × ${rec['mid']:.2f})")
            
            st.markdown("---")
    
    # 显示完整期权列表
    st.subheader("📋 完整期权列表")
    
    display = df.copy()
    display["yield_ann_cost_basis"] = display["yield_ann_cost_basis"].apply(format_percentage)
    display["yield_ann_cash"] = display["yield_ann_cash"].apply(format_percentage)
    display["p_assign"] = display["p_assign"].apply(format_percentage)
    display["p_touch"] = display["p_touch"].apply(format_percentage)
    display["mid"] = display["mid"].apply(format_currency)
    display["premium_contract"] = display["premium_contract"].apply(format_currency)
    display["breakeven"] = display["breakeven"].apply(format_currency)
    display["iv"] = display["iv"].apply(lambda v: f"{v*100:.2f}%" if np.isfinite(v) else "—")
    display["delta_call"] = display["delta_call"].apply(lambda v: f"{v:.3f}" if np.isfinite(v) else "—")
    
    st.dataframe(
        display[
            [
                "expiration",
                "dte",
                "strike",
                "mid",
                "premium_contract",
                "iv",
                "delta_call",
                "breakeven",
                "yield_ann_cost_basis",
                "yield_ann_cash",
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
    
    # 添加说明
    with st.expander("📊 Sell Call 策略说明", expanded=False):
        st.markdown("""
        **Covered Call 策略说明：**
        - **策略原理**: 在持有股票的基础上，卖出看涨期权获得权利金收入
        - **年化收益率**: 基于您的持仓成本计算，反映相对于投入资金的收益
        - **被指派概率**: 到期时股价高于执行价，需要以执行价卖出股票的概率
        - **盈亏平衡点**: 执行价 + 权利金，股价超过此点开始亏损
        - **最大收益**: 权利金收入（股价不超过执行价时）
        - **最大风险**: 股价大幅上涨时，只能以执行价卖出，错失上涨收益
        
        **适用场景：**
        - 对股票长期看好，但短期内预期涨幅有限
        - 希望通过期权增加持仓收益
        - 愿意承担股价上涨时被提前卖出的风险
        
        **风险提示：**
        - 股价大幅上涨时，收益被限制在执行价
        - 被指派后需要以执行价卖出股票
        - 期权交易具有时间价值衰减风险
        """)

def main() -> None:
    st.set_page_config(page_title="luluwangzi的期权策略", layout="wide")
    
    # 添加侧边栏导航
    with st.sidebar:
        st.title("🧭 导航")
        
        # 检查是否有跳转到主页的请求
        if 'switch_to_home' in st.session_state and st.session_state['switch_to_home']:
            default_page = "主页"
            # 清除跳转标志
            del st.session_state['switch_to_home']
        else:
            default_page = "主页"
        
        page = st.selectbox("选择页面", ["主页", "强烈推荐", "Sell Call", "关于"], index=["主页", "强烈推荐", "Sell Call", "关于"].index(default_page))
    
    if page == "关于":
        show_about_page()
    elif page == "强烈推荐":
        analyze_nasdaq100_recommendations()
    elif page == "Sell Call":
        show_sell_call_page()
    else:  # 主页
        st.title("luluwangzi的期权策略")
        
        with st.sidebar:
            st.header("参数")
            # 检查是否有从强烈推荐页面传递的股票代码
            default_symbol = "AAPL"
            if 'selected_symbol' in st.session_state:
                default_symbol = st.session_state['selected_symbol']
                # 清除session state，避免重复使用
                del st.session_state['selected_symbol']
            
            symbol = st.text_input("股票代码（如 AAPL, MSFT, SPY）", value=default_symbol).upper().strip()
            years = st.slider("历史回看年数", min_value=3, max_value=25, value=10, step=1)
            rf = st.number_input("无风险利率 r（年化）", min_value=0.0, max_value=0.20, value=0.045, step=0.005, format="%.3f")
            q = st.number_input("股息率 q（年化，近似）", min_value=0.0, max_value=0.10, value=0.0, step=0.005, format="%.3f")
            dte_range = st.slider("到期天数范围（DTE）", min_value=1, max_value=365, value=(1, 45), step=1)
            delta_abs_range = st.slider("目标 |Delta| 范围（卖出看跌）", min_value=0.01, max_value=0.95, value=(0.15, 0.35), step=0.01)
            st.caption("注：Delta 为看跌期权的绝对值筛选区间")

        # Price history and MDD
        hist = fetch_price_history(symbol, years)
        if hist.empty:
            st.error("未能获取历史数据，请检查股票代码或稍后重试。")
            return

        mdd = compute_max_drawdown(hist["Close"]) 
        spot = estimate_spot_price(symbol, hist)

        # 计算历史最高价信息
        hist_high_price = hist["Close"].max()
        hist_high_date = hist["Close"].idxmax()
        hist_high_days_ago = (datetime.now(timezone.utc) - hist_high_date).days
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("现价", format_currency(spot) if spot is not None else "—")
        with col2:
            st.metric("最大回撤", format_percentage(mdd.max_drawdown_pct))
        with col3:
            # 历史最高价显示
            if hist_high_days_ago <= 730:  # 24个月内
                time_str = f"{hist_high_days_ago}天前"
            else:
                time_str = f"{hist_high_days_ago//365}年前"
            st.metric("历史最高价", f"${hist_high_price:.2f}", delta=time_str, help=f"历史最高价: ${hist_high_price:.2f} ({hist_high_date.strftime('%Y-%m-%d')})")
        with col4:
            # 最大回撤低点显示
            if mdd.trough_date and mdd.peak_date:
                trough_days_ago = (datetime.now(timezone.utc) - mdd.trough_date).days
                if trough_days_ago <= 730:  # 24个月内
                    time_str = f"{trough_days_ago}天前"
                else:
                    time_str = f"{trough_days_ago//365}年前"
                
                peak_price = hist.loc[mdd.peak_date, 'Close']
                trough_price = hist.loc[mdd.trough_date, 'Close']
                st.metric("最大回撤低点", f"${trough_price:.2f}", delta=time_str, help=f"从${peak_price:.2f}回撤到${trough_price:.2f} ({mdd.trough_date.strftime('%Y-%m-%d')})")
            else:
                st.metric("最大回撤低点", "—")

        st.plotly_chart(plot_price_and_drawdown(hist, mdd.series), use_container_width=True)
        
        # 添加说明
        with st.expander("📊 指标说明", expanded=False):
            st.markdown("""
            **历史数据指标说明：**
            - **现价**: 当前股票市场价格
            - **最大回撤**: 历史上从某个峰值到谷值的最大跌幅百分比
            - **历史最高价**: 显示价格和时间，整个历史期间的最高价格
            - **最大回撤低点**: 显示价格和时间，以及从哪个峰值回撤而来
            
            **显示格式说明：**
            - 主值显示价格（如 $183.15）
            - Delta显示相对时间（如 "30天前"）
            - 悬停提示显示完整信息（价格、具体日期、回撤详情）
            
            **重要区别：**
            - **历史最高价** ≠ **最大回撤的峰值**
            - 历史最高价是绝对的最高价格
            - 最大回撤的峰值是导致最大回撤的那个峰值（可能不是历史最高价）
            
            **回撤分析意义：**
            - 回撤越小，说明股票价格相对稳定
            - 回撤越大，说明股价波动较大，卖出看跌期权风险相对较高
            - 建议结合历史回撤情况选择合适的期权策略
            """)

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

        # 收益优先策略推荐
        st.subheader("🎯 收益优先策略推荐")
        recommendations = get_yield_priority_recommendations(df, top_n=3)
        
        if not recommendations.empty:
            st.success(f"基于收益优先策略，为您推荐以下 {len(recommendations)} 个最优期权：")
            st.markdown("**筛选条件**: 年化收益率 > 15%，被指派概率 < 40%，成交量 > 50")
            
            for idx, (_, rec) in enumerate(recommendations.iterrows(), 1):
                st.markdown(f"### 推荐 #{idx}")
                display_recommendation_card(rec, symbol)
        else:
            st.warning("当前筛选条件下未找到符合收益优先策略的期权，建议调整参数或查看下方完整列表。")
            st.markdown("**建议**: 可以适当放宽 DTE 范围或 Delta 范围来获得更多选择。")

        st.markdown("---")
        st.subheader("📊 完整期权列表")

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

