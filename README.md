# Streamlit Snake Game

A simple Snake game built with Streamlit, featuring looping BGM and a SQLite-backed leaderboard.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit (e.g., `http://localhost:8501`). If the background music does not autoplay due to browser policy, click Play in the audio controls at the top.

## How to play

- Use on-screen arrow buttons to change direction
- Click Pause/Resume to pause or continue
- Eat red seeds to grow and score points; you'll see “你太棒了！” when you eat one
- Avoid hitting walls or yourself; the game ends on collision

## Leaderboard

When the game ends and your score is greater than 0, enter your name and submit to add your score to the top-10 leaderboard stored in a local SQLite database (`leaderboard.db`).

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to Streamlit Community Cloud and create a new app
3. Select the repo and set the entrypoint to `app.py`
4. Deploy. Note that `leaderboard.db` will be ephemeral on most serverless deployments.

---
title: MDD & CSP Analyzer
emoji: 📉
colorFrom: purple
colorTo: yellow
sdk: streamlit
sdk_version: 1.49.1
app_file: app.py
pinned: false
---

## 美股历史最大回撤与卖出认购期权（Sell Put）收益分析应用

本应用基于 Streamlit 构建，提供：

- 历史价格的最大回撤（Max Drawdown）分析与可视化
- 美股卖出看跌期权（Cash-Secured Put, CSP）收益率与风险估算
- **🎯 收益优先策略推荐** - 智能筛选最优期权
- 年化收益率、盈亏平衡点、到期被指派概率（P(ITM)）与触碰概率（Probability of Touch）的近似估计

### 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动应用：
```bash
streamlit run app.py
```

### 功能说明

- 通过 `yfinance` 获取历史价格与期权链数据
- 最大回撤按日频价格计算，展示价格与回撤曲线
- **收益优先策略推荐**：
  - 自动筛选年化收益率 > 15% 的期权
  - 控制被指派概率 < 30%
  - 确保成交量 > 100 的流动性
  - 按收益风险比排序推荐
- 对所选到期日区间与 Delta 区间的卖出看跌期权，计算：
  - 期权权利金（使用中间价，若无则回退到最后成交价）
  - 年化收益率（按现金担保金额，即执行价）
  - 盈亏平衡点（执行价 − 权利金）
  - 到期被指派概率 P(S_T < K) ≈ N(−d2)
  - 触碰概率（近似）≈ min(1, 2 × P(ITM))

### 部署说明

本应用已配置支持 Streamlit Cloud 部署：
- 自动检测 `app.py` 作为主应用文件
- 依赖包已在 `requirements.txt` 中定义
- 支持实时数据获取和期权分析

> 以上风险度量基于 Black–Scholes 假设与近似，实际结果可能与市场情况有所偏离，仅用于研究和教育目的。

# finance