## 美股历史最大回撤与卖出认购期权（Sell Put）收益分析应用

本应用基于 Streamlit 构建，提供：

- 历史价格的最大回撤（Max Drawdown）分析与可视化
- 美股卖出看跌期权（Cash-Secured Put, CSP）收益率与风险估算
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
- 对所选到期日区间与 Delta 区间的卖出看跌期权，计算：
  - 期权权利金（使用中间价，若无则回退到最后成交价）
  - 年化收益率（按现金担保金额，即执行价）
  - 盈亏平衡点（执行价 − 权利金）
  - 到期被指派概率 P(S_T < K) ≈ N(−d2)
  - 触碰概率（近似）≈ min(1, 2 × P(ITM))

> 以上风险度量基于 Black–Scholes 假设与近似，实际结果可能与市场情况有所偏离，仅用于研究和教育目的。

# finance