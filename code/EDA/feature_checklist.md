# Feature Checklist

每个 `(stock_id, time_id)` 窗口（10 分钟）→ 每个 feature 一个数字。

## 通用统计量（下面简称 **STATS**）

对一个窗口内的时间序列 `s` 计算：

- [x] `mean`
- [x] `std`
- [x] `zscore_last`：`(s 最后一个值 − mean) / std`，衡量窗口末尾偏离多少
- [x] `iqr`：`q75 − q25`
- [x] `ema_last`：指数移动平均在窗口最后一刻的值（span 待定，比如 30s / 60s）

> 注意：价格类（wap、trade price）每个窗口都被归一化到 1 附近，不同窗口之间的**水平**不可比。
> 价格类的 `mean` / `ema_last` 意义不大，建议改用 log return 或相对量（如 `iqr / mean`、`last / mean − 1`）。
> `std`、`iqr` 这类衡量离散程度的统计量没问题。

---

## Book

### 价差 spread

- [x] `spread = (ask_price1 − bid_price1) / wap1` → STATS

### WAP

- [x] `wap1`、`wap2`、`wap12` → STATS 里衡量离散程度的那些（`std`、`iqr`、`max − min`）
- [x] `|wap1_first − 1|`、`|wap1_last − 1|`：偏离参考价的幅度
  - 实测：取绝对值后和 target 相关 0.45，但去掉 rv1 后只剩 0.02–0.05
  - 参考价大约在窗口开始前 ~5 分钟（偏离的方差随时间线性增长反推），所以 `|wap1_first − 1|` ≈ 窗口开始前几分钟的价格变化，是 book 里看不到的额外信息
- [x] `rv1`、`rv2`、`rv12`（已有：log return 平方和开根号）



### 挂单量 size

- [x] `tot_size = bid_size1 + bid_size2 + ask_size1 + ask_size2` → STATS



### 盘口不平衡 size imbalance

- [x] `imb1  = mean(bid_size1) − mean(ask_size1)`
- [x] `imb12 = mean(bid_size1 + bid_size2) − mean(ask_size1 + ask_size2)`
- [x] 建议另加归一化版本（算在 **size** 上）：`imb1 = (bid_size1 − ask_size1) / (bid_size1 + ask_size1)`，`imb12` 同理用两档之和
  - 范围 −1 到 1，不同股票之间可比
  - **先逐行算 imb，再对它算 STATS**（不要先对 size 求平均再算比例）
  - 重点：`std` / `iqr`（盘口翻转得多剧烈），另加 `mean(|imb|)`（不管方向的不平衡程度）



### 数据密度

- [x] `n_quotes`：窗口内有多少个 seconds 出现了 book 更新（即行数）

---



## Trade



### 成交价 price

- [x] `price` → STATS（同样注意价格水平问题）
- [x] 建议：用成交价算的 RV（`rv_trade`）



### 成交次数

- [x] `n_trades`：窗口内的成交记录数
- [x] `trade_ratio = n_trades / n_quotes`



### 买卖方向

- [x] 判断每笔成交是主动买还是主动卖：成交价离 ask 近 → 买方主动；离 bid 近 → 卖方主动
  - 需要把 trade 和 book 按 `seconds_in_bucket` 对齐（`pd.merge_asof`，取成交时刻或之前最近的盘口）
  - 和 mid 比较：`price > mid` → buy，`price < mid` → sell，`== mid` → 无法判断
- [x] `buy_frac = n_buy / (n_buy + n_sell)`（用比例代替 sell/buy，避免除以 0）



### 成交量 volume

- [x] `size` → STATS（价格都在 1 附近，直接用 size 即可）
- [x] `total_volume = sum(size)`
- [ ] 前后半段对比：`volume_2nd_half / volume_1st_half`（后 300s vs 前 300s），看成交是否往窗口末尾集中

---



## 之后可以加的（EDA 里已经发现有用）

- [ ] **时间切片**：上面的 STATS 只用窗口后 300s / 后 150s 再算一遍（近期信息更有预测力）
- [ ] **市场整体特征**：每个 time_id 下，所有股票某个特征的平均值，如 `mkt_rv1 = mean(log rv1)`
- [ ] **杠杆效应（leverage effect）**：跌之后波动上升
  - `ret_win = log(wap1_last / wap1_first)`（带符号）
  - `mkt_ret = 每个 time_id 下 ret_win 的平均`（更重要）
  - 实测（target 去掉 rv1 后的残差，time_id block bootstrap）：`ret_win` Spearman −0.026，95% CI [−0.036, −0.016]
  - 效应来自市场层面：time_id 平均收益 vs 平均残差 −0.089；同一 time_id 内个股相对涨跌 ≈ 0
- [ ] **相对市场**：`log rv1 − mkt_rv1`
- [ ] **股票分组特征**：按相关性聚类，算组内平均
- [ ] **股票层面固定特征**：只在训练集上估计，如每只股票的平均 RV、对市场的 beta

> 所有市场 / 分组层面的特征只能用**当前窗口**的数据（rv1 等），不能用 target。

---



## 工程问题



### price × 1000 会不会让 LightGBM 更好 split？

不会，不需要。

- 决策树只看值的**排序**：分裂点是"x < 阈值"，任何单调变换（乘常数、取 log）都不改变能分出哪些组。
- LightGBM 先把每个特征按**分位数**分成最多 `max_bin`（默认 255）个箱。数值再小，只要有区分度，就会被分到不同的箱里。
- float32 在 1 附近的精度约为 1e-7，远小于价格的最小变动（约 5e-5），也不存在精度问题。

真正影响效果的是特征本身是否**跨窗口可比**（见上面关于价格水平的注意），而不是数值大小。

### 其他

- [ ] 按股票循环计算特征，避免 16GB 内存不够
- [ ] 验证集按 `time_id` 分组切分（GroupKFold）
- [ ] 训练目标：log(target) + 缩放系数 c，或原始 target + `sample_weight = 1/y²`