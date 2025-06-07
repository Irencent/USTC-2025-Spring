import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
# 设置 matplotlib 字体以支持中文
mpl.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体

# 1. 读取 JSON 数据
with open("./vegetable_prices.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. 清洗和结构化：解析有效的日期和价格记录
records = []
for entry in raw_data.values():
    date_str, price_dict = entry
    if not isinstance(date_str, str):
        continue
    try:
        date = datetime.strptime(date_str, "%Y年%m月%d日")
    except ValueError:
        continue
    for vegetable, price in price_dict.items():
        records.append({"date": date, "vegetable": vegetable, "price": price})

df = pd.DataFrame(records)

# 3. 数据透视：按日期为索引，蔬菜种类为列
pivot_df = df.groupby(["date", "vegetable"]).mean(numeric_only=True).reset_index()
pivot_df = pivot_df.pivot(index="date", columns="vegetable", values="price").sort_index()

# 4. 缺失值处理：前向填充
pivot_df = pivot_df.ffill()


# 5. 时间聚合：按月求平均价格
monthly_avg = pivot_df.resample("M").mean()
monthly_avg_interpolated = monthly_avg.interpolate() # 插值处理缺失值


# 6. 长期趋势：3个月滑动平均
rolling_avg = monthly_avg.rolling(window=3, min_periods=1).mean()

# 7. 可视化/输出（仅展示月均价，趋势数据可选）
monthly_avg_interpolated[["西红柿", "黄瓜", "土豆"]].plot(figsize=(10, 6))
plt.title("蔬菜月均价格趋势")
plt.xlabel("月份")
plt.ylabel("价格（元）")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 选取代表性蔬菜（西红柿）进行季节性分析和趋势分析
veg_name = "西红柿"
ts_data = monthly_avg_interpolated[veg_name].dropna()

# === 季节性分析：按月分组后求多年平均，查看是否存在季节性模式 ===
seasonal_pattern = ts_data.groupby(ts_data.index.month).mean()

# === 长期趋势分析：使用低通滤波（rolling）可视化趋势 ===
rolling_trend = ts_data.rolling(window=12, center=True).mean()

# 创建两个图：季节性图 + 趋势图
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# --- 图1：月度平均（多年叠加） --季节性模式
axes[0].plot(seasonal_pattern.index, seasonal_pattern.values, marker='o', linestyle='-')
axes[0].set_title(f"{veg_name} 月度平均价格（多年） - 季节性特征")
axes[0].set_xlabel("月份")
axes[0].set_ylabel("平均价格（元）")
axes[0].grid(True)

# --- 图2：滚动趋势图（12个月滑动平均） -- 长期趋势
axes[1].plot(ts_data.index, ts_data, label="原始价格", color='lightgray')
axes[1].plot(rolling_trend.index, rolling_trend, label="12个月滑动平均", color='red')
axes[1].set_title(f"{veg_name} 月均价格长期趋势")
axes[1].set_xlabel("日期")
axes[1].set_ylabel("价格（元）")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()




from statsmodels.tsa.statespace.sarimax import SARIMAX

# 选取代表性蔬菜进行季节性时间序列建模，如西红柿
veg_name = "西红柿"
ts_data = monthly_avg[veg_name].dropna()

# 拟合 SARIMA 模型（自动化建模参数选择可用pmdarima或网格搜索，但此处手动指定）
# SARIMA(p,d,q)(P,D,Q,s)模型中s=12为月度季节性
model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

# 未来6个月预测
forecast = results.get_forecast(steps=6)
forecast_df = forecast.summary_frame()

# 合并实际和预测值
forecast_df["date"] = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=6, freq="M")
forecast_df["vegetable"] = veg_name
forecast_df = forecast_df[["date", "vegetable", "mean", "mean_ci_lower", "mean_ci_upper"]]

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(ts_data.index, ts_data, label="实际价格", color='blue')
plt.plot(forecast_df["date"], forecast_df["mean"], label="预测价格", color='orange')
plt.fill_between(forecast_df["date"], forecast_df["mean_ci_lower"], forecast_df["mean_ci_upper"], color='lightgray', alpha=0.5, label="置信区间")
plt.title(f"{veg_name}价格预测")
plt.xlabel("日期")
plt.ylabel("价格（元）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 假设你已有 monthly_avg（包含每种蔬菜月均价的 DataFrame）
# 选取“西红柿”的时间序列数据
veg_name = "西红柿"
ts_data = monthly_avg[veg_name].dropna()

# 建立 SARIMA 模型
model = SARIMAX(
    ts_data,
    order=(1, 1, 1),                # 非季节部分 p,d,q
    seasonal_order=(1, 1, 1, 12),   # 季节部分 P,D,Q,s （s=12表示年季节性）
    enforce_stationarity=False,
    enforce_invertibility=False
)

# 拟合模型
results = model.fit(disp=False)

# 输出模型摘要
print(results.summary())

# 输出 AIC 值验证模型优劣
print(f"\nAIC 值：{results.aic:.2f}")

# 可视化拟合结果（拟合 vs 实际）
plt.figure(figsize=(10, 6))
plt.plot(ts_data, label="实际价格", color="blue")
plt.plot(results.fittedvalues, label="模型拟合值", color="orange")
plt.title(f"{veg_name}价格的SARIMA拟合效果")
plt.xlabel("日期")
plt.ylabel("价格（元）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import statsmodels.api as sm
residuals = results.resid

# 残差 ACF 图：看是否存在自相关
sm.graphics.tsa.plot_acf(residuals.dropna(), lags=30)
plt.title("残差的自相关函数 ACF")
plt.show()

# 残差 QQ 图：检验是否服从正态分布
sm.qqplot(residuals.dropna(), line='s')
plt.title("残差的 QQ 图")
plt.show()

