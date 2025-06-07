# %% 导入必需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA  # 导入 ARIMA 模型
import statsmodels.graphics.tsaplots as sgt  # 用于 ACF/PACF 图
from statsmodels.stats.diagnostic import acorr_ljungbox  # Ljung-Box 测试
from statsmodels.tsa.stattools import adfuller  # 导入 ADF 检验
import warnings
import matplotlib.dates as mdates
import os
import itertools  # 用于生成阶数组合

# 忽略常见的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore",
                        category=sm.tools.sm_exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore",
                        category=UserWarning)  # 忽略一些 statsmodels 的 UserWarning
# warnings.filterwarnings("ignore", category=sm.errors.SpecificationWarning) # 如果遇到特定的SpecificationWarning可以取消注释

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 设置 Matplotlib 支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei' 等
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败: {e}. 图表中的中文可能无法显示。")

# %% 0. 创建输出目录和定义结果文件
output_dir_diag = "img_diag"  # 诊断图目录 (保持不变)
output_dir_arima = "img_arima_auto_v2"  # 新目录存放改进后的自动 ARIMA 结果
os.makedirs(output_dir_diag, exist_ok=True)
os.makedirs(output_dir_arima, exist_ok=True)
results_file_path = "analysis_results_arima_auto_v2.txt"  # 新的结果文件名

# %% 1. 加载数据 (与之前相同)
print("--- 1. 加载数据 ---")
file_path = "vegetable_prices_1.csv"
try:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"CSV 文件 '{file_path}' 使用 utf-8 编码加载成功。")
    except UnicodeDecodeError:
        print("UTF-8 解码失败，尝试 gbk 编码...")
        df = pd.read_csv(file_path, encoding='gbk')
        print(f"CSV 文件 '{file_path}' 使用 gbk 编码加载成功。")
    print(f"初始加载行数: {len(df)}")
except FileNotFoundError:
    print(f"错误：在当前目录下未找到文件 '{file_path}'。")
    df = pd.DataFrame(columns=['name', 'price', 'date', 'period'])
except Exception as e:
    print(f"文件加载或处理过程中发生其他错误: {e}")
    df = pd.DataFrame(columns=['name', 'price', 'date', 'period'])

# %% 2. 数据预处理 (与之前相同)
print("\n--- 2. 数据预处理 ---")
if not df.empty:
    if 'period' in df.columns:
        df = df.drop(columns=['period'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    initial_rows = len(df)
    df = df.dropna(subset=['date', 'price'])
    df = df[df['price'] > 0]
    cleaned_rows = len(df)
    print(f"因日期/价格无效或非正数价格，删除了 {initial_rows - cleaned_rows} 行。")

    duplicates = df.duplicated(subset=['name', 'date'], keep=False)
    if duplicates.any():
        print(f"发现 {duplicates.sum()} 条重复记录... 对价格取平均值...")
        df = df.groupby(['name', 'date'], as_index=False)['price'].mean()
        print(f"对重复项取平均后的数据行数：{len(df)}")

    df = df.sort_values(by=['name', 'date'])

    if not df.empty:
        start_date = df['date'].min()
        df['time_numeric'] = (df['date'] - start_date).dt.days
        df['log_price'] = np.log(df['price'])
        print(f"数据日期范围：{start_date.date()} 到 {df['date'].max().date()}")
        print("已添加 'time_numeric' 和 'log_price' 列。")
        print("尝试为每个蔬菜设置日期索引和频率...")

    else:
        print("清洗后数据帧为空，无法继续。")

    print(f"\n预处理完成。总有效记录数：{len(df)}")
else:
    print("数据帧为空，跳过预处理和分析。")

# %% 3. 选择用于详细分析的蔬菜 (与之前相同)
if not df.empty:
    vegetable_counts = df['name'].value_counts()
    min_data_points = 100
    min_days_span = 365 * 2
    eligible_vegetables = []
    for name, count in vegetable_counts.items():
        if count >= min_data_points:
            veg_data = df[df['name'] == name]
            span = (veg_data['date'].max() - veg_data['date'].min()).days
            if span >= min_days_span:
                eligible_vegetables.append(name)

    if eligible_vegetables:
        top_n = min(10, len(eligible_vegetables))  # 保持分析蔬菜数量较少以控制时间
        selected_vegetables = vegetable_counts.loc[
            eligible_vegetables].nlargest(top_n).index.tolist()
        print(f"\n已选择前 {top_n} 种蔬菜进行详细分析:")
        print(selected_vegetables)
    else:
        selected_vegetables = []
        print(f"\n警告：未找到满足标准的蔬菜。")
else:
    selected_vegetables = []

# %% 4. 自动寻找最优 ARIMA 阶数并建模 (改进版)

# *** 定义搜索范围 (可在此处调整) ***
p_max_search = 2  # 扩大 p 的搜索范围
d_max_search = 1  # d 通常 0 或 1 足够
q_max_search = 2  # 扩大 q 的搜索范围


# 定义自动搜索函数 (增加错误处理细节)
def find_best_arima_order_robust(endog, exog, p_max, d_max, q_max):
    """
    通过 AIC 寻找最优 ARIMA(p,d,q) 阶数 (增强错误处理)
    """
    best_aic = np.inf
    best_order = None
    p = range(p_max + 1)
    d = range(d_max + 1)
    q = range(q_max + 1)
    pdq_combinations = list(itertools.product(p, d, q))
    tested_orders = 0

    print(
        f"  正在搜索 {len(pdq_combinations)} 种 ARIMA(p,d,q) 组合 (p<={p_max}, d<={d_max}, q<={q_max})..."
    )

    for order in pdq_combinations:
        if order == (0, 0, 0):
            continue
        # 检查差分后是否仍有足够数据
        if len(endog) <= order[1]:
            continue  # 如果差分阶数大于序列长度，跳过

        current_exog = exog  # d=0 时使用原始 exog
        if order[1] > 0:
            # 如果进行差分，外生变量也需要对应调整（或者从模型中移除趋势项）
            # 简单起见，如果 d=1，我们这里只使用季节性外生变量，并移除趋势项
            # 注意：这是一个简化处理，更严谨的方法可能需要差分外生变量或使用趋势差分
            # current_exog = exog[['cos1', 'sin1', 'cos2', 'sin2']]
            # 或者保持 exog 不变，让模型内部处理
            pass  # 保持 exog 不变，让模型处理

        try:
            tested_orders += 1
            model = ARIMA(
                endog=endog,
                exog=exog,
                order=order,  # 保持 exog 完整传入
                enforce_stationarity=False,
                enforce_invertibility=False)
            # 增加 maxiter 尝试解决收敛问题，但可能增加时间
            results = model.fit(method_kwargs={"maxiter": 200, "disp": False})
            current_aic = results.aic
            # print(f"    尝试 order={order}, AIC={current_aic:.2f}") # 可选打印
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = order
        except (np.linalg.LinAlgError, ValueError,
                IndexError) as e:  # 捕捉常见的拟合错误
            # print(f"    尝试 order={order} 失败 (LinAlg/Value/Index Error): {e}") # 可选打印
            continue
        except Exception as e:  # 捕捉其他未知错误
            # print(f"    尝试 order={order} 失败 (其他错误): {e}") # 可选打印
            continue

    print(
        f"  搜索完成，共尝试 {tested_orders} 个模型。最优阶数 (基于 AIC): {best_order} (AIC: {best_aic:.2f})"
    )
    return best_order


fitted_models_auto_arima = {}

# 打开文件准备写入结果
with open(results_file_path, 'w', encoding='utf-8') as f_out:
    if not df.empty and selected_vegetables:
        print("\n--- 4. 自动搜索与 ARIMA 模型建立 (改进版) ---")
        f_out.write("--- 4. 自动搜索与 ARIMA 模型建立 (改进版) ---\n")
        P = 365.25

        for veg_name in selected_vegetables:
            print(f"\n--- 分析：{veg_name} ---")
            f_out.write(f"\n--- 分析：{veg_name} ---\n")

            df_veg = df[df['name'] == veg_name].copy()

            try:
                df_veg = df_veg.set_index('date').sort_index()
                if df_veg.index.has_duplicates:
                    print(f"警告：{veg_name} 存在重复日期，保留第一个。")
                    df_veg = df_veg[~df_veg.index.duplicated(keep='first')]

                inferred_freq = pd.infer_freq(df_veg.index)
                if inferred_freq:
                    df_veg = df_veg.asfreq(inferred_freq)
                    print(f"为 {veg_name} 设置频率: {inferred_freq}")
                else:
                    print(f"警告：无法为 {veg_name} 推断出规则频率。")
                    # 考虑是否填充缺失值
                    # df_veg = df_veg.asfreq('D').interpolate(method='time')

                # 确保有足够数据点
                min_required_points = max(p_max_search + d_max_search +
                                          q_max_search + 1, 10)  # 最小所需点数
                if df_veg.shape[0] < min_required_points:
                    raise ValueError(
                        f"设置索引/频率后数据点不足 ({df_veg.shape[0]})，至少需要 {min_required_points} 个点"
                    )

            except Exception as e_index:
                print(f"为 {veg_name} 设置日期索引或检查数据点时出错: {e_index}。跳过。")
                f_out.write(f"为 {veg_name} 设置日期索引或检查数据点时出错: {e_index}。跳过。\n")
                continue

            endog = df_veg['log_price'].dropna()

            if endog.empty or len(endog) < min_required_points:
                print(f"跳过 {veg_name}: 内生变量为空或数据点过少 ({len(endog)})。")
                f_out.write(f"跳过 {veg_name}: 内生变量为空或数据点过少 ({len(endog)})。\n")
                continue

            # --- 诊断步骤 ---
            f_out.write("\n诊断信息:\n")
            # 1. ADF Test
            try:
                adf_result = adfuller(endog)
                adf_pvalue = adf_result[1]
                adf_str = f"  ADF 检验 p 值 (log_price): {adf_pvalue:.4f}"
                print(adf_str)
                f_out.write(adf_str + "\n")
                if adf_pvalue > 0.05:
                    f_out.write("    (注: p > 0.05, 序列可能非平稳, 建议搜索包含 d=1 的阶数)\n")
                    # d_search_suggestion = 1 # 可以基于此调整搜索，但为简单起见，我们仍搜索 d=0,1
                else:
                    f_out.write("    (注: p <= 0.05, 序列可能平稳, 建议搜索包含 d=0 的阶数)\n")
                    # d_search_suggestion = 0
            except Exception as e_adf:
                print(f"ADF 检验出错: {e_adf}")
                f_out.write(f"  ADF 检验出错: {e_adf}\n")

            # 2. ACF/PACF Plot
            fig_diag, axs_diag = plt.subplots(1, 2, figsize=(12, 4))
            try:
                sgt.plot_acf(endog,
                             ax=axs_diag[0],
                             lags=40,
                             title=f'{veg_name} - ACF (log_price)')
                sgt.plot_pacf(endog,
                              ax=axs_diag[1],
                              lags=40,
                              title=f'{veg_name} - PACF (log_price)',
                              method='ywm')
                plt.tight_layout()
                safe_veg_name_diag = "".join(
                    c if c.isalnum() or c in ['_', '-', ' '] else '_'
                    for c in veg_name)
                plot_filename_diag = os.path.join(
                    output_dir_diag, f"{safe_veg_name_diag}_ACF_PACF.png")
                plt.savefig(plot_filename_diag)
                print(f"诊断图表已保存到: {plot_filename_diag}")
                f_out.write(f"  ACF/PACF 图已保存到: {plot_filename_diag}\n")
                f_out.write(
                    "    (注: ACF/PACF 图用于辅助手动选择 ARIMA 的 p 和 q 阶数)\n")  # 仍然有用
            except Exception as e_plot_diag:
                print(f"绘制或保存诊断图出错: {e_plot_diag}")
                f_out.write(f"  绘制或保存诊断图出错: {e_plot_diag}\n")
            finally:
                plt.close(fig_diag)

            # 准备外生变量
            time_numeric_indexed = (endog.index - endog.index.min()).days
            exog_dict = {
                'time_numeric': time_numeric_indexed,
                'cos1': np.cos(2 * np.pi * time_numeric_indexed / P),
                'sin1': np.sin(2 * np.pi * time_numeric_indexed / P),
                'cos2': np.cos(4 * np.pi * time_numeric_indexed / P),
                'sin2': np.sin(4 * np.pi * time_numeric_indexed / P)
            }
            exog = pd.DataFrame(exog_dict, index=endog.index)

            # *** 自动寻找最优阶数 (使用更新后的函数和范围) ***
            # !!! 警告：p_max=2, q_max=2 会比之前更耗时 !!!
            best_order = find_best_arima_order_robust(endog,
                                                      exog,
                                                      p_max=p_max_search,
                                                      d_max=d_max_search,
                                                      q_max=q_max_search)

            if best_order is None:
                print(f"在指定范围内无法为 {veg_name} 找到合适的 ARIMA 阶数。")
                f_out.write(f"在指定范围内无法为 {veg_name} 找到合适的 ARIMA 阶数。\n")
                continue

            f_out.write(f"\n自动选择的最优阶数 (基于 AIC): {best_order}\n")
            print(f"为 {veg_name} 使用自动选择的阶数 ARIMA{best_order} 进行最终拟合...")
            f_out.write(
                f"为 {veg_name} 使用自动选择的阶数 ARIMA{best_order} 进行最终拟合...\n")

            try:
                # 最终拟合
                final_model = ARIMA(endog=endog,
                                    exog=exog,
                                    order=best_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                final_result = final_model.fit(method_kwargs={
                    "maxiter": 300,
                    "disp": False
                })  # 增加迭代次数
                fitted_models_auto_arima[veg_name] = final_result

                # --- 验证与输出 ---
                f_out.write("最终模型摘要:\n")
                summary_str = final_result.summary().as_text()
                f_out.write(summary_str)
                f_out.write("\n")

                residuals = final_result.resid

                # Ljung-Box 检验
                try:
                    lags_to_test = min(10, len(residuals) // 2 - 1)
                    if lags_to_test >= 1:
                        lb_test = acorr_ljungbox(residuals,
                                                 lags=[lags_to_test],
                                                 return_df=True)
                        lb_pvalue = lb_test['lb_pvalue'].iloc[0]
                        lb_str = f"Ljung-Box 检验 (滞后 {lags_to_test}) p 值: {lb_pvalue:.4f}\n"
                        print(lb_str.strip())
                        f_out.write(lb_str)
                        if lb_pvalue < 0.05:
                            f_out.write(
                                "  (注: p < 0.05, 残差仍可能存在显著自相关性，模型可能仍需改进/检查或扩大搜索范围)\n"
                            )
                        else:
                            f_out.write(
                                "  (注: p >= 0.05, 无显著证据表明残差存在自相关性，模型诊断通过)\n")
                    else:
                        f_out.write("Ljung-Box 检验: 残差数量不足。\n")
                        print("Ljung-Box 检验: 残差数量不足。")
                except Exception as e_lb:
                    print(f"Ljung-Box 检验出错: {e_lb}")
                    f_out.write(f"Ljung-Box 检验出错: {e_lb}\n")

                # --- 可视化 ---
                fig, axs = plt.subplots(3, 2, figsize=(15, 12))
                fig.suptitle(
                    f'自动 ARIMA 模型分析：{veg_name} (Best Order={best_order})',
                    fontsize=16)

                predicted_values = final_result.predict(start=exog.index[0],
                                                        end=exog.index[-1],
                                                        exog=exog,
                                                        typ='levels')
                axs[0, 0].plot(endog.index,
                               endog,
                               label='实际 log(Price)',
                               alpha=0.7)
                axs[0, 0].plot(predicted_values.index,
                               predicted_values,
                               label='拟合 log(Price)',
                               color='red',
                               linestyle='--')
                axs[0, 0].set_title('实际值 vs. 拟合值 (Log Prices)')
                axs[0, 0].set_xlabel('日期')
                axs[0, 0].set_ylabel('Log(Price)')
                axs[0, 0].legend()
                axs[0, 0].xaxis.set_major_locator(mdates.YearLocator())
                axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                axs[0, 0].tick_params(axis='x', rotation=45)

                axs[0, 1].plot(residuals.index,
                               residuals,
                               label='Residuals',
                               alpha=0.7)
                axs[0, 1].axhline(0, color='red', linestyle='--')
                axs[0, 1].set_title('残差 vs. 时间')
                axs[0, 1].set_xlabel('日期')
                axs[0, 1].set_ylabel('残差')
                axs[0, 1].xaxis.set_major_locator(mdates.YearLocator())
                axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                axs[0, 1].tick_params(axis='x', rotation=45)

                sgt.plot_acf(residuals, ax=axs[1, 0], lags=30)
                axs[1, 0].set_title('残差的 ACF 图')
                sgt.plot_pacf(residuals, ax=axs[1, 1], lags=30, method='ywm')
                axs[1, 1].set_title('残差的 PACF 图')
                sm.qqplot(residuals, line='s', ax=axs[2, 0], alpha=0.4)
                axs[2, 0].set_title('残差的 Q-Q 图')
                axs[2, 1].hist(residuals, bins=30, density=True, alpha=0.7)
                axs[2, 1].set_title('残差直方图')
                axs[2, 1].set_xlabel('残差值')
                axs[2, 1].set_ylabel('密度')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                safe_veg_name = "".join(
                    c if c.isalnum() or c in ['_', '-', ' '] else '_'
                    for c in veg_name)
                plot_filename = os.path.join(
                    output_dir_arima,
                    f"{safe_veg_name}_ARIMA_Auto_{best_order}_analysis.png")
                try:
                    plt.savefig(plot_filename)
                    print(f"图表已保存到: {plot_filename}")
                except Exception as e_save:
                    print(f"保存图表 {plot_filename} 时出错: {e_save}")

                plt.close(fig)

            except Exception as e:
                err_msg = f"无法为 {veg_name} 拟合或验证最优 ARIMA 模型 ({best_order})：{e}\n"
                print(err_msg.strip())
                f_out.write(err_msg)
                fitted_models_auto_arima[veg_name] = None

    # %% 5. 结果解释示例（基于自动选择的 ARIMA 结果） - 写入文件
    f_out.write("\n--- 5. 自动 ARIMA 结果解释示例 ---\n")
    if fitted_models_auto_arima:
        example_veg_auto = next(
            (name for name, model in fitted_models_auto_arima.items()
             if model is not None), None)

        if example_veg_auto and fitted_models_auto_arima[
                example_veg_auto] is not None:
            print(f"\n--- 5. 自动 ARIMA 结果解释示例 ({example_veg_auto}) ---")
            f_out.write(f"({example_veg_auto})\n")
            result = fitted_models_auto_arima[example_veg_auto]
            best_order_for_interp = result.model.order
            params = result.params
            pvalues = result.pvalues

            interpretation_lines = []
            interpretation_lines.append(
                f"模型: 自动选择的 ARIMA {best_order_for_interp} with Exog")
            interpretation_lines.append(
                f"(基于 AIC 在 p<={p_max_search}, d<={d_max_search}, q<={q_max_search} 范围内搜索)"
            )

            interpretation_lines.append("\n外生变量 (趋势/季节性) 系数:")
            param_names = params.index.tolist()
            exog_names_in_params = [
                name
                for name in ['time_numeric', 'cos1', 'sin1', 'cos2', 'sin2']
                if name in param_names
            ]
            intercept_name = next(
                (p for p in param_names
                 if p.lower() == 'intercept' or p.lower() == 'const'), None)
            if intercept_name:
                exog_names_in_params.insert(0, intercept_name)

            for name in exog_names_in_params:
                if name in params.index:
                    coeff = params[name]
                    p_value = pvalues[name]
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    interpretation_lines.append(
                        f"  - {name}: {coeff:.4f} (p={p_value:.3f}) {sig}")
                else:
                    interpretation_lines.append(f"  - {name}: (未在模型参数中找到)")

            interpretation_lines.append("\nARIMA 部分系数:")
            arima_params = [
                p for p in param_names
                if p.startswith(('ar.', 'ma.', 'sigma2'))
            ]
            has_ar_ma = any(p.startswith(('ar.', 'ma.')) for p in arima_params)

            if has_ar_ma:
                for name in arima_params:
                    if name != 'sigma2':
                        coeff = params[name]
                        p_value = pvalues[name]
                        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        interpretation_lines.append(
                            f"  - {name}: {coeff:.4f} (p={p_value:.3f}) {sig}")
            else:
                interpretation_lines.append(f"  (最优模型无 AR/MA 项或阶数为 (0,d,0))")

            if 'sigma2' in params.index:
                interpretation_lines.append(
                    f"\n残差方差 (sigma2): {params['sigma2']:.4f}")

            interpretation_lines.append("\n模型诊断:")
            interpretation_lines.append(
                "  - Ljung-Box 检验: (见上方具体 p 值) 用于判断残差是否存在自相关。p < 0.05 提示模型可能仍需改进。"
            )
            interpretation_lines.append("  - 残差的 ACF/PACF 图: 辅助判断残差是否接近白噪声。")
            interpretation_lines.append("  - Q-Q 图/直方图: 用于检查残差是否接近正态分布。")

            for line in interpretation_lines:
                print(line)
                f_out.write(line + "\n")

        else:
            msg = "无法生成自动 ARIMA 解释示例，因为没有模型成功拟合。\n"
            print(msg.strip())
            f_out.write(msg)
    else:
        msg = "跳过自动 ARIMA 解释，因为没有模型被拟合。\n"
        print(msg.strip())
        f_out.write(msg)

# 文件写入结束
print(f"\n自动 ARIMA 分析结果已写入文件: {results_file_path}")
print(f"诊断图表已保存到目录: {output_dir_diag}")
print(f"自动 ARIMA 分析图表已保存到目录: {output_dir_arima}")
print("\n--- 分析完成 ---")