import pandas as pd
import sys  # 用于获取脚本文件名，使输出更清晰

# --- 配置 ---
input_filename = 'vegetable_prices.csv'
output_filename = 'vegetable_prices_1.csv'
baseline_period = 7149
baseline_date_str = '2025-03-31'  # 请注意，此日期基于用户原始请求

# --- 获取当前日期用于输出信息 (可选) ---
# 注意：根据您的请求，日期计算的基准是固定的 2025-03-31，这里获取当前日期仅为示例
try:
    from datetime import datetime
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    print(f"脚本运行日期: {current_date_str}")
except ImportError:
    print("无法导入 datetime 模块，将跳过显示当前日期。")

print(f"--- 开始处理脚本: {sys.argv[0]} ---")

# --- 主要逻辑 ---
try:
    # --- 1. 读取数据 ---
    print(f"\n[步骤 1/5] 正在读取文件: {input_filename}...")
    # 指定 encoding='utf-8' 或 'gbk' 或 'utf-8-sig' 以正确处理中文字符
    try:
        df = pd.read_csv(input_filename, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            print("尝试使用 gbk 编码读取...")
            df = pd.read_csv(input_filename, encoding='gbk')
        except UnicodeDecodeError:
            print("尝试使用 utf-8-sig 编码读取...")
            df = pd.read_csv(input_filename, encoding='utf-8-sig')
    print(f"成功读取 {len(df)} 行数据。")

    # --- 2. 处理缺失值 ---
    print(f"\n[步骤 2/5] 检查并处理 'period' 列的缺失值 (NaN)...")
    initial_rows = len(df)
    # 删除 'period' 列值为 NaN 的所有行
    df.dropna(subset=['period'], inplace=True)
    removed_rows = initial_rows - len(df)

    if removed_rows > 0:
        print(f"注意：已从数据中删除 {removed_rows} 行，因为它们的 'period' 列为空或无效。")
    else:
        print("'period' 列中未发现缺失值。")
    print(f"处理缺失值后剩余 {len(df)} 行数据。")

    # --- 3. 确保 'period' 列为数值类型 ---
    # (这一步是额外的保险，确保即使读入了非数字字符串也能处理)
    print(f"\n[步骤 3/5] 确保 'period' 列为数值类型...")
    initial_rows_before_numeric = len(df)
    # errors='coerce' 会将无法转换为数字的值变成 NaN
    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    # 再次删除可能因类型转换产生的 NaN 值
    df.dropna(subset=['period'], inplace=True)
    removed_rows_after_numeric = initial_rows_before_numeric - len(df)

    if removed_rows_after_numeric > 0:
        print(
            f"注意：在转换为数值类型后，额外删除了 {removed_rows_after_numeric} 行，因为它们的 'period' 包含非数字内容。"
        )
    print(f"类型检查后最终有效数据为 {len(df)} 行。")

    # 如果所有行都被删除了，则无需继续
    if len(df) == 0:
        print("\n警告：所有行都因 'period' 列无效而被删除，无法进行日期计算。")
        print(f"--- 脚本处理结束 (无有效数据) ---")
        sys.exit()  # 退出脚本

    # --- 4. 计算日期 ---
    print(f"\n[步骤 4/5] 开始计算日期...")
    # 将基准日期字符串转换为 pandas 的 Timestamp 对象
    baseline_date = pd.to_datetime(baseline_date_str)

    # 定义计算日期的函数 (现在可以假设 period 总是有效的数字)
    def calculate_date(period):
        try:
            # 计算与基准周期的天数差 (period 现在保证是数字)
            # 显式转换为int可能更安全，如果period可能是浮点数的话
            days_difference = baseline_period - int(period)
            # 从基准日期减去天数差得到目标日期
            calculated_date = baseline_date - pd.Timedelta(
                days=days_difference)
            # 将日期格式化为 'YYYY-MM-DD' 字符串
            return calculated_date.strftime('%Y-%m-%d')
        except Exception as e:
            # 添加一个备用的错误捕获，虽然理论上不应发生
            print(f"警告：在为 period '{period}' 计算日期时发生意外错误: {e}。返回空日期。")
            return None  # 或者 pd.NaT

    # 应用函数到 'period' 列，并将结果填充到 'date' 列
    df['date'] = df['period'].apply(calculate_date)
    print("日期计算完成。")

    print("\n处理后数据的前几行预览:")
    print(df.head())

    # --- 5. 保存结果 ---
    print(f"\n[步骤 5/5] 正在保存结果到文件: {output_filename}...")
    # 将更新后的 DataFrame 保存到新的 CSV 文件
    # index=False 表示不将 DataFrame 的索引写入 CSV 文件
    # encoding='utf-8-sig' 通常能更好地兼容 Excel 打开包含中文的 CSV 文件
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"结果已成功保存到 {output_filename}。")

    print(f"\n--- 脚本处理成功结束 ---")

# --- 错误处理 ---
except FileNotFoundError:
    print(f"\n错误：输入文件 '{input_filename}' 未找到。请确保文件存在于脚本运行的目录下。")
    print(f"--- 脚本处理失败 ---")
except KeyError as e:
    print(f"\n错误：输入文件中缺少必要的列: {e}。请检查文件格式是否正确（需要包含 'period' 列等）。")
    print(f"--- 脚本处理失败 ---")
except Exception as e:
    print(f"\n处理过程中发生未预料的错误: {e}")
    # 尝试打印 DataFrame 信息以帮助调试（如果 df 已被定义）
    if 'df' in locals():
        print("\nDataFrame 信息 (发生错误时):")
        try:
            df.info()
            print("\n'period' 列数据类型和可能存在问题的样本:")
            if 'period' in df.columns:
                print(df['period'].dtype)
                # 查找NaN或其他可能的值
                print(df[pd.isna(df['period']) | ~df['period'].
                         apply(lambda x: isinstance(x, (int, float)))].head())
            else:
                print("'period' 列不存在！")
        except Exception as e_info:
            print(f"(获取详细信息时出错: {e_info})")
    print(f"--- 脚本处理失败 ---")