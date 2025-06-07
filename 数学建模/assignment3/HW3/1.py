import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time
import re
from tqdm import tqdm  # 用于显示进度条
import datetime
import os  # 用于检查文件是否存在

# --- 配置参数 ---
BASE_URL = "https://ysjt.ustc.edu.cn"
LIST_PAGE_BASE = "https://ysjt.ustc.edu.cn/wjxx/"
HEADERS = {
    'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept':
    'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Connection': 'keep-alive',
    'Referer': LIST_PAGE_BASE  # 添加 Referer 头
}
REQUEST_DELAY = 0.5  # 请求之间的延迟（秒），避免给服务器造成过大压力
MAX_INDEX_PAGES_TO_SCRAPE = 100  # 定义要爬取的目录页总数
OUTPUT_CSV = "ustc_vegetable_prices_final.csv"  # 输出CSV文件名
OUTPUT_EXCEL = "ustc_vegetable_prices_final.xlsx"  # 输出Excel文件名


# --- 辅助函数 ---
def get_soup(url):
    """获取URL内容并返回BeautifulSoup对象，包含错误处理。"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)  # 设置超时时间
        response.raise_for_status()  # 检查请求是否成功
        response.encoding = response.apparent_encoding  # 自动检测编码
        # 优先使用 lxml 解析器，如果未安装则回退到 html.parser
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except Exception:
            soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.Timeout:
        print(f"请求超时: {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求错误 {url}: {e}")
        return None
    except Exception as e:
        print(f"解析 {url} 时出错: {e}")
        return None


def format_date(date_str_input):
    """将 YYYY-MM-DD 或 YYYY年M月D日 格式化为 YYYY/MM/DD。"""
    if not date_str_input:
        return "Unknown Date"
    date_str_input = date_str_input.strip()
    try:
        # 尝试 YYYY-MM-DD 格式
        dt = datetime.datetime.strptime(date_str_input, '%Y-%m-%d')
        return dt.strftime('%Y/%m/%d')
    except ValueError:
        try:
            # 尝试 YYYY年M月D日 格式 (去除可能存在的空格)
            date_str_cleaned = re.sub(r'(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日',
                                      r'\1-\2-\3', date_str_input)
            parts = date_str_cleaned.split('-')
            if len(parts) == 3:
                dt = datetime.datetime(int(parts[0]), int(parts[1]),
                                       int(parts[2]))
                return dt.strftime('%Y/%m/%d')
            else:
                return date_str_input  # 返回原始字符串如果格式无法识别
        except (ValueError, TypeError, AttributeError):
            return date_str_input  # 解析失败则返回原始字符串


def extract_links_and_dates_from_index(soup, index_url):
    """
    从索引页的Soup对象中提取符合条件的子页面链接和对应的日期。
    返回列表，每个元素是 (子页面URL, 格式化后的日期字符串)。
    """
    links_dates = []
    # 定位包含新闻列表的主要 div
    news_div = soup.find('div', id='wp_news_w71')
    if not news_div:
        print(f"警告: 在页面 {index_url} 未找到 ID 为 'wp_news_w71' 的 div")
        return []

    # 查找列表项，基于您提供的第一个HTML示例，结构为 <li class="cle">
    list_items = news_div.find_all('li', class_='cle')

    # 如果找不到 li.cle，尝试查找表格行作为备选（以防万一）
    if not list_items:
        table = news_div.find('table', class_='wp_article_list_table')
        if table:
            list_items = table.find_all('tr')

    if not list_items:
        print(f"警告: 在页面 {index_url} 的新闻区域未找到列表项 (li.cle 或 table tr).")
        return []

    for item in list_items:
        # 查找链接<a>标签，预期在<div class="text...">内
        link_container = item.find('div', class_='text')
        link_tag = link_container.find(
            'a', href=True) if link_container else item.find('a', href=True)

        # 查找日期标签，预期在<div class="time...">内
        time_container = item.find('div', class_='time')
        time_tag = time_container if time_container else item.find(
            ['div', 'span'], class_='time')  # 更广泛的备选

        if link_tag and time_tag:
            link_text = link_tag.get_text(strip=True)
            # *** 使用正则表达式检查链接文本是否符合"蔬菜价格信息（第X期）"格式 ***
            if re.search(r'蔬菜价格信息（第\d+期）', link_text):
                href = link_tag['href']
                # *** 拼接完整 URL (处理相对路径) ***
                full_url = urljoin(BASE_URL, href)
                # *** 提取索引页上的日期文本 ***
                date_str_raw = time_tag.get_text(strip=True)
                # *** 格式化日期 ***
                formatted_date = format_date(date_str_raw)
                if full_url and formatted_date != "Unknown Date":  # 确保URL和日期都有效
                    links_dates.append((full_url, formatted_date))
                    # print(f"找到链接: {full_url} 日期: {formatted_date}") # 调试信息
    return links_dates


def extract_price_data_from_meta(soup, url):
    """
    从子页面的Soup对象中，通过解析<meta name='description'>标签提取蔬菜名和价格。
    返回字典列表: [{'Vegetable': 名称, 'Price (元/斤)': 价格}, ...]
    """
    data = []
    # *** 定位 meta description 标签 ***
    meta_tag = soup.find(
        'meta', attrs={'name': re.compile(r'description',
                                          re.IGNORECASE)})  # 忽略大小写

    if not meta_tag or 'content' not in meta_tag.attrs or not meta_tag[
            'content']:
        print(f"警告: 在页面 {url} 未找到有效的 <meta name='description'> 标签或其内容为空")
        return []  # 如果找不到或内容为空，返回空列表

    content_string = meta_tag['content']

    # *** 使用正则表达式从 meta content 提取 "名称 价格" 对 ***
    # 匹配中文字符（可能包含'/'）作为名称，后面跟数字（可能含小数点）作为价格
    matches = re.findall(r'([\u4e00-\u9fa5/]+)\s*([\d.]+)', content_string)

    if not matches:
        print(f"警告: 在页面 {url} 的 meta 描述中未能用正则表达式匹配到任何'名称-价格'对")
        return []

    processed_count = 0
    for name, price_str in matches:
        name = name.strip()
        price_str = price_str.strip()

        # *** 过滤掉已知的非蔬菜条目 ***
        if name in ["名称", "单价", "价格单位", "元/斤", "期", "蔬菜价格信息", "第"
                    ] or "价格单位" in name or not name or name == '/':
            continue

        if price_str:
            try:
                # 清理价格字符串，只保留数字和小数点
                price_str_cleaned = re.sub(r'[^\d.]', '', price_str)
                if price_str_cleaned:
                    price = float(price_str_cleaned)
                    data.append({
                        'Vegetable': name,
                        'Price (元/斤)': price  # 以浮点数存储价格
                    })
                    processed_count += 1
            except ValueError:
                # print(f"警告: 无法将价格 '{price_str}' 转换为数字 (名称: '{name}', 页面: {url})")
                pass  # 如果价格无法转换成数字，则跳过该条目

    # if processed_count == 0 and matches:
    #     print(f"警告: 在 {url} 中找到匹配项但全部被过滤。")

    return data


# --- 主程序逻辑 ---

# 1. 生成所有索引页面的 URL (1-10 .htm, 11-100 .psp)
list_page_urls = []
for i in range(1, 11):
    list_page_urls.append(f"{LIST_PAGE_BASE}list{i}.htm")
for i in range(11, MAX_INDEX_PAGES_TO_SCRAPE + 1):
    list_page_urls.append(f"{LIST_PAGE_BASE}list{i}.psp")
print(f"准备扫描 {len(list_page_urls)} 个索引页面...")

# 2. 从所有索引页面收集目标子页面的 URL 和日期
subpage_info_list = []
for index_url in tqdm(list_page_urls, desc="扫描索引页"):
    index_soup = get_soup(index_url)
    if index_soup:
        links_dates = extract_links_and_dates_from_index(index_soup, index_url)
        if links_dates:
            subpage_info_list.extend(links_dates)
    time.sleep(REQUEST_DELAY)  # 每个索引页请求后暂停

# 3. 去重，确保每个子页面只处理一次（保留第一次找到的日期）
unique_subpage_info = {}
for url, date_str in subpage_info_list:
    if url not in unique_subpage_info:
        unique_subpage_info[url] = date_str

print(f"\n从索引页找到 {len(unique_subpage_info)} 个符合条件的独立子页面链接。")
if not unique_subpage_info:
    print("未能找到任何符合条件的子页面链接，程序退出。")
    exit()

# 4. 遍历独立子页面，提取数据
all_final_data = []
print("开始爬取子页面并解析 meta description...")
for url, date_str in tqdm(unique_subpage_info.items(), desc="爬取子页面"):
    sub_soup = get_soup(url)
    if sub_soup:
        # *** 调用解析 meta 标签的函数 ***
        price_data_list = extract_price_data_from_meta(sub_soup, url)
        if price_data_list:
            for item in price_data_list:
                # *** 添加从索引页获取的日期 ***
                item['日期'] = date_str
                # item['Source URL'] = url # 可选：保留来源URL用于调试
                all_final_data.append(item)
    time.sleep(REQUEST_DELAY)  # 每个子页面请求后暂停

# 5. 处理并输出结果
if all_final_data:
    # 创建 Pandas DataFrame
    df = pd.DataFrame(all_final_data)

    # *** 整理 DataFrame 列名和顺序 ***
    if 'Vegetable' in df.columns and 'Price (元/斤)' in df.columns and '日期' in df.columns:
        df_final = df[['Vegetable', 'Price (元/斤)', '日期']]  # 选择并排序
        df_final = df_final.rename(columns={
            'Vegetable': '蔬菜',
            'Price (元/斤)': '单价'
        })  # 重命名为中文
    else:
        print("错误: 最终数据缺少必要的列。无法格式化输出。")
        print("找到的列:", df.columns if 'df' in locals() else "DataFrame 未创建")
        df_final = pd.DataFrame()  # 创建空的以避免后续错误

    print("\n--- 爬取完成 ---")

    if not df_final.empty:
        print(f"总共提取到 {len(df_final)} 条价格记录。")
        print("\n数据样本 (前5条):")
        print(df_final.head())
        # print("\n数据样本 (后5条):") # 可以取消注释查看末尾数据
        # print(df_final.tail())

        # --- 保存到文件 ---
        try:
            # 保存前检查文件是否存在
            if os.path.exists(OUTPUT_CSV):
                print(f"提示: 输出文件 {OUTPUT_CSV} 已存在，将被覆盖。")
            if os.path.exists(OUTPUT_EXCEL):
                print(f"提示: 输出文件 {OUTPUT_EXCEL} 已存在，将被覆盖。")

            # 保存为 CSV (使用 utf-8-sig 编码确保 Excel 正确显示中文)
            df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"\n数据已成功保存到 CSV 文件: {OUTPUT_CSV}")

            # 保存为 Excel (需要安装 openpyxl)
            try:
                df_final.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
                print(f"数据已成功保存到 Excel 文件: {OUTPUT_EXCEL}")
            except ImportError:
                print("\n提示: 未找到 'openpyxl' 库，无法保存为 Excel 文件。")
                print("请使用以下命令安装: pip install openpyxl")
            except Exception as e_excel:
                print(f"\n保存到 Excel 文件时出错: {e_excel}")

        except Exception as e_csv:
            print(f"\n保存到 CSV 文件时出错: {e_csv}")
    else:
        print("\n最终 DataFrame 为空，没有数据可保存。请检查之前的警告信息。")

else:
    print("\n未能从任何子页面提取到有效数据。")