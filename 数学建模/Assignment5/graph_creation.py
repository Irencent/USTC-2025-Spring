import matplotlib.pyplot as plt
import math
import networkx as nx
from matplotlib import font_manager
import matplotlib as mpl
from scipy.spatial.distance import euclidean
# 设置字体为 Pingfang SC（适用于 macOS）
font_path = "/Volumes/Macintosh HD/System/Library/Fonts/STHeiti Medium.ttc"
prop = font_manager.FontProperties(fname=font_path)

# 设置字体为自定义的中文字体
mpl.rcParams['font.sans-serif'] = [prop.get_name()]
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 在图中强制加入必须经过的路径
def enforce_must_visit_paths(graph, must_visit_paths):
    for path in must_visit_paths:
        node1, node2, path_name = path
        # 为强制路径设定较小的权重（或者特殊的惩罚机制）
        graph[node1][node2]["weight"] = 0.1  # 强制路径的权重更小

# 初始化无向图
campus_graph = nx.Graph()

# 构建地点字典和路径数据（你可以直接使用之前提供的 locations 和 path_data）
# 最终修正版节点数据
locations = {
    # === 大门 (坐标已根据图片更新) ===
    "G_N": {"name_cn": "北门", "name_en": "North Gate", "coords": (419, 853)},
    "G_W": {"name_cn": "西门", "name_en": "West Gate", "coords": (85, 544)},
    "G_E": {"name_cn": "东门", "name_en": "East Gate", "coords": (944, 386)},
    "G_S": {"name_cn": "南门", "name_en": "South Gate", "coords": (534, 132)}, 

    # === 建筑 (B1-B20, 坐标已根据图片更新, 已添加 B13) ===
    "B1": {"name_cn": "1号学生公寓", "name_en": "Student Dormitory 1", "coords": (450, 194)},
    "B2": {"name_cn": "2号学生公寓", "name_en": "Student Dormitory 2", "coords": (263, 297)},
    "B3": {"name_cn": "3号学生公寓", "name_en": "Student Dormitory 3", "coords": (262, 441)},
    "B4": {"name_cn": "4号学生公寓", "name_en": "Student Dormitory 4", "coords": (263, 192)},
    "B5": {"name_cn": "5号学生公寓", "name_en": "Student Dormitory 5", "coords": (187, 195)},
    "B6": {"name_cn": "6号学生公寓", "name_en": "Student Dormitory 6", "coords": (90, 374)},
    "B7": {"name_cn": "7号学生公寓", "name_en": "Student Dormitory 7", "coords": (187, 439)},
    "B8": {"name_cn": "8号学生公寓", "name_en": "Student Dormitory 8", "coords": (302, 543)},
    "B9": {"name_cn": "师生服务中心", "name_en": "Student-Teacher Service Center", "coords": (439, 135)},
    "B10": {"name_cn": "学生食堂", "name_en": "Dining Hall", "coords": (332, 392)},
    "B11": {"name_cn": "体育场", "name_en": "Athletic Field", "coords": (340, 604)},
    "B12": {"name_cn": "体育馆", "name_en": "The Gymnasium", "coords": (268, 604)},
    "B13": {"name_cn": "教工食堂", "name_en": "Staff Canteen", "coords": (479, 853)},
    "B14": {"name_cn": "教工公寓", "name_en": "Staff Dormitory", "coords": (520, 599)},
    "B15": {"name_cn": "行政服务中心", "name_en": "Administration Center", "coords": (730, 790)},
    "B16": {"name_cn": "图书教育中心", "name_en": "Library & Edu Service Center", "coords": (450, 392)},
    "B17": {"name_cn": "1号学科楼", "name_en": "Academic Building 1", "coords": (736, 589)},
    "B18": {"name_cn": "信息科学技术大楼", "name_en": "School of Info Science & Tech", "coords": (740, 384)},
    "B19": {"name_cn": "2号学科楼", "name_en": "Academic Building 2", "coords": (744, 280)},
    "B20": {"name_cn": "3号学科楼", "name_en": "Academic Building 3", "coords": (534, 289)},

    # === 交叉路口 (坐标已根据图片更新) ===
    "I1": {"name_cn": "交叉口", "name_en": "Intersection", "coords": (146, 144)},
    "I2": {"name_cn": "交叉口", "name_en": "Intersection", "coords": (81, 861)},
    "I3": {"name_cn": "交叉口", "name_en": "Intersection", "coords": (262, 855)},
    "I4": {"name_cn": "交叉口", "name_en": "Intersection", "coords": (920, 846)},
    "I5": {"name_cn": "交叉口", "name_en": "Intersection", "coords": (922, 531)},
    "I6": {"name_cn": "交叉口", "name_en": "Intersection", "coords": (920, 275)},
    "I7": {"name_cn": "交叉口", "name_en": "Intersection", "coords": (922, 127)},
}

# Path data
# 最终修正版路径数据
_path_data = [
    # --- 主干道 ---
    ("I2", "I3", "North Zhuoyue Road"), ("I3", "G_N", "North Zhuoyue Road"), ("G_N", "B13", "North Zhuoyue Road"), ("B13", "I4", "North Zhuoyue Road"),
    ("I1", "B6", "West Zhuoyue Road"), ("B6", "G_W", "West Zhuoyue Road"), ("G_W", "I2", "West Zhuoyue Road"), 
    ("I1", "B9", "South Zhuoyue Road"), ("B9", "G_S", "South Zhuoyue Road"), ("I7", "G_S", "South Zhuoyue Road"),
    ("I4", "I5", "East Zhuoyue Road"), ("I5", "G_E", "East Zhuoyue Road"), ("G_E", "I6", "East Zhuoyue Road"), ("I6", "I7", "East Zhuoyue Road"),
    ("I3", "B11", "Zhaojiuzhang Road"), ("B11", "B10", "Zhaojiuzhang Road"), ("B10", "B1", "Zhaojiuzhang Road"), ("B20", "B1", "Zhaojiuzhang Road"),
    ("B20", "B19", "Zhaozhongyao Road"), ("B19", "I6", "Zhaozhongyao Road"),
    ("G_N", "B14", "Hualuogeng Road"), ("B16", "B14", "Hualuogeng Road"), ("B16", "B20", "Hualuogeng Road"), ("B20", "G_S", "Hualuogeng Road"),
    ("B12", "B11", "Yongheng Road"), ("B14", "B11", "Yongheng Road"), ("B14", "B17", "Yongheng Road"), ("B15", "B17", "Yongheng Road"),
    ("B10", "B16", "Sipei Bridge"),
    ("G_W", "B8", "Xiapeisu Road"),
    
    # --- 支路与建筑连接路径 ---
    ("B6", "B7", "UR"), ("B7", "B3", "UR"), ("B5", "B7", "UR"), ("B3", "B8", "UR"), 
    ("B3", "B2", "UR"), ("B3", "B10", "UR"), ("B2", "B10", "UR"), ("B2", "B5", "UR"), 
    ("B5", "B4", "UR"), ("B1", "B4", "UR"), ("B1", "B9", "UR"), ("B1", "B2", "UR"), 
    ("B15", "B17", "UR"), ("B18", "B17", "UR"), ("B18", "B19", "UR"), 
]

# --- 创建图谱 ---
# 1. 初始化一个空的无向图
G = nx.Graph()

# 2. 向图中添加节点，并赋予节点属性（名称和坐标）
for node_id, attrs in locations.items():
    G.add_node(node_id, **attrs)

# 3. 向图中添加边，并赋予边属性（名称和权重/长度）
for start_id, end_id, name in _path_data:
    p1 = locations[start_id]["coords"]
    p2 = locations[end_id]["coords"]
    distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    G.add_edge(start_id, end_id, name=name, weight=round(distance, 2))

print("图谱创建成功！")
print(f"图中共有 {G.number_of_nodes()} 个节点。")
print(f"图中共有 {G.number_of_edges()} 条边。")


# --- 可视化图谱 ---
# 4. 绘制图谱以进行验证
plt.figure(figsize=(16, 10))

# 获取所有节点的坐标用于绘图
pos = nx.get_node_attributes(G, 'coords')

# 绘制节点、边和标签
nx.draw_networkx_nodes(G, pos, node_size=150, node_color='skyblue')
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

# 设置中文字体，以防标签显示为方块（请确保你的系统中有这个字体）
plt.rcParams['font.sans-serif'] = ['Heiti TC'] # 或者 'SimHei', 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

plt.title("中国科学技术大学高新园区 - 完整校园图谱", size=18)
plt.tight_layout()
plt.show()

# --- 步骤 2: 识别必需的游览组件 ---

# 1. 定义必须访问的20个地标节点的ID
V_req = [f'B{i}' for i in range(1, 21)]

# 2. 定义必须走完的7条主要道路的名称关键字
#    使用关键字可以匹配到所有路段（例如 "Zhuoyue" 会匹配 "North Zhuoyue Road" 等）
required_road_names = [
    "Zhuoyue",      # 卓越路 (东西南北)
    "Hualuogeng",   # 华罗庚路
    "Zhaojiuzhang", # 赵九章路
    "Zhaozhongyao", # 赵忠尧路
    "Xiapeisu",     # 夏培肃路
    "Yongheng",     # 永恒路
    "Sipei",        # 思佩桥
]

# 3. 从完整图谱 G 中提取出所有必需的边
E_req = []
for u, v, data in G.edges(data=True):
    # 检查边的名称是否包含任何一个必需道路的关键字
    if any(road_name in data.get('name', '') for road_name in required_road_names):
        E_req.append((u, v))

print("\n--- 步骤 2: 必需组件识别结果 ---")
print(f"必须访问的节点数量: {len(V_req)} 个")
print(f"必须走完的边（路段）数量: {len(E_req)} 个")


# --- 可选：可视化必需的组件 ---
# 这可以帮助我们直观地检查“任务清单”是否正确
plt.figure(figsize=(16, 10))
plt.title("必需的游览组件 (红色高亮)", size=18)

# 1. 绘制完整的校园图谱作为底图 (灰色)
pos = nx.get_node_attributes(G, 'coords')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray')
nx.draw_networkx_edges(G, pos, edge_color='lightgray')

# 2. 高亮绘制必需的节点 (红色)
nx.draw_networkx_nodes(G, pos, nodelist=V_req, node_size=150, node_color='red')

# 3. 高亮绘制必需的边 (红色)
nx.draw_networkx_edges(G, pos, edgelist=E_req, edge_color='red', width=2.0)

# 绘制所有节点的标签以便参考
nx.draw_networkx_labels(G, pos, font_size=8)
plt.show()

# --- 步骤 3: 连接必需的组件 ---

# 1. 创建一个仅包含必需组件的临时图，用于寻找连通分量
#    这个图包含所有必需的边，以及所有必需的节点
temp_G = nx.Graph(E_req)
temp_G.add_nodes_from(V_req)

# 2. 找出所有独立的“岛屿”（连通分量）
components = list(nx.connected_components(temp_G))

print(f"\n--- 步骤 3: 连接组件结果 ---")
print(f"在连接之前，必需的组件被分成了 {len(components)} 个独立的'岛屿'。")

connecting_edges = []
if len(components) > 1:
    # 3. 如果存在多个岛屿，则需要将它们连接起来
    #    按组件大小排序，将最大的作为“主大陆”
    components.sort(key=len, reverse=True)
    main_component = components[0]
    other_components = components[1:]

    # 4. 遍历所有小岛屿，找到连接到主大陆的最短路径
    for component in other_components:
        min_dist = float('inf')
        best_path = None

        # 寻找最短的“桥梁”
        for u in component:
            for v in main_component:
                # 在完整的校园图G中计算实际最短路径
                dist = nx.shortest_path_length(G, source=u, target=v, weight='weight')
                if dist < min_dist:
                    min_dist = dist
                    best_path = nx.shortest_path(G, source=u, target=v, weight='weight')
        
        print(f"找到一条连接“岛屿”的路径，长度为: {min_dist:.2f}")

        # 将最短路径（桥梁）转换为边的列表，并添加到 connecting_edges 中
        path_edges = list(zip(best_path[:-1], best_path[1:]))
        connecting_edges.extend(path_edges)

# E_req_connected 是包含了原始必需边和新增连接边的完整列表
E_req_connected = E_req + connecting_edges
print(f"为了连接所有组件，新增了 {len(connecting_edges)} 段连接路径。")


# --- 可选：可视化连接后的网络 ---
plt.figure(figsize=(16, 10))
plt.title("连接后的必需网络 (橙色为新增的连接路径)", size=18)

# 1. 绘制完整的校园图谱作为底图 (灰色)
pos = nx.get_node_attributes(G, 'coords')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray')
nx.draw_networkx_edges(G, pos, edge_color='lightgray')

# 2. 绘制原始的必需组件 (红色)
nx.draw_networkx_nodes(G, pos, nodelist=V_req, node_size=150, node_color='red')
nx.draw_networkx_edges(G, pos, edgelist=E_req, edge_color='red', width=2.0)

# 3. 高亮绘制新增的连接路径 (橙色)
nx.draw_networkx_edges(G, pos, edgelist=connecting_edges, edge_color='orange', width=2.5, style='dashed')

# 绘制所有节点的标签以便参考
nx.draw_networkx_labels(G, pos, font_size=8)
plt.tight_layout()
plt.show()

# --- 步骤 4: 求解邮递员问题 ---

# 1. 基于连接后的必需边，创建一个新的“多重图”(MultiGraph)
G_req_connected = nx.MultiGraph(E_req_connected)

# 2. 找出这个网络中所有度为奇数的节点
odd_degree_nodes = [n for n, d in G_req_connected.degree() if d % 2 != 0]

print(f"\n--- 步骤 4: 邮递员问题求解结果 ---")
print(f"必需网络中有 {len(odd_degree_nodes)} 个奇数度节点: {odd_degree_nodes}")

# 3. 在这些奇数度节点之间，计算所有可能的配对，并以它们在完整校园图G中的最短距离为权重
#    创建一个新的、完全连接的图，用于寻找最优匹配
G_odd_complete = nx.Graph()
for i in range(len(odd_degree_nodes)):
    for j in range(i + 1, len(odd_degree_nodes)):
        u = odd_degree_nodes[i]
        v = odd_degree_nodes[j]
        dist = nx.shortest_path_length(G, source=u, target=v, weight='weight')
        G_odd_complete.add_edge(u, v, weight=dist)

# 4. 找到能使总权重最小的完美匹配方案
#    这个匹配方案就告诉我们应该如何将奇数度节点两两配对，以最高效率“消除”它们
min_matching = nx.min_weight_matching(G_odd_complete, weight='weight')

print(f"计算出的最优配对方案是: {min_matching}")

# 5. 根据最优配对方案，生成必需的“重复路线”（Deadheading Paths）
deadhead_edges = []
for u, v in min_matching:
    path = nx.shortest_path(G, source=u, target=v, weight='weight')
    path_edges = list(zip(path[:-1], path[1:]))
    deadhead_edges.extend(path_edges)

deadhead_length = sum(G[u][v]['weight'] for u, v in deadhead_edges)
print(f"为实现最优游览，需要重复走的总路线长度为: {deadhead_length:.2f}")

# --- 可选：可视化重复路线 ---
plt.figure(figsize=(16, 10))
plt.title("最终游览网络 (蓝色虚线为必须重复走的路线)", size=18)

# 1. 绘制完整的校园图谱作为底图 (灰色)
pos = nx.get_node_attributes(G, 'coords')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray')
nx.draw_networkx_edges(G, pos, edge_color='lightgray')

# 2. 绘制原始的必需组件 (红色)
nx.draw_networkx_nodes(G, pos, nodelist=V_req, node_size=150, node_color='red')
nx.draw_networkx_edges(G, pos, edgelist=E_req, edge_color='red', width=2.0)

# 3. 绘制连接路径 (橙色)
nx.draw_networkx_edges(G, pos, edgelist=connecting_edges, edge_color='orange', width=2.5, style='dashed')

# 4. 绘制计算出的重复路线 (蓝色)
nx.draw_networkx_edges(G, pos, edgelist=deadhead_edges, edge_color='blue', width=2.5, style='dashed')

# 绘制所有节点的标签以便参考
nx.draw_networkx_labels(G, pos, font_size=8)
plt.tight_layout()
plt.show()

# --- 步骤 5: 整合并分析最终路线 ---

# 1. 创建一个最终的“多重图”(MultiGraph)，因为最优路线可能包含重复走过的路段
G_final_tour = nx.MultiGraph()
# 添加所有必需的、连接的和重复的边
all_tour_edges = E_req_connected + deadhead_edges
for u, v in all_tour_edges:
    G_final_tour.add_edge(u, v, weight=G[u][v]['weight'])

# 2. 验证最终的图是否是“欧拉图”（所有节点的度都为偶数）
is_eulerian = all(d % 2 == 0 for n, d in G_final_tour.degree())
print(f"\n--- 步骤 5: 最终路线分析结果 ---")
print(f"最终的游览网络是否是欧拉图? {is_eulerian}")

# 3. 计算最终路线的总长度
total_length = G_final_tour.size(weight='weight')
print(f"恭喜！计算出的最短游览总用时（距离）为: {total_length:.2f}")

# 4. 生成具体的游览路线序列
#    我们可以选择一个方便的地点作为起点，比如南门 'G_South'
start_node = 'G_S'
tour_edges = list(nx.eulerian_circuit(G_final_tour, source=start_node))

# 将边的序列转换为节点的访问序列
tour_nodes = [tour_edges[0][0]] + [v for u, v in tour_edges]

print(f"路线起点: {locations[start_node]['name_cn']}")
print("建议的游览顺序 (前20步):")
for i in range(min(20, len(tour_nodes))):
    node_id = tour_nodes[i]
    node_name = locations[node_id]['name_cn']
    print(f"  第{i+1}步: {node_name} ({node_id})")

# 5. 最终的可视化与步骤4相同，因为它最清晰地展示了路线的构成
plt.figure(figsize=(16, 10))
plt.title("最终游览路线构成图 (红:必需, 橙:连接, 蓝:重复)", size=18)
pos = nx.get_node_attributes(G, 'coords')
# 底图
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray')
nx.draw_networkx_edges(G, pos, edge_color='lightgray')
# 必需组件
nx.draw_networkx_nodes(G, pos, nodelist=V_req, node_size=150, node_color='red')
nx.draw_networkx_edges(G, pos, edgelist=E_req, edge_color='red', width=2.0)
# 连接路径
nx.draw_networkx_edges(G, pos, edgelist=connecting_edges, edge_color='orange', width=2.5, style='dashed')
# 重复路线
nx.draw_networkx_edges(G, pos, edgelist=deadhead_edges, edge_color='blue', width=2.5, style='dashed')
# 标签
nx.draw_networkx_labels(G, pos, font_size=8)

plt.tight_layout()
plt.show()


