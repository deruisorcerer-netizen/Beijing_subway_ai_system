import os
import json
import networkx as nx
import random

# ================= 配置路径 =================
INPUT_DIR = r"G:\Subway_ai_system\raw_lines"
OUTPUT_DIR = r"G:\Subway_ai_system\data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "subway_sft_pro.json")

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def build_subway_graph():
    G = nx.Graph()
    line_info = {} # 存储 线路 -> 站点列表
    
    # 1. 扫描所有线文件
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"检测到 {len(files)} 条线路文件，开始解析...")

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if not lines: continue
            
            # 根据 的格式解析第一行线路名
            # 例如: "1号线（含八通线）" -> "1号线（含八通线）"
            raw_line_name = lines[0]
            line_name = raw_line_name.split("] ")[-1] if "] " in raw_line_name else raw_line_name
            stations = lines[1:]
            
            line_info[line_name] = stations
            
            # 2. 构建图结构
            for i in range(len(stations)):
                current_station = stations[i]
                
                # 更新节点属性（记录所属线路）
                if G.has_node(current_station):
                    if line_name not in G.nodes[current_station]['lines']:
                        G.nodes[current_station]['lines'].append(line_name)
                else:
                    G.add_node(current_station, lines=[line_name])
                
                # 添加边（相邻车站）
                if i > 0:
                    prev_station = stations[i-1]
                    G.add_edge(prev_station, current_station, line=line_name)
                    
    return G, line_info

def generate_sft_json(G, line_info):
    sft_data = []
    all_stations = list(G.nodes())

    # --- A. 基础站点查询 (每站一条) ---
    for station in all_stations:
        lines = G.nodes[station]['lines']
        sft_data.append({
            "instruction": f"查询站点信息：{station}站属于哪条线？",
            "input": "",
            "output": f"{station}站是北京地铁网络中的一个站点，它所属的线路包括：{'、'.join(lines)}。"
        })

    # --- B. 线路组成查询 ---
    for line_name, stations in line_info.items():
        sft_data.append({
            "instruction": f"请列出北京地铁{line_name}的所有站点。",
            "input": "",
            "output": f"{line_name}共包含以下站点：{' -> '.join(stations)}。"
        })

    # --- C. 换乘逻辑 (自动识别度大于1的节点) ---
    transfer_stations = [n for n, d in G.nodes(data=True) if len(d['lines']) > 1]
    for ts in transfer_stations:
        lines = G.nodes[ts]['lines']
        sft_data.append({
            "instruction": f"在{ts}站可以换乘哪些线路？",
            "input": "",
            "output": f"{ts}站是一个换乘站，您可以解析在此换乘：{'、'.join(lines)}。"
        })

    # --- D. 最优路径规划 (利用 Dijkstra 算法生成 1000 组真实路径) ---
    print("正在生成路径规划样本...")
    for _ in range(1000):
        start, end = random.sample(all_stations, 2)
        try:
            path = nx.shortest_path(G, start, end)
            # 模拟更智能的回答风格
            path_str = " -> ".join(path)
            sft_data.append({
                "instruction": f"我想从{start}坐地铁去{end}，该怎么走？",
                "input": "",
                "output": f"建议乘坐方案如下：从【{start}】出发，依次经过 {path_str}，最终抵达【{end}】。请注意站内换乘广播。"
            })
        except nx.NetworkXNoPath:
            continue

    return sft_data

if __name__ == "__main__":
    subway_graph, subway_lines = build_subway_graph()
    final_dataset = generate_sft_json(subway_graph, subway_lines)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"🎉 处理完成！")
    print(f"总计解析线路: {len(subway_lines)}")
    print(f"生成 SFT 样本数: {len(final_dataset)}")
    print(f"文件保存至: {OUTPUT_FILE}")