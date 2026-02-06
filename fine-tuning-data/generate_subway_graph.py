import os
import json
import glob
import random

# ================= 配置区域 =================
INPUT_DIR = "raw_lines"  # 存放txt文件的文件夹
OUTPUT_FILE = "subway_sft_data.json" # 输出的SFT文件名

# 提问模板（增加数据的多样性）
PROMPT_TEMPLATES = [
    "请介绍一下北京地铁{station}站的线路和相邻站点信息。",
    "我在{station}，这里有几号线？下一站可以去哪？",
    "{station}站的通达情况如何？",
    "我想了解{station}站的换乘信息及邻站。"
]

# ================= 核心逻辑 =================

def parse_line_file(file_path):
    """解析单个线路txt文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip().split('\n')
    
    # 清洗掉空行
    content = [line.strip() for line in content if line.strip()]
    
    if not content:
        return None, []

    # 1. 获取线路名称 (处理 这种前缀)
    raw_title = content[0]
    # 如果有source标记，去掉它
    if "]" in raw_title:
        line_name = raw_title.split("]")[-1].strip()
    else:
        line_name = raw_title
    
    # 2. 获取站点列表
    stations = content[1:]
    
    return line_name, stations

def build_station_graph(data_dir):
    """
    构建站点图谱
    结构: {
        "西单": {
            "lines": ["1号线", "4号线"],
            "neighbors": {
                "1号线": ["复兴门", "天安门西"],
                "4号线": ["灵境胡同", "宣武门"]
            }
        }
    }
    """
    station_graph = {}
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    print(f"检测到 {len(txt_files)} 个线路文件，开始处理...")

    for file_path in txt_files:
        line_name, stations = parse_line_file(file_path)
        if not stations:
            continue
            
        print(f"正在处理: {line_name} ({len(stations)}站)")

        for i, station in enumerate(stations):
            if station not in station_graph:
                station_graph[station] = {
                    "lines": [],
                    "neighbors": {}
                }
            
            # 记录线路
            if line_name not in station_graph[station]["lines"]:
                station_graph[station]["lines"].append(line_name)
            
            # 记录相邻站
            # 前一站
            if i > 0:
                prev_station = stations[i-1]
                if line_name not in station_graph[station]["neighbors"]:
                    station_graph[station]["neighbors"][line_name] = []
                station_graph[station]["neighbors"][line_name].append(prev_station)
            
            # 后一站
            if i < len(stations) - 1:
                next_station = stations[i+1]
                if line_name not in station_graph[station]["neighbors"]:
                    station_graph[station]["neighbors"][line_name] = []
                station_graph[station]["neighbors"][line_name].append(next_station)

    return station_graph

def generate_sft_dataset(station_graph):
    """将图谱转换为SFT JSON格式"""
    dataset = []

    for station, info in station_graph.items():
        # 构造回答 (Answer)
        lines_str = "、".join(info["lines"])
        
        neighbor_texts = []
        for line, neighbors in info["neighbors"].items():
            # 去重并格式化相邻站
            unique_neighbors = sorted(list(set(neighbors)))
            n_str = "，".join(unique_neighbors)
            neighbor_texts.append(f"- {line}：可前往 {n_str}")
        
        neighbor_block = "\n".join(neighbor_texts)

        # 组合标准回答
        answer = (
            f"{station}站目前有 {len(info['lines'])} 条线路经过，分别是：{lines_str}。\n\n"
            f"相邻站点详情如下：\n{neighbor_block}"
        )

        # 构造训练样本 (生成 2 个不同问法的样本，增强鲁棒性)
        # 随机取样 2 个模板，如果不够就全取
        selected_templates = random.sample(PROMPT_TEMPLATES, min(2, len(PROMPT_TEMPLATES)))
        
        for tmpl in selected_templates:
            question = tmpl.format(station=station)
            
            entry = {
                "conversations": [
                    {
                        "from": "user",
                        "value": question
                    },
                    {
                        "from": "assistant",
                        "value": answer
                    }
                ]
            }
            dataset.append(entry)
    
    return dataset

# ================= 执行入口 =================

if __name__ == "__main__":
    # 1. 检查目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误：请创建一个名为 '{INPUT_DIR}' 的文件夹，并将txt文件放入其中。")

    # 2. 构建图谱
    graph = build_station_graph(INPUT_DIR)

    # 3. 生成数据集
    sft_data = generate_sft_dataset(graph)

    # 4. 保存文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\n成功！已生成 {len(sft_data)} 条SFT样本，保存在 {OUTPUT_FILE}")
    print("样本示例：")
    print(json.dumps(sft_data[0], ensure_ascii=False, indent=2))