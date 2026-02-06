# generate_rationale.py
# 用法示例（在 Windows powershell/cmd 中）:
#   python generate_rationale.py --raw_lines_dir "G:\\Subway_ai_system\\raw_lines" --out "G:\\Subway_ai_system\\generated_rationale.jsonl"
#
# 该脚本会：
# - 从 raw_lines_dir 下读取所有 .txt 文件来构建线路图（每个 .txt 首行为线路名，接下来每行一个站名）
# - 可选读取 graph.json / subway_sft_pro.json (路径在同目录) 做对比与抽取 pairs
# - 输出 Alpaca 格式 jsonl，每行一个 {"instruction","input","output"}，output 包含详细的推导（rationale）和最终路线

import os
import json
import argparse
from collections import defaultdict, deque
from types import SimpleNamespace
import re
import unicodedata
# 在脚本顶部 import
import heapq

def augment_edge_lines_from_lines(lines_dict, edge_lines):
    """
    确保 edge_lines 对于每个相邻站对都被记录（从 lines_dict 逐条扫描）
    lines_dict: {line_name: [station1,station2,...]}
    edge_lines: existing dict (a,b) -> set(lines)
    """
    added = 0
    for line, stations in lines_dict.items():
        for i in range(len(stations)-1):
            a = stations[i]; b = stations[i+1]
            key = tuple(sorted((a,b)))
            if line not in edge_lines.get(key, set()):
                edge_lines.setdefault(key, set()).add(line)
                added += 1
    # 可选打印
    if added:
        print(f"augment_edge_lines: 为 edge_lines 新增了 {added} 条边的线路标注。")
    return edge_lines

def build_expanded_graph(station_lines, edge_lines, default_travel_time=2.0):
    """
    构建扩展图：节点为 (station, line) 表示在某站且乘坐某条线到达该站。
    同站不同 line 之间有换乘边（权重 = transfer_penalty，后续在 dijkstra 中加入）。
    edge_lines: dict (a,b) sorted tuple -> set(lines)
    station_lines: dict station -> set(lines)
    返回:
      adj: dict node -> list of (neighbor_node, weight, edge_info)
    说明：
      - 对于相邻站 a-b，如果 edge_lines[(a,b)] 包含 lineL，那么我们在 (a,lineL) <-> (b,lineL) 上加 travel_time 边
      - 另外，为了支持起点/终点可能不指定 line，后续可把虚拟起点/终点连接到所有相关 (station,line)
    """
    adj = defaultdict(list)
    # add edges for line-contiguous travel
    for (a,b), lines in edge_lines.items():
        for line in lines:
            u = (a, line)
            v = (b, line)
            # travel time default (可后续扩展按区间具体值)
            t = default_travel_time
            adj[u].append((v, t, {"type":"ride","line":line, "edge":(a,b)}))
            adj[v].append((u, t, {"type":"ride","line":line, "edge":(a,b)}))
    # add transfer edges at same station between different lines with zero weight here;
    # we will add transfer penalty in dijkstra when switching lines, or we can explicitly add edges with penalty.
    # For simplicity, add transfer edges with weight 0 here and handle penalty when expanding origin/neighbor.
    for station, lines in station_lines.items():
        lines_list = list(lines)
        for i in range(len(lines_list)):
            for j in range(i+1, len(lines_list)):
                li = lines_list[i]; lj = lines_list[j]
                u = (station, li); v = (station, lj)
                # We add transfer edges with zero here; penalty added in dijkstra optionally
                adj[u].append((v, 0.0, {"type":"transfer","line_from":li,"line_to":lj}))
                adj[v].append((u, 0.0, {"type":"transfer","line_from":lj,"line_to":li}))
    return adj

def dijkstra_with_transfer(neighbors, station_lines, edge_lines, start, goal,
                           default_travel_time=2.0, transfer_penalty=4.0):
    """
    基于扩展图的 Dijkstra，返回按分钟最短路径的站点序列（若无法到达返回 None）。
    参数说明：
      neighbors: 原始邻接集合（未直接使用，但可作检查）
      station_lines: station -> set(lines)
      edge_lines: (a,b) -> set(lines)
      start/goal: 站名字符串
      default_travel_time: 每站间默认耗时（分钟）
      transfer_penalty: 换乘惩罚（分钟）
    思路：
      - 扩展节点为 (station, line)
      - 起点连接到所有 (start,line) 起点候选，初始 cost 0
      - 当在 Dijkstra 中从 (s,line1) 走到 (s2,line2):
          * 如果 line1 == line2 and it's a ride edge -> cost += travel_time
          * If switching lines at same station -> cost += transfer_penalty
    返回 station path (list of stations)。
    """
    # build expanded adjacency
    adj = build_expanded_graph(station_lines, edge_lines, default_travel_time=default_travel_time)

    # heap: (cost, node, prev_node)
    heap = []
    dist = {}
    prev = {}
    visited = set()

    # push all possible start-line nodes
    start_lines = station_lines.get(start, set()) or {None}
    goal_lines = station_lines.get(goal, set()) or {None}

    # special handling if start==goal
    if start == goal:
        return [start]

    # Initialize distances for nodes corresponding to start
    start_nodes = []
    for ln in station_lines.get(start, []):
        node = (start, ln)
        heapq.heappush(heap, (0.0, node, None))
        dist[node] = 0.0
        prev[node] = None

    # If start has no line labels (rare), fail
    if not heap:
        return None

    while heap:
        cost, node, p = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        prev[node] = p
        station, cur_line = node

        # check goal (any node with station==goal)
        if station == goal:
            # reconstruct path in expanded nodes then map to stations
            # find the node in prev chain
            path_nodes = []
            cur = node
            while cur is not None:
                path_nodes.append(cur)
                cur = prev.get(cur)
            path_nodes.reverse()
            # collapse to station list (remove consecutive duplicates)
            station_path = []
            for st, ln in path_nodes:
                if not station_path or station_path[-1] != st:
                    station_path.append(st)
            return station_path

        # relax neighbors
        for (nbr, w, info) in adj.get(node, []):
            nbr_station, nbr_line = nbr
            add_cost = 0.0
            if info.get("type") == "ride":
                # ride along same line: w accounts for travel time
                add_cost = w
            elif info.get("type") == "transfer":
                # transfer between lines: penalize
                add_cost = transfer_penalty
            else:
                add_cost = w
            new_cost = cost + add_cost
            if new_cost < dist.get(nbr, float('inf')):
                dist[nbr] = new_cost
                prev[nbr] = node
                heapq.heappush(heap, (new_cost, nbr, node))
    return None

# 示例：如何在原脚本中调用替代 bfs_shortest_path
# 替换点：path = bfs_shortest_path(neighbors, a, b)
# 改成：
# path = dijkstra_with_transfer(neighbors, station_lines, edge_lines, a, b,
#                               default_travel_time=2.0, transfer_penalty=4.0)

def normalize_name(s):
    """统一字符串：去 BOM、trim、NFKC（把全角转半角）、压缩空白。"""
    if s is None:
        return s
    s = s.strip().lstrip("\ufeff").rstrip("\u200b")
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def contains_chinese(s):
    """判断字符串里是否含有中文汉字字符（基本汉字区）。"""
    if not s:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', s))

def extract_line_number_from_filename(fname):
    """
    从文件名尝试提取线路数字，例如 line_1.txt -> 1, Line-2.txt -> 2, 3_line.txt -> 3
    返回字符串数字或 None
    """
    bn = os.path.basename(fname)
    m = re.search(r'(?i)(?:line[_\-\s]*|^)(\d{1,2})(?:\D|$)', bn)
    if m:
        return m.group(1)
    # 备选：直接找第一个独立数字
    m2 = re.search(r'(\d{1,2})', bn)
    return m2.group(1) if m2 else None

# 可扩展的英文到中文线路名映射（按需补充）
english_to_chinese_map = {
    # 常见示例：文件名或首行含 airport/CapitalAirport -> 首都机场线
    "airport_capital": "首都机场线",
    "airport_daxing": "大兴机场线",
    "changping":"昌平线",
    "fangshan":"房山线",
    "yizhuang":"亦庄线",
    "line 1": "1号线",
    "line1": "1号线",
    "line 2": "2号线",
    # 根据你实际文件名补充更多映射
}

def map_english_to_chinese(text):
    """尝试把英文短语映射为中文线路名（小写匹配）"""
    if not text:
        return None
    t = text.lower().strip()
    # 精确匹配 map 键
    if t in english_to_chinese_map:
        return english_to_chinese_map[t]
    # 含键即可匹配（例如文件名里包含 'airport'）
    for k, v in english_to_chinese_map.items():
        if k in t:
            return v
    return None
def load_lines_from_txt(folder):
    """
    读取目录下所有 .txt 文件，解析为 {line_name: [station1, station2, ...]}
    规则：
      - 读取每个 txt，第一行优先作为线路名（先 normalize）
      - 如果第一行包含中文 -> 直接用作线路名
      - 否则尝试用文件名推断如 'line_1' -> '1号线'，或用 english_to_chinese_map 映射
      - 如果都失败，仍使用文件名（去扩展名）作为线路名（normalize 后）
    """
    lines = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.txt'):
            continue
        fp = os.path.join(folder, fname)
        with open(fp, 'r', encoding='utf-8') as f:
            raw_rows = [r.rstrip("\n\r") for r in f.readlines()]
        # 去空行并 normalize each line
        rows = [normalize_name(r) for r in raw_rows if r and r.strip()]
        if not rows:
            continue

        first = rows[0]
        # 规范化文件名（无扩展名）做候选
        file_base = os.path.splitext(os.path.basename(fname))[0]
        file_base_norm = normalize_name(file_base)

        chosen_line_name = None
        # 1) 如果首行含中文，直接采用
        if contains_chinese(first):
            chosen_line_name = first
        else:
            # 2) 尝试 english->chinese map（匹配首行）
            mapped = map_english_to_chinese(first)
            if mapped:
                chosen_line_name = mapped
            else:
                # 3) 尝试用文件名去匹配（先 map，再 attempt number）
                mapped_fn = map_english_to_chinese(file_base_norm)
                if mapped_fn:
                    chosen_line_name = mapped_fn
                else:
                    num = extract_line_number_from_filename(file_base_norm)
                    if num:
                        chosen_line_name = f"{num}号线"
                    else:
                        # 4) 退回：使用首行（英文）或文件名作为线路名（normalized）
                        #    仍保持首行原样（方便你 later manual mapping）
                        chosen_line_name = first or file_base_norm

        # 最后规范化站名列表
        stations = [normalize_name(s) for s in rows[1:]] if len(rows) > 1 else []
        # store with chosen_line_name normalized as well
        chosen_line_name = normalize_name(chosen_line_name)
        lines[chosen_line_name] = stations
    return lines

def build_graph_from_lines(lines_dict):
    """
    构建无向图：
    nodes: station name (string)
    edges: adjacency with lines info stored
    返回:
      neighbors: dict station -> set(neighbor stations)
      station_lines: dict station -> set(lines)
      edge_lines: dict (a,b) sorted tuple -> set(lines)
    """
    neighbors = defaultdict(set)
    station_lines = defaultdict(set)
    edge_lines = defaultdict(set)
    for line, stations in lines_dict.items():
        for i, s in enumerate(stations):
            station_lines[s].add(line)
            if i > 0:
                a = stations[i-1]; b = s
                neighbors[a].add(b); neighbors[b].add(a)
                edge_lines[tuple(sorted((a,b)))].add(line)
    return neighbors, station_lines, edge_lines

def bfs_shortest_path(neighbors, start, goal):
    """
    无权最短路径（按站数）。返回站点列表（含 start,goal），若无路径返回 None。
    """
    if start == goal:
        return [start]
    if start not in neighbors or goal not in neighbors:
        return None
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        for nb in neighbors[cur]:
            if nb not in prev:
                prev[nb] = cur
                if nb == goal:
                    # 回溯路径
                    path = [goal]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(nb)
    return None
def choose_lines_for_path(path, edge_lines, station_lines, lines_dict=None):
    """
    更稳健的分段算法：
    - 在每个起点 i，先收集 candidate_lines = union(edge_lines for edges starting at i)
    - 对每个 candidate 计算它能连续覆盖 path[i..j] 的最大 j（coverage）
    - 选择 coverage 最大的 candidate 作为当前段的线路；若没有 candidate，再退回 station_lines
    - 返回 [{'line': chosen_line_or_None, 'stations': [...]}, ...]
    
    给出站点路径，尝试把路径分段并标注使用哪条线，以减少换乘（贪心选取连续段上的同一条线）
    返回 list of segments: [{'line': line_name, 'stations': [s_i...s_j]}]
    算法（贪心）:
      - 从 path[0] 开始，尝试在后续边上寻找一条线能覆盖尽可能长的连续区间（即对每相邻边取交集），
        当交集为空时结束当前段并开始新段。
    """
    segments = []
    n = len(path)
    i = 0
    while i < n-1:
        # collect candidate lines from upcoming edges (small horizon to avoid huge union)
        candidate_lines = set()
        # look ahead up to remaining edges (or cap to e.g. 20)
        max_look = min(n - 1 - i, 50)
        for k in range(i, i + max_look):
            ekey = tuple(sorted((path[k], path[k+1])))
            candidate_lines.update(edge_lines.get(ekey, set()))
        # If still empty and we have lines_dict, attempt to scan each line for adjacency to find candidates
        if not candidate_lines and lines_dict:
            for ln, sts in lines_dict.items():
                # simple adjacency check within each line
                for idx in range(len(sts)-1):
                    if (sts[idx] == path[i] and sts[idx+1] == path[i+1]) or (sts[idx] == path[i+1] and sts[idx+1] == path[i]):
                        candidate_lines.add(ln)
                        break

        # if still empty, fallback to station_lines of current station
        if not candidate_lines:
            candidate_lines = set(station_lines.get(path[i], []))

        # If after all attempts candidate_lines empty, we cannot identify a line for this edge
        if not candidate_lines:
            # create a single-segment with None line for this one edge and move on
            segments.append({'line': None, 'stations': path[i:i+2]})
            i += 1
            continue

        # For each candidate, compute how far it can cover continuously from i
        best_line = None
        best_j = i + 1  # at least covers edge i-(i+1)
        for cand in sorted(candidate_lines):  # deterministic order
            j = i + 1
            while j < n:
                ekey = tuple(sorted((path[j-1], path[j])))
                # if this edge is served by cand, extend; else stop
                if cand in edge_lines.get(ekey, set()):
                    j += 1
                else:
                    break
            # j is exclusive index where coverage stops; coverage length = j - i
            if j > best_j:
                best_j = j
                best_line = cand

        # If no candidate extended beyond single edge, pick deterministic candidate (sorted)
        if best_line is None:
            best_line = sorted(candidate_lines)[0] if candidate_lines else None
            seg_end = i + 1
        else:
            seg_end = best_j - 1  # seg covers stations i..seg_end inclusive

        # build segment stations range: path[i:seg_end+1]
        seg_stations = path[i:seg_end+1]
        segments.append({'line': best_line, 'stations': seg_stations})
        i = seg_end
    return segments


def format_rationale(instr_from, instr_to, path, segments, station_lines):
    """
    生成 output 字符串，包含思路分步（rationale）和最终路线（Final Answer）。
    """
    if path is None:
        return f"思路：我先检查图中是否存在从 {instr_from} 到 {instr_to} 的连通路径。经搜索，未找到可达路径。\n\nFinal Answer:\n无法从 {instr_from} 到达 {instr_to}（图中无连通路径或站名输入错误）。"
    lines_from = ", ".join(sorted(station_lines.get(instr_from, []))) or "未知线路"
    lines_to = ", ".join(sorted(station_lines.get(instr_to, []))) or "未知线路"
    s = []
    s.append(f"思路：")
    s.append(f"1) 查询起点与终点所在线路：{instr_from} 在 {lines_from}；{instr_to} 在 {lines_to}。")
    s.append(f"2) 在站点图上计算无权最短路径（按换乘/站点数最少的近似策略）：")
    s.append(f"   站点序列： {' -> '.join(path)}")
    s.append(f"3) 对路径进行分段，选取每段尽量不换乘的线路：")
    for seg in segments:
        line = seg['line'] or '（未标注线路）'
        s.append(f"   - 在线路 {line} 上经过：{' -> '.join(seg['stations'])}")
    s.append("")
    s.append("Final Answer:")
    s.append("路线（按站点顺序）： " + " -> ".join(path))
    # 简短路线（换乘说明）
    transfers = []
    last_line = None
    for seg in segments:
        if seg['line'] != last_line:
            transfers.append(seg['line'])
            last_line = seg['line']
    transfers_str = "，然后换乘 ".join([t for t in transfers if t])
    if transfers_str:
        s.append(f"换乘说明：按顺序乘坐 {transfers_str}。")
    else:
        s.append("换乘说明：无换乘，单线可达。")
    return "\n".join(s)

def extract_pairs_from_subway_sft(path_to_subway_sft):
    """
    从 subway_sft_pro.json 中粗暴提取 '从 X 坐地铁去 Y' 这类 instruction 的 (X,Y) 对作为处理候选。
    """
    pairs = []
    try:
        with open(path_to_subway_sft, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return pairs
    for item in data:
        instr = item.get('instruction','')
        # 找 "我想从 X 坐地铁去 Y" 或 "我想从 X 坐地铁去 Y，该怎么走"
        # 这里做简单中文匹配（根据你的语料常见形式做调整）
        if instr.startswith("我想从") and "坐地铁去" in instr:
            try:
                after = instr[len("我想从"):]
                parts = after.split("坐地铁去", 1)
                if len(parts) == 2:
                    a = parts[0].strip()
                    b = parts[1].split("，")[0].strip()
                    if a and b:
                        pairs.append((a,b))
            except Exception:
                continue
    return pairs

def main(args):
    raw_dir = args.raw_lines_dir
    out_file = args.out
    subway_sft = args.subway_sft
    graph_json = args.graph_json

    # 1. 读取线路文本
    lines = load_lines_from_txt(raw_dir)
    if not lines:
        print("警告：未在 raw_lines_dir 读取到任何线路文本（请检查路径）。")
    neighbors, station_lines, edge_lines = build_graph_from_lines(lines)
    edge_lines = augment_edge_lines_from_lines(lines, edge_lines)

    print(f"已加载 {len(lines)} 条线路，构建站点数 {len(station_lines)}。")

    # 2. 准备待生成的 pairs：优先从 subway_sft 提取若干示例；否则可以让用户手工提供
    pairs = []
    if subway_sft and os.path.exists(subway_sft):
        ext_pairs = extract_pairs_from_subway_sft(subway_sft)
        print(f"从 {subway_sft} 提取到 {len(ext_pairs)} 个候选 A->B 对。")
        pairs.extend(ext_pairs)

    # 若 pairs 为空，为示例添加少量 pairs（可删除）
    if not pairs:
        # 取 line 中的若干示例起止点（首尾/中间）
        sample_pairs = []
        for ln, sts in lines.items():
            if len(sts) >= 2:
                sample_pairs.append((sts[0], sts[-1]))
            if len(sample_pairs) >= 20:
                break
        pairs = sample_pairs

    # 3. 生成 jsonl 输出
    with open(out_file, 'w', encoding='utf-8') as fout:
        for a,b in pairs:
            path = dijkstra_with_transfer(neighbors, station_lines, edge_lines, a, b,default_travel_time=2.0, transfer_penalty=4.0)
            segments = path and choose_lines_for_path(path, edge_lines, station_lines, lines_dict=lines) or []
            output_text = format_rationale(a, b, path, segments, station_lines)
            sample = {"instruction": f"从 {a} 到 {b} 的路线（请给出推导步骤和最终路线）",
                      "input": "",
                      "output": output_text}
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"已写入 {out_file}（共 {len(pairs)} 条样本）。")

if __name__ == "__main__":
    args=SimpleNamespace(
        raw_lines_dir=r"G:\Subway_ai_system\\raw_lines",
        out=r"G:\Subway_ai_system\data\shortest_graph_datasets.jsonl",
        subway_sft=r"G:\Subway_ai_system\data\subway_sft_pro.json",
        graph_json=r"G:\Subway_ai_system\data\\graph.json"
    )
    main(args)
