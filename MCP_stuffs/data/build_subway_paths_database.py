"""
build_subway_paths.py

用途：
- 读取 RAW_LINES_DIR 下的所有 .txt（每个文件为一条地铁线，文件第一行可为线路名或首站）
- 根据 STRATEGY ( "distance" 或 "time" ) 计算所有 A -> B 的最短路径（包含相邻站）
- 把结果（双向 A->B）写入 SQLite DB OUT_DB_PATH

修改这里的变量即可（不要交互式输入）：
RAW_LINES_DIR, OUT_DB_PATH, STRATEGY, TRAVEL_TIME_PER_EDGE, TRANSFER_TIME

依赖：Python 3.8+（仅标准库）
"""
import os
import glob
import sqlite3
import json
from pathlib import Path
import heapq
from collections import defaultdict

# -------------------------
# 配置区（请按需修改）
# -------------------------
RAW_LINES_DIR = r"G:\Subway_ai_system\\raw_lines"   # <- 把这里改成你本地文件夹路径（不需要输入）
OUT_DB_PATH   = r"G:\Subway_ai_system\\MCP_stuffs\data\AtoB_time_efficient.sqlite"  # <- 输出的 SQLite 文件
STRATEGY      = "time"   # "distance" 或 "time". 选择优化目标
TRAVEL_TIME_PER_EDGE = 2  # 相邻站坐地铁所需分钟（整数）
TRANSFER_TIME = 4         # 换乘惩罚（分钟）
# -------------------------

# Helper: read .txt files and parse station sequences
def parse_lines_from_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    lines = {}  # line_name -> [station1, station2, ...]
    if not files:
        raise RuntimeError(f"No .txt files found in {folder}")
    for p in files:
        with open(p, "r", encoding="utf-8-sig") as f:
            raw = [ln.rstrip("\n\r") for ln in f]
        # strip empties
        cleaned = [ln.strip() for ln in raw if ln and ln.strip()]
        if not cleaned:
            continue
        # Heuristic: if first line looks like a line name (contains '线' or non-station pattern), treat as header
        first = cleaned[0]
        rest = cleaned[1:] if len(cleaned) >= 2 else cleaned
        # If first line contains '线' or contains non-station characters, use as line name
        if ("线" in first) or (any(c.isalpha() for c in first) and len(rest) >= 2):
            line_name = first
            stations = rest
        else:
            # fallback to filename stem as line name
            line_name = Path(p).stem
            stations = cleaned
        # final cleanup: remove blanks
        stations = [s for s in stations if s]
        lines[line_name] = stations
    return lines

# Dijkstra on a graph represented as adjacency dict: node -> list of (neighbor, weight)
def dijkstra_all_nodes(adj):
    """For each source node, run Dijkstra and return dict of (src -> (dist_map, prev_map))"""
    all_results = {}
    nodes = list(adj.keys())
    for src in nodes:
        dist = {n: float('inf') for n in nodes}
        prev = {n: None for n in nodes}
        dist[src] = 0
        h = [(0, src)]
        while h:
            d,u = heapq.heappop(h)
            if d > dist[u]:
                continue
            for v,w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(h, (nd, v))
        all_results[src] = (dist, prev)
    return all_results

def reconstruct_path_prev(prev_map, src, dst):
    if prev_map[dst] is None and src != dst:
        return None
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        if cur == src:
            break
        cur = prev_map[cur]
    path.reverse()
    if path[0] != src:
        return None
    return path

# Utility to ensure folder path exists for DB
def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# Main flow
def main():
    print("Parsing lines from:", RAW_LINES_DIR)
    lines = parse_lines_from_folder(RAW_LINES_DIR)
    print(f"Found {len(lines)} lines.")

    # Build station list and mappings
    station_to_id = {}
    stations = []
    for seq in lines.values():
        for st in seq:
            if st not in station_to_id:
                station_to_id[st] = len(stations)
                stations.append(st)
    N = len(stations)
    print(f"Total stations discovered: {N}")

    # Build "lines" table data (keep sequences)
    # Also build station -> lines mapping
    station_lines = defaultdict(list)  # station_name -> list of line_names
    for line_name, seq in lines.items():
        for st in seq:
            station_lines[st].append(line_name)

    # Prepare DB
    ensure_parent_dir(OUT_DB_PATH)
    if os.path.exists(OUT_DB_PATH):
        os.remove(OUT_DB_PATH)
    conn = sqlite3.connect(OUT_DB_PATH)
    cur = conn.cursor()

    # Create tables
    cur.execute("CREATE TABLE stations (id INTEGER PRIMARY KEY, name TEXT UNIQUE)")
    cur.execute("CREATE TABLE lines (id INTEGER PRIMARY KEY, name TEXT, stations_json TEXT)")
    cur.execute("CREATE TABLE edges (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, line_name TEXT, UNIQUE(a,b,line_name))")
    cur.execute("""CREATE TABLE paths (
                    id INTEGER PRIMARY KEY,
                    from_id INTEGER,
                    to_id INTEGER,
                    distance_steps INTEGER,  -- number of station hops (if applicable)
                    time_minutes REAL,       -- total minutes (if applicable)
                    path_json TEXT
                )""")

    # Insert stations
    for i, name in enumerate(stations):
        cur.execute("INSERT INTO stations (id, name) VALUES (?, ?)", (i, name))

    # Insert lines
    for line_name, seq in lines.items():
        cur.execute("INSERT INTO lines (name, stations_json) VALUES (?, ?)", (line_name, json.dumps(seq, ensure_ascii=False)))

    # Insert edges (unique undirected) for reference (line-level adjacency)
    edges_set = set()
    for line_name, seq in lines.items():
        for a, b in zip(seq, seq[1:]):
            ia = station_to_id[a]; ib = station_to_id[b]
            key = (min(ia,ib), max(ia,ib), line_name)
            if key not in edges_set:
                edges_set.add(key)
                cur.execute("INSERT INTO edges (a,b,line_name) VALUES (?,?,?)", (key[0], key[1], line_name))
    conn.commit()
    print(f"Inserted stations ({len(stations)}), lines ({len(lines)}), edges ({len(edges_set)}) into DB.")

    # Depending on STRATEGY, build graph(s) and compute all-pairs shortest
    if STRATEGY == "distance":
        print("Computing distance-optimal shortest paths (edge weight = 1 per adjacent station)...")
        # Build adjacency on station-level
        adj = {i: [] for i in range(N)}
        for (a,b,line_name) in edges_set:
            adj[a].append((b, 1.0))
            adj[b].append((a, 1.0))
        # Run Dijkstra from every station
        results = dijkstra_all_nodes(adj)
        total_paths = 0
        batch = []
        for src in range(N):
            dist_map, prev = results[src]
            for dst in range(N):
                if dist_map[dst] == float('inf'):
                    continue
                path_ids = reconstruct_path_prev(prev, src, dst)
                if path_ids is None:
                    continue
                path_names = [stations[i] for i in path_ids]
                distance_steps = int(dist_map[dst])
                # Estimate time min as distance_steps * TRAVEL_TIME_PER_EDGE (no transfer accounted here)
                time_minutes = distance_steps * TRAVEL_TIME_PER_EDGE
                batch.append((src, dst, distance_steps, time_minutes, json.dumps(path_names, ensure_ascii=False)))
                total_paths += 1
        cur.executemany("INSERT INTO paths (from_id,to_id,distance_steps,time_minutes,path_json) VALUES (?,?,?,?,?)", batch)
        conn.commit()
        print(f"Stored {total_paths} ordered paths (A->B).")

    elif STRATEGY == "time":
        print("Computing time-optimal shortest paths (travel and transfer penalties considered)...")
        # Build expanded nodes: station + line -> node id
        # For stations that appear on a line, we create node (station_name, line_name)
        node_idx = {}
        idx_node = []
        for line_name, seq in lines.items():
            for st in seq:
                key = (st, line_name)
                if key not in node_idx:
                    node_idx[key] = len(idx_node)
                    idx_node.append(key)
        M = len(idx_node)
        print(f"Expanded nodes (station,line) count: {M}")

        # Build adjacency on expanded graph
        # adjacency: node_id -> list of (neighbor_node_id, weight_minutes)
        adj = {i: [] for i in range(M)}

        # Add travel edges along lines (consecutive stations)
        for line_name, seq in lines.items():
            for a,b in zip(seq, seq[1:]):
                na = node_idx[(a, line_name)]
                nb = node_idx[(b, line_name)]
                # both directions
                adj[na].append((nb, TRAVEL_TIME_PER_EDGE))
                adj[nb].append((na, TRAVEL_TIME_PER_EDGE))

        # Add transfer edges at same station between different lines
        # For each station, find all its line-nodes and fully connect with transfer cost
        for st, line_list in station_lines.items():
            # if only one line, no transfer edges needed
            if len(line_list) < 2:
                continue
            node_ids = [node_idx[(st, ln)] for ln in line_list if (st, ln) in node_idx]
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    a = node_ids[i]; b = node_ids[j]
                    adj[a].append((b, TRANSFER_TIME))
                    adj[b].append((a, TRANSFER_TIME))

        # Dijkstra from every expanded node
        all_results = dijkstra_all_nodes(adj)

        # For each pair of stations (A,B), we need to pick the best combination:
        # min over src_node in nodes_of(A), dst_node in nodes_of(B) of total_time
        total_paths = 0
        batch = []
        # precompute station -> expanded-node-ids
        station_nodes = {}
        for st in stations:
            station_nodes[station_to_id[st]] = [ node_idx[(st, ln)] for ln in station_lines[st] if (st, ln) in node_idx ]

        for src_id in range(N):
            src_nodes = station_nodes.get(src_id, [])
            if not src_nodes:
                continue
            # For each possible starting expanded node, we have precomputed distances
            # We'll compute for dsts by testing combos
            # For speed, we can compute a map: dst_exp_node -> (best_time, src_exp_node)
            best_to = {}
            for s_node in src_nodes:
                dist_map, prev_map = all_results[s_node]
                for t_node in range(M):
                    d = dist_map[t_node]
                    if d == float('inf'):
                        continue
                    prev_best = best_to.get(t_node)
                    if (prev_best is None) or (d < prev_best[0]):
                        best_to[t_node] = (d, s_node, prev_map)  # store prev_map for reconstruction if needed

            # Now for each destination station, find minimal among its expanded nodes
            for dst_id in range(N):
                dst_nodes = station_nodes.get(dst_id, [])
                if not dst_nodes:
                    continue
                best_time = float('inf')
                best_pair = None  # (start_exp_node, end_exp_node, prev_map_for_start)
                best_prev_map = None
                for t_node in dst_nodes:
                    entry = best_to.get(t_node)
                    if entry is None:
                        continue
                    d, s_node, prev_map = entry
                    if d < best_time:
                        best_time = d
                        best_pair = (s_node, t_node)
                        best_prev_map = prev_map
                if best_pair is None:
                    continue
                # reconstruct expanded-node path using prev map from the chosen s_node
                s_node, t_node = best_pair
                path_exp = reconstruct_path_prev(best_prev_map, s_node, t_node)
                if path_exp is None:
                    continue
                # project expanded-node path to station names (collapse consecutive duplicates)
                path_stations = []
                last_st = None
                for node in path_exp:
                    st_name, line_name = idx_node[node]
                    if st_name != last_st:
                        path_stations.append(st_name)
                        last_st = st_name
                distance_steps = max(len(path_stations)-1, 0)  # number of hops
                time_minutes = float(best_time)
                batch.append((src_id, dst_id, distance_steps, time_minutes, json.dumps(path_stations, ensure_ascii=False)))
                total_paths += 1

        # Insert to DB
        cur.executemany("INSERT INTO paths (from_id,to_id,distance_steps,time_minutes,path_json) VALUES (?,?,?,?,?)", batch)
        conn.commit()
        print(f"Stored {total_paths} ordered paths (A->B) with time-optimal metric.")

    else:
        raise RuntimeError("STRATEGY must be 'distance' or 'time'")

    # Final counts
    num_stations = cur.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
    num_paths = cur.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
    print(f"Finished. Stations: {num_stations}, Paths stored: {num_paths}")
    conn.close()
    print("SQLite DB created at:", OUT_DB_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
