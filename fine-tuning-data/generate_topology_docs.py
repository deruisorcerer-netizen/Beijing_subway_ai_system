# generate_topology_docs.py
import os, json, datetime

RAW_DIR = "raw_lines"   # 放你的每条线路txt，每行一个站名（第一行可写线路名）
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

topology = {"lines": [], "edges": [], "metadata": {"generated_at": None, "source":"user_txt"}}
docs = []
sft = []

for fname in sorted(os.listdir(RAW_DIR), key=str.lower):
    if not fname.endswith(".txt"): continue
    line_id = os.path.splitext(fname)[0]
    with open(os.path.join(RAW_DIR, fname), "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # use first line as line display name if it contains "线" or "号线"
    line_name = lines[0] if len(lines)>0 and ("号线" in lines[0] or "线" in lines[0]) else line_id
    station_names = lines[1:] if (len(lines)>0 and ("号线" in lines[0] or "线" in lines[0])) else lines
    station_list = []
    for idx,name in enumerate(station_names, start=1):
        sid = f"{line_id.upper()}_ST_{idx:03d}"
        station = {"id": sid, "name": name, "lat": None, "lon": None, "attributes": {}}
        station_list.append(station)
        docs.append({"doc_id": sid, "text": f"{name}站（ID:{sid}），属于{line_name}。", "metadata": {"line": line_name, "station_name": name}})
    topology["lines"].append({"line_id": line_id, "line_name": line_name, "stations": station_list})
    for i in range(len(station_list)-1):
        e = {"edge_id": f"{line_id}_E_{i+1:03d}", "source": station_list[i]["id"], "target": station_list[i+1]["id"], "distance_m": None, "track_type": "双线", "signal": None}
        topology["edges"].append(e)
    if len(station_list) >= 2:
        sft.append({
            "instruction": f"给出 {station_list[0]['name']} 到 {station_list[-1]['name']} 的行程概要（列出经过站点）。",
            "input": f"线名：{line_name}",
            "output": " → ".join([s["name"] for s in station_list])
        })

topology["metadata"]["generated_at"] = datetime.datetime.now().isoformat()
with open(os.path.join(OUT_DIR,"topology.json"), "w", encoding="utf-8") as f:
    json.dump(topology, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT_DIR,"docs.jsonl"), "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

with open(os.path.join(OUT_DIR,"sft_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(sft, f, ensure_ascii=False, indent=2)

print("生成完成：", os.path.join(OUT_DIR,"topology.json"))
print("docs:", os.path.join(OUT_DIR,"docs.jsonl"))
print("sft:", os.path.join(OUT_DIR,"sft_dataset.json"))
