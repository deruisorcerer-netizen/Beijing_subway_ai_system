"""
mcp_server.py

轻量教学版 MCP-style HTTP Server (FastAPI)
- 读取两个 SQLite DB（distance 和 time）
- 暴露:
  - GET  /mcp/capabilities         -> 列出可用工具及参数说明 (machine-readable)
  - POST /rpc                      -> 简单 JSON-RPC 2.0 风格入口 (method "tools.list" 或 "tools.call")
  - GET  /tools/{tool_name}        -> 便捷的 REST 测试端点 (query params ?from=...&to=...)
  - GET  /health                   -> 健康检查

修改顶部变量后直接运行：
python mcp_server.py
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
from typing import Dict, Any, Optional
import os
import logging

# --------------------
# 配置区 - 修改这里
# --------------------
DATA_DIR = r"G:\Subway_ai_system\\MCP_stuffs\data"  # <- 只改这一行，指向包含两个 DB 的文件夹
DB_DISTANCE = os.path.join(DATA_DIR, "AtoB_distance_efficient.sqlite")  # 文件名可根据你实际情况改
DB_TIME     = os.path.join(DATA_DIR, "AtoB_time_efficient.sqlite")
HOST = "0.0.0.0"
PORT = 8000
# --------------------

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

app = FastAPI(title="Educational MCP-style Subway Server")

# 简单 CORS，方便前端或测试脚本访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 工具元数据（用来模拟 MCP 的 capability manifest）
TOOLS = {
    "get_distance_efficient_path": {
        "name": "get_distance_efficient_path",
        "description": "Return A->B path optimized for distance (fewest station hops). Params: from (station name), to (station name).",
        "params_schema": {
            "type": "object",
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"}
            },
            "required": ["from", "to"]
        },
        "db": "distance"
    },
    "get_time_efficient_path": {
        "name": "get_time_efficient_path",
        "description": "Return A->B path optimized for time (transfer penalties and travel time accounted). Params: from (station name), to (station name).",
        "params_schema": {
            "type": "object",
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"}
            },
            "required": ["from", "to"]
        },
        "db": "time"
    }
}

# --------------------
# Helper: open sqlite connections and helpers
# --------------------
def open_conn(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DB not found: {path}")
    # allow multithread access from uvicorn workers if needed
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

logger.info("Opening databases...")
try:
    conn_distance = open_conn(DB_DISTANCE)
except Exception as e:
    logger.warning(f"Distance DB open failed: {e}")
    conn_distance = None

try:
    conn_time = open_conn(DB_TIME)
except Exception as e:
    logger.warning(f"Time DB open failed: {e}")
    conn_time = None


from difflib import get_close_matches
import logging
# 1. 增加全局站名缓存，用于模糊匹配
STATION_CACHE = []

def get_all_stations(conn):
    global STATION_CACHE
    if not STATION_CACHE:
        cur = conn.cursor()
        cur.execute("SELECT name FROM stations")
        STATION_CACHE = [r["name"] for r in cur.fetchall()]
    return STATION_CACHE

def find_station_id(conn: sqlite3.Connection, station_name: str) -> Optional[int]:
    cur = conn.cursor()
    # 先尝试精确匹配
    cur.execute("SELECT id FROM stations WHERE name = ? COLLATE NOCASE LIMIT 1", (station_name,))
    r = cur.fetchone()
    if r: return r["id"]
    
    # 模糊匹配：解决用户输入“燕山”但数据库里是“燕山站”的问题
    all_names = get_all_stations(conn)
    matches = get_close_matches(station_name, all_names, n=1, cutoff=0.6)
    if matches:
        corrected_name = matches[0]
        logger.info(f"将 '{station_name}' 纠正为 '{corrected_name}'")
        cur.execute("SELECT id FROM stations WHERE name = ?", (corrected_name,))
        r = cur.fetchone()
        return r["id"] if r else None
    return None

def query_path_by_ids(conn: sqlite3.Connection, from_id: int, to_id: int):
    cur = conn.cursor()
    # 1. 依然先获取路径
    cur.execute("SELECT distance_steps, time_minutes, path_json FROM paths WHERE from_id = ? AND to_id = ? LIMIT 1", (from_id, to_id))
    r = cur.fetchone()
    if not r: return None

    stations_path = json.loads(r["path_json"])
    
    # 提前一次性获取所有站名的 ID 映射，避免在循环里反复查数据库
    placeholders = ','.join(['?'] * len(stations_path))
    cur.execute(f"SELECT name, id FROM stations WHERE name IN ({placeholders})", stations_path)
    name_to_id = {row["name"]: row["id"] for row in cur.fetchall()}

    detailed_segments = []
    last_line = None
    
    # 遍历路径
    for i in range(len(stations_path) - 1):
        s1, s2 = stations_path[i], stations_path[i+1]
        id1, id2 = name_to_id.get(s1), name_to_id.get(s2)
        
        if not id1 or not id2: continue

        # 查线路名
        cur.execute("SELECT line_name FROM edges WHERE (a=? AND b=?) OR (a=? AND b=?) LIMIT 1", (id1, id2, id2, id1))
        edge_row = cur.fetchone()
        line = edge_row["line_name"] if edge_row else "未知线路"
        
        # 优化后的合并逻辑：使用 last_line 变量
        if line != last_line:
            # 记录换乘信息
            detailed_segments.append(f"【{line}】: {s1} -> {s2}")
            last_line = line
        else:
            # 同一线，只追加站名
            detailed_segments[-1] += f" -> {s2}\n"

    return {
        "distance_steps": r["distance_steps"],
        "time_minutes": r["time_minutes"],
        "path": "\n".join(detailed_segments)
    }
    
#针对输入模糊的异常处理
def do_get_path_logic(from_name: str, to_name: str, conn):
    from_id = find_station_id(conn, from_name)
    to_id = find_station_id(conn, to_name)
    
    # 这里的改进：精准指出哪个站找不到
    if from_id is None or to_id is None:
        missing = []
        if from_id is None: missing.append(f"起点 '{from_name}'")
        if to_id is None: missing.append(f"终点 '{to_name}'")
        
        # 抛出具体的错误信息
        raise HTTPException(
            status_code=404, 
            detail=f"站点识别失败：数据库中找不到 {' 和 '.join(missing)}。请尝试输入官方全称（如：光熙门站，而非光熙家园）。"
        )

# --------------------
# Tool implementations
# --------------------
def do_get_distance_efficient_path(from_name: str, to_name: str) -> Dict[str, Any]:
    if conn_distance is None:
        raise HTTPException(status_code=500, detail="Distance DB not available on server.")
    from_id = find_station_id(conn_distance, from_name)
    to_id = find_station_id(conn_distance, to_name)
    if from_id is None or to_id is None:
        raise HTTPException(status_code=404, detail=f"Station not found. from_id={from_id}, to_id={to_id}")
    row = query_path_by_ids(conn_distance, from_id, to_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No path found in distance DB for that ordered pair.")
    return {"from": from_name, "to": to_name, "distance_steps": row["distance_steps"], "time_minutes": row["time_minutes"], "path": row["path"], "db": os.path.basename(DB_DISTANCE)}

def do_get_time_efficient_path(from_name: str, to_name: str) -> Dict[str, Any]:
    if conn_time is None:
        raise HTTPException(status_code=500, detail="Time DB not available on server.")
    from_id = find_station_id(conn_time, from_name)
    to_id = find_station_id(conn_time, to_name)
    if from_id is None or to_id is None:
        raise HTTPException(status_code=404, detail=f"Station not found. from_id={from_id}, to_id={to_id}")
    row = query_path_by_ids(conn_time, from_id, to_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No path found in time DB for that ordered pair.")
    return {"from": from_name, "to": to_name, "distance_steps": row["distance_steps"], "time_minutes": row["time_minutes"], "path": row["path"], "db": os.path.basename(DB_TIME)}

# --------------------
# HTTP endpoints
# --------------------

@app.get("/health")
async def health():
    return {"status": "ok", "distance_db": bool(conn_distance), "time_db": bool(conn_time)}

@app.get("/mcp/capabilities")
async def capabilities():
    """Return a machine-readable description of available tools (simple manifest)."""
    return {"tools": list(TOOLS.values())}

@app.post("/rpc")
async def rpc_entry(request: Request):
    """
    Minimal JSON-RPC 2.0 style entrypoint.
    Expected payload:
    {
      "jsonrpc": "2.0",
      "id": <id>,
      "method": "tools.list" | "tools.call",
      "params": { ... }
    }

    For method "tools.call", params should be:
    {
      "name": "get_distance_efficient_path",
      "arguments": { "from": "...", "to": "..." }
    }
    """
    payload = await request.json()
    # basic validation
    if "method" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'method' in RPC payload")
    method = payload["method"]
    req_id = payload.get("id", None)
    try:
        if method == "tools.list":
            result = list(TOOLS.values())
        elif method == "tools.call":
            params = payload.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not name or not isinstance(arguments, dict):
                raise HTTPException(status_code=400, detail="Invalid 'params' for tools.call")
            from_name = arguments.get("from")
            to_name   = arguments.get("to")
            if not from_name or not to_name:
                raise HTTPException(status_code=400, detail="Missing 'from' or 'to' in arguments")
            if name == "get_distance_efficient_path":
                result = do_get_distance_efficient_path(from_name, to_name)
            elif name == "get_time_efficient_path":
                result = do_get_time_efficient_path(from_name, to_name)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown tool name: {name}")
        else:
            raise HTTPException(status_code=404, detail=f"Unknown RPC method: {method}")
        # JSON-RPC style response
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})
    except HTTPException as e:
        # return JSON-RPC error format
        err = {"code": e.status_code, "message": e.detail}
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": err}, status_code=400)

# Convenience REST endpoints for manual testing
@app.get("/tools/{tool_name}")
async def tool_get(tool_name: str, from_name: str, to_name: str):
    if tool_name == "get_distance_efficient_path":
        return do_get_distance_efficient_path(from_name, to_name)
    elif tool_name == "get_time_efficient_path":
        return do_get_time_efficient_path(from_name, to_name)
    else:
        raise HTTPException(status_code=404, detail="Tool not found")

# Shutdown: close DBs gracefully
@app.on_event("shutdown")
def shutdown():
    try:
        if conn_distance:
            conn_distance.close()
        if conn_time:
            conn_time.close()
        logger.info("Closed DB connections.")
    except Exception:
        pass

# if run as script, start uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host=HOST, port=PORT, log_level="info", reload=False)
