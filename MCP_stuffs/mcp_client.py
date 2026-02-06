import openai
import requests
import re

# 配置
LLM_API_URL = "http://127.0.0.1:8001/v1"
# 注意：确保你的 MCP Server 正在 8000 端口运行
MCP_SERVER_URL = "http://127.0.0.1:8000" 

client = openai.OpenAI(api_key="empty", base_url=LLM_API_URL)

def call_mcp_tool(tool_name, from_station, to_station):
    url = f"{MCP_SERVER_URL}/tools/{tool_name}"
    params = {"from_name": from_station, "to_name": to_station}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # 💡 注意这里：去掉 .get('result')，直接获取 json
            data = response.json() 
            
            # 打印一下，方便你在黑窗口调试看数据对不对
            print(f"DEBUG Server Response: {data}")
            
            steps = data.get('distance_steps', '-')
            time = data.get('time_minutes', '-')
            path_info = data.get('path', '无具体路径')
            return f"📍 路径规划：\n{path_info}\n\n📊 统计数据：共经过 {steps} 站，预计耗时 {time} 分钟。"
        else:
        # 提取服务器返回的精确 detail
            try:
                err_detail = response.json().get('detail', '计算出错')
                return f"❌ {err_detail},站名输入模糊，请输入正确站名（如：光熙门站，而非光熙家园）"
            except:
                return "❌ 地铁服务器响应异常。"
    except Exception as e:
        return f"无法连接到地铁服务器: {str(e)}"

def run_ai_agent(user_input):
    # 1. 构造强约束的 System Prompt
    system_prompt = """你是一个北京地铁专家。用户会向你咨询地铁路径。
    如果需要查询路径，请**必须**按照以下格式回复，不要有任何多余文字：
    CALL:tool_name(from="起点站", to="终点站")
    可选工具名：get_distance_efficient_path (最短路程), get_time_efficient_path (最快时间)
    
    例如：CALL:get_time_efficient_path(from="积水潭", to="西直门")
    """
    
    # 2. 调用微调后的 Qwen 模型
    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1 # 降低随机性，让格式更固定
    )
    
    content = response.choices[0].message.content.strip()
    
    # 3. 解析模型是否发出了调用指令
    # 使用正则表达式提取：CALL:工具名(from="xxx", to="xxx")
    match = re.search(r'CALL:(\w+)\(from="(.*?)", to="(.*?)"\)', content)
    
    if match:
        tool_name, start, end = match.groups()
        print(f"--- 正在调用工具: {tool_name} ---")
        return call_mcp_tool(tool_name, start, end)
    else:
        # 如果 AI 直接回答了（没有触发工具），则直接返回 AI 的话
        return content