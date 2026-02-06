import streamlit as st
from mcp_client import run_ai_agent

st.set_page_config(page_title="北京地铁 AI 助手", page_icon="🚇")

st.title("🚇 北京地铁智能助手")
st.caption("微调 Qwen2.5 + MCP 实时路径规划")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("例如：我想从积水潭去西直门，怎么走最快？"):
    # 用户输入展示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 思考并响应
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            # 调用中枢逻辑
            response_text = run_ai_agent(prompt)
            st.markdown(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})