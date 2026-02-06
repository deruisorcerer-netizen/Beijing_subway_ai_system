# 🚇 北京地铁万事通 (Beijing Subway Master Agent)

**“北京地铁万事通”** 是一款基于 **Qwen2.5-7B** 微调模型与 **MCP (Model Context Protocol)** 协议构建的智能出行 Agent。该项目通过大模型语义理解与结构化数据库调用的深度集成，解决了北京地铁复杂线路环境下的“ A 到 B ”路径规划问题，并支持“路程最短”与“时间最优”的双重决策逻辑。

---

## 🌟 核心特性

- **精调大模型驱动**：基于 Qwen2.5-7B-Instruct 进行 SFT 训练，具备强大的地铁领域语义理解能力。
    
- **MCP 标准架构**：采用 Model Context Protocol 协议，实现模型推理层与工具执行层（SQLite）的完全解耦。
    
- **双路由导航引擎**：
    
    - **Distance-Efficient**：基于物理距离的最短路径算法。
        
    - **Time-Efficient**：基于运行时间的最快到达算法。
        
- **端到端 Agent 逻辑**：Agent 作为 MCP Host，能够根据用户意图（如“我赶时间”或“我想少坐几站”）自动选择最优工具路径。
  -
项目概述：
	通过大模型微调+MCP协议下的数据库搭建和调用实现解决在北京地铁线路图中从A to B问题的Agent。
		1.大模型微调利用Qwen2.5 - Instruct 7B模型，在LLaMA-Factory开源微调训练平台上训练
		2.微调数据来源于本项目文件中raw_lines文件夹中的txt文档，皆为地铁线路名+各站点的名称，通过fune-tuning-data中的脚本文件，生成同一文件夹下的微调数据。
		3.数据库搭建利用MCP_stuffs\data路径下的脚本文件build_subway_paths_database.py，分别按照路程最短和时间最优的方式生成SQLite数据库。
		4.将两数据库作为一个MCP Server并暴露寻找路程最短路径和寻找时间最优路径两个接口（拥有两个路由）。
		5.将AI应用（无实体，就是一个说法，实际上代表着整个前端）作为一个MCP Host，控制着一个MCP Client,这个Client利用微调后的Qwen模型自动判断根据用户需求而需要调用哪个路由或者方法去使用
---

## 🏗️ 系统架构

本项目由以下四个核心模块组成：

1. **数据引擎 (Data Engine)**：解析 `raw_lines` 中的拓扑数据，通过算法脚本生成加权图数据库（SQLite）。
    
2. **模型层 (Model Layer)**：在 LLaMA-Factory 环境下完成领域知识微调，提供符合 `CALL:tool_name` 规范的指令输出。
    
3. **协议服务器 (MCP Server)**：暴露双路由 API 接口，负责处理逻辑运算并返回结构化路径数据。
    
4. **交互终端 (Application Host)**：基于 Streamlit 的前端应用，控制 MCP Client 实现闭环对话。
    

---

## 📂 目录结构

Plaintext

```
Beijing_subway_ai_system/
├── MCP_stuffs/             # MCP Server 实现与双策略数据库
│   └── data/               # 数据库构建脚本与 .sqlite 文件
├── fine-tuning-data/       # SFT 训练集生成脚本与预处理数据
├── raw_lines/              # 原始地铁线路拓扑文本 (txt)
├── LLaMA-Factory_api/      # 对接微调框架的推断适配层
├── requirements.txt        # 项目依赖清单
└── Start.bat               # 一键启动脚本
```
---

## 🚀 快速开始

### 1. 环境初始化

克隆本项目到本地，并创建 Python 虚拟环境：

Bash

```
python -m venv venv
# Windows 激活环境
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 依赖项配置 (关键)

为了保证系统正常运行，请手动完成以下步骤：

- **微调框架**：克隆 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 至主目录。
    
- **接口移植**：将本库 `LLaMA-Factory_api` 下的文件拷贝至 `LLaMA-Factory/src` 中。
    
- **模型部署**：从 Hugging Face 下载 `Qwen_Subway_ai_rational_ultra` 权重，放置于项目根目录。
    

### 3. 启动系统

双击根目录下的 **`Start.bat`**。

系统将并行启动 **MCP Server**（8000端口）、**Model API**（8001端口）以及 **Streamlit 前端**。当控制台均显示 `Uvicorn running` 后，即可在浏览器中使用。

---

## 🖼️ 项目展示
| 界面预览 | 路径规划示例 | 逻辑调用展示 |
| :--- | :--- | :--- |
| ![界面截图](Pasted%20image%2020260206124846.png) | ![规划示例](Pasted%20image%2020260206124943.png) | ![逻辑展示](Pasted%20image%2020260206125033.png) |

---
使用方法：
	下载仓库到本地，按照requirment.txt中的要求设置虚拟环境venv；
	在  https://github.com/hiyouga/LlamaFactory  这个连接里下载LLaMA-Factory，并放进本项目的主文件夹里；
	将本项目主文件夹下 LLaMA-Factory_api 里面所有的项目都拷贝到你刚才下载的LLaMA-Factory\src里面；
	在Hugging Face上下载Qwen_Subway_ai_rational_ultra 7B 微调后大模型，并将下载后的文件放进本项目的主文件夹下。
	双击本项目主文件夹下的Start.bat，等待全部信息加载成功，即Server控制台和模型控制台都出现了“INFO:     Uvicorn running on【你的本地网址】”，并且前端已经在你默认的浏览器中打开。好！你就可以开始享受本项目从A到B的便捷导航了。
---

> **注**：本项目目前主要针对北京地铁核心线路，建议使用标准站名（如“光熙门站”）以获取最高准确度。
