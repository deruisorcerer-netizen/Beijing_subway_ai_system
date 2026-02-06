MCP官方规范，学习server和client的集合
网址端：
	https://modelcontextprotocol.io/docs/getting-started/intro

建立MCP Server：
	对于利用Stdio规范的服务器来说：
		由于Clients通过管道传输来的数据是stdin，遵循Stdio规范，因此在Server输出的时候，应当通过管道将正式的Json-RPC返回值利用stdout的形式传输回去。
		不能出现print()等非stdout的输出形式。不要有”print("正在计算中...")“类似的废话。
	对于HTTP规范的服务器来说：
		Server通过特定的网络端口与Client交流，回复内容被封装在HTTP响应体里。
MCP Server分为三种类型：
	· 实时工具型 天气、股票、搜索引擎
	· 计算工具型 python解释器、规划器
	· 预计算知识型  Static/Precomputed Knowledage Server