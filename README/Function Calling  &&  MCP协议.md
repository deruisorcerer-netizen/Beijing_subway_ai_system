方案一：后端在大模型与用户之间。

![[Pasted image 20260202181721.png]]
用户发起的提问是不会直接输入给大模型的，千问等大模型只是进行聊天对话的工具，对于工具的调用是隶属于ai应用程序端的（那能否让ai像一个人一样，了解到了用户的请求后，自己用API去访问后端的数据库等应用程序，然后将读到的信息自己感知，然后进行输出）

方案二：大模型与用户直接对接，自己调用工具和数据库
大模型在对话的过程中只能判断用户的需求，并自动生成调用所需的参数，最终将文字通过符合约定格式的调用请求传给第三方插件。
其中，OpenAI引入了一种具体、可实现的Function calling技术。
	**在模型层面**，让大模型能根据自然语言上下文请求，选择正确的函数，并生成有效的参数，去调用第三方插件；
	**在API层面**，让模型额外开放对Function Calling的支持。比如ChatGPT就提供了functions的方法列表，里面有规定的json格式。
![[Pasted image 20260202183140.png]]

AI应用程序就是社会中所说的AI agent。
大模型可以接受的调用指令对每个厂商而言都是不同的，当AI应用一方在模型的API中填写完了自己第三方插件中的API json列表后，模型会自己生成一个符合约定的调用指令（对于每一个输入的函数）。这时AI应用一方应针对这个API调用指令制作针对性的字符串匹配，接受指令，并在自己的插件中调用函数。

由于你也不能要求不同厂商针对市场上如此之多的调用需求针对性的开发function_calling方法，而且市场上也有在同一应用中切换调用不同大模型的需求，但不同厂商的API不同，因此产生了MCP协议


解决方案：
	给ai应用（ai agent）设计标准化配置，调用新工具时只需要增加一个配置即可
		*本地服务*：拉取任意的工具包到本地，运行此工具包，ai应用自动获取工具信息，并自动完成调用。
		*远程服务接入*：通过ai应用的标准化配置，自动访问远程服务，自动获取工具描述配置，并自动完成调用。
	工具与agent之间的交互必须标准化：
		通信协议、接口、数据交换、配置内容都需要标准化。

**MCP协议：Model Context Protocol**
	其实就是在ai应用中工具Client与这个工具的Server之间的一个统一收发接口的抽象层。
	[What is the Model Context Protocol (MCP)? - Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro)
	上面是MCP协议的官方文档
	![[mcp-simple-diagram.avif]]
MCP协议就类似于Type-C接口，不同的agents和后端的生产力插件之间利用MCP协议统一了连接方式和端口，实现了沟通。
MCP遵循client-server架构，本地MCP服务器使用Stdio传输方式，远程MCP服务器使用流式HTTP传输方式。其中架构的核心是MCP Host、Client和Server
	MCP Host：作为一个平台、agent或ai应用协调和管理多个MCP clients
	MCP Client：向MCP Server传输约定格式指令和接收Server返回的结果，让host去使用这些返回值。
	MCP Server：为Client提供结果的程序。
每当Host需要连接到一个MCP Server的时候，就在实例化并维护一个MCP Clients与这个Server对接，进行数据传输。![[9adab079-9726-4975-9f07-afa674b95971.png]]
一种MCP协议的可视化有趣表达：![[image.png]]
![[whiteboard_exported_image.png]]
在MCP协议下，当代agent的运行逻辑
	1. 在Host（agent）刚启动的时候，Host（agent）先于MCP服务器握手（），在MCP Host中实例化可用Clients（向MCP Server发送查询有用工具信息的请求，并接收有用工具返回值）；
	2. 用户向Host（agent）提出自然语言需求，Host（agent）将自然语言需求传给大模型。大模型将需求转化为符合规范的MCP指令并返回Host（agent）；
	3. 对应的Client向Server发出工具使用申请（将得到的MCP指令传输给Server），Server将MCP指令转译成自己的指令方法（Server提供商提供），调用具体方法、工具、计算单元，并将最终输出的结果返回给Client（不用协议，就是自然文本的返回）；
	4. Host（agent）将所有Client收到的返回值打包与用户的输入一起转发给大模型，进行自然语言处理和整合（按照人类语法和逻辑输出）；
	5. 大模型将输出传送给Host（agent），(可以进行一些微加工，比如添加agent公司特别内容等)，然后把整体的输出打包传输给用户作为返回输出的结果。
