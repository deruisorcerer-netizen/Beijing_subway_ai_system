from llamafactory.chat import ChatModel
from llamafactory.api.app import create_app
import uvicorn
import os

def main():
    # 手动定义所有参数，避开命令行解析器的检查
    args = {
        "stage": "sft",
        "model_name_or_path": "G:/Subway_ai_system/Qwen_Subway_ai_rational_ultra",
        "template": "qwen",
        "infer_backend": "huggingface",
        "quantization_bit": 4,
        "do_sample": False
    }
    
    # 初始化模型
    chat_model = ChatModel(args)
    
    # 创建 API 应用
    app = create_app(chat_model)
    
    # 强制在 8001 端口启动
    print("正在启动 Subway AI API 服务，端口：8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)

if __name__ == "__main__":
    main()