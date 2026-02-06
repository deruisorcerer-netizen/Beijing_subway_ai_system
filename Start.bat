@echo off
chcp 65001 > nul

:: 获取脚本所在的文件夹路径 (末尾带 \)
set BASE_DIR=%~dp0

:: 定义相对路径变量
set VENV_PATH=%BASE_DIR%venv\Scripts\activate
set PROJECT_ROOT=%BASE_DIR%

title Subway AI 一键启动相对路径版

echo ====================================================
echo    当前运行目录: %BASE_DIR%
echo ====================================================

:: 1. 启动 MCP Server
:: 注意：%BASE_DIR% 已经包含了末尾的 \，所以后面直接接文件夹名
start "MCP_Server" cmd /k "chcp 65001 > nul && call "%VENV_PATH%" && cd /d "%BASE_DIR%MCP_stuffs" && python mcp_server.py"

:: 2. 启动 API (调用同样位于项目下的子脚本)
start "LLM_API" cmd /k "chcp 65001 > nul && call "%BASE_DIR%LLaMA-Factory\src\start_api.bat""

:: 3. 启动 UI
start "Subway_UI" cmd /k "chcp 65001 > nul && call "%VENV_PATH%" && cd /d "%BASE_DIR%MCP_stuffs" && streamlit run app_gui.py"

:CHECK_MCP
set /a mcp_retry+=1
netstat -ano | findstr :8000 > nul
if %errorlevel% equ 0 (
    echo [OK] MCP 数据服务器已就绪 (Port: 8000)
) else (
    if %mcp_retry% LSS 10 (
        timeout /t 2 > nul
        goto CHECK_MCP
    ) else (
        echo [!!] MCP 服务器启动较慢，请检查窗口日志。
    )
)

:CHECK_API
set /a api_retry+=1
netstat -ano | findstr :8001 > nul
if %errorlevel% equ 0 (
    echo [OK] 微调大模型 API 已就绪 (Port: 8001)
) else (
    if %api_retry% LSS 20 (
        echo 正在等待模型载入显存 (尝试 %api_retry%/20)...
        timeout /t 3 > nul
        goto CHECK_API
    ) else (
        echo [!!] API 未能在预定时间内响应，可能正在加载大型权重。
    )
)

echo ----------------------------------------------------
echo 全部检测完成！
echo 如果 [OK] 状态已出现，你可以开始使用浏览器进行对话了。
echo ====================================================
pause