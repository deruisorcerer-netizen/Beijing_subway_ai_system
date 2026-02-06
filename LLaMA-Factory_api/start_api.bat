@echo off
:: 获取当前脚本所在的文件夹
set CURRENT_DIR=%~dp0

:: 切换到当前脚本所在的目录
cd /d "%CURRENT_DIR%"

:: 这里的相对路径向上跳两级找到 venv (或者根据你的实际结构调整)
:: 如果 venv 就在项目根目录，通常是 ..\..\venv
call "%CURRENT_DIR%..\..\venv\Scripts\activate"

:: 运行 API
python run_api.py

pause