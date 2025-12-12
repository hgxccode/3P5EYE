@echo off
chcp 65001 >nul
echo ========================================
echo 三庭五眼人脸修复系统 - 启动服务
echo ========================================
echo.

:: 检查虚拟环境
if not exist venv (
    echo [错误] 虚拟环境不存在，请先运行 deploy.bat
    pause
    exit /b 1
)

:: 激活虚拟环境
call venv\Scripts\activate

:: 启动服务
echo 正在启动服务...
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止服务
echo ========================================
echo.
python app.py

pause
