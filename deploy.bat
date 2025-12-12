@echo off
chcp 65001 >nul
echo ========================================
echo 三庭五眼人脸修复系统 - Windows部署脚本
echo ========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

:: 创建虚拟环境
if not exist venv (
    echo [1/4] 创建虚拟环境...
    python -m venv venv
) else (
    echo [1/4] 虚拟环境已存在
)

:: 激活虚拟环境
echo [2/4] 激活虚拟环境...
call venv\Scripts\activate

:: 安装依赖
echo [3/4] 安装依赖...
if exist offline_packages (
    pip install --no-index --find-links=offline_packages flask opencv-python numpy 2>nul
    if errorlevel 1 (
        echo     离线包安装失败，尝试在线安装...
        pip install flask opencv-python numpy
    ) else (
        echo     离线包安装成功
    )
) else (
    echo     未找到离线包目录，使用在线安装...
    pip install flask opencv-python numpy
)

:: 创建必要目录
echo [4/4] 创建必要目录...
if not exist static\uploads mkdir static\uploads
if not exist static\fontawesome\css mkdir static\fontawesome\css
if not exist static\fontawesome\webfonts mkdir static\fontawesome\webfonts

echo.
echo ========================================
echo 部署完成！
echo ========================================
echo.
echo 启动服务请运行: start.bat
echo 或手动执行:
echo   venv\Scripts\activate
echo   python app.py
echo.
echo 访问地址: http://localhost:5000
echo ========================================
pause
