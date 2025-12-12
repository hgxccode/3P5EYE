@echo off
chcp 65001 >nul
echo ========================================
echo Python离线依赖包下载脚本
echo ========================================
echo.
echo 此脚本将下载项目所需的Python依赖包到 offline_packages 目录
echo 请确保当前机器可以访问互联网
echo.

:: 创建目录
if not exist offline_packages mkdir offline_packages

echo 开始下载依赖包...
echo.

:: 下载依赖包（适用于Windows x64）
pip download -d offline_packages -r requirements.txt

if errorlevel 1 (
    echo.
    echo [警告] 部分包下载失败，尝试指定平台...
    pip download -d offline_packages --platform win_amd64 --python-version 38 --only-binary=:all: flask opencv-python numpy
)

echo.
echo ========================================
echo 下载完成！
echo ========================================
echo.
echo 离线包保存在: offline_packages 目录
echo.
echo 在内网机器上安装时使用:
echo   pip install --no-index --find-links=offline_packages flask opencv-python numpy
echo.
pause
