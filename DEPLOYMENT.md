# 三庭五眼人脸五官修复系统 - 内网部署指南

## 一、项目结构

```
3part5eyes/
├── app.py                 # Flask后端主程序
├── requirements.txt       # Python依赖
├── templates/
│   └── index.html        # 前端页面
├── static/
│   ├── uploads/          # 上传文件临时目录
│   └── fontawesome/      # Font Awesome离线资源（需下载）
└── examples/             # 示例图片
```

## 二、环境准备

### 2.1 Python环境要求
- Python 3.8 或更高版本
- pip 包管理器

### 2.2 下载离线依赖包

**在有网络的机器上执行以下操作：**

```bash
# 创建依赖下载目录
mkdir offline_packages

# 下载所有依赖包到本地
pip download -d offline_packages flask opencv-python numpy

# 如果目标机器是不同操作系统/架构，需要指定平台
# Windows x64:
pip download -d offline_packages --platform win_amd64 --python-version 38 --only-binary=:all: flask opencv-python numpy

# Linux x64:
pip download -d offline_packages --platform manylinux2014_x86_64 --python-version 38 --only-binary=:all: flask opencv-python numpy
```

### 2.3 下载Font Awesome离线资源

**方法一：手动下载**
1. 访问 https://fontawesome.com/download
2. 下载 "Free for Web" 版本
3. 解压后将 `webfonts` 和 `css` 文件夹复制到 `static/fontawesome/` 目录

**方法二：使用CDN下载脚本**
在有网络的机器上运行以下Python脚本：

```python
import os
import urllib.request

# 创建目录
os.makedirs('static/fontawesome/css', exist_ok=True)
os.makedirs('static/fontawesome/webfonts', exist_ok=True)

# Font Awesome 6.4.0 文件列表
base_url = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0'

files = {
    'css/all.min.css': 'css/all.min.css',
    'webfonts/fa-solid-900.woff2': 'webfonts/fa-solid-900.woff2',
    'webfonts/fa-solid-900.ttf': 'webfonts/fa-solid-900.ttf',
    'webfonts/fa-regular-400.woff2': 'webfonts/fa-regular-400.woff2',
    'webfonts/fa-regular-400.ttf': 'webfonts/fa-regular-400.ttf',
    'webfonts/fa-brands-400.woff2': 'webfonts/fa-brands-400.woff2',
    'webfonts/fa-brands-400.ttf': 'webfonts/fa-brands-400.ttf',
}

for remote, local in files.items():
    url = f'{base_url}/{remote}'
    path = f'static/fontawesome/{local}'
    print(f'下载: {url}')
    try:
        urllib.request.urlretrieve(url, path)
        print(f'  -> 保存到: {path}')
    except Exception as e:
        print(f'  -> 下载失败: {e}')

print('下载完成！')
```

## 三、打包项目文件

### 3.1 需要打包的文件清单

```
3part5eyes/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── uploads/          (空目录，需创建)
│   └── fontawesome/      (离线资源)
│       ├── css/
│       │   └── all.min.css
│       └── webfonts/
│           ├── fa-solid-900.woff2
│           ├── fa-solid-900.ttf
│           ├── fa-regular-400.woff2
│           ├── fa-regular-400.ttf
│           ├── fa-brands-400.woff2
│           └── fa-brands-400.ttf
├── offline_packages/     (离线Python包)
└── examples/             (可选，示例图片)
```

### 3.2 打包命令

```bash
# 压缩项目（不包含虚拟环境和临时文件）
tar -czvf 3part5eyes_offline.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='venv' \
    3part5eyes/

# 或者使用 zip（Windows）
# 右键项目文件夹 -> 发送到 -> 压缩(zipped)文件夹
```

## 四、内网机器部署步骤

### 4.1 解压项目

```bash
# Linux/Mac
tar -xzvf 3part5eyes_offline.tar.gz
cd 3part5eyes

# Windows
# 右键解压到当前文件夹
```

### 4.2 创建Python虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4.3 安装离线依赖

```bash
# 使用离线包安装
pip install --no-index --find-links=offline_packages flask opencv-python numpy

# 或者如果内网有PyPI镜像
pip install -i http://内网镜像地址/simple --trusted-host 内网镜像地址 flask opencv-python numpy
```

### 4.4 确保目录结构正确

```bash
# 创建上传目录（如果不存在）
mkdir -p static/uploads
```

### 4.5 启动服务

```bash
# 开发模式（仅本机访问）
python app.py

# 生产模式（允许局域网访问）
# 方法1：修改app.py最后一行
# app.run(debug=False, host='0.0.0.0', port=5000)

# 方法2：使用命令行参数
flask run --host=0.0.0.0 --port=5000

# 方法3：使用Waitress（推荐用于Windows生产环境）
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app

# 方法4：使用Gunicorn（推荐用于Linux生产环境）
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 4.6 访问应用

- 本机访问: http://localhost:5000
- 局域网访问: http://服务器IP:5000

## 五、修改前端使用离线Font Awesome

需要修改 `templates/index.html` 文件，将CDN链接改为本地路径：

**修改前：**
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
```

**修改后：**
```html
<link rel="stylesheet" href="/static/fontawesome/css/all.min.css">
```

同时需要修改 `static/fontawesome/css/all.min.css` 文件中的字体路径：

将所有 `../webfonts/` 替换为 `/static/fontawesome/webfonts/`

或者保持相对路径不变（`../webfonts/`），因为CSS相对于自身路径解析。

## 六、防火墙配置（如需局域网访问）

### Windows:
```powershell
# 允许入站5000端口
netsh advfirewall firewall add rule name="Flask App" dir=in action=allow protocol=TCP localport=5000
```

### Linux:
```bash
# 使用firewalld
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload

# 使用ufw
sudo ufw allow 5000/tcp
```

## 七、常见问题

### Q1: 启动报错 "No module named 'cv2'"
**A:** OpenCV未正确安装，检查离线包是否与目标系统匹配（Windows/Linux/Mac，32/64位）

### Q2: 页面图标显示为方框
**A:** Font Awesome资源未正确加载，检查：
1. `static/fontawesome/` 目录是否存在
2. CSS文件中的字体路径是否正确
3. 浏览器开发者工具(F12)查看网络请求是否有404错误

### Q3: 上传图片失败
**A:** 检查 `static/uploads/` 目录是否存在且有写入权限

### Q4: 局域网无法访问
**A:** 检查：
1. 是否使用 `host='0.0.0.0'` 启动
2. 防火墙是否放行5000端口
3. 服务器IP是否正确

## 八、一键部署脚本

### Windows (deploy.bat)
```batch
@echo off
echo 三庭五眼人脸修复系统 - 部署脚本
echo ================================

:: 创建虚拟环境
if not exist venv (
    echo 创建虚拟环境...
    python -m venv venv
)

:: 激活虚拟环境
call venv\Scripts\activate

:: 安装依赖
echo 安装依赖...
pip install --no-index --find-links=offline_packages flask opencv-python numpy 2>nul
if errorlevel 1 (
    echo 离线包安装失败，尝试在线安装...
    pip install flask opencv-python numpy
)

:: 创建必要目录
if not exist static\uploads mkdir static\uploads

:: 启动服务
echo 启动服务...
echo 访问地址: http://localhost:5000
python app.py

pause
```

### Linux/Mac (deploy.sh)
```bash
#!/bin/bash
echo "三庭五眼人脸修复系统 - 部署脚本"
echo "================================"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install --no-index --find-links=offline_packages flask opencv-python numpy 2>/dev/null
if [ $? -ne 0 ]; then
    echo "离线包安装失败，尝试在线安装..."
    pip install flask opencv-python numpy
fi

# 创建必要目录
mkdir -p static/uploads

# 启动服务
echo "启动服务..."
echo "访问地址: http://localhost:5000"
python app.py
```

## 九、版本信息

- Python: 3.8+
- Flask: 最新稳定版
- OpenCV: 最新稳定版
- Font Awesome: 6.4.0
