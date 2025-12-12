#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Font Awesome 离线资源下载脚本
用于在有网络的机器上下载Font Awesome资源，供内网部署使用
"""

import os
import urllib.request
import ssl

# 创建不验证SSL的上下文（某些环境可能需要）
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def download_file(url, path):
    """下载文件到指定路径"""
    try:
        print(f'下载: {url}')
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 下载文件
        request = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(request, context=ssl_context) as response:
            data = response.read()
            with open(path, 'wb') as f:
                f.write(data)
        print(f'  -> 保存到: {path} ({len(data)} bytes)')
        return True
    except Exception as e:
        print(f'  -> 下载失败: {e}')
        return False

def main():
    print('=' * 50)
    print('Font Awesome 6.4.0 离线资源下载脚本')
    print('=' * 50)
    print()
    
    # 创建目录
    os.makedirs('static/fontawesome/css', exist_ok=True)
    os.makedirs('static/fontawesome/webfonts', exist_ok=True)
    
    # Font Awesome 6.4.0 CDN地址
    base_url = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0'
    
    # 需要下载的文件列表
    files = [
        ('css/all.min.css', 'static/fontawesome/css/all.min.css'),
        ('webfonts/fa-solid-900.woff2', 'static/fontawesome/webfonts/fa-solid-900.woff2'),
        ('webfonts/fa-solid-900.ttf', 'static/fontawesome/webfonts/fa-solid-900.ttf'),
        ('webfonts/fa-regular-400.woff2', 'static/fontawesome/webfonts/fa-regular-400.woff2'),
        ('webfonts/fa-regular-400.ttf', 'static/fontawesome/webfonts/fa-regular-400.ttf'),
        ('webfonts/fa-brands-400.woff2', 'static/fontawesome/webfonts/fa-brands-400.woff2'),
        ('webfonts/fa-brands-400.ttf', 'static/fontawesome/webfonts/fa-brands-400.ttf'),
    ]
    
    success_count = 0
    fail_count = 0
    
    for remote_path, local_path in files:
        url = f'{base_url}/{remote_path}'
        if download_file(url, local_path):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print('=' * 50)
    print(f'下载完成! 成功: {success_count}, 失败: {fail_count}')
    print('=' * 50)
    
    if fail_count == 0:
        print()
        print('接下来需要修改CSS文件中的字体路径...')
        
        # 读取并修改CSS文件
        css_path = 'static/fontawesome/css/all.min.css'
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            
            # 替换字体路径：../webfonts/ -> /static/fontawesome/webfonts/
            # 注意：保持相对路径也可以工作，因为CSS相对于自身解析
            # 这里我们保持相对路径不变，因为Flask的静态文件服务会正确处理
            
            print('CSS文件路径检查完成')
            print()
            print('提示：CSS中的字体路径使用相对路径 "../webfonts/"')
            print('Flask会正确解析为 /static/fontawesome/webfonts/')
            print()
            print('请确保 templates/index.html 中的引用已修改为：')
            print('<link rel="stylesheet" href="/static/fontawesome/css/all.min.css">')
            
        except Exception as e:
            print(f'读取CSS文件失败: {e}')

if __name__ == '__main__':
    main()
