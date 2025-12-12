import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import time
import base64
import requests
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Apply affine transform calculated using src_tri and dst_tri to src and
    output an image of size.
    """
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    """
    Warps a rectangular region from img1 to img2 defined by t1 and t2 triangles.
    """
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []
    
    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])
    
    if size[0] > 0 and size[1] > 0 and img1_rect.size > 0:
        img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
        
        img2_rect = img2_rect * mask

        # Copy triangular region of the rectangular patch to the output image
        # Be careful with boundaries
        y1, y2 = r2[1], r2[1]+r2[3]
        x1, x2 = r2[0], r2[0]+r2[2]
        
        # Ensure we don't go out of bounds
        if y1 < img2.shape[0] and x1 < img2.shape[1] and y2 > 0 and x2 > 0:
             # Slice intersection
             img2_slice = img2[y1:y2, x1:x2]
             
             # If slice size matches (it should unless out of bounds clipping happened implicitly, but numpy handles slicing gracefully by returning smaller)
             # We need to make sure mask and img2_rect are also sliced if they are larger than the target area (e.g. at image edges)
             
             h_slice, w_slice = img2_slice.shape[:2]
             if h_slice != r2[3] or w_slice != r2[2]:
                 # This is a simplified handling, for robust edge cases more logic is needed
                 # But for now assuming points are within image
                 pass
             else:
                img2[y1:y2, x1:x2] = img2[y1:y2, x1:x2] * ((1.0, 1.0, 1.0) - mask)
                img2[y1:y2, x1:x2] = img2[y1:y2, x1:x2] + img2_rect

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_face', methods=['POST'])
def detect_face():
    """
    Detect face landmarks and contours in an uploaded image.
    Returns detected facial features as points.
    """
    try:
        # Get the image from request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image data
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to load image'}), 400
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': '未检测到人脸，请确保图片中有清晰的正面人脸'})
        
        # Use the first (largest) face
        x, y, w, h = faces[0]
        
        # Get face region
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(15, 15))
        
        # Sort eyes by x position to get left and right
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Build landmarks dictionary for frontend
        landmarks = {}
        
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Max 2 eyes
            # Inner and outer corners (approximate)
            eye_inner_x = x + ex + int(ew * 0.2) if i == 0 else x + ex + int(ew * 0.8)
            eye_outer_x = x + ex + int(ew * 0.8) if i == 0 else x + ex + int(ew * 0.2)
            eye_corner_y = y + ey + eh // 2
            
            if i == 0:
                landmarks['left_inner_eye'] = [int(eye_inner_x), int(eye_corner_y)]
                landmarks['left_outer_eye'] = [int(eye_outer_x), int(eye_corner_y)]
            else:
                landmarks['right_inner_eye'] = [int(eye_inner_x), int(eye_corner_y)]
                landmarks['right_outer_eye'] = [int(eye_outer_x), int(eye_corner_y)]
        
        # Estimate other facial landmarks based on face proportions
        face_center_x = x + w // 2
        
        # Eyebrow center (approximately at 1/3 of face height from top)
        landmarks['eyebrow_center'] = [int(face_center_x), int(y + h * 0.33)]
        
        # Nose bottom (approximately at 2/3 of face height from top)
        landmarks['nose_bottom'] = [int(face_center_x), int(y + h * 0.67)]
        
        # Lip center (approximately at 3/4 of face height from top)
        landmarks['lip_center'] = [int(face_center_x), int(y + h * 0.78)]
        
        # Chin bottom
        landmarks['chin_bottom'] = [int(face_center_x), int(y + h)]
        
        # Forehead top
        landmarks['forehead_top'] = [int(face_center_x), int(y)]
        
        # Face contour points (simplified ellipse-like contour)
        contour = []
        for angle in range(0, 360, 15):
            rad = np.radians(angle)
            cx = face_center_x + int((w // 2) * np.cos(rad))
            cy = y + h // 2 + int((h // 2) * np.sin(rad))
            contour.append([int(cx), int(cy)])
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'contour': contour,
            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Get data from request
        target_file = request.files.get('target') or request.files.get('target_image')
        source_file = request.files.get('source') or request.files.get('source_image')
        
        if not target_file or not source_file:
            return jsonify({'success': False, 'error': 'Missing target or source image'}), 400
        
        # Parse points (JSON strings) - supports both [[x,y],...] and [{x:...,y:...},...]
        target_points_data = json.loads(request.form['target_points'])
        source_points_data = json.loads(request.form['source_points'])
        
        # Read images directly from file bytes
        target_bytes = np.frombuffer(target_file.read(), np.uint8)
        source_bytes = np.frombuffer(source_file.read(), np.uint8)
        
        img_target = cv2.imdecode(target_bytes, cv2.IMREAD_COLOR)
        img_source = cv2.imdecode(source_bytes, cv2.IMREAD_COLOR)
        
        if img_target is None or img_source is None:
            return jsonify({'success': False, 'error': 'Failed to load images'}), 400

        # Convert points to list of tuples - handle both formats
        points_target = []
        points_source = []
        
        for p in target_points_data:
            if isinstance(p, dict):
                points_target.append((int(p['x']), int(p['y'])))
            else:
                # Format is [x, y]
                points_target.append((int(p[0]), int(p[1])))
            
        for p in source_points_data:
            if isinstance(p, dict):
                points_source.append((int(p['x']), int(p['y'])))
            else:
                # Format is [x, y]
                points_source.append((int(p[0]), int(p[1])))
        
        if len(points_target) < 3 or len(points_source) < 3:
            return jsonify({'success': False, 'error': '需要至少3个点才能进行三角剖分'}), 400
        
        if len(points_target) != len(points_source):
            return jsonify({'success': False, 'error': f'点数不匹配：目标{len(points_target)}个，源{len(points_source)}个'}), 400
            
        # Create a copy of target image for warping
        img_warped = np.copy(img_target).astype(np.float32)
        img_source_float = img_source.astype(np.float32)
        
        # Delaunay Triangulation on Source Points
        rect = (0, 0, img_source.shape[1], img_source.shape[0])
        subdiv = cv2.Subdiv2D(rect)
        
        # Insert points into subdiv - filter valid points
        valid_source_indices = []
        for i, p in enumerate(points_source):
            if 0 <= p[0] < img_source.shape[1] and 0 <= p[1] < img_source.shape[0]:
                subdiv.insert(p)
                valid_source_indices.append(i)
            
        triangle_list = subdiv.getTriangleList()
        
        # Map triangles to indices
        source_indices = []
        for t in triangle_list:
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            
            # Find indices of these points in the points_source list
            indices = []
            for pt in pts:
                found = False
                for i, p in enumerate(points_source):
                    if abs(pt[0] - p[0]) < 1.0 and abs(pt[1] - p[1]) < 1.0:
                        indices.append(i)
                        found = True
                        break
                if not found:
                    break
            
            if len(indices) == 3:
                source_indices.append(indices)
        
        # Warp each triangle
        for indices in source_indices:
            t_source = [points_source[indices[0]], points_source[indices[1]], points_source[indices[2]]]
            t_target = [points_target[indices[0]], points_target[indices[1]], points_target[indices[2]]]
            
            try:
                warp_triangle(img_source_float, img_warped, t_source, t_target)
            except Exception as e:
                print(f"Warning: Triangle warp failed: {e}")
                continue
            
        # Convert back to uint8
        img_warped = np.clip(img_warped, 0, 255).astype(np.uint8)
        
        # 创建融合遮罩 - 使用凸包定义填补区域
        mask = np.zeros(img_target.shape[:2], dtype=np.uint8)
        hull_points = cv2.convexHull(np.array(points_target, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull_points, 255)
        
        # 计算凸包的中心点作为seamlessClone的插入点
        moments = cv2.moments(hull_points)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            # 如果凸包面积为0，使用凸包的边界框中心
            rect = cv2.boundingRect(hull_points)
            center_x = rect[0] + rect[2] // 2
            center_y = rect[1] + rect[3] // 2
        
        center = (center_x, center_y)
        
        # 使用OpenCV的seamlessClone进行泊松融合
        # NORMAL_CLONE保留源图像的颜色，同时平滑边界
        try:
            output = cv2.seamlessClone(img_warped, img_target, mask, center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            # 如果seamlessClone失败，回退到简单混合
            print(f"seamlessClone failed: {e}, falling back to simple blend")
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            output = (img_warped.astype(np.float32) * mask_3ch + 
                      img_target.astype(np.float32) * (1 - mask_3ch))
            output = np.clip(output, 0, 255).astype(np.uint8)

        # Encode to base64
        _, buffer = cv2.imencode('.png', output)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/face_compare', methods=['POST'])
def face_compare():
    """
    人像比对接口 - 调用科达API进行人脸搜索
    支持完整的API参数：
    - algorithmList: 算法列表
    - algorithmStrategy: 算法策略 (default/all)
    - count: 最大返回记录数
    - similarityThreshold: 相似度阈值
    - startTime/endTime: 时间范围
    - deviceList: 设备列表
    - linkedQuery: 是否关联查询
    - multipleArrage: 综合排序 (0=算法分组, 1=整合排序)
    - returnImageData: 是否返回base64图片数据
    - caseId/caseName: 案件信息
    - comparisonReason: 比对事由
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': '未提供请求数据'}), 400
        
        # 获取图片数据（base64格式）
        image_data = data.get('imageData')
        if not image_data:
            return jsonify({'success': False, 'error': '未提供图片数据'}), 400
        
        # 移除base64前缀（如果有）
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # API配置
        api_url = "http://10.151.151.83:8080/dataexchangeserver/apiPayload/qt_keda_imageSearchGj/ODI3Y2QxZTgtNGVlZS00ZDUxLThkYjMtNGE0MGMzYWJmMGRl"
        
        # 计算默认时间范围（最近30天）
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        # 构建请求参数 - 支持所有API参数
        request_params = {
            "imageData": image_data,
            "algorithmStrategy": data.get('algorithmStrategy', 'all'),
            "multipleArrage": int(data.get('multipleArrage', 1)),
            "count": int(data.get('count', 10)),
            "similarityThreshold": int(data.get('similarityThreshold', 80)),
            "startTime": data.get('startTime', start_time.strftime("%Y-%m-%d %H:%M:%S")),
            "endTime": data.get('endTime', end_time.strftime("%Y-%m-%d %H:%M:%S")),
            "linkedQuery": str(data.get('linkedQuery', "0")),
            "returnImageData": str(data.get('returnImageData', "1"))
        }
        
        # 可选参数 - 算法列表
        if data.get('algorithmList'):
            request_params['algorithmList'] = data.get('algorithmList')
        
        # 可选参数 - 设备列表
        if data.get('deviceList'):
            request_params['deviceList'] = data.get('deviceList')
        
        # 可选参数 - 案件信息
        if data.get('caseId'):
            request_params['caseId'] = data.get('caseId')
        if data.get('caseName'):
            request_params['caseName'] = data.get('caseName')
        
        # 可选参数 - 比对事由
        if data.get('comparisonReason'):
            request_params['comparisonReason'] = data.get('comparisonReason')
        
        # 可选参数 - 身份证号和姓名
        if data.get('sfzh'):
            request_params['sfzh'] = data.get('sfzh')
        if data.get('name'):
            request_params['name'] = data.get('name')
        
        # 发送API请求
        headers = {
            'Content-Type': 'application/json'
        }
        
        print(f"[DEBUG] 发送API请求到: {api_url}")
        print(f"[DEBUG] 请求参数: algorithmStrategy={request_params.get('algorithmStrategy')}, count={request_params.get('count')}, similarityThreshold={request_params.get('similarityThreshold')}")
        print(f"[DEBUG] 时间范围: {request_params.get('startTime')} 至 {request_params.get('endTime')}")
        
        response = requests.post(api_url, json=request_params, headers=headers, timeout=60)
        
        print(f"[DEBUG] API响应状态码: {response.status_code}")
        print(f"[DEBUG] API响应内容: {response.text[:500] if response.text else 'Empty'}...")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[DEBUG] 解析后的结果: code={result.get('code')}, status={result.get('status')}, message={result.get('message')}")
            
            # 解析返回结果 - 兼容多种返回格式
            # 检查是否是后端服务错误
            if result.get('code') == 'INTERNAL_SERVER_ERROR' or 'error' in str(result.get('msg', '')).lower():
                error_msg = result.get('msg', '未知错误')
                # 提取更友好的错误信息
                if 'timed out' in error_msg.lower() or 'timeout' in error_msg.lower():
                    return jsonify({
                        'success': False,
                        'error': '人脸比对服务响应超时，请稍后重试。这可能是由于服务器繁忙或图片过大导致。',
                        'detail': error_msg
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'人脸比对服务内部错误: {error_msg}',
                        'detail': error_msg
                    })
            
            if result.get('code') == '0' or result.get('code') == 0 or result.get('status') == 200 or result.get('result') == '0':
                # 处理成功
                compare_results = []
                
                # 检查result字段的类型
                result_data = result.get('result')
                if result_data == '0':
                    # result为'0'表示成功但无数据
                    return jsonify({
                        'success': True,
                        'results': [],
                        'totalCount': 0,
                        'message': '比对完成，未找到匹配结果'
                    })
                
                if result_data and isinstance(result_data, list):
                    for algorithm_result in result_data:
                        algorithm_name = algorithm_result.get('algorithmName', '未知算法')
                        algorithm_vendor = algorithm_result.get('algorithmVendor', '未知厂商')
                        
                        if algorithm_result.get('algorithmResultList'):
                            for face_result in algorithm_result['algorithmResultList']:
                                compare_results.append({
                                    'algorithmName': algorithm_name,
                                    'algorithmVendor': algorithm_vendor,
                                    'similarity': face_result.get('similarity', '0'),
                                    'shotTime': face_result.get('shotTime', ''),
                                    'deviceName': face_result.get('deviceName', ''),
                                    'faceUrl': face_result.get('faceUrl', ''),
                                    'faceData': face_result.get('faceData', ''),
                                    'overviewUrl': face_result.get('overviewUrl', ''),
                                    'overviewData': face_result.get('overviewData', ''),
                                    'personUrl': face_result.get('personUrl', ''),
                                    'personData': face_result.get('personData', ''),
                                    'faceId': face_result.get('faceId', ''),
                                    'name': face_result.get('name', ''),
                                    'idNumber': face_result.get('idNumber', '')
                                })
                
                return jsonify({
                    'success': True,
                    'results': compare_results,
                    'totalCount': len(compare_results),
                    'timeElapsed': result.get('timeElapsed', 0)
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f"API返回错误: {result.get('message', '未知错误')}",
                    'code': result.get('code')
                })
        else:
            return jsonify({
                'success': False,
                'error': f"API请求失败，状态码: {response.status_code}"
            }), 500
            
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'API请求超时，请稍后重试'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': 'API连接失败，请检查网络连接'}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
