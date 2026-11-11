# YOUO-ANPR · 车辆实时跟踪与车牌识别

基于 **YOLO 预训练模型** 的车辆检测跟踪 + 车牌识别系统，**全程不需要训练任何模型**：

- 车辆检测跟踪：`ultralytics` 预训练 YOLO 权重（COCO 的 car / bus / truck / motorcycle）+ 内置 **ByteTrack** 多目标跟踪
- 车牌定位：颜色掩膜 + 字符连通域聚类 + 边缘形态学多候选；若放入专用车牌 YOLO 权重则自动优先使用
- 车牌识别：**EasyOCR**（中文 + 英文），带车牌字符白名单、结果清洗、长度规则校验
- 多帧融合：同一 `track_id` 多次识别结果加权投票，减少单帧误识别
- 输出：实时画面标注、识别事件列表（CSV）、车牌抓拍小图、可选录像
- 两种使用方式：命令行（摄像头 / 视频 / 图片）与 Flask Web 界面（MJPEG 实时流 + 事件面板）

## 目录结构

```
YOUO-ANPR/
├── app.py                # Flask Web 服务（实时流 / 上传图片 / 上传视频）
├── main.py               # 命令行入口（摄像头 / 视频 / 单图）
├── anpr/
│   ├── config.py         # 全部可调参数
│   ├── detector.py       # YOLO 检测 + ByteTrack 跟踪
│   ├── plate_locator.py  # 车牌定位（启发式 + 可选专用模型）
│   ├── ocr.py            # EasyOCR 封装
│   ├── plate_text.py     # 号码清洗 / 校验 / 投票
│   ├── pipeline.py       # 端到端流程编排
│   ├── draw.py           # 绘制（支持中文标签）
│   ├── logger.py         # 事件 CSV 落盘
│   └── service.py        # Web 后台视频作业
├── templates/index.html  # Web 页面
├── static/               # 前端样式与脚本
├── models/               # 可选：放置专用车牌检测权重 plate.pt
├── outputs/              # 录像、事件 CSV、车牌抓拍
├── uploads/              # Web 上传的媒体
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

> 若 `import cv2` 报错或弹窗提示不支持 GUI，说明 `easyocr` 依赖把 OpenCV 换成了旧版 headless 包，执行：
> `pip install --force-reinstall --no-deps opencv-python`
>
> 若 `pyclipper` 编译失败（Python 版本过新缺 wheel），可跳过它：
> `pip install --no-deps easyocr`（本项目的识别流程不依赖 pyclipper）

首次运行会自动下载 `yolo26n.pt`（约 5 MB）和 EasyOCR 中英文模型（约 100 MB），需要联网。

## 命令行使用

```bash
# 摄像头实时识别（按 q 或 Esc 退出）
python main.py --source 0

# 处理视频文件并录制结果到 outputs/
python main.py --source demo.mp4 --save

# 单张图片识别，结果写入 outputs/<名称>_anpr.jpg
python main.py --image car.jpg --no-show

# 服务器 / 无显示器环境（不弹窗）
python main.py --source demo.mp4 --no-show --save
```

常用参数：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--source` | 摄像头序号或视频路径 | `0` |
| `--weights` | 车辆检测预训练权重 | `yolo26n.pt` |
| `--plate-weights` | 专用车牌 YOLO 权重（可选） | 自动查找 `models/plate.pt` |
| `--imgsz` | 检测输入尺寸，越大越准越慢 | `640` |
| `--conf` | 车辆检测置信度阈值 | `0.35` |
| `--device` | `auto` / `cpu` / `0`（GPU） | `auto` |
| `--ocr-every` | 同一目标每 N 帧识别一次 | `6` |
| `--min-votes` | 确认车牌所需最少识别次数 | `2` |
| `--save` / `--output` | 录制标注视频 | 关闭 |
| `--loop` | 视频文件循环播放 | 关闭 |

## Web 界面

```bash
python app.py                 # http://127.0.0.1:5000
python app.py --port 8000 --camera 1
```

页面功能：

- **启动摄像头**：浏览器内观看带检测框、车牌框与号码的 MJPEG 实时流
- **上传视频**：后台线程处理，实时刷新画面与识别记录
- **识别图片**：返回标注图与识别结果卡片（弹出窗口）
- **识别记录**：时间、车牌、车型、置信度、抓拍缩略图（点击查看原图）

接口一览：`GET /api/status`、`POST /api/start`、`POST /api/stop`、`POST /api/upload`、`GET /video_feed`、`GET /media/<path>`。

## 识别流程

1. `VehicleTracker.update(frame)` → YOLO 检测车辆，ByteTrack 分配 `track_id`
2. 取车辆框下部区域作为车牌 ROI
3. 生成候选：蓝色/绿色/黄色掩膜多尺度闭运算、字符连通域按行聚类、边缘 blackhat/tophat 形态学
4. 每个候选透视矫正裁剪 → EasyOCR 识别（放大 + CLAHE 增强，车牌字符白名单）
5. 清洗号码：去分隔符、混淆字符纠正（O→0 等）、长度规则（普通 7 位 / 新能源 8 位）
6. 候选排序：优先格式合法的号码，其次比置信度；仍无合法结果时用 OCR 文本行在整车下半部兜底
7. 按 `track_id` 加权投票，命中次数达到 `min_votes` 后确认并输出事件 + 抓拍

## 提升准确率

1. **放一个专用车牌检测模型**（推荐）：把任意车牌 YOLO 权重命名为 `models/plate.pt`，程序会自动优先使用它定位车牌，启发式逻辑作为兜底。
2. 摄像头尽量正对车辆、分辨率不低于 720p；夜晚建议补光。
3. 目标较小时调大 `--imgsz`（如 960/1280），或减小 `--ocr-every` 提高识别频次。
4. CPU 环境可改用更小的检测权重（`yolo26n.pt` / `yolo11n.pt`）。

## 输出文件

- `outputs/events.csv`：识别事件流水（时间、车牌、车型、置信度、是否合法、帧号）
- `outputs/plates/*.jpg`：确认车牌时的抓拍小图
- `outputs/record_*.mp4`：`--save` 录制的标注视频

## 常见问题

**摄像头打不开**：Windows 下优先用 `cv2.CAP_DSHOW` 打开（已内置），若仍失败请更换 `--source 1/2`，或先用 `python -c "import cv2;print(cv2.VideoCapture(0).isOpened())"` 排查。

**画面很卡**：CPU 推理本身较慢，属正常现象。可降低 `--imgsz`、增大 `--ocr-every`，或使用 GPU（`--device 0`）。注：安装 CUDA 版 PyTorch（最关键，当前装的是 CPU 版）

**识别到的是车身文字而不是车牌**：说明车牌过小或模糊。可放入 `models/plate.pt` 专用模型，或提高输入分辨率。

**首次运行很慢**：正在下载预训练权重与 OCR 模型，仅首次需要。
