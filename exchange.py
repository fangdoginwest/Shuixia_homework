from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/underwater_v1/weights/best.pt')

# 导出为 ONNX 格式
success = model.export(
    format='onnx',
    imgsz=640,  # 输入尺寸
    simplify=True,  # 简化模型
    opset=12,  # ONNX 算子集版本
    dynamic=False  # 固定输入尺寸
)