from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("best.pt")
    success = model.export(format="onnx", imgsz=640)
    print("导出结果：", success)
