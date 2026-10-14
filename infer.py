from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test.MP4", show=True, save=True)

print("推理完成！请去 runs/detect 文件夹下查看生成的视频。")