from ultralytics import YOLO
import time

if __name__ == '__main__':
    test_img = r"D:\算法\test.MP4"
    conf = 0.25
    iou = 0.45

    # ---------- pt推理 ----------
    model_pt = YOLO(r"D:\算法\best.pt")
    model_pt(test_img, conf=conf, iou=iou)
    t0 = time.time()
    res_pt = model_pt(test_img, conf=conf, iou=iou, save=True, save_dir="results/pt")
    t_pt = time.time() - t0
    print(f"PT耗时：{t_pt:.4f} s")

    # ---------- onnx推理 ----------
    model_onnx = YOLO(r"D:\算法\best.onnx")
    model_onnx(test_img, conf=conf, iou=iou)
    t1 = time.time()
    res_onnx = model_onnx(test_img, conf=conf, iou=iou, save=True, save_dir="results/onnx")
    t_onnx = time.time() - t1
    print(f"ONNX耗时：{t_onnx:.4f} s")

    num_pt = len(res_pt[0].boxes)
    num_onnx = len(res_onnx[0].boxes)
    print(f"PT检测目标数：{num_pt}")
    print(f"ONNX检测目标数：{num_onnx}")
