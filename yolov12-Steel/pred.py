import numpy as np
from ultralytics import YOLO
import cv2
import math

# 设置字体样式
font = cv2.FONT_HERSHEY_DUPLEX


# 在图像上添加带背景的文本
def add_text_with_background(
        image,
        text,
        position,
        font_face,
        font_scale,
        text_color,
        bg_color,
        thickness=1,
        padding=5,
):
    # 获取文本大小
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )

    # 计算文本框的大小（包括内边距）
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding + baseline

    # 确保文本框不会超出图像边界
    x, y = position
    x = max(0, min(x, image.shape[1] - box_width))
    y = max(box_height, min(y, image.shape[0]))

    # 绘制背景矩形
    cv2.rectangle(image, (x, y - box_height), (x + box_width, y), bg_color, -1)

    # 绘制文本
    text_position = (x + padding, y - padding - baseline)
    cv2.putText(
        image, text, text_position, font_face, font_scale, text_color, thickness
    )


# 获取适合文本宽度的最优字体大小
def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=2
        )
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


# 加载YOLO模型
model = YOLO("./runs/detect/yolov12s_300e/weights/best.pt")

# 定义类别名称
classNames = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


# 进行预测
def pred(img_path, stream=False):
    # 检查输入是图片路径还是图像数组
    if isinstance(img_path, str):
        # 如果是字符串，则加载图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误：无法加载图像 {img_path}")
            return None, None
    else:
        # 如果已经是图像数组，直接使用
        img = img_path

    orig = img.copy()

    # 使用YOLO模型进行预测
    results = model(img, stream=stream)
    for r in results:
        boxes = r.boxes
        if boxes is not None:  # 添加检查，确保有检测结果
            for box in boxes:
                # 获取边界框的坐标
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # 获取置信度并进行四舍五入
                conf = math.ceil((box.conf[0] * 100)) / 100

                # 获取类别索引并转换为类别名称
                cls = box.cls[0]
                name = classNames[int(cls)]
                print(f"{name} {conf}")

                # 获取适合文本的最优字体大小
                # font_scale = get_optimal_font_scale(f"{name} {conf}", w)
                thick = 1 if w < 210 else 2

                # 根据类别名称选择颜色并绘制边界框和文本
                cv2.rectangle(
                    img=img,
                    pt1=(x1, y1),
                    pt2=(x1 + w, y1 + h),
                    color=(0, 102, 255),  # bgr
                    thickness=2,
                )
                add_text_with_background(
                    img,
                    f"{name} {conf}",
                    (x1, y1),
                    font,
                    1.1,
                    (255, 255, 255),
                    (0, 102, 255),  # bgr
                    thick,
                    5,
                )

    return orig, img


# 主程序入口
if __name__ == "__main__":
    # 进行预测
    original_img, result_img = pred("/Users/luojiehao/Desktop/服务器/Steel-defect-model-based-on-yolov12/yolov12钢材检测/data/test/images/crazing_21_jpg.rf.ba60da711d22af5e1933388cca662731.jpg")

    if result_img is not None:
        # 显示结果（可选）
        cv2.imshow('Original', original_img)
        cv2.imshow('Detection Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果（可选）
        cv2.imwrite('detection_result.jpg', result_img)
        print("检测完成，结果已保存为 detection_result.jpg")
    else:
        print("检测失败")