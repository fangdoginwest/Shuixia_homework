import cv2
import argparse
import time
from inference import YOLOv8Inference

# 固定窗口大小
cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv8 Inference', 1280, 720)  # 固定窗口大小


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference on Atlas 200I DK A2')
    parser.add_argument('--model', type=str, required=True, help='Path to OM model')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--display-scale', type=float, default=0.6, help='Display window scale (0.1-1.0)')
    parser.add_argument('--skip-frames', type=int, default=0, help='Number of frames to skip between processing')
    parser.add_argument('--show-fps', action='store_true', help='Show FPS on video')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for detections')
    args = parser.parse_args()

    # 初始化推理引擎
    detector = YOLOv8Inference(args.model)

    # 打开视频文件
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 控制变量
    paused = False
    frame_count = 0
    display_counter = 0  # 用于控制显示频率

    # 创建可调整大小的窗口
    cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8 Inference',
                     int(width * args.display_scale),
                     int(height * args.display_scale))

    print(f"Starting video processing. Press 'SPACE' to pause, '→' to step, '←' to step back, 'Q' to quit.")

    # 性能计数器
    last_time = time.time()
    fps_counter = 0
    avg_fps = 0

    while cap.isOpened():
        if not paused:
            # 跳过指定帧数
            for _ in range(args.skip_frames + 1):
                ret, frame = cap.read()
                frame_count += 1
                if not ret:
                    break

            if not ret:
                break

            # 执行推理
            detections = detector.infer(frame)

            # 绘制结果
            frame = detector.draw_results(frame, detections)

            # 添加调试信息
            print(f"Frame {frame_count}: Detections count = {len(detections)}")

            # 显示进度
            progress = f"Frame: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)"
            cv2.putText(frame, progress, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # 显示状态
            status = "PAUSED" if paused else "PLAYING"
            cv2.putText(frame, f"Status: {status}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if paused else (0, 255, 0), 1)

            # 保存帧
            out.write(frame)

            # 计算实时FPS
            fps_counter += 1
            if time.time() - last_time >= 1.0:
                avg_fps = fps_counter / (time.time() - last_time)
                fps_counter = 0
                last_time = time.time()

            if args.show_fps:
                cv2.putText(frame, f"Realtime FPS: {avg_fps:.1f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 显示处理后的帧（每2帧显示一次以提升性能）
            if display_counter % 2 == 0:
                display_frame = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
                cv2.imshow('YOLOv8 Inference', display_frame)
            display_counter += 1

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # 空格键暂停/继续
            paused = not paused
        elif key == ord('q'):  # 退出
            break
        elif key == 83:  # 右箭头 (前进一帧)
            if paused:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    detections = detector.infer(frame)
                    frame = detector.draw_results(frame, detections)
                    display_frame = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
                    cv2.imshow('YOLOv8 Inference', display_frame)
                    out.write(frame)
        elif key == 81:  # 左箭头 (后退一帧)
            if paused and frame_count > 1:
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 2)
                frame_count -= 2
                paused = False

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    detector.release()

    # 打印统计信息
    print("\n===== Processing Summary =====")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {detector.total_time:.2f} seconds")
    print(f"Average FPS: {frame_count / detector.total_time:.2f}")
    print("Detection counts:")
    for class_name, count in detector.detection_counts.items():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()