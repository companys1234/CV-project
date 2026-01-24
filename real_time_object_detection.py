import cv2
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO  # Для YOLOv8
import supervision as sv  # Для визуализации детекций


# =================== 1. ЗАГРУЗКА МОДЕЛИ YOLO ===================
def load_yolo_model(model_name='yolov8n.pt'):
    """
    Загрузка модели YOLO для детекции объектов
    Доступные модели: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    """
    print(f"Загрузка модели {model_name}...")

    # Устанавливаем устройство (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Загружаем модель YOLO
    model = YOLO(model_name)
    model.to(device)

    # Получаем имена классов COCO
    class_names = model.names

    print(f"Модель загружена. Доступно классов: {len(class_names)}")
    return model, class_names, device


# =================== 2. ФУНКЦИИ ДЛЯ ДЕТЕКЦИИ ===================
def detect_objects_in_frame(model, frame, device, confidence_threshold=0.5):
    """
    Детекция объектов в кадре
    """
    # Конвертируем BGR в RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Выполняем детекцию
    results = model(frame_rgb, conf=confidence_threshold, device=device)

    # Обрабатываем результаты
    detections = []

    if results and len(results) > 0:
        result = results[0]

        # Получаем bounding boxes, confidence scores и class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Формируем список детекций
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': model.names[class_id],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'width': x2 - x1,
                'height': y2 - y1
            })

    return detections, frame_rgb


def draw_detections(frame, detections, show_labels=True, show_confidence=True):
    """
    Рисует bounding boxes и метки на кадре
    """
    frame_with_detections = frame.copy()

    # Цвета для разных классов
    colors = {
        'person': (0, 255, 0),  # Зеленый
        'car': (255, 0, 0),  # Синий
        'truck': (0, 0, 255),  # Красный
        'bus': (255, 255, 0),  # Голубой
        'bicycle': (255, 0, 255),  # Фиолетовый
        'motorcycle': (0, 255, 255)  # Желтый
    }

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']

        # Выбираем цвет для класса
        color = colors.get(class_name, (255, 255, 255))  # Белый по умолчанию

        # Рисуем bounding box
        cv2.rectangle(frame_with_detections,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      color, 2)

        # Рисуем метку с классом и уверенностью
        if show_labels:
            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"

            # Фон для текста
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_with_detections,
                          (int(x1), int(y1) - text_size[1] - 10),
                          (int(x1) + text_size[0], int(y1)),
                          color, -1)

            # Текст
            cv2.putText(frame_with_detections, label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Рисуем центр объекта
        center_x, center_y = detection['center']
        cv2.circle(frame_with_detections,
                   (int(center_x), int(center_y)),
                   3, color, -1)

    return frame_with_detections


# =================== 3. ФУНКЦИИ СТАТИСТИКИ И АНАЛИЗА ===================
def analyze_detections(detections, frame_width, frame_height):
    """
    Анализ детекций для статистики
    """
    analysis = {
        'total_objects': len(detections),
        'by_class': {},
        'in_zones': {'center': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
        'average_confidence': 0,
        'largest_object': None,
        'smallest_object': None
    }

    if not detections:
        return analysis

    total_confidence = 0
    max_area = 0
    min_area = float('inf')

    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']

        # Статистика по классам
        analysis['by_class'][class_name] = analysis['by_class'].get(class_name, 0) + 1

        # Общая уверенность
        total_confidence += confidence

        # Анализ зон
        center_x, center_y = detection['center']

        if center_x < frame_width * 0.33:
            analysis['in_zones']['left'] += 1
        elif center_x > frame_width * 0.66:
            analysis['in_zones']['right'] += 1

        if center_y < frame_height * 0.33:
            analysis['in_zones']['top'] += 1
        elif center_y > frame_height * 0.66:
            analysis['in_zones']['bottom'] += 1

        if (frame_width * 0.33 < center_x < frame_width * 0.66 and
                frame_height * 0.33 < center_y < frame_height * 0.66):
            analysis['in_zones']['center'] += 1

        # Размер объекта
        area = detection['width'] * detection['height']
        if area > max_area:
            max_area = area
            analysis['largest_object'] = detection
        if area < min_area:
            min_area = area
            analysis['smallest_object'] = detection

    analysis['average_confidence'] = total_confidence / len(detections)

    return analysis


# =================== 4. ОСНОВНАЯ ФУНКЦИЯ С ДВУМЯ ОКНАМИ ===================
def real_time_object_detection_with_dual_window(camera_id=0, model_name='yolov8n.pt'):
    """Запуск детекции объектов в реальном времени с двумя окнами"""

    # Загружаем модель
    print("Инициализация модели детекции...")
    model, class_names, device = load_yolo_model(model_name)

    # Открываем камеру
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Не удалось подключиться к камере")
        return

    # Настройка параметров камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Создаем окна
    cv2.namedWindow('📷 Исходное видео', cv2.WINDOW_NORMAL)
    cv2.namedWindow('🔍 Детекция объектов', cv2.WINDOW_NORMAL)
    cv2.namedWindow('📊 Статистика', cv2.WINDOW_NORMAL)

    # Размещаем окна
    cv2.resizeWindow('📷 Исходное видео', 640, 480)
    cv2.resizeWindow('🔍 Детекция объектов', 640, 480)
    cv2.resizeWindow('📊 Статистика', 640, 480)

    cv2.moveWindow('📷 Исходное видео', 100, 100)
    cv2.moveWindow('🔍 Детекция объектов', 750, 100)
    cv2.moveWindow('📊 Статистика', 1400, 100)

    # Параметры детекции
    confidence_threshold = 0.5
    detection_interval = 0.1  # 10 FPS для детекции

    # Переменные для контроля
    last_detection_time = time.time()
    frame_count = 0
    start_time = time.time()

    # Буфер для статистики
    detection_history = []

    print("\n" + "=" * 80)
    print("СИСТЕМА ДЕТЕКЦИИ ОБЪЕКТОВ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 80)
    print("Окна:")
    print("  1. 📷 Исходное видео - Прямой эфир с камеры")
    print("  2. 🔍 Детекция объектов - Результаты обнаружения")
    print("  3. 📊 Статистика - Анализ детекций")
    print("\nУправление:")
    print("  'q' - Выход")
    print("  's' - Сохранить кадр с детекцией")
    print("  '+' - Увеличить порог уверенности")
    print("  '-' - Уменьшить порог уверенности")
    print("  '1' - Показать/скрыть метки")
    print("  '2' - Показать/скрыть уверенность")
    print("  '3' - Включить/выключить трекинг центра")
    print("  'c' - Сбросить статистику")
    print("=" * 80)

    # Настройки отображения
    show_labels = True
    show_confidence = True
    show_center_points = True

    # Основной цикл
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break

        frame_count += 1
        current_time = time.time()

        # ============ ОКНО 1: ИСХОДНОЕ ВИДЕО ============
        original_display = frame.copy()

        # Добавляем информацию
        fps = frame_count / (current_time - start_time)
        cv2.putText(original_display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(original_display, f"Порог: {confidence_threshold:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Рисуем зоны анализа
        h, w = frame.shape[:2]
        cv2.rectangle(original_display, (int(w * 0.33), int(h * 0.33)),
                      (int(w * 0.66), int(h * 0.66)), (0, 255, 255), 1)

        cv2.imshow('📷 Исходное видео', original_display)

        # ============ ВЫПОЛНЕНИЕ ДЕТЕКЦИИ ============
        if current_time - last_detection_time >= detection_interval:
            # Детекция объектов
            detections, processed_frame = detect_objects_in_frame(
                model, frame, device, confidence_threshold
            )

            # Анализ детекций
            analysis = analyze_detections(detections, w, h)

            # Сохраняем в историю
            detection_history.append({
                'time': current_time,
                'detections': detections,
                'analysis': analysis
            })

            # Ограничиваем историю
            if len(detection_history) > 100:
                detection_history.pop(0)

            last_detection_time = current_time

            # ============ ОКНО 2: ДЕТЕКЦИЯ ОБЪЕКТОВ ============
            detection_display = draw_detections(
                frame.copy(), detections, show_labels, show_confidence
            )

            # Добавляем количество объектов
            cv2.putText(detection_display, f"Объектов: {len(detections)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('🔍 Детекция объектов', detection_display)

            # ============ ОКНО 3: СТАТИСТИКА ============
            stats_display = np.zeros((480, 640, 3), dtype=np.uint8)
            stats_display[:] = (30, 30, 30)  # Темный фон

            y_offset = 40

            # Заголовок
            cv2.putText(stats_display, "СТАТИСТИКА ДЕТЕКЦИИ", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40

            # Общая информация
            cv2.putText(stats_display, f"Всего объектов: {analysis['total_objects']}",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30

            cv2.putText(stats_display, f"Средняя уверенность: {analysis['average_confidence']:.2f}",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30

            # Распределение по классам
            cv2.putText(stats_display, "Распределение по классам:",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25

            for class_name, count in analysis['by_class'].items():
                cv2.putText(stats_display, f"  {class_name}: {count}",
                            (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Распределение по зонам
            y_offset += 10
            cv2.putText(stats_display, "Распределение по зонам:",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 25

            zones = ['center', 'left', 'right', 'top', 'bottom']
            for zone in zones:
                cv2.putText(stats_display, f"  {zone}: {analysis['in_zones'][zone]}",
                            (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # История детекций (график)
            if len(detection_history) > 1:
                y_offset += 10
                cv2.putText(stats_display, "Тенденция (последние 50 кадров):",
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                y_offset += 25

                # Создаем простой график
                recent_counts = [d['analysis']['total_objects']
                                 for d in detection_history[-50:]]
                if recent_counts:
                    max_count = max(recent_counts) if max(recent_counts) > 0 else 1
                    graph_height = 100
                    graph_width = 400
                    graph_x = 100
                    graph_y = y_offset + 50

                    # Рисуем оси
                    cv2.rectangle(stats_display, (graph_x, graph_y),
                                  (graph_x + graph_width, graph_y + graph_height),
                                  (100, 100, 100), 1)

                    # Рисуем график
                    points = []
                    for i, count in enumerate(recent_counts):
                        x = graph_x + int(i * graph_width / len(recent_counts))
                        y = graph_y + graph_height - int((count / max_count) * graph_height)
                        points.append((x, y))

                    for i in range(len(points) - 1):
                        cv2.line(stats_display, points[i], points[i + 1],
                                 (0, 255, 255), 2)

            cv2.imshow('📊 Статистика', stats_display)

        # ============ ОБРАБОТКА КЛАВИШ ============
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Сохраняем кадр с детекцией
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'detection_{timestamp}.jpg', detection_display)
            print(f"Сохранено: detection_{timestamp}.jpg")
        elif key == ord('+'):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Порог уверенности: {confidence_threshold:.2f}")
        elif key == ord('-'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"Порог уверенности: {confidence_threshold:.2f}")
        elif key == ord('1'):
            show_labels = not show_labels
            print(f"Метки: {'ВКЛ' if show_labels else 'ВЫКЛ'}")
        elif key == ord('2'):
            show_confidence = not show_confidence
            print(f"Уверенность: {'ВКЛ' if show_confidence else 'ВЫКЛ'}")
        elif key == ord('3'):
            show_center_points = not show_center_points
            print(f"Точки центра: {'ВКЛ' if show_center_points else 'ВЫКЛ'}")
        elif key == ord('c'):
            detection_history = []
            print("Статистика сброшена")

    # ============ ЗАВЕРШЕНИЕ ============
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("ФИНАЛЬНАЯ СТАТИСТИКА:")
    print("=" * 80)
    print(f"Всего кадров: {frame_count}")
    print(f"Общее время: {total_time:.1f} сек")
    print(f"Средний FPS: {frame_count / total_time:.1f}")

    if detection_history:
        total_detections = sum([len(d['detections']) for d in detection_history])
        avg_detections = total_detections / len(detection_history)
        print(f"Среднее объектов на кадр: {avg_detections:.2f}")

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


# =================== 5. ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ===================
def detect_in_image(image_path, model_name='yolov8n.pt', confidence_threshold=0.5):
    """Детекция объектов на изображении"""
    print(f"Обработка изображения: {image_path}")

    # Загружаем модель
    model, class_names, device = load_yolo_model(model_name)

    # Читаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось загрузить изображение")
        return

    # Выполняем детекцию
    detections, _ = detect_objects_in_frame(model, image, device, confidence_threshold)

    # Рисуем детекции
    result_image = draw_detections(image, detections)

    # Показываем результат
    cv2.imshow('Результат детекции', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Выводим статистику
    print(f"\nНайдено объектов: {len(detections)}")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class_name']} ({det['confidence']:.2f}) - "
              f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")

    # Сохраняем результат
    output_path = image_path.replace('.', '_detected.')
    cv2.imwrite(output_path, result_image)
    print(f"Результат сохранен: {output_path}")


# =================== 6. ЗАПУСК ПРОГРАММЫ ===================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Детекция объектов в реальном времени')
    parser.add_argument('--camera', type=int, default=0, help='ID камеры')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
                                 'yolov8l.pt', 'yolov8x.pt'],
                        help='Модель YOLO для использования')
    parser.add_argument('--image', type=str, help='Путь к изображению для детекции')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Порог уверенности')

    args = parser.parse_args()


    # Проверяем доступность камеры
    def check_camera():
        for i in range(3):
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                print(f"Найдена камера с индексом {i}")
                cap_test.release()
                return i
            cap_test.release()
        return 0


    if args.image:
        # Режим детекции на изображении
        detect_in_image(args.image, args.model, args.confidence)
    else:
        # Режим реального времени
        camera_id = check_camera()
        print(f"Используется камера: {camera_id}")
        print(f"Используется модель: {args.model}")

        # Установка библиотек (если нужно)
        try:
            import ultralytics
        except ImportError:
            print("\nУстановите необходимые библиотеки:")
            print("pip install ultralytics")
            print("pip install supervision")
            exit(1)

        # Запускаем систему детекции
        real_time_object_detection_with_dual_window(camera_id, args.model)