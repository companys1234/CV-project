import cv2
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

# =================== 1. ОПРЕДЕЛЕНИЕ КЛАССОВ CIFAR-10 ===================
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# =================== 2. ТРАНСФОРМАЦИИ И ЗАГРУЗКА ДАННЫХ ===================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Загружаем датасет CIFAR-10 (у вас было CIFAR10 в коде)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# =================== 3. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ ===================
def create_model():
    """Создает и возвращает модель ResNet18 для CIFAR-10"""
    model = models.resnet18(pretrained=True)

    # Изменяем последний слой для CIFAR-10 (10 классов)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # Переносим модель на CPU (или GPU если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    model = model.to(device)

    return model, device


def train_model(model, trainloader, device, num_epochs=5):
    """Обучение модели"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Начинаем обучение...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Вычисляем точность
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{i + 1}/{len(trainloader)}], '
                      f'Loss: {running_loss / 100:.4f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0

    print('Обучение завершено!')
    return model


# =================== 4. ФУНКЦИИ ДЛЯ ОБРАБОТКИ КАДРОВ ===================
def preprocess_frame_for_model(frame, device):
    """Подготовка кадра камеры для модели"""
    # Изменяем размер до 32x32 как в CIFAR-10
    frame_resized = cv2.resize(frame, (32, 32))

    # Конвертируем BGR (OpenCV) в RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Конвертируем в PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Применяем трансформации
    input_tensor = transform(pil_image)

    # Добавляем batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Перемещаем на нужное устройство
    input_batch = input_batch.to(device)

    return input_batch, frame_resized


def predict_frame(model, frame, device):
    """Предсказание класса для кадра"""
    # Подготавливаем кадр
    input_batch, frame_resized = preprocess_frame_for_model(frame, device)

    # Делаем предсказание
    with torch.no_grad():
        model.eval()
        outputs = model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Получаем топ-3 предсказания
    top3_prob, top3_catid = torch.topk(probabilities, 3)

    # Формируем результат
    predictions = []
    for i in range(top3_prob.size(0)):
        class_id = top3_catid[i].item()
        class_name = CIFAR10_CLASSES[class_id]
        confidence = top3_prob[i].item()
        predictions.append((class_name, confidence))

    return predictions, frame_resized


# =================== 5. ОСНОВНАЯ ФУНКЦИЯ С ДВУМЯ ОКНАМИ ===================
def real_time_classification_with_dual_window(camera_id=0):
    """Запуск классификации в реальном времени с двумя окнами"""

    # Создаем и обучаем модель
    print("Инициализация модели...")
    model, device = create_model()

    # Если нужно заново обучить модель, раскомментируйте:
    # model = train_model(model, trainloader, device, num_epochs=5)

    # Загружаем предобученные веса (если есть)
    # model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device))

    # Открываем камеру
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Не удалось подключиться к камере")
        return

    # Настройка параметров камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Создаем окна
    cv2.namedWindow('📷 Камера', cv2.WINDOW_NORMAL)
    cv2.namedWindow('🤖 Классификация', cv2.WINDOW_NORMAL)

    # Размещаем окна рядом
    cv2.resizeWindow('📷 Камера', 640, 480)
    cv2.resizeWindow('🤖 Классификация', 640, 480)
    cv2.moveWindow('📷 Камера', 100, 100)
    cv2.moveWindow('🤖 Классификация', 750, 100)

    # Переменные для контроля FPS
    last_prediction_time = time.time()
    prediction_interval = 1.0  # 1 кадр в секунду
    last_predictions = []
    last_processed_frame = None

    print("\n" + "=" * 60)
    print("СИСТЕМА КЛАССИФИКАЦИИ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 60)
    print("Левое окно: Прямой эфир с камеры")
    print("Правое окно: Результат классификации (обновляется 1 раз в секунду)")
    print("\nУправление:")
    print("  'q' - Выход")
    print("  's' - Сохранить текущий результат")
    print("  '1' - Увеличить интервал до 2 сек")
    print("  '2' - Уменьшить интервал до 0.5 сек")
    print("  ' ' (пробел) - Сделать снимок сейчас")
    print("=" * 60)

    frame_count = 0
    start_time = time.time()

    while True:
        # Читаем кадр с камеры
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break

        frame_count += 1

        # ============ ЛЕВОЕ ОКНО: КАМЕРА ============
        camera_display = frame.copy()

        # Добавляем информацию на кадр камеры
        current_time = time.time()
        fps = frame_count / (current_time - start_time)

        # Время до следующего предсказания
        time_to_next = max(0, prediction_interval - (current_time - last_prediction_time))

        # Текстовая информация
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Кадр: {frame_count}",
            f"Следующее предсказание через: {time_to_next:.1f}с",
            f"Интервал: {prediction_interval}с"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(camera_display, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        # Индикатор готовности к предсказанию
        if time_to_next < 0.1:  # Почти время для предсказания
            cv2.circle(camera_display, (600, 40), 15, (0, 255, 0), -1)
        else:
            cv2.circle(camera_display, (600, 40), 15, (0, 0, 255), -1)

        cv2.imshow('📷 Камера', camera_display)

        # ============ ПРАВОЕ ОКНО: КЛАССИФИКАЦИЯ ============
        current_time = time.time()
        make_prediction = False

        # Проверяем клавиши для принудительного предсказания
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Пробел - предсказать сейчас
            make_prediction = True
        elif current_time - last_prediction_time >= prediction_interval:
            make_prediction = True

        if make_prediction:
            # Делаем предсказание
            predictions, processed_frame = predict_frame(model, frame, device)
            last_predictions = predictions
            last_processed_frame = processed_frame
            last_prediction_time = current_time

            # Выводим результат в консоль
            print(f"\n[{time.strftime('%H:%M:%S')}] Предсказание #{frame_count}:")
            for i, (class_name, confidence) in enumerate(predictions):
                print(f"  {i + 1}. {class_name}: {confidence:.1%}")

        # Создаем отображение для окна классификации
        if last_processed_frame is not None and len(last_predictions) > 0:
            # Создаем большое изображение для отображения
            classification_display = np.zeros((480, 640, 3), dtype=np.uint8)
            classification_display[:] = (40, 40, 40)  # Темно-серый фон

            # Добавляем обработанный кадр (увеличенный)
            small_frame = cv2.resize(last_processed_frame, (200, 200))
            classification_display[50:250, 50:250] = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)

            # Добавляем заголовок
            cv2.putText(classification_display, "РЕЗУЛЬТАТ КЛАССИФИКАЦИИ",
                        (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Добавляем топ-3 предсказания
            y_offset = 100
            for i, (class_name, confidence) in enumerate(last_predictions):
                # Цвет в зависимости от уверенности
                if confidence > 0.7:
                    color = (0, 255, 0)  # Зеленый
                elif confidence > 0.3:
                    color = (0, 255, 255)  # Желтый
                else:
                    color = (0, 165, 255)  # Оранжевый

                # Текст предсказания
                text = f"{i + 1}. {class_name}: {confidence:.1%}"
                cv2.putText(classification_display, text,
                            (260, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Полоса уверенности
                bar_width = int(confidence * 200)
                cv2.rectangle(classification_display,
                              (260, y_offset + 10),
                              (260 + bar_width, y_offset + 25),
                              color, -1)

                y_offset += 50

            # Добавляем время последнего обновления
            time_text = f"Обновлено: {time.strftime('%H:%M:%S', time.localtime(last_prediction_time))}"
            cv2.putText(classification_display, time_text,
                        (260, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Легенда
            cv2.putText(classification_display, "Высокая уверенность (>70%)",
                        (260, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(classification_display, "Средняя уверенность (30-70%)",
                        (260, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(classification_display, "Низкая уверенность (<30%)",
                        (260, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.imshow('🤖 Классификация', classification_display)
        else:
            # Показываем сообщение об ожидании
            waiting_display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting_display, "ОЖИДАНИЕ ПЕРВОГО ПРЕДСКАЗАНИЯ...",
                        (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('🤖 Классификация', waiting_display)

        # ============ ОБРАБОТКА КЛАВИШ ============
        if key == ord('q'):  # Выход
            break
        elif key == ord('s'):  # Сохранить результат
            if last_processed_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # Сохраняем оригинальный кадр
                cv2.imwrite(f'capture_{timestamp}.jpg', frame)
                # Сохраняем результаты
                with open(f'results_{timestamp}.txt', 'w') as f:
                    f.write(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Кадр: {frame_count}\n")
                    f.write("Предсказания:\n")
                    for i, (class_name, confidence) in enumerate(last_predictions):
                        f.write(f"  {i + 1}. {class_name}: {confidence:.1%}\n")
                print(f"Сохранено: capture_{timestamp}.jpg и results_{timestamp}.txt")
        elif key == ord('1'):  # Увеличить интервал
            prediction_interval = 2.0
            print(f"Интервал изменен на {prediction_interval} сек")
        elif key == ord('2'):  # Уменьшить интервал
            prediction_interval = 0.5
            print(f"Интервал изменен на {prediction_interval} сек")

    # ============ ЗАВЕРШЕНИЕ ============
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("СТАТИСТИКА РАБОТЫ:")
    print("=" * 60)
    print(f"Всего кадров: {frame_count}")
    print(f"Общее время: {total_time:.1f} сек")
    print(f"Средний FPS: {frame_count / total_time:.1f}")

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()


# =================== 6. ЗАПУСК ПРОГРАММЫ ===================
if __name__ == "__main__":
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


    camera_id = check_camera()

    # Запускаем систему классификации
    real_time_classification_with_dual_window(camera_id)