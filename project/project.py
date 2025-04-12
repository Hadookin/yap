from ultralytics import YOLO
import cv2

# Загрузка предобученной модели YOLO
model = YOLO('yolov8n.pt')  # Вы можете выбрать другую модель, например yolov8s.pt, yolov8m.pt и т.д.

# Путь к изображению, на котором будем распознавать объекты
image_path = 'C:\\Users\\Artem\\Desktop\\project\\shapes.png'

# Чтение изображения с помощью OpenCV
image = cv2.imread(image_path)

# Применение модели для распознавания объектов
results = model(image)

# Визуализация результатов
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        r = box.xyxy[0].astype(int)
        cls = int(box.cls[0])
        conf = box.conf[0]
        
        # Отрисовка прямоугольника вокруг объекта
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
        
        # Добавление метки с классом и вероятностью
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Отображение результата
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()