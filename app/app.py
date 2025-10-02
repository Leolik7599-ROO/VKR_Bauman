import os                                                                           # Для работы с ОС, например на наличие файла - помогает ловить ошибки
import json                                                                         # Для работы с файлами JSON - хранение метаданных
import joblib                                                                       # Для сохранения и загрузки объектов Python (scaler)
import pandas as pd                                                                 # Для работы с табличными данными
import numpy as np                                                                  # Для работы с массивами
from flask import Flask, render_template, request, redirect, url_for, flash         # Для создания веб-приложения
from sklearn.preprocessing import StandardScaler                                    # Для предварительной обработки данных
from sklearn.model_selection import train_test_split                                # Для разделения на обучающую и тестовую выборки
import torch                                                                        # Для построения и обучения нейронных сетей
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

# Некоторые настройки
DATA_PATH = os.getenv("DATA_CSV", "standard_for_model.csv")                         # Путь к CSV файлу с данными, если не переменной окружения, то файл берется по названию
MODEL_PATH = 'model.pth'                                                            # Путь для сохранения обученной модели
SCALER_PATH = 'scaler.pkl'                                                          # Путь для сохранения scaler
META_PATH = 'meta.json'                                                             # Путь для сохранения метаданных (названия признаков)
DEVICE = torch.device('cpu')                                                        # Жесткое обозначение места выполнения предсказаний

# Сама модель
class Recommendation_system(nn.Module):
    def __init__(self, input_size):                                                 # Структура нейросети
        super(Recommendation_system, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):                                                           # Прямой проход через нейросеть
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Все функции для подготовки данных к обучению
def prepare_data(dataset, target_col='Соотношение матрица-наполнитель'):                 # Определение целевого столбца (переменной)
    X = dataset.drop([target_col], axis=1).values                                        # Удаление целевой переменной из обучающего dataset
    y = dataset[target_col].values                                                       # Целевая переменная теперь содержит столбец со значениями соотношения матрица-наполнитель
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)             # Разделение на обучающую и валидационную выборки
    X_temp, X_test, y_temp, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)   # Разделение на обучающую и тестовую выборки
    scaler = StandardScaler()                                                                           # Стандартизация данных
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=8):                                       # Создание загрузчиков данных PyTorch
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))   # Создание тензоров из массивов
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),                                                 # Создание пакетов для каждой выборки
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),                                                  # итерация по данным батчами
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

def evaluate_model(model, data_loader, criterion):                                  # Оценка модели на заданном загрузчике данных
    model.eval()                                                                    # перевод модели в режим оценки
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            total_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy().flatten())
    avg_loss = total_loss / len(data_loader.dataset)                                # Вычисление потери
    r2 = r2_score(all_targets, all_predictions)                                     # Вычисление метрики R2
    return avg_loss, r2

def train_and_save_model(dataset):                                                       # Функция для обучения и сохранения модели
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(dataset)       # Подготовка данных
    train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=8)    # Создание загрузчиков данных
    input_size = X_train.shape[1]                                                        # Будет равно количеству столбцов в обучающей выборке
    model = Recommendation_system(input_size).to(DEVICE)                                 # Инициализация модели
    criterion = nn.MSELoss()                                                             # Функция потерь
    optimizer = optim.Adam(model.parameters(), lr=0.001)                                 # Оптимизатор

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    epochs = 100

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_loss, val_r2 = evaluate_model(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)                                    # Сохраняем лучшую модель
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping")
            break
        print(f'Epoch {epoch+1}/{epochs} train_loss={running_loss/len(train_loader):.4f} val_loss={val_loss:.4f} val_r2={val_r2:.4f}')

    # Загрузка лучших весов и оценка на тесте
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    test_loss, test_r2 = evaluate_model(model, test_loader, criterion)
    print(f'Final test loss: {test_loss:.4f}, R2: {test_r2:.4f}')

    # Сохранение scaler и метаданных (названия признаков)
    joblib.dump(scaler, SCALER_PATH)
    features = list(dataset.drop(['Соотношение матрица-наполнитель'], axis=1).columns)
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump({'feature_names': features}, f, ensure_ascii=False)
    return model, scaler, features

# Загрузка данных и модели при старте
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Не найден CSV файл: {DATA_PATH}. Положите ваш файл с именем standard_for_model.csv или укажите путь через DATA_CSV env var.") # Выводим на случай ошибки 

dataset = pd.read_csv(DATA_PATH)
if 'Соотношение матрица-наполнитель' not in dataset.columns:
    raise KeyError("В CSV отсутствует столбец 'Соотношение матрица-наполнитель'")               # Тоже введена проверка на всякий случай, если исходный датасет поврежден или загружен в папку не тот, если столбец целевой переменно отсутствует

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(META_PATH):
    print("Загружаем модель и scaler из файлов...")                                             # Если файлы модели, scaler и метаданные есть, то осуществляется загрузка и выводится сообщение, для понимания, на каком этапе процесс
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    feature_names = meta['feature_names']
    scaler = joblib.load(SCALER_PATH)
    # Создание модели с правильным input size и загрузка весов
    model = Recommendation_system(len(feature_names)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    print("Обучаем модель на основе CSV (это может занять время)...")              # Если файлы не были найдены, обучение модели производится на основе предоставленного файла
    model, scaler, feature_names = train_and_save_model(dataset)
    model.to(DEVICE)
    model.eval()

# Функционал Flask приложения
app = Flask(__name__)
app.secret_key = 'change_this_secret'

@app.route('/', methods=['GET'])                                                        # Определение маршрута для главной страницы
def index():
    # Подготовка средних значений для отображения в форме
    defaults = {}
    for f in feature_names:
        defaults[f] = float(dataset[f].mean())
    return render_template('index.html', features=feature_names, defaults=defaults)     # Форма для ввода значений признаков

@app.route('/predict', methods=['POST'])                                                # Определение маршрута для страницы предсказания
def predict():
    # Сбор значений из формы
    try:
        vals = []
        for f in feature_names:
            v = request.form.get(f)
            if v is None:
                return "Missing feature: " + f, 400
            vals.append(float(v.replace(',', '.')))
        X = np.array(vals).reshape(1, -1)
        X_scaled = scaler.transform(X)
        with torch.no_grad():
            t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
            pred = model(t).cpu().numpy().flatten()[0]
        return render_template('index.html', features=feature_names, defaults={k: float(v) for k,v in zip(feature_names, vals)}, prediction=pred)
    except Exception as e:
        flash(str(e))
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Flask debug выключить в продакшне
    app.run(host='0.0.0.0', port=5000, debug=True)
