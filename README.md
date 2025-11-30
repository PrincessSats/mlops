# HW1: DVC & MLflow

Домашнее задание по модулю «Основы воспроизводимого машинного обучения. Жизненный цикл MLOps».

Проект показывает минимальный, но полноценный MLOps-контур:

- версионирование данных через DVC;
- воспроизводимый пайплайн подготовки данных и обучения;
- логирование экспериментов в MLflow (параметры, метрики, артефакты);
- запуск обучения одной командой: `dvc repro`.

---

## Структура проекта

```text
.
├── data/                # данные под управлением DVC
│   ├── raw/             # сырые данные (iris.csv под DVC)
│   └── processed/       # train/test сплиты
├── src/                 # исходный код
│   ├── download_data.py # сохранение датасета iris в CSV
│   ├── prepare.py       # подготовка данных и разбиение на train/test
│   └── train.py         # обучение модели и логирование в MLflow
├── dvc.yaml             # описание DVC-пайплайна (stages: prepare, train)
├── dvc.lock             # зафиксированное состояние пайплайна
├── params.yaml          # гиперпараметры (split + модель)
├── requirements.txt     # зависимости проекта
└── README.md

## Как запустить

```bash
git clone <repo_url>
cd mlops-hw1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Вытянуть данные
dvc pull   # если настроен remote

# Запустить пайплайн
dvc repro

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db