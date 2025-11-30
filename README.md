# HW1: DVC & MLflow

## Структура проекта

├── data/              # данные под управлением DVC  
│   ├── raw/           # сырые данные (iris.csv под DVC)  
│   └── processed/     # train/test сплиты  
├── src/               # исходный код  
│   ├── prepare.py     # подготовка данных  
│   └── train.py       # обучение модели и логирование в MLflow  
├── dvc.yaml           # описание DVC-пайплайна  
├── params.yaml        # гиперпараметры  
├── requirements.txt   # зависимости  
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