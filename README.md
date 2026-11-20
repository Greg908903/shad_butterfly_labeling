# Детекция бабочек

Репозиторий для инференса **Grounding DINO tiny**, **OWLv2 base** и
**Qwen3-VL-2B-Instruct** на задаче выделения бабочек на изображениях и подсчёта метрик.

## Код

- `predict.py` — инференс на папке изображений, сохранение рамок и визуализаций.
- `models.py` — функции `load_model()` и `predict()` для поддерживаемых моделей.
- `evaluate.py` — подсчёт метрик и сохранение таблицы результатов.

## Установка

Проверено на Ubuntu 24.04.5 LTS, Python 3.12.3 и NVIDIA V100 32 GB.
В `requirements.txt` закреплён PyTorch `2.9.1+cu126` для совместимости с V100.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Конфиги

- `configs/config.yaml` — папки входа/выхода, устройство и выбор модели.
- `configs/model/*.yaml` — checkpoint, промпт и параметры каждой модели.
- `configs/evaluate.yaml` — входные файлы и настройки подсчёта метрик.

## Запуск

Из корня проекта. Изображения — в `data/dataset_images/`, имена вида `123.jpg`.
Другую папку можно указать через `images_dir=/path/to/images`.

```bash
python predict.py model=grounding_dino
python predict.py model=owlv2
python predict.py model=qwen
python evaluate.py
```

## Выход

- `outputs/<модель>/<дата>/` — предсказания JSONL, журнал запуска и визуализации.
- `outputs/evaluation/<дата>/` — метрики JSON, таблицы CSV/PNG и результаты по кадрам.
