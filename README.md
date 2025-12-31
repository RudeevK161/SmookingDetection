# smooking-detection

Данный репозиторий содержит код проект SmookingDetection.

Задача: Разработка системы компьютерного зрения для автоматического определения признаков курения у человека на основе анализа фотографий.
Техническая суть:
●	Бинарная классификация изображений на две категории: "курящий" / "некурящий"
Зачем:
●	Контроль соблюдения антитабачного законодательства
●	Автоматизация наблюдения в общественных местах
●	Анализ поведения в зонах, свободных от курения

Metrics:

- Accuracy
- F1 score

Baseline model is simple convolutional neural network.
Final model (finetuned ViT)

В виду того, что я создавл проект на ноутбуке, без gpu, метрики получились, ниже заявленных, чем на kaggle, однако если увеличить число эпох, то можно достичь следующих метрик 
- Accuracy: 0.9
- F1 score: 0.8


Training loss is crossentropy.

Dataset:

- [Smooking]([https://www.kaggle.com/datasets/zedsden/mushroom-classification-dataset](https://www.kaggle.com/datasets/sujaykapadnis/smoking))
- Датасет уже поделен на три части: training, validation, testing.

Если нужны артефакты, то можно их скачать здесь, однако для проверки можно обучить модель на малом числе эпох, которое установлено в конфиге.
[here](https://drive.google.com/drive/folders/1y5kkbgmNVoyA4DfdhyQP6XPOh45n4h6z?usp=drive_link)

## Setup

1. Install
   [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).
2. Install the project dependencies:

```bash
poetry install
```

optional: set conda environment

Также для скачивания данных, если с dvc не получилось, то вам нужну положить kaggle api token в
~/.kaggle/kaggle.json

## Train

1. Вначале запустите mlflow server
2. Чтобы начать обучение, запустите следующую команду:

```bash
poetry run python smooking_detection/train.py
```

Note: data is downloaded automatically.

## Pruduction preparation
Модель сохраняется в triton/export/onnx/model.onnx, но отлаживать этот момент буду после, т к в задание 2 не входит, а хочется сделать.
1. You need to put .onnx model in the `triton/export/onnx/model.onnx` file (if
   you fully trained model then it's already made onnx).
2. To prepare the production tensorrt model for triton go to `triton` folder and
   use [commands](triton/export_to_trt.md).

## Infer

 Чтобы запустить инференс, у вас должна быть обученная модель в checkpoints, т к config ссылается на этот путь.
 
Запуск без параметров для картинки по умолчанию, которая лежит infer_imges - sm.PNG:

```bash
poetry run python smooking_detection/infer.py
```
Если указывать пути вручную:

```bash
poetry run python mushroom_classification/infer.py \
    image_path=infer_images/nsm.png \
    model_path=checkpoints/best_model_weights.pth
```
If you want to infer via triton see [example notebook](triton/test.ipynb).
