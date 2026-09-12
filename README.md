# Мультимодальное распознавание и анализ психоэмоциональных состояний человека

Цель работы — разработка собственного механизма внимания для свёрточных нейронных сетей, адаптированного под задачу распознавания эмоций по изображению лица, обеспечивающего повышение точности классификации психоэмоциональных состояний человека по сравнению с существующими подходами.

В рамках проекта реализован и обучен единый backbone (VGG16-подобная сеть с residual-связями) с более чем 20 различными механизмами внимания — от базовых (SE-Net, CBAM) до наиболее современных решений 2023–2024 годов — с целью их честного сравнения на одном датасете и выявления наиболее перспективных архитектурных решений для последующей разработки собственного механизма.

## Содержание

- [Датасеты](#датасеты)
- [Итоговое сравнение механизмов внимания](#итоговое-сравнение-механизмов-внимания)
- [Базовые эксперименты (FER-2013)](#базовые-эксперименты-fer-2013)
- [Служебные файлы](#служебные-файлы)
- [Запуск](#запуск)

## Датасеты

- **FER-2013** — использовался на начальном этапе для базовых архитектур (полносвязная сеть, AlexNet и его вариации).
- **RAF-DB (aligned)** — основной датасет для всех последующих экспериментов с механизмами внимания.

## Итоговое сравнение механизмов внимания

Все модели ниже построены на общем VGG16-подобном backbone и обучались на RAF-DB (aligned) при одинаковых условиях, что делает точность моделей напрямую сравнимой между собой.

| Модель | Файл | Точность | Параметры | Архитектура | Статья |
|---|---|---|---|---|---|
| Distract Your Attention Network | `DAN.py` | **0.7930** | ~18.9 млн | VGG16 + Distract Your Attention (4 головы) после последнего блока | Wen, Lin, Wang, Xu — *Distract Your Attention: Multi-Head Cross Attention Network for Facial Expression Recognition* |
| Coordinate Attention Network | `CordAttNet.py` | 0.79 | ~15.3 млн | VGG16 + Coordinate Attention в каждом блоке | Hou, Zhou, Feng — *Coordinate Attention for Efficient Mobile Network Design*, CVPR 2021 |
| Channel-wise Spatially Autocorrelated Attention | `CSANet.py` | 0.7865 | ~15.38 млн | VGG16 + CSA в каждом блоке | Nikzad, Gao, Zhou — *CSA-Net: Channel-wise Spatially Autocorrelated Attention Networks*, 2024 |
| Efficient Local Attention | `ELANet.py` | 0.7832 | ~15.42 млн | VGG16 + ELA в каждом блоке | Xu, Wan — *ELA: Efficient Local Attention for Deep Convolutional Neural Networks*, 2024 |
| Non-Local Attention Network | `NonLocalAttention.py` | 0.78 | ~15.6 млн | VGG16 + Non-Local Attention после 4-го и 5-го блоков | Wang, Girshick, Gupta, He — *Non-local Neural Networks*, CVPR 2018 |
| Dual Attention Network | `DualAttNet.py` | 0.78 | ~15.6 млн | VGG16 + Dual Attention после 4-го и 5-го блоков | Fu, Liu, Tian, Li, Bao, Fang, Lu — *Dual Attention Network for Scene Segmentation*, CVPR 2019 |
| Channel Prior Convolutional Attention | `CPCA.py` | 0.7692 | ~16.18 млн | VGG16 + CPCA в каждом блоке | Huang, Chen, Zou, Lu, Chen — *Channel Prior Convolutional Attention for Medical Image Segmentation*, 2023 |
| Shared Multi-Semantic Spatial Attention | `SCSANet.py` | 0.7669 | ~15.32 млн | VGG16 + SCSA в каждом блоке | Si, Xu, Zhu, Zhang, Dong, Chen, Li — *SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention*, 2024 |
| Squeeze-and-Excitation Network | `SENet.py` | 0.77 | ~15.4 млн | VGG16 + SE-блок после каждого блока | Hu, Shen, Sun — *Squeeze-and-Excitation Networks* |
| Pyramid Squeeze Attention | `EPSANet.py` | 0.7448 | ~18.68 млн | VGG16 + PSA в каждом блоке | Zhang, Zu, Lu, Zou, Meng — *EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network*, ACCV 2022/2023 |
| Parameter-Free Attention Network | `SimAMNet.py` | 0.75 | ~15.0 млн | VGG16 + SimAM в каждом блоке | Yang, Zhang, Wang, Li — *SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks*, ICML 2021 |
| Triplet Attention Network | `TripletAttNet.py` | 0.76 | ~15.3 млн | VGG16 + Triplet Attention в каждом блоке | Misra, Nalamada, Arasanipalai, Hou — *Rotate to Attend: Convolutional Triplet Attention Module*, WACV 2021 |
| Frequency Channel Attention Network | `FcaNet.py` | 0.71 | ~15.4 млн | VGG16 + FcaNet после 4-го и 5-го блоков | Qin, Zhang, Wu, Li — *FcaNet: Frequency Channel Attention Networks*, ICCV 2021 |
| AttentionNet (CBAM) | `CBAM.py` | 0.71 | ~50.7 млн | VGG16 + CBAM (межканальное и внутриканальное внимание) | Woo, Park, Lee, Kweon — *CBAM: Convolutional Block Attention Module*, ECCV 2018 |
| SelfAttentionNet | `SelfAttentionNet.py` | 0.70 | ~51.6 млн | VGG16 + самовнимание (self-attention) в 3, 4 и 5 блоках | Wang, Girshick, Gupta, He — *Non-local Neural Networks*, CVPR 2018 |
| VGG16 (RAF-DB) | `VGG16_refDB.py` | 0.67 | ~50.85 млн | 5 блоков по 2 свёрточных слоя с возможностью пропуска блока, классификатор — ПНН с 2 скрытыми слоями | — |

> Столбец «Параметры» отражает полное число обучаемых весов модели (backbone + классификатор + модуль внимания).

## Базовые эксперименты (FER-2013)

Ранний этап работы — подбор и обкатка базовой архитектуры на FER-2013, до перехода на RAF-DB и эксперименты с механизмами внимания.

| Модель | Файл | Точность | Архитектура |
|---|---|---|---|
| Полносвязная НС | `fullNn.py` | 0.37 | Вход 2304 нейрона → 2 скрытых слоя (256, 128) → выход 7 классов. BatchNorm, ReLU |
| AlexNet | `AlexNet.py` | 0.56 | 5 свёрточных + 2 полносвязных слоя. BatchNorm, ReLU. Переобучение начинается с 30-й эпохи |
| AlexNetResNet | `AlexNetResNet.py` | 0.60 | 6 свёрточных слоёв в 3 блоках по 2, каждый блок с возможностью пропуска (residual). BatchNorm, ReLU |
| AlexResNetAugumentation | `AlexResNetAug.py` | 0.62 | Аналогична предыдущей, добавлена случайная аугментация (поворот, отражение, изменение контраста/яркости, blur) |
| AlexResNetAugExtraLayers | `AlexResNetAugExtraLayers.py` | — | Расширенный вариант AlexResNetAug с дополнительными слоями |
| VGG16 | `VGG16.py` | 0.61 | 5 блоков по 2 свёрточных слоя с возможностью пропуска блока, классификатор — ПНН с 2 скрытыми слоями |

## Служебные файлы

| Файл | Назначение |
|---|---|
| `dataset_class_gen.py` | Вспомогательный скрипт для генерации класса датасета |
| `mediapipe_neyro.py` | Эксперимент с извлечением признаков лица через MediaPipe |
| `mnist_net.py` | Тестовая сеть на MNIST (отладка пайплайна обучения) |
| `requirements.txt` | Зависимости проекта |

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Каждый файл модели самодостаточен: содержит определение датасета, архитектуры, цикл обучения (50 эпох, сохранение лучшей по Val Accuracy модели в `best_models/`) и итоговую оценку на тестовой выборке. Запуск:

```bash
python <имя_файла>.py
```

Перед запуском убедитесь, что путь к датасету (RAF-DB aligned / FER-2013) в начале файла соответствует вашей структуре директорий.
