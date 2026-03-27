# neuro - нейросетевой фреймворк на F#

Легковесный учебно-практический фреймворк для нейросетей, написанный с нуля на F# без внешних ML-библиотек.
Проект показывает полный цикл: от подготовки данных и матричных операций до обучения, валидации и запуска реальных примеров (`Iris`, `MNIST`, `CNN MNIST`, `TicTacToe`).

---

## Содержание

1. [Что это за проект](#что-это-за-проект)
2. [Ключевые возможности](#ключевые-возможности)
3. [Быстрый старт](#быстрый-старт)
4. [Как устроено обучение](#как-устроено-обучение)
5. [Модули и API](#модули-и-api)
6. [Примеры](#примеры)
7. [Структура проекта](#структура-проекта)

---

## Что это за проект

`neuro` - это консольное приложение (`net9.0`), в котором реализована модульная мини-экосистема для нейросетей:

- базовые операции над матрицами;
- функции активации и их производные;
- функции потерь и градиенты;
- полносвязные слои (`Dense`) и прямой/обратный проход;
- оптимизаторы (`SGD`, `Momentum`, `Adam`, `GradientClipping`);
- утилиты для датасетов;
- тренировочный цикл и вычисление точности;
- набор демонстрационных сценариев и встроенные тесты.

Проект особенно полезен, если нужно понять механику backpropagation и оптимизации, а не только «использовать готовый фреймворк».

---

## Ключевые возможности

- Обучение многослойной сети как списка слоев `DenseLayer list`.
- Поддержка слоев: `Dense` и `Dropout`.
- Базовая поддержка сверточных архитектур: `Conv2D`, `MaxPool2D`, `AvgPool2D`, `Flatten/Reshape`, `BatchNorm2D`.
- Поддержка активаций: `Sigmoid`, `ReLU`, `Tanh`, `Softmax`, `Linear`.
- Поддержка функций потерь: `MSE`, `CrossEntropy`, `BinaryCrossEntropy`.
- Поддержка оптимизаторов: `SGD`, `Momentum`, `Adam`, `GradientClipping`.
- Подготовка данных: нормализация, перемешивание, батчинг, split.
- Практические примеры на реальных данных (`Iris`, `MNIST`) и игровой задаче (`TicTacToe`).
- Отдельный пример CNN на MNIST (`Examples/CNNMNIST.fs`).
- В CNN-тренере есть поддержка `SGD`, `Momentum`, `Adam`, `GradientClipping`.
- В CNN-сценарии зафиксирован quality gate: `test accuracy >= 85%`.
- Большой набор встроенных проверок в `Examples/Tests.fs`.

---

## Быстрый старт

### Требования

- .NET SDK 9.0+
- macOS / Linux / Windows

### Сборка и запуск

```bash
dotnet build neuro.sln
dotnet run --project neuro/neuro.fsproj
```

После запуска откроется меню:

- `1` - Comprehensive Framework Tests
- `2` - Iris Classification
- `3` - MNIST Classification
- `4` - TicTacToe
- `5` - MNIST CNN (Conv2D)
- `6` - Hyperparameter Lab
- `7` - MNIST Regularization Experiments
- `8` - Interpretability Tools
- `9` - Benchmark Harness
- `0` - Exit

### Важные данные

Для примеров используются CSV-файлы в `neuro/Examples/Data/`:

- `Iris.csv`
- `mnist_train.csv`
- `mnist_test.csv`

---

## Как устроено обучение

Ниже путь данных в `Trainer.train`:

```text
Dataset
  -> (optional) split train/val
  -> shuffle
  -> batch
  -> forward (по всем слоям)
  -> loss + dLoss
  -> backward (по слоям в обратном порядке)
  -> optimizer update
  -> метрики эпохи (train loss, optional val loss)
```

Важные детали реализации:

- Слои создаются через `Layers.createLayer` с масштабированной инициализацией весов.
- На прямом проходе кешируются входы/преактивации/выходы слоев для корректного backward.
- Для `Softmax` производная обрабатывается отдельно в логике backprop.
- В конце каждой эпохи сохраняется `TrainLoss`, при наличии validation - еще и `ValLoss`.
- `Trainer.accuracy` сравнивает argmax предсказания и целевой one-hot класс.

---

## Модули и API

### `Core/Matrix.fs`

Базовый модуль линейной алгебры над `float[,]`, на котором построены слои, лоссы и оптимизаторы.

Типы:

- использует стандартный двумерный массив F# `float[,]`.

Функции:

- `zeros rows cols` - создает матрицу нулей `rows x cols`.
- `ones rows cols` - создает матрицу единиц `rows x cols`.
- `random rows cols` - создает матрицу случайных значений в диапазоне около `[-0.05, 0.05]`.
- `init rows cols f` - обертка над `Array2D.init` для кастомной инициализации.
- `copy matrix` - возвращает полную копию матрицы.
- `multiply a b` - матричное умножение `a * b`.
- `transpose matrix` - транспонирует матрицу.
- `add a b` - поэлементное сложение матриц одинаковой размерности.
- `subtract a b` - поэлементное вычитание.
- `hadamard a b` - поэлементное произведение.
- `scale scalar matrix` - умножение каждого элемента матрицы на скаляр.
- `map f matrix` - применяет функцию `f` к каждому элементу матрицы.
- `sum matrix` - сумма всех элементов.
- `mean matrix` - среднее всех элементов.

Нюанс реализации: `multiply` использует небольшой micro-optimization (`if aik <> 0.0`) и цикл по `k`, что уменьшает лишние операции при разреженных/частично нулевых данных.

### `Core/Activation.fs`

Модуль активаций и их производных.

Типы:

- `ActivationFunction = Sigmoid | ReLU | Tanh | Softmax | Linear`.

Функции:

- `activate activation value` - скалярная активация:
  - `Sigmoid`: `1 / (1 + exp(-x))`
  - `ReLU`: `max(0, x)`
  - `Tanh`: `tanh(x)`
  - `Linear`: `x`
  - `Softmax`: исключение (нужен вектор/строка, а не скаляр).
- `derivative activation value` - скалярная производная активации:
  - `Sigmoid`: `s(x) * (1 - s(x))`
  - `ReLU`: `1` при `x > 0`, иначе `0`
  - `Tanh`: `1 - tanh(x)^2`
  - `Linear`: `1`
  - `Softmax`: исключение (обрабатывается в backward отдельно).
- `apply activation matrix` - применяет активацию ко всей матрице:
  - для `Softmax` нормализует **каждую строку** в распределение вероятностей;
  - для остальных активаций применяет `activate` поэлементно.

Нюанс реализации: `Softmax` численно стабилизирован через вычитание максимума по строке перед `exp`.

### `Losses.fs`

Модуль функций потерь и производных по предсказанию сети.

Типы:

- `LossFunction = MSE | CrossEntropy | BinaryCrossEntropy`.

Функции:

- `compute loss yPred yTrue` - возвращает скаляр loss по батчу.
  - `MSE`: средний квадрат ошибки по всем элементам.
  - `CrossEntropy`: `-sum(yTrue * log(yPred)) / batchSize`.
  - `BinaryCrossEntropy`: бинарная кросс-энтропия по элементам, усреднение по числу объектов.
- `gradient loss yPred yTrue` - возвращает `dLoss/dyPred` той же размерности, что `yPred`.
  - `MSE`: `2 * (yPred - yTrue)`.
  - `CrossEntropy`: `(yPred - yTrue)`.
  - `BinaryCrossEntropy`: аналитическая производная BCE с защитой от деления на 0.

Нюанс реализации: в `CrossEntropy` и `BinaryCrossEntropy` используется `eps = 1e-8`, а `yPred` дополнительно ограничивается в `(eps, 1-eps)` для численной устойчивости.

### `Layers.fs`

Ядро прямого и обратного прохода полносвязной сети.

Типы:

- `DenseLayer`:
  - `Weights: float[,]` размером `[outputSize, inputSize]`
  - `Bias: float[]` длиной `outputSize`
  - `Activation`, `InputSize`, `OutputSize`
  - `Kind` (`Dense` или `Dropout`)
  - `DropoutRate` (используется только для `Dropout`).
- `NeuralNetwork = DenseLayer list` - сеть как список слоев.

Функции:

- `createLayer inputSize outputSize activation` - создает слой с Xavier-подобной инициализацией: масштаб `sqrt(6 / (in + out))`, bias = 0.
- `createDropout size rate` - создает dropout-слой размера `size` с вероятностью зануления `rate`.
- `forward layer input` - прямой проход слоя:
  - считает `z = input * W^T + b`
  - применяет активацию
  - возвращает `(z, output)`.
- `forwardTraining layer input` - прямой проход в training-режиме (для `Dropout` генерирует маску и применяет inverted dropout).
- `forwardNetwork network input` - прогоняет вход через все слои:
  - возвращает `(finalOutput, cache)`;
  - `cache` хранит `(z, output)` по слоям в порядке обратного прохода.
- `backward layer gradOutput input z output` - backward для одного слоя:
  - строит `gradActivation` (для `Softmax` используется `gradOutput` как есть);
  - для `Dropout` использует сохраненную маску из `z`;
  - считает `gradWeights`, `gradBias` (усредняет по batch);
  - считает `gradInput` для предыдущего слоя;
  - возвращает `(gradInput, gradWeights, gradBias)`.

Нюанс реализации: параметр `output` в текущей версии передается в `backward`, но для вычислений не используется - он оставлен для совместимости интерфейса с кешем trainer-а.

### `Optimizers.fs`

Модуль обновления параметров сети.

Типы:

- `Optimizer`:
  - `SGD of learningRate`
  - `Momentum of learningRate * momentum`
  - `Adam of learningRate * beta1 * beta2 * epsilon`
  - `GradientClipping of learningRate * threshold`.
- `OptimizerState`:
  - `Velocities` - состояние для `Momentum`
  - `AdamM`, `AdamV` - 1-й и 2-й моменты для `Adam`
  - `Step` - шаг оптимизатора.

Функции:

- `initState network optimizer` - готовит state под выбранный оптимизатор:
  - для `SGD` и `GradientClipping` состояние пустое;
  - для `Momentum` создаются нулевые скорости;
  - для `Adam` создаются нулевые `m/v`, `Step = 1`.
- `update optimizer state gradients network` - применяет шаг оптимизации ко всем слоям:
  - входные `gradients` задаются списком `(gradWeights, gradBias)` по слоям;
  - обновляет веса/смещения и возвращает `(newState, updatedNetwork)`.

Алгоритмы внутри `update`:

- `SGD`: `param <- param - lr * grad`.
- `Momentum`: `v <- momentum * v + lr * grad`, затем `param <- param - v`.
- `Adam`: обновление `m`, `v`, bias correction (`mHat`, `vHat`) и шаг `lr * mHat / (sqrt(vHat) + eps)`.
- `GradientClipping`: ограничивает каждый градиент порогом `[-threshold, threshold]` и делает обычный SGD-шаг.

### `Data.fs`

Модуль подготовки и трансформации датасета.

Типы:

- `Dataset = { Features: float[,]; Labels: float[,] }`.

Функции:

- `createDataset features labels` - конструктор записи `Dataset`.
- `map f dataset` - применяет `f` ко всем признакам (`Features`), labels не трогает.
- `normalize dataset` - z-score нормализация по каждому признаку:
  - `x' = (x - mean) / std`;
  - если `std = 0`, значение ставится в `0`.
- `shuffle dataset` - случайно перемешивает объекты, сохраняя соответствие `Features <-> Labels`.
- `batch batchSize dataset` - режет датасет на список мини-батчей.
- `split ratio dataset` - делит на `(train, val)` по первым `ratio * n` объектам.

Нюансы:

- `split` не перемешивает данные сам по себе; обычно перед ним вызывается `shuffle`.
- `batch` всегда возвращает последний батч, даже если он меньше `batchSize`.

### `Trainer.fs`

Оркестратор полного цикла обучения.

Типы:

- `TrainingConfig`:
  - `Epochs`, `BatchSize`
  - `Optimizer`, `Loss`
  - `Verbose`
  - `ValidationSplit: float option`.
- `TrainingMetrics`:
  - `TrainLoss: float list`
  - `ValLoss: float list`
  - `EpochsCompleted: int`.

Значения и функции:

- `defaultConfig` - базовая конфигурация (`100` эпох, `batch=32`, `SGD 0.01`, `MSE`).
- `train config network dataset` - основной training loop:
  - опционально делит train/val;
  - на каждой эпохе делает `shuffle -> batch -> forward -> loss -> backward -> update`;
  - копит `TrainLoss`, при наличии val считает `ValLoss`;
  - возвращает `(metrics, trainedNetwork)`.
- `predict network features` - прямой проход сети и возврат матрицы предсказаний.
- `accuracy network dataset` - accuracy по `argmax` (подходит для one-hot классификации).

Нюанс: `ValidationSplit = None` отключает валидацию полностью; в этом случае `ValLoss` остается пустым списком.

### `ConvLayers.fs`

Модуль слоев для сверточной сети и mixed-пайплайна (`Tensor4D` + dense-голова).

Типы:

- `LayerData = Tensor4D | Matrix`.
- `Conv2DLayer`, `MaxPool2DLayer`, `FlattenLayer`.
- `CNNLayer = Conv2D | MaxPool2D | Flatten | Dense`.
- `CNNNetwork = CNNLayer list`.

Функции:

- `createConv2D` - создание сверточного слоя.
- `createMaxPool2D` - max-pooling слой.
- `createAvgPool2D` - average-pooling слой.
- `createFlatten` - flatten слой (`Tensor4D -> Matrix`).
- `createReshapeToMatrix`, `createReshapeToTensor` - явные reshape-слои между `Tensor4D` и `Matrix`.
- `createBatchNorm2D` - batch normalization по каналам.
- `matrixToTensor`, `tensorToMatrix` - преобразование форматов входа/выхода.
- `forwardLayer`, `forwardNetwork` - прямой проход.
- `backwardLayer`, `backwardNetwork` - обратный проход.
- `initOptimizerState` - инициализация состояния оптимизатора для CNN.
- `updateNetwork` - обновление параметров `Conv2D` / `Dense` / `BatchNorm2D` через `SGD`, `Momentum`, `Adam`, `GradientClipping`.
- `updateNetworkSGD` - backward-compatible обертка для SGD.

### `TrainerCNN.fs`

Тренировочный модуль для CNN.

Типы:

- `CNNTrainingConfig` - конфиг обучения (включая выбор оптимизатора).
- `CNNTrainingMetrics` - метрики обучения.

Функции:

- `train` - полный training loop для mixed CNN-сети.
- `predict` - инференс.
- `accuracy` - точность по argmax.

Нюансы текущей реализации:

- вход в `train`/`predict` задается как `float[,]` + явный shape (`channels`, `height`, `width`);
- если `Optimizer = None`, используется `SGD LearningRate` для обратной совместимости.
- при `Optimizer = Some ...` поддерживаются `SGD`, `Momentum`, `Adam`, `GradientClipping`.

### `Examples/*`

Практическая часть проекта с готовыми сценариями и проверками.

`Examples/Iris.fs`:

- `loadDataFromCsv filePath` - читает Iris CSV, парсит 4 признака и строит one-hot метки 3 классов.
- `stratifiedSplit ratio dataset` - делает стратифицированный train/val split, сохраняя баланс классов.
- `run()` - полный pipeline примера: загрузка, нормализация, split, обучение, отчет по метрикам и предсказаниям.

`Examples/MNIST.fs`:

- `loadMnistFromCsv filePath maxSamples` - загружает MNIST CSV, переводит пиксели в `[0, 1]`, кодирует one-hot метки.
- `printDigit image width` - печатает изображение цифры в ASCII-виде для визуальной проверки.
- `run()` - полный запуск MNIST-сценария: загрузка, обучение, оценка, разбор предсказаний и ошибок.

`Examples/CNNMNIST.fs`:

- `run()` - CNN-сценарий для MNIST (`Conv2D -> BatchNorm -> MaxPool -> Conv2D -> BatchNorm -> AvgPool -> Flatten -> Dense -> Softmax`) с обучением через `TrainerCNN`.
- пример использует бинарную постановку (`цифры 0 vs 1`) для стабильной и быстрой демонстрации качества.
- число эпох увеличено (в текущем конфиге: `8`).
- после обучения применяется quality gate: при `test accuracy < 85%` сценарий завершится ошибкой.

`Examples/TicTacToe.fs`:

- `createEmptyBoard()` - создает пустое поле 3x3.
- `boardToString board` - рендерит поле в строку для консоли.
- `boardToFeatures board` - кодирует поле в one-hot вектор длины 27.
- `checkWinner board` - определяет победителя (`X`/`O`) или отсутствие победы.
- `isDraw board` - проверяет ничью (нет пустых клеток).
- `gameOver board` - объединенная проверка конца игры.
- `minimax board isMaximizing` - вычисляет minimax-оценку позиции.
- `generatePositionsForPlayer player` - рекурсивно генерирует игровые позиции и оптимальные ходы для заданного игрока.
- `createDatasetForPlayer player` - собирает датасет `(features, labels)` из сгенерированных позиций.
- `trainAI epochs` - обучает сеть для выбора хода и выводит метрики.
- `getNetworkMove network board` - выбирает ход сети по максимуму вероятности.
- `makeMove board row col player` - делает ход, если клетка свободна.
- `getPlayerMove()` - считывает и валидирует ход человека из консоли.
- `playGame network` - интерактивная игра человек vs сеть.
- `testAI network gamesCount` - прогон сети против случайного игрока и сбор статистики.
- `run()` - входная точка примера: обучение, авто-тест AI и опциональная игра с человеком.

`Examples/Tests.fs`:

- `testDataFunctions()` - проверки `Data` (map/normalize/shuffle/batch/split).
- `testMatrixOperations()` - проверки базовых матричных операций.
- `testActivationFunctions()` - проверки активаций и softmax.
- `testLossFunctions()` - проверки `MSE`, `CrossEntropy`, `BinaryCrossEntropy` и их градиентов.
- `testOptimizers()` - smoke/integration тесты для `SGD`, `Momentum`, `Adam`, `GradientClipping`.
- `testActivationsAndLosses()` - проверка совместимости разных комбинаций activation/loss.
- `testFullTrainingCycle()` - end-to-end тест учебного цикла на синтетической классификации.
- `testGradients()` - численная проверка градиентов (finite differences).
- `testCNNBlock()` - проверки CNN-блока: конвертация tensor/matrix, shape-checks forward/backward и accuracy-gate `>= 85%` на синтетическом датасете для `Adam` и `Momentum`.
- `run()` - запускает полный набор тестов и печатает итоговый статус.

`Examples/HyperparameterLab.fs`:

- `run()` - мини-лаборатория гиперпараметров (grid + random search по activation/lr/optimizer/batch).
- Формирует ранжированный leaderboard с `Run ID`/`Seed` и сохраняет отчеты в `Examples/Reports/hyperparameter_leaderboard.md` и `Examples/Reports/hyperparameter_leaderboard.csv`.

`Examples/RegularizationMNIST.fs`:

- `run()` - серия MNIST-экспериментов с регуляризацией.
- Поддерживает `label smoothing`, `L2 weight decay`, `mixup-style` и `cutmix-style` аугментации.
- Сохраняет сводные отчеты в `Examples/Reports/regularization_leaderboard.md` и `Examples/Reports/regularization_leaderboard.csv`.

`Examples/Interpretability.fs`:

- `run()` - инструменты интерпретируемости.
- Строит confusion matrix + per-class precision/recall/F1 для dense MNIST.
- Генерирует saliency maps (PGM) для CNN MNIST и сохраняет артефакты в `Examples/Reports/saliency/`.

`Examples/BenchmarkHarness.fs`:

- `run()` - benchmark-матрица по сценариям (Iris/MNIST), оптимизаторам и layer-предустановкам.
- Измеряет `accuracy`, `final loss`, время выполнения и приблизительную delta-памяти.
- Экспортирует отчеты в `Examples/Reports/benchmark_report.md` и `Examples/Reports/benchmark_report.csv`.

---

## Примеры

### 1) Iris (`Examples/Iris.fs`)

Что делает пример:

- читает `Iris.csv`;
- строит one-hot метки для 3 классов;
- нормализует признаки;
- делает стратифицированный split 80/20;
- обучает сеть `4 -> 8(ReLU) -> 3(Softmax)`;
- печатает accuracy, динамику loss и подробные предсказания на валидации.

Почему это полезно: компактная мультиклассовая задача, удобная для быстрой проверки корректности обучения.

### 2) MNIST (`Examples/MNIST.fs`)

Что делает пример:

- загружает MNIST из CSV;
- нормализует пиксели в диапазон `[0, 1]`;
- обучает сеть `784 -> 128(ReLU) -> 64(ReLU) -> 10(Softmax)`;
- оценивает качество на тесте и показывает выборочные предсказания.

В коде предусмотрен лимит выборок для более быстрого эксперимента (`train: 10000`, `test: 2000`).

### 3) TicTacToe (`Examples/TicTacToe.fs`)

Что делает пример:

- генерирует обучающие позиции через `minimax`;
- кодирует поле 3x3 в one-hot вектор из 27 признаков;
- обучает сеть `27 -> 128 -> 64 -> 9` предсказывать лучший ход;
- тестирует AI против случайного игрока;
- дает режим игры человека против обученной сети.

Это хороший пример того, как комбинировать алгоритмическую экспертную логику (Minimax) и supervised learning.

### 4) MNIST CNN (`Examples/CNNMNIST.fs`)

Что делает пример:

- загружает подмножество MNIST и формирует бинарный датасет (`0` vs `1`);
- обучает компактную сверточную сеть `Conv2D -> BatchNorm -> MaxPool -> Conv2D -> BatchNorm -> AvgPool -> Flatten -> Dense -> Dense(Softmax)`;
- обучает модель дольше базовой версии (в текущем конфиге `8` эпох);
- показывает train/test accuracy и динамику loss;
- проверяет quality gate: `test accuracy >= 85%`.

### 5) Comprehensive Tests (`Examples/Tests.fs`)

Набор тестов проверяет:

- корректность data pipeline;
- матричные операции;
- активации и лоссы;
- работу оптимизаторов;
- полный цикл обучения;
- численную проверку градиентов (сравнение analytical vs numerical);
- отдельный CNN-блок (форматы данных, формы тензоров и accuracy-gate на синтетике).

Этот файл можно рассматривать как «живую спецификацию» проекта.

### 6) Hyperparameter Lab (`Examples/HyperparameterLab.fs`)

Что делает пример:

- запускает grid search + random search для Iris;
- перебирает `learning rate`, `optimizer`, `batch size`, `hidden activation`;
- выводит топ запусков и сохраняет leaderboard.

### 7) MNIST Regularization (`Examples/RegularizationMNIST.fs`)

Что делает пример:

- запускает baseline и регуляризационные абляции;
- сравнивает `label smoothing`, `L2 weight decay`, `mixup-style`, `cutmix-style` и их комбинации;
- строит ранжированный отчет по качеству и затратам.

### 8) Interpretability Tools (`Examples/Interpretability.fs`)

Что делает пример:

- считает confusion matrix и per-class метрики на MNIST;
- обучает бинарный CNN (0 vs 1) и сохраняет saliency maps для тестовых примеров.

### 9) Benchmark Harness (`Examples/BenchmarkHarness.fs`)

Что делает пример:

- прогоняет матрицу экспериментов для Iris и MNIST;
- сравнивает presets архитектур и оптимизаторов;
- экспортирует markdown benchmark-отчет.

---

## Структура проекта

```text
some-code/
  neuro.sln
  neuro/
    Core/
      Matrix.fs
      Activation.fs
    Data.fs
    Layers.fs
    ConvLayers.fs
    Losses.fs
    Optimizers.fs
      Trainer.fs
      TrainerCNN.fs
      Experiments.fs
      Examples/
        Iris.fs
        MNIST.fs
        CNNMNIST.fs
        HyperparameterLab.fs
        RegularizationMNIST.fs
        Interpretability.fs
        BenchmarkHarness.fs
        TicTacToe.fs
        Tests.fs
        Reports/
        Data/
          Iris.csv
          mnist_train.csv
        mnist_test.csv
    Program.fs
    neuro.fsproj
```
