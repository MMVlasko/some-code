using System.Globalization;
using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Controls.Shapes;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Storage;
using Path = System.IO.Path;

namespace Neuro.Mobile;

public sealed class IrisPage : ContentPage
{
    private const int MinEpochs = 10;
    private const int MaxEpochs = 2000;
    private const int EpochStep = 10;
    private const int DefaultBatchSize = 16;
    private const double DefaultLearningRate = 0.001;

    private readonly Label _statusLabel;
    private readonly Label _progressLabel;
    private readonly Label _timingLabel;
    private readonly Label _metricsLabel;
    private readonly Label _epochsValueLabel;
    private readonly ProgressBar _trainingProgressBar;
    private readonly Button _decreaseEpochsButton;
    private readonly Button _increaseEpochsButton;
    private readonly Button _trainButton;
    private readonly Button _deleteModelButton;
    private readonly Button _predictButton;
    private readonly Entry _batchSizeEntry;
    private readonly Entry _learningRateEntry;
    private readonly Entry[] _featureEntries;
    private readonly Label _predictionLabel;

    private IrisModel? _model;
    private bool _isTraining;
    private int _selectedEpochs = 300;
    private readonly string _modelPath = Path.Combine(FileSystem.Current.AppDataDirectory, "models", "iris-model.json");

    public IrisPage()
    {
        Title = "Iris";
        BackgroundColor = Color.FromArgb("#F3ECDD");

        _statusLabel = CreateTextLabel("Train the model or load a saved model.");
        _progressLabel = CreateTextLabel("Training progress will appear here.");
        _timingLabel = CreateSecondaryLabel("No active training session.");
        _metricsLabel = CreateTextLabel("No model trained yet.");
        _predictionLabel = CreateTextLabel("Prediction result will appear here.");

        _trainingProgressBar = new ProgressBar
        {
            Progress = 0.0,
            ProgressColor = Color.FromArgb("#245F54"),
            BackgroundColor = Color.FromArgb("#E7DDCA")
        };

        _epochsValueLabel = new Label
        {
            FontSize = 18,
            FontAttributes = FontAttributes.Bold,
            TextColor = Color.FromArgb("#3C5A55"),
            Text = _selectedEpochs.ToString(),
            HorizontalTextAlignment = TextAlignment.Center,
            VerticalTextAlignment = TextAlignment.Center
        };

        _decreaseEpochsButton = CreateCircleButton("−");
        _decreaseEpochsButton.Clicked += (_, _) => ChangeEpochs(-EpochStep);

        _increaseEpochsButton = CreateCircleButton("+");
        _increaseEpochsButton.Clicked += (_, _) => ChangeEpochs(EpochStep);

        _batchSizeEntry = CreateNumericEntry(DefaultBatchSize.ToString(CultureInfo.InvariantCulture));
        _learningRateEntry = CreateNumericEntry(DefaultLearningRate.ToString(CultureInfo.InvariantCulture));

        _trainButton = CreatePrimaryButton($"Train On Device ({_selectedEpochs} epochs)");
        _trainButton.Clicked += OnTrainClicked;

        _deleteModelButton = CreateDangerButton("Delete Saved Model");
        _deleteModelButton.Clicked += OnDeleteModelClicked;
        _deleteModelButton.IsEnabled = false;

        _featureEntries =
        [
            CreateNumericEntry("5.1"),
            CreateNumericEntry("3.5"),
            CreateNumericEntry("1.4"),
            CreateNumericEntry("0.2")
        ];

        _predictButton = CreatePrimaryButton("Predict Flower Class");
        _predictButton.Clicked += OnPredictClicked;

        Content = new ScrollView
        {
            Content = new VerticalStackLayout
            {
                Padding = new Thickness(24, 32),
                Spacing = 18,
                Children =
                {
                    CreateHeader("Iris Classification", "Train on-device and classify flowers from 4 features."),
                    CreateSettingsCard(),
                    _trainButton,
                    _trainingProgressBar,
                    _progressLabel,
                    _timingLabel,
                    _metricsLabel,
                    _statusLabel,
                    CreatePredictionCard(),
                    _deleteModelButton
                }
            }
        };

        TryLoadSavedModel();
        UpdateInteractiveState();
    }

    private Border CreateSettingsCard() =>
        CreateCard(new VerticalStackLayout
        {
            Spacing = 12,
            Children =
            {
                CreateSectionTitle("Training Settings"),
                new HorizontalStackLayout
                {
                    Spacing = 14,
                    Children =
                    {
                        _decreaseEpochsButton,
                        CreateValuePill("Epochs", _epochsValueLabel),
                        _increaseEpochsButton
                    }
                },
                CreateLabeledEntry("Batch Size", _batchSizeEntry),
                CreateLabeledEntry("Learning Rate", _learningRateEntry)
            }
        });

    private Border CreatePredictionCard() =>
        CreateCard(new VerticalStackLayout
        {
            Spacing = 12,
            Children =
            {
                CreateSectionTitle("Predict"),
                CreateLabeledEntry("Sepal length", _featureEntries[0]),
                CreateLabeledEntry("Sepal width", _featureEntries[1]),
                CreateLabeledEntry("Petal length", _featureEntries[2]),
                CreateLabeledEntry("Petal width", _featureEntries[3]),
                _predictButton,
                _predictionLabel
            }
        });

    private async void OnTrainClicked(object? sender, EventArgs e)
    {
        if (_isTraining)
        {
            return;
        }

        if (!TryReadTrainingSettings(out var batchSize, out var learningRate))
        {
            return;
        }

        SetTrainingState(true);
        _trainingProgressBar.Progress = 0;
        _progressLabel.Text = "Loading bundled Iris dataset...";
        _timingLabel.Text = "Epoch timing will appear after the first epoch.";
        _metricsLabel.Text = "Training is running. Prediction will unlock when the model is ready.";
        _statusLabel.Text = "Training in progress.";

        try
        {
            var lines = await MobileAssetLoader.LoadLinesAsync("Data/Iris.csv");
            var epochs = _selectedEpochs;
            var result = await Task.Run(() =>
            {
                var trained = IrisMobile.Train(
                    lines,
                    epochs,
                    batchSize,
                    learningRate,
                    new Action<MobileTrainingProgress>(progress =>
                        MainThread.BeginInvokeOnMainThread(() =>
                        {
                            _trainingProgressBar.Progress = progress.ProgressRatio;
                            _progressLabel.Text = $"Epoch {progress.Epoch}/{progress.TotalEpochs}  Loss {progress.TrainLoss:F4}";
                            _timingLabel.Text = $"Epoch {progress.EpochSeconds:F2}s  ETA {FormatDuration(progress.EtaSeconds)}";
                        })));

                IrisMobile.SaveModel(trained.Model, _modelPath);
                return trained;
            });

            _model = result.Model;
            _trainingProgressBar.Progress = 1.0;
            _metricsLabel.Text =
                $"Train acc {result.TrainAccuracy:P1}  Val acc {result.ValidationAccuracy:P1}  Loss {result.FirstLoss:F4} -> {result.FinalLoss:F4}";
            _progressLabel.Text = $"Training finished after {result.EpochsCompleted} epochs.";
            _timingLabel.Text = $"Model saved to {_modelPath}";
            _statusLabel.Text = "Model ready. Enter flower features and run prediction.";
        }
        catch (Exception ex)
        {
            _statusLabel.Text = $"Training failed: {ex.Message}";
            _timingLabel.Text = "Training stopped before completion.";
        }
        finally
        {
            SetTrainingState(false);
        }
    }

    private void OnPredictClicked(object? sender, EventArgs e)
    {
        if (_model is null)
        {
            _statusLabel.Text = "Train or load a model first.";
            return;
        }

        var features = new double[4];
        for (var i = 0; i < _featureEntries.Length; i++)
        {
            if (!double.TryParse(_featureEntries[i].Text, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
            {
                _statusLabel.Text = "All four feature values must be valid numbers.";
                return;
            }

            features[i] = parsed;
        }

        var prediction = _model.Predict(features);
        _predictionLabel.Text =
            $"{prediction.PredictedClassName}  |  " +
            $"{prediction.Probabilities[0]:P1}, {prediction.Probabilities[1]:P1}, {prediction.Probabilities[2]:P1}";
        _statusLabel.Text = "Prediction completed.";
    }

    private void TryLoadSavedModel()
    {
        try
        {
            if (!IrisMobile.ModelExists(_modelPath))
            {
                _deleteModelButton.IsEnabled = false;
                return;
            }

            _model = IrisMobile.LoadModel(_modelPath);
            _metricsLabel.Text = "Loaded the last saved Iris model from device storage.";
            _progressLabel.Text = "Saved model ready.";
            _timingLabel.Text = $"Loaded from {_modelPath}";
            _statusLabel.Text = "Saved model loaded. You can predict immediately or retrain.";
            _deleteModelButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            _metricsLabel.Text = "Saved model could not be loaded.";
            _timingLabel.Text = ex.Message;
        }
    }

    private void OnDeleteModelClicked(object? sender, EventArgs e)
    {
        try
        {
            IrisMobile.DeleteModel(_modelPath);
            _model = null;
            _trainingProgressBar.Progress = 0;
            _metricsLabel.Text = "Saved model deleted.";
            _progressLabel.Text = "No saved model on device.";
            _timingLabel.Text = "Train a new model to enable prediction.";
            _predictionLabel.Text = "Prediction result will appear here.";
            _statusLabel.Text = "Saved model removed. Train the model to continue.";
        }
        catch (Exception ex)
        {
            _statusLabel.Text = $"Delete failed: {ex.Message}";
        }

        UpdateInteractiveState();
    }

    private void ChangeEpochs(int delta)
    {
        _selectedEpochs = Math.Clamp(_selectedEpochs + delta, MinEpochs, MaxEpochs);
        _epochsValueLabel.Text = _selectedEpochs.ToString();
        _trainButton.Text = $"Train On Device ({_selectedEpochs} epochs)";
        UpdateInteractiveState();
    }

    private bool TryReadTrainingSettings(out int batchSize, out double learningRate)
    {
        batchSize = 0;
        learningRate = 0;

        if (!int.TryParse(_batchSizeEntry.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out batchSize) || batchSize <= 0)
        {
            _statusLabel.Text = "Batch size must be a positive integer.";
            return false;
        }

        if (!double.TryParse(_learningRateEntry.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out learningRate) || learningRate <= 0)
        {
            _statusLabel.Text = "Learning rate must be a positive number.";
            return false;
        }

        return true;
    }

    private void SetTrainingState(bool isTraining)
    {
        _isTraining = isTraining;
        UpdateInteractiveState();
    }

    private void UpdateInteractiveState()
    {
        _trainButton.IsEnabled = !_isTraining;
        _decreaseEpochsButton.IsEnabled = !_isTraining && _selectedEpochs > MinEpochs;
        _increaseEpochsButton.IsEnabled = !_isTraining && _selectedEpochs < MaxEpochs;
        _batchSizeEntry.IsEnabled = !_isTraining;
        _learningRateEntry.IsEnabled = !_isTraining;
        _predictButton.IsEnabled = !_isTraining && _model is not null;
        _deleteModelButton.IsEnabled = !_isTraining && IrisMobile.ModelExists(_modelPath);
    }

    private static Border CreateCard(View content) =>
        new()
        {
            StrokeShape = new RoundRectangle { CornerRadius = 20 },
            Stroke = Color.FromArgb("#C8B99E"),
            BackgroundColor = Colors.White,
            Padding = new Thickness(16),
            Content = content
        };

    private static View CreateHeader(string title, string subtitle) =>
        new VerticalStackLayout
        {
            Spacing = 6,
            Children =
            {
                new Label
                {
                    Text = title,
                    FontSize = 28,
                    FontAttributes = FontAttributes.Bold,
                    TextColor = Color.FromArgb("#3C5A55")
                },
                new Label
                {
                    Text = subtitle,
                    FontSize = 15,
                    TextColor = Color.FromArgb("#3C5A55")
                }
            }
        };

    private static Label CreateSectionTitle(string text) =>
        new()
        {
            Text = text,
            FontSize = 18,
            FontAttributes = FontAttributes.Bold,
            TextColor = Color.FromArgb("#3C5A55")
        };

    private static Label CreateTextLabel(string text) =>
        new()
        {
            Text = text,
            FontSize = 14,
            TextColor = Color.FromArgb("#3C5A55")
        };

    private static Label CreateSecondaryLabel(string text) =>
        new()
        {
            Text = text,
            FontSize = 13,
            TextColor = Color.FromArgb("#6D837E")
        };

    private static Border CreateValuePill(string title, Label valueLabel) =>
        new()
        {
            StrokeShape = new RoundRectangle { CornerRadius = 18 },
            Stroke = Color.FromArgb("#D8CCB0"),
            BackgroundColor = Color.FromArgb("#F8F2E6"),
            Padding = new Thickness(20, 12),
            Content = new VerticalStackLayout
            {
                Spacing = 2,
                Children =
                {
                    new Label
                    {
                        Text = title,
                        FontSize = 12,
                        TextColor = Color.FromArgb("#6D837E"),
                        HorizontalTextAlignment = TextAlignment.Center
                    },
                    valueLabel
                }
            }
        };

    private static HorizontalStackLayout CreateLabeledEntry(string label, Entry entry) =>
        new()
        {
            Spacing = 12,
            Children =
            {
                new Label
                {
                    Text = label,
                    WidthRequest = 120,
                    VerticalTextAlignment = TextAlignment.Center,
                    TextColor = Color.FromArgb("#3C5A55")
                },
                entry
            }
        };

    private static Entry CreateNumericEntry(string text) =>
        new()
        {
            Text = text,
            Keyboard = Keyboard.Numeric,
            BackgroundColor = Color.FromArgb("#F8F2E6"),
            TextColor = Color.FromArgb("#173D38"),
            HorizontalOptions = LayoutOptions.Fill
        };

    private static Button CreateCircleButton(string text) =>
        new()
        {
            Text = text,
            WidthRequest = 52,
            HeightRequest = 52,
            CornerRadius = 26,
            FontSize = 24,
            FontAttributes = FontAttributes.Bold,
            BackgroundColor = Color.FromArgb("#245F54"),
            TextColor = Colors.White
        };

    private static Button CreatePrimaryButton(string text) =>
        new()
        {
            Text = text,
            BackgroundColor = Color.FromArgb("#245F54"),
            TextColor = Colors.White,
            CornerRadius = 14
        };

    private static Button CreateDangerButton(string text) =>
        new()
        {
            Text = text,
            BackgroundColor = Color.FromArgb("#F0D6D6"),
            TextColor = Color.FromArgb("#8B3A3A"),
            CornerRadius = 14
        };

    private static string FormatDuration(double seconds)
    {
        var duration = TimeSpan.FromSeconds(Math.Max(0.0, seconds));
        return duration.TotalHours >= 1 ? duration.ToString(@"h\:mm\:ss") : duration.ToString(@"m\:ss");
    }
}
