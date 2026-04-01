using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Controls.Shapes;
using Microsoft.Maui.Graphics;
using Microsoft.Maui.Storage;
using Path = System.IO.Path;

namespace Neuro.Mobile;

public sealed class TicTacToePage : ContentPage
{
    private const int MinEpochs = 10;
    private const int MaxEpochs = 300;
    private const int EpochStep = 10;

    private readonly Button[,] _cells = new Button[3, 3];
    private readonly Label _statusLabel;
    private readonly Label _progressLabel;
    private readonly Label _timingLabel;
    private readonly Label _metricsLabel;
    private readonly Label _epochsValueLabel;
    private readonly ProgressBar _trainingProgressBar;
    private readonly Button _decreaseEpochsButton;
    private readonly Button _increaseEpochsButton;
    private readonly Button _trainButton;
    private readonly Button _newGameButton;
    private readonly Button _deleteModelButton;

    private TicTacToeModel? _model;
    private int[,] _board = TicTacToeMobile.CreateEmptyBoard();
    private bool _isTraining;
    private bool _gameFinished;
    private int _selectedEpochs = 90;
    private readonly string _modelPath = Path.Combine(FileSystem.Current.AppDataDirectory, "models", "tictactoe-model.json");

    public TicTacToePage()
    {
        
        Title = "Tic-Tac-Toe";
        BackgroundColor = Color.FromArgb("#F3ECDD");

        _statusLabel = new Label
        {
            FontSize = 18,
            FontAttributes = FontAttributes.Bold,
            TextColor = Color.FromArgb("#3C5A55"),
            Text = "Train the model to start a game."
        };

        _progressLabel = new Label
        {
            FontSize = 14,
            TextColor = Color.FromArgb("#3C5A55"),
            Text = "Training progress will appear here."
        };

        _timingLabel = new Label
        {
            FontSize = 13,
            TextColor = Color.FromArgb("#6D837E"),
            Text = "No active training session."
        };

        _metricsLabel = new Label
        {
            FontSize = 14,
            TextColor = Color.FromArgb("#3C5A55"),
            Text = "No model trained yet."
        };

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

        _decreaseEpochsButton = CreateEpochButton("−");
        _decreaseEpochsButton.Clicked += (_, _) => ChangeEpochs(-EpochStep);

        _increaseEpochsButton = CreateEpochButton("+");
        _increaseEpochsButton.Clicked += (_, _) => ChangeEpochs(EpochStep);

        _trainButton = new Button
        {
            Text = $"Train On Device ({_selectedEpochs} epochs)",
            BackgroundColor = Color.FromArgb("#245F54"),
            TextColor = Colors.White,
            CornerRadius = 14
        };
        _trainButton.Clicked += OnTrainClicked;

        _newGameButton = new Button
        {
            Text = "New Game",
            BackgroundColor = Color.FromArgb("#D8CCB0"),
            TextColor = Color.FromArgb("#173D38"),
            CornerRadius = 14,
            IsEnabled = false
        };
        _newGameButton.Clicked += (_, _) => ResetBoard("Your turn.");

        _deleteModelButton = new Button
        {
            Text = "Delete Saved Model",
            BackgroundColor = Color.FromArgb("#F0D6D6"),
            TextColor = Color.FromArgb("#8B3A3A"),
            CornerRadius = 14,
            IsEnabled = false
        };
        _deleteModelButton.Clicked += OnDeleteModelClicked;

        var boardGrid = new Grid
        {
            RowSpacing = 10,
            ColumnSpacing = 10,
            HorizontalOptions = LayoutOptions.Center,
            VerticalOptions = LayoutOptions.Center
        };

        for (var index = 0; index < 3; index++)
        {
            boardGrid.RowDefinitions.Add(new RowDefinition(new GridLength(88, GridUnitType.Absolute)));
            boardGrid.ColumnDefinitions.Add(new ColumnDefinition(new GridLength(88, GridUnitType.Absolute)));
        }

        for (var row = 0; row < 3; row++)
        {
            for (var col = 0; col < 3; col++)
            {
                var cellButton = new Button
                {
                    FontSize = 32,
                    FontAttributes = FontAttributes.Bold,
                    BackgroundColor = Colors.White,
                    BorderColor = Color.FromArgb("#245F54"),
                    BorderWidth = 2,
                    CornerRadius = 20
                };

                var currentRow = row;
                var currentCol = col;
                cellButton.Clicked += (_, _) => OnBoardTapped(currentRow, currentCol);

                _cells[row, col] = cellButton;
                boardGrid.Add(cellButton, col, row);
            }
        }

        Content = new ScrollView
        {
            Content = new VerticalStackLayout
            {
                Padding = new Thickness(24, 32),
                Spacing = 18,
                Children =
                {
                    new Label
                    {
                        Text = "Tic-Tac-Toe Neural Demo",
                        FontSize = 28,
                        FontAttributes = FontAttributes.Bold,
                        TextColor = Color.FromArgb("#3C5A55")
                    },
                    new Label
                    {
                        Text = "Configure training, train on-device, then play against the resulting model.",
                        FontSize = 15,
                        TextColor = Color.FromArgb("#3C5A55")
                    },
                    CreateSettingsCard(),
                    _trainButton,
                    _trainingProgressBar,
                    _progressLabel,
                    _timingLabel,
                    _metricsLabel,
                    _statusLabel,
                    boardGrid,
                    _newGameButton,
                    _deleteModelButton
                }
            }
        };

        TryLoadSavedModel();
        RefreshBoard();
    }

    private Border CreateSettingsCard() =>
        new()
        {
            StrokeShape = new RoundRectangle { CornerRadius = 20 },
            Stroke = Color.FromArgb("#C8B99E"),
            BackgroundColor = Colors.White,
            Padding = new Thickness(16),
            Content = new VerticalStackLayout
            {
                Spacing = 12,
                Children =
                {
                    new Label
                    {
                        Text = "Training Settings",
                        FontSize = 18,
                        FontAttributes = FontAttributes.Bold,
                        TextColor = Color.FromArgb("#3C5A55")
                    },
                    new HorizontalStackLayout
                    {
                        Spacing = 12,
                        VerticalOptions = LayoutOptions.Center,
                        Children =
                        {
                            new Label
                            {
                                Text = "Epochs",
                                FontSize = 16,
                                VerticalTextAlignment = TextAlignment.Center,
                                TextColor = Color.FromArgb("#3C5A55")
                            }
                        }
                    },
                    new HorizontalStackLayout
                    {
                        Spacing = 14,
                        VerticalOptions = LayoutOptions.Center,
                        Children =
                        {
                            _decreaseEpochsButton,
                            new Border
                            {
                                StrokeShape = new RoundRectangle { CornerRadius = 18 },
                                Stroke = Color.FromArgb("#D8CCB0"),
                                BackgroundColor = Color.FromArgb("#F8F2E6"),
                                Padding = new Thickness(20, 12),
                                Content = new VerticalStackLayout
                                {
                                    Spacing = 2,
                                    HorizontalOptions = LayoutOptions.Center,
                                    Children =
                                    {
                                        new Label
                                        {
                                            Text = "Selected",
                                            FontSize = 12,
                                            TextColor = Color.FromArgb("#6D837E"),
                                            HorizontalTextAlignment = TextAlignment.Center
                                        },
                                        _epochsValueLabel
                                    }
                                }
                            },
                            _increaseEpochsButton
                        }
                    }
                }
            }
        };

    private async void OnTrainClicked(object? sender, EventArgs e)
    {
        if (_isTraining)
        {
            return;
        }

        SetTrainingState(true);
        _trainingProgressBar.Progress = 0.0;
        _progressLabel.Text = $"Preparing training dataset for {_selectedEpochs} epochs...";
        _timingLabel.Text = "Epoch timing will appear after the first completed epoch.";
        _statusLabel.Text = "Training in progress.";
        _metricsLabel.Text = "Training is running. The game will unlock when the model is ready.";

        try
        {
            var epochs = _selectedEpochs;
            var result = await Task.Run(() =>
            {
                var trained = TicTacToeMobile.Train(
                    epochs,
                    new Action<MobileTrainingProgress>(progress =>
                        MainThread.BeginInvokeOnMainThread(() =>
                        {
                            _trainingProgressBar.Progress = progress.ProgressRatio;
                            _progressLabel.Text =
                                $"Epoch {progress.Epoch}/{progress.TotalEpochs}  Loss {progress.TrainLoss:F4}";
                            _timingLabel.Text =
                                $"Epoch {progress.EpochSeconds:F2}s  ETA {FormatDuration(progress.EtaSeconds)}";
                        })));

                TicTacToeMobile.SaveModel(trained.Model, _modelPath);
                return trained;
            });

            _model = result.Model;
            _metricsLabel.Text =
                $"Train acc {result.TrainAccuracy:P1}  Test acc {result.TestAccuracy:P1}  Loss {result.FirstLoss:F4} -> {result.FinalLoss:F4}";
            _trainingProgressBar.Progress = 1.0;
            _progressLabel.Text = $"Training finished after {result.EpochsCompleted} epochs.";
            _timingLabel.Text = $"Model saved to {_modelPath}";
            ResetBoard("Training completed. You can play now.");
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

    private void OnBoardTapped(int row, int col)
    {
        if (_isTraining || _model is null || _gameFinished)
        {
            return;
        }

        if (!TicTacToeMobile.TryMakeMove(_board, row, col, MobileCell.X))
        {
            _statusLabel.Text = "That cell is already occupied.";
            return;
        }

        RefreshBoard();
        if (TryFinishGame())
        {
            return;
        }

        var aiMove = _model.ApplyAiMove(_board);
        RefreshBoard();

        if (TryFinishGame())
        {
            return;
        }

        _statusLabel.Text = $"AI moved to row {aiMove.Row}, column {aiMove.Col}. Your turn.";
    }

    private bool TryFinishGame()
    {
        var status = TicTacToeMobile.EvaluateBoard(_board);
        switch (status)
        {
            case MobileGameStatus.InProgress:
                _gameFinished = false;
                return false;
            case MobileGameStatus.XWins:
                _statusLabel.Text = "You won.";
                break;
            case MobileGameStatus.OWins:
                _statusLabel.Text = "AI won.";
                break;
            default:
                _statusLabel.Text = "Draw.";
                break;
        }

        _gameFinished = true;
        RefreshBoard();
        return true;
    }

    private void ResetBoard(string statusText)
    {
        _board = TicTacToeMobile.CreateEmptyBoard();
        _gameFinished = false;
        _statusLabel.Text = statusText;
        RefreshBoard();
    }

    private void SetTrainingState(bool isTraining)
    {
        _isTraining = isTraining;
        _trainButton.IsEnabled = !isTraining;
        _decreaseEpochsButton.IsEnabled = !isTraining && _selectedEpochs > MinEpochs;
        _increaseEpochsButton.IsEnabled = !isTraining && _selectedEpochs < MaxEpochs;
        _newGameButton.IsEnabled = !isTraining && _model is not null;
        _deleteModelButton.IsEnabled = !isTraining && TicTacToeMobile.ModelExists(_modelPath);
        RefreshBoard();
    }

    private Button CreateEpochButton(string text) =>
        new()
        {
            Text = text,
            WidthRequest = 52,
            HeightRequest = 52,
            CornerRadius = 26,
            FontSize = 24,
            FontAttributes = FontAttributes.Bold,
            BackgroundColor = Color.FromArgb("#245F54"),
            TextColor = Colors.White,
            Shadow = new Shadow
            {
                Brush = new SolidColorBrush(Color.FromArgb("#33245F54")),
                Offset = new Point(0, 6),
                Radius = 12,
                Opacity = 0.6f
            }
        };

    private void ChangeEpochs(int delta)
    {
        var updated = Math.Clamp(_selectedEpochs + delta, MinEpochs, MaxEpochs);
        if (updated == _selectedEpochs)
        {
            return;
        }

        _selectedEpochs = updated;
        _epochsValueLabel.Text = _selectedEpochs.ToString();
        _trainButton.Text = $"Train On Device ({_selectedEpochs} epochs)";
        _decreaseEpochsButton.IsEnabled = !_isTraining && _selectedEpochs > MinEpochs;
        _increaseEpochsButton.IsEnabled = !_isTraining && _selectedEpochs < MaxEpochs;
    }

    private void TryLoadSavedModel()
    {
        try
        {
            if (!TicTacToeMobile.ModelExists(_modelPath))
            {
                _statusLabel.Text = "Train the model to start a game.";
                return;
            }

            _model = TicTacToeMobile.LoadModel(_modelPath);
            _metricsLabel.Text = "Loaded the last saved model from device storage.";
            _progressLabel.Text = "Saved model ready.";
            _timingLabel.Text = $"Loaded from {_modelPath}";
            _statusLabel.Text = "Saved model loaded. You can play now or retrain.";
            _deleteModelButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            _metricsLabel.Text = "Saved model could not be loaded.";
            _timingLabel.Text = ex.Message;
        }
    }

    private static string FormatDuration(double seconds)
    {
        var duration = TimeSpan.FromSeconds(Math.Max(0.0, seconds));
        if (duration.TotalHours >= 1)
        {
            return duration.ToString(@"h\:mm\:ss");
        }

        return duration.ToString(@"m\:ss");
    }

    private void OnDeleteModelClicked(object? sender, EventArgs e)
    {
        try
        {
            TicTacToeMobile.DeleteModel(_modelPath);
            _model = null;
            _board = TicTacToeMobile.CreateEmptyBoard();
            _gameFinished = false;
            _trainingProgressBar.Progress = 0.0;
            _metricsLabel.Text = "Saved model deleted.";
            _progressLabel.Text = "No saved model on device.";
            _timingLabel.Text = "Train a new model to play again.";
            _statusLabel.Text = "Saved model removed. Train the model to start a game.";
            _deleteModelButton.IsEnabled = false;
            RefreshBoard();
        }
        catch (Exception ex)
        {
            _statusLabel.Text = $"Delete failed: {ex.Message}";
        }
    }

    private void RefreshBoard()
    {
        for (var row = 0; row < 3; row++)
        {
            for (var col = 0; col < 3; col++)
            {
                var value = _board[row, col];
                var button = _cells[row, col];
                button.Text = value switch
                {
                    1 => "X",
                    2 => "O",
                    _ => string.Empty
                };
                button.TextColor = value switch
                {
                    1 => Color.FromArgb("#C94C4C"),
                    2 => Color.FromArgb("#3F72AF"),
                    _ => Color.FromArgb("#245F54")
                };
                button.IsEnabled = !_isTraining && !_gameFinished && _model is not null && value == 0;
            }
        }

        _newGameButton.IsEnabled = !_isTraining && _model is not null;
    }
}
