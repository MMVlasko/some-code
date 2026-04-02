using Microsoft.Maui.Controls;
using Microsoft.Maui.Controls.Shapes;
using Microsoft.Maui.Graphics;

namespace Neuro.Mobile;

public sealed class MnistCanvasPage : ContentPage
{
    private readonly MnistModel _model;
    private readonly Action<string> _onPredictionReady;
    private readonly DigitCanvasDrawable _canvasDrawable;
    private readonly GraphicsView _canvasView;
    private readonly Label _predictionLabel;

    public MnistCanvasPage(MnistModel model, Action<string> onPredictionReady)
    {
        _model = model;
        _onPredictionReady = onPredictionReady;

        Title = "Draw Digit";
        BackgroundColor = Color.FromArgb("#F3ECDD");

        _canvasDrawable = new DigitCanvasDrawable();
        _canvasView = new GraphicsView
        {
            Drawable = _canvasDrawable,
            BackgroundColor = Colors.White,
            HorizontalOptions = LayoutOptions.Fill,
            VerticalOptions = LayoutOptions.Fill
        };
        _canvasView.StartInteraction += OnCanvasInteraction;
        _canvasView.DragInteraction += OnCanvasInteraction;

        _predictionLabel = new Label
        {
            Text = "Draw a digit and tap Predict.",
            FontSize = 15,
            TextColor = Color.FromArgb("#3C5A55")
        };

        var clearButton = CreateSecondaryButton("Clear");
        clearButton.Clicked += (_, _) =>
        {
            _canvasDrawable.Clear();
            _canvasView.Invalidate();
            _predictionLabel.Text = "Canvas cleared.";
        };

        var predictButton = CreatePrimaryButton("Predict");
        predictButton.Clicked += (_, _) => PredictCurrentDigit();

        var closeButton = CreateDangerButton("Close");
        closeButton.Clicked += async (_, _) => await Navigation.PopModalAsync();

        var header = CreateHeader();
        var canvasShell = CreateCanvasShell();
        var bottomBar = CreateBottomBar(clearButton, predictButton, closeButton);

        var layoutGrid = new Grid();
        layoutGrid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
        layoutGrid.RowDefinitions.Add(new RowDefinition(GridLength.Star));
        layoutGrid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));

        Grid.SetRow(header, 0);
        Grid.SetRow(canvasShell, 1);
        Grid.SetRow(bottomBar, 2);

        layoutGrid.Children.Add(header);
        layoutGrid.Children.Add(canvasShell);
        layoutGrid.Children.Add(bottomBar);

        Content = layoutGrid;
    }

    private void OnCanvasInteraction(object? sender, TouchEventArgs e)
    {
        if (e.Touches.Length == 0)
        {
            return;
        }

        _canvasDrawable.Paint(e.Touches[0], new RectF(0, 0, (float)_canvasView.Width, (float)_canvasView.Height));
        _canvasView.Invalidate();
    }

    private void PredictCurrentDigit()
    {
        var prediction = _model.Predict(_canvasDrawable.GetPixels());
        var top3 = prediction.Probabilities
            .Select((value, index) => new { Index = index, Value = value })
            .OrderByDescending(x => x.Value)
            .Take(3)
            .Select(x => $"{x.Index}:{x.Value:P1}");

        var text = $"Digit {prediction.PredictedDigit}  |  {string.Join("  ", top3)}";
        _predictionLabel.Text = text;
        _onPredictionReady(text);
    }

    private View CreateHeader() =>
        new VerticalStackLayout
        {
            Padding = new Thickness(24, 24, 24, 12),
            Spacing = 6,
            Children =
            {
                new Label
                {
                    Text = "Draw a Digit",
                    FontSize = 28,
                    FontAttributes = FontAttributes.Bold,
                    TextColor = Color.FromArgb("#3C5A55")
                } 
            }
        };

    private View CreateCanvasShell() =>
        new Border
        {
            Margin = new Thickness(24, 0),
            Stroke = Color.FromArgb("#C8B99E"),
            StrokeShape = new RoundRectangle { CornerRadius = 24 },
            BackgroundColor = Colors.White,
            Padding = new Thickness(12),
            Content = _canvasView
        };

    private View CreateBottomBar(Button clearButton, Button predictButton, Button closeButton) =>
        new VerticalStackLayout
        {
            Padding = new Thickness(24, 16, 24, 24),
            Spacing = 12,
            Children =
            {
                _predictionLabel,
                new HorizontalStackLayout
                {
                    Spacing = 12,
                    Children =
                    {
                        clearButton,
                        predictButton,
                        closeButton
                    }
                }
            }
        };

    private static Button CreatePrimaryButton(string text) =>
        new()
        {
            Text = text,
            BackgroundColor = Color.FromArgb("#245F54"),
            TextColor = Colors.White,
            CornerRadius = 14
        };

    private static Button CreateSecondaryButton(string text) =>
        new()
        {
            Text = text,
            BackgroundColor = Color.FromArgb("#D8CCB0"),
            TextColor = Color.FromArgb("#173D38"),
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
}
