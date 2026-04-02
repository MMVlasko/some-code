using Microsoft.Maui.Controls;
using Microsoft.Maui.Controls.Shapes;
using Microsoft.Maui.Graphics;

namespace Neuro.Mobile;

public sealed class MainPage : ContentPage
{
    public MainPage()
    {
        Title = "Neuro Mobile";
        BackgroundColor = Color.FromArgb("#F3ECDD");

        var ticTacToeButton = CreateDemoButton("Open Tic-Tac-Toe Demo", "#245F54", Colors.White);
        ticTacToeButton.Clicked += async (_, _) => await Navigation.PushAsync(new TicTacToePage());

        var irisButton = CreateDemoButton("Open Iris Demo", "#245F54", Colors.White);
        irisButton.Clicked += async (_, _) => await Navigation.PushAsync(new IrisPage());

        var mnistButton = CreateDemoButton("Open MNIST Demo", "#245F54", Colors.White);
        mnistButton.Clicked += async (_, _) => await Navigation.PushAsync(new MnistPage());

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
                        Text = "Neuro Demo Menu",
                        FontSize = 30,
                        FontAttributes = FontAttributes.Bold,
                        TextColor = Color.FromArgb("#3C5A55")
                    },
                    new Label
                    {
                        Text = "Choose a demo to run on the device. All three demos now train on-device through the shared F# core: Tic-Tac-Toe for gameplay, Iris for flower classification, and MNIST for handwritten digit recognition.",
                        FontSize = 15,
                        TextColor = Color.FromArgb("#3C5A55")
                    },
                    CreateDemoCard(
                        "Tic-Tac-Toe",
                        "Train a lightweight model on-device and play against it.",
                        ticTacToeButton),
                    CreateDemoCard(
                        "Iris Classification",
                        "Train a classifier on-device, inspect metrics and predict flower classes from custom input.",
                        irisButton),
                    CreateDemoCard(
                        "MNIST",
                        "Train on the bundled MNIST dataset and recognize digits from a touch canvas.",
                        mnistButton)
                }
            }
        };
    }

    private static Border CreateDemoCard(string title, string description, Button actionButton) =>
        new()
        {
            StrokeShape = new RoundRectangle { CornerRadius = 24 },
            Stroke = Color.FromArgb("#C8B99E"),
            BackgroundColor = Colors.White,
            Padding = new Thickness(18),
            Content = new VerticalStackLayout
            {
                Spacing = 12,
                Children =
                {
                    new Label
                    {
                        Text = title,
                        FontSize = 22,
                        FontAttributes = FontAttributes.Bold,
                        TextColor = Color.FromArgb("#3C5A55")
                    },
                    new Label
                    {
                        Text = description,
                        FontSize = 14,
                        TextColor = Color.FromArgb("#3C5A55")
                    },
                    actionButton
                }
            }
        };

    private static Button CreateDemoButton(string text, string backgroundHex, Color textColor) =>
        new()
        {
            Text = text,
            BackgroundColor = Color.FromArgb(backgroundHex),
            TextColor = textColor,
            CornerRadius = 14,
            Padding = new Thickness(16, 12)
        };
}
