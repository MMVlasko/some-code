using Microsoft.Maui.Controls;

namespace Neuro.Mobile;

public sealed class App : Application
{
    protected override Window CreateWindow(IActivationState? activationState)
    {
        var navigationPage = new NavigationPage(new MainPage())
        {
            BarBackgroundColor = Color.FromArgb("#F3ECDD"),
            BarTextColor = Color.FromArgb("#3C5A55")
        };

        return new Window(navigationPage);
    }
}
