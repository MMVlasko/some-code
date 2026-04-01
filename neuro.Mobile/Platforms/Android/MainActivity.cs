using Android.App;
using Android.Content.PM;
using Android.Graphics;
using Android.OS;
using Android.Views;
using Android.Graphics.Drawables;
using AndroidX.Core.View;
using Color = Android.Graphics.Color;

namespace Neuro.Mobile;

[Activity(
    Theme = "@style/Maui.SplashTheme",
    MainLauncher = true,
    ConfigurationChanges = ConfigChanges.ScreenSize
        | ConfigChanges.Orientation
        | ConfigChanges.UiMode
        | ConfigChanges.ScreenLayout
        | ConfigChanges.SmallestScreenSize
        | ConfigChanges.Density)]
public class MainActivity : MauiAppCompatActivity
{
    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);
        ApplyChromeColors();
    }

    protected override void OnResume()
    {
        base.OnResume();
        ApplyChromeColors();
    }

    private void ApplyChromeColors()
    {
        var pageColor = Color.Rgb(243, 236, 221);

        if (Window is not null)
        {
            Window.DecorView?.SetBackgroundColor(pageColor);

            Window?.SetStatusBarColor(pageColor);

            var controller = WindowCompat.GetInsetsController(Window, Window.DecorView);
            if (controller is not null)
            {
                controller.AppearanceLightStatusBars = true;
            }
        }

        SupportActionBar?.SetBackgroundDrawable(new ColorDrawable(pageColor));
    }
}
