using Microsoft.Maui.Storage;

namespace Neuro.Mobile;

public static class MobileAssetLoader
{
    public static async Task<string[]> LoadLinesAsync(string assetPath)
    {
        await using var stream = await FileSystem.OpenAppPackageFileAsync(assetPath);
        using var reader = new StreamReader(stream);
        var lines = new List<string>();

        while (await reader.ReadLineAsync() is { } line)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                lines.Add(line);
            }
        }

        return lines.ToArray();
    }
}
