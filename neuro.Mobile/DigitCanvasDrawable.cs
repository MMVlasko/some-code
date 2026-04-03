using Microsoft.Maui.Graphics;

namespace Neuro.Mobile;

public sealed class DigitCanvasDrawable : IDrawable
{
    private const int GridSize = 28;
    private readonly float[,] _pixels = new float[GridSize, GridSize];

    public void Draw(ICanvas canvas, RectF dirtyRect)
    {
        canvas.SaveState();
        canvas.FillColor = Colors.White;
        canvas.FillRoundedRectangle(dirtyRect, 18);

        var cellWidth = dirtyRect.Width / GridSize;
        var cellHeight = dirtyRect.Height / GridSize;

        for (var row = 0; row < GridSize; row++)
        {
            for (var col = 0; col < GridSize; col++)
            {
                var intensity = _pixels[row, col];
                if (intensity <= 0f)
                {
                    continue;
                }

                var shade = 1f - Math.Clamp(intensity, 0f, 1f);
                canvas.FillColor = new Color(shade, shade, shade);
                canvas.FillRectangle(
                    dirtyRect.Left + col * cellWidth,
                    dirtyRect.Top + row * cellHeight,
                    cellWidth + 0.5f,
                    cellHeight + 0.5f);
            }
        }

        canvas.StrokeColor = Color.FromArgb("#D8CCB0");
        canvas.StrokeSize = 1;
        canvas.DrawRoundedRectangle(dirtyRect, 18);
        canvas.RestoreState();
    }

    public void Paint(PointF point, RectF bounds)
    {
        if (bounds.Width <= 0 || bounds.Height <= 0)
        {
            return;
        }

        var col = (int)(point.X / bounds.Width * GridSize);
        var row = (int)(point.Y / bounds.Height * GridSize);

        PaintCell(row, col, 1f);
        PaintCell(row - 1, col, 0.55f);
        PaintCell(row + 1, col, 0.55f);
        PaintCell(row, col - 1, 0.55f);
        PaintCell(row, col + 1, 0.55f);
        PaintCell(row - 1, col - 1, 0.25f);
        PaintCell(row - 1, col + 1, 0.25f);
        PaintCell(row + 1, col - 1, 0.25f);
        PaintCell(row + 1, col + 1, 0.25f);
    }

    public void Clear()
    {
        Array.Clear(_pixels);
    }

    public double[] GetPixels()
    {
        var flattened = new double[GridSize * GridSize];
        for (var row = 0; row < GridSize; row++)
        {
            for (var col = 0; col < GridSize; col++)
            {
                flattened[row * GridSize + col] = _pixels[row, col];
            }
        }

        return flattened;
    }

    private void PaintCell(int row, int col, float intensity)
    {
        if (row < 0 || row >= GridSize || col < 0 || col >= GridSize)
        {
            return;
        }

        _pixels[row, col] = Math.Max(_pixels[row, col], intensity);
    }
}
