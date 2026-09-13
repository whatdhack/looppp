"""Build docs/images/loop-tour.gif: a captioned slideshow of the single-notebook screenshots.

Run from the repo root after replacing screenshots in docs/images/:
    uv run --with pillow python docs/make_tour_gif.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMG = Path(__file__).resolve().parent / "images"
OUT = IMG / "loop-tour.gif"
W, H = 1000, 600                 # canvas
CAPTION_H = 64
PAD = 20
HOLD_MS, LAST_HOLD_MS, FADE_FRAMES, FADE_MS = 3200, 4500, 5, 70

SLIDES = [
    (["loop-attach-gpu.png"], "1  Open loop.py in molab, attach the GPU"),
    (["loop-settings.png", "loop-test-model.png"], "2  Settings, then test the model"),
    (["loop-progress.png"], "3  Start: live progress and attempts"),
    (["loop-best-kernel.png"], "4  Best kernel, ready to download"),
    (["loop-wandb-gpu-metrics.png"], "5  W&B system metrics: GPU"),
    (["loop-wandb-system-metrics.png"], "6  W&B system metrics: network, disk, memory"),
    (["loop-aria-run-analysis.png"], "7  Ask ARIA about a running loop"),
    ([("loop-aria-run-analysis.png", (1290, 0, 1693, 405)), ("loop-aria-run-analysis.png", (1290, 405, 1693, 590))],
     "8  ARIA's reading of the run", "row"),
]

def _font(name: str, size: int):
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default(size=size)


bold = _font("DejaVuSans-Bold.ttf", 24)
small = _font("DejaVuSans.ttf", 15)


def _open(item) -> Image.Image:
    """item: file name, or (file name, crop box)."""
    name, box = (item, None) if isinstance(item, str) else item
    im = Image.open(IMG / name).convert("RGB")
    return im.crop(box) if box else im


def slide(paths: list, caption: str, n: int, layout: str = "column") -> Image.Image:
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, CAPTION_H], fill=(33, 37, 41))
    d.text((PAD, CAPTION_H // 2), caption, font=bold, fill=(255, 255, 255), anchor="lm")
    d.text((W - PAD, CAPTION_H // 2), f"looppp · {n}/{len(SLIDES)}", font=small, fill=(173, 181, 189), anchor="rm")

    shots = [_open(p) for p in paths]
    gap = 14
    box_w, box_h = W - 2 * PAD, H - CAPTION_H - 2 * PAD
    extra = gap * (len(shots) - 1)
    if layout == "row":
        scale = min((box_w - extra) / sum(im.width for im in shots), box_h / max(im.height for im in shots), 1.0)
    else:
        scale = min(box_w / max(im.width for im in shots), (box_h - extra) / sum(im.height for im in shots), 1.0)
    shots = [im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS) for im in shots]
    if layout == "row":
        x = (W - (sum(im.width for im in shots) + extra)) // 2
        top = CAPTION_H + PAD + (box_h - max(im.height for im in shots)) // 2
        for im in shots:
            d.rectangle([x - 2, top - 2, x + im.width + 1, top + im.height + 1], outline=(206, 212, 218), width=2)
            canvas.paste(im, (x, top))
            x += im.width + gap
    else:
        y = CAPTION_H + PAD + (box_h - (sum(im.height for im in shots) + extra)) // 2
        for im in shots:
            x = (W - im.width) // 2
            d.rectangle([x - 2, y - 2, x + im.width + 1, y + im.height + 1], outline=(206, 212, 218), width=2)
            canvas.paste(im, (x, y))
            y += im.height + gap
    return canvas


slides = [slide(spec[0], spec[1], i + 1, *spec[2:]) for i, spec in enumerate(SLIDES)]

frames, durations = [], []
for i, s in enumerate(slides):
    frames.append(s)
    durations.append(LAST_HOLD_MS if i == len(slides) - 1 else HOLD_MS)
    nxt = slides[(i + 1) % len(slides)]
    for k in range(1, FADE_FRAMES + 1):
        frames.append(Image.blend(s, nxt, k / (FADE_FRAMES + 1)))
        durations.append(FADE_MS)

# one shared palette built from all slides keeps colours stable between frames and the file small
strip = Image.new("RGB", (W, H * len(slides)))
for i, s in enumerate(slides):
    strip.paste(s, (0, H * i))
palette_img = strip.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
pal_frames = [f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in frames]

pal_frames[0].save(OUT, save_all=True, append_images=pal_frames[1:], duration=durations, loop=0,
                   optimize=True, disposal=1)
total_s = sum(durations) / 1000
print(f"{OUT} {OUT.stat().st_size // 1024} KB, {len(frames)} frames, {total_s:.1f}s per loop")
