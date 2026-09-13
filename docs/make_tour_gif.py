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
]

def _font(name: str, size: int):
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default(size=size)


bold = _font("DejaVuSans-Bold.ttf", 24)
small = _font("DejaVuSans.ttf", 15)


def slide(paths: list[str], caption: str, n: int) -> Image.Image:
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, CAPTION_H], fill=(33, 37, 41))
    d.text((PAD, CAPTION_H // 2), caption, font=bold, fill=(255, 255, 255), anchor="lm")
    d.text((W - PAD, CAPTION_H // 2), f"looppp · {n}/{len(SLIDES)}", font=small, fill=(173, 181, 189), anchor="rm")

    shots = [Image.open(IMG / p).convert("RGB") for p in paths]
    gap = 14
    box_w, box_h = W - 2 * PAD, H - CAPTION_H - 2 * PAD
    stack_w = max(im.width for im in shots)
    stack_h = sum(im.height for im in shots) + gap * (len(shots) - 1)
    scale = min(box_w / stack_w, (box_h - gap * (len(shots) - 1)) / (stack_h - gap * (len(shots) - 1)), 1.0)
    shots = [im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS) for im in shots]
    total_h = sum(im.height for im in shots) + gap * (len(shots) - 1)
    y = CAPTION_H + PAD + (box_h - total_h) // 2
    for im in shots:
        x = (W - im.width) // 2
        d.rectangle([x - 2, y - 2, x + im.width + 1, y + im.height + 1], outline=(206, 212, 218), width=2)
        canvas.paste(im, (x, y))
        y += im.height + gap
    return canvas


slides = [slide(p, c, i + 1) for i, (p, c) in enumerate(SLIDES)]

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
