import cv2
import numpy as np
from collections import Counter

COLORS = {
    "Qizil": (0, 0, 255),
    "To‘q qizil": (0, 0, 139),
    "Yashil": (0, 255, 0),
    "Zumrad yashil": (0, 100, 0),
    "Ko‘k": (255, 0, 0),
    "Moviy": (139, 0, 0),
    "Sariq": (0, 255, 255),
    "Och sariq": (173, 255, 47),
    "Binafsha": (128, 0, 128),
    "Pushti": (203, 192, 255),
    "Jigarrang": (42, 42, 165),
    "Qora": (0, 0, 0),
    "Oq": (255, 255, 255),
    "Kulrang": (128, 128, 128),
    "To‘q kulrang": (64, 64, 64),
    "Och kulrang": (192, 192, 192),
    "Och yashil": (144, 238, 144),
    "Moviy-yashil": (175, 238, 238),
    "Qizg‘ish sariq": (220, 20, 60),
    "Nilufar": (255, 228, 196),
    "Olov rang": (255, 69, 0),
    "Zumrad": (80, 200, 120)
}

def color_distance(c1, c2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def find_nearest_color(bgr_pixel):
    min_dist = float('inf')
    nearest_name = "Noma'lum"
    for name, color in COLORS.items():
        dist = color_distance(bgr_pixel, color)
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
    return nearest_name

def draw_palette(image, color_stats):
    h, w = 60, 60
    num_colors = len(color_stats)
    target_width = image.shape[1]
    palette_width = w * num_colors

    if palette_width > target_width:
        w = target_width // num_colors
        if w < 30:
            w = 30
            num_cols = target_width // w
            color_stats = color_stats[:num_cols]
            num_colors = len(color_stats)

    palette = np.zeros((h * 2, w * num_colors, 3), dtype=np.uint8)
    x_offset = 0
    for name, bgr, percent in color_stats:
        color_bgr = np.array(bgr, dtype=np.uint8)
        cv2.rectangle(palette, (x_offset, 0), (x_offset + w, h), color_bgr.tolist(), -1)
        cv2.putText(palette, name[:6], (x_offset + 2, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(palette, f"{percent:.0f}%", (x_offset + 2, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        x_offset += w

    palette = cv2.resize(palette, (target_width, h * 2))

    combined = np.vstack((image, palette))
    return combined

image_path = "image.png"
image = cv2.imread(image_path)

if image is None:
    print(f"XATO: '{image_path}' topilmadi!")
    exit()

original = image.copy()

image = cv2.resize(image, (300, 300))
pixels = image.reshape((-1, 3))

color_names = [find_nearest_color(pixel) for pixel in pixels]
color_counts = Counter(color_names)
total_pixels = len(pixels)

color_stats = []
for name, count in color_counts.most_common():
    percent = (count / total_pixels) * 100
    if percent >= 0.1:
        bgr = COLORS.get(name, (128, 128, 128))
        color_stats.append((name, bgr, percent))

print("\n" + "=" * 70)
print("     RANG TAHLILI (22 ta rang)")
print("=" * 70)
for name, _, percent in color_stats:
    bar = "█" * max(1, int(percent // 2))
    print(f"{name:15} | {percent:5.1f}% | {bar}")

resized_original = cv2.resize(original, (300, 300))
result_image = draw_palette(resized_original, color_stats)

cv2.imshow("Rang tahlili + Palitra", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
