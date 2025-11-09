import cv2
import numpy as np

def get_color_name(bgr_color):
    b, g, r = map(int, bgr_color)
    if r > g and r > b:
        if r > 200: return "Qizil"
        elif r > 100: return "To'q sariq"
        else: return "Qizil-jigarrang"
    elif g > r and g > b:
        if g > 200: return "Yashil"
        elif g > 100: return "Yashil-sariq"
        else: return "Qorong‘i yashil"
    elif b > r and b > g:
        if b > 200: return "Ko‘k"
        elif b > 100: return "Moviy"
        else: return "Qorong‘i ko‘k"
    else:
        return "Aralash / Oq"

def get_shape_name(approx):
    sides = len(approx)
    if sides == 3:
        return "Uchburchak"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Kvadrat"
        else:
            return "To‘rtburchak"
    elif sides == 5:
        return "Beshburchak"
    elif sides == 6:
        return "Olti burchak"
    elif sides > 6:
        return "Doira"
    else:
        return "Noma'lum"

def find_dominant_colors(image, k=5):
    data = image.reshape((-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    counts = np.bincount(labels.flatten())

    return centers, counts

image_path = "image.png"
image = cv2.imread(image_path)

if image is None:
    print("XATO: Rasm topilmadi! Yo‘lni tekshiring:", image_path)
    exit()

original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("\n" + "="*50)
print("     SHAKLLAR VA RANGLAR TAHLILI")
print("="*50)

shakllar_soni = 0
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 500:
        continue

    shakllar_soni += 1

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    shape_name = get_shape_name(approx)

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_color_tuple = cv2.mean(image, mask=mask)[:3]  # (B, G, R)
    mean_color_np = np.array(mean_color_tuple)
    mean_color_int = mean_color_np.astype(int)

    color_name = get_color_name(mean_color_int)

    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)
    cv2.putText(image, f"{shape_name}: {color_name}", (cX - 70, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    print(f"Shakl {shakllar_soni}:")
    print(f"   ↳ Shakl: {shape_name}")
    print(f"   ↳ Rang: {color_name}")
    print(f"   ↳ BGR: {mean_color_int}")
    print(f"   ↳ Maydon: {area:.0f} px")
    print("-" * 40)

if shakllar_soni == 0:
    print("Hech qanday shakl topilmadi. Rasmni tekshiring yoki sifatni oshiring.")

print("\nDOMINANT RANGLAR (TOP 5)")
print("-" * 40)

colors, counts = find_dominant_colors(original, k=5)
total_pixels = np.sum(counts)

for i in range(len(colors)):
    percentage = (counts[i] / total_pixels) * 100
    color_bgr = colors[i]
    color_name = get_color_name(color_bgr)
    print(f"Rang {i+1}:")
    print(f"   ↳ Nomi: {color_name}")
    print(f"   ↳ BGR: {color_bgr}")
    print(f"   ↳ Foiz: {percentage:.1f}%")
    print("-" * 30)

cv2.imshow("Natija: Shakllar va Ranglar", image)
cv2.imshow("Konturlar (Canny)", edged)

print("\nEkranni yopish uchun 'q' tugmasini bosing...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nDastur muvaffaqiyatli yakunlandi!")