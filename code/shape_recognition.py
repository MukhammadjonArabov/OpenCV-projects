import cv2

image_path = "../images1.jpg"
image = cv2.imread(image_path)

if image is None:
    print("XATO: 'image.jpg' topilmadi! Rasm qo'ying.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = image.copy()
shakl_soni = 0

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 500:
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    sides = len(approx)

    if sides == 3:
        shape = "UCHBURCHAK"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h)
        shape = "KVADRAT" if 0.95 <= aspect <= 1.05 else "TO'RTBURCHAK"
    elif sides == 5:
        shape = "BESHBURCHAK"
    elif sides == 6:
        shape = "OLTI BURCHAK"
    elif sides > 6:
        shape = "AYLANA"
    else:
        shape = "NOMA'LUM"

    shakl_soni += 1

    cv2.drawContours(result, [cnt], -1, (0, 255, 0), 3)

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(result, (cx, cy), 7, (255, 0, 0), -1)
        cv2.putText(result, shape, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    print(f"{shakl_soni:2}. {shape:12} | Maydon: {int(area):6} px | Nuqtalar: {sides}")


cv2.imwrite("result_shapes.jpg", result)

cv2.imshow("Shakllar aniqlandi", result)
cv2.waitKey(0)
cv2.destroyAllWindows()