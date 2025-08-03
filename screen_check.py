import cv2

img = cv2.imread("screenshot_20250430_011522.jpg")
height, width = img.shape[:2]

# 畫格線
for i in range(1, 4):
    x = int(i * width / 4)
    cv2.line(img, (x, 0), (x, height), (0, 255, 0), 2)

cell_width = width / 4
cell_height = height / 40

for j in range(1, 40):
    y = int(j * height / 40)
    cv2.line(img, (0, y), (width, y), (0, 255, 0), 2)

for col in range(4):
    for row in range(40):
        x = int(col * cell_width)
        y = int(row * cell_height)
        label = f"{col}-{row}"
        cv2.putText(img, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)

cv2.imwrite("screenshot_with_grid_4_40_l.jpg", img)