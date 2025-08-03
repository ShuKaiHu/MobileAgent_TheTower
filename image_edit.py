# 需要安裝 Pillow: pip install pillow
from PIL import Image

# 打開原始圖片
img = Image.open('Dimond.jpeg')

# 裁剪鑽石區域 (left, upper, right, lower) 需根據實際位置調整
# diamond = img.crop((40, 780, 120, 860))

# 旋轉到正的方向（例如逆時針90度）
img = img.rotate(19, expand=True)

# 保存新圖片
img.save('diamond+19.png')