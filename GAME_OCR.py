from appium import webdriver
from appium.options.ios import XCUITestOptions
import time
import cv2
import numpy as np
import pytesseract
import requests
import datetime
import io
from PIL import Image, ImageDraw
import random  # 確保有引入
import pytesseract

def setup_driver():
    options = XCUITestOptions()
    options.platform_name = "iOS"
    options.platform_version = "18.0"
    options.device_name = "Shu-Kai Hu 的 iPhone"
    options.udid = "00008101-000405C63E20001E"
    options.bundle_id = "com.TechTreeGames.TheTower"
    options.use_new_wda = False
    options.use_prebuilt_wda = True
    options.wda_launch_timeout = 60000
    options.xcode_signing_id = "iPhone Developer"
    return webdriver.Remote("http://localhost:4723", options=options)

def get_screen_size(driver):
    screen_size = driver.get_window_size()
    screen_width = screen_size['width']
    screen_height = screen_size['height']
    print(f"螢幕尺寸: {screen_width} x {screen_height}")
    return screen_width, screen_height

def convert_to_screen(x, y, real_width, real_height, screen_width, screen_height):
    screen_x = int(x / real_width * screen_width)
    screen_y = int(y / real_height * screen_height)
    return screen_x, screen_y

# 同時找 BATTLE 和 RETRY
def detect_targets(screenshot_png):
    nparr = np.frombuffer(screenshot_png, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥加強辨識：把亮色的文字轉成純白，其他變黑
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    targets = []  # (文字, 中心點座標)

    for i in range(n_boxes):
        text = data['text'][i].strip().upper()
        if text in ["BATTLE", "RETRY"]:
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            center_x = x + w // 2
            center_y = y + h // 2
            targets.append((text, center_x, center_y))

            color = (0, 255, 0) if text == "BATTLE" else (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.circle(img, (center_x, center_y), 10, color, thickness=-1)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 顯示處理後的畫面（原本畫上去的）
    # cv2.imshow("Target Detection", img)
    cv2.waitKey(1)

    return targets

def analyze_block(img, grid_x, grid_y, cols, rows):
    real_height, real_width = img.shape[:2]
    cell_width = real_width // cols
    cell_height = real_height // rows
    center_x = grid_x * cell_width + cell_width // 2
    center_y = grid_y * cell_height + cell_height // 2
    half_box = 10  # 正方形邊長的一半 (20x20 的正方形)
    start_x = max(center_x - half_box, 0)
    start_y = max(center_y - half_box, 0)
    end_x = min(center_x + half_box, real_width)
    end_y = min(center_y + half_box, real_height)
    region = img[start_y:end_y, start_x:end_x]
    reshaped = region.reshape(-1, 3)
    unique_colors = np.unique(reshaped, axis=0)
    is_single_color = (len(unique_colors) == 1)
    color = unique_colors[0] if is_single_color else None
    return is_single_color, color, (start_x, start_y, end_x, end_y)

def save_debug_region(img, start_x, start_y, end_x, end_y, grid_x, grid_y):
    debug_img = img.copy()
    cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"region_{grid_x}_{grid_y}_{timestamp}.jpg"
    cv2.imwrite(filename, debug_img)
    print(f"已儲存 {filename}")

def real_touch(driver, x, y):
    driver.execute_script("mobile: tap", {"x": x, "y": y})

def upload_screenshot(screenshot_png):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 用 Pillow 把原本的 PNG 壓成 JPEG
    img = Image.open(io.BytesIO(screenshot_png))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=50)  # quality 越低檔案越小，50是中等畫質
    buffer.seek(0)

    files = {'file': (f"screenshot_{now}.jpg", buffer, 'image/jpeg')}
    try:
        response = requests.post("https://shukaihu.com/upload_screenshot", files=files)
        if response.status_code == 200:
            print(f"✅ 成功上傳截圖 {now}")
        else:
            print(f"⚠️ 上傳失敗，狀態碼: {response.status_code}")
    except Exception as e:
        print(f"❌ 上傳錯誤：{e}")

def find_keywords_and_mark(img_cv, keywords=["WAVE", "DEFENSE", "ATTACK", "UTILITY"]):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    results = []

    for i in range(len(data['text'])):
        word = data['text'][i].strip().upper()
        if word in keywords:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cx, cy = x + w // 2, y + h // 2
            draw.ellipse((cx-5, cy-5, cx+5, cy+5), fill='red')
            draw.text((x, y - 12), word, fill='blue')
            results.append((word, cx, cy))
    
    return img_pil, results

def find_all_text_and_mark(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    results = []

    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word:  # 忽略空字串
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cx, cy = x + w // 2, y + h // 2
            draw.ellipse((cx-5, cy-5, cx+5, cy+5), fill='red')
            draw.text((x, y - 12), word, fill='blue')
            results.append((word, cx, cy))

    return img_pil, results


def detect_game_over_and_wave(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

    retry_found = False
    home_found = False
    wave_result = None

    for i in range(len(data['text'])):
        text = data['text'][i].strip().upper()
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        cx, cy = x + w // 2, y + h // 2

        # 判斷是否有結束畫面
        if text == "RETRY" and abs(cx - 108) < 40 and abs(cy - 588) < 40:
            retry_found = True
        if text == "HOME" and abs(cx - 281) < 40 and abs(cy - 587) < 40:
            home_found = True

    if retry_found and home_found:
        for i in range(len(data['text'])):
            text = data['text'][i].strip().upper()
            if text == "WAVE":
                # 嘗試讀取下一個文字作為數字（波數）
                if i + 1 < len(data['text']):
                    wave_text = data['text'][i + 1].strip()
                    wave_result = wave_text
                break

    return retry_found and home_found, wave_result




def main():
    driver = setup_driver()
    screen_width, screen_height = 390, 844  # iPhone 螢幕尺寸

    try:
        time.sleep(20)
        # 擷取螢幕並解碼
        screenshot = driver.get_screenshot_as_png()
        img_cv = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        real_height, real_width = img_cv.shape[:2]

        # OCR 偵測所有文字
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

        # 開始畫圖
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        results = []

        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                cx, cy = x + w // 2, y + h // 2
                draw.ellipse((cx-5, cy-5, cx+5, cy+5), fill='red')
                draw.text((x, y - 12), word, fill='blue')

                sx, sy = convert_to_screen(cx, cy, real_width, real_height, screen_width, screen_height)
                results.append((word, sx, sy))

        # 儲存圖片
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_text_{timestamp}.jpg"
        img_pil.save(filename)
        print(f"✅ 已儲存標記圖片：{filename}")

        # 印出結果
        print("🔍 偵測到以下畫面文字（轉為 iPhone 螢幕座標）：")
        for word, sx, sy in results:
            print(f" - {word}: ({sx}, {sy})")

    finally:
        driver.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# 藍 = [200 158  25] = [(49, 809)]
# 紅 = [ 88  64 205] = [(147, 809)]
# 黃 = [ 17 184 221] = [(245, 809)]
# 綠 = [ 75 214  96] = [(343, 809)]
# 這些顏色是從截圖中抓出來的，可能會隨著遊戲更新而改變