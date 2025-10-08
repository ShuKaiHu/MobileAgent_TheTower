import re
import time

# 全域開關：是否保存任何圖片（debug/截圖）
SAVE_IMAGES = False
import cv2
import numpy as np
import pytesseract
import datetime
import csv
import os
from PIL import Image
from appium import webdriver
from appium.options.ios import XCUITestOptions
import random
import datetime
import cv2
import numpy as np
import time


# ========================
#  Appium 驅動初始化設定
# ======================== 
def setup_driver():
    options = XCUITestOptions()
    options.platform_name = "iOS"
    options.device_name = "iPhone"
    options.udid = "00008101-000405C63E20001E"
    options.bundle_id = "com.TechTreeGames.TheTower"
    options.updated_wda_bundle_id = "com.shukaihu.WebDriverAgentRunner"

    options.set_capability("wdaLocalPort", 8100)
    options.set_capability("showXcodeLog", True)
    options.set_capability("skipLogCapture", True)
    options.set_capability("wdaStartupRetries", 3)
    options.set_capability("wdaStartupRetryInterval", 10000)
    options.set_capability("waitForQuiescence", False)
    options.set_capability("newCommandTimeout", 1200)

    return webdriver.Remote("http://localhost:4723", options=options)

# =========================
#  取得螢幕寬高 (Appium)
# =========================
def get_screen_size(driver):
    screen_size = driver.get_window_size()
    return screen_size['width'], screen_size['height']

# ===================================================
#  轉換影像座標 (x,y) 成螢幕點擊座標 (螢幕解析度對齊)
# ===================================================
def convert_to_screen(x, y, real_width, real_height, screen_width, screen_height):
    screen_x = int(x / real_width * screen_width)
    screen_y = int(y / real_height * screen_height)
    return screen_x, screen_y

# =========================
#  點擊螢幕座標
# =========================
def real_touch(driver, x, y):
    driver.execute_script("mobile: tap", {"x": x, "y": y})

# =========================
#  判斷區塊是否為單一顏色
# =========================
def analyze_block(img, grid_x, grid_y, cols, rows):
    real_height, real_width = img.shape[:2]
    cell_width = real_width // cols
    cell_height = real_height // rows
    center_x = grid_x * cell_width + cell_width // 2
    center_y = grid_y * cell_height + cell_height // 2
    half_box = 10
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

# =========================
#  影像旋轉
# =========================
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

# =========================
#  OCR 區域讀取數字
# =========================
def read_number_in_region(img, x1, y1, x2, y2):
    region = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    print(f"\U0001f522 OCR 結果（{x1},{y1} ~ {x2},{y2}）：{text}")
    return text

# =========================
#  存圖區域 (除錯用)
# =========================
def save_crop_region(img, x1, y1, x2, y2):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cropped = img[y1:y2, x1:x2]
    filename = f"region_{x1}_{y1}_{x2}_{y2}_{timestamp}.jpg"
    if SAVE_IMAGES:
        cv2.imwrite(filename, cropped)
        print(f"🖼️ 儲存區域截圖：{filename}")

def detect_game_over_and_wave(img_cv, save_debug=False):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

    found_retry = False
    wave_number = None
    coins_number = None
    tier_number = None

    for i, text in enumerate(data['text']):
        text_clean = text.strip().upper()
        if text_clean == "RETRY":
            found_retry = True
        if text_clean == "WAVE":
            for j in range(i + 1, min(i + 4, len(data['text']))):
                candidate = re.sub(r"[^\d]", "", data['text'][j])
                if candidate.isdigit():
                    wave_number = candidate
                    break
        if text_clean == "TIER":
            for j in range(i + 1, min(i + 4, len(data['text']))):
                candidate = re.sub(r"[^\d]", "", data['text'][j])
                if candidate.isdigit():
                    tier_number = candidate
                    break

    # 原始畫面位置
    screen_x1, screen_y1 = 264, 515
    screen_x2, screen_y2 = 338, 555

    # 原圖尺寸
    real_h, real_w = img_cv.shape[:2]
    screen_w, screen_h = 390, 844

    # 嘗試往下找最多10次
    for offset_try in range(10):
        offset_y = offset_try * 5  # 每次往下5px
        y1 = int((screen_y1 + offset_y) / screen_h * real_h)
        y2 = int((screen_y2 + offset_y) / screen_h * real_h)
        x1 = int(screen_x1 / screen_w * real_w)
        x2 = int(screen_x2 / screen_w * real_w)

        coins_roi = img_cv[y1:y2, x1:x2]
        coins_text = pytesseract.image_to_string(coins_roi, config='--psm 7').strip()
        match = re.search(r'[\d\.]+[KMBT]?', coins_text.upper())
        if match:
            value = match.group(0).upper()
            multiplier = 1
            for suffix, factor in zip(['K', 'M', 'B', 'T'], [1_000, 1_000_000, 1_000_000_000, 1_000_000_000_000]):
                if suffix in value:
                    multiplier = factor
                    value = value.replace(suffix, '')
                    break
            try:
                coins_number = str(int(float(value) * multiplier))
                break
            except ValueError:
                coins_number = None

    screenshot_path = None
    cropped_path = None
    if save_debug:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"screenshot_full_{timestamp}.jpg"
        cropped_path = f"screenshot_crop_{timestamp}.jpg"
        img_copy = img_cv.copy()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if SAVE_IMAGES:
            cv2.imwrite(screenshot_path, img_copy)
            cv2.imwrite(cropped_path, coins_roi)

    return found_retry, wave_number, coins_number, tier_number, screenshot_path, cropped_path

def save_wave_log(start_time, end_time, wave, coins, tier, screenshot_path=None, cropped_path=None):
    file_exists = os.path.exists("wave_log.csv")
    with open("wave_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Start Time", "End Time", "Wave", "Coins", "Tier", "Screenshot", "Crop"])
        writer.writerow([
            start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            wave or "",
            coins or "",
            tier or "",
            screenshot_path or "",
            cropped_path or ""
        ])
    update_wave_log_summary()
    # 🧹 清除每個 Tier 超過 10 筆的舊資料
    try:
        import pandas as pd
        df = pd.read_csv("wave_log.csv")
        df["End Time"] = pd.to_datetime(df["End Time"], errors="coerce")
        df["Tier"] = pd.to_numeric(df["Tier"], errors="coerce", downcast="integer")
        df = df.dropna(subset=["End Time", "Tier"])
        # df = df.sort_values("End Time", ascending=False)
        df = df.sort_values(["Tier", "End Time"], ascending=[True, True])
        df = df.groupby("Tier").head(10)
        df.to_csv("wave_log.csv", index=False)
    except Exception as e:
        print(f"⚠️ 清理舊資料失敗：{e}")

def update_wave_log_summary():
    import pandas as pd
    from pandas.errors import ParserError

    csv_path = "wave_log.csv"
    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
    except ParserError:
        with open(csv_path, "r") as f:
            lines = f.readlines()
        # 自動偵測欄位數
        header_line = lines[0] if lines else ""
        col_count = header_line.count(",") + 1 if header_line else 0
        valid_lines = [line for line in lines if line.count(",") == (col_count - 1)]
        with open("wave_log_clean.csv", "w") as f:
            f.writelines(valid_lines)
        df = pd.read_csv("wave_log_clean.csv")

    df = df.dropna(subset=["Start Time", "End Time", "Coins", "Tier"])
    df["Start Time"] = df["Start Time"].astype(str).str.split(".").str[0]
    df["End Time"] = df["End Time"].astype(str).str.split(".").str[0]
    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
    df["End Time"] = pd.to_datetime(df["End Time"], errors="coerce")
    df["Coins"] = pd.to_numeric(df["Coins"], errors="coerce")
    df["Tier"] = pd.to_numeric(df["Tier"], errors="coerce", downcast="integer")
    df = df.dropna(subset=["Start Time", "End Time", "Coins", "Tier"])

    df["Duration (min)"] = (df["End Time"] - df["Start Time"]).dt.total_seconds() / 60
    df = df[df["Duration (min)"] > 0]
    df["Coins Per Minute"] = df["Coins"] / df["Duration (min)"]
    df["Duration (min)"] = df["Duration (min)"].round(1)
    df["Coins Per Minute"] = df["Coins Per Minute"].round(1)

    # 保留每個 Tier 最新的 10 筆，先按 Tier 升冪、End Time 降冪排序，取最新10筆，再重排為 Tier 升冪、End Time 升冪
    df = df.sort_values(["Tier", "End Time"], ascending=[True, False])
    df = df.groupby("Tier").head(10)
    df = df.sort_values(["Tier", "End Time"], ascending=[True, True])

    # 🧹 同步清理原始 wave_log.csv，只保留每個 Tier 最新 10 筆，且排序
    df.to_csv("wave_log.csv", index=False)

    summary = df.groupby("Tier", as_index=False).agg(
        Amount=("Coins Per Minute", "count"),
        Average_Wave=("Wave", "mean"),
        Average_Coins_Per_Minute=("Coins Per Minute", "mean")
    )
    summary["Average Wave"] = summary["Average_Wave"].round().astype(int)
    summary["Average Coins Per Minute"] = summary["Average_Coins_Per_Minute"].round().astype(int)
    summary = summary[["Tier", "Amount", "Average Wave", "Average Coins Per Minute"]]
    summary.to_csv("wave_log_sum.csv", index=False)
    print("💾 已儲存至 wave_log_sum.csv")

def draw_circle_and_save(screenshot_np, center_x, center_y):
    """在影像中畫圓圈並儲存 debug 圖片"""
    debug_img = screenshot_np.copy()
    cv2.circle(debug_img, (center_x, center_y), radius=50, color=(0, 0, 255), thickness=5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_diamond_{timestamp}.png"
    if SAVE_IMAGES:
        cv2.imwrite(filename, debug_img)
        print(f"🖼️ 已儲存 debug 圖片：{filename}")



# =========================
#  鑽石圖示偵測與點擊
# =========================
def detect_and_click_diamond(driver, img, template_path="diamond_f.png", threshold=0.8):
    import math
    # 點擊前讀取數字
    number_before = read_number_in_region(img, 100, 350, 250, 400)

    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        print("❌ 找不到鑽石圖")
        return False

    # 先進行基本偵測，確認畫面中有鑽石再進行後續動作
    result_check = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val_check, _, _ = cv2.minMaxLoc(result_check)
    if max_val_check < threshold:
        # print("🔍 沒有偵測到鑽石，跳過")
        return False

    # 若偵測到鑽石，開始依角度點擊
    screen_width, screen_height = get_screen_size(driver)
    real_height, real_width = img.shape[:2]

    center_x, center_y = 614, 793  # 預估圓心（太陽）座標
    radius = 160

    for angle in range(0, 360, 10):
        radians = math.radians(angle)
        cx_img = int(center_x + radius * math.cos(radians))
        cy_img = int(center_y + radius * math.sin(radians))
        cx_screen, cy_screen = convert_to_screen(cx_img, cy_img, real_width, real_height, screen_width, screen_height)

        try:
            real_touch(driver, cx_screen, cy_screen)
        except Exception as e:
            print(f"⚠️ 點擊錯誤：{e}")

        time.sleep(0.1)

    # 點擊後重新截圖與讀取數字
    screenshot_after = driver.get_screenshot_as_png()
    img_after = cv2.imdecode(np.frombuffer(screenshot_after, np.uint8), cv2.IMREAD_COLOR)
    number_after = read_number_in_region(img_after, 100, 350, 250, 400)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"💎 {timestamp} ▶ 點擊鑽石 🔢 數字變化：{number_before} ➡ {number_after}")
    return True


def save_crop_region(img, x1, y1, x2, y2):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cropped = img[y1:y2, x1:x2]
    filename = f"region_{x1}_{y1}_{x2}_{y2}_{timestamp}.jpg"
    if SAVE_IMAGES:
        cv2.imwrite(filename, cropped)
        print(f"🖼️ 儲存區域截圖：{filename}")

def read_number_in_region(img, x1, y1, x2, y2):
    import pytesseract

    roi = img[y1:y2, x1:x2]

    # 🎯 色彩過濾：只保留接近白色的像素
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])      # 色相、飽和度、亮度下限
    upper_white = np.array([180, 50, 255])   # 色相、飽和度、亮度上限
    mask = cv2.inRange(hsv, lower_white, upper_white)
    filtered = cv2.bitwise_and(roi, roi, mask=mask)

    # 灰階 + 二值化
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=config)
    cleaned = ''.join(filter(str.isdigit, text))

    # 🌟 限制四位數
    if len(cleaned) != 4:
        cleaned = ""

    # print(f"🔢 OCR 結果（{x1},{y1} ~ {x2},{y2}）：{cleaned if cleaned else text.strip()}")
    return cleaned if cleaned else text.strip()



def draw_debug_points(img, points, filename="debug_click_positions.png"):
    debug_img = img.copy()
    for (x, y) in points:
        cv2.circle(debug_img, (x, y), radius=10, color=(0, 0, 255), thickness=-1)
    if SAVE_IMAGES:
        cv2.imwrite(filename, debug_img)
        print(f"📸 已儲存點擊位置圖：{filename}")






def main():
    driver = setup_driver()
    screen_width, screen_height = get_screen_size(driver)
    game_state = "waiting"
    round_start = None
    SAVE_UNKNOWN_WAVE_SCREENSHOT = False
    last_save_time = time.time()  # ← 新增：紀錄上次截圖時間

    while True:
        screenshot = driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)

        if game_state == "waiting":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            for text in data['text']:
                if text.strip().upper() == "BATTLE":
                    real_touch(driver, 195, 681)
                    round_start = datetime.datetime.now()  # ✅ 保留 datetime 格式
                    time.sleep(1)
                    game_state = "playing"
                    break
            time.sleep(1)

        elif game_state == "playing":
            # 🔍 每回合檢查是否有鑽石圖案出現
            detect_and_click_diamond(driver, img)
            # 🖼️ 每回合儲存裁切畫面（這邊請替換成你要的區域）

            screenshot = driver.get_screenshot_as_png()
            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            is_over, wave_number, coins_number, tier_number, full_path, crop_path = detect_game_over_and_wave(img, save_debug=False)

            if is_over:
                round_end = datetime.datetime.now()  # ← 保留 datetime 物件
                duration_sec = int((round_end - round_start).total_seconds())
                round_end_str = round_end.strftime("%Y-%m-%d %H:%M:%S")
                print(f"🏁 Game Over | Wave: {wave_number} | Coins: {coins_number} | Tier: {tier_number} | ⏱ {round_end_str} | ⌛ {duration_sec}s")  
                game_state = "game_over"
                continue

            # 原本的隨機點擊判斷（移至 is_over 判斷之後）
            grid_x, grid_y, cols, rows = 2, 26, 4, 40
            is_single_color, color, _ = analyze_block(img, grid_x, grid_y, cols, rows)
            if is_single_color and (color == np.array([88, 64, 205])).all():
                real_touch(driver, *random.choice([(150, 625), (350, 625), (350, 700)]))
            else:
                real_touch(driver, 147, 809)
            time.sleep(1)

        elif game_state == "game_over":
            for attempt in range(10):
                screenshot = driver.get_screenshot_as_png()
                img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
                # 不再保存任何圖片，停用 debug 截圖
                is_over, wave_number, coins_number, tier_number, full_path, crop_path = detect_game_over_and_wave(img, save_debug=False)
                if is_over:
                    round_end = datetime.datetime.now()
                    if not wave_number and SAVE_UNKNOWN_WAVE_SCREENSHOT:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        pass  # 已停用保存未知 WAVE 圖片
                    save_wave_log(round_start, round_end, wave_number, coins_number, tier_number, full_path, crop_path)
                    real_touch(driver, 108, 588)
                    round_start = datetime.datetime.now()
                    time.sleep(3)
                    game_state = "playing"
                    break
                time.sleep(1)


if __name__ == "__main__":
    main()
    # update_wave_log_summary()
