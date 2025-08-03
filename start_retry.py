import re
import time
import cv2
import numpy as np
import pytesseract
import datetime
import io
import requests
import random
import csv
import os
from PIL import Image
from appium import webdriver
from appium.options.ios import XCUITestOptions

def setup_driver():
    options = XCUITestOptions()
    options.platform_name = "iOS"
    options.platform_version = "18.4.1"
    options.device_name = "iPhone"
    options.udid = "00008101-000405C63E20001E"
    options.bundle_id = "com.TechTreeGames.TheTower"
    options.use_new_wda = False
    options.use_prebuilt_wda = True
    options.wda_launch_timeout = 60000
    options.xcode_signing_id = "iPhone Developer"
    return webdriver.Remote("http://localhost:4723", options=options)

def get_screen_size(driver):
    screen_size = driver.get_window_size()
    return screen_size['width'], screen_size['height']

def convert_to_screen(x, y, real_width, real_height, screen_width, screen_height):
    screen_x = int(x / real_width * screen_width)
    screen_y = int(y / real_height * screen_height)
    return screen_x, screen_y

def real_touch(driver, x, y):
    driver.execute_script("mobile: tap", {"x": x, "y": y})

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
        cv2.imwrite(screenshot_path, img_copy)
        cv2.imwrite(cropped_path, coins_roi)

    return found_retry, wave_number, coins_number, tier_number, screenshot_path, cropped_path

def save_wave_log(round_num, start_time, end_time, wave, coins, tier, screenshot_path=None, cropped_path=None):
    file_exists = os.path.exists("wave_log.csv")
    with open("wave_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Start Time", "End Time", "Wave", "Coins", "Tier", "Screenshot", "Crop"])
        writer.writerow([
            round_num,
            start_time,
            end_time,
            wave or "",
            coins or "",
            tier or "",
            screenshot_path or "",
            cropped_path or ""
        ])
    update_wave_log_summary()


# 新增：統計每個 Tier 的平均 coins per minute 並寫入 wave_log_sum.csv
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
        valid_lines = [line for line in lines if line.count(",") == 7]  # 預期8欄位 => 7個逗號
        with open("wave_log_clean.csv", "w") as f:
            f.writelines(valid_lines)
        df = pd.read_csv("wave_log_clean.csv")

    df = df.dropna(subset=["Start Time", "End Time", "Coins", "Tier"])

    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
    df["End Time"] = pd.to_datetime(df["End Time"], errors="coerce")
    df["Coins"] = pd.to_numeric(df["Coins"], errors="coerce")
    df["Tier"] = pd.to_numeric(df["Tier"], errors="coerce")

    df = df.dropna(subset=["Start Time", "End Time", "Coins", "Tier"])
    df["Duration (min)"] = (df["End Time"] - df["Start Time"]).dt.total_seconds() / 60
    df = df[df["Duration (min)"] > 0]
    df["Coins Per Minute"] = df["Coins"] / df["Duration (min)"]

    summary = df.groupby("Tier")["Coins Per Minute"].mean().reset_index()
    summary = summary.rename(columns={"Coins Per Minute": "Average Coins Per Minute"})
    summary["Average Coins Per Minute"] = summary["Average Coins Per Minute"].round().astype(int)
    summary.to_csv("wave_log_sum.csv", index=False)

def detect_and_click_diamond(driver, img, template_path="diamond_f.png", threshold=0.8):
    def rotate_image(img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    def convert_to_screen(x, y, real_width, real_height, screen_width, screen_height):
        screen_x = int(x / real_width * screen_width)
        screen_y = int(y / real_height * screen_height)
        return screen_x, screen_y

    def read_number_in_region(img, x1, y1, x2, y2):
        region = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        return text

    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        print("❌ 找不到鑽石圖")
        return False

    screen_width, screen_height = get_screen_size(driver)
    real_height, real_width = img.shape[:2]

    for angle in range(0, 360, 30):
        rotated = rotate_image(template, angle)
        result = cv2.matchTemplate(img, rotated, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            h, w = rotated.shape[:2]
            cx_img = max_loc[0] + w // 2
            cy_img = max_loc[1] + h // 2
            cx_screen, cy_screen = convert_to_screen(cx_img, cy_img, real_width, real_height, screen_width, screen_height)

            # 💡 點擊前讀數字
            num_before = read_number_in_region(img, 100, 350, 250, 400)

            real_touch(driver, cx_screen, cy_screen)
            time.sleep(0.3)

            # 💡 點擊後更新畫面再讀一次
            screenshot_after = driver.get_screenshot_as_png()
            img_after = cv2.imdecode(np.frombuffer(screenshot_after, np.uint8), cv2.IMREAD_COLOR)
            num_after = read_number_in_region(img_after, 100, 350, 250, 400)

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"💎 {now_str} ▶ 點擊鑽石 🔢 數字變化：{num_before} ➡ {num_after}")
            return True
    return False

# def save_crop_region(img, x1, y1, x2, y2):
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     cropped = img[y1:y2, x1:x2]
#     filename = f"region_{x1}_{y1}_{x2}_{y2}_{timestamp}.jpg"
#     cv2.imwrite(filename, cropped)
#     print(f"🖼️ 儲存區域截圖：{filename}")

def read_number_in_region(img, x1, y1, x2, y2):
    region = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 7').strip()
    print(f"🔢 OCR 結果（{x1},{y1} ~ {x2},{y2}）：{text}")
    return text

def main():
    driver = setup_driver()
    screen_width, screen_height = get_screen_size(driver)
    game_state = "waiting"
    round_count = 0
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
                    round_count += 1
                    round_start = datetime.datetime.now()  # ✅ 保留 datetime 格式
                    time.sleep(2)
                    game_state = "playing"
                    break
            time.sleep(1)

        elif game_state == "playing":
            # 🔍 每回合檢查是否有鑽石圖案出現
            detect_and_click_diamond(driver, img)
            # 🖼️ 每回合儲存裁切畫面（這邊請替換成你要的區域）
            

            grid_x, grid_y, cols, rows = 2, 26, 4, 40
            is_single_color, color, _ = analyze_block(img, grid_x, grid_y, cols, rows)
            if is_single_color and (color == np.array([88, 64, 205])).all():
                real_touch(driver, *random.choice([(150, 600), (350, 700)]))
            else:
                real_touch(driver, 147, 809)
            time.sleep(2)
            screenshot = driver.get_screenshot_as_png()
            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            is_over, wave_number, coins_number, tier_number, full_path, crop_path = detect_game_over_and_wave(img, save_debug=False)

            if is_over:
                round_end = datetime.datetime.now()  # ← 保留 datetime 物件
                duration_sec = int((round_end - round_start).total_seconds())
                round_end_str = round_end.strftime("%Y-%m-%d %H:%M:%S")
                print(f"🏁 Game Over | Round {round_count} | Wave: {wave_number} | Coins: {coins_number} | Tier: {tier_number} | ⏱ {round_end_str} | ⌛ {duration_sec}s")  
                game_state = "game_over"

        elif game_state == "game_over":
            for attempt in range(10):
                screenshot = driver.get_screenshot_as_png()
                img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
                is_over, wave_number, coins_number, tier_number, full_path, crop_path = detect_game_over_and_wave(img, save_debug=True)
                if is_over:
                    round_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if not wave_number and SAVE_UNKNOWN_WAVE_SCREENSHOT:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"wave_unknown_{timestamp}.jpg", img)
                    save_wave_log(round_count, round_start, round_end, wave_number, coins_number, tier_number, full_path, crop_path)
                    real_touch(driver, 108, 588)
                    round_count += 1
                    round_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    time.sleep(3)
                    game_state = "playing"
                    break
                time.sleep(1)





if __name__ == "__main__":
    main()


    