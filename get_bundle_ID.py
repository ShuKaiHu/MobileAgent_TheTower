from appium import webdriver
from appium.options.ios import XCUITestOptions

# 建立 options
options = XCUITestOptions()
options.platform_name = "iOS"
options.platform_version = "18.0"    # 換成你的 iOS 版本
options.device_name = "Shu-Kai Hu 的 iPhone"  # 換成你的 iPhone名字
options.udid = "00008101-000405C63E20001E"    # 換成你的 iPhone UDID
options.use_new_wda = False
options.use_prebuilt_wda = True
options.xcode_signing_id = "iPhone Developer"

# 不指定 bundle_id！保持空白，代表連接現有畫面
driver = webdriver.Remote("http://localhost:4723", options=options)

# 抓目前活躍中的 App 資訊
app_info = driver.execute_script("mobile: activeAppInfo")

# 顯示結果
print("目前活躍中的 App：")
print(f"Bundle ID: {app_info.get('bundleId')}")
print(f"App 名字: {app_info.get('name')}")

# 結束連線
driver.quit()