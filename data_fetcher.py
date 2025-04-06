import requests
import json
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import math
import logging

# --- 配置區域 ---
API_ENDPOINT = "YOUR_API_ENDPOINT_HERE" # <--- 請替換成你的 API 端點
MAX_DATA_NAMES_PER_CALL = 80
TIME_INTERVAL_HOURS = 1
OUTPUT_DIR = "./Data/"
MAX_WORKERS = 10 # 平行處理的執行緒數量 (可依據你的 CPU 和網路狀況調整)
REQUEST_TIMEOUT = 60 # API 請求超時時間 (秒)

# 設定日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 模擬 API ---
# 為了測試，我們先模擬一個 API 回應。實際使用時請移除或註解掉這部分。
def mock_api_call(payload):
    """模擬 API 呼叫，返回符合格式的假資料"""
    try:
        data = json.loads(payload)
        data_names = data.get("data_name", [])
        start_time_str = data.get("start_time")
        end_time_str = data.get("end_time")

        if not all([data_names, start_time_str, end_time_str]):
            return {"error": "Missing parameters"}, 400

        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)

        results = []
        current_time = start_time
        while current_time < end_time: # API 通常包含 start 但不含 end
            for name in data_names:
                # 模擬偶爾的資料遺漏 (例如，每 10 筆漏掉 1 筆)
                if np.random.rand() > 0.1:
                    results.append({
                        "data_name": name,
                        "data_value": np.random.rand() * 100,
                        "date_time": current_time.strftime('%Y-%m-%d %H:%M:%S')
                    })
            current_time += timedelta(minutes=1)

        # 模擬 API 延遲
        time.sleep(np.random.uniform(0.5, 1.5))
        return results, 200

    except Exception as e:
        return {"error": str(e)}, 500

# --- 核心功能 ---

def fetch_data_chunk(data_names_chunk, start_chunk_time, end_chunk_time):
    """
    為一個特定的 data_name 列表和時間區間呼叫 API。

    Args:
        data_names_chunk (list): 要查詢的 data_name 列表 (最多 MAX_DATA_NAMES_PER_CALL 個)。
        start_chunk_time (datetime): 這個區塊的開始時間。
        end_chunk_time (datetime): 這個區塊的結束時間。

    Returns:
        list: 從 API 返回的資料列表 (list of dictionaries)，如果失敗則返回 None。
    """
    payload = {
        "data_name": data_names_chunk,
        "start_time": start_chunk_time.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_chunk_time.strftime('%Y-%m-%d %H:%M:%S')
    }
    payload_json = json.dumps(payload)
    # logging.debug(f"Fetching: {len(data_names_chunk)} names from {start_chunk_time} to {end_chunk_time}")

    try:
        # --- 實際 API 呼叫 (替換掉 mock) ---
        # response = requests.post(API_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        # response.raise_for_status() # 如果狀態碼不是 2xx，則拋出異常
        # return response.json()
        # --- 使用模擬 API ---
        response_data, status_code = mock_api_call(payload_json)
        if status_code != 200:
            logging.error(f"API Error ({status_code}) for {start_chunk_time}-{end_chunk_time}, names: {data_names_chunk[:5]}... - Response: {response_data}")
            return None
        return response_data
        # --- 結束模擬 ---

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {start_chunk_time}-{end_chunk_time}, names: {data_names_chunk[:5]}... - Error: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response for {start_chunk_time}-{end_chunk_time}, names: {data_names_chunk[:5]}... - Error: {e}")
        # logging.error(f"Raw response text: {response.text[:500]}") # Log raw response if needed
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during fetch for {start_chunk_time}-{end_chunk_time}, names: {data_names_chunk[:5]}... - Error: {e}")
        return None


def process_and_store_data(all_fetched_data, all_data_names, overall_start_time, overall_end_time):
    """
    處理收集到的數據，填補缺失值，並以高效的 .npy 格式儲存。

    Args:
        all_fetched_data (list): 包含所有 API 回應的列表 (每個元素是 list of dict)。
        all_data_names (list): 所有請求的 data_name 列表。
        overall_start_time (datetime): 整個請求的總開始時間。
        overall_end_time (datetime): 整個請求的總結束時間。
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    # 1. 合併並轉換為 DataFrame 以便處理
    logging.info("Aggregating fetched data...")
    flat_data = [item for sublist in all_fetched_data if sublist is not None for item in sublist]
    if not flat_data:
        logging.warning("No data was successfully fetched.")
        return

    df = pd.DataFrame(flat_data)
    if df.empty:
        logging.warning("Fetched data resulted in an empty DataFrame.")
        return

    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.set_index('date_time')
    df['data_value'] = pd.to_numeric(df['data_value'], errors='coerce') # 轉換為數字，無法轉換的變 NaN

    # 2. 處理每個 data_name
    logging.info("Processing and saving data for each data_name...")
    # 建立完整的時間索引（每分鐘）
    full_time_index = pd.date_range(start=overall_start_time, end=overall_end_time - timedelta(minutes=1), freq='T') # freq='T' means minute frequency

    for data_name in tqdm(all_data_names, desc="Processing & Saving"):
        df_single_name = df[df['data_name'] == data_name][['data_value']].copy()

        # 去除重複的時間戳 (保留第一個)
        df_single_name = df_single_name[~df_single_name.index.duplicated(keep='first')]

        # 重新索引以包含所有時間點，缺失處填 NaN
        df_reindexed = df_single_name.reindex(full_time_index)

        # 填補缺失值：使用下一個有效觀測值向前填充 (backward fill / bfill)
        df_filled = df_reindexed.bfill()

        # 可選：如果第一個值就是 NaN，bfill 後仍然是 NaN，可以選擇用 ffill 補全開頭
        # df_filled = df_filled.ffill() # 如果需要的話

        # 移除仍然是 NaN 的行 (如果 bfill 和 ffill 都無法填補，表示該 data_name 完全沒有數據)
        df_filled = df_filled.dropna()

        if df_filled.empty:
            logging.warning(f"No valid data points found for {data_name} after processing. Skipping save.")
            continue

        # 3. 轉換為 NumPy Structured Array
        # 將 datetime index 轉換為 Unix timestamp (整數，秒或納秒) 以便存儲
        # 使用 nanoseconds (int64) 較為精確且是 pandas 內部表示
        timestamps_ns = df_filled.index.values.astype(np.int64)
        values = df_filled['data_value'].values.astype(np.float64) # 確保為 float64

        # 創建 structured array
        structured_array = np.empty(len(df_filled), dtype=[('timestamp_ns', 'i8'), ('value', 'f8')])
        structured_array['timestamp_ns'] = timestamps_ns
        structured_array['value'] = values

        # 4. 儲存為 .npy 文件
        output_path = os.path.join(OUTPUT_DIR, f"{data_name}.npy")
        try:
            np.save(output_path, structured_array)
            # logging.debug(f"Successfully saved data for {data_name} to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save data for {data_name} to {output_path}. Error: {e}")

    logging.info("Data processing and saving complete.")


def parallel_fetch_and_store(all_data_names, overall_start_time_str, overall_end_time_str):
    """
    主函數：分塊、平行獲取、處理並儲存數據。

    Args:
        all_data_names (list): 所有需要獲取的 data_name。
        overall_start_time_str (str): 總體開始時間字串 (e.g., "2023-10-26 00:00:00")。
        overall_end_time_str (str): 總體結束時間字串 (e.g., "2023-10-27 00:00:00")。
    """
    try:
        overall_start_time = pd.to_datetime(overall_start_time_str)
        overall_end_time = pd.to_datetime(overall_end_time_str)
    except ValueError as e:
        logging.error(f"Invalid date format: {e}. Please use YYYY-MM-DD HH:MM:SS.")
        return

    logging.info(f"Starting data fetch for {len(all_data_names)} data names from {overall_start_time} to {overall_end_time}")

    # 1. 創建所有需要的 API 呼叫任務
    tasks_to_submit = []
    current_time = overall_start_time
    while current_time < overall_end_time:
        # 計算每個時間區塊的結束時間
        end_chunk_time = current_time + timedelta(hours=TIME_INTERVAL_HOURS)
        # 確保不超過總結束時間
        end_chunk_time = min(end_chunk_time, overall_end_time)

        # 將 data_names 分塊
        for i in range(0, len(all_data_names), MAX_DATA_NAMES_PER_CALL):
            data_names_chunk = all_data_names[i : i + MAX_DATA_NAMES_PER_CALL]
            # 添加任務：(data_name 列表, 區塊開始時間, 區塊結束時間)
            tasks_to_submit.append((data_names_chunk, current_time, end_chunk_time))

        # 移動到下一個時間區塊的開始
        current_time = end_chunk_time
        # 如果剛好到達結束時間，則跳出迴圈
        if current_time >= overall_end_time:
            break


    all_fetched_data = []
    total_tasks = len(tasks_to_submit)
    logging.info(f"Total API calls to make: {total_tasks}")

    # 2. 使用 ThreadPoolExecutor 平行執行任務
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 submit 和 as_completed 可以更好地處理結果和錯誤
        future_to_task = {executor.submit(fetch_data_chunk, *task): task for task in tasks_to_submit}

        # 使用 tqdm 顯示進度條
        for future in tqdm(as_completed(future_to_task), total=total_tasks, desc="Fetching Data"):
            task_info = future_to_task[future] # 獲取原始任務信息用於日誌記錄
            try:
                result = future.result() # 獲取任務結果
                if result is not None:
                    all_fetched_data.append(result)
                else:
                    # fetch_data_chunk 內部已經記錄了詳細錯誤
                    logging.warning(f"Task failed (returned None): {len(task_info[0])} names, {task_info[1]} to {task_info[2]}")
            except Exception as exc:
                logging.error(f"Task generated an exception: {len(task_info[0])} names, {task_info[1]} to {task_info[2]}. Error: {exc}", exc_info=True)

    logging.info(f"Finished fetching data. Got results from {len(all_fetched_data)} successful API calls.")

    # 3. 處理和儲存數據
    process_and_store_data(all_fetched_data, all_data_names, overall_start_time, overall_end_time)

# --- 讀取儲存的數據 ---
def load_data(data_name):
    """
    從 .npy 文件讀取指定 data_name 的數據，並轉換回 pandas DataFrame。

    Args:
        data_name (str): 要讀取的 data_name。

    Returns:
        pandas.DataFrame: 包含 'date_time' (index) 和 'value' 的 DataFrame，如果文件不存在或出錯則返回 None。
    """
    filepath = os.path.join(OUTPUT_DIR, f"{data_name}.npy")
    if not os.path.exists(filepath):
        logging.error(f"Data file not found: {filepath}")
        return None
    try:
        structured_array = np.load(filepath)
        if structured_array.size == 0:
             logging.warning(f"Loaded empty array from {filepath}")
             return pd.DataFrame(columns=['value']) # 返回空的 DataFrame 但有欄位

        # 將 Unix timestamp (nanoseconds) 轉換回 datetime 對象
        timestamps_ns = structured_array['timestamp_ns']
        datetimes = pd.to_datetime(timestamps_ns, unit='ns')

        # 創建 DataFrame
        df = pd.DataFrame({'value': structured_array['value']}, index=datetimes)
        df.index.name = 'date_time'
        logging.info(f"Successfully loaded data for {data_name} from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Failed to load or process data from {filepath}. Error: {e}")
        return None

# --- 範例使用 ---
if __name__ == "__main__":
    # 假設你有 200 個 data_name
    example_data_names = [f"Sensor_{i:03d}" for i in range(1, 201)]

    # 設定要抓取的總時間範圍
    start_datetime_str = "2023-10-26 08:00:00"
    end_datetime_str = "2023-10-26 12:00:00" # 抓取 4 小時的數據

    # 執行主要流程
    parallel_fetch_and_store(example_data_names, start_datetime_str, end_datetime_str)

    # --- 驗證讀取 ---
    print("\n--- Verifying saved data ---")
    example_load_name = "Sensor_005"
    loaded_df = load_data(example_load_name)

    if loaded_df is not None:
        print(f"\nLoaded data for {example_load_name}:")
        if not loaded_df.empty:
            print(loaded_df.head())
            print("...")
            print(loaded_df.tail())
            print(f"Total data points loaded: {len(loaded_df)}")
            # 檢查是否有 NaN 值 (理論上不應該有，除非數據完全缺失)
            print(f"Any NaN values after loading? {loaded_df['value'].isnull().any()}")
        else:
            print("Loaded DataFrame is empty.")

    example_load_name_missing = "Sensor_XXX" # 嘗試讀取不存在的數據
    load_data(example_load_name_missing)

    # 檢查一個可能完全沒有數據的例子 (如果 mock API 模擬了這種情況)
    # 假設 Sensor_199 在模擬中可能沒有數據
    # loaded_empty_df = load_data("Sensor_199")
    # if loaded_empty_df is not None:
    #     print(f"\nLoaded data for Sensor_199 (potentially empty): {loaded_empty_df.empty}")
