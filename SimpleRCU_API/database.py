import asyncio
import aiosqlite
import time
import datetime
import logging
from typing import List, Dict, Any

# --- 配置 ---
DATABASE_URL = "test.db"
WRITE_GRACE_PERIOD = 0.1  # 秒 (寫入後的寬限期)
DB_TIMEOUT = 5.0 # 資料庫操作超時

# --- 日誌設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 全域變數 (應用程式生命週期管理) ---
db_connection: aiosqlite.Connection | None = None
write_lock = asyncio.Lock() # 用於同步寫入操作
last_write_time = 0.0       # 記錄上次寫入完成的時間戳

# --- 資料庫初始化 ---
async def init_db():
    """初始化資料庫連接並創建表（如果不存在）"""
    global db_connection, write_lock, last_write_time
    logger.info(f"Connecting to database: {DATABASE_URL}")
    db_connection = await aiosqlite.connect(DATABASE_URL, timeout=DB_TIMEOUT)
    db_connection.row_factory = aiosqlite.Row # 返回類似字典的行

    async with db_connection.cursor() as cursor:
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    await db_connection.commit()
    logger.info("Database initialized and table 'items' ensured.")
    # 初始化鎖和時間戳
    write_lock = asyncio.Lock()
    last_write_time = time.monotonic() # 使用 monotonic clock 避免系統時間更改問題

async def close_db():
    """關閉資料庫連接"""
    global db_connection
    if db_connection:
        await db_connection.close()
        logger.info("Database connection closed.")
        db_connection = None

# --- 資料庫操作 ---

async def add_item(name: str, description: str | None) -> Dict[str, Any]:
    """
    添加一個新項目到資料庫 (同步寫入)。
    使用 write_lock 確保同一時間只有一個寫入操作。
    """
    if not db_connection:
        raise ConnectionError("Database is not connected.")

    global last_write_time

    logger.info("Attempting to acquire write lock...")
    async with write_lock: # <--- 寫入鎖定 critical section
        logger.info("Write lock acquired. Performing write operation...")
        start_time = time.monotonic()
        try:
            async with db_connection.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO items (name, description) VALUES (?, ?)",
                    (name, description)
                )
                item_id = cursor.lastrowid # 獲取剛插入的 ID
                await db_connection.commit()

                # 獲取剛插入的完整記錄以返回
                await cursor.execute("SELECT * FROM items WHERE id = ?", (item_id,))
                new_item_row = await cursor.fetchone()

            # 更新上次寫入時間 *在鎖釋放之前*
            last_write_time = time.monotonic()
            duration = time.monotonic() - start_time
            logger.info(f"Write operation completed in {duration:.4f} seconds. Item ID: {item_id}")

            if new_item_row:
                # 將 aiosqlite.Row 轉換為字典
                return dict(new_item_row)
            else:
                 # 理論上不應發生，但作為防禦性程式設計
                logger.error(f"Failed to retrieve the newly inserted item with ID: {item_id}")
                raise RuntimeError("Failed to retrieve the newly inserted item after commit.")

        except aiosqlite.Error as e:
            logger.error(f"Database write error: {e}")
            # 考慮是否需要 rollback，儘管 commit() 失敗可能隱含了 rollback
            # await db_connection.rollback() # 取決於具體錯誤和需求
            raise # 重新拋出異常，讓 FastAPI 處理
        # 鎖在此處自動釋放 (async with 結束)
        logger.debug("Write lock released.")


async def get_items() -> List[Dict[str, Any]]:
    """
    從資料庫讀取所有項目 (並行讀取優化 + Grace Period)。
    讀取操作會檢查是否有寫入正在進行或剛結束 (在 Grace Period 內)，
    如果是，則會短暫等待。讀取本身不持有長時間鎖，允許多個讀取並行。
    """
    if not db_connection:
        raise ConnectionError("Database is not connected.")

    # --- Grace Period 和寫入鎖定檢查 ---
    while True:
        current_time = time.monotonic()
        is_writing = write_lock.locked()
        time_since_last_write = current_time - last_write_time

        if not is_writing and time_since_last_write >= WRITE_GRACE_PERIOD:
            # 沒有寫入正在進行，且已超過寬限期，可以開始讀取
            logger.debug("Read condition met (no active write, grace period passed).")
            break
        elif is_writing:
            logger.info("Read waiting: Write lock is active.")
            await asyncio.sleep(0.01) # 短暫 sleep，讓出控制權，避免忙等待
        else: # not is_writing and time_since_last_write < WRITE_GRACE_PERIOD
            wait_time = WRITE_GRACE_PERIOD - time_since_last_write
            logger.info(f"Read waiting: Inside grace period. Waiting for {wait_time:.4f} more seconds.")
            await asyncio.sleep(wait_time)
            # 等待後需要重新檢查條件，因為可能有新的寫入開始了
            continue # 回到 while 迴圈頂部重新檢查

    # --- 執行實際的並行讀取 ---
    logger.info("Performing read operation...")
    start_time = time.monotonic()
    try:
        async with db_connection.cursor() as cursor:
            await cursor.execute("SELECT id, name, description, created_at FROM items ORDER BY created_at DESC")
            rows = await cursor.fetchall()
        duration = time.monotonic() - start_time
        logger.info(f"Read operation completed in {duration:.4f} seconds. Fetched {len(rows)} items.")
        # 將 aiosqlite.Row 列表轉換為字典列表
        return [dict(row) for row in rows]
    except aiosqlite.Error as e:
        logger.error(f"Database read error: {e}")
        raise