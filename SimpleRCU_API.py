from fastapi import FastAPI, HTTPException, Depends, status
from typing import List
import asyncio
import logging

from models import ItemCreate, ItemRead # 從 models.py 導入 Pydantic 模型
from database import ( # 從 database.py 導入
    init_db,
    close_db,
    add_item as db_add_item,
    get_items as db_get_items,
    db_connection # 可以用來檢查連接狀態
)

# --- 日誌設定 (可以與 database.py 共享或獨立) ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- FastAPI 應用實例 ---
app = FastAPI(
    title="Async DB Operations Demo",
    description="Demonstrates synchronized writes and parallel reads with grace period.",
    version="1.0.0",
)

# --- 生命周期事件處理 ---
@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化資料庫"""
    logger.info("Application startup: Initializing database...")
    try:
        await init_db()
    except Exception as e:
        logger.critical(f"Failed to initialize database during startup: {e}", exc_info=True)
        # 在生產環境中，這裡可能需要更健壯的處理，例如重試或退出應用
        raise RuntimeError(f"Database initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時關閉資料庫連接"""
    logger.info("Application shutdown: Closing database connection...")
    await close_db()
    logger.info("Application shutdown complete.")

# --- 依賴項 (可選，用於檢查連接) ---
async def get_db_connection_status():
    """檢查資料庫是否已連接"""
    if db_connection is None:
        logger.error("API called but database is not connected.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available.",
        )
    # 可以選擇性地在這裡執行一個快速的 ping 檢查
    # try:
    #     await db_connection.execute("SELECT 1")
    # except Exception as e:
    #     logger.error(f"Database connection check failed: {e}")
    #     raise HTTPException(
    #         status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    #         detail="Database connection lost.",
    #     )
    # logger.debug("Database connection confirmed.")


# --- API 路由 ---

@app.post("/items/", response_model=ItemRead, status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate, _=Depends(get_db_connection_status)):
    """
    創建一個新項目。此操作將會同步執行。
    """
    logger.info(f"Received request to create item: {item.name}")
    try:
        # 調用 database.py 中實現了同步寫入的函數
        created_item_dict = await db_add_item(name=item.name, description=item.description)
        # 使用 Pydantic 模型進行驗證和轉換
        return ItemRead.parse_obj(created_item_dict) # Pydantic v1
        # return ItemRead.model_validate(created_item_dict) # Pydantic v2
    except ConnectionError as e:
         logger.error(f"Connection error during item creation: {e}")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception(f"Error creating item '{item.name}': {e}") # 使用 exception 記錄堆疊跟踪
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating item.")

@app.get("/items/", response_model=List[ItemRead])
async def read_items(_=Depends(get_db_connection_status)):
    """
    讀取所有項目。此操作經過並行優化，並包含讀取寬限期。
    """
    logger.info("Received request to read items.")
    try:
        # 調用 database.py 中實現了並行讀取和 Grace Period 的函數
        items_list_dict = await db_get_items()
        # 使用 Pydantic 模型進行驗證和轉換
        return [ItemRead.parse_obj(item_dict) for item_dict in items_list_dict] # Pydantic v1
        # return [ItemRead.model_validate(item_dict) for item_dict in items_list_dict] # Pydantic v2
    except ConnectionError as e:
         logger.error(f"Connection error during item reading: {e}")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception(f"Error reading items: {e}") # 使用 exception 記錄堆疊跟踪
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while reading items.")

# --- 運行應用 (如果直接執行此文件) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # 使用 reload=True 方便開發調試
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
