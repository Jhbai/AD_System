from pydantic import BaseModel
import datetime

class ItemBase(BaseModel):
    name: str
    description: str | None = None

class ItemCreate(ItemBase):
    pass # 寫入時使用的模型

class ItemRead(ItemBase):
    id: int
    created_at: datetime.datetime

    class Config:
        orm_mode = True # Pydantic v1 style, or from_attributes = True for v2