from pydantic import BaseModel

class HouseRequest(BaseModel):
    size: float