from pydantic import BaseModel

class Agent1Output(BaseModel):
    name: str
    mobile_number: int 
    salary_in_inr: float
