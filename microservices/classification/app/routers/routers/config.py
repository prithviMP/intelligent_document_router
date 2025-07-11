from fastapi import APIRouter

router = APIRouter()

#  Actual implementations here
def update_priority_config(high: int, medium: int):
    # update logic
    print(f"Updated priority thresholds → High: {high}, Medium: {medium}")

def add_department(name: str, keywords: list[str]):
    # department config logic
    print(f"Added new department → {name}, with keywords: {keywords}")

@router.put("/config/priority/")
def update_priority(high: int, medium: int):
    update_priority_config(high, medium)
    return {"message": "Priority config updated"}

@router.post("/config/departments/")
def add_new_department(name: str, keywords: list[str]):
    add_department(name, keywords)
    return {"message": f"Department '{name}' added"}
