from datetime import date, timedelta

PRIORITY_CONFIG = {"high": 3, "medium": 8}

def calculate_priority(doc_date: date) -> str:
    today = date.today()
    delta = (doc_date - today).days
    if delta <= PRIORITY_CONFIG["high"]:
        return "high"
    elif delta <= PRIORITY_CONFIG["medium"]:
        return "medium"
    else:
        return "low"
