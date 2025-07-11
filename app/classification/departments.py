
DEPARTMENTS = {
    1: {
        "name": "hr",
        "keywords": ["employee", "recruitment", "salary", "leave"]
    },
    2: {
        "name": "it", 
        "keywords": ["server", "network", "software", "hardware"]
    },
    3: {
        "name": "legal",
        "keywords": ["contract", "agreement", "law", "compliance"]
    },
    4: {
        "name": "finance",
        "keywords": ["invoice", "budget", "payment", "tax", "receipt", "expense", "cost"]
    },
    5: {
        "name": "marketing",
        "keywords": ["campaign", "brand", "social media", "promotion"]
    },
    6: {
        "name": "sales",
        "keywords": ["client", "deal", "lead", "target"]
    },
    7: {
        "name": "engineering",
        "keywords": ["design", "specs", "prototype", "build"]
    },
    8: {
        "name": "support",
        "keywords": ["ticket", "issue", "help", "response"]
    },
    9: {
        "name": "general",
        "keywords": ["document", "file", "general", "misc", "other"]
    }
}

# Legacy compatibility
DEPARTMENT_KEYWORDS = {dept["name"]: dept["keywords"] for dept in DEPARTMENTS.values()}

HIGH_PRIORITY_KEYWORDS = ["by EOD", "by eod", "by today", "by tomorrow", "urgent", "asap", "emergency"]

def get_department_by_id(dept_id: int):
    return DEPARTMENTS.get(dept_id)

def get_department_id_by_name(name: str):
    for dept_id, dept_info in DEPARTMENTS.items():
        if dept_info["name"].lower() == name.lower():
            return dept_id
    return None
