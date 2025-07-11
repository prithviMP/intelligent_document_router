# Microservices Developer Guide

This guide provides a step-by-step, beginner-friendly walkthrough for developing, extending, and understanding each microservice in the Intelligent Document Classifier and Router system. It covers folder structure, SQLAlchemy models, Pydantic models, API endpoints, service communication, and best practices for scalable development.

---

## Table of Contents
1. [Project Structure Overview](#project-structure-overview)
2. [Microservice Folder Structure](#microservice-folder-structure)
3. [Creating SQLAlchemy Models and Tables](#creating-sqlalchemy-models-and-tables)
4. [Writing Pydantic Models](#writing-pydantic-models)
5. [API Routers and Endpoints](#api-routers-and-endpoints)
6. [Service-to-Service Communication](#service-to-service-communication)
7. [Shared Libraries and Utilities](#shared-libraries-and-utilities)
8. [Adding New Features or Endpoints](#adding-new-features-or-endpoints)
9. [Best Practices](#best-practices)

---

## 1. Project Structure Overview

```
intelligent_document_router/
├── microservices/
│   ├── api_gateway/
│   ├── classification/
│   ├── routing_engine/
│   ├── content_analysis/
│   └── workflow_integration/
├── libs/
│   ├── database/
│   └── utils/
├── infrastructure/
│   ├── db/
│   ├── rabbitmq/
│   └── storage/
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
└── docker-compose.yml
```

- **microservices/**: Each service is isolated and self-contained.
- **libs/**: Shared code (e.g., database models, utilities).
- **infrastructure/**: Database, message broker, storage setup.
- **tests/**: Unit and integration tests.
- **docs/**: Documentation and guides.

---

## 2. Microservice Folder Structure

Each microservice follows a similar structure for consistency and maintainability:

```
microservices/<service_name>/
├── app/
│   ├── main.py           # FastAPI entrypoint
│   ├── models.py         # SQLAlchemy models (if service-specific)
│   ├── routers/          # API routers (endpoints)
│   ├── middleware/       # Custom middleware (optional)
│   └── dependencies.py   # Dependency functions (optional)
├── adapters/             # External service adapters (optional)
├── rules/                # Business rules (optional)
├── requirements.txt      # Python dependencies
└── Dockerfile            # Container build file
```

**Example: api_gateway**
```
microservices/api_gateway/
├── app/
│   ├── main.py
│   ├── dependencies.py
│   ├── routers/
│   │   ├── documents.py
│   │   ├── auth.py
│   │   └── health.py
│   ├── middleware/
│   │   ├── rate_limit.py
│   │   └── logging.py
│   └── ...
├── adapters/
├── rules/
├── requirements.txt
└── Dockerfile
```

---

## 3. Creating SQLAlchemy Models and Tables

- **Shared models** (used by multiple services) go in `libs/database/models.py`.
- **Service-specific models** go in `microservices/<service>/app/models.py`.

**Example: Shared Model (libs/database/models.py)**
```python
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_name = Column(String(255), nullable=False)
    # ... other fields ...
```

**Example: Service-Specific Model (classification/app/models.py)**
```python
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class ClassificationJob(Base):
    __tablename__ = 'classification_jobs'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    doc_type = Column(String(50))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**How to create tables:**
- Use Alembic for migrations (recommended), or
- Use `Base.metadata.create_all(engine)` in a script for quick setup.

---

## 4. Writing Pydantic Models

Pydantic models define the request and response schemas for your API endpoints.

**Example:**
```python
from pydantic import BaseModel
from typing import Optional

class ClassifyRequest(BaseModel):
    document_id: str
    content: Optional[str] = None

class ClassifyResponse(BaseModel):
    doc_type: str
    confidence: float
```

- Place Pydantic models in the same file as the endpoint or in a separate `schemas.py` file.

---

## 5. API Routers and Endpoints

- Place routers in `app/routers/`.
- Each router handles a logical group of endpoints (e.g., documents, auth).

**Example: documents.py**
```python
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from libs.database.models import Document

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file, create DB record, call classification service
    ...
```

- Register routers in `main.py`:
```python
from app.routers import documents
app.include_router(documents.router, prefix="/documents", tags=["documents"])
```

---

## 6. Service-to-Service Communication

- Use HTTP (via FastAPI + httpx) or message queues (RabbitMQ) for communication.
- Example: API Gateway calls Classification Service after document upload.

**Example: Calling another service**
```python
import httpx

response = await httpx.post("http://classification:8001/classify", json={"document_id": doc_id})
if response.status_code == 200:
    result = response.json()
    # Use result
```

- Use environment variables for service URLs.
- For async communication, use RabbitMQ (see infrastructure setup).

---

## 7. Shared Libraries and Utilities

- Place shared models in `libs/database/models.py`.
- Place shared DB connection logic in `libs/database/connection.py`.
- Place utility functions in `libs/utils/`.
- Import shared code in microservices using:
```python
from libs.database.models import Document
```

---

## 8. Adding New Features or Endpoints

**Step-by-step:**
1. **Create a new router:**
   - Add a file in `app/routers/` (e.g., `notifications.py`).
2. **Define endpoints:**
   - Use FastAPI decorators (`@router.get`, `@router.post`, etc.).
3. **Add Pydantic models** for request/response.
4. **Register the router** in `main.py`.
5. **(If needed) Add new SQLAlchemy models** in `models.py`.
6. **(If needed) Add new tables** via Alembic or `Base.metadata.create_all()`.
7. **Test your endpoint** (add tests in `tests/unit/` or `tests/integration/`).

---

## 9. Best Practices
- Keep each microservice self-contained.
- Use shared code from `libs/` to avoid duplication.
- Use environment variables for configuration.
- Write unit and integration tests for all endpoints.
- Document your endpoints with OpenAPI (FastAPI auto-generates docs at `/docs`).
- Use Docker for local development and deployment.
- Use Alembic for database migrations.
- Follow the folder structure for easy onboarding and scalability.

---

## Example: Adding a New Endpoint to Classification Service

1. **Create a new router file:** `microservices/classification/app/routers/jobs.py`
2. **Define endpoints:**
```python
from fastapi import APIRouter
from .models import ClassificationJob

router = APIRouter()

@router.get("/jobs")
def list_jobs():
    # Return list of jobs
    ...
```
3. **Register the router in `main.py`:**
```python
from app.routers import jobs
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
```
4. **Add tests in `tests/unit/classification/test_jobs.py`**

---

## Example: Creating a New SQLAlchemy Model

1. **Add to `models.py`:**
```python
class NewModel(Base):
    __tablename__ = 'new_table'
    id = Column(Integer, primary_key=True)
    ...
```
2. **Create migration or run `Base.metadata.create_all()`**

---

## Example: Folder Structure for a New Microservice

```
microservices/new_service/
├── app/
│   ├── main.py
│   ├── models.py
│   ├── routers/
│   └── middleware/
├── adapters/
├── rules/
├── requirements.txt
└── Dockerfile
```

---

For more details, see the code in each microservice and the shared `libs/` directory. Follow this guide for consistent, scalable, and maintainable development! 

---

## 10. Step-by-Step: Creating a Table, SQLAlchemy Model, Pydantic Model, and API Route

This section walks you through the full process of adding a new resource to a microservice, from database table to API endpoint.

### Step 1: Create a SQLAlchemy Model (Database Table)
- **Where:** `microservices/<service>/app/models.py` (or `libs/database/models.py` for shared models)
- **Purpose:** Defines the structure of your database table.

**Example:**
```python
# microservices/classification/app/models.py
from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ExampleItem(Base):
    __tablename__ = 'example_items'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
```

### Step 2: Create the Table in the Database
- **Recommended:** Use Alembic for migrations (see Alembic docs)
- **Quick test:** Add this to a script and run it:
```python
from sqlalchemy import create_engine
from app.models import Base
engine = create_engine('postgresql://user:pass@localhost/dbname')
Base.metadata.create_all(engine)
```

### Step 3: Create Pydantic Models (Request/Response Schemas)
- **Where:** Same file as your route, or a new `schemas.py` in the same folder.
- **Purpose:** Defines the data shape for API requests and responses. Pydantic models are NOT used for the database, only for API validation and docs.

**Example:**
```python
# microservices/classification/app/schemas.py
from pydantic import BaseModel
from typing import Optional

class ExampleItemCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ExampleItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
```

### Step 4: Create a FastAPI Route Using the Models
- **Where:** `microservices/<service>/app/routers/example_items.py`
- **Purpose:** Implements the API endpoint using the SQLAlchemy and Pydantic models.

**Example:**
```python
# microservices/classification/app/routers/example_items.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models import ExampleItem
from app.schemas import ExampleItemCreate, ExampleItemResponse

router = APIRouter()

@router.post("/example_items", response_model=ExampleItemResponse)
def create_example_item(item: ExampleItemCreate, db: Session = Depends(get_db)):
    db_item = ExampleItem(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

### Step 5: Register the Router in main.py
- **Where:** `microservices/<service>/app/main.py`

**Example:**
```python
from app.routers import example_items
app.include_router(example_items.router, prefix="/example_items", tags=["example_items"])
```

---

### Summary Table
| Step | What | File/Folder | Purpose |
|------|------|-------------|---------|
| 1 | SQLAlchemy Model | `app/models.py` | Defines DB table |
| 2 | Create Table | (script or Alembic) | Makes table in DB |
| 3 | Pydantic Model | `app/schemas.py` | API request/response |
| 4 | FastAPI Route | `app/routers/` | API endpoint |
| 5 | Register Router | `app/main.py` | Expose endpoint |

---

**Tip:**
- SQLAlchemy models = database tables (persistent storage)
- Pydantic models = API schemas (validation, docs, serialization)
- Keep them in separate files for clarity in larger projects 