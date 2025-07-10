# Intelligent Document Classifier and Router - Architecture & Folder Structure Guide

This guide explains the folder structure, code organization, and extension points for the Intelligent Document Classifier and Router, based on the requirements in plan1.md.

---

## 📁 Root Directory Structure

```
intelligent_document_router/
├── .github/workflows/     # CI/CD pipeline definitions (GitHub Actions)
├── docs/                  # Architecture diagrams, technical docs, PRDs
├── libs/                  # Shared Python libraries (models, utils)
│   ├── database/          # SQLAlchemy models, DB connection logic
│   └── utils/             # Common utilities (logging, helpers)
├── microservices/         # All microservices (each in its own folder)
│   ├── api_gateway/       # FastAPI entry point, auth, routing
│   ├── classification/    # AI-powered document classification
│   ├── routing_engine/    # Smart document routing logic
│   ├── content_analysis/  # Entity extraction, content analysis
│   └── workflow_integration/ # Slack/Jira/email integrations
├── infrastructure/        # Infrastructure as code (DB, MQ, storage)
│   ├── db/                # PostgreSQL Dockerfile, init.sql
│   ├── rabbitmq/          # RabbitMQ config (if needed)
│   └── storage/           # MinIO/S3 config (if needed)
├── tests/                 # Unit and integration tests
│   ├── unit/              # Unit tests for each service
│   └── integration/       # Integration/E2E tests
├── docker-compose.yml     # Main compose file for all services
├── .env                   # Environment variables for all services
├── README.md              # Project overview and quickstart
├── todo.md                # Implementation progress tracker
└── SETUP_COMPLETE.md      # Setup summary and next steps
```

---

## 📦 Folder-by-Folder Explanation

### `.github/workflows/`
- **Purpose:** CI/CD pipeline definitions (e.g., build, test, deploy)
- **Add:** YAML files for GitHub Actions, e.g., `ci-cd.yml`

### `docs/`
- **Purpose:** Architecture diagrams, PRDs, technical documentation
- **Add:** Markdown docs, Mermaid diagrams, onboarding guides

### `libs/`
- **Purpose:** Shared code used by multiple microservices
- **database/`**: SQLAlchemy models, DB session/connection logic
- **utils/`**: Logging, error handling, helper functions
- **Add:** New shared models, utility functions, or adapters

### `microservices/`
- **Purpose:** All business logic, each service in its own folder
- **api_gateway/`**: FastAPI app, authentication, main API endpoints
  - `app/`: Main FastAPI app, routers, middleware, dependencies
    - `routers/`: Add new API endpoints (e.g., `/documents`, `/auth`)
    - `middleware/`: Custom middleware (logging, rate limiting)
    - `dependencies.py`: Dependency injection (auth, DB session)
    - `main.py`: FastAPI app entry point
  - `adapters/`: (Optional) HTTP clients for other services
  - `rules/`: (Optional) Business rule definitions
  - `requirements.txt`, `Dockerfile`: Service dependencies and build
- **classification/`**: Document type detection, spaCy/NLP logic
- **routing_engine/`**: Rule engine, workload balancing, assignment
- **content_analysis/`**: Entity extraction, key phrase, summary
- **workflow_integration/`**: Slack, Jira, email, webhooks
- **Add:** New microservices for additional features

### `infrastructure/`
- **Purpose:** Infrastructure as code for DB, MQ, storage
- **db/`**: PostgreSQL Dockerfile, schema init.sql
- **rabbitmq/`**: RabbitMQ config (if custom setup needed)
- **storage/`**: MinIO/S3 config (if custom setup needed)
- **Add:** Custom Dockerfiles, config scripts

### `tests/`
- **Purpose:** All automated tests
- **unit/`**: Unit tests for each microservice
- **integration/`**: Integration/E2E tests across services
- **Add:** New test modules as you add features

---

## 🛠️ Where to Add What Code

- **New API endpoint?**
  - Add a new file in `microservices/api_gateway/app/routers/` and register it in `main.py`.
- **New database model?**
  - Add to `libs/database/models.py` and run migrations/init.
- **New microservice?**
  - Create a new folder in `microservices/`, add `app/`, `Dockerfile`, `requirements.txt`.
- **Shared logic?**
  - Add to `libs/` (e.g., `libs/utils/` for helpers, `libs/database/` for DB logic).
- **Infrastructure change?**
  - Update `infrastructure/` and `docker-compose.yml`.
- **New test?**
  - Add to `tests/unit/` or `tests/integration/` as appropriate.

---

## 🚦 Onboarding Guide for New Developers

1. **Read `README.md` and `ARCHITECTURE_GUIDE.md` for project overview.**
2. **Clone the repo and copy `.env.example` to `.env`.**
3. **Start infrastructure:**
   ```bash
   docker-compose up db rabbitmq minio redis -d
   ```
4. **Run the API Gateway locally:**
   ```bash
   cd microservices/api_gateway
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
5. **Explore the codebase:**
   - API endpoints: `microservices/api_gateway/app/routers/`
   - DB models: `libs/database/models.py`
   - Shared logic: `libs/`
   - Add new features/tests as described above
6. **Run tests:**
   ```bash
   pytest
   ```
7. **Check progress in `todo.md` and update as you complete tasks.**

---

## 🧩 Extending the System
- **Add new document types:** Update classification logic in `classification/` and models in `libs/database/models.py`.
- **Add new routing rules:** Update `routing_engine/` and `libs/database/models.py`.
- **Integrate new tools (Slack, Jira, etc):** Add adapters in `workflow_integration/`.
- **Improve search/analytics:** Extend `content_analysis/` and add endpoints to API Gateway.

---

## 📚 Reference: plan1.md
- All folder and service responsibilities are mapped directly from the PRD and technical design in `plan1.md`.
- For detailed feature specs, see `plan1.md` and `README.md`.

---

**For any questions, see the docs/ folder or contact the project maintainers.** 

---

## 🧑‍💻 Example: Creating an API Endpoint in Each Microservice

Below are minimal examples for adding a simple `/ping` endpoint to each microservice. This helps new developers quickly understand where to add code and how to register routes.

### 1. API Gateway (`microservices/api_gateway`)
**File:** `microservices/api_gateway/app/routers/ping.py`
```python
from fastapi import APIRouter
router = APIRouter()

@router.get("/ping")
def ping():
    return {"message": "pong from API Gateway"}
```
**Register in** `app/main.py`:
```python
from app.routers import ping
app.include_router(ping.router, prefix="/ping", tags=["ping"])
```

### 2. Classification Service (`microservices/classification`)
**File:** `microservices/classification/app/main.py`
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong from Classification Service"}
```

### 3. Routing Engine (`microservices/routing_engine`)
**File:** `microservices/routing_engine/app/main.py`
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong from Routing Engine"}
```

### 4. Content Analysis (`microservices/content_analysis`)
**File:** `microservices/content_analysis/app/main.py`
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong from Content Analysis Service"}
```

### 5. Workflow Integration (`microservices/workflow_integration`)
**File:** `microservices/workflow_integration/app/main.py`
```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong from Workflow Integration Service"}
```

---

**How to test:**
- Start the relevant service (e.g., with `uvicorn app.main:app --reload` or via Docker Compose)
- Visit `http://localhost:<service-port>/ping` in your browser or use `curl`

**Tip:**
- For more complex APIs, create a `routers/` directory in each service and organize endpoints as in the API Gateway example.
- Always register new routers in your FastAPI `main.py` using `app.include_router(...)`.

--- 