# ✈️ Wanderlust Travels RAG Chatbot

AI-powered travel assistant using Azure OpenAI, Azure AI Search, and LLMOps best practices.

## 🌟 Features

*   **RAG-Powered Responses**: Retrieves information from travel policies, routes, and FAQs
*   **Azure AI Integration**: Uses Azure OpenAI chat/embedding deployments and Azure AI Search
*   **Ragas Evaluation**: Evaluates faithfulness, answer relevancy, context precision, and context recall
*   **Governance & Guardrails**:
    *   Safety Validator: Detects prompt injection and unsafe content
    *   Compliance Checker: Detects and blocks PII (GDPR compliance)
*   **Observability**: Optional MLflow tracking
*   **Production-Ready**: Docker containerization for easy deployment

---

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **Azure Subscription** with:
    *   Azure OpenAI chat and embedding deployments
    *   Azure AI Search (Free tier works)
    *   Azure Content Safety (Optional)
*   **Azure OpenAI chat and embedding deployments** (also used by Ragas)

---

## 🚀 Setup & Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd wanderlust-travel-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Azure services
Run `cp .env.sample .env`, then set your Azure OpenAI and Azure AI Search resource
credentials and deployment/index settings. `AZURE_OPENAI_DEPLOYMENT_NAME` must name a
chat deployment, and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` must name an embeddings
deployment. The application validates required settings when the search engine starts.
Azure Content Safety and MLflow are optional; leave their settings blank to disable them.
Never commit `.env`.

### 5. Prepare Data
Place source PDFs directly in `data/` and new or updated PDFs/CSVs directly in
`document_updates/` (no subfolders):

```
data/
├── ai-schedule-change-policy-updated.pdf
├── air-india-coc.pdf
├── *.pdf
└── golden_dataset.json

document_updates/
├── *.pdf
└── *.csv
```

The ingestion pipeline indexes both directories. Re-running ingestion upserts stable
source-page chunks; it does not automatically remove old index records when a source
file is deleted or becomes shorter, so perform an index rebuild for deletions.
---

## 🏃‍♂️ Usage

### 1. Start MLflow (Optional - for tracking)
```bash
./start_mlflow.sh
```
*   View dashboard at `http://localhost:5000`

**Troubleshooting MLflow:**
```bash
# If MLflow doesn't start, use restart script
./restart_mlflow.sh

# Or manually:
pkill -f "mlflow server"
mlflow server --host 127.0.0.1 --port 5000
```

When using Azure ML tracking from Codespaces, an
`InteractiveBrowserCredential` state-mismatch error means Azure authentication
did not complete; it is not an ingestion failure. If tracking is not needed,
leave `MLFLOW_TRACKING_URI` blank in `.env`. To keep Azure ML tracking enabled,
authenticate non-interactively (for example, install Azure CLI and run
`az login --use-device-code`) or configure service-principal credentials.
Ingestion continues without MLflow when tracking authentication fails.

### 2. Ingest Documents
Populate Azure AI Search index with your travel knowledge base:
```bash
python src/ingestion.py

or
python -m src.ingestion
```

**Output:** the ingestion command exits nonzero and logs the failed batches if indexing
or its retrieval verification fails. It prints concise stage messages, an indexing
progress bar, and a final result summary; verbose Azure SDK HTTP traces are suppressed.
PDF text extraction happens before indexing. A `KeyboardInterrupt` inside
`pypdf` indicates the process was stopped during loading; `INGESTION_LIMIT`
limits indexed chunks only and does not skip PDF loading.

### 3. Run the Application
Launch the Streamlit interface:
```bash
streamlit run src/app.py
```
*   Access at `http://localhost:8501`

### 4. Test Queries
Try these example queries:
- "What are the baggage rules for international flights?"
- "Do I need a visa to travel from India to UK?"
- "What is the cancellation policy for Air India?"
- "What is the refund policy if my flight is delayed?"
- "What countries require aircraft disinsection?"
---


## 📊 Evaluation (Ragas)

### Run Evaluation
Evaluate chatbot performance using Ragas metrics:
```bash
python src/evaluate.py
```

Ragas runs against the Azure OpenAI chat and embedding deployments and writes
`reports/evaluation_summary.json` plus per-question scores to
`reports/evaluation_detailed.csv`. The command exits `0` only when all four metrics
meet their 0.70 minimum thresholds; otherwise it exits nonzero.

### Evaluation Metrics Explained

The system uses **Ragas** to evaluate RAG performance:

1. **Faithfulness (0-1)**: Measures if the answer is grounded in retrieved context
   - Checks for hallucinations or unsupported claims
   - **Threshold**: ≥ 0.70

2. **Answer Relevancy (0-1)**: Measures if the answer addresses the question
   - Evaluates how well the response answers what was asked
   - **Threshold**: ≥ 0.70

3. **Context Precision (0-1)**: Measures quality of retrieved documents
   - Evaluates if the top-ranked documents are most relevant
   - Higher score = better ranking

4. **Context Recall (0-1)**: Measures if all relevant info was retrieved
   - Checks if retrieved context contains all information needed
   - Higher score = more complete retrieval

All four metric thresholds are applied by the evaluation pass/fail check.

### Golden Dataset Structure

Located at: `data/golden_dataset.json`

Format (a JSON list of cases):
```json
[
  {
    "question": "What is the refund policy?",
    "ground_truth": "Expected answer...",
    "contexts": ["Optional reference context"],
    "source": "policy_document.pdf",
    "category": "refund_policies"
  }
]
```

**Categories** (based on `data_loader.py`):
- `air_india_policies`: Air India specific policies
- `us_dot_regulations`: U.S. Department of Transportation regulations
- `booking_policies`: General booking and reservation policies
- `refund_policies`: Refund and compensation policies
- `privacy_policies`: Privacy and data protection policies
- `general`: General travel information

---


## 🔍 Testing

### Run All Tests
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=governance --cov=guardrails

# Run specific test file
pytest tests/test_search_engine.py -v

# Run with output
pytest tests/ -v -s
```

### Test Coverage
Current test coverage includes:
- Configuration loading
- Search engine initialization
- Document loading
- Governance checks
- Vector store integration

---


## 🛡️ Governance & Guardrails

The system implements multi-layer safety:

### 1. **Input Validation** (Before Processing)
   - **PII Detection**: Detects email, phone, passport, SSN, credit card
   - **Prompt Injection Detection**: Blocks malicious prompts
   - **Content Safety**: Azure Content Safety + keyword filtering
   - **Blocked queries** return: "Query blocked by security checks"

### 2. **Output Validation** (After Generation)
   - **PII Leakage Prevention**: Ensures no sensitive info in response
   - **Unsafe Content Filtering**: Blocks inappropriate content
   - **Blocked responses** return: "Response didn't pass safety checks"

### 3. **Compliance**
   - **GDPR Compliance**: PII detection and redaction
   - **Audit Logging**: All governance checks logged
   - **Transparency**: Clear violation messages
```

---
## 🐳 Docker Deployment

### Build and Run Locally
```bash
# Build image
docker build -t wanderlust-chatbot .

# Run container
docker run -p 8501:8501 \
  -e AZURE_OPENAI_API_KEY="your-key" \
  -e AZURE_SEARCH_ENDPOINT="your-endpoint" \
  -e AZURE_SEARCH_KEY="your-key" \
  wanderlust-chatbot
```

### Deploy to Cloud
The Dockerfile is ready for deployment to:
- Azure Container Instances
- Azure App Service
- Azure Kubernetes Service
- Any container platform

---

## 📂 Project Structure

```
wanderlust-travel-chatbot/
├── governance/              # Governance & compliance
│   ├── __init__.py
│   ├── compliance_checker.py   # PII detection and GDPR compliance
│   ├── governance_gate.py      # Main governance orchestrator
│   └── safety_validator.py     # Content safety and prompt injection detection
├── guardrails/              # Safety guardrails
│   ├── __init__.py
│   ├── content_safety.py   # Keyword-based content filtering
│   └── pii_detector.py     # Regex-based PII detection
├── src/                     # Application code
│   ├── monitoring/          # Logging & metrics
│   │   ├── __init__.py
│   │   ├── logger.py           # Structured JSON logging
│   │   ├── metrics_collector.py # MLflow metrics tracking
│   │   └── trace_collector.py  # OpenTelemetry tracing
│   ├── app.py              # Streamlit UI
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # Document loading from data/
│   ├── ingestion.py        # Index creation and batch ingestion
│   ├── search_engine.py    # RAG search engine
│   ├── vector_store.py     # Azure AI Search wrapper
│   └── evaluate.py         # Ragas evaluation script
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_search_engine.py
│   └── test_search.py
├── data/                   # Knowledge base (PDFs directly here)
│   ├── *.pdf               # All travel policy documents
│   └── golden_dataset.json # Evaluation test cases
├── document_updates/       # New documents; same-name files override data/
├── reports/                # Generated evaluation reports
│   ├── evaluation_summary.json
│   ├── evaluation_detailed.csv
│   └── category_breakdown.json
├── .env                    # Environment variables (not in git)
├── env.sample              # Sample environment file
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── start_mlflow.sh         # MLflow startup script
├── restart_mlflow.sh       # MLflow restart script
└── README.md              # This file
```

---

## 🎯 Architecture
```
User Query
    ↓
Governance Gate (Input Validation)
    ↓
RAG Pipeline
    ├─ Azure AI Search (Vector Retrieval)
    └─ Azure OpenAI GPT-4 (Generation)
    ↓
Governance Gate (Output Validation)
    ↓
Response to User
```

**Components:**

1. **Data Loading**: `TravelDataLoader` loads PDFs from `data/` folder
2. **Document Categorization**: Auto-categorizes based on filename
3. **Text Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
4. **Embeddings**: Azure OpenAI text-embedding-3-small
5. **Vector Store**: Azure AI Search (no ChromaDB)
6. **LLM**: Azure OpenAI GPT-4
7. **Governance**: Multi-layer input/output validation
8. **Monitoring**: MLflow + Azure Monitor

---

## 📊 Monitoring & Observability

### MLflow Tracking
- **Experiments**: Track ingestion and search operations
- **Parameters**: Log query parameters, model configs
- **Metrics**: Track latency, token usage, retrieval count
- **Artifacts**: Save responses and evaluation results

**Access MLflow UI:**
```bash
./start_mlflow.sh
# Open http://localhost:5000
```

### Azure Monitor (Optional)
- **Application Insights**: Track requests, exceptions, dependencies
- **Structured Logging**: JSON logs with context
- **Distributed Tracing**: OpenTelemetry integration

### Logging
All components use structured JSON logging:
```python
from src.monitoring import logger

logger.info("Search completed", 
    query="baggage rules",
    results_count=5,
    latency_ms=245
)
```

---

## 🔧 Troubleshooting

### Issue: No documents indexed
```bash
# Verify data directory structure
ls -R data/

# Re-run ingestion
python src/ingestion.py
```

### Issue: Search returns no results
```bash
# Test Azure AI Search connection
python -c "from src.search_engine import TravelSearchEngine; engine = TravelSearchEngine(); print('Connected!')"
```

### Issue: MLflow not starting
```bash
# Kill existing processes
pkill -f mlflow

# Start fresh
./restart_mlflow.sh
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments
- Azure AI Services
- LangChain
- Streamlit
- MLflow
```