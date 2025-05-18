
## 🔗 Live Applications

- 🚀 **FastAPI API (with Swagger Docs)**: [API_Link:](https://interview-api-b-tech-mca.onrender.com/docs)
- 🌐 **Streamlit UI App**: [Streamlit_App_Link:](https://interview-ai-app-nq82tdrwdr4xmdrcrhwwmj.streamlit.app/)


# Interview AI App

This repository contains two main components:

1. **FastAPI Backend API** - A REST API that loads a trained ML model to predict candidate selection based on input features.
2. **Streamlit Frontend UI** - A user-friendly web app to input candidate details and get predictions with visualization.

---


---

## Setup and Run

### 1. FastAPI Backend API

**Requirements:**

- Python 3.8+
- Install dependencies:
  
```bash
cd FastAPI_api
pip install -r requirements.txt

Open the UI in your browser at the URL shown in terminal.
Run locally:

streamlit run app.py

Deployment:

    Deploy this folder to Streamlit Cloud or any similar service.

    Ensure interview_model.pkl is included in deployment.

Notes

    Both API and UI use a saved ML model (.pkl) trained on the interview placement dataset.

    The UI and API are independent but use the same prediction logic.
Contact

Created by Vineet Saini

For questions or help, please raise an issue or contact : [Email:] (saini.vineet784@gmail.com)

