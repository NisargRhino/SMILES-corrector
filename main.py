import uvicorn
import torch
import pandas as pd
import uuid
import os
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.modelling import initialize_model, correct_SMILES

# ----------- Setup -------------------
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.options("/correct")
async def preflight_handler(request: Request, rest_of_path: str):
    return JSONResponse(status_code=204)


# Model setup
folder_out = "Data/"
data_source = "PAPYRUS_200"
threshold = 200
invalid_type = "multiple"
num_errors = 12
device = torch.device("cpu")

dummy_error_source = "Data/papyrus_rnn_XS.csv"
os.makedirs("Data", exist_ok=True)
if not os.path.exists(dummy_error_source):
    pd.DataFrame({
        "SMILES": ["C1=CC=CC=C1"],
        "SMILES_TARGET": ["C1=CC=CC=C1"]
    }).to_csv(dummy_error_source, index=False)

model, out, SRC = initialize_model(
    folder_out=folder_out,
    data_source=data_source,
    error_source=dummy_error_source,
    device=device,
    threshold=threshold,
    epochs=30,
    layers=3,
    batch_size=16,
    invalid_type=invalid_type,
    num_errors=num_errors
)

# ----------- Correction Logic -------------------
def correct_user_smiles(user_smiles):
    temp_path = f"temp_input_{uuid.uuid4().hex[:6]}.csv"
    pd.DataFrame({'SMILES': [user_smiles]}).to_csv(temp_path, index=False)
    
    try:
        valids, df_output = correct_SMILES(model, out, temp_path, device, SRC)
        corrected = df_output["CORRECT"].iloc[0] if "CORRECT" in df_output.columns else None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return corrected

# ----------- API Routes -------------------
class SMILESRequest(BaseModel):
    smiles: str

@app.get("/")
def home():
    return {"message": "SMILES Corrector API is running."}

@app.post("/correct")
def correct(smiles_req: SMILESRequest):
    try:
        corrected = correct_user_smiles(smiles_req.smiles)
        return JSONResponse(content={"original": smiles_req.smiles, "corrected": corrected})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
