import pandas as pd
import os
import uuid
import torch

from src.modelling import initialize_model, correct_SMILES

def correct_user_smiles(user_smiles, model, out, device, SRC, temp_path):
       
    pd.DataFrame({'SMILES': [user_smiles]}).to_csv(temp_path, index=False)
    
    valids, df_output = correct_SMILES(model, out, temp_path, device, SRC)
    print(valids)
    print(df_output)
    return df_output["CORRECT"].iloc[0] if "CORRECT" in df_output.columns else None
# ----------------------------
# Load model once before input loop
if __name__ == "__main__":
    
    dummy_error_source = "SMILES-corrector/Data/papyrus_rnn_XS.csv"
    os.makedirs("Data", exist_ok=True)

    if not os.path.exists(dummy_error_source):
        pd.DataFrame({
            "SMILES": ["C1=CC=CC=C1"],
            "SMILES_TARGET": ["C1=CC=CC=C1"]  # Can be the same for dummy
        }).to_csv(dummy_error_source, index=False)
    folder_out = "SMILES-corrector/Data/"
    data_source = "PAPYRUS_200"
    threshold = 200
    invalid_type = 'multiple'
    num_errors = 12

    device = torch.device("cpu")

    model, out, SRC = initialize_model(
        folder_out=folder_out,
        data_source=data_source,
        error_source=dummy_error_source,  # only used to initialize tokenizer
        device=device,
        threshold=threshold,
        epochs=30,
        layers=3,
        batch_size=16,
        invalid_type=invalid_type,
        num_errors=num_errors
    )


    while True:
        user_input = input("💬 Enter an invalid SMILES (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            break
        try:
            temp_path = f"temp_input_{uuid.uuid4().hex[:6]}.csv"
            corrected = correct_user_smiles(user_input, model, out, device, SRC, temp_path)
            print(f"✅ Corrected SMILES: {corrected}\n")
            os.remove(temp_path)
        except Exception as e:
            os.remove(temp_path)
            print(f"❌ Error: {e}\n")
