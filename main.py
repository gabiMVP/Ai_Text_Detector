from transformers import AutoTokenizer
import lightning
import torch
from train import Litning_AI_TEXT_CLASSIFICAITON_Model

base_model = "google-bert/bert-base-cased"
model_path = "./deploy/model/ai_detect.ckpt"
tokenizer_path = "./deploy/model/tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import time
import uuid
from fastapi import FastAPI, Depends, HTTPException, status
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pydantic import BaseModel
import jwt
from fastapi.middleware.cors import CORSMiddleware

print(torch.__version__)
print(lightning.__version__)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = Litning_AI_TEXT_CLASSIFICAITON_Model.load_from_checkpoint(model_path, model_ID=base_model)
model.eval()
model = model.to(device)

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:8001",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    text: str


def detect_ai_string(input_text):
    inputs = tokenizer.encode_plus(input_text,
                                   max_length=tokenizer.model_max_length, padding="max_length",
                                   truncation=True,
                                   return_tensors="pt",
                                   return_attention_mask=True,
                                   return_token_type_ids=False
                                   )
    # entry = {
    #
    #     # "input_ids": inputs["input_ids"].flatten(),
    #     # "attention_mask": inputs["attention_mask"].flatten()
    #     "input_ids": inputs["input_ids"],
    #     "attention_mask": inputs["attention_mask"]
    # }
    #
    # # inputs = {key: value.to(device) for key, value in inputs.items()}
    # do not flatten in inference mode as we need the extra dim for the bert model inside to consider it the batch size
    print(input_text)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model.forward(
        input_ids, attention_mask
    )
    print(input_ids)
    return output.detach().item()


# Status endpoint
@app.get("/")
def read_root():
    return {"status": "server is running. Use the /detect_ai_text endpoint for ai detection"}


# Translation endpoint
@app.post("/detect_ai_text")
def get_prediction(input_data: InputData):
    print(input_data.text)
    result = detect_ai_string(input_data.text )
    print(result)
    return {
        "score": result
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
