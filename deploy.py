from transformers import AutoTokenizer
import lightning
import torch
from train import Litning_AI_TEXT_CLASSIFICAITON_Model
base_model = "google-bert/bert-base-cased"
model_path = "./deploy/model"
tokenizer_path = "./deploy/model/tokenizer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained(
#     base_model
# )
# tokenizer.save_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = Litning_AI_TEXT_CLASSIFICAITON_Model.load_from_checkpoint('model/ai_detect.ckpt')
model.eval()
model = model.to(device)
