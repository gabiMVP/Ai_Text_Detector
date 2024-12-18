# Using dataset Provided here https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset
import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import Trainer
from lightning.pytorch import LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertModel
import textwrap
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
import tensorboard


class Detect_AI_Dataset(Dataset):
    def __init__(self, dataFrame, model_ID):
        # store the image and mask filepaths, and augmentation
        self.df = dataFrame
        self.weight_dtype = torch.float16
        self.text = dataFrame.loc[:, "text"]
        self.labels = dataFrame.loc[:, "generated"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ID
        )

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.text)

    def __getitem__(self, idx):
        # grab the image path from the current index
        text = self.text[idx]
        labels = self.labels[idx]
        ret = self.tokenizer.encode_plus(text,
                                         max_length=self.tokenizer.model_max_length, padding="max_length",
                                         truncation=True,
                                         return_tensors="pt",
                                         return_attention_mask=True,
                                         return_token_type_ids=False
                                         )
        return {
            # .flatten() to remove extra dim in the middle
            "input_ids": ret["input_ids"].flatten(),
            "attention_mask": ret["attention_mask"].flatten(),
            "targets": torch.tensor(labels, dtype=torch.long)
        }


class Detect_AI_DataModule(LightningDataModule):
    def __init__(self, train_df, test_df, val_df, batch_size, model_id):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.model_id = model_id

    def setup(self, stage=None):
        self.train_dataset = Detect_AI_Dataset(
            self.train_df, self.model_id
        )
        self.test_dataset = Detect_AI_Dataset(
            self.test_df, self.model_id
        )
        self.val_dataset = Detect_AI_Dataset(
            self.val_df, self.model_id
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4

        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )


class Litning_AI_TEXT_CLASSIFICAITON_Model(LightningModule):

    def __init__(self, model_ID):
        super().__init__()
        self.base = BertModel.from_pretrained(model_ID)
        self.drop = torch.nn.Dropout(p=0.2)
        # we know this model has output 768 after testing it
        # 1 output between 0 and 1
        self.outputLayer = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = torch.nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.F1Score = BinaryF1Score()
        self.Recall = BinaryRecall()

    def forward(self, input_ids, attention_mask):
        pooler_output = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask
        )['pooler_output']
        output = self.drop(pooler_output)
        output = self.outputLayer(pooler_output)
        return self.sigmoid(output).flatten()

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['targets'].to(device)

        outputs = self(input_ids, attention_mask)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        # return {"loss": loss, "predictions": outputs, "labels": labels}
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['targets'].to(device)

        outputs = self(input_ids, attention_mask)
        loss = self.loss(outputs, labels)
        self.log("eval_loss", loss, prog_bar=True, logger=True)
        accuracy = self.accuracy(outputs, labels)
        self.log("eval_accuracy", accuracy, prog_bar=True, logger=True)
        F1Score = self.F1Score(outputs, labels)
        self.log("eval_F1Score", F1Score, prog_bar=True, logger=True)
        Recall = self.Recall(outputs, labels)
        self.log("eval_Recall", Recall, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['targets'].to(device)

        outputs = self(input_ids, attention_mask)
        loss = self.loss(outputs, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        accuracy = self.accuracy(outputs, labels)
        self.log("test_accuracy", accuracy, prog_bar=True, logger=True)
        F1Score = self.F1Score(outputs, labels)
        self.log("test_F1Score", F1Score, prog_bar=True, logger=True)

        Recall = self.Recall(outputs, labels)
        self.log("test_Recall", Recall, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(params=self.parameters(), lr=2e-5)
        optimizer = transformers.AdamW(params=self.parameters(), lr=2e-5, correct_bias=False)
        return optimizer


def get_new_row_break_up_text(tokenizer, text, generated):
    ret1 = tokenizer(text)
    output_len_sequence = len(ret1['input_ids'])
    parts_to_break = math.ceil(output_len_sequence / tokenizer.model_max_length)
    string_len = math.ceil(len(text) / parts_to_break)
    parts = textwrap.wrap(text, string_len)
    # single_entry = pd.Series({'text': text, 'generated': generated})
    list = []
    for part in parts:
        # any sequence with <10 words we ignore
        if (len(part.split(" "))) < 50:
            continue
        list.append({'text': part, 'generated': generated})
    return list


def main():
    model_ID = "google-bert/bert-base-cased"

    BATCH_SIZE = 8
    EPOCHS = 20

    dataframe = pd.read_csv("./data/Training_Essay_Data.csv")
    print("original dataframe human examples  : " + str(dataframe[dataframe.generated == 0].shape[0]))
    print("original dataframe AI examples  : " + str(dataframe[dataframe.generated == 1].shape[0]))
    tokenizer = AutoTokenizer.from_pretrained(
        model_ID
    )
    isAlready_pre_precessed = os.path.exists('./data/Training_Data_Final.csv')
    if not isAlready_pre_precessed:
        listRows = []
        for idx, row in dataframe.iterrows():
            text = row['text']
            generated = row['generated']
            # to maximize the dataset we break long sequences in more sequences that we know the model can handle using tokenizer.model_max_length
            single_entry = get_new_row_break_up_text(tokenizer, text, generated)
            listRows.extend(single_entry)
        dataframe2 = pd.DataFrame(listRows)
        dataframe2.to_csv('./data/Training_Data_Final.csv', index=False)

    dataframe_used = pd.read_csv('./data/Training_Data_Final.csv')

    examples_AI = dataframe_used[dataframe_used.generated == 1].shape[0]
    examples_human = dataframe_used[dataframe_used.generated == 0].shape[0]
    print("new Dataset human examples  : " + str(examples_human))
    print("new Dataset dataframe AI examples  : " + str(examples_AI))

    assert (examples_AI + examples_human == dataframe_used.shape[0])
    print(examples_human / examples_AI)
    # we make sure our dataset is 50 50
    df1 = dataframe_used[dataframe_used.generated == 0].sample(examples_AI)
    df2 = dataframe_used[dataframe_used.generated == 1]
    newDatframe = pd.concat([df1, df2], ignore_index=True)
    # don't forget to stratify so the proportions of ai and human are 50 50 in all splits
    df_train, df_test = train_test_split(newDatframe, test_size=0.2, stratify=newDatframe['generated'])
    df_test, dt_val = train_test_split(df_test, test_size=0.5, stratify=df_test['generated'])
    print("train dataframe human examples  : " + str(df_train[df_train.generated == 0].shape[0]))
    print("train dataframe AI examples  : " + str(df_train[df_train.generated == 1].shape[0]))
    print("test dataframe human examples  : " + str(df_test[df_test.generated == 0].shape[0]))
    print("test dataframe AI examples  : " + str(df_test[df_test.generated == 1].shape[0]))
    print("validation dataframe human examples  : " + str(dt_val[dt_val.generated == 0].shape[0]))
    print("validation dataframe AI examples  : " + str(dt_val[dt_val.generated == 1].shape[0]))
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    dt_val.reset_index(drop=True, inplace=True)
    data_Module = Detect_AI_DataModule(df_train, df_test, dt_val, BATCH_SIZE, model_ID)
    data_Module.setup()

    m = next(iter(data_Module.train_dataloader()))
    n = next(iter(data_Module.test_dataloader()))
    print(m['input_ids'].shape)
    print(m['attention_mask'].shape)
    print(m['targets'].shape)

    model = BertModel.from_pretrained(model_ID)

    resutl = model(
        input_ids=m['input_ids'],
        attention_mask=m['attention_mask']
    )
    last_hidden_state = resutl['last_hidden_state']
    pooled_output = resutl['pooler_output']
    print(last_hidden_state.shape)
    # get the shape of the poller output
    # Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function

    print(pooled_output.shape)

    ai_detect_model = Litning_AI_TEXT_CLASSIFICAITON_Model(model_ID)
    z = next(iter(data_Module.train_dataloader()))
    output = ai_detect_model(
        input_ids=z['input_ids'],
        attention_mask=z['attention_mask']
    )
    print(output)

    logger = TensorBoardLogger("logs", name="ai_detect_model")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="train_loss",
        mode="min",
        dirpath="content/weights",
        filename="ai_detector-{epoch:02d}-{train_loss:.2f}",
    )

    EPOCHS = 20

    trainer = Trainer(
        precision="16-mixed",
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="gpu",
        devices=1
    )

    ai_detect_model.to(device)
    trainer.validate(model=ai_detect_model, dataloaders=data_Module.val_dataloader())
    # trainer.fit(ai_detect_model,datamodule=data_Module)
    # trainer.save_checkpoint("stable.ckpt")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
