# Create an AI text detector using a LLM 

The purpose is to create a model that can detect Ai written text with great accuracy 

### Dataset Aquisition
In this implementation I used this dataset from Kaggle:
https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset


### Dataset preprocessing 

The dataset is not that large, so it has to be used to its full potential\
To do this we split the long sentences in the dataset in sentences that fit the maximum model max length,\
that otherwise would have been truncated to fit the model max length 

original dataframe human examples  : 17508\
original dataframe AI examples  : 11637\
new Dataset human examples  : 25995\
new Dataset dataframe AI examples  : 13507


So we gained a few thousand examples

We need to make sure the data is split 50 /50 to make sure the model after training will be unbiased so we take 13507 human examples to match the 13507 Ai examples\
Then we do a 80 10 10 split on the dataframe and we get:

train dataframe human examples  : 10805\
train dataframe AI examples  : 10806\
test dataframe human examples  : 1351\
test dataframe AI examples  : 1350\
validation dataframe human examples  : 1351\
validation dataframe AI examples  : 1351


### Implementation notes:

I used the Pytorch Lightning framework\
the base model  selected was "google-bert/bert-base-cased"\
We take the output from the pooler_output which is the\
last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function\
This gives us a pretty good summary of the semantic content of the input\
After this we add our classification layer\
torch.nn.Linear(768, 1)\
which we activate with a sigmoid activation function\
Our optimization function is AdamW with learning rate of 2e-5 like in the Bert paper \
The loss is Binary Cross Entropy loss\
The metrics used BinaryAccuracy,BinaryF1Score,BinaryRecall 

Training is done for 10 epochs


### Results :


| metric | value     | 
|--------|-----------|
| eval_F1Score      | 	   0.9715201258659363 | 
| eval_Recall      | 	 0.991117715835571 | 
| eval_accuracy      | 	0.9840858578681946 | 
| eval_loss      | 	0.0723654925823211 | 

| metric | value     | 
|--------|-----------|
| test_F1Score      | 	   0.9604597687721252 | 
| test_Recall      | 	 0.9851906895637512 | 
| test_accuracy      | 	0.9785264730453491  | 
| test_loss      | 	0.07475403696298599  | 


