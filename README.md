# FactVsOp

This is the code for our COLING 2020 paper:
<br> [Fact vs. Opinion: the Role of Argumentation Features in News Classification](https://aclanthology.org/2020.coling-main.540.pdf)
<br> Tariq Alhindi, Smaranda Muresan, Daniel Preotiuc-Pietro

It has a script for training rnn+bert model in a python script (train_rnn_bert.py) or jupyter notebook (rnn+bert.ipynb), which are equivalent.
It also has a script for training the svm model and extracting BERT embedding features.

To train a rnn+bert model, you need:
1. fine-tune a bert sentence-level classifier for argument component tagging (claim, premise, other), which can be done using
[huggingface script for text classification](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification)
2. extract bert embeddings by running a code similar to [extract_bert_features.py](https://github.com/Tariq60/FactVsOp/blob/master/src/features/extract_bert_features.py)
3. train the rnn+bert model using argumentation features and embeddings by running a code similar to [train_rnn_bert.py](https://github.com/Tariq60/FactVsOp/blob/master/src/models/train_rnn_bert.py)

Current scripts are not currently ready for use as is. Modification of data and model directories is needed.
I will aim to provide a more usable version of the scripts soon.
