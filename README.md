# AskDoc


Initial Tasks:

-Develop a dataset from /r/askdocs. 
--Figure out best format for data. ie first column post text, 2nd column text from most upvoted commnt, 3rd column text for second most upvoted comment, etc.

-Convert dataset to Tfrecords file using Llion's script

-Set up BERT to encode post text and answer text, and put a similarity scorer on top of BERT 
--Which similarity scorer? NCE loss?
--I believe that one of the members has unlimited TPU instances. Will confirm. But even if we don't, colab offers free TPU, so maybe code it for TPU use either way. 
--Useful Colab notebook, end-to-end fine tuning + prediction using BERT on TPU https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb
--May or may not want to fine tune BERT (not fine tune BERT for now?0

TFRecords data download:
https://drive.google.com/drive/folders/1Q6Em4Y5PMSOMl_E-HwEMeE9v76XzhBQv?usp=sharing
