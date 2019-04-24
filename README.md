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

CSV data download with BioBERT embeddings:
https://drive.google.com/drive/folders/1kYD57uStDd4kXyb3JOYCTQd92Al6Il4K?usp=sharing

4-21-19
CSV data of /r/AskDocs Questions and Answers. 
https://drive.google.com/open?id=1t5tWWv5xkkU-YerJZabu-QMBajaqAnDB

4-22-19
Ryan's updated webmed data with BERT embeddings added for questions and answers. Ryan collected all the data from webmd, not just where the old dataset left off from May 2017, so this should replace the old webmd data. 
https://drive.google.com/open?id=1cmlfAO7pnf1kYdoCLe1zH1d0JeFJuknn

4-24-19
TF Records from csv files.
https://drive.google.com/drive/folders/1wRc1jtl5Q0objpfualNFwpg4H575tmks?usp=sharing

TODO:

1. Imbalance label problem
2. Cross entropy loss not decreasing(Santosh)
3. Evaluation and prediction, including Faiss
4. Demo
5. BERT finetune model(Jay)
