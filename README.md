# DocProduct

## Initial Tasks:

-Develop a dataset from /r/askdocs. 
--Figure out best format for data. ie first column post text, 2nd column text from most upvoted commnt, 3rd column text for second most upvoted comment, etc.

-Convert dataset to Tfrecords file using Llion's script

-Set up BERT to encode post text and answer text, and put a similarity scorer on top of BERT 
--Which similarity scorer? NCE loss?
--I believe that one of the members has unlimited TPU instances. Will confirm. But even if we don't, colab offers free TPU, so maybe code it for TPU use either way. 
--Useful Colab notebook, end-to-end fine tuning + prediction using BERT on TPU https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb
--May or may not want to fine tune BERT (not fine tune BERT for now?0

## Scripts

- [x] `train_ffn.py`: Train FFN model.
- [x] `train_bertffn.py`: Train BertFFN model.
- [x] `gpt2_main.py`(Alex): Train generation model.
- [x] `train_data_to_embedding.py`: Convert data to trained FFN model/BertFFN model embeddings for faiss training.
- [x] `train_faiss_topk_to_gpt2.py`(Santosh): Convert faiss top k results to gpt2 training data.
- [ ] `eval_topk.py`(Jay): Evaluation of top k results.
- [x] `inference_question_to_topk.py`: Given question, return top k answers.
- [ ] `inference_question_to_generated_answer.py`: Given question, return generated answer.

### Training pipeline

train_ffn&train_bertffn -> train_data_to_embedding -> train_faiss -> train_faiss_topk_to_gpt2 -> train_gpt2



## Useful link

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

4-27-19
FFNN test embeddings. 
https://drive.google.com/open?id=1ee6I9OHrCiN-wv5BtZX75nyOiWuWTnfs
Trained on about 300 epochs. Won't be the final embeddings, but can be used to get an idea of how well the architecture is performing. 

4-28-19
Current FFNN checkpoint
https://drive.google.com/open?id=1TxR8UBzoBtlewO3lt5efLQgGP_3BRPZm

4-28-19
BERT+FFNN MSE checkpoint
https://drive.google.com/open?id=1U3FRlbSDa0oNz5Mx9PvJad4QagvQh4TI

5-1-19
BERT+FFNN Crossentropy checkpoint
https://drive.google.com/drive/folders/1zMC1Nm8wChJoVY4PDMnjzl7LSsQmhy_h?usp=sharing

5-1-19
Ryan's updated HealthTap data

https://drive.google.com/open?id=1f1f9Caf-xZ4d-FCOD70zUmY7r-1qPp4G

5-2-19
here's the updated healthtap data with the BERT embeds
https://drive.google.com/open?id=1JDp99yeYYRD-Xjk_NhJY_womBQZmOX0z
it's a 5 gb csv file

5-2-19
FAISS top-K text data for GPT-2 training, processed from 4-27-19 FFNN embeds
https://drive.google.com/open?id=13IxjpUohvfscSWUEKZwSY7Lbl6-pqun2

5-2-19
Updated consolodated data, just the pure text, in CSV format
https://drive.google.com/folderview?id=1PymmjbrgfOIs-HJ7oBmjZKH8j4rYsGZj

5-2-19
Final FFNN checkpoint https://drive.google.com/open?id=10K6TiolxsFNZWfSyelzcJ7jm-TG5B0Nl

5-5-19
Finetune model embedding
https://1drv.ms/u/s!An_n1-LB8-2dgfpRUPHMtWcK9qJrIQ

5-5-19
GPT2 training data
https://1drv.ms/u/s!An_n1-LB8-2dgfpS66JqPuBXCRqQHQ
