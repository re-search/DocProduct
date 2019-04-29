# BERT

GPT-2 model from ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

For generating answers to novel questions by conditioning on K-nearest questions/answers in our database

![](architecture.png)

# Acknowledgement

Based on [Minimaxir's GPT-2-simple](https://github.com/CyberZHG/keras-bert)

# Usage

Put text data in gpt_2/data/ and change the txt_file variable in main, then run it:

```
python main.py
```