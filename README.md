![python](https://img.shields.io/badge/python%20-3.7.1-brightgreen.svg) ![tensorflow](https://img.shields.io/badge/tensorflow-2.0.0--alpha0-orange.svg) ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

# Doc Product: Medical Q&A with Deep Language Models

Download trained models and embedding file [here](https://1drv.ms/u/s!An_n1-LB8-2dgoAYrXYnnBSA4d5dsg?e=i3mnFH).

<p align="center">
  <img src="https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/000/806/964/datas/gallery.jpg">
</p>

## Quality medical information is valuable to everyone, but it's not always readily available. Doc Product aims to fix that.

Whether you've hit your head and are unsure if you need to see a doctor, caught a bad bug halfway up the Himalayas with no idea how to treat it, or made a pact with the ancient spaghetti gods to never accept healthcare from human doctors, *Doc Product* has you covered with up to date information and unique AI-generated advice to address your medical concerns.

We wanted to use TensorFlow 2.0 to explore how well state-of-the-art natural language processing models like [BERT](https://arxiv.org/abs/1810.04805) and [GPT-2](https://openai.com/blog/better-language-models/) could respond to medical questions by retrieving and conditioning on relevant medical data, and this is the result.

## How we built Doc Product

As a group of friends with diverse backgrounds ranging from broke undergrads to data scientists to top-tier NLP researchers, we drew inspiration for our design from various different areas of machine learning. By combining the power of [transformer architectures](https://arxiv.org/abs/1706.03762), [latent vector search](https://github.com/facebookresearch/faiss), [negative sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), and [generative pre-training](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) within TensorFlow 2.0's flexible deep learning framework, we were able to come up with a novel solution to a difficult problem that at first seemed like a herculean task.

<div style="text-align:center"><img src="https://i.imgur.com/wzWt039.png" /></div>

- 700,000 medical questions and answers scraped from Reddit, HealthTap, WebMD, and several other sites
- Fine-tuned TF 2.0 [BERT](https://arxiv.org/abs/1810.04805) with [pre-trained BioBERT weights](https://arxiv.org/abs/1901.08746) for extracting representations from text
- Fine-tuned TF 2.0 [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) with OpenAI's GPT-2-117M parameters for generating answers to new questions
- Network heads for mapping question and answer embeddings to metric space, made with a Keras.Model feedforward network
- Over a terabyte of TFRECORDS, CSV, and CKPT data

If you're interested in the whole story of how we built *Doc Product* and the details of our architecture, [take a look at our GitHub README](https://github.com/Santosh-Gupta/DocProduct)!

## Challenges

Our project was wrought with too many challenges to count, from compressing astronomically large datasets, to re-implementing the entirety of BERT in TensorFlow 2.0, to running GPT-2 with 117 million parameters in Colaboratory, to rushing to get the last parts of our project ready with a few hours left until the submission deadline. Oddly enough, the biggest challenges were often when we had disagreements about the direction that the project should be headed. However, although we'd disagree about what the best course of action was, in the end we all had the same end goal of building something meaningful and potentially valuable for a lot of people. That being said, we would always eventually be able to sit down and come to an agreement and, with each other's support and late-night pep talks over Google Hangouts, rise to the challenges and overcome them together.

## What's next?

Although *Doc Product* isn't ready for widespread commercial use, its surprisingly good performance shows that advancements in general language models like [BERT](https://arxiv.org/abs/1810.04805) and [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) have made previously intractable problems like medical information processing accessible to deep NLP-based approaches. Thus, we hope that our work serves to inspire others to tackle these problems and explore the newly open NLP frontier themselves.

Nevertheless, we still plan to continue work on *Doc Product*, specifically expanding it to take advantage of the 345M, 762M, and 1.5B parameter versions of GPT-2 as OpenAI releases them as part of their [staged release program](https://openai.com/blog/better-language-models/#update). We also intend to continue training the model, since we still have quite a bit more data to go through.

NOTE: We are currrently working on research in scientific/medical NLP and information retrieval. **If you're interested in collaborating, shoot us an e-mail at Research2Vec@gmail.com!**

## Try it out!

### Install from pip

You can install *Doc Product* directly from pip and run it on your local machine. Here's the code to install *Doc Product*, along with TensorFlow 2.0 and FAISS:

```
!wget  https://anaconda.org/pytorch/faiss-cpu/1.2.1/download/linux-64/faiss-cpu-1.2.1-py36_cuda9.0.176_1.tar.bz2
#To use GPU FAISS use
# !wget  https://anaconda.org/pytorch/faiss-gpu/1.2.1/download/linux-64/faiss-gpu-1.2.1-py36_cuda9.0.176_1.tar.bz2
!tar xvjf faiss-cpu-1.2.1-py36_cuda9.0.176_1.tar.bz2
!cp -r lib/python3.6/site-packages/* /usr/local/lib/python3.6/dist-packages/
!pip install mkl

!pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
!pip install https://github.com/Santosh-Gupta/DocProduct/archive/master.zip
```
 
Our repo contains scripts for generating **.tfrefords** data, training *Doc Product* on your own Q&A data, and running *Doc Product* to get answers for medical questions. Please see the **Google Colaboratory demos** section below for code samples to load data/weights and run our models.

### Colaboratory demos

[Take a look at our Colab demos!](https://drive.google.com/open?id=1kXqsE4N0MgfktsEJpZZn470yOZ3UQ10F) We plan on adding more demos as we go, allowing users to explore more of the functionalities of *Doc Product*. All new demos will be added to the same Google Drive folder.

The demos include code for installing *Doc Product* via pip, downloading/loading pre-trained weights, and running *Doc Product*'s retrieval functions and fine-tuning on your own Q&A data.

#### Run our interactive retrieval model to get answers to your medical questions
https://colab.research.google.com/drive/11hAr1qo7VCSmIjWREFwyTFblU2LVeh1R

<p align="center">
  <img src="https://i.imgur.com/Z8DOXuJ.png">
</p>

#### Train your own medical Q&A retrieval model
https://colab.research.google.com/drive/1Rz2rzkwWrVEXcjiQqTXhxzLCW5cXi7xA

<p align="center">
  <img src="https://snag.gy/WPdV5T.jpg">
</p>

#### [Experimental] Run the full Doc Product pipeline with BERT, FCNN, FAISS, and GPT-2 to get your medical questions answered by state-of-the-art AI.
The end-to-end *Doc Product* demo is still **experimental**, but feel free to try it out!
https://colab.research.google.com/drive/1Bv7bpPxIImsMG4YWB_LWjDRgUHvi7pxx

<p align="center">
  <img src="https://snag.gy/WU1YPE.jpg">
</p>

## What it does

Our BERT has been trained to encode medical questions and medical information. A user can type in a medical question, and our model will retrieve the most relevant medical information to that question.

## Data

We created datasets from several medical question and answering forums. The forums are WebMD, HealthTap, eHealthForums, iClinic, Question Doctors, and Reddit.com/r/AskDocs

## Architecture 

The architecture consists of a fine-tuned bioBert (same for both questions and answers) to convert text input to an embedding representation. The embedding is then input into a FCNN (a different one for the questions and answers) to develop an embedding which is used for similarity lookup. The top similar questions and answers are then used by GPT-2 to generate an answer. The full architecture is shown below. 

Lets take a look at the first half of the diagram above above in more detail, the training of the BERT and the FCNNs. A detailed figure of this part is shown below

<p align="center">
  <img src="https://i.imgur.com/IRCyKIL.jpg?1">
</p>

During training, we take a batch of medical questions and their corresponding medical answers, and convert them to bioBert embeddings. The same Bert weights are used for both the questions and answers. 

![DoctorBert](https://i.imgur.com/Lpjjcvk.jpg)

These embeddings are then inputted into a FCNN layer. There are separate FCNN layers for both the question and answer embeddings. 

![DoctorBert](https://i.imgur.com/6HwikW2.jpg)

Now here's where things get a little tricky. Usually embedding similarity training involves negative samples, like how word2vec uses NCE loss. However, we can not use NCE loss in our case since the embeddings are generated during each step, and the weights change during each training step. 

So instead of NCE loss, what we did was compute the dot product for every combination of the question and answer embeddings within our batch. This is shown in the figure below

![DoctorBert](https://i.imgur.com/KOyiCJU.jpg)

Then, a softmax is taken across the rows; for each question, all of it's answer combinations are softmaxed. 

![DoctorBert](https://i.imgur.com/X6N84Gd.jpg)

Finally, the loss used is cross entropy loss. The softmaxed matrix is compared to a ground truth matrix; the correct combinations of questions and answers are labeled with a '1', and all the other combinations are labeled with a '0'. 

## Technical Obstacles we ran into

### Data Gathering and Wrangling

The data gathering was tricky because the formatting of all of the different medical sites was significantly different. Custom work needed to be done for each site in order to pull questions and answers from the correct portion of the HTML tags. Some of the sites also had the possibility of multiple doctors responding to a single question so we needed a method of gathering multiple responses to individual questions. In order to deal with this, we created multiple rows for every question-answer pair. From here we needed to run the model through BERT and store the outputs from one of the end layers in order to make BioBERT embeddings we could pass through the dense layers of our feed-forward neural network(FFNN). 768 dimension vectors were stored for both the question and answers and concatenated with the corresponding text in a CSV file. We tried various different formats for more compact and faster loading and sharing, but CSV ended up being the easiest and most flexible method. After the BioBERT embeddings were created and stored the similarity training process was done and then FFNN embeddings were created that would capture the similarity of questions to answers. These were also stored along with the BioBERT embeddings and source text for later visualization and querying.

### Combining Models Built in TF 1.X and TF 2.0

The embedding models are built in TF 2.0 which utilizes the flexibility of eager execution of TF 2.0. However, GPT2 model that we use are are built in TF 1.X. Luckily, we can train two models separately. While inference, we need to maintain disable eager execution with tf.compat.v1.disable_eager_execution and maintain two separate sessions. We also need to take care of the GPU memory of two sessions to avoid OOM.

## Accomplishments that we're proud of

### Robust Model with Careful Loss and Architecture Design

One obvious approach to retrieve answers based on user’s questions is that we use a powerful encoder(BERT) to encode input questions and questions in our database and do a similarity search. There is no training involves and the performance of this approach totally rely on the encoder. Instead, we use separate Feed-forward networks for questions and answers and calculate cosine similarity between them. Inspired by the negative sampling of word2vec paper, we treat other answers in the same batch as negative samples and calculate cross entropy loss. This approach makes the questions embeddings and answers embeddings in one pair as close as possible in terms of Euclidean distance. It turns out that this approach yields more robust results than doing similarity search directly using BERT embedding vector.

### High-performance Input Pipeline

The preprocessing of BERT is complicated and we totally have around 333K QA pairs and over 30 million tokens. Considering shuffle is very important in our training, we need the shuffle buffer sufficiently large to properly train our model. It took over 10 minutes to preprocess data before starting to train model in each epoch. So we used the tf.data and TFRecords to build a high-performance input pipeline. After the optimization, it only took around 20 seconds to start training and no GPU idle time. 

Another problem with BERT preprocessing is that it pads all data to a fixed length. Therefore, for short sequences, a lot of computation and GPU memory are wasted. This is very important especially with big models like BERT. So we rewrite the BERT preprocessing code and make use of [tf.data.experimental.bucket_by_sequence_length](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/experimental/bucket_by_sequence_length) to bucket sequences with different lengths and dynamically padding sequences. By doing this, we achieved a longer max sequence length and faster training.

### Imperative BERT Model

After some modification, the Keras-Bert is able to run in tf 2.0 environment. However, when we try to use the Keras-Bert as a sub-model in our embedding models, we found the following two problems.
- It uses the functional API. Functional API is very flexible, however, it’s still symbolic. That means even though eager execution is enabled, we still cannot use the traditional python debugging method at run time. In order to fully utilize the power of eager execution, we need to build the model using tf.keras.Model
- We are not directly using the input layer of Keras-Bert and ran into this [issue](https://github.com/tensorflow/tensorflow/issues/27543). It’s not easy to avoid this bug without changing our input pipeline.

As a result, we decided to re-implement an imperative version of BERT. We used some components of Keras-Bert(Multihead Attention, Checkpoint weight loading, etc) and write the call method of Bert. Our implementation is easier to debug and compatible with both flexible eager mode and high-performance static graph mode.

### Answer Generation with Auxiliary Inputs

Users may experience multiple symptoms in various condition, which makes the perfect answer might be a combination of multiple answers. To tackle that, we make use of the powerful GPT2 model and feed the model the questions from users along with Top K auxiliary answers that we retrieved from our data. The GPT2 model will be based on the question and the Top K answers and generate a better answer. To properly train the GPT2 model, we create the training data as following: we take every question in our dataset, do a similarity search to obtain top K+1 answer, use the original answer as target and other answers as auxiliary inputs. By doing this we get the same amount of GPT2 training data as the embedding model training data.

## What we learned

Bert is fantastic for encoding medical questions and answers, and developing robust vector representations of those questions/answers. 

We trained a fine-tuned version of our model which was initialized with Naver's bioBert. We also trained a version where the bioBert weights were frozen, and only trained the two FCNNs for the questions and answers. While we expected the fine-tuned version to work well, we were surprised at how robust later was. This suggests that bioBert has innate capabilities in being able to encode the means of medical questions and answers. 

## What's next for Information Retrieval w/BERT for Medical Question Answering 

Explore if there's any practical use of this project outside of research/exploratory purposes. A model like this should not be used in the public for obtaining medical information. But perhaps it can be used by trained/licenced medical professionals to gather information for vetting. 


Explore applying the same method to other domains (ie history information retrieval, engineering information retrieval, etc.).

Explore how the recently released sciBert (from Allen AI) compares against Naver's bioBert. 

## DISCLAIMER

The recommendations of an open-source AI application is not a substitute for professional medical care. If your condition is worsening, please go to your primary care provider. If you are having an emergency, please go to the nearest hospital or call your country's emergency number.

The purpose of this project is to explore the capabilities of deep learning language models. Although the application will help find publicly available medical advice, you are following it AT YOUR OWN RISK.

<p align="center">
  <img src="https://i.ytimg.com/vi/nPemP-Q0Xn8/hqdefault.jpg">
</p>

## Thanks!

We give our thanks to the TensorFlow team for providing the #PoweredByTF2.0 Challenge as a platform through which we could share our work with others, and a special thanks to Dr. Llion Jones, whose insights and guidance had an important impact on the direction of our project.
