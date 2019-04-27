#Intro

Two notebooks need to be run simultaneously, one for running a service that will take requests to convert text into Bert embeddings, the other will be 
the client notebook that will give requests, and also run the FFNN model to convert these Bert embeddings to FFNN embeddings. 

#Instructions

Open both notebooks and do Not run anything yet

make sure both notebooks are on GPU mode. Make sure they are both connected, the top right of the notebook should NOT read 'busy' or 'connecting'. 
Note: These notebooks won't work in CPU or TPU mode (perhaps there's a workaround)

Run the first line of code in TextToBertEmbedToFFNNEmbed.ipynb, 
"from google.colab import drive
drive.mount('/content/gdrive')"
DO NOT INSTALL TF 2.0 IN THIS NOTEBOOK YET otherwise Bert as a Service won't run

Run all the code in BERTServiceForFFNNConversions.ipynb

Look at the output for the last cell in this notebook. VERY IMPORTANT: Will until the output of this cell says something like:
"
I:WORKER-0:[__i:gen:542]:ready and listening!
I:WORKER-2:[__i:gen:542]:ready and listening!
I:WORKER-3:[__i:gen:542]:ready and listening!
I:WORKER-1:[__i:gen:542]:ready and listening!
I:VENTILATOR:[__i:_ru:163]:all set, ready to serve request!
"

Wait, 10 more seconds, and then go to to the 2nd notebook

Finally, run all the cells in TextToBertEmbedToFFNNEmbed.ipynb, , including installing TF 2.0. Some how installing TF 2.0 after bert service is running will not interfere with it.  
