# End-to-End Financial Summarization example using Keras and Hugging Face Transformers

* [Notebook](./summarization-notebook.ipynb)


Welcome to this end-to-end Financial Summarization (NLP) example using Keras and Hugging Face Transformers. In this demo, we will use the Hugging Faces `transformers` and `datasets` library together with `Tensorflow` & `Keras` to fine-tune a pre-trained seq2seq transformer for financial summarization.

We are going to use the [Trade the Evebt](https://paperswithcode.com/paper/trade-the-event-corporate-events-detection) dataset for abstractive text summarization. The benchmark dataset contains 303893​ news articles range from 2020/03/01 to 2021/05/06. The articles are downloaded from the [PRNewswire](https://www.prnewswire.com/) and [Businesswire](https://www.businesswire.com/).


More information for the dataset can be found at the [repository](https://github.com/Zhihan1996/TradeTheEvent/tree/main/data).


We are going to use all of the great Feature from the Hugging Face ecosystem like model versioning and experiment tracking as well as all the great features of Keras like Early Stopping and Tensorboard.
