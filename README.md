# Aspect Based Sentiment Analysis

## Alban Sarfati, Kenz Bensmaine and Salah Azekour


Aspect based sentiment analysis is a type of sentiment analysis that involves identifying the sentiment polarities (negative, neutral, or positive) of a sentence towards specific "opinion targets" mentioned within it.

## Dataset 
We conduct our project on a dataset composed of Restaurant reviews. The dataset is in TSV format, one instance per line.
For example: 

negative SERVICE#GENERAL Wait staff 0:10 Wait staff is blantently unappreciative of your business!

The training set is composed of 1503 instances with max_sentence_size=355 and min_sentence_size=9, while the dev set is composed of 376 instances with max_sentence_size = 355 and min_sentence_size=10

## Statistics of the dataset

### Polarity Weights

The dataset is balanced across the polarity of the sentences. The weights for each polarity class are as follows:

| Polarity | Training | Dev      |
| -------- | -------- |----------|
| positive | 0.701929 | 0.702128 |
| negative | 0.259481 | 0.260638 |
| neutral  | 0.038589 | 0.037234 |

### Aspect Categories Weights

The dataset contains sentences belonging to different aspect categories. The weights for each aspect category are as follows:

| Aspect Category           | Training | Dev      |
| -------------------------| -------- |----------|
| SERVICE#GENERAL           | 0.174983 | 0.162234 |
| AMBIENCE#GENERAL          | 0.125083 | 0.101064 |
| RESTAURANT#GENERAL        | 0.091816 | 0.125000 |
| FOOD#STYLE_OPTIONS        | 0.065203 | 0.045213 |
| FOOD#PRICES               | 0.038589 | 0.031915 |
| DRINKS#QUALITY            | 0.027279 | 0.007979 |
| RESTAURANT#MISCELLANEOUS  | 0.025948 | 0.026596 |
| DRINKS#STYLE_OPTIONS      | 0.017299 | 0.015957 |
| RESTAURANT#PRICES         | 0.013307 | 0.015957 |
| LOCATION#GENERAL          | 0.010645 | 0.015957 |
| DRINKS#PRICES             | 0.008649 | 0.018617 |

## Proposed Methodology 

To solve this task we used pre-trained language model based on Attention mechanism, which enforce the model to pay
more attention to context words with closer semantic relations with the target.

The the dataset has been processed by cleaning the text and encoding the labels ('positive'=2, 'neutral'=1, 'negative'=0).

### Embedding Layer

We apply pre-trained case BERT-base to generate word vectors of sequence. We transform the given context and target
to “[CLS] + context + [SEP]” and “target + [SEP]” respectively. The indices of input sequence tokens in the vocabulary is their concatenation, the attention mask, to avoid performing attention on padding token indices,
and segment token indices, to indicate first and second portions of the inputs, were also engineered. These three inputs were subject to a padding and truncation method. 

### Model

We use pre-trained case BERT-base as our Attentional Encoder Network, we get the average pooling, concatenate them as
the final comprehensive representation and use dropout plus
a full connected layer to project the concatenated
vector into the space of the targeted classes (polarities).


### Loss

The objective function to be optimized is the cross-entropy loss with L2
regularization, while AdamW optimizer is applied to update all the parameters.

### Experimental Settings

We fine-tuned the model by searching the hyper-parameters in the space {'lr': [2e-5, 5e-5],
                'l2reg': [0.1, 0.01],
                'dropout': [0.1, 0.2],
                'batch_size': [16, 32, 64]} leading to the maximum accuracy on the evaluation dataset. The pytorch seed has been set at 42 in order to obtain reproducible results. The maximum number of epoch is 20 which is enough to attain convergence according to our early stopping method. 
The best hyper-parameters are stored to src/savings/ path.
The experiment was runned on the default Colab GPU.

## Results

The hyperparamters leading to the maximum accuracy on the dev dataset are {'lr': 2e-05, 'l2reg': 0.1, 'dropout': 0.2, 'batch_size': 32, 'epoch': 14} allowing to attain 0.861 with the random seed 42.

By unsetting the seed, using the best hyper-parameters and runing 5 times, we get these results:

Dev accs: [84.04, 84.57, 86.44, 84.57, 84.04]

Mean Dev Acc.: 84.73 (0.89)
 
Exec time: 1315.23 s. ( 263 per run )

path:  src/savings/results_bert-base-cased_lr2e-05_batchsize32_dropout0.2_l2reg0.1.txt

## Note

Our methods save the best models (in src/savings/) during training to retrieve them during evaluation. Hence, some storage capacity is needed during training, at the end of the run, the files are deleted. 