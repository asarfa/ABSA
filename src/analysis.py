class AnalysisData:
    """
    This class allows to do the main analysis of the input data
    """

    def __init__(self, dataset):
        assert(sum(dataset.isnull().sum())==0)
        self.count_label = dataset["polarity"].value_counts(normalize=True)
        self.count_aspect_category = dataset["aspect_category"].value_counts(normalize=True)
        #self.count_target_term = dataset["clean_target_term"].value_counts()
        len_sentences = [len(s) for s in dataset['clean_sentence']]
        self.max_sentence_size = max(len_sentences)
        self.min_sentence_size = min(len_sentences)
        words_sentences = [len(s.split()) for s in dataset['clean_sentence']]
        self.max_words_sentence = max(words_sentences)
        self.min_words_sentence = min(words_sentences)


class AnalysisModel:
    """
    This class allows to do the main analysis of the specified model
    """

    def __init__(self, config):
        self.max_sentence_size = config.max_position_embeddings
        self.embedding_size = config.hidden_state
        self.n_hidden_layer = config.num_hidden_layers
        self.vocab_size = config.vocab_size
