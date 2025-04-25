import time
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd


class TextSummarizer:
    def __init__(self, text, verbose=False):
        if verbose:
            self.start_time = time.time()
            print('\n  - (nlp model) Starting...')

        self.verbose = verbose

        nltk.download("stopwords", quiet=not verbose)
        self.stopwords_list = stopwords.words("english")

        self.text = text
        self.sentences = []

    def _text_preprocess(self):
        if self.verbose:
            print('  - (nlp model) Text preprocessing...')

        processed_text = [word for word in self.text.split() if word not in self.stopwords_list]

        translator = str.maketrans({key: " " for key in string.punctuation})
        clean_text = self.text.translate(translator)
        clean_text = [word.lower() for word in clean_text.split() if word not in self.stopwords_list]

        processed_text = " ".join(processed_text)
        clean_text = " ".join(clean_text)

        return processed_text, clean_text

    def _get_text_tokens(self):
        if self.verbose:
            print('  - (nlp model) Text tokenizing...')

        processed_text, clean_text = self._text_preprocess()

        self.sentences = sent_tokenize(processed_text)
        sentences_lower = sent_tokenize(processed_text.lower())

        return sentences_lower

    def _text_vectorization(self):
        sentences_lower = self._get_text_tokens()

        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(sentences_lower)
        tfidf.transpose().toarray()

        if self.verbose:
            print(f'  - (nlp model) Vectorized text matrix shape (TFIDF): {tfidf.shape}')

        return vectorizer, tfidf

    def _form_scores(self):
        vectorizer, tfidf_matrix = self._text_vectorization()

        # Formation of word scores
        word_scores = tfidf_matrix.sum(axis=0).transpose()
        word_scores_df = pd.DataFrame(word_scores,
                                      index=vectorizer.get_feature_names_out(),
                                      columns=["score"]).sort_values(by=["score"],
                                                                     ascending=False)
        word_scores_df.drop(index=[word for word in word_scores_df.index
                                   if word in self.stopwords_list + ["like", "yeah", "something", "even"]],
                            inplace=True)

        # Formation of sentence scores
        sentence_scores = tfidf_matrix.sum(axis=1)
        sentence_scores_df = pd.DataFrame(sentence_scores,
                                          columns=["score"]).sort_values(by=["score"],
                                                                         ascending=False)
        sentence_scores_df.reset_index(inplace=True)

        if self.verbose:
            print(f'  - (nlp model) Word scores: \n    {word_scores_df}\n')
            print(f'  - (nlp model) Sentence scores: \n    {sentence_scores_df}\n')

        return word_scores_df, sentence_scores_df

    def get_summary(self, sentences_top_n=2, words_top_n=3):
        w_scores, s_scores = self._form_scores()
        s_indexes = (s_scores.head(sentences_top_n)["index"].astype(int))

        summary = " ".join(self.sentences[i] for i in s_indexes)
        keywords = ", ".join(w_scores.head(words_top_n).index.tolist())

        if self.verbose:
            print(f'  - (nlp model) Running for {(time.time() - self.start_time):.2f} seconds\n')

        print(f'Video text:\n{self.text}\n')
        print(f'Summary:\n{summary}\n')
        print(f'Keywords:\n{keywords}\n')
