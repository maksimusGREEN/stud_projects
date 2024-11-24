from razdel import tokenize
from tqdm import tqdm
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from collections import Counter
from corus import load_lenta
import re
import torch
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))


def get_corpus(path='lenta-ru-news.csv.gz', num_doc=1000):
    records = load_lenta(path)
    corpus = []
    for record in records:
        corpus.append(record.text)
    return corpus[:num_doc]


def process_document(doc):
    doc = doc.lower()
    doc = re.sub(r'[^а-яё/\s]', '', doc)
    tokens = [token.text for token in tokenize(doc)]
    words = [word for word in tokens if word not in stop_words]
    return ' '.join(words)


def corpus_prepros(data: list) -> list:
    # Use joblib to process documents in parallel
    processed_corpus = Parallel(n_jobs=-1)(delayed(process_document)(doc) for doc in tqdm(data))
    return ' '.join(processed_corpus)


def generate_cbows(text, window_size):
    cbows = []
    words = text.split()
    for i, target_word in enumerate(words):
        context_words = words[max(0, i - window_size):i] + words[i + 1:i + window_size + 1]
        if len(context_words) == window_size * 2:
            cbows.append((context_words, target_word))
    return cbows


def generate_skip_grams(text, window_size):
    skip_grams = []
    words = text.split()
    for i, target_word in enumerate(words):
        context_words = words[max(0, i - window_size):i] + words[i + 1:i + window_size + 1]
        for context_word in context_words:
            skip_grams.append((target_word, context_word))
    return skip_grams


def build_vocab(corpus, vocab_size=10000):
    words = [word for word in corpus.split()]
    word_count = sorted(Counter(words).items(), key=lambda x: -x[1])
    vocab = {'UNKNW': 0}
    for idx, (word, _) in enumerate(word_count):
        if idx<vocab_size-1:
            vocab[word] = idx+1
        else: break
    return vocab


def prepare_cbow_input(cbow_pairs, vocab):
    result = []
    for context_words, target_word in cbow_pairs:
        features = torch.tensor([vocab.get(word, vocab['UNKNW']) for word in context_words], dtype=torch.long)
        target = torch.tensor(vocab.get(target_word, vocab['UNKNW']), dtype=torch.long)
        result.append((features, target))
    return result


def prepare_sg_input(skip_gram_pairs, vocab):
    result = []
    for target_word, context_word in skip_gram_pairs:
        target = torch.tensor(vocab.get(target_word, vocab['UNKNW']), dtype=torch.long)
        context = torch.tensor(vocab.get(context_word, vocab['UNKNW']), dtype=torch.long)
        result.append((target, context))
    return result


def data_preparation(data, method):
    vocab = build_vocab(data)
    if method=='sg':
        skip_grams = generate_skip_grams(data, window_size=3)
        skip_gram_vector_pairs = prepare_sg_input(skip_grams, vocab)
        return skip_gram_vector_pairs, vocab
    elif method=='cbow':
        cbows = generate_cbows(data, window_size=3)
        cbow_vector_pairs = prepare_cbow_input(cbows, vocab)
        return cbow_vector_pairs, vocab
    else:
        raise AttributeError(f'The "method" parameter must take the values "sg" or "cbow", not {method}')