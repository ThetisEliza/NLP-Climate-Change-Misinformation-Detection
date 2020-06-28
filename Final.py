import os
import sys
from urllib import request

import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from nltk import sent_tokenize
from nltk import word_tokenize
import datetime
import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
import tensorflow.keras as keras
from tqdm import tqdm

Glove_link = 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'
Glove_PATH = 'Glove.6B'
GLove_file = 'glove.6B.300d.txt'

GLOVE = os.path.join(Glove_PATH, GLove_file)
VOCAB_SIZE = 1000
TOKEN_SIZE = 128
STATIC = [False]


import zipfile
import re
def un_zip(file_name):
    zip_file = zipfile.ZipFile(file_name)
    path = re.sub(r'.zip', '', file_name)
    print(path)
    if os.path.isdir(path):
        print('Path exist')
    else:
        os.mkdir(path)
    for names in zip_file.namelist():
        zip_file.extract(names,path)
    zip_file.close()

def check_Glove():
    if not os.path.exists(GLOVE):
        print('Glove not found')

        try:
            # TASK 1: Download the Glove pack
            if not os.path.exists('Glove.6B.zip'):
                print('Downloading Glove..')
                request.urlretrieve(Glove_link, 'Glove.6B.zip')
                print('Downloaded')

            # TASK 2: Uuzip the pack
            print('Unzip Glove')
            un_zip('Glove.6B.zip')
            print('Glove Unzipped')
            # TASK 3: Check the validation again
            if not os.path.exists(GLOVE):
                raise Exception()
        except:
            raise Exception('Fetching Glove failed')

def JsonToDf(*files):
    texts = []
    labels = []
    for file in files:
        df = pd.read_json(file, encoding='utf-8')
        text = list(df.loc['text'].values)
        try:
            label = list(df.loc['label'].values)
        except:
            label = [0 for _ in range(len(text))]
        texts.extend(text)
        labels.extend(label)

    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })

    return df

def replace_str(text):
    return re.sub(u"(\u201c|\u201d)", '"', re.sub(u"(\u2018|\u2019)", "'", text))

def generate_embedding_matrix(word_index, EMBEDDING_DIM = 300):
    word_index = sorted(word_index.items(), key=lambda x: x[1])
    # print('Word index', len(word_index))
    GLOVE_DIR = 'glove.6B'
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    import re
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in tqdm(word_index):

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            word = re.sub(r".*n't", 'not', word)
            word = re.sub(r"'.*", '', word)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                pass
    print('Embedding matrix built')
    return embedding_matrix

class DataGen:

    def gen_new_data(self, max_len=128, vocab_size=10000, F=50, L=50,):

        DATASET_PATH = 'Dataset'

        train_pos = 'train.json'
        train_negs = ['neg_final.json']
        # train_negs = []
        dev = 'dev.json'
        test = 'test-unlabelled.json'
        trains = [os.path.join(DATASET_PATH, train_pos)]+[os.path.join(DATASET_PATH, train_neg) for train_neg in train_negs]


        self.vocab_size = vocab_size - 4
        self.max_len = max_len
        self.F = F
        self.L = L
        self.token_dict = defaultdict(int)
        train_df = JsonToDf(*trains)
        dev_df = JsonToDf(os.path.join(DATASET_PATH, dev))
        test_df = JsonToDf(os.path.join(DATASET_PATH, test))

        # print(train_df.head())
        # print(dev_df.head())
        # print(test_df.head())
        #
        # print(train_df.shape)
        # print(dev_df.shape)
        # print(test_df.shape)

        self.train_text = train_df['text'].values
        self.dev_text = dev_df['text'].values
        self.test_text = test_df['text'].values
        self.train_label = train_df['label'].values
        self.dev_label= dev_df['label'].values
        # self.test_label = test_df['label'].values


        train_tokens_lists, train_labels, train_map = self.process_dataframe(train_df, True)
        dev_tokens_lists, dev_labels, dev_map = self.process_dataframe(dev_df)
        test_tokens_lists, test_labels, test_map = self.process_dataframe(test_df)
        self.make_word_index()

        train_X, train_Y = self.process_tokens(train_tokens_lists, train_labels)
        dev_X, dev_Y = self.process_tokens(dev_tokens_lists, dev_labels)
        test_X, _ = self.process_tokens(test_tokens_lists, test_labels)

        # print(self.reverse_code(train_X[0]))
        # print(self.reverse_code(dev_X[0]))
        # print(self.reverse_code(test_X[0]))

        return train_X, train_Y, dev_X, dev_Y, test_X,




    def reverse_code(self, text_array):
        word_index = self.word_index
        reverse_index = {v:k for k,v in word_index.items()}
        return ' '.join([reverse_index.get(t, '?') for t in text_array])

    def process_tokens(self, tokens_lists, labels):
        codes, labels = self.sentence_coding(tokens_lists, labels)
        return codes, labels

    def process_dataframe(self, df, is_train=False):
        texts = df['text']
        labels = df['label']
        if is_train:
            t_l_tuple = list(zip(texts, labels))
            random.shuffle(t_l_tuple)
            texts, labels = zip(*t_l_tuple)
            texts, labels = self.filter_text(texts, labels)
        all_sents, all_labels, map = self.convert_to_sentences_and_order_map(texts, labels)
        tokens_lists, labels = self.convert_to_tokens_list(all_sents, all_labels)
        return tokens_lists, labels, map

    def filter_text(self, texts, labels):
        new_texts = []
        new_labels = []
        for text, label in list(zip(texts, labels)):
            text = replace_str(text)
            if re.search('(climate|environment|energy|warm|greenhouse|carbon|forest|emission|ipcc|coal|sea|ocean|co2)', text, re.IGNORECASE):
                new_texts.append(text)
                new_labels.append(label)
        return new_texts, new_labels

    def convert_to_sentences_and_order_map(self, texts, labels):
        text_map = {}
        sentence_order = 0
        all_sents = []
        all_labels = []
        for i, (text, label) in enumerate(list(zip(texts, labels))):
            sents = self.get_sentences(text)
            for sent in sents:
                text_map[sentence_order] = i
                all_sents.append(sent)
                all_labels.append(label)
                sentence_order += 1
        return all_sents, all_labels, text_map


    def convert_to_tokens_list(self, all_sents, all_labels):
        tokens_lists = []

        for text in all_sents:
            sents = []
            paras = re.split(r'\n', text)
            for p in paras:
                sents.extend(sent_tokenize(p))
            tokens_list = []
            for sent in sents:
                tokens = word_tokenize(sent)
                tokens_list.append('<START>')
                for token in tokens:
                    token = token.lower()
                    self.token_dict[token] += 1
                tokens_list.extend(tokens)
            tokens_lists.append(tokens_list)
        return tokens_lists, all_labels

    def make_word_index(self):
        vocab_size = self.vocab_size
        token_sort = sorted(self.token_dict.items(), key=lambda x: x[1], reverse=True)
        token_sort = token_sort[:vocab_size]

        word_index = {}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3
        i = 4
        for token, _ in token_sort:
            word_index[token] = i
            i += 1
        self.word_index = word_index

    def sentence_coding(self, tokens_lists, labels):
        X = np.zeros(shape=(len(tokens_lists), self.max_len))
        for i, tokens in enumerate(tokens_lists):
            for j in range(min(len(tokens), self.max_len)):
                X[i][j] = self.word_index.get(tokens[j], 2)
        return np.array(X), np.array(labels)



    def get_sentences(self, text):
        text = text.lower()
        paras = re.split(r'\n', text)
        sents = []
        for p in paras:
            sents.extend(sent_tokenize(p))
        paras = list(set(paras[:2])|set(paras[-8:]))
        return [text]

    def convert_input_text_to_array(self, text):
        tokens_list = [word_tokenize(text)]
        X, Y = self.sentence_coding(tokens_list, [])
        return X

class TextCNN:

    def fit_and_predict(self, embedding_matrix, train_data, train_labels, test_data, test_labels, pred_data,
                        ConV_output, ConV_num, pool_size, static_):
        train_ratio = 0.8

        train_len = int(len(train_data) * train_ratio)

        x_val = train_data[train_len:]
        partial_x_train = train_data[:train_len]

        y_val = train_labels[train_len:]
        partial_y_train = train_labels[:train_len]

        main_input = keras.Input(shape=partial_x_train[0].shape, dtype='float64')
        word_len, EMBEDDING_DIM = embedding_matrix.shape

        embedder = keras.layers.Embedding(word_len, EMBEDDING_DIM, weights=[embedding_matrix], trainable=not static_)
        # embedder = keras.layers.Embedding(len(word_index) + 1, EMBEDDING_DIM,trainable=True)

        embed = embedder(main_input)
        cnns = []
        for i in range(ConV_num):
            cnnl = keras.layers.Conv1D(ConV_output, i + 1, padding='same', strides=1, activation='relu')(embed)
            cnnl = keras.layers.MaxPooling1D(pool_size=pool_size)(cnnl)
            cnns.append(cnnl)
        cnn = keras.layers.concatenate(cnns, axis=-1)
        flat = keras.layers.Flatten()(cnn)
        drop = keras.layers.Dropout(0.2)(flat)
        # dense = keras.layers.Dense(256, activation='relu')(drop)
        main_output = keras.layers.Dense(1, activation='sigmoid')(drop)

        model = keras.Model(inputs=main_input, outputs=main_output)
        # model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=10,
                            batch_size=500,
                            validation_data=(x_val, y_val),
                            verbose=0)
        self.model = model
        # print(model.evaluate(test_data, test_labels))

        # results = model.evaluate(dev_X, dev_labels, verbose=1)



        pred_y = model.predict(test_data)
        predicted_label = np.where(pred_y > 0.5, 1, 0)
        predicted_label = predicted_label.T[0]
        # predicted_label = self.back_to_texts_label(test_map, predicted_label)
        # print(predicted_label)
        # test_labels = self.back_to_texts_label(test_map, test_labels)
        # print(test_labels)

        p, r, f1, _ = precision_recall_fscore_support(test_labels, predicted_label, average='weighted')
        print(ConV_output, ConV_num, pool_size)
        print('p:{}, r:{}, f{}'.format(p, r, f1))
        self.p = p

        if not self.model:
            raise Exception('Model not trained')
        else:
            model = self.model
            pred_Y = model.predict(pred_data)
            # print(pred_Y)

            pred_Y = np.where(pred_Y > 0.5, 1, 0)
            pred_Y = pred_Y.T[0]
            print(sum(pred_Y))
            pred_dict = {}
            name = 'test-'
            for i, y in enumerate(pred_Y):
                pred_dict[name + str(i)] = {'label': str(y)}
            import zipfile

            # pred_dict = {}
            with open('C:\\Users\\73639\\Desktop\\results\\test-output.json', 'w') as f:
                # print(pred_dict)
                json.dump(pred_dict, f)

            file_name = datetime.datetime.now().strftime('%d_%H_%M_%S')
            z = zipfile.ZipFile(
                'C:\\Users\\73639\\Desktop\\results\\' + file_name + ' ' + str(round(self.p * 100)) + 'out.zip', 'w',
                zipfile.ZIP_STORED)
            z.write('C:\\Users\\73639\\Desktop\\results\\test-output.json', 'test-output.json')
            z.close()

        return model, p, r, f1


    def back_to_texts_label(self, map, labels):
        count_map = defaultdict(int)
        for (para_id, text_id), label in list(zip(map.items(), labels)):
            count_map[text_id] += label
        print(count_map)
        return [1 if count_map[i]>=1 else 0 for i in range(len(count_map))]

class LSTM:

    def fit_and_predict(self, embedding_matrix, train_data, train_labels, test_data, test_labels, pred_data,
                        LSTM_output, LSTM_num, static_):
        train_ratio = 0.8

        train_len = int(len(train_data) * train_ratio)

        x_val = train_data[train_len:]
        partial_x_train = train_data[:train_len]

        y_val = train_labels[train_len:]
        partial_y_train = train_labels[:train_len]

        word_len, EMBEDDING_DIM = embedding_matrix.shape

        model = keras.Sequential()
        model.add(keras.layers.Embedding(word_len, EMBEDDING_DIM, weights=[embedding_matrix], trainable=not static_))
        for i in range(LSTM_num):
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output)))
            LSTM_output //= 2
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=10,
                            validation_data=(x_val, y_val),
                            verbose=0

                            )
        self.model = model
        print(model.evaluate(test_data, test_labels))

        pred_y = model.predict(test_data)
        predicted_label = np.where(pred_y > 0.5, 1, 0)
        p, r, f1, _ = precision_recall_fscore_support(test_labels, predicted_label, average='weighted')
        print('p:{}, r:{}, f{}'.format(p, r, f1))
        self.p = p

        if not self.model:
            raise Exception('Model not trained')
        else:
            model = self.model
            pred_Y = model.predict(pred_data)
            # print(pred_Y)

            pred_Y = np.where(pred_Y > 0.5, 1, 0)
            pred_Y = pred_Y.T[0]
            print(sum(pred_Y))
            pred_dict = {}
            name = 'test-'
            for i, y in enumerate(pred_Y):
                pred_dict[name + str(i)] = {'label': str(y)}
            import zipfile

            # pred_dict = {}
            with open('C:\\Users\\73639\\Desktop\\results\\test-output.json', 'w') as f:
                # print(pred_dict)
                json.dump(pred_dict, f)

            file_name = datetime.datetime.now().strftime('%d_%H_%M_%S')
            z = zipfile.ZipFile(
                'C:\\Users\\73639\\Desktop\\results\\' + file_name + ' ' + str(round(self.p * 100)) + 'out.zip', 'w',
                zipfile.ZIP_STORED)
            z.write('C:\\Users\\73639\\Desktop\\results\\test-output.json', 'test-output.json')
            z.close()

        return model, p, r, f1

class BOW:
    def fit_and_predict(self, train_text, train_labels, test_text, test_labels, pred_text):
        train_ratio = 0.8

        vectorizer = CountVectorizer()
        # print(len(train_text))

        train_data = vectorizer.fit_transform(train_text).toarray()
        test_data = vectorizer.transform(test_text).toarray()
        pred_data = vectorizer.transform(pred_text).toarray()

        # print(train_data.shape)
        # print(test_data.shape)
        # print(pred_data.shape)

        train_len = int(len(train_data) * train_ratio)

        x_val = train_data[train_len:]
        partial_x_train = train_data[:train_len]

        y_val = train_labels[train_len:]
        partial_y_train = train_labels[:train_len]

        model = keras.Sequential()

        model.add(keras.layers.Dense(256, activation='relu', input_shape=train_data[0].shape))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=10,
                            validation_data=(x_val, y_val),
                            verbose=0
                            )
        self.model = model
        # print(model.evaluate(test_data, test_labels))

        pred_y = model.predict(test_data)
        predicted_label = np.where(pred_y > 0.5, 1, 0)
        p, r, f1, _ = precision_recall_fscore_support(test_labels, predicted_label, average='weighted')
        print('p:{}, r:{}, f{}'.format(p, r, f1))
        self.p = p

        if not self.model:
            raise Exception('Model not trained')
        else:
            model = self.model
            pred_Y = model.predict(pred_data)
            # print(pred_Y)

            pred_Y = np.where(pred_Y > 0.5, 1, 0)
            pred_Y = pred_Y.T[0]
            print(sum(pred_Y))
            pred_dict = {}
            name = 'test-'
            for i, y in enumerate(pred_Y):
                pred_dict[name + str(i)] = {'label': str(y)}
            import zipfile

            # pred_dict = {}
            with open('C:\\Users\\73639\\Desktop\\results\\test-output.json', 'w') as f:
                # print(pred_dict)
                json.dump(pred_dict, f)

            file_name = datetime.datetime.now().strftime('%d_%H_%M_%S')
            z = zipfile.ZipFile(
                'C:\\Users\\73639\\Desktop\\results\\' + file_name + ' ' + str(round(self.p * 100)) + 'out.zip', 'w',
                zipfile.ZIP_STORED)
            z.write('C:\\Users\\73639\\Desktop\\results\\test-output.json', 'test-output.json')
            z.close()
        return model, p, r, f1


def baseline_test():
    dg = DataGen()
    dg.gen_new_data()
    bow = BOW()
    bow.fit_and_predict(dg.train_text, dg.train_label, dg.dev_text, dg.dev_label, dg.test_text)

def static_test():
    dg = DataGen()
    train_X, train_Y, dev_X, dev_Y, test_X = dg.gen_new_data(max_len=TOKEN_SIZE, vocab_size=VOCAB_SIZE)
    embedding_matrix = generate_embedding_matrix(dg.word_index)

    results = []
    best_model = None
    for static_ in STATIC:
        cnn = TextCNN()
        model, p, r, f = cnn.fit_and_predict(embedding_matrix, train_X, train_Y, dev_X, dev_Y, test_X, 128, 4, 16, static_=static_)
        if not static_:
            best_model = model

        results.append('Static:{}, Model:{}, Precision:{}, Recall:{}, F1-Score{}'.format(
            static_,
            'TextCNN',
            round(p, 2),
            round(r, 2),
            round(f, 2)
        ))

        lstm = LSTM()
        model, p, r, f = lstm.fit_and_predict(embedding_matrix, train_X, train_Y, dev_X, dev_Y, test_X, 128, 1, static_=static_)

        results.append('Static:{}, Model:{}, Precision:{}, Recall:{}, F1-Score{}'.format(
            static_,
            'LSTM',
            round(p, 2),
            round(r, 2),
            round(f, 2)
        ))

    for s in results:
        print(s)
    return dg, best_model

def test(model, dg):
    while True:
        text =input('input your content\n')
        X = dg.convert_input_text_to_array(text)
        result = model.predict(X)
        print('Prediction result:')
        print('Yes' if result[0][0]>0.5 else 'No')


if __name__ == '__main__':
    try:
        # STEP 1: Check the Validation of Glove
        print('STEP 1: Check the Validation of Glove')
        print('Check the validation of Glove')
        check_Glove()
        print('Glove validated')

        # STEP 2: Test
        print('STEP 2: Test')
        print('Start test')
        baseline_test()
        dg, tested_model = static_test()
        print('Test Finished')

        # STEP 3: Model Check
        print('STEP 3: Model Check')
        print('Check model')
        try:
            model = keras.models.load_model('model.h5')
        except:
            tested_model.save('model.h5')
            model = tested_model

        # STEP 4: Test
        print('STEP 4: Test')
        test(model, dg)


    except:
        print('System Error')
        sys.exit()








