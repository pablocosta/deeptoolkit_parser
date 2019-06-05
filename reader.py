from torchtext import data
from torchtext.vocab import FastText
import torch

def reader(file_name):

    conllx = {}
    sentence = []
    pos_seq = []
    heads_seq = []
    token = []
    pos = []
    head = []

    with open(file_name, mode='r') as f:
        for line in f:
            contents = line.split('\t')
            if contents[0].isdigit():
                token.append(contents[1].lower())
                pos.append(contents[3])
                head.append(int(contents[6].strip()))
            else:
                # end of sentence
                sentence.append(token)

                pos_seq.append(pos)
                heads_seq.append(head)
                token = []
                pos = []
                head = []
                continue
    conllx['sentence'] = sentence
    conllx['pos'] = pos_seq
    conllx['head'] = heads_seq

    return conllx


class DataPreprocessor(object):
    def __init__(self):
        self.generate_fields()

    def preprocess(self, data_path, file_name, max_len=None, vocab=None, vocab_pos=None):
        self.max_len = max_len
        data         = reader(data_path)
        # Generating torchtext dataset class
        dataset = self.generate_data(data)

        # Building field vocabulary
        if vocab is None:
            self.words_field.build_vocab(data['sentence'], max_size=30000)
        else:
            self.words_field.vocab = vocab

        if vocab_pos is None:
            self.pos_field.build_vocab(data['pos'], min_freq=0)
        else:
            self.pos_field.vocab = vocab_pos

        wrd_vocab, pos_vocab = self.generate_vocabs()

        self.save_data(file_name, dataset)

        return dataset, wrd_vocab, pos_vocab

    def save_data(self, data_file, dataset):
        example_pos  = vars(dataset)["examples"]

        dataset = {'examples': example_pos}

        torch.save(dataset, data_file)

    def generate_fields(self):

        self.words_field = data.Field(tokenize=data.get_tokenizer('spacy'),
                                      use_vocab=True,
                                      init_token='<ROOT>',
                                      unk_token='<UNK>',
                                      pad_token='<PAD>',
                                      batch_first=True)
        self.pos_field = data.Field(tokenize=data.get_tokenizer('spacy'), use_vocab=True, init_token='<ROOT>', batch_first=True)
        self.head_field = data.Field(tokenize=data.get_tokenizer('spacy'), use_vocab=False, init_token='<PAD>', pad_token='<PAD>', batch_first=True)
        self.sentence_id_field = data.Field(tokenize=data.get_tokenizer('spacy'), use_vocab=False, batch_first=True)

        self.fields = [('words', self.words_field),
                   ('pos', self.pos_field),
                   ('head', self.head_field),
                   ('sentence_id', self.sentence_id_field)]


    def load_data(self, data_file, vocab_word, vocab_pos):
        # Loading saved data
        dataset = torch.load(data_file)

        examples = dataset["examples"]


        # Generating torchtext dataset class
        dataset = data.Dataset(fields=self.fields, examples=examples)


        # Building field vocabulary
        if vocab_word is None:
            self.words_field.build_vocab(dataset, max_size=30000)

        else:
            self.words_field.vocab = vocab_word

        if vocab_pos is None:
            self.pos_field.build_vocab(dataset, max_size=30000)
            wrd_vocab, pos_vocab = self.generate_vocabs()
        else:
            self.pos_field.vocab = vocab_pos





        return dataset, wrd_vocab, pos_vocab


    def generate_data(self, senteces):


        return data.Dataset(self._get_examples(self.fields, senteces), self.fields)

    def _get_examples(self, fields, sentences):
        sentence_id = [[i] for i in range(len(sentences['sentence']))]
        ex = []
        for sentence, pos, head, sentence_id in zip(sentences['sentence'], sentences['pos'], sentences['head'], sentence_id):

            if len(sentence) <= self.max_len and len(sentence) != 0:
                ex.append(data.Example.fromlist([sentence, pos, head, sentence_id], fields))
        return ex

    def generate_vocabs(self):
        # Define string to index vocabs
        wrd_vocab = self.words_field.vocab
        pos_vocab = self.pos_field.vocab


        return wrd_vocab, pos_vocab