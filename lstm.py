import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# training_data = [
#     ("The dog ate the apple".split(), ['DET', 'NN', 'V', 'DET', 'NN']),
#     ("Everybody read that book".split(), ['NN', 'V', 'DET', 'NN'])
# ]
# word_to_ix = {}
# for sent, tags in training_data:
#     for word in sent:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
# tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}
#
#
# EMBEDDING_DIM = 6
# HIDDEN_DIM = 6
#
# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)
#
# for epoch in range(300):
#     for sentence, tags in training_data:
#         model.zero_grad()
#         model.hidden = model.init_hidden()
#
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = prepare_sequence(tags, tag_to_ix)
#
#         tag_scores = model(sentence_in)
#         loss = loss_function(tag_scores, targets)
#         loss.backward()
#         optimizer.step()
#
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)


class LSTMTaggerWithCharEmbeds(nn.Module):

    def __init__(self, we_dim, ce_dim, c_dim, h_dim, tagset_size, word_to_ix, char_to_ix):
        super(LSTMTaggerWithCharEmbeds, self).__init__()
        self.we_dim = we_dim
        self.c_dim = c_dim
        self.h_dim = h_dim
        self.word_to_ix = word_to_ix
        self.char_to_ix = char_to_ix

        self.w_embedding = nn.Embedding(len(word_to_ix), we_dim)
        self.c_embedding = nn.Embedding(len(char_to_ix), ce_dim)
        self.c_lstm = nn.LSTM(ce_dim, c_dim)

        self.lstm = nn.LSTM(we_dim+c_dim, h_dim)

        self.hidden2tag = nn.Linear(h_dim, tagset_size)

    def make_idxs(self, seq, to_ix):
        idxs = [to_ix[o] for o in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def init_hidden(self, dim):
        return (torch.zeros(1, 1, dim),
                torch.zeros(1, 1, dim))

    def forward(self, sentence):
        embeds = torch.zeros(len(sentence), self.we_dim + self.c_dim)
        w_embeds = self.w_embedding(self.make_idxs(sentence, self.word_to_ix))
        for i, word in enumerate(sentence):
            c_embeds = self.c_embedding(self.make_idxs(word.lower(), self.char_to_ix))
            c_hidden = self.init_hidden(self.c_dim)
            c_out, c_hidden = self.c_lstm(
                c_embeds.view(len(c_embeds), 1, -1), c_hidden)
            embeds[i] = torch.cat([w_embeds[i], c_hidden[0][0][0]], 0)

        hidden = self.init_hidden(self.h_dim)
        out, hidden = self.lstm(embeds.view(len(embeds), 1, -1), hidden)

        tag_space = self.hidden2tag(out.view(len(out), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


training_data = [
    ("The dog ate the apple".split(), ['DET', 'NN', 'V', 'DET', 'NN']),
    ("Everybody read that book".split(), ['NN', 'V', 'DET', 'NN'])
]
word_to_ix = {}
c_to_ix = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
                        'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13,
                        'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
                        'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, }
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}

WORD_EMBED_DIM = 5
CHAR_EMBED_DIM = 3
CHAR_DIM = 3
HIDDEN_DIM = 8

model = LSTMTaggerWithCharEmbeds(WORD_EMBED_DIM, CHAR_EMBED_DIM, CHAR_DIM, HIDDEN_DIM, len(tag_to_ix), word_to_ix, c_to_ix)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    tag_scores = model(training_data[0][0])
    print('BEFORE:', tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()

        targets = model.make_idxs(tags, tag_to_ix)
        tag_scores = model(sentence)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print('EPOCH:', epoch)

with torch.no_grad():
    tag_scores = model(training_data[0][0])
    print('AFTER:', tag_scores)
