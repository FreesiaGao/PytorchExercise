import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# CONTEXT_SIZE = 2
# EMBEDDING_DIM = 10
#
# test_sentence = "When forty winters shall besiege thy brow, " \
#                 "And dig deep trenches in thy beauty's field, " \
#                 "Thy youth's proud livery so gazed on now, " \
#                 "Will be a totter'd weed of small worth held: " \
#                 "Then being asked, where all thy beauty lies, " \
#                 "Where all the treasure of thy lusty days; " \
#                 "To say, within thine own deep sunken eyes, " \
#                 "Were an all-eating shame, and thriftless praise. " \
#                 "How much more praise deserv'd thy beauty's use, " \
#                 "If thou couldst answer 'This fair child of mine " \
#                 "Shall sum my count, and make my old excuse,' " \
#                 "Proving his beauty by succession thine! " \
#                 "This were to be new made when thou art old, " \
#                 "And see thy blood warm when thou feel'st it cold.".split()
# trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2])
#             for i in range(len(test_sentence) - 2)]
#
# print(trigrams[:3])
#
# vocab = set(test_sentence)
# word_to_ix = {word: i for i, word in enumerate(vocab)}
#
# losses = []
# loss_function = nn.NLLLoss()
# model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
#
# for epoch in range(1000):
#     total_loss = torch.Tensor([0])
#     for context, target in trigrams:
#         context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
#
#         model.zero_grad()
#         log_probs = model(context_idxs)
#         loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#     losses.append(total_loss)
#     if epoch % 100 == 0:
#         print('EPOCh:', epoch)
#
# print(losses[999])
# print(model.embeddings(torch.tensor([word_to_ix['when']], dtype=torch.long)))
# print(model(torch.tensor([word_to_ix['When'], word_to_ix['forty']], dtype=torch.long)))
# print(word_to_ix)


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size * 2, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

raw_text = "We are about to study the idea of a computational process. " \
           "Computational processes are abstract beings that inhabit computers. " \
           "As they evolve, processes manipulate other abstract things called data. " \
           "The evolution of a process is directed by a pattern of rules " \
           "called a program. People create programs to direct processes. In effect, " \
           "we conjure the spirits of the computer with our spells.".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text)-2):
    context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:3])


def make_context_vecter(context, word_to_ix):
    return torch.tensor([word_to_ix[word] for word in context], dtype=torch.long)


model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = []
for epoch in range(150):
    total_loss = torch.tensor([0])
    for context, target in data:
        context_ix = make_context_vecter(context, word_to_ix)

        model.zero_grad()
        log_probs = model(context_ix)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)
    if epoch % 50 == 0:
        print('EPOCH:', epoch)

print(losses[149])
print(model.embeddings(torch.tensor([word_to_ix['We']], dtype=torch.long)))
print(model(torch.tensor([word_to_ix['We'], word_to_ix['are'], word_to_ix['to'], word_to_ix['study']], dtype=torch.long)))
print(word_to_ix)
