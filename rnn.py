import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):

    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output


rnn = RNN(50, 20, 10)

loss_fn = nn.MSELoss()

batch_size = 10
# recurrent times
TIMESTEPS = 5

# create some fake data
batch = torch.randn(batch_size, 50)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 10)

# backward once for one batch
# the input sequent size is TIMESTEPS, embedding is 50d
loss = 0
for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()
