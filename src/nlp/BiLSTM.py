import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.lstm_forward = nn.LSTM(input_size, hidden_size, num_layers)
        self.lstm_backward = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        hidden_forward = torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)
        out_forward = []
        for i, input_t in enumerate(x):
            output_forward, hidden_forward = self.lstm_forward(input_t.view(1, 1, -1), hidden_forward)
            out_forward.append(output_forward)

        hidden_backward =  torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size)
        out_backward = []

        for i, input_t in reversed(x):
            output_backward, hidden_backward = self.lstm_backward(input_t.view(1, 1, -1), hidden_backward)
            out_backward.append(out_backward)
            out_backward.reverse()

        outputs = [torch.cat((out_forward[i], out_backward[i]), dim=2) for i in range(len(x))]
        output_seq = torch.cat(outputs, dim=0)
        output = self.fc(output_seq.view(len(x), -1))
        return output

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = 28
    sequence_lenght = 28
    num_layers = 2
    hidden_size = 128
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 2

    text_field = Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
    label_field = LabelField(dtype=torch.float)

    train_data, test_data = IMDB.splits(text_field, label_field)

    text_field.build_vocab(train_data, max_Size=10000, vectors='glove.6B.100d')
    label_field.build_vocab(train_data)

    train_iter, test_iter = BucketIterator.splits((train_data, test_data), batch_size=batch_size, device=device)

    model = BiLSTM(input_size=len(text_field.vocab), hidden_size=hidden_size, output_size=1)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for batch in train_iter:
            optimizer.zero_grad()

            text, labels = batch.text, batch.label
            predictions = model(text)
            loss = loss_fn(predictions.squeeze(), labels)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_iter:
            text, labels = batch.text, batch.label
            predictions = model(text)
            predicted_labels = torch.round(torch.sigmoid(predictions))
            correct += (predicted_labels == labels).cum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f'Total accuracy : {accuracy:.3f}')