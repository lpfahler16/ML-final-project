import torch

# ATTEMPT


class AttemptNNClassifier(torch.nn.Module):
    input_size = 4   # input size
    hidden_size = 50  # width of hidden layer
    output_size = 3   # number of output neurons

    def __init__(self):

        super().__init__()
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        y_output = self.log_softmax(x)
        return y_output

# CONVERT


class ConvertNNClassifier(torch.nn.Module):
    input_size = 4   # input size
    hidden_size = 50  # width of hidden layer
    output_size = 4   # number of output neurons

    def __init__(self):

        super().__init__()
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        y_output = self.log_softmax(x)
        return y_output


# CONVERSION

class ConversionNNClassifier(torch.nn.Module):

    input_size = 4   # input size
    hidden_size = 50  # width of hidden layer
    output_size = 2   # number of output neurons

    def __init__(self):

        super().__init__()
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        y_output = self.log_softmax(x)
        return y_output
