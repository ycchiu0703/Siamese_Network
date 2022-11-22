import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models import Phrases
import torch
import torch.nn.functional as F


class SIMPLE_model(torch.nn.Module):
    def __init__(self, input_size , hidden_dim, output_size):
        super(SIMPLE_model,self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_dim // 2, num_layers = 1, bidirectional=True)
        self.Conv1 = torch.nn.Conv1d(1, hidden_dim, kernel_size = 3, stride=1, padding="same")
        self.pool1 = torch.nn.MaxPool1d(kernel_size = 2, stride=2)
        self.Conv2 = torch.nn.Conv1d(64, 128, kernel_size = 3, stride=1,padding="same")
        self.fc1 = torch.nn.Linear(4096,512)  
        self.fc2 = torch.nn.Linear(512,64)  
        self.fc5 = torch.nn.Linear(64,output_size)   
        
    def forward(self,din):
        din = din.unsqueeze(1)
        dout,_ = self.lstm(din)
        dout = F.relu(self.Conv1(dout))
        dout = self.pool1(dout)
        dout = F.relu(self.Conv2(dout))
        dout = torch.flatten(dout)
        dout = F.relu(self.fc1(dout))
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc5(dout))
        return dout

# Extract syscall
def ExtractSyscallSeq(SeqStr):
    SeqStr = ast.literal_eval(SeqStr)
    return [syscall for time, syscall in SeqStr]

def train():
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0001, weight_decay=0)
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data,target in train_loader:
            optimizer.zero_grad()
            output = model(data.float()) 
            loss = lossfunc(output,target[0]) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        train_acc()
        test()

def test():
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.float())
            lbl = torch.argmax(labels, dim=1)[0]
            out = torch.argmax(outputs)
            correct += (out==lbl)
    print('Accuracy of the network on the test data: %d %%' % (
        100 * correct / len(test_loader.dataset)))
    return 100.0 * correct / len(test_loader.dataset)

def train_acc():
    correct = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            outputs = model(images.float())
            lbl = torch.argmax(labels, dim=1)[0]
            out = torch.argmax(outputs)
            correct += (out==lbl)
    print('Accuracy of the network on the train data: %d %%' % (
        100 * correct / len(train_loader.dataset)))
    return 100.0 * correct / len(train_loader.dataset)

def load_dataset(path):
    dataset = pd.read_csv(path)

    mal_fam_cont = {'mirai': 0, 'unknown': 0, 'penguin': 0, 'fakebank': 0, 'fakeinst': 0, 'mobidash': 0, 'berbew': 0, 'wroba': 0}

    # Drop too few samples of famliy(metasploit, zbot, hiddad)
    drop_index = []
    dataset = dataset.drop(['CLASS', 'BEH'], axis=1)

    # Balance family of dataset
    for i in range(len(dataset)):
        if dataset['FAM'][i] in mal_fam_cont and mal_fam_cont[dataset['FAM'][i]] < 10:
            mal_fam_cont[dataset['FAM'][i]] += 1
        else:
            drop_index.append(i)

    dataset = dataset.drop(drop_index)
    dataset = dataset.reset_index(drop = True)

    # Encoding labels
    mal_fam_label = {'mirai': 0, 'unknown': 1, 'penguin': 2, 'fakebank': 3, 'fakeinst': 4, 'mobidash': 5, 'berbew': 6, 'wroba': 7}
    label = []
    for fam in dataset.FAM:
        label.append(mal_fam_label[fam])
    return dataset, label

def data_preprocessing(dataset, label):

    # Extract syscall seq
    Seq = list()
    Seq = [ ExtractSyscallSeq(seq) for seq in dataset.SEQUENCE ]

    max_len = 0
    for i in range(80):
        max_len = max(max_len, len(Seq[i]))

    # bigram w2v
    bigram_transformer = Phrases(Seq)
    model = Word2Vec(bigram_transformer[Seq], min_count = 1)
    word_dict = dict(zip(model.wv.index_to_key, model.wv.vectors))

    # 
    X = []
    for i in Seq:
        tmp = np.zeros(100)
        for j in i:
            tmp += word_dict[j]
        X.append(tmp)

    # normalization
    X = np.array(X) / np.max(np.array(X), axis=0)

    # Split dataset to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size=0.2, shuffle=True)

    # one-hot encoding labels
    Y_train_vec = []
    for i in range(len(Y_train)):
        y_temp = [0]*8 
        y_temp[Y_train[i]] = 1
        Y_train_vec.append(y_temp)

    Y_test_vec = []
    for i in range(len(Y_test)):
        y_temp = [0]*8 
        y_temp[Y_test[i]] = 1
        Y_test_vec.append(y_temp)
    
    # Set train loader
    train_features_init = np.array(X_train)
    train_label_init = np.array(Y_train_vec)
    train_features_init = torch.from_numpy(train_features_init)
    train_label_init = np.array(train_label_init).astype('float32')
    train_label_init = torch.from_numpy(train_label_init).to(torch.float)

    strace_train = torch.utils.data.TensorDataset(train_features_init,train_label_init)
    
    # Set test loader
    test_features_init = np.array(X_test)
    test_label_init = np.array(Y_test_vec)
    test_features_init = torch.from_numpy(test_features_init)
    test_label_init = np.array(test_label_init).astype('float32')
    test_label_init = torch.from_numpy(test_label_init).to(torch.float)

    strace_test = torch.utils.data.TensorDataset(test_features_init, test_label_init)

    return strace_train, strace_test

    
if __name__ == "__main__":
    dataset, label = load_dataset(r'/mnt/bigDisk/weiren/Labels_TimeSyscallSeqs.csv')
    strace_train, strace_test =  data_preprocessing(dataset, label)
    n_epochs = 10000
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(strace_train, batch_size = batch_size, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(strace_test, batch_size = batch_size, num_workers = 0)

    model = SIMPLE_model(input_size=100, hidden_dim=64, output_size=8)
    train()