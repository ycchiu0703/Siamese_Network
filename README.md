# Outline

* Dataset Generation
* Siamese Network
    * Generate Syscall Image
    * Model-Siamese Network
    * Experiment Result
* Compared Methods
    * Preprocessing
    * Model-SIMPLE
    * Model-CNN
* Conclusion
* Future Work
* Reference

[HackMD](https://hackmd.io/L9i8zYMxTb661-YRsmLjQg?view)

# Dataset Generation

我們將樣本從VirusTotal下載，利用每個樣本的Hash name (SHA1)來獲取相對應的Report，再將樣本放入實驗室的Sandbox中，提取出Syscall的資訊以及相對應的時間序列，藉此組成本次實驗的資料集

* 如下圖所示資料集中總共有29221筆資料，並包含12個類別

```python=
mirai       :  20475
unknown     :  8621
penguin     :  10
fakebank    :  47
fakeinst    :  21
mobidash    :  12
berbew      :  13
wroba       :  14
metasploit  :  5
zbot        :  1
skeeyah     :  1
hiddad      :  1
```

![](https://i.imgur.com/AYjsLWA.png)

# Siamese network

## Generate Syscall Image

### Balance dataset

我們觀察發現：其中類別`mirai`佔了總資料的70%左右，同時類別`mirai`+`unknown`更是佔了99.6%，而`zbot` + `skeeyah` + `hiddad`僅佔總資料的0.01%，由此可見資料集中的樣本分佈相當不平衡。

為了解決樣本分佈不平衡的問題，我們決定只取樣本數相對多的家族(家族樣本>=10)，同時為了在樣本分佈和現實場景分佈之間取得權衡，我們只在各家族中取至多50個樣本來組成我們的資料集

* 如下圖所示平衡後的資料集中總共有219筆資料，並包含8個類別

```python=
mirai       :  50
unknown     :  50
penguin     :  10
fakebank    :  47
fakeinst    :  21
mobidash    :  12
berbew      :  13
wroba       :  14
```

![](https://i.imgur.com/Uag6XLq.png)

### Organize time series

每一筆使用的syscall和執行時間序列如下所示 :

```python=
[('1659223215.715424', 'execve'), 
 ('1659223215.755751', 'rt_sigprocmask'),
 ('1659223215.768365', 'rt_sigaction'),
 ('1659223215.793790', 'rt_sigaction'),
 ('1659223215.804770', 'socket'),...]
```

校對執行時間，每個sample皆從時間0.0開始

```python=
syscall_list = []

for i in range(len(dataset)):
    tmp_syscall = []
    tmp = str(dataset["SEQUENCE"][i])
    tmp = tmp.replace('-','0')
    # print(i)
    tmp_seq = ast.literal_eval(tmp)
    end_t = float(tmp_seq[len(tmp_seq) - 1][0])
    start_t = float(tmp_seq[0][0])

    for j in range(len(tmp_seq)):
        try:
            time = float(tmp_seq[j][0])
            time -= start_t
            tmp_syscall.append([time,tmp_seq[j][1]])
        except:
            pass
    syscall_list.append(tmp_syscall)
```

### SyscallCategory 

我們參考[Searchable Linux Syscall Table for x86 and x86_64](https://filippo.io/linux-syscall-table/)，將Syscall分成8類，並新增paper中定義的`HighFreq`作為一類，如下所示:

```python=
SyscallCategory_Original = {
    'kernel' : ['rt_sigaction', 'uname', 'setgid', 'getpriority', 'time', 'clone', 'restart_syscall', 'sysinfo', 'prlimit64', 'geteuid', 'kill', 'umask', 'getppid', 'set_robust_list', 'setresuid', 'nanosleep', 'rt_sigprocmask', 'prctl', 'times', 'mmap', 'setsid', 'vfork', 'wait4', 'getuid', 'gettid', 'set_tid_address', 'fork', 'rt_sigsuspend', 'setpriority', 'ptrace', 'get_thread_area', 'exit', 'alarm', 'setpgid', 'setresgid', 'set_thread_area', 'sigaltstack', 'getrlimit', 'getpid', 'futex', 'setrlimit', 'getegid', 'tgkill', 'setuid', 'exit_group', 'clock_gettime', 'getgid', 'rt_sigtimedwait', 'setitimer', 'getpgrp', 'gettimeofday','mmap2','ni_syscall','sigreturn','waitpid'],

    'fs' : ['stat', 'lseek', 'readlinkat', 'chroot', 'sendfile', 'umount2', 'symlink', 'flock', 'dup2', 'getcwd', 'chdir', 'fstat', 'mount', 'rmdir', 'execve', 'mkdir', 'epoll_wait', 'openat', 'eventfd2', 'readv', 'rename', 'epoll_create1', 'fchmod', 'pipe', 'unlink', 'pipe2', 'fcntl', 'open', 'read', 'write', 'lstat', 'chmod', 'readlink', 'getdents64', 'utimes', 'ioctl', 'select', 'access', 'close', 'poll', 'getdents', 'epoll_ctl', 'ftruncate','_llseek','_newselect','fcntl64','fstat64','llseek','lstat64','renameat2','stat64'],

    'net' : ['accept', 'connect', 'sendto', 'shutdown', 'getsockname', 'getpeername', 'listen', 'socketpair', 'socket', 'setsockopt', 'getsockopt', 'recvfrom', 'recvmsg', 'bind','recv','send','sendfile64','socketcall'],

    'mm' : ['mprotect', 'brk', 'munmap', 'madvise'],

    'ipc' : ['shmdt', 'shmget'],

    'printk' : ['syslog'],

    'sched' : ['sched_getaffinity'],
        
    'HighFreq' : ['_newselect', 'close', 'connect', 'fcntl', 'get_thread_area', 'getsockopt', 'open', 'read', 'recv', 'recvfrom', 'rt_sigaction', 'rt_sigprocmask', 'sendto', 'socket', 'time']
        
    'other' : [ 'getegid32','geteuid32','getgid32','getuid32','setgid32','setresuid32','setuid32','sysctl','ugetrlimit'],
    }
```

我們觀察發現:`ipc`和`printk`種類的Syscall出現次數過少，而其中kernel種類的syscall占大多數，因此依照行為將其切分成3種，並將`ipc`和`printk`合併至`others`中，如下所示:

```python=
SyscallCategory = {
    'kernel_signal' : ['rt_sigaction', 'restart_syscall','kill', 'rt_sigprocmask','rt_sigsuspend','sigaltstack','tgkill','rt_sigtimedwait','sigreturn'],

    'kernel_sys' : ['uname', 'setgid', 'getpriority', 'sysinfo', 'prlimit64','geteuid','umask', 'getppid', 'setresuid','prctl', 'times','mmap','setsid','getuid', 'gettid', 'setpriority','setpgid', 'setresgid','getrlimit','getpid','setrlimit','getegid','setuid','getgid','getpgrp','mmap2','ni_syscall'],

    'kernel_others' : ['time','gettimeofday','alarm','clone','vfork','set_tid_address', 'fork','set_robust_list','futex','nanosleep','wait4','exit','exit_group','waitpid','ptrace','get_thread_area','set_thread_area','clock_gettime','setitimer'],

    'fs' : ['stat', 'lseek', 'readlinkat', 'chroot', 'sendfile', 'umount2', 'symlink', 'flock', 'dup2', 'getcwd', 'chdir', 'fstat', 'mount', 'rmdir', 'execve', 'mkdir', 'epoll_wait', 'openat', 'eventfd2', 'readv', 'rename', 'epoll_create1', 'fchmod', 'pipe', 'unlink', 'pipe2', 'fcntl', 'open', 'read', 'write', 'lstat', 'chmod', 'readlink', 'getdents64', 'utimes', 'ioctl', 'select', 'access', 'close', 'poll', 'getdents', 'epoll_ctl', 'ftruncate','_llseek','_newselect','fcntl64','fstat64','llseek','lstat64','renameat2','stat64'],

    'net' : ['accept', 'connect', 'sendto', 'shutdown', 'getsockname', 'getpeername', 'listen', 'socketpair', 'socket', 'setsockopt', 'getsockopt', 'recvfrom', 'recvmsg', 'bind','recv','send','sendfile64','socketcall'],

    'mm' : ['mprotect', 'brk', 'munmap', 'madvise'],

    'sched' : ['sched_getaffinity'],
    
    'HighFreq' : ['_newselect','close','connect','fcntl','get_thread_area','getsockopt','open','read','recv','recvfrom','rt_sigaction','rt_sigprocmask','sendto','socket','time'],

    'others' : [ 'getegid32','geteuid32','getgid32','getuid32','setgid32','setresuid32','setuid32','sysctl','ugetrlimit','syslog', 'shmdt', 'shmget']
}
```
我們根據兩種不同的Syscall分類方式，進行孿生網路的實驗，並比較兩者之間的差異，第一種分類方式在本文中都稱為"Original SyscallCategory"，第二種則是我們根據上述考量修改的分類方式，在本文中稱為"SyscallCategory"。

### Generate Grayscale Image

雖然論文中使用RGB產生Image，但我們認為灰階圖可以更好的表示每個Syscall類別出現的頻率，也可以視作為一種Frequency Encoding，所以使用灰階圖取代RGB-Image。

```python=
t_step = 16
column, row = t_step, len(syscallCategory)
imgs = []

for i in range(len(syscall_list)):
    time_step = np.linspace(0.0,float(syscall_list[i][-1][0]),num = t_step + 1)
    time_step = time_step[1:]
    k = 0
    step = [[0 for _ in range(row)] for _ in range(column)]
    for j in range(len(syscall_list[i])):
        if syscall_list[i][j][0] <= time_step[k]:
            syscall_fam = get_key(syscall_list[i][j][1])
            for h in syscall_fam:
                step[k][fam[h]] += 1
        else:
            k += 1
            syscall_fam = get_key(syscall_list[i][j][1])
            for h in syscall_fam:
                step[k][fam[h]] += 1                        
    imgs.append(step)

```

## Model-Siamese Network

以下為孿生網路架構:

```python=
class SiameseNetwork(nn.Module):
    """
    Siamese network for image similarity estimation.
    The network is composed of two identical networks, one for each input.
    The output of each network is concatenated and passed to a linear layer. 
    The output of the linear layer passed through a sigmoid function.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # use resnet18 without pretrained weights
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the classifier (linear layers)
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output
```

## Experiment Result

### Original Syscall Category + Siamese Network

Training loss如下圖所示，後面比較淡的是真實的曲線，前面亮橘色的是大概的趨勢，我們可以透過觀察真實曲線發現loss在訓練的過程中不斷震盪，但大致趨勢都保持在0.4-0.5之間。

![](https://i.imgur.com/6LrAQqs.png)

Testing loss如下圖所示，可以發現大致趨勢有明顯下降。

![](https://i.imgur.com/CgiHNMP.png)

Testing Accuracy如下圖所示，在訓練過程中有上升的趨勢。

![](https://i.imgur.com/kbta5lS.png)


### Syscall Category + Siamese Network

Training loss如下圖所示，我們可以發現loss在訓練的過程中雖然也大致趨勢都保持在0.4-0.5之間，但與上面相比平均的loss還是略大一些

![](https://i.imgur.com/Yr9utTR.png)

Testing loss如下圖所示，可以發現大致趨勢雖然也有下降，但與上面相比平均的loss還是略大一些。

![](https://i.imgur.com/TliNEF1.png)

Testing Accuracy如下圖所示，雖然也有明顯上升的趨勢，但相較於上面仍然略差一些。

![](https://i.imgur.com/rMCDni0.png)

# Compared Methods

我們根據比較論文進行實作並改良，沿用了大部分比較論文的架構並加上自己的想法進行實驗，下圖為比較論文中的模型架構：

![](https://i.imgur.com/RFUtHxP.png)

## Preprocessing

由於比較論文的目標是在做few-shot learning，所以我們從資料集中每個家族各取10筆形成我們新的資料集。
* 如下圖所示資料集中總共有80筆資料，並包含8個類別

```python=
mirai       :  10
unknown     :  10
penguin     :  10
fakebank    :  10
fakeinst    :  10
mobidash    :  10
berbew      :  10
wroba       :  10
```


我們將比較論文模型架構中的Word Embedding移到Data Preprocessing中，利用[`gensim.models.phrases`](https://radimrehurek.com/gensim/models/phrases.html)套件進行Word 2 Vector，將每個Syscall embedding 成100維的向量，並將每個樣本中每個embedding後的Syscall向量加總。

```python=
# bigram w2v
bigram_transformer = Phrases(Seq)
model = Word2Vec(bigram_transformer[Seq], min_count = 1)
word_dict = dict(zip(model.wv.index_to_key, model.wv.vectors))

# syscall embedding
X = []
for i in Seq:
    tmp = np.zeros(100)
    for j in i:
        tmp += word_dict[j]
    X.append(tmp)

```

接下來對每個樣本做正規化，並將結果作為我們樣本的特徵。

```python=
# normalization
X = np.array(X) / np.max(np.array(X), axis=0)
```

資料集共80筆，我們將資料集以8:2切分成訓練集和測試集，訓練集64筆，測試集16筆。

```python=
# Split dataset to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)
```

![](https://i.imgur.com/QV8bcAv.png)


## Model-SIMPLE

以下是我們修改後的模型架構，包含BiLSTM層和兩個卷積層，並都使用relu作為激活函數:

```python=
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
```

### Experiment Result

|                  | Training | Testing  |
| --------         | -------- | -------- |
| **Accuracy**     | 78%      | 62%      |

## Model-CNN

為了評估比較模型架構，我們實做了CNN模型進行比較，架構如下圖:

```python=
class CNN_model(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(CNN_model, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, kernel_size=5, stride=1, padding="same")
        self.conv2 = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding="same")
        self.mx_pool1 = torch.nn.MaxPool1d(kernel_size=4, stride=4)     
        self.fc1 = torch.nn.Linear(25, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.mx_pool1(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x)

        return x[0]
```

### Experiment Result

|              | Training | Testing |
| ------------ | -------- | ------- |
| **Accuracy** | 82%      | 56%     |

# Conclusion



| Methods|Training Accuracy|Testing Accuracy|
| ------ | -------- | -------- |
|Original Syscall Category + Siamese Network|--   |74.3%|
|Syscall Category + Siamese Network|--|73.8% |
|W2V + Model-SIMPLE|78%|62%|
|W2V + Model-CNN|82%|56%|

藉由觀察上面實驗結果，我們發現：
1. 在孿生網路和W2V的方法進行比較，孿生網路的準確率明顯優於W2V的方法。
2. 在孿生網路中，不同的Syscall分類方法得到相近的結果，差距僅在0.5%左右，證明在不同的Syscall分類方法上，孿生網路都可以得到不錯的結果。
3. 在model-SIMPLE和model-CNN之間，加入BiLSTM層為準確率提高了6%左右，證明比較論文的模型架構比傳統的CNN模型表現更好。

# Future Work

進行完本次實驗後，我們提出了幾點，未來可以在持續改進的方向：
1. 由於我們在解決各家族樣本數不平衡的問題時，在約兩萬筆`mirai`和約八千筆`unknown`中只各隨機取了50筆樣本，我們猜測取的樣本無法表現出這兩個類別樣本的特性，造成孿生網路的準確率無法提高，未來可以嘗試設計出某些方法，可以取出較具有代表性的樣本以提高準確率。
2. 在比較論文方面，效果不如預期，我們猜測是因為樣本數過少，即便是Few-shot learning，也需要一定量的樣本，才可以達到較高的準確率。


# Reference

1. TANG, Mingdong; QIAN, Quan. Dynamic API call sequence visualisation for malware classification. IET Information Security, 2019, 13.4: 367-377.

2. WANG, Peng; TANG, Zhijie; WANG, Junfeng. A novel few-shot malware classification approach for unknown family recognition with multi-prototype modeling. Computers & Security, 2021, 106: 102273.