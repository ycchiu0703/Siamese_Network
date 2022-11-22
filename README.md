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

[Traditional Chinese Report @ HackMD](https://hackmd.io/L9i8zYMxTb661-YRsmLjQg?view)

# Dataset Generation

We download the samples from VirusTotal, use the Hash name (SHA1) of each sample to obtain the corresponding Report, then put the samples into the Sandbox of the laboratory, extract the Syscall information and the corresponding time series, and thus compose Dataset for this experiment.

* As shown in the figure below, there are a total of 29221 data in the data set, and it contains 12 categories.

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

We observed that the category `mirai` accounted for about 70% of the total data, and the category `mirai`+`unknown` accounted for 99.6%, while `zbot` + `skeeyah` + `hiddad` only accounted for the total data 0.01% of , it can be seen that the distribution of samples in the data set is quite unbalanced.

In order to solve the problem of unbalanced sample distribution, we decided to only take families with a relatively large number of samples (family samples>=10). samples to form our data set

* As shown in the figure below, the balanced data set has a total of 219 data and contains 8 categories.

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

The syscall and execution time series used by each transaction are as follows:

```python=
[('1659223215.715424', 'execve'), 
 ('1659223215.755751', 'rt_sigprocmask'),
 ('1659223215.768365', 'rt_sigaction'),
 ('1659223215.793790', 'rt_sigaction'),
 ('1659223215.804770', 'socket'),...]
```

Proofread execution time, each sample starts from time 0.0.

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

We refer to [Searchable Linux Syscall Table for x86 and x86_64](https://filippo.io/linux-syscall-table/) to divide Syscall into 8 categories, and add `HighFreq` defined in the paper as a category, as follows:

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

We observed that the occurrences of `ipc` and `printk` types of Syscalls are too few, and the kernel type syscalls account for the majority, so they are divided into three types according to the behavior, and `ipc` and `printk` are merged into` others`, as follows:

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

We conduct twin network experiments based on two different Syscall classification methods and compare the differences between the two. The first classification method is called "Original SyscallCategory" in this article, and the second is based on the above Consider the classification of modifications, referred to in this document as "SyscallCategory".

### Generate Grayscale Image

Although the paper uses RGB to generate the Image, we believe that the grayscale image can better represent the frequency of each Syscall category, and can also be regarded as a Frequency Encoding, so the grayscale image is used instead of the RGB-Image.

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

The following is the Siamese network architecture:

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

The training loss is shown in the figure below. 
The lighter one in the back is the real curve, and the brighter orange in the front is the approximate trend. 
We can observe the real curve and find that the loss is constantly oscillating during the training process, but the general trend remains at 0.4- between 0.5.

![](https://i.imgur.com/6LrAQqs.png)

Testing loss is shown in the figure below, and it can be found that the general trend has decreased significantly.

![](https://i.imgur.com/CgiHNMP.png)

Testing Accuracy As shown in the figure below, there is an upward trend during the training process.

![](https://i.imgur.com/kbta5lS.png)


### Syscall Category + Siamese Network

Training loss is shown in the figure below. We can find that although the general trend of loss remains between 0.4-0.5 during the training process, the average loss is still slightly larger than the above.

![](https://i.imgur.com/Yr9utTR.png)

Testing loss is shown in the figure below. It can be seen that although the general trend has also declined, the average loss is still slightly larger than the above.

![](https://i.imgur.com/TliNEF1.png)

Testing Accuracy is shown in the figure below. Although there is an obvious upward trend, it is still slightly worse than the above.

![](https://i.imgur.com/rMCDni0.png)

# Compared Methods

We implemented and improved based on the comparative papers, and followed the structure of most of the comparative papers and added our own ideas for experiments. The following figure shows the model architecture in the comparative papers:

![](https://i.imgur.com/RFUtHxP.png)

## Preprocessing

Since the goal of the comparison paper is to do few-shot learning, we take 10 records from each family in the data set to form our new data set.

* As shown in the figure below, there are a total of 80 data in the data set, and it contains 8 categories

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


We moved the Word Embedding in the comparative paper model architecture to Data Preprocessing, using [`gensim.models.phrases`](https://radimrehurek.com/gensim/models/phrases.html) to make Word 2 Vector.
Convert each Syscall embedding into a 100-dimensional vector, and sum the Syscall vectors after each embedding in each sample.

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

Next, regularize each sample and use the result as the feature of our sample.

```python=
# normalization
X = np.array(X) / np.max(np.array(X), axis=0)
```

There are a total of 80 records in the data set. We divide the data set into a training set and a test set at 8:2, with 64 records in the training set and 16 records in the test set.

```python=
# Split dataset to train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size = 0.2, shuffle = True)
```

![](https://i.imgur.com/QV8bcAv.png)


## Model-SIMPLE

The following is our modified model architecture, which contains a BiLSTM layer and two convolutional layers, and both use relu as the activation function:

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

In order to evaluate and compare the model architecture, we implemented the CNN model for comparison. The architecture is as follows:

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



| Methods                                   |Training Accuracy|Testing Accuracy|
| ------                                    | --------        | --------       |
|Original Syscall Category + Siamese Network|--               |74.3%           |
|Syscall Category + Siamese Network         |--               |73.8%           |
|W2V + Model-SIMPLE                         |78%              |62%             |
|W2V + Model-CNN                            |82%              |56%             |

By observing the above experimental results, we found that:
1. Comparing the Siamese network and W2V methods, the accuracy of the twin network is significantly better than the W2V method.
2. In the Siamese network, different Syscall classification methods get similar results, and the gap is only about 0.5%, which proves that the Siamese network can get good results on different Syscall classification methods.
3. Between model-SIMPLE and model-CNN, adding the BiLSTM layer increases the accuracy rate by about 6%, which proves that the model architecture of the comparison paper performs better than the traditional CNN model.

# Future Work

After conducting this experiment, we put forward two points, which can be continuously improved in the future:
1. Since we only randomly selected 50 samples from about 20,000 `mirai` and about 8,000 `unknown` when we solved the problem of unbalanced samples in each family, we guessed that the samples we took could not show the two Due to the characteristics of each category of samples, the accuracy of the twin network cannot be improved. In the future, we can try to design some methods that can take out more representative samples to improve the accuracy.
2. In terms of comparing papers, the effect is not as expected. We guess that the number of samples is too small. Even Few-shot learning requires a certain amount of samples to achieve a high accuracy rate.

# Reference

1. [TANG, Mingdong; QIAN, Quan. Dynamic API call sequence visualisation for malware classification. IET Information Security, 2019, 13.4: 367-377.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ifs.2018.5268)

2. [WANG, Peng; TANG, Zhijie; WANG, Junfeng. A novel few-shot malware classification approach for unknown family recognition with multi-prototype modeling. Computers & Security, 2021, 106: 102273.](https://www.sciencedirect.com/science/article/pii/S0167404821000973)