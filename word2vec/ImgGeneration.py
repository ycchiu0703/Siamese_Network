import pandas as pd
import numpy as np
import ast

syscallCategory = {
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

fam = {
    'kernel_signal':0,
    'kernel_sys':1,
    'kernel_others':2,
    'fs':3,
    'net':4,
    'mm':5,
    'sched':6,
    'HighFreq':7,
    'others':8
}

def get_key(target):
    return_key = []
    for key,value in syscallCategory.items():
        if target in value:
            return_key.append(key)
            # return key
    if len(return_key) > 0:
        return return_key
    else:
        return ['other']

###generate dataset
dataset = pd.read_csv(r'/mnt/bigDisk/weiren/Labels_TimeSyscallSeqs.csv')
new_dataset = []
cnt_mirai = 0
cnt_unknown = 0
drop_fam = ['metasploit','zbot','skeeyah','hiddad']
drop_index = []
dataset = dataset.drop(['CLASS', 'BEH'], axis=1)
for i in range(len(dataset)):
    if dataset['FAM'][i] == 'mirai':
        if cnt_mirai > 50:
            drop_index.append(i)
        else:
            cnt_mirai += 1
    elif dataset['FAM'][i] == 'unknown':
        if cnt_unknown > 50 :
            drop_index.append(i)
        else :
            cnt_unknown += 1
    elif dataset['FAM'][i] in drop_fam:
        drop_index.append(i)

dataset = dataset.drop(drop_index)
dataset = dataset.reset_index(drop = True)

label = np.array(dataset["FAM"])


#syscall_time_correction
syscall_list = []
for i in range(len(dataset)):
    tmp_syscall = []
    tmp = str(dataset["SEQUENCE"][i])
    tmp = tmp.replace('-','0')
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
    
## generate_img(16*9)
# 16 time step 
# 9 class of syscall   
column, row = 16, 9
imgs = []
for i in range(len(syscall_list)):
    time_step = np.linspace(0.0,float(syscall_list[i][-1][0]),num=17)
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
imgs = np.array(imgs)


# save imgs and label
np.save("./img_data.npy",imgs)
np.save("./img_label.npy",label)