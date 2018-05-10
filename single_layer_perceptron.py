import numpy as np
import glob
from numpy import binary_repr
from random import uniform


def rep(string):
    return str(string).replace('', ' ')


n = 4
th = 0.7
ita = 1
# n = int(input('Enter number of bits : '))
# th = float(input('Enter a threshold value : '))
# ita = float(input('Enter an adaption rate : '))

w = []
cols = []
for j in range(0, n):
    cols.append(j)
    tmp = uniform(0, 1)
    w.append(tmp)

# w = [0.3, 0.4, 0.5, 0.7, 0.6, 0.3, 0.4, 0.5, 0.1, 0.2]  # sample for 8 bits
initial_w = np.copy(w)


f1 = open('dataset/perceptron_data.txt', 'w')
f2 = open('dataset/perceptron_input.txt', 'w')

seq = pow(2, n)
for j in range(0, seq):
    b = binary_repr(j, width=n)
    b = rep(b)
    f2.write(b + '\n')

    if j%2==1:
        continue

    if j < seq/2:
        f1.write(b + '0\n')
    else:
        f1.write(b + '1\n')

f1.close()
f2.close()

dat = glob.glob('dataset/perceptron_data.txt')
inp = []

for f in dat:
    inp.append(np.loadtxt(f, usecols=cols))

dat = np.loadtxt('dataset/perceptron_data.txt')
d = dat[:, n].astype(np.int)

train = len(d)

x = np.vstack(inp)
y = np.copy(d)+2

i = 0
cnt = 0
delta = 0

while y[train - 1] != d[train - 1] and i < train:
    cnt += 1
    val = 0
    # print('\ncnt = ', cnt, ', i = ', i)

    for j in range(0, n):
        val += w[j]*x[i][j]
    # print('x = ', x[i], ', calculated output = ', val)

    if val < th:
        y[i] = 0
    else:
        y[i] = 1

    tmp = np.copy(w)
    correction = False
    delta = d[i]-y[i]
    # print('tmp = ', tmp)

    for j in range(0, n):
        w[j] += ita * delta * x[i][j]

        if w[j] != tmp[j]:
            correction = True

    # print('w = ', w)
    # print('delta & correction : ', delta, correction)

    if correction == False:
        i += 1
    else:
        i = 0


print('\nNumber of bits = ', n)
print('threshold = ', th)
print('adaption rate = ', ita)
print('Number of training cases = ', train)
print('Number of iterations = ', cnt)
print('\ninitial weight = ', initial_w)
print('\nfinal weight = ', w, '\n')


test = seq
outp = []
# test = int(input('\nEnter no of test cases : '))
f = open('dataset/perceptron_input.txt', 'r')

for i in range(0, test):
    val = 0
    # print('\nCase ', i+1, ' : ')
    # print('Enter input', n, ' bits with spaces : ')

    line = f.readline()

    # tmp = input()
    tmp = line
    inp = list(map(int, tmp.strip().split(' ')))

    for j in range(0, n):
        val += w[j] *inp[j]

    if val < th:
        output = 0
    else:
        output = 1
    # print(inp, ' = ', output)
    outp.append(output)


f.close()
right = 0

for j in range (0,seq):
    if j<seq/2:
        tmp = 0
    else:
        tmp = 1
    # print(j, tmp, outp[j])
    if tmp == outp[j]:
        right += 1
    else:
        b = binary_repr(j, width=n)
        b = rep(b)
        print('-> false for test case : ', b)

accuracy=(right/seq)*100
print('\nnumber of right outputs = ', right)
print('number of test cases = ', seq)
print('Accuracy = ', accuracy, '%')

print('\nExit()')