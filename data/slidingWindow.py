import numpy as np
from sklearn.preprocessing import normalize

n_days_lookahead = int(input('Please input the length of days lookahead in {5, 7, 15, 30, 45, 60, 90, 120}: '))

if(n_days_lookahead not in [5, 7, 15, 30, 45, 60, 90, 120]):
    print('Input does not meet requirements.')
    exit()

data_type = str(input('Please specify the coverage of the data {A - Manufacturer 1, B - Manufacturer 2, C - Manufacturer 1 & 2, D - Unbalanced}: '))


if(data_type not in ['A', 'B', 'C', 'D']):
    print('Input does not meet requirements.')
    exit()

dit_str = {'A': 'mc1', 'B': 'mc2', 'C': 'mc1_mc2', 'D': 'unbalanced'}

if n_days_lookahead == 5:
    prime_array = [1]
elif n_days_lookahead == 7:
    prime_array = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 5, 7]
elif n_days_lookahead == 15:
    prime_array = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 5, 5, 7, 7, 13]
elif n_days_lookahead == 30:
    prime_array = [1, 1, 2, 2, 3, 3, 5, 5, 7, 7, 13, 17, 23, 29]
elif n_days_lookahead == 45:
    prime_array = [1, 2, 3, 5, 7, 13, 17, 23, 29, 31, 37, 41, 43]
elif n_days_lookahead == 60:
    prime_array = [1, 2, 3, 5, 7, 13, 17, 23, 29, 31, 37, 41, 43, 47, 53]
else:
    prime_array = [1, 3, 5, 7, 13, 17, 23, 43, 47, 59]

window_size = 30
slide = 1


def generatePair(X_smart_good, X_smart_bad, random_state=42):
    pair_smart = []
    y = []
    np.random.seed(random_state)
    for i in range(0, len(X_smart_good)):
        pair_smart.append(X_smart_good[i])
        y.append(1)

    for i in range(0, len(X_smart_bad)):
        pair_smart.append(X_smart_bad[i])
        y.append(0)

    return np.asarray(pair_smart), np.asarray(y)


def generatePairSN(X_good, X_bad, random_state=42):
    pair = []
    y = []
    np.random.seed(random_state)
    # similar_pair_good
    for i in range(len(X_good)):
        pair.append(np.stack((X_good[i], X_good[np.random.randint(0, len(X_good))]), axis=0))
        y.append(1)
    # similar_pair_bad
    for i in range(len(X_bad)):
        pair.append(np.stack((X_bad[i], X_bad[np.random.randint(0, len(X_bad))]), axis=0))
        y.append(1)
    # unsimilar_pair
    for i in range(len(X_bad)):
        for j in range(0, 5):
            pair.append(np.stack((X_bad[i], X_good[np.random.randint(0, len(X_good))]), axis=0))
            y.append(0)
        for j in range(5, 10):
            pair.append(np.stack((X_good[np.random.randint(0, len(X_good))], X_bad[i]), axis=0))
            y.append(0)

    return np.asarray(pair), np.asarray(y)


def slidingWindow(data, size=14, slide=1, n_days_lookahead=0):
    currentDiskId = data[0][0]
    ret_list = []
    i = 0
    while(i < len(data) - (size-1)*slide):
        if(currentDiskId != data[i+(size-1)*slide][0]):
            i = i+(size-1)*slide
            currentDiskId = data[i][0]
        else:
            if(abs(data[i][2] - data[i+(size-1)*slide][2]) > slide*25*size):
                i += 1
            else:
                ret_list.append(data[i:i+size*slide:slide, 1:])
                i += np.random.choice(prime_array)

    return np.asarray(ret_list)


def generateBadSample(data, size=14, slide=1, n_days_lookahead=0):
    currentDiskId = data[0][0]
    nextDiskId = 0
    ret_list = []

    i = 0
    while(currentDiskId != -1):
        while(i < len(data)):
            if i == len(data) - 1:
                nextDiskId = -1
                break
            else:
                if(currentDiskId == data[i+1][0]):
                    i += 1
                else:
                    nextDiskId = data[i+1][0]
                    break

        i_ = i - n_days_lookahead - (size-1) * slide

        while(i_ + (size-1)*slide < i):
            while(abs(data[i_][2] - data[i_+(size-1)*slide][2]) > slide*25*size):
                i_ += 1

            ret_list.append(data[i_:i_+size*slide:slide, 1:])
            i_ += np.random.choice(prime_array)

        i += 1
        currentDiskId = nextDiskId

    return np.asarray(ret_list)


good_data_mc1_m1 = np.load('../smart-preprocess/npy/partial_statistics_mc1_model1.npy', allow_pickle=True)
good_data_mc1_m2 = np.load('../smart-preprocess/npy/partial_statistics_mc1_model2.npy', allow_pickle=True)
good_data_mc2_m1 = np.load('../smart-preprocess/npy/partial_statistics_mc2_model1.npy', allow_pickle=True)
good_data_mc2_m2 = np.load('../smart-preprocess/npy/partial_statistics_mc2_model2.npy', allow_pickle=True)
bad_data_mc1_m1 = np.load('../smart-preprocess/npy/bad_mc1_model1.npy', allow_pickle=True)
bad_data_mc1_m2 = np.load('../smart-preprocess/npy/bad_mc1_model2.npy', allow_pickle=True)
bad_data_mc2_m1 = np.load('../smart-preprocess/npy/bad_mc2_model1.npy', allow_pickle=True)
bad_data_mc2_m2 = np.load('../smart-preprocess/npy/bad_mc2_model2.npy', allow_pickle=True)


good_data_mc1_m1_list = []
good_data_mc1_m2_list = []
good_data_mc2_m1_list = []
good_data_mc2_m2_list = []
i = 0
j = 0

# eliminate invalid data
while(i < len(good_data_mc1_m1[0]) and j < len(good_data_mc1_m2[0])):
    if good_data_mc1_m1[0][i][0] == good_data_mc1_m2[0][j][0]:
        good_data_mc1_m1_list.append(good_data_mc1_m1[0][i])
        good_data_mc1_m2_list.append(good_data_mc1_m2[0][j])
        i += 1
        j += 1
    elif good_data_mc1_m1[0][i][0] < good_data_mc1_m2[0][j][0]:
        while(good_data_mc1_m1[0][i][0] < good_data_mc1_m2[0][j][0]):
            if(i < len(good_data_mc1_m1[0])):
                i += 1
            else:
                break
    else: # good_data_mc1_m1[0][i][0] > good_data_mc1_m2[0][j][0]
        while(good_data_mc1_m1[0][i][0] > good_data_mc1_m2[0][j][0]):
            if(j < len(good_data_mc1_m2[0])):
                j += 1
            else:
                break
i = 0
j = 0
while(i < len(good_data_mc2_m1[0]) and j < len(good_data_mc2_m2[0])):
    if good_data_mc2_m1[0][i][0] == good_data_mc2_m2[0][j][0]:
        good_data_mc2_m1_list.append(good_data_mc2_m1[0][i])
        good_data_mc2_m2_list.append(good_data_mc2_m2[0][j])
        i += 1
        j += 1
    elif good_data_mc2_m1[0][i][0] < good_data_mc2_m2[0][j][0]:
        while(good_data_mc2_m1[0][i][0] < good_data_mc2_m2[0][j][0]):
            if(i < len(good_data_mc2_m1[0])):
                i += 1
            else:
                break
    else: # good_data_mc2_m1[0][i][0] > good_data_mc2_m2[0][j][0]
        while(good_data_mc2_m1[0][i][0] > good_data_mc2_m2[0][j][0]):
            if(j < len(good_data_mc2_m2[0])):
                j += 1
            else:
                break

good_data_mc1_m1 = np.asarray(good_data_mc1_m1_list)
good_data_mc1_m2 = np.asarray(good_data_mc1_m2_list)
good_data_mc2_m1 = np.asarray(good_data_mc2_m1_list)
good_data_mc2_m2 = np.asarray(good_data_mc2_m2_list)

state = np.random.get_state()
np.random.set_state(state)
good_data_slidden_mc1_m1 = slidingWindow(good_data_mc1_m1, size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
np.random.set_state(state)
good_data_slidden_mc1_m2 = slidingWindow(good_data_mc1_m2, size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
print(good_data_slidden_mc1_m1.shape)
print(good_data_slidden_mc1_m2.shape)
np.random.set_state(state)
bad_data_slidden_mc1_m1 = generateBadSample(bad_data_mc1_m1[0], size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
np.random.set_state(state)
bad_data_slidden_mc1_m2 = generateBadSample(bad_data_mc1_m2[0], size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
print(bad_data_slidden_mc1_m1.shape)
print(bad_data_slidden_mc1_m2.shape)

state = np.random.get_state()
np.random.set_state(state)
good_data_slidden_mc2_m1 = slidingWindow(good_data_mc2_m1, size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
np.random.set_state(state)
good_data_slidden_mc2_m2 = slidingWindow(good_data_mc2_m2, size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
print(good_data_slidden_mc2_m1.shape)
print(good_data_slidden_mc2_m2.shape)
np.random.set_state(state)
bad_data_slidden_mc2_m1 = generateBadSample(bad_data_mc2_m1[0], size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
np.random.set_state(state)
bad_data_slidden_mc2_m2 = generateBadSample(bad_data_mc2_m2[0], size=window_size, slide=slide, n_days_lookahead=n_days_lookahead)
print(bad_data_slidden_mc2_m1.shape)
print(bad_data_slidden_mc2_m2.shape)

# len of mc1
len_good_mc1 = len(good_data_slidden_mc1_m1)
len_bad_mc1 = len(bad_data_slidden_mc1_m1)

# len of mc2
len_good_mc2 = len(good_data_slidden_mc2_m1)
len_bad_mc2 = len(bad_data_slidden_mc2_m1)

# 1. normalize
data_m1 = np.vstack((good_data_slidden_mc1_m1, good_data_slidden_mc2_m1, bad_data_slidden_mc1_m1, bad_data_slidden_mc2_m1))[:, :, 1:]
print(data_m1.shape)
data_m2 = np.vstack((good_data_slidden_mc1_m2, good_data_slidden_mc2_m2, bad_data_slidden_mc1_m2, bad_data_slidden_mc2_m2))[:, :, 2:]
print(data_m2.shape)
data = np.dstack((data_m1, data_m2))
a1 = len(data)
a2 = len(data[0])
a3 = len(data[0][0])
data = data.reshape((-1, a3), order='C')
data = normalize(data, axis=0, norm='max')
data = data.reshape((-1, a2, a3), order='C')
print(data.shape)

# 2. generate general training set and testing set
X_good_mc1 = data[0:len_good_mc1]
X_good_mc2 = data[len_good_mc1:len_good_mc1+len_good_mc2]
X_bad_mc1 = data[len_good_mc1+len_good_mc2:len_good_mc1+len_good_mc2+len_bad_mc1]
X_bad_mc2 = data[len_good_mc1+len_good_mc2+len_bad_mc1:]
print(X_good_mc1.shape, X_good_mc2.shape, X_bad_mc1.shape, X_bad_mc2.shape)

np.random.shuffle(X_good_mc1)
np.random.shuffle(X_good_mc2)
np.random.shuffle(X_bad_mc1)
np.random.shuffle(X_bad_mc2)

if data_type in ('A', 'B', 'C'):
    if len(X_bad_mc1) > 30000:
        X_good_mc1 = X_good_mc1[0:30000]
        X_bad_mc1 = X_bad_mc1[0:30000]
    else:
        X_good_mc1 = X_good_mc1[0:len(X_bad_mc1)]

    if len(X_bad_mc2) > 30000:
        X_good_mc2 = X_good_mc2[0:30000]
        X_bad_mc2 = X_bad_mc2[0:30000]
    else:
        X_good_mc2 = X_good_mc2[0:len(X_bad_mc2)]
elif data_type in ('D'):
    if len(X_bad_mc1) > 30000:
        X_good_mc1 = X_good_mc1[0:150000]
        X_bad_mc1 = X_bad_mc1[0:30000]
    else:
        X_good_mc1 = X_good_mc1[0:5*len(X_bad_mc1)]

    if len(X_bad_mc2) > 30000:
        X_good_mc2 = X_good_mc2[0:150000]
        X_bad_mc2 = X_bad_mc2[0:30000]
    else:
        X_good_mc2 = X_good_mc2[0:5*len(X_bad_mc2)]


X_good_mc1_train = X_good_mc1[0:int(0.8*len(X_good_mc1))]
X_good_mc1_test = X_good_mc1[int(0.8*len(X_good_mc1)):]
X_good_mc2_train = X_good_mc2[0:int(0.8*len(X_good_mc2))]
X_good_mc2_test = X_good_mc2[int(0.8*len(X_good_mc2)):]
X_bad_mc1_train = X_bad_mc1[0:int(0.8*len(X_bad_mc1))]
X_bad_mc1_test = X_bad_mc1[int(0.8*len(X_bad_mc1)):]
X_bad_mc2_train = X_bad_mc2[0:int(0.8*len(X_bad_mc2))]
X_bad_mc2_test = X_bad_mc2[int(0.8*len(X_bad_mc2)):]


# 3. generate specified training set and testing set
if data_type in ('A'):
    X_good_train = X_good_mc1_train
    X_good_test = X_good_mc1_test
    X_bad_train = X_bad_mc1_train
    X_bad_test = X_bad_mc1_test

elif data_type in ('B'):
    X_good_train = X_good_mc2_train
    X_good_test = X_good_mc2_test
    X_bad_train = X_bad_mc2_train
    X_bad_test = X_bad_mc2_test

elif data_type in ('C', 'D'):
    X_good_train = np.vstack((X_good_mc1_train, X_good_mc2_train))
    X_good_test = np.vstack((X_good_mc1_test, X_good_mc2_test))
    X_bad_train = np.vstack((X_bad_mc1_train, X_bad_mc2_train))
    X_bad_test = np.vstack((X_bad_mc1_test, X_bad_mc2_test))

X_smart_test = np.vstack((X_good_test, X_bad_test))
y_smart_test = np.vstack((np.ones((len(X_good_test), 1)), np.zeros((len(X_bad_test), 1))))
np.save('./' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy', X_smart_test)
np.save('./' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy', y_smart_test)

X_smart_train, y_smart_train = generatePair(X_good_train, X_bad_train)
print(X_smart_train.shape)
state = np.random.get_state()
np.random.set_state(state)
np.random.shuffle(X_smart_train)
np.random.set_state(state)
np.random.shuffle(y_smart_train)
np.save('./' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_train.npy', X_smart_train)
np.save('./' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/train_labels.npy', y_smart_train)
X_smart_test = np.vstack((X_good_test, X_bad_test))
y_smart_test = np.vstack((np.ones((len(X_good_test), 1)), np.zeros((len(X_bad_test), 1))))
np.save('./' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/smart_test.npy', X_smart_test)
np.save('./' + dit_str[data_type] + '/' + str(n_days_lookahead) + '_days_lookahead/test_labels.npy', y_smart_test)
