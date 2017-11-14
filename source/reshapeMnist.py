#coding:utf-8
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# mnistデータよみこみ
with open('mnist.pkl', 'rb') as file:
    dataset_mnist = pickle.load(file)
data = dataset_mnist['train_img'].reshape((60000, 1,  28, 28))
data = np.asarray( data, dtype = np.float32 ) / 255.0
labels = dataset_mnist['train_label']
print(labels[0:9])

valid_data = dataset_mnist['test_img'].reshape((-1, 1,  28, 28))
valid_labels = dataset_mnist['test_label']

# カテゴリごとのサンプル数をカウント
train_num = np.zeros(10, dtype=int)

# データ、ラベルの配列の雛形
class0 = np.empty((0,28,28), dtype=float)
class0 = np.empty((0,28,28), dtype=float)
class1 = np.empty((0,28,28), dtype=float)
class2 = np.empty((0,28,28), dtype=float)
class3 = np.empty((0,28,28), dtype=float)
class4 = np.empty((0,28,28), dtype=float)
class5 = np.empty((0,28,28), dtype=float)
class6 = np.empty((0,28,28), dtype=float)
class7 = np.empty((0,28,28), dtype=float)
class8 = np.empty((0,28,28), dtype=float)
class9 = np.empty((0,28,28), dtype=float)
class0_labels = np.empty((0,1), dtype=int)
class1_labels = np.empty((0,1), dtype=int)
class2_labels = np.empty((0,1), dtype=int)
class3_labels = np.empty((0,1), dtype=int)
class4_labels = np.empty((0,1), dtype=int)
class5_labels = np.empty((0,1), dtype=int)
class6_labels = np.empty((0,1), dtype=int)
class7_labels = np.empty((0,1), dtype=int)
class8_labels = np.empty((0,1), dtype=int)
class9_labels = np.empty((0,1), dtype=int)

for i in tqdm(range(60000)):
    if(labels[i] == 0):
        class0 = np.append(class0, data[i], axis=0)
        class0_labels = np.append(class0_labels, labels[i])
        train_num[0] += 1

    if(labels[i] == 1):
        class1 = np.append(class1, data[i], axis=0)
        class1_labels = np.append(class1_labels, labels[i])
        train_num[1] += 1

    if(labels[i] == 2):
        class2 = np.append(class2, data[i], axis=0)
        class2_labels = np.append(class2_labels, labels[i])
        train_num[2] += 1

    if(labels[i] == 3):
        class3 = np.append(class3, data[i], axis=0)
        class3_labels = np.append(class3_labels, labels[i])
        train_num[3] += 1

    if(labels[i] == 4):
        class4 = np.append(class4, data[i], axis=0)
        class4_labels = np.append(class4_labels, labels[i])
        train_num[4] += 1

    if(labels[i] == 5):
        class5 = np.append(class5, data[i], axis=0)
        class5_labels = np.append(class5_labels, labels[i])
        train_num[5] += 1

    if(labels[i] == 6):
        class6 = np.append(class6, data[i], axis=0)
        class6_labels = np.append(class6_labels, labels[i])
        train_num[6] += 1

    if(labels[i] == 7):
        class7 = np.append(class7, data[i], axis=0)
        class7_labels = np.append(class7_labels, labels[i])
        train_num[7] += 1

    if(labels[i] == 8):
        class8 = np.append(class8, data[i], axis=0)
        class8_labels = np.append(class8_labels, labels[i])
        train_num[8] += 1

    if(labels[i] == 9):
        class9 = np.append(class9, data[i], axis=0)
        class9_labels = np.append(class9_labels, labels[i])
        train_num[9] += 1

dataset = {}
dataset['class0'] = class0
dataset['class1'] = class1
dataset['class2'] = class2
dataset['class3'] = class3
dataset['class4'] = class4
dataset['class5'] = class5
dataset['class6'] = class6
dataset['class7'] = class7
dataset['class8'] = class8
dataset['class9'] = class9


dataset['class0_labels'] = class0_labels
dataset['class1_labels'] = class1_labels
dataset['class2_labels'] = class2_labels
dataset['class3_labels'] = class3_labels
dataset['class4_labels'] = class4_labels
dataset['class5_labels'] = class5_labels
dataset['class6_labels'] = class6_labels
dataset['class7_labels'] = class7_labels
dataset['class8_labels'] = class8_labels
dataset['class9_labels'] = class9_labels

dataset['test_img'] = valid_data
dataset['test_label'] = valid_labels

print("Creating pickle file ...")
with open("mnist_reshape.pkl", 'wb') as f:
    #pickle.dump(dataset, f, -1)
    pickle.dump(dataset, f, protocol=2)

print("Done!")

#print np.random.choice(dataset['class0'],2,replace=False)
choice = np.random.choice(len(class0), 3, replace=False)
#print np.random.rand() # ここは違う
#print choice
test = class0[choice]
test2 = class0_labels[choice]
print(test.shape)
print(test2.shape)
print(choice)
#print(class5.shape)
#print(class5_labels.shape)
#img_show(class5[0])
#print(class5_labels[0])
