import torch
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device = torch.device(device)
print(device)

import numpy as np
import mat73
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold

trans = transforms.ToTensor()

train_losses = []
train_acc = []
train_data = []
val_data = []
test_data = []

train_stage1 = 0
train_stage2 = 0
train_stage3 = 0
train_stage4 = 0

val_stage1 = 0
val_stage2 = 0
val_stage3 = 0
val_stage4 = 0

test_stage1 = 0
test_stage2 = 0
test_stage3 = 0
test_stage4 = 0

for j in range(1, 33):  # 1~32 -> train/val
    path = "/home/user1/다운로드/cwtnew_0418/CWTData_s" + str(j) + ".mat"
    data = mat73.loadmat(path)

    cwtDataInfo = data.get("CWTData")
    Power = cwtDataInfo.get("Power")
    stg = cwtDataInfo.get("stg")

    timelen = int(Power.shape[1] / 600)
    print("sleep stage num: ", timelen)

    for i in range(1, timelen + 1):
        start_time = (i - 1) * 600
        end_time = i * 600
        b = Power[0:149, start_time: end_time]
        train_Data = np.array(b, dtype=np.float32)

        # N1+N2
        if stg[i - 1] == 3 or stg[i - 1] == 4:
            a = (trans(train_Data), int(0))
            train_data.append(a)
            train_stage1 += 1
        # N3
        elif stg[i - 1] == 5:
            a = (trans(train_Data), int(1))
            train_data.append(a)
            train_stage2 += 1
        # wake
        elif stg[i - 1] == 6:
            a = (trans(train_Data), int(2))
            train_data.append(a)
            train_stage3 += 1
        # REM
        elif stg[i - 1] == 7:
            a = (trans(train_Data), int(3))
            train_data.append(a)
            train_stage4 += 1

#test data
for j in range(36, 39):  # 36, 38 -> test
    if j == 37: continue
    path = "/home/user1/다운로드/cwtnew_0418/CWTData_s" + str(j) + ".mat"
    data = mat73.loadmat(path)
    print(path)

    cwtDataInfo = data.get("CWTData")
    Power = cwtDataInfo.get("Power")
    stg = cwtDataInfo.get("stg")

    timelen = int(Power.shape[1] / 600)
    trainData = [[[0] * 600] * 149] * timelen
    print("sleep stage num: ", timelen)

    for i in range(1, timelen + 1):
        start_time = (i - 1) * 600
        end_time = i * 600
        b = Power[0:149, start_time: end_time]
        train_Data = np.array(b, dtype=np.float32)

        # N1+N2
        if stg[i - 1] == 3 or stg[i - 1] == 4:
            a = (trans(train_Data), int(0))
            test_data.append(a)
            test_stage1 += 1
        # N3
        elif stg[i - 1] == 5:
            a = (trans(train_Data), int(1))
            test_data.append(a)
            test_stage2 += 1
        # wake
        elif stg[i - 1] == 6:
            a = (trans(train_Data), int(2))
            test_data.append(a)
            test_stage3 += 1
        # REM
        elif stg[i - 1] == 7:
            a = (trans(train_Data), int(3))
            test_data.append(a)
            test_stage4 += 1

print(len(train_data))
print(len(test_data))

# check stages
print('\ntrain 1: %d,   2: %d,  3: %d, 4: %d' % (train_stage1, train_stage2, train_stage3, train_stage4))
print('test 1: %d,   2: %d,  3: %d, 4: %d' % (test_stage1, test_stage2, test_stage3, test_stage4))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.fc1 = nn.Linear(149504, 500)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

cnn = CNN()
cnn = cnn.cuda()

# class imbalance 해결을 위한 weight 부여
weight_stage1 = 0.2
weight_stage2 = 1.6
weight_stage3 = 0.2
weight_stage4 = 1.0

normedWeights = [weight_stage1, weight_stage2, weight_stage3, weight_stage4]
normedWeights = torch.FloatTensor(normedWeights).to(device)
criterion = torch.nn.CrossEntropyLoss(normedWeights)
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

# Define k-fold cross-validation
fold_count = 5
kfold = KFold(n_splits=fold_count, shuffle=False)

EPOCHS = 15
train_accu = [[] for _ in range(EPOCHS)]
train_losses = [[] for _ in range(EPOCHS)]
eval_losses = [[] for _ in range(EPOCHS)]
eval_accu = [[] for _ in range(EPOCHS)]
best = (0, -1, 1e9, []) # (fold, valid_idx, loss, model.state)



# append result by fold -> by epoch
def add_fold_result(epoch, d, loss):
    if epoch in d:
        d[epoch].append(loss)
    else:
        d[epoch] = [loss]
    return d

def avg_group_by_fold(losses):
   avg_value = []
   for f in range(len(losses)):
       if not losses[f]: break
       avg_value.append(np.array(losses[f]).mean())
   #print(losses)
   return avg_value
   #return [np.array(losses[f_key]).mean() for f_key in losses]


# for tracking by fold
train_losses_by_fold = [[] for _ in range(fold_count)]
valid_losses_by_fold = [[] for _ in range(fold_count)]

for epoch in range(EPOCHS):
    running_loss = 0
    correct = 0
    total = 0

    for fold, (train_index, valid_index) in enumerate(kfold.split(train_data)):
        # Split dataset and loader
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)  # index 생성
        val_subsampler = torch.utils.data.SubsetRandomSampler(valid_index)  # index 생성
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, sampler=val_subsampler)

        N1N2 = 0
        N3 = 0
        wake = 0
        REM = 0
        if epoch == 0:
            print("train len : %d" % len(train_index))
            print("valid len :%d" % len(valid_index))

        running_loss = 0.0
        correct = 0
        total = 0

        # Train
        cnn.train()
        for index, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()  # 기울기 초기화
            output = cnn(data)
            loss = criterion(output, target)
            loss.backward()  # 역전파
            optimizer.step()

            running_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()



        train_loss = running_loss / len(train_index)
        train_losses_by_fold[fold].append(train_loss)
        accu = 100. * correct / total
        train_accu[epoch].append(accu)
        train_losses[epoch].append(train_loss)

        running_loss = 0
        correct = 0
        total = 0

        # valid loss (just for check)
        with torch.no_grad():
            for index, (data, target) in enumerate(valid_loader):
                data, target = data.cuda(), target.cuda()
                cnn.eval()

                output = cnn(data)

                loss = criterion(output, target)
                running_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total += target.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()

            val_loss = running_loss / len(valid_index)
            valid_losses_by_fold[fold].append(val_loss)
            accu = 100. * correct / total
            eval_losses[epoch].append(val_loss)
            eval_accu[epoch].append(accu)

        print('[%d] fold=%d, train loss: %.6f, valid loss: %.6f' % (epoch + 1, fold, train_loss, val_loss))

for epoch in range(1):
    cnn.eval()  # test case 학습 방지
    running_loss = 0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = cnn(data)

            loss = criterion(output, target)
            running_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)

            target = target.data.cpu().numpy()
            y_true.extend(target)

    test_loss = running_loss / len(test_loader.dataset)
    accu = 100. * correct / total
    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))

    print(np.shape(y_true))
    print(np.shape(y_pred))

    print(y_true[1])
    print(y_pred[1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(cm)

    accu_0 = 100. * (cm[0][0] / test_stage1)
    accu_1 = 100. * (cm[1][1] / test_stage2)
    accu_2 = 100. * (cm[2][2] / test_stage3)
    accu_3 = 100. * (cm[3][3] / test_stage4)

    print('label 0 accuracy: %.3f' % (accu_0))
    print('label 1 accuracy: %.3f' % (accu_1))
    print('label 2 accuracy: %.3f' % (accu_2))
    print('label 3 accuracy: %.3f' % (accu_3))


plt.figure(figsize=(10, 5))
plt.plot(avg_group_by_fold(train_losses), label="train loss")
plt.plot(avg_group_by_fold(eval_losses), label="test loss")
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Losses')
plt.show()

plt.figure(figsize=(25, 5))
for fold in range(fold_count):
    plt.subplot(1, fold_count, fold + 1)
    plt.plot(train_losses_by_fold[fold], label="train loss")
    plt.plot(valid_losses_by_fold[fold], label="valid loss")
    plt.title(f'Loss (fold={fold})')
    plt.xlabel("epoch")
    plt.axvline(best[0], color='red')
    plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(avg_group_by_fold(train_accu), label="train accuracy")
plt.plot(avg_group_by_fold(eval_accu), label="test accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['Train', 'Valid'])
plt.title("Train vs Valid Accuracy")
plt.show()