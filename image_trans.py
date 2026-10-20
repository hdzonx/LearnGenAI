import torch
import sys
print("virtual env. active?: ",sys.prefix != sys.base_prefix)  # True = venv ativo   
import torch.nn as nn
import torchvision 
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
transform = T.Compose([T.ToTensor(),
                        T.Normalize([0.5],[0.5])
                       ])

train_set=torchvision.datasets.FashionMNIST(
    root=".",
    train=True,
    download=True,
    transform=transform
)

test_set=torchvision.datasets.FashionMNIST(
    root=".",
    train=False,
    download=True,
    transform=transform
)

#print(train_set[0])

text_labels=['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

plt.figure(dpi=300, figsize=(8,4))
for i in range(24):
    ax = plt.subplot(3, 8, i+1)
    img = train_set[i][0]
    img = img/2+0.5
    img = img.reshape(28,28)
    plt.imshow(img, cmap="binary")
    plt.axis('off')
    plt.title(text_labels[train_set[i][1]],
fontsize=4)
plt.show()

#Iterates through each item x in the original train_set.
#If the second element of that item is 0 or 9, it is included
#in the new binary_train_set.
binary_train_set = [x for x in train_set if x[1] in [0,9]]
binary_test_set = [x for x in test_set if x[1] in [0,9]]

#Creating batchs for training and test
batch_size = 64
binary_train_loader = torch.utils.data.DataLoader(
    binary_train_set,
    batch_size=batch_size,
    shuffle=True
) 

binary_test_loader = torch.utils.data.DataLoader(
    binary_test_set,
    batch_size=batch_size,
    shuffle=True
)

#creating a binary classification model
device = "cpu"

binary_model = nn.Sequential(
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,32),
    nn.ReLU(),
    nn.Dropout(p=0.25),
    nn.Linear(32,1),
).to(device)

lr = 0.001
optimizer = torch.optim.Adam(binary_model.parameters(), lr=lr)
loss_fn = nn.BCEWithLogitsLoss()


#Training a binary classification model
for i in range(50):    #A
    tloss=0
    for n,(imgs,labels) in enumerate(binary_train_loader):    #B
        imgs=imgs.reshape(-1,28*28)    #C
        imgs=imgs.to(device)
        labels=torch.FloatTensor(\
          [x if x==0 else 1 for x in labels])    #D
        labels=labels.reshape(-1,1).to(device)
        preds=binary_model(imgs)    
        loss=loss_fn(preds,labels)    #E
        optimizer.zero_grad()
        loss.backward()    #F
        optimizer.step()
        tloss+=loss
    tloss=tloss/n
    print(f"at epoch {i}, loss is {tloss}")


#Calculation the accuracy of predictions
results =[]
for imgs, labels in binary_test_loader:
    imgs = imgs.reshape(-1,28*28).to(device)
    labels = (labels/9).reshape(-1,1).to(device)
    preds = binary_model(imgs)
    pred10 = torch.where(preds>0.5,1,0)
    correct = (pred10==labels)
    results.append(correct.detach().cpu()\
                   .numpy().mean())

accuracy = np.array(results).mean()               
print(f"the accuracy of the predictons is {accuracy}")




