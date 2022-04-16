from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from helper_function.utils import get_dataset
from helper_function.utils import args_parser
from helper_function.utils import test_inference
from helper_function.utils import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar

args = args_parser()

if args.gpu == 1:
    torch.cuda.set_device(0)
device = 'cuda' if args.gpu else 'cpu'

# load datasets
trdata, tsdata, _ = get_dataset(args)

# BUILD MODEL
if args.model == 'cnn':
    if args.dataset == 'mnist':
        glmodel = CNNMnist(args=args)
    elif args.dataset == 'fmnist':
        glmodel = CNNFashion_Mnist(args=args)
    elif args.dataset == 'cifar':
        glmodel = CNNCifar(args=args)
elif args.model == 'mlp':
    imsize = trdata[0][0].shape
    input_len = 1
    for x in imsize:
        input_len *= x
        glmodel = MLP(dim_in=input_len, dim_hidden=64,
                           dim_out=args.num_classes)
else:
    exit('Error: unrecognized model')
    
glmodel.to(device)
glmodel.train()
print(glmodel)

#Hyperparameters
if args.optimizer == 'sgd':
    optim = torch.optim.SGD(glmodel.parameters(), lr=args.lr,
                                momentum=0.5)
elif args.optimizer == 'adam':
    optim = torch.optim.Adam(glmodel.parameters(), lr=args.lr,
                                 weight_decay=1e-4)

trloader = DataLoader(trdata, batch_size=64, shuffle=True)
crit = torch.nn.NLLLoss().to(device)
loss_epoch = []

for ep in tqdm(range(args.epochs)):
    loss_batch = []

    for batch_idx, (img, cat) in enumerate(trloader):
        img, cat = img.to(device), cat.to(device)

        optim.zero_grad()
        out = glmodel(img)
        loss = crit(out, cat)
        loss.backward()
        optim.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep+1, batch_idx * len(img), len(trloader.dataset),
                100. * batch_idx / len(trloader), loss.item()))
        loss_batch.append(loss.item())

    avg_loss = sum(loss_batch)/len(loss_batch)
    print('\nTrain loss:', avg_loss)
    loss_epoch.append(avg_loss)
    
# Plot loss
plt.figure()
plt.plot(range(len(loss_epoch)), loss_epoch)
plt.xlabel('epochs')
plt.ylabel('Train loss')
plt.savefig('results/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                             args.epochs))

# testing
acc_test, loss_test = test_inference(args, glmodel, tsdata)
print('Test on', len(tsdata), 'samples')
print("Test Accuracy: {:.2f}%".format(100*acc_test))
