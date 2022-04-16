# Federated_vs_Centralized_Learning
The repository provides both the federated and centralized based learning on MNIST, CIFAR, and FashionMNIST using MLP and CNN based networks.

The implementation of federated learning is based on the details provided in the paper  
Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

The method works on CIFAR10, Fashion MNIST and MNIST datasets for both non-IID and IID cases, respectively.  
In case of opting non-IID, the data may or may not be divided in a uniform manner amongst the users.  

The purpose of this repository is not to provide optimized results, rather it serves as a learning material for learning how to implement vanilla Federated Learning approach.

The Requirements are provided in the requirements.txt file.  

For dataset, there are two options, either the datasets can be downloaded manually and then be arranged in the data folder. However, it is recommended that the method downloads it automatically from torchvision datasets as it will ensure to have error-less placement of folders and data. 

You can try it on your custom dataset, just copy your data into the data directory and write a wrapper function on pytorch dataset class in the utils.py file.  

For centralized learning, write the following command.

python Centralized_learning.py --model=mlp --dataset=mnist --epochs=10 --gpu=1
**if you want to run on CPU, make gpu = 0, you can use mnist, fmnist, or cifar for dataset, and you can use mlp or cnn for model. You can also vary number of epochs, accordingly.

For Federated learning, write the following command.

python Federated_learning.py --model=mlp --dataset=mnist --epochs=10 --iid=1 --gpu==1
**if you want to run on CPU, make gpu = 0, you can use mnist, fmnist, or cifar for dataset, and you can use mlp or cnn for model. You can also vary number of epochs, and iid variable, accordingly.

You can also change default values for  
--lr: 0.01 (default)
--seed: 1 (default)
--num_users: 100 (default)
--frac: 0.1 (default) --> fraction of users to be used for federated updates.
--local_ep: 10 (default) --> local training epochs for each user
--local_bs: 10 (default) --> batch size of local updates for each user
Results on MNIST Dataset
Centralized Learning MLP (MNIST) ![nn_mnist_mlp_10](https://user-images.githubusercontent.com/26203136/163686416-db183fc9-4cd0-408c-974a-c4e7f4b5d4f7.png)

Federated Learning MLP (MNIST) ![FL_mnist_mlp_10_C 0 1 _iid 1 _E 10 _B 10 _loss](https://user-images.githubusercontent.com/26203136/163686450-f36f7722-d215-4908-9d4c-5830304a3484.png)

Centralized Learning CNN (CIFAR10) ![nn_cifar_cnn_10](https://user-images.githubusercontent.com/26203136/163686460-a2b2ec67-a62d-475d-a78b-350162ce0698.png)

Federated Learning CNN (CIFAR10) ![FL_cifar_cnn_10_C 0 1 _iid 1 _E 10 _B 10 _loss](https://user-images.githubusercontent.com/26203136/163686474-9b205479-4314-4550-8686-59495572e886.png)

For further reading you can refer to the following papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
