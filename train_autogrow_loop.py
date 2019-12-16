import torch
import torchvision
import numpy as np
from decimal import Decimal
from LeNet import AutoGrowFC

if __name__ == '__main__':    
    #All tunable parameter
    # learning tuned parameter
    momen_rate = 0.9
    learn_rate = 0.01
    n_train_batch = 64
    n_test_batch = 256
    n_inspect_batch = 1000
    main_net_epooch = 1
    whole_net_epooch = 2
    grad_thres = 0.25
    decay_rate = 0 
    base_drop_rate = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_lyrs = [784,20,20,10]
    
    #Download MNIST data
    # preprocessing configuration for downloaded MNIST data
    transformImg = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),               #convert image to tensor
                        torchvision.transforms.Normalize([0.5],[0.5])])  #normalize in tensor format
    # load train set and test set
    train_set = torchvision.datasets.MNIST(root='../data',
                    train=True, download=True, transform=transformImg)
    test_set = torchvision.datasets.MNIST(root='../data',
                    train=False, download=True, transform=transformImg)
    
    # Pruning and retraining
    AutoGrowNet = AutoGrowFC(n_lyrs, lr=learn_rate, 
            momentum=momen_rate, l2_decay=decay_rate, bias=True)
    AutoGrowNet.to_(device)
    
    # Put the data to the handler
    AutoGrowNet.insert_train_data(train_set, batch_size=n_train_batch, num_workers=4)
    AutoGrowNet.insert_test_data(test_set, batch_size=n_test_batch, num_workers=4)
    
    for i in range(20):
        print("GROWING PROCESS", i, "size now",AutoGrowNet.main_net.size)
        
        # Train the main network withuot (non growing process)
        AutoGrowNet.train(main_net_epooch, with_test=True)
        
        # Create jumping connection
        w1,w2,sc = AutoGrowNet.calc_grad_matrix(n_inspect_batch)
        j_idx, j_node = AutoGrowNet.evaluate_grad(sc,grad_thres,verbose=True)
        AutoGrowNet.load_jump_weight(w1, w2, j_idx, j_node,lr=learn_rate, momentum=momen_rate)
        
        print("Test result after jump layer created")
        # Test jumper consistency
        AutoGrowNet.test_with_jump()
        
        # Retrain with jumper
        AutoGrowNet.train_with_jump(whole_net_epooch,reset_optim=True)
        
        # transver weight from jumper to main network
        AutoGrowNet.transfer_weight()
        print("Test result after weight transfered")
        AutoGrowNet.test()