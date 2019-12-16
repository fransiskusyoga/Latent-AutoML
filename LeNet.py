import torch
import numpy as np
import collections

#Define LeNet5 network
class LeNet5(torch.nn.Module):
	# initilization
	def __init__(self):
		super(LeNet5, self).__init__()
		self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=6,
			kernel_size=5, stride=1, padding=2, bias=True)
		self.maxpool1 =  torch.nn.MaxPool2d(kernel_size=2)
		self.conv2 = torch.nn.Conv2d(in_channels=6,out_channels=16,
			kernel_size=5, stride=1, padding=0, bias=True)
		self.maxpool2 =  torch.nn.MaxPool2d(kernel_size=2)
		self.fc1 = torch.nn.Linear(16*5*5,120)
		self.fc2 = torch.nn.Linear(120, 84)
		self.fc3 = torch.nn.Linear(84, 10)
	# forward propagation
	def forward(self, x):
		x = torch.nn.functional.relu(self.conv1(x))
		self.L1 = self.maxpool1(x)
		x = torch.nn.functional.relu(self.conv2(self.L1))
		self.L2 = self.maxpool2(x)
		x = self.L2.view(-1, 16*5*5)
		self.L3 = torch.nn.functional.relu(self.fc1(x))
		self.L4 = torch.nn.functional.relu(self.fc2(self.L3))
		x = self.fc3(self.L4)
		return(x)
        
#Define LeNet300 network
class LeNet300(torch.nn.Module):
	# initilization
	def __init__(self):
		super(LeNet300, self).__init__()
		self.fc1 = torch.nn.Linear(28*28,300)
		self.fc2 = torch.nn.Linear(300, 100)
		self.fc3 = torch.nn.Linear(100, 10)
	# forward propagation
	def forward(self, x):
		self.L1 = x.view(-1, 28*28)
		self.L2 = torch.nn.functional.relu(self.fc1(self.L1))
		self.L3 = torch.nn.functional.relu(self.fc2(self.L2))
		x = self.fc3(self.L3)
		return(x)
	# forward propagation with dropout
	def forward_dropout(self, x, drop_rate=0.3):
		self.L1 = x.view(-1, 28*28)
		self.L2 = torch.nn.functional.relu(self.fc1(self.L1))
		self.L2 = torch.nn.functional.dropout(self.L2,p=drop_rate)
		self.L3 = torch.nn.functional.relu(self.fc2(self.L2))
		self.L3 = torch.nn.functional.dropout(self.L3,p=drop_rate)
		x = self.fc3(self.L3)
		return(x)

#Define LeNet5 network
class LeNetFC(torch.nn.Module):
    # initilization
    def __init__(self,perceptrons):
        super(LeNetFC, self).__init__()
        self.size = perceptrons.copy()
        self.n_layers = len(perceptrons)-1
        n_prev = perceptrons[0]
        self.layers =  []
        for idx,n_curr in enumerate(perceptrons[1:]):
            self.layers.append(torch.nn.Linear(n_prev,n_curr))
            self.add_module("hidden_layer"+str(idx), self.layers[-1])
            n_prev = n_curr
            
    # forward propagation
    def forward(self, x):
        self.L = x.view(-1, self.size[0])
        for i in range(self.n_layers-1):
            self.L = torch.nn.functional.relu(self.layers[i](self.L))
        self.L = self.layers[-1](self.L)
        return(self.L)# Inherit from Function
        
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class LinearModule(torch.nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
class AutoFC(torch.nn.Module):
    # initilization
    def __init__(self,perceptrons,bias=False,enable_hook=False):
        super(AutoFC, self).__init__()
        self.size = perceptrons.copy()
        self.n_layers = len(perceptrons)-1
        if (self.n_layers<2):
            print('Input should be list with at least 3 integer data,')
            print('At least there should be 1 hidden layer')
            print('minimum requirement :')
            print('  layers = [n_input, n_hidden_1, n_output]')
            return
            
        # Create layer
        self.layers =  []
        n_prev = perceptrons[0]
        for idx,n_curr in enumerate(perceptrons[1:]):
            self.layers.append(torch.nn.Linear(n_prev,n_curr,bias=bias))
            self.add_module("hidden_layer"+str(idx), self.layers[-1])
            n_prev = n_curr
        # initiate jumping connection
        self.jump_layers = [None for i in range(self.n_layers-1)]
        
        # Activate hook
        self.grad_in = [None for i in range(self.n_layers)]
        self.grad_out = [None for i in range(self.n_layers)]
        self.hook_hand = [None for i in range(self.n_layers)]
        if enable_hook:
            self.enable_back_hooks()
    
    # function to add jumping connection
    #   start_brach is a value between 0 to (self.n_layers-2)
    def create_jumper(self,start_branch):
        if self.jump_layers[start_branch] is None:
            self.jump_layers[start_branch] = torch.nn.Linear(
                        self.size[start_branch],self.size[start_branch+2], bias=False)
            self.add_module("jumping_layer"+str(start_branch), self.layers[start_branch])
    
    def load_jump_weight(self,start_branch,weight):
        if self.jump_layers[start_branch] is None:
            self.jump_layers[start_branch] = torch.nn.Linear(
                    self.size[start_branch],self.size[start_branch+2], bias=False)
            self.jump_layers[start_branch] = self.jump_layers[start_branch].to(
                    self.layers[0].weight.device)
            self.add_module("jumping_layer"+str(start_branch), self.layers[start_branch])
            self.jump_layers[start_branch].weight.data = weight.clone().detach()
        else:
            self.jump_layers[start_branch].weight.data += weight
            
            
    
    def remove_jumper(self,start_brach=0):
        print("Not implemented yet")
        
    # All back ward hooks function
    # define callback function
    def back_hook(self, n_of_arry):
        def back_hook_chld(module, grad_input, grad_output):
            self.grad_in[n_of_arry] = grad_input;
            self.grad_out[n_of_arry] = grad_output; 
        return back_hook_chld
    # Enable the hooks
    def enable_back_hooks(self):
        for i in range(self.n_layers):
            self.hook_hand[i] = self.layers[i].register_backward_hook(self.back_hook(i))
    # Remove all hooks
    def disable_back_hooks(self):
        for i in range(self.n_layers):
            self.hook_hand[i].remove()
            self.grad_in[i] = self.grad_out[i] = None
        
    def resize_layer(self):
        print('It is not implemented yet')
        
    # forward propagation with smaller foot print
    def forward(self, x):
        L = [None, None, None]
        #first two layer without any jumping connection
        L[0] = x.view(-1, self.size[0])
        L[1] = torch.nn.functional.relu(self.layers[0](L[0]))
        
        #hidden layer with jumping connection
        for idx,jumper in enumerate(self.jump_layers[:-1]):
            i = idx+2
            L[i%3] = self.layers[idx+1](L[(i+2)%3])
            if jumper is not None:
                L[i%3] += jumper(L[(i+1)%3])
            L[i%3] = torch.nn.functional.relu(L[i%3])
            
        #last layer do not use relu
        i = self.n_layers;
        L[i%3] = self.layers[-1](L[(i+2)%3])
        if self.jump_layers[-1] is not None:
            L[i%3] += self.jump_layers[-1](L[(i+1)%3])
        return(L[i%3])
    
        # forward propagation with bigger foot print
    # you c an use this if you need to access the output of each layer
    def forward_big_fp(self, x):
        self.L = [None for i in range(self.n_layers+1)]
        #first two layer without any jumping connection
        self.L[0] = x.view(-1, self.size[0])
        self.L[1] = torch.nn.functional.relu(self.layers[0](self.L[0]))
        
        #hidden layer with jumping connection
        for idx,jumper in enumerate(self.jump_layers[:-1]):
            i = idx+2
            self.L[i] = self.layers[i-1](self.L[i-1])
            if jumper is not None:
                self.L[i] += jumper(self.L[i-2])
            self.L[i] = torch.nn.functional.relu(self.L[i])
            
        #last layer do not use relu
        self.L[-1] = self.layers[-1](self.L[-2])
        if self.jump_layers[-1] is not None:
            self.L[-1] += self.jump_layers[-1](self.L[-3])
        return(self.L[-1])
    
    # Delete intermediate layer of forward propagation
    def delete_forw_state(self):
        del self.L

class JumperFC():
    #Initialization
    def __init__(self,n, x_2,x_1,x_0,x_m1=None, new_jump=True, bias=False, 
                    acti_func=torch.nn.functional.relu, device='cpu'):
        # save all necesary parameter in local variable
        self.idx      = n 
        self.size   = [x_2,x_1,x_0,x_m1]
        
        #activation function
        self.acti_func = acti_func
        
        # New perceptron in intermediate layer
        #   new perceptron iin intermediate layer need two matrix,.
        #   the first matrix is bet ween input to intermediate the
        #   second layer is between intermadiate to output.
        self.L1   = torch.nn.Linear(x_0,x_1,bias=bias)
        if bias:
            self.L1.bias.data.zero_()
        self.p_L1 = torch.nn.Parameter(torch.ones(x_1))
        self.L2   = torch.nn.Linear(x_1,x_2,bias=False)
        
        # Fisrt jumping connection if required
        self.Lj1 = None
        self.Lj1_exist = False
        if new_jump:
            self.Lj1_exist = True
            self.Lj1  = torch.nn.Linear(x_0,x_2,bias=False)
        
        # Second jumping connection if required
        self.Lj2  = None
        self.Lj2n = None #negative of lj that will decay
        self.Lj2_exist = False
        if x_m1 is not None: 
            self.Lj2_exist = True
            self.Lj2 = torch.nn.Linear(x_m1,x_1, bias=False)
            self.Lj2n= torch.nn.Linear(x_m1,x_1, bias=False)
            self.Lj2n.weight.data = -self.Lj2.weight.data
        
        # sent the variable to the correct device
        self.to_(device)
        
    # remove jumping layer 1 and 2 if wee want tore use the same
    #   Jumper FC for multiple iteration.
    def reset(self):
        self.Lj1 = None
        self.Lj1_exist = False
        self.Lj2 = None
        self.Lj2n= None
        self.Lj2_exist = False
        
    # load tensor to the weight of layer 
    #   if size of input layer do not match the prior layer 
    #   the corespoding linear module will be reinitated
    def load_L1(self, layer, bias = False):
        self.L1 = torch.nn.Linear(layer.size(1),layer.size(0), bias=bias)
        self.L1.weight.data = layer
        if (bias):
            self.L1.bias.data.zero_()
        self.p_L1.data = torch.ones(layer.size(0)).to(self.device) #reset p of activation function
        self.size[1] = layer.size(0)
        self.size[2] = layer.size(1)
    
    def load_L2(self, layer):
        self.L2 = torch.nn.Linear(layer.size(1),layer.size(0), bias=False)
        self.L2.weight.data = layer
        self.size[0] = layer.size(0)
        self.size[1] = layer.size(1)
    
    def load_Lj1(self, layer):
        self.Lj1 = torch.nn.Linear(layer.size(1),layer.size(0), bias=False)
        self.Lj1.weight.data = layer
        self.size[0] = layer.size(0)
        self.size[2] = layer.size(1)
        self.Lj1_exist = True
        
    def init_Lj2(self, n_in, n_out):
        self.Lj2 = torch.nn.Linear(n_in,n_out, bias=False)
        self.Lj2n = torch.nn.Linear(n_in,n_out, bias=False)
        self.Lj2n.weight.data = -self.Lj2.weight.data
        self.size[1] = n_out
        self.size[3] = n_in
        self.Lj2_exist = True
        
    def load_all(self, lyr_L1, lyr_L2, lyr_Lj1=None, Lj2_inp = 0, bias=False):
        self.reset()
        check = (lyr_L1.size(0) == lyr_L2.size(1))
        if lyr_Lj1 is not None:
            check = check and (lyr_L1.size(1) == lyr_Lj1.size(1))
            check = check and (lyr_L2.size(0) == lyr_Lj1.size(0))
        if check:
            self.load_L1(lyr_L1,bias)
            self.load_L2(lyr_L2)
            if lyr_Lj1 is not None:
                self.load_Lj1(lyr_Lj1)
            if (Lj2_inp !=0):
                self.init_Lj2(Lj2_inp,lyr_L1.size(0))
        else:
            print('The size of the inputed layer is not match')
        
    #change the location of tensor
    def to_(self,device):
        self.device = device
        self.L1  = self.L1.to(device)
        self.L2  = self.L2.to(device)
        self.p_L1.data = self.p_L1.data.to(device)
        if (self.Lj1 is not None):
            self.Lj1 = self.Lj1.to(device)
        if (self.Lj2 is not None):
            self.Lj2 = self.Lj2.to(device)
            self.Lj2n = self.Lj2n.to(device)
            
    # calculate the output of first connection
    def calc_main(self,inputs,):  
        H = self.L1(inputs)
        H = self.acti_func(H)*(1-self.p_L1) + H*self.p_L1
        H = self.L2(H)
        if self.Lj1_exist:
            H += self.Lj1(inputs)
        return H
    # calcualte the result of second jumping configuration
    def calc_side(self,inputs):
        if self.Lj2_exist:
            return (self.Lj2(inputs) + self.Lj2n(inputs))
        else:
            return None
    # callculate all contribution
    def calc_all(self, inputs, inputs_prev=None):
        H = self.L1(inputs)
        if (inputs_prev is not None) and self.Lj2_exist:
            H += self.Lj2(inputs_prev) + self.Lj2n(inputs_prev)
        H = self.acti_func(H)*(1-self.p_L1) + H*self.p_L1
        H = self.L2(H)
        if self.Lj1_exist:
            H += self.Lj1(inputs)
        return H
        
    def calc_L1_reg(self, l1_rate):
        return l1_rate*torch.sum(torch.abs(self.p_L1))
        
    def optim_parameters(self):
        all_optim = [{'params' : self.L1.parameters()}, 
                     {'params' : self.L2.parameters()}]
        if (self.Lj1 is not None):
            all_optim.append({'params' : self.Lj1.parameters()})
        if (self.Lj2 is not None):
            all_optim.append({'params' : self.Lj2.parameters()})
            # the negative side have 0 learning rate but always decay
            # all_optim.append({'params' : self.Lj2n.parameters(),'lr' : 0, 'momentum' :0, 'weight_decay' : 0.1})
        return all_optim    
    
    
class AutoGrowFC():
    def __init__(self,perceptrons,bias=False, lr=0.01, l2_decay=0, 
                      l1_decay=0, momentum=0.9, device='cpu'):
        # All variables related to main neural net
        self.bias = bias
        self.main_net = AutoFC(perceptrons,bias=bias,enable_hook=False)
        self.optim_main = torch.optim.SGD(self.main_net.parameters(), lr=lr,
                        momentum=momentum, weight_decay = l2_decay)
        self.loss_func = torch.nn.CrossEntropyLoss()
        
        # All variables related to jump neural net 
        self.jump_net = None
        self.optim_jump = None
        
        #set loader to none
        self.train_loader = None
        self.test_loader = None
        
        #load the network to assigned device
        self.to_(device)
    
    # change the neural network device 
    def to_(self, device):
        self.main_net = self.main_net.to(device)
        if self.jump_net is not None:
            self.jump_net.to_(device)
        self.device = device
        
    # attach training data set to loader
    def insert_train_data(self, train_data, batch_size=64, num_workers=4):
        self.train_loader = torch.utils.data.DataLoader(train_data, 
                    shuffle=True, batch_size=batch_size, num_workers=num_workers)   
        self.n_train = len(train_data)
        self.n_train_batch = batch_size
        
    # attach testing sata set to loader
    def insert_test_data(self, test_data, batch_size=256, num_workers=4):
        self.test_loader = torch.utils.data.DataLoader(test_data, 
                    batch_size=batch_size, num_workers=num_workers)
        self.n_test = len(test_data)
        
    def train(self, train_epooch, with_test=True):
        for epooch in range(train_epooch):
            #  TRAINNING
            train_loss = 0.0
            train_accuracy   = 0.0
            for inputs,labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Make gradients zero for parameters 'W', 'b'
                self.optim_main.zero_grad()    
                # forward, backward pass with parameter update
                outputs = self.main_net(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optim_main.step()
                # calculate accuracy
                _, preds = torch.max(outputs,1)
                train_accuracy += torch.sum(preds == labels.data)
                # calculate loss
                train_loss += loss.item() * inputs.size(0)
            train_accuracy = float(train_accuracy)/self.n_train
            train_loss = train_loss/self.n_train
            
            #Print result from training
            print('Epooch',epooch,
                  '\n  Train loss:',train_loss,', acc:', float(train_accuracy))
            
            # TESTING
            if with_test:
                self.test()
                
                
    def test(self):  
        test_accuracy= 0
        test_loss = 0
        for inputs,labels in self.test_loader:
            with torch.no_grad():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # forward, backward pass with parameter update
                outputs = self.main_net(inputs)
                loss = self.loss_func(outputs, labels)
                # calculate accuracy
                _, preds = torch.max(outputs,1)
                test_accuracy += torch.sum(preds == labels.data)
                # calculate loss
                test_loss += loss.item() * inputs.size(0)
        test_accuracy = float(test_accuracy)/self.n_test
        test_loss = test_loss/self.n_test
        
        #PRINT TREIN AND TESTING RESULT,
        print('  Test  loss:', test_loss,', acc:', float(test_accuracy))  

    def save_model(self, file_name):
        print('not implemented yet')
        
    def load_model(self, file_name):
        print('not implemented yet')
        
    def calc_grad_matrix(self, inspect_point=1000):
        n_inspect_itr = inspect_point//self.n_train_batch + 1
        #calculate the matrix
        self.main_net.enable_back_hooks()
        # Initiate optimation matrix
        m_opt = [torch.zeros(self.main_net.size[i+2],self.main_net.size[i]) for i in range(self.main_net.n_layers-1)]
        w1 = [None for i in range(self.main_net.n_layers-1)]
        w2 = [None for i in range(self.main_net.n_layers-1)]
        wj = [None for i in range(self.main_net.n_layers-1)]
        sc = [None for i in range(self.main_net.n_layers-1)]
        # Calculate optimation matix
        for i in range(n_inspect_itr):
            # Load data
            inputs, labels = next(iter(self.train_loader))
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Calculate the gradient
            self.optim_main.zero_grad()
            outputs = self.main_net.forward_big_fp(inputs)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            # Calculate optimation matrix
            for i in range (self.main_net.n_layers-1):
                m_opt[i] += torch.matmul(self.main_net.grad_out[i+1][0].cpu().t(),
                                self.main_net.L[i].cpu())
        # Calculate new matrix
        for i in range (self.main_net.n_layers-1): 
            w2[i], sc[i], w1[i] = np.linalg.svd(m_opt[i].detach(),full_matrices=False)
            w1[i] = torch.Tensor(w1[i]/w1[i].std())*self.main_net.layers[i].weight.std()
            w2[i] = torch.Tensor(w2[i]/w2[i].std())*self.main_net.layers[i+1].weight.std()
        
        # #remove unecessaary variable
        # self.main_net.delete_forw_state()
        
        # return tupple
        return (w1,w2,sc)
        
    
    def evaluate_grad(self, sc_mats, thrs_magnitude=0.25, verbose=False):
        max_sc = 0
        jump_idx = -1
        for idx,sc in enumerate(sc_mats):
            if sc[0]> max_sc:
                jump_idx = idx
                max_sc = sc[0]
        jump_node = (sc_mats[jump_idx]>=thrs_magnitude).sum()
        if verbose:
            print("Jumping connection from layer", jump_idx,"with",jump_node,"node")
        return (jump_idx,jump_node)
        
    def load_jump_weight(self, w1, w2, branch_idx, n_node, lr=0.01, momentum=0.9, l2_decay=0):
        self.jump_net = JumperFC(branch_idx,self.main_net.size[branch_idx+2],
                n_node,self.main_net.size[branch_idx],new_jump=True, bias=self.bias)
        with torch.no_grad():
            wj = (-torch.matmul(w2[branch_idx][:,0:n_node],w1[branch_idx][0:n_node,:]))
        if (branch_idx != 0) and (self.main_net.jump_layers[branch_idx-1] is not None):
            node_prev = self.main_net.size[branch_idx-1]
        else:
            node_prev = 0
        self.jump_net.load_all(w1[branch_idx][0:n_node,:], w2[branch_idx][:,0:n_node],
                        wj,Lj2_inp=node_prev, bias=self.bias)
        self.jump_net.to_(self.device)
        self.optim_jump = torch.optim.SGD(self.jump_net.optim_parameters(), lr = lr, 
                            momentum=momentum, weight_decay = l2_decay)
    
    def optims_reset(self):
        self.optim_main.state = collections.defaultdict(dict)
        self.optim_jump.state = collections.defaultdict(dict)
    
    def optims_zero_grad(self):
        self.optim_main.zero_grad()
        self.optim_jump.zero_grad()
    
    def optims_step(self):
        self.optim_main.step()
        self.optim_jump.step()
        
    def train_with_jump(self,train_epooch,reset_optim=False,with_test=True):
        if reset_optim:
            self.optim_main.state = collections.defaultdict(dict)
            self.optim_jump.state = collections.defaultdict(dict)
        
        for epooch in range(train_epooch):
            #  TRAINNING
            train_loss = 0.0
            train_accuracy   = 0.0
            for inputs,labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Make gradients zero for parameters 'W', 'b'
                self.optim_main.zero_grad()    
                self.optim_jump.zero_grad()
                # forward
                outputs = self.forward_with_jump(inputs)
                # backward pass with parameter update
                loss = self.loss_func(outputs, labels) 
                loss.backward()
                self.optim_main.step()
                self.optim_jump.step()
                # decay LJ2n only if exist
                self.jump_net.p_L1.data = 0.99*self.jump_net.p_L1.data
                if self.jump_net.Lj2_exist:
                    self.jump_net.Lj2n.weight.data = 0.99*self.jump_net.Lj2n.weight.data
                # calculate accuracy
                _, preds = torch.max(outputs,1)
                train_accuracy += torch.sum(preds == labels.data)
                # calculate loss
                train_loss += loss.item() * inputs.size(0)
            train_accuracy = float(train_accuracy)/self.n_train
            train_loss = train_loss/self.n_train
            
            #PRINT Trrain result
            print('Epooch',epooch,
                  '\n  Train loss:',train_loss,', acc:', float(train_accuracy))
            
            # TESTING
            if with_test:
                self.test_with_jump()
    
    def test_with_jump(self):
        # Last Test
        test_accuracy= 0
        test_loss = 0
        for inputs,labels in self.test_loader:
            with torch.no_grad():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # forward propagarion
                outputs = self.forward_with_jump(inputs)
                loss = self.loss_func(outputs, labels)
                # calculate accuracy
                _, preds = torch.max(outputs,1)
                test_accuracy += torch.sum(preds == labels.data)
                # calculate loss
                test_loss += loss.item() * inputs.size(0)
        test_accuracy = float(test_accuracy)/self.n_test
        test_loss = test_loss/self.n_test
        print('  Test loss:', test_loss,', acc:', float(test_accuracy))
    
    def forward_with_jump(self,x):
        L = [None, None, None, None]
        #first two layer without any jumping connection
        L[0] = x.view(-1, self.main_net.size[0])
        L[1] = torch.nn.functional.relu(self.main_net.layers[0](L[0]))
        
        #hidden layer with jumping connection
        for idx,jumper in enumerate(self.main_net.jump_layers[:-1]):
            i = idx+2
            L[i%4] = self.main_net.layers[idx+1](L[(i+3)%4])
            if jumper is not None:
                L[i%4] += jumper(L[(i+2)%4])
            if (idx == self.jump_net.idx):
                L[i%4] += self.jump_net.calc_all(L[(i+2)%4],L[(i+1)%4])
            L[i%4] = torch.nn.functional.relu(L[i%4])
            
        #last layer do not use relu
        i = self.main_net.n_layers;
        L[i%4] = self.main_net.layers[-1](L[(i+3)%4])
        if self.main_net.jump_layers[-1] is not None:
            L[i%4] += self.main_net.jump_layers[-1](L[(i+2)%4])
        if (i-2 == self.jump_net.idx):
            L[i%4] += self.jump_net.calc_all(L[(i+2)%4],L[(i+1)%4])
        return(L[i%4])
    
    # Set the P in jjumper activation function to 0
    # befor setting the P ot zero w
    def nullify_p_af(self):
        w_to_transf = self.jump_net.p_L1.data*self.jump_net.L2.weight.data
        w_to_transf = torch.matmul(w_to_transf,self.jump_net.L1.weight.data)
        self.jump_net.Lj1.weight.data += w_to_transf
        self.jump_net.L2.weight.data = (1.0-self.jump_net.p_L1.data)*self.jump_net.L2.weight.data
        if self.bias:
            self.main_net.layers[self.jump_net.idx+1].bias.data += torch.matmul(self.jump_net.L2.weight.data,
                    self.jump_net.p_L1.data*self.jump_net.L1.bias.data)
        self.jump_net.p_L1.data.zero_()
    
    # Tranfer werright from jumper net to main net
    def transfer_weight(self):
        self.nullify_p_af()
        branch_idx = self.jump_net.idx
        self.main_net.layers[branch_idx] = concat_linear(
                self.main_net.layers[branch_idx],self.jump_net.L1, 
                common_input=True)
        self.main_net.add_module("hidden_layer"+str(branch_idx),
                self.main_net.layers[branch_idx])
        self.main_net.size[branch_idx+1] = self.main_net.layers[branch_idx].out_features
        self.main_net.layers[branch_idx+1] = concat_linear(
                self.main_net.layers[branch_idx+1],self.jump_net.L2, 
                common_input=False)
        self.main_net.add_module("hidden_layer"+str(branch_idx+1),
                self.main_net.layers[branch_idx+1])
        if self.jump_net.Lj1_exist:
            self.main_net.load_jump_weight(branch_idx,self.jump_net.Lj1.weight.data)
        if self.jump_net.Lj2_exist:
            self.main_net.jump_layers[branch_idx-1] = concat_linear(
                    self.main_net.jump_layers[branch_idx-1],self.jump_net.Lj2, 
                    common_input=True)
            self.main_net.add_module("jumping_layer"+str(branch_idx-1),
                    self.main_net.layers[branch_idx-1])
        if (branch_idx+1 < len(self.main_net.jump_layers)):
            if self.main_net.jump_layers[branch_idx+1] is not None:
                in_feat = self.main_net.size[branch_idx+1] - self.main_net.jump_layers[branch_idx+1].in_features
                out_feat = self.main_net.jump_layers[branch_idx+1].out_features
                zeroLinear = torch.nn.Linear(in_feat, out_feat, bias=False)
                zeroLinear.weight.data.zero_()
                self.main_net.jump_layers[branch_idx+1] = concat_linear(
                        self.main_net.jump_layers[branch_idx-1],zeroLinear,
                        common_input = False)
                    
            
# Concatenate two layer into one bigger layer
#   parameter:
#   - common_input: if th enetwork that gonna be concatenated share 
#     common input. It can be indicated by the number of input featres.
#     if the number of input features is same you must set this as true.
#.    if this parameters is set to False then the concatenation will proceed
#     as common output.
#   - linear_1    : this module is module that will occupy weight and bias
#     in lower index. information about bias and device location is fetched from
#     module.
def concat_linear(linear_1, linear_2, common_input=True):
    bias = linear_1.bias is not None
    device = linear_1.weight.device
    if common_input:
        input_sz = linear_1.weight.size(1)
        output_sz1 = linear_1.weight.size(0)
        output_sz2 = linear_2.weight.size(0)
        output_sz =  output_sz1 + output_sz2
        linear_cat = torch.nn.Linear(input_sz,output_sz,bias=bias)
        linear_cat = linear_cat.to(device)
        linear_cat.weight.data[0:output_sz1,:] = linear_1.weight.data
        linear_cat.weight.data[output_sz1:output_sz,:] = linear_2.weight.data
        if bias:
            linear_cat.bias.data[0:output_sz1] = linear_1.bias.data
            linear_cat.bias.data[output_sz1:output_sz] = linear_2.bias.data
    else:
        input_sz1 = linear_1.weight.size(1)
        input_sz2 = linear_2.weight.size(1)
        input_sz = input_sz1 + input_sz2
        output_sz = linear_1.weight.size(0)
        linear_cat = torch.nn.Linear(input_sz,output_sz,bias=bias)
        linear_cat = linear_cat.to(device)
        linear_cat.weight.data[:,0:input_sz1] = linear_1.weight.data
        linear_cat.weight.data[:,input_sz1:input_sz] = linear_2.weight.data
        if bias:
            linear_cat.bias.data = linear_1.bias.data.clone()
    return linear_cat
    
