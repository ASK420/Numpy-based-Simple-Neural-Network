from nn import *
import numpy as np

#Setting inputs and respective outputs
ip = np.array([[0,0],[0,1],[1,0],[1,1]])
op = np.array([[1,0],[0,1],[0,1],[1,0]])

#Creating model by declaring layers
layer1 = Layer(2,10,activation='sigmoid')
layer2 = Layer(10,2,activation='softmax')

#Array for recording errors
error = np.empty((15000,10))

#Main training loop running for 
for i in range(15000):
    index = np.random.choice(op.shape[0],1)
    layer1.train(np.squeeze(ip[index]))
    layer2.train(layer1.out)
    layer2.backprop(getError(layer2,op[[index]],derivative=True))
    layer1.backprop(layer2.delip)
    error[i] = layer2.delip
    #print('Output o:',layer2.out)
    # print('WEights o:',layer2.out)
    #print('Output e:', op[index])
    #print('Input:', np.squeeze(ip[index]))
    #print(getError(layer2,op[index],derivative=True))
    #print(layer2.delip)
    print('Completed {0} steps'.format(i))

#Testing Function that takes ntework input and predicts output 
def forward(num):
    print('ip:',np.squeeze(num))
    layer1.train(np.squeeze(num))
    layer2.train(layer1.out)
    return layer2.out

#Testing loop. 
for i in range(20):
    index = np.random.choice(op.shape[0], 1)
    num = np.squeeze(ip[index])
    ans = forward(num)
    print(ans)
    
#Save the parameters of network and its training error based on layers
layer1.save('layer1')
layer2.save('layer2')
np.save('error.npy',error)
