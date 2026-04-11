
import numpy as np
import copy

class FeedforwardNeuralNetwork:
    def __init__(self,
                 num_layers,
                 num_features,
                 num_hidden_units:dict,
                 num_classes,
                 activation_func,
                 activation_func_prime,
                 theta=None):

        self.activation_func=activation_func
        self.activation_func_prime=activation_func_prime
        self.num_layers=num_layers

        self.num_features=num_features
        self.num_hidden_units=num_hidden_units
        self.num_classes=num_classes

        if theta==None:
            self.theta=self.generate_random_theta()
        else:
            self.theta=theta

        self.output=None

    def generate_random_theta(self):

        theta=[np.random.randn(self.num_unit(i+1), self.num_unit(i)+1) *0.1
                for i in range(self.num_layers - 1)]
        return theta

    def num_unit(self,l):

        if l==0:
            return self.num_features
        
        elif l==self.num_layers-1:
            return self.num_classes
        
        elif 0<l and l <self.num_layers-1:
            return self.num_hidden_units[l]
        
        else:
            raise IndexError
        
    def feed_forward(self,X,theta):

        z_list=list()
        activations_list=list()

        for l in range(self.num_layers):

            if l==0:
                z_list.append(X)
                activations_list.append(X)

            else:

                pre_activation_no_bias=activations_list[l-1]
                pre_activation = np.hstack([np.ones((pre_activation_no_bias.shape[0],1)), pre_activation_no_bias])

                curr_z=pre_activation.dot(theta[l-1].T)
                z_list.append(curr_z)

                curr_activation=self.activation_func(curr_z)

                activations_list.append(curr_activation)

        return activations_list,z_list
        

    def cost_function(self,h_theta,Y,theta):

        m=Y.shape[0]
        
        landa=1e-3

        if self.activation_func.__name__== 'sigmoid':
            eps = 1e-12
            h = np.clip(h_theta, eps, 1 - eps)
            
            cost=-np.sum(Y*(np.log(h)) + (1-Y)*(np.log(1-h)))/ m

            regulization=0
            for weight in theta:

                y=np.power(weight[:,1:],2)
                regulization+=np.sum(y)

            regulization = (landa/(2*m)) * regulization
            cost += regulization
        
        return cost
    

    def back_prop(self,Y,activations,z_list,theta):
        landa=1e-3
        g_prime=self.activation_func_prime
        num_layers=self.num_layers

        delta=[None]*num_layers
        delta[num_layers-1]=activations[-1]-Y

        for l in range(num_layers-2,0,-1):
            theta_l=theta[l]
            delta[l]=((delta[l+1]).dot( theta_l[:,1:]))* g_prime(z_list[l])


        delta_weight_sum=[None]* (num_layers-1)
        for l in range(0,num_layers-1):
            delta_weight_sum[l]=np.zeros((self.num_unit(l+1),self.num_unit(l)+1))

        m = Y.shape[0]
        for l in range(0,num_layers-1):
            delta_next=delta[l+1]
            delta_weight_sum[l]+=(delta_next.T).dot(np.hstack((np.ones((m,1)),activations[l])))
 
        derivatives=[None]*(num_layers-1)
        for l in range(0,num_layers-1):
            derivatives[l]=(delta_weight_sum[l])/m
            derivatives[l][:,1:] += (landa/m )* theta[l][:,1:]

        return derivatives
        
    def gradient_checking(self,X,Y,derivatives,theta):

        gradient = [np.zeros_like(t) for t in theta]
        epsilon = 1e-5  


        num_theta=[]
        for l in range(len(theta)):
            num_theta.append((theta[l].shape[0]) * (theta[l].shape[1]))

        for L in range(len(theta)):
            for _ in range(num_theta[L]//33):

                i=np.random.randint(0,theta[L].shape[0])
                j=np.random.randint(0,theta[L].shape[1])

                theta_plus = copy.deepcopy(theta)
                theta_minus = copy.deepcopy(theta)

                theta_plus[L][i,j]+=epsilon
                theta_minus[L][i,j]-=epsilon

                h_plus,_=self.feed_forward(X,theta_plus)
                h_minus,_=self.feed_forward(X,theta_minus)

                gradient[L][i,j]=(self.cost_function(h_plus[-1],Y,theta_plus)
                                    -self.cost_function(h_minus[-1],Y,theta_minus))/(2*epsilon)

                diff = abs( gradient[L][i,j] - derivatives[L][i,j] )
                if diff > 1e-6:
                    print((i,j)," at Layer",L,"FAILED. diff =", diff)
                    return False
        print("GRADIENT CHECK PASSED")
        return True

    def gradient_descent(self,derivatives,theta,alpha):

        for L in range(len(theta)):
            theta[L]-=alpha*derivatives[L]

        return theta

    def train(self,X,Y,num_epoch):
        costs=[]
        alpha=0.005
        for _ in range(num_epoch):
            activations, z_list=self.feed_forward(X,self.theta)
            h_theta=activations[-1]
            cost=self.cost_function(h_theta,Y,self.theta)
            costs.append(cost)
            derivatives=self.back_prop(Y,activations,z_list,self.theta)

            if _==0:
                Flag=self.gradient_checking(X,Y,derivatives,self.theta)
                if Flag==False:
                    raise RuntimeError
            self.theta=self.gradient_descent(derivatives,self.theta,alpha)
        return costs
    
    def predict(self,X):
        activations,_=self.feed_forward(X,self.theta)
        self.output=(activations[-1]>0.5).astype(int)
        return self.output
    

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    A = sigmoid(z)
    return A * (1 - A)
