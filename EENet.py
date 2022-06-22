from packages import *
from EENetClass import Exploitation, Exploration, Decision_maker


class EE_Net:
    def __init__(self, dim, n_arm, lr_1 = 0.01, lr_2 = 0.01, lr_3 = 0.01, pooling_step_size = 1000, hidden=100):
        #Network 1
        self.f_1 = Exploitation(dim, n_arm, lr_1, pooling_step_size)
        
        # number of dimensions of aggregated for f_2  
        f_2_dim = self.f_1.total_param // pooling_step_size + 1
        #Network 2
        self.f_2 = Exploration(f_2_dim , 100, lr_2)
        
        #Network 3
        self.f_3 = Decision_maker(2, 20, lr_3) 
        
        self.arm_select = 0
        
        self.exploit_scores = []
        self.explore_scores = []
        self.ee_scores = []
        self.grad_list = []

        
    def predict(self, context, t):  # np.array 
        self.exploit_scores, self.grad_list = self.f_1.output_and_gradient(context)
        self.explore_scores = self.f_2.output(self.grad_list)
    
        self.ee_scores = np.concatenate((self.exploit_scores, self.explore_scores), axis=1)
        
        if t < 500:
            # linear decision maker
            suml = self.exploit_scores + self.exploit_scores
            self.arm_select = np.argmax(suml)
        else:
            # neural decision maker
            self.arm_select = self.f_3.select(self.ee_scores)
        return self.arm_select
    
    def update(self, context, r_1, t):
        # update exploitation network
        self.f_1.update(context[self.arm_select], r_1)
        
        # update exploration network
        f_1_predict = self.exploit_scores[self.arm_select][0]
        r_2 = r_1 - f_1_predict
        self.f_2.update(self.grad_list[self.arm_select], r_2)
        
        # add additional samples to exploration net when the selected arm is not optimal
        if t < 500:
            if r_1 == 0:
                index = 0
                for i in self.grad_list:
                    c = (1/np.sqrt(t+10))
                    if index != self.arm_select:
                        self.f_2.update(i, c)
                    index += 1
        
        # update decision maker
        self.f_3.update(self.ee_scores[self.arm_select], float(r_1))

    def train(self):
        #train networks
        loss_1 = self.f_1.train()
        loss_2 = self.f_2.train()
        loss_3 = self.f_3.train()
        return loss_1, loss_2, loss_3
    