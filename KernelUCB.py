from packages import *
from sklearn.metrics.pairwise import rbf_kernel
from torch.distributions.multivariate_normal import MultivariateNormal

class KernelUCB:
    def __init__(self, dim, lamdba=1, nu=1):
        self.dim = dim
        self.lamdba = lamdba
        self.nu = nu
        self.x_t = None
        self.r_t = None
        self.history_len = 0
        self.scale = self.lamdba * self.nu
        self.U_t = None
        self.K_t = None
    
    def select(self, context):
        a, f = context.shape
        if self.history_len == 0:
            mu_t = torch.zeros((a,), device=torch.device('cuda'))
            sigma_t = self.scale * torch.ones((a,), device=torch.device('cuda'))
        else:
            c_t = torch.from_numpy(context).float().cuda()
            delta_t = c_t.reshape((a, 1, -1)) - self.x_t.reshape((1, self.history_len, -1))
            k_t = torch.exp(- delta_t.norm(dim=2))
            # print(k_t)
            mu_t = k_t.matmul(self.U_t.matmul(self.r_t))
            sigma_t = self.scale * (torch.ones((a,), device=torch.device('cuda')) - torch.diag(k_t.matmul(self.U_t.matmul(k_t.T))))

        r = mu_t + torch.sqrt(sigma_t)
        return torch.argmax(r).cpu()

    def train(self, context, reward):
        f = context.shape[0]
        if self.history_len < 1000:
            if self.x_t is None:
                self.x_t = torch.from_numpy(context).float().cuda().reshape((1, -1))
                self.r_t = torch.tensor(reward, device=torch.device('cuda'), dtype=torch.float).reshape((-1,))
                self.K_t = torch.tensor(1, device=torch.device('cuda'), dtype=torch.float).reshape((1, 1))
            else:
                c_t = torch.from_numpy(context).float().cuda().reshape((1, -1))
                r_t = torch.tensor(reward, device=torch.device('cuda'), dtype=torch.float).reshape((-1,))
                delta_t = c_t.reshape((1, 1, -1)) - self.x_t.reshape((1, self.history_len, -1))
                self.x_t = torch.cat((self.x_t, c_t), dim=0)
                self.r_t = torch.cat((self.r_t, r_t), dim=0)
                # print(self.x_t.shape, self.r_t.shape, self.K_t.shape)
                k_t = torch.exp(- delta_t.norm(dim=2)).reshape((-1, 1))
                a = torch.cat((k_t.T, torch.ones((1, 1), dtype=torch.float, device=torch.device('cuda'))), dim=1)
                b = torch.cat((self.K_t, k_t), dim=1)
                self.K_t = torch.cat((b, a) , dim=0)
            self.history_len += 1
            self.U_t = torch.inverse(self.K_t + self.lamdba * torch.eye(self.history_len, device=torch.device('cuda'))) 
        return 0
    
    
        