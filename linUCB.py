from packages import *
from load_data import load_yelp, Bandit_multi, load_mnist_1d, load_movielen, load_disin, load_disin_20


class Linearucb:
    # Brute-force Linear TS with full inverse
    def __init__(self, dim, lamdba=0.001, nu=1, style='ts'):
        self.dim = dim
        self.U = lamdba * np.eye(dim)
        self.Uinv = 1 / lamdba * np.eye(dim)
        self.nu = nu
        self.jr = np.zeros((dim, ))
        self.mu = np.zeros((dim, ))
        self.lamdba = lamdba
        self.style = style

    def select(self, context):
        sig = np.diag(np.matmul(np.matmul(context, self.Uinv), context.T))
        r = np.dot(context, self.mu) + np.sqrt(self.lamdba * self.nu) * sig
        return np.argmax(r), np.linalg.norm(self.mu), np.mean(sig), np.mean(r)
        
    
    def train(self, context, reward):
        self.jr += reward * context
        self.U += np.matmul(context.reshape((-1, 1)), context.reshape((1, -1)))
        # fast inverse for symmetric matrix
        zz , _ = sp.linalg.lapack.dpotrf(self.U, False, False)
        Linv, _ = sp.linalg.lapack.dpotri(zz)
        self.Uinv = np.triu(Linv) + np.triu(Linv, k=1).T
        self.mu = np.dot(self.Uinv, self.jr)
        return 0
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LinUCB')
    parser.add_argument('--dataset', default='mnist', type=str, help='yelp, disin, mnist, movielens')
    args = parser.parse_args()
   

    for i in range(10):
        if args.dataset == "yelp":
            b = load_yelp()
        elif args.dataset == "disin":
            b = load_disin()
        elif args.dataset == "mnist":
            b = load_mnist_1d()
        elif args.dataset == "movielens":
            b = load_movielen()
        else:
            b = Bandit_multi(dataset, 1, 0)
        
        summ = 0
        regrets = []
        lin = Linearucb(b.dim)
        for t in range(10000):
            context, rwd = b.step()
            arm_select, c, d, e = lin.select(context)
            r = rwd[arm_select]
            reg = np.max(rwd) - r
            summ+=reg
            regrets.append(summ)
            if t > 2000:
                if t%100 == 0:
                    lin.train(context[arm_select],r)
            else:
                if t%10 == 0:
                    lin.train(context[arm_select],r)

            if t % 50 == 0:
                print('{}: {:}'.format(t, summ))

        print("round:", t, summ)

    
    
    
    