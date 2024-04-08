import torch, os, time, random, generator, discri, classify, utils
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
from art.attacks.inference import model_inversion 
from art.estimators.classification import KerasClassifier
from art.attacks.inference import model_inversion 
from art.estimators.classification import KerasClassifier
import numpy as np

device = "cuda"
num_classes = 10
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)
root_path = "./Attack/Kmnist_based"

save_img_dir = os.path.join(root_path, "Final_imgs_Mnist")

def inversion(G, D, T, iden, lr=1e-2, momentum=0.9, lamda=4000, iter_times=30, clip_range=1):
        iden = iden.view(-1).long()
        criterion = nn.CrossEntropyLoss()
        bs = iden.shape[0]
        
        G.eval()
        D.eval()
        T.eval()
        

        max_score = torch.zeros(bs)
        max_iden = torch.zeros(bs)
        z_hat = torch.zeros(bs, 100)
        
        for random_seed in range(1000):
                tf = time.time()
                
                torch.manual_seed(random_seed) 
                torch.manual_seed(random_seed) 
                np.random.seed(random_seed) 
                random.seed(random_seed)

                z = torch.randn(bs, 100).float()
                z.requires_grad = True
                v = torch.zeros(bs, 100).float()
                        
                for i in range(iter_times):
                        fake = G(z)
                        label = D(fake)
                        out = T(fake)[-1]
                        
                        
                        
                        if z.grad is not None:
                                z.grad.data.zero_()

                        Prior_Loss = - label.mean()
                        Iden_Loss = criterion(out, iden)
                        
                        Total_Loss = Prior_Loss + lamda * Iden_Loss

                        Total_Loss.backward()
                        
                        v_prev = v.clone()
                        gradient = z.grad.data
                        v = momentum * v - lr * gradient
                        z = z + ( - momentum * v_prev + (1 + momentum) * v)
                        z = torch.clamp(z.detach(), -clip_range, clip_range).float()
                        z.requires_grad = True

                        Prior_Loss_val = Prior_Loss.item()
                        Iden_Loss_val = Iden_Loss.item()

                        #if (i+1) % 300 == 0:
                        fake_img = G(z.detach())
                                
                        eval_prob = T(fake_img)[-1]
                        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                        #save_img_dir= os.path.join(root_path, "Kmnist")
                        #save_img_dir = os.path.join(temp_dir, str(eval_iden.item()))
                                
                        if not os.path.exists(save_img_dir):
                                os.mkdir(save_img_dir)
                        acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
                        utils.save_tensor_images(fake_img.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(random_seed+i)), nrow = 8)
                                

                        #print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
                        
                fake = G(z)
                score = T(fake)[-1]
                
                eval_iden = torch.argmax(score, dim=1).view(-1)
                
                cnt = 0
                for i in range(bs):
                        gt = iden[i].item()
                        if score[i, gt].item() > max_score[i].item():
                                max_score[i] = score[i, gt]
                                max_iden[i] = eval_iden[i]
                                z_hat[i, :] = z[i, :]
                        if eval_iden[i].item() == gt:
                                cnt += 1
                        
                interval = time.time() - tf
                print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))

        correct = 0
        for i in range(bs):
                gt = iden[i].item()
                if max_iden[i].item() == gt:
                        correct += 1
        
        acc = correct * 1.0 / bs
        print("Acc:{:.2f}".format(acc))

if __name__ == "__main__":
        target_path = "./Attack/attack_models/KMNIST_T.tar"
        T = classify.CNN(10)
        ckp_T = torch.load(target_path)['state_dict']
        utils.load_my_state_dict(T, ckp_T)

        my_attack=model_inversion.MIFace(ckp_T)

       

        g_path = "./Attack/attack_models/MNIST_G_Kmnist.tar"
        G = generator.GeneratorMNIST()
        G = nn.DataParallel(G)
        ckp_G = torch.load(g_path)['state_dict']
        utils.load_my_state_dict(G, ckp_G)

        d_path = "./Attack/attack_models/MNIST_D_Kmnist.tar"
        D = discri.DGWGAN32()
        D = nn.DataParallel(D)
        ckp_D = torch.load(d_path)['state_dict']
        utils.load_my_state_dict(D, ckp_D)

        iden = torch.zeros(1)
        for i in range(1):
            iden[i] = 6

        inversion(G, D, T, iden)
        
