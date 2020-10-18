# coding =utf-8
import math
import os
import shutil
from models.U_Nets import *
from models import DeepLab
from data.dataset import Dataset
from torch.utils.data import DataLoader

from opt import opt
import numpy as np
import random
from validation_utils import Eval, FWIOU
from predict import predict
from torch.optim.lr_scheduler import StepLR
from torch import optim
from adams import AdamW,Nadam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速
    torch.backends.cudnn.enabled = True
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)

def save_model(model, model_save_path, current_epoch):
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    save_path = "net_" + str(current_epoch) + ".pth"
    save_path = os.path.join(model_save_path,save_path)
    torch.save(model.state_dict(), save_path)

end_lr = 1e-6
def get_base_lr(step,total_step):
    base_lr = 1e-3
    if step < 0.5*total_step:
        base_lr =1e-3
    elif step < 0.9*total_step:
        base_lr =0.5 * 1e-3
    else:
        base_lr = 0.5  * ((total_step-step)/(0.1*total_step)) * 1e-3
    return base_lr

def lr_function(step, warm_step, total_step):
    base_lr = get_base_lr(step,total_step)
    if step < warm_step:
        lr = step / warm_step * base_lr
    else:
        lr = end_lr+0.5* (base_lr - end_lr ) * ( 1 + math.cos((step - warm_step)/(total_step- step) * math.pi))
    return lr

def adjust_lr(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


'''def train1(epoch, train_dataLoader, optimizer, criterion, writer, global_step, WARMUP_STEP, TOTAL_STEP):
    print("*******  begin train  *******")
    model.train()
    all_loss = 0.0
    for batch_id, (data, target_img) in enumerate(train_dataLoader):
        global_step += 1
        lr = lr_function(global_step, WARMUP_STEP, TOTAL_STEP)
        adjust_lr(optimizer, lr)
        data, target_img = data.to(opt.device), target_img.to(opt.device)
        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out, target_img)

        current_loss = loss.item()
        all_loss += current_loss
        avg_loss = all_loss / (batch_id + 1)
        if batch_id % 100==0:
            print("[batch_id   {},   avg_loss {:.8f}   ,  lr:{:.8f}] ".format(batch_id,avg_loss, lr))
        loss.backward()
        optimizer.step()
    print("[***********************       all  avg_loss {:.8f}   ,  lr:{:.8f}]".format(avg_loss,lr))'''

def train(epoch, train_dataLoader, optimizer, criterion, writer, global_step, WARMUP_STEP, TOTAL_STEP):
    print("*******  begin train  *******")
    model.train()
    all_loss = 0.0
    with tqdm(total= len(train_dataLoader), unit='batch') as pbar:
        for batch_id, (data, target_img) in enumerate(train_dataLoader):
            global_step+=1
            lr = lr_function(global_step, WARMUP_STEP, TOTAL_STEP)
            adjust_lr(optimizer,lr)
            data,target_img = data.to(opt.device),  target_img.to(opt.device)
            optimizer.zero_grad()
            out = model(data)

            loss = criterion(out,target_img)

            current_loss = loss.item()
            all_loss += current_loss
            avg_loss  = all_loss/(batch_id+1)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            postfix_message={"cur_loss":"%.6f"%(current_loss) , "avg_loss ":"%.6f"%(avg_loss) , "lr":"%.8f"%(lr)}
            pbar.set_postfix(log = postfix_message)

            writer.add_scalar("Train/ avg loss",avg_loss,global_step=(epoch-1)*len(train_dataLoader)+batch_id)
            writer.flush()


def validation(val_dataLoader, criterion):
    print("*******  begin validation  *******")
    model.eval()
    all_loss = 0.0

    EVAL = Eval(opt.n_classes)
    EVAL.reset()
    #Fwiou =FWIOU(opt.n_classes)
    with torch.no_grad():
        with tqdm(total=len(val_dataLoader), unit='batch') as pbar:
            for batch_id, (data, target_mask) in enumerate(val_dataLoader):
                data, target_mask = data.to(opt.device), target_mask.to(opt.device)
                out = model(data)
                loss = criterion(out, target_mask)
                current_loss = loss.data.item()
                all_loss += current_loss

                out = out.data.cpu().numpy()
                target_mask = target_mask.data.cpu().numpy()

                EVAL.add_batch(target_mask, out.argmax(axis=1))

                '''for ii in range(len(out)):
                    every_out = out[ii]
                    every_out = np.transpose(every_out, (1, 2, 0))   #chage to 256,256,8
    
                    predict = every_out.argmax(axis=2)
                    seg_img = np.zeros((256, 256), dtype=np.uint16)
                    for c in range(opt.n_classes):
                        seg_img[predict == c] = c
    
                    every_target_mask = target_mask[ii]
                    Fwiou.update(predict=seg_img , gt=every_target_mask)'''
                pbar.update(1)
        print('[ validation ] [average loss:{}]'.format(all_loss/len(val_dataLoader)))
        PA = EVAL.Pixel_Accuracy()
        MPA = EVAL.Mean_Pixel_Accuracy()
        MIoU = EVAL.Mean_Intersection_over_Union()
        FWIoU = EVAL.Frequency_Weighted_Intersection_over_Union()

        print('[ validation ] [PA1: {:.8f}], [MPA1: {:.8f}], [MIoU1: {:.8f}], [FWIoU1: {:.8f}]'.format(PA, MPA,MIoU, FWIoU))
        #Fwiou.calculate_fwiou()



if __name__ == "__main__":
    set_seed(1)
    check_mkdir(opt.tensorboard_path)
    writer =SummaryWriter(opt.tensorboard_path)
    #writer = None

    train_data = Dataset(images_path=opt.images_path, labels_path= opt.labels_path,mode="train",
                         train_val_scale = opt.train_val_scale, use_augment=True)
    train_dataLoader = DataLoader(train_data, opt.batch, shuffle=True, num_workers=opt.num_workers)
    val_data = Dataset(images_path=opt.images_path, labels_path=opt.labels_path, mode="val",
                       train_val_scale = opt.train_val_scale)
    val_dataLoader = DataLoader(val_data,  opt.batch, shuffle=True, num_workers=opt.num_workers)

    #model = AttU_Net(img_ch= 3,output_ch=opt.n_classes).to(opt.device)
    model = DeepLab(output_stride=16,class_num=opt.n_classes,pretrained=True,bn_momentum=0.1,freeze_bn=False)


    if torch.cuda.device_count() > 1:
        print("use many GPUS！")
        model = nn.DataParallel(model,device_ids=[0,1,2])
    #model.load_state_dict(torch.load("/data/zhujie/checkpoints/net_30.pth"))
    model.to(device=opt.device)

    criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    optimizer = Nadam(model.parameters(),lr = opt.lr)
    #optimizer = optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)
    #ptimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    #optimizer= AdamW(model.parameters(),lr=opt.lr, weight_decay=opt.weight_decay)

    TOTAL_STEP = len(train_dataLoader)*opt.epoch + 1
    WARMUP_RATIO = 0.1
    WARMUP_STEP = TOTAL_STEP * WARMUP_RATIO

    global_step = 0
    for epoch in range(1, opt.epoch + 1):
        print("[----------------------------------epoch {} ---------------------------------]".format(epoch))
        train(epoch,train_dataLoader, optimizer, criterion,writer, global_step, WARMUP_STEP, TOTAL_STEP)
        global_step += len(train_dataLoader)
        if epoch%5==0:
            save_model(model, opt.model_save_path, epoch)
        validation(val_dataLoader,criterion)
        #reset dataset,shuffle the img
        train_data = Dataset(images_path=opt.images_path, labels_path=opt.labels_path, mode="train",
                             train_val_scale=opt.train_val_scale, use_augment=True)
        train_dataLoader = DataLoader(train_data, opt.batch, shuffle=True, num_workers=opt.num_workers)
        val_data = Dataset(images_path=opt.images_path, labels_path=opt.labels_path, mode="val",
                           train_val_scale=opt.train_val_scale)
        val_dataLoader = DataLoader(val_data, opt.batch, shuffle=True, num_workers=opt.num_workers)
    predict(model, opt.test_input_path, opt.test_output_path, opt.n_classes, opt.test_weights_path)
