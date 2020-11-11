# coding =utf-8
import math
import os
import torch
import torch.nn as nn
from models import DeepLab
from data.dataset import Dataset
from torch.utils.data import DataLoader
from config import get_args
import numpy as np
import random
from utils import Eval, write_log
from tqdm import tqdm
#import segmentation_models_pytorch as smp

from lr_scheduler import LR_Scheduler

# from torch import optim
# from adams import AdamW,Nadam

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

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


def save_model(model, model_save_path, current_epoch):
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    save_path = "net_" + str(current_epoch) + ".pth"
    save_path = os.path.join(model_save_path,save_path)
    torch.save(model.state_dict(), save_path)

# end_lr = 1e-5
# def get_base_lr(step,total_step):
#     base_lr = 1e-3
#     if step < 0.5*total_step:
#         base_lr = 1e-3
#     elif step < 0.7*total_step:
#         base_lr = 0.5 * 1e-3
#     elif step < 0.9*total_step:
#         base_lr = 0.25 * 1e-3
#     else:
#         base_lr = 0.1 * 1e-3
#     return base_lr
#
# def lr_function(step, warm_step, total_step):
#     base_lr = get_base_lr(step,total_step)
#     if step < warm_step:
#         lr = step / warm_step * base_lr
#     else:
#         lr = end_lr+ 0.5* (base_lr - end_lr ) * ( 1 + math.cos((step - warm_step)/(total_step- step) * math.pi))
#     return lr
#
# def adjust_lr(optimizer,lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def train(epoch, train_dataLoader, optimizer, criterion, args, best_pred):
    print("*******  begin train  *******")
    model.train()
    all_loss = 0.0
    with tqdm(total= len(train_dataLoader), unit='batch') as pbar:
        for batch_id, (data, target_img) in enumerate(train_dataLoader):
            #global_step+=1
            # lr = lr_function(global_step, WARMUP_STEP, TOTAL_STEP)
            # adjust_lr(optimizer,lr)
            scheduler(optimizer, batch_id, epoch, best_pred)

            data,target_img = data.to(device),  target_img.to(device)
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
    log_message ="**train** [ epoch  {} ] , [ avg_loss  {:.8f} ] , [ lr  {:.8f} ]".format(epoch,avg_loss,lr)
    write_log(args.log_file_name,log_message)


def validation(val_dataLoader, criterion, args, best_pred):
    print("*******  begin validation  *******")
    model.eval()
    all_loss = 0.0

    EVAL = Eval(args.n_classes)
    EVAL.reset()
    with torch.no_grad():
        with tqdm(total=len(val_dataLoader), unit='batch') as pbar:
            for batch_id, (data, target_mask) in enumerate(val_dataLoader):
                data, target_mask = data.to(device), target_mask.to(device)
                out = model(data)
                loss = criterion(out, target_mask)
                current_loss = loss.data.item()
                all_loss += current_loss

                out = out.data.cpu().numpy()
                target_mask = target_mask.data.cpu().numpy()
                EVAL.add_batch(target_mask, out.argmax(axis=1))

                pbar.update(1)
    print('[ validation ] [average loss:{}]'.format(all_loss/len(val_dataLoader)))
    PA = EVAL.Pixel_Accuracy()
    MPA = EVAL.Mean_Pixel_Accuracy()
    MIoU = EVAL.Mean_Intersection_over_Union()
    FWIoU = EVAL.Frequency_Weighted_Intersection_over_Union()
    print('[ validation ] [PA1: {:.8f}], [MPA1: {:.8f}], [MIoU1: {:.8f}], [FWIoU1: {:.8f}]'.format(PA, MPA,MIoU, FWIoU))

    log_message ='**validation** [average loss: {:.8f} ] \n[PA1: {:.8f}], [MPA1: {:.8f}], [MIoU1: {:.8f}], [FWIoU1: {:.8f}]'\
        .format(all_loss/len(val_dataLoader),PA, MPA,MIoU, FWIoU)
    write_log(args.log_file_name,log_message)

    new_pred = FWIoU
    if new_pred > best_pred:
        print("Got the best FWIOU!")
        return new_pred
    return best_pred




if __name__ == "__main__":
    best_pred = 0.0     #FWIOU 最好的预测精度
    set_seed(1)
    dataset_train_scale = 1.0
    dataset_val_scale = 0.9
    num_workers = 4
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #数据集 data loader
    train_data = Dataset(images_path=args.images_path, labels_path= args.labels_path,mode="train",
                         train_val_scale = dataset_train_scale, use_augment=args.data_augment)
    train_dataLoader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=num_workers)
    val_data = Dataset(images_path=args.images_path, labels_path=args.labels_path, mode="val",
                         train_val_scale = dataset_val_scale)
    val_dataLoader = DataLoader(val_data,  args.batch_size, shuffle=True, num_workers=num_workers)

    #lr schedule
    scheduler = LR_Scheduler(args.lr_scheduler, args.optimizer_lr,
                             args.epochs, len(train_dataLoader), warmup_epochs=5)

    #model = AttU_Net(img_ch= 3,output_ch=opt.n_classes).to(opt.device)
    model = DeepLab(output_stride=16,class_num=args.n_classes,pretrained=True,bn_momentum=0.1,freeze_bn=False)

    if torch.cuda.device_count() > 1:
        print("use many GPUS！")
        model = nn.DataParallel(model,device_ids=[0,1,2])
    #model.load_state_dict(torch.load("/data/zhujie/checkpoints/net_30.pth"))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    #optimizer = Nadam(model.parameters())
    #optimizer = optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.optimizer_lr,
                                momentum=args.optimizer_momentum,
                                weight_decay=args.optimizer_weight_decay,
                                nesterov=args.optimizer_nesterov,
                                )

    # TOTAL_STEP = len(train_dataLoader)*(args.epochs) + 1
    # WARMUP_RATIO = 0.1
    # WARMUP_STEP = TOTAL_STEP * WARMUP_RATIO
    #global_step = 0

    for epoch in range(1, args.epochs + 1):
        print("[----------------------------------epoch {} ---------------------------------]".format(epoch))
        train(epoch,train_dataLoader, optimizer, criterion, args, best_pred)

        save_model(model, args.checkpoints_path, epoch)

        best_pred = validation(val_dataLoader,criterion, args, best_pred)
