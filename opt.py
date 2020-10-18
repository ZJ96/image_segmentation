import torch

class OPT():
    epoch = 120
    batch = 100
    n_classes = 8
    train_val_scale = 0.95   #train and val dataset scale
    images_path = "/data/zhangzhenghao/train/image"
    labels_path = "/data/zhangzhenghao/train/label"
    num_workers = 8


    lr = 0.001
    lr_decay = 0.95
    weight_decay = 0  # 损失函数

    train_print_num = 25
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model_save_path = "/data/zhangzhenghao/checkpoints/"

    test_weights_path = "/data/zhangzhenghao/checkpoints/u_net_120.pth"
    test_input_path = "/data/zhangzhenghao/image_B"
    test_output_path = "/data/zhangzhenghao/results/"

    tensorboard_path = "/data/zhangzhenghao/tensorboard/"


opt = OPT()