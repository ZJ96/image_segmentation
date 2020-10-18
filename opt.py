import torch

class OPT():
    epoch = 120
    batch = 100
    n_classes = 8
    train_val_scale = 0.9   #train and val dataset scale
    images_path = "/data/zhujie/train/image"
    labels_path = "/data/zhujie/train/label"
    num_workers = 8


    lr = 0.001
    lr_decay = 0.95
    weight_decay = 0  # 损失函数

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_path = "/data/zhujie/checkpoints/"

    test_weights_path = "/data/zhujie/checkpoints/net_120.pth"
    test_input_path = "/data/zhujie/image_B"
    test_output_path = "/data/zhujie/results/"

    tensorboard_path = "/data/zhujie/tensorboard/"


opt = OPT()