import torch

class OPT():
    epoch = 120
    batch = 100
    n_classes = 8
    train_val_scale = 0.9   #train and val dataset scale
    images_path = "/data/zhujie/train/image"   #train img path
    labels_path = "/data/zhujie/train/label"   #train label path
    num_workers = 8

    # lr
    lr = 0.001
    lr_decay = 0.95
    weight_decay = 0

    #device gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_save_path = "/data/zhujie/checkpoints/"      #the path for saving the model

    # predict params
    test_weights_path = "/data/zhujie/checkpoints/net_120.pth"     #the model weight path for testing
    test_input_path = "/data/zhujie/image_B"         #test img path
    test_output_path = "/data/zhujie/results/"       #the path for outputing the label

    tensorboard_path = "/data/zhujie/tensorboard/"    #tensorboard path


opt = OPT()