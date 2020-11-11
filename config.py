import argparse
import time

def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--model-name",type=str, default="DEEPLABV3+")

    parser.add_argument("--epochs", type=int, default=128,    help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=40,    help="number of batch size, (default, 100)")

    parser.add_argument("--n-classes", type=int, default=17,    help="number of classes, (default, 17)")
    parser.add_argument("--images-path", type = str, default="./data/train/image")
    parser.add_argument("--labels-path", type = str, default="./data/train/label")

    parser.add_argument("--data-augment", type=bool, default=True)
    parser.add_argument("--checkpoints-path", type=str, default="./checkpoints")

    parser.add_argument("--log-file-name",type= str,default="train_logs.log")

    #lr schedule
    parser.add_argument("--lr-scheduler",type = str,default="poly",help="cos,poly,step")
    #train parameters SGD
    parser.add_argument("--optimizer-lr", type=float, default=0.01 , help="sgd lr")
    parser.add_argument("--optimizer-momentum", type=float, default=0.9,    help="sgd momentum")
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.0001 , help="sgd weight decay")
    parser.add_argument("--optimizer-nesterov", type=bool, default= False,    help="sgd nesterov")

    #predict params
    # parser.add_argument('--test-weights-path', type=str, default="./checkpoints/net_120.pth")
    # parser.add_argument('--test-input-path', type=str, default="./data/test/image")
    # parser.add_argument('--test-output-path', type=str, default="./data/test/results/")

    args = parser.parse_args()

    print("---------------------- args ---------------------- ")
    for k in list(vars(args).keys()):
        print("{}  :  {}  ".format(k,vars(args)[k]))
    print("---------------------- args ---------------------- ")
    print("****** start train time ******: ", time.strftime("%Y %m/%d  %H:%M:%S"))
    print("\n")

    return args
