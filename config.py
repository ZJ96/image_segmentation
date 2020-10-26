import argparse

def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument("--model-name",type=str, default="DEEPLABV3+")

    parser.add_argument("--epochs", type=int, default=10,    help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4,    help="number of batch size, (default, 100)")

    parser.add_argument("--n-classes", type=int, default=8,    help="number of classes, (default, 17)")
    parser.add_argument("--images-path", type = str, default="./data/train/image")
    parser.add_argument("--labels-path", type = str, default="./data/train/label")

    parser.add_argument("--data-augment", type=bool, default=True)
    parser.add_argument("--checkpoints-path", type=str, default="./checkpoints")

    #predict params
    parser.add_argument('--test-weights-path', type=str, default="./checkpoints/net_120.pth")
    parser.add_argument('--test-input-path', type=str, default="./data/test/image")
    parser.add_argument('--test-output-path', type=str, default="./data/test/results/")

    args = parser.parse_args()

    print("    ---------------------- args ---------------------- ")
    for k in list(vars(args).keys()):
        print("    {}   :   {}  ".format(k,vars(args)[k]))
    print("    ---------------------- args ---------------------- \n")

    return args

a = get_args()