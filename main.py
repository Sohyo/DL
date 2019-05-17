import argparse

from networks import OurResNet, OurDenseNet


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--filename', type=str, default='default',
                        help='The nice file name to store output.')
    parser.add_argument('--feature_extract', default=False, action='store_true',
                        help='When this argument is supplied feature extraction instead of fine-tuing is used.')
    parser.add_argument('--use_densenet', default=False, action='store_true',
                        help='When this argument is supplied a densenet instead of a resnet is used.')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        help='Please decide which optimizer you want to use: Adam or Adadelta')

    args = parser.parse_args()

    return args.epochs, args.filename, args.feature_extract, args.optimizer, args.use_densenet


if __name__ == '__main__':
    epochs, filename, feature_extract, optimizer, use_densenet = parse_arguments()

    if use_densenet:
        net = OurDenseNet(epochs=epochs, feature_extract=feature_extract, optimizer=optimizer)
    else:
        net = OurResNet(epochs=epochs, feature_extract=feature_extract, optimizer=optimizer)

    metrics = net.run()
    net.save_metrics(filename, metrics)
