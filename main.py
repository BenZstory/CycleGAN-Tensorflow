
from CycleGAN import CycleGAN
import argparse
from ops import *
from utils import *
"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of CycleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--epoch', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')
    parser.add_argument('--do_resnet', type=bool, default=True, help='Doing residual block ? or not ?')
    parser.add_argument('--dis_layer', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--res_block', type=int, default=6, help='The number of res_block')
    parser.add_argument('--norm', type=str, default='instance', help='instance or batch')
    parser.add_argument('--lambdaA', type=float, default=10.0, help='reconstruction_loss')
    parser.add_argument('--lambdaB', type=float, default=10.0, help='reconstruction_loss')
    parser.add_argument('--identity', type=float, default=0.5, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss.'
                                      'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning_rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam')
    parser.add_argument('--pool_size', type=int, default=50, help='discriminator pool')
    parser.add_argument('--dataset', type=str, default='cat2dog', help='dataset_name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = CycleGAN(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()