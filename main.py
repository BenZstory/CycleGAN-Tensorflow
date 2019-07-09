from CycleGAN import CycleGAN
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of CycleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='summer2winter', help='dataset_name')
    parser.add_argument('--augment_flag', type=int, default=0, help='Image augmentation use or not')
    parser.add_argument('--do_random_hue', type=int, default=0, help='Whether do random hue on domain_B')

    parser.add_argument('--epoch', type=int, default=2, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=1000000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--print_step', type=int, default=5, help='The number of steps that gonna print in one time.')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--gan_w', type=float, default=1.0, help='weight of adversarial loss')
    parser.add_argument('--cycle_w', type=float, default=10.0, help='weight of cycle loss')
    parser.add_argument('--identity_w', type=float, default=5.0, help='weight of identity loss')
    parser.add_argument('--counter_penalty_w', type=float, default=1.0, help='weight of identity loss')
    parser.add_argument('--semantic_w', type=float, default=0.0002, help='weight of identity loss')
    parser.add_argument('--pix_loss_w', type=float, default=5.0, help='weight of identity loss')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=9, help='The number of residual blocks')

    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    parser.add_argument('--from_checkpoint', type=str, default='',
                        help='Directory to load the checkpoints')
    parser.add_argument('--skip', type=int, default=0,
                        help='if 0, no skip in generator, if 1 or 2, two different kinds skip in generator.')
    parser.add_argument('--restore_partly', type=int, default=0, help='if restore partly')

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