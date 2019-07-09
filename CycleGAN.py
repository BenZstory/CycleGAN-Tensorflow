from ops import *
from utils import *
from glob import glob
import time
from datetime import datetime
from tensorflow.contrib.data import batch_and_drop_remainder
import tensorflow as tf

class CycleGAN(object) :
    def __init__(self, sess, args):
        self.model_name = 'CycleGAN'
        self.sess = sess
        self.args_dict = vars(args)
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.from_checkpoint = args.from_checkpoint
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag
        self.do_random_hue = args.do_random_hue
        self.skip = args.skip
        self.restore_partly = args.restore_partly

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.print_step = args.print_step
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.gan_w = args.gan_w
        self.cycle_w = args.cycle_w
        self.identity_w = args.identity_w
        self.counter_penalty_w = args.counter_penalty_w
        self.semantic_w = args.semantic_w
        self.pix_loss_w = args.pix_loss_w
        self.pix_loss_w_base = self.pix_loss_w
        self.semantic_w_base = self.semantic_w

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        """ working on dir params """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = "{}_{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type, current_time)

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(os.path.join(self.sample_dir, "imgs"))

        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_dir)
        self.log_dir = os.path.join(args.log_dir, self.model_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.trainP_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainP'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset), len(self.trainP_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, scope='conv_'+str(i+1))
                x = instance_norm(x, scope='down_ins_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            # Bottle-neck
            for i in range(self.n_res) :
                x = resblock(x, channel, scope='resblock_'+str(i))

            # Up-Sampling
            for i in range(2) :
                x = deconv(x, channel//2, kernel=3, stride=2, scope='deconv_'+str(i+1))
                x = instance_norm(x, scope='up_ins_'+str(i+1))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x

    # this is a skip version in which we add conv_block with devconv-IN output
    def generator_with_skip(self, x, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            input = x

            #  conv_0
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            e0 = relu(x)

            # down-sample conv_1
            channel = channel * 2
            x = conv(e0, channel, kernel=3, stride=2, pad=1, scope='conv_1')
            x = instance_norm(x, scope='down_ins_1')
            e1 = relu(x)

            # down-sample conv_2
            channel = channel * 2
            x = conv(e1, channel, kernel=3, stride=2, pad=1, scope='conv_2')
            x = instance_norm(x, scope='down_ins_2')
            e2 = relu(x)

            # Bottle-neck
            x = e2
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

            # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_1')
            x = instance_norm(x, scope='up_ins_1')
            x = relu(x+e1)

            # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_2')
            x = instance_norm(x, scope='up_ins_2')
            x = relu(x + e0)

            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x + input)

            return x


    def generator_with_skip2(self, x, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            input = x

            #  conv_0
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            e0 = relu(x)

            # down-sample conv_1
            channel = channel * 2
            x = conv(e0, channel, kernel=3, stride=2, pad=1, scope='conv_1')
            x = instance_norm(x, scope='down_ins_1')
            e1 = relu(x)

            # down-sample conv_2
            channel = channel * 2
            x = conv(e1, channel, kernel=3, stride=2, pad=1, scope='conv_2')
            x = instance_norm(x, scope='down_ins_2')
            e2 = relu(x)

            # Bottle-neck
            x = e2
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

            # up-sample deconv_0
            x = tf.concat([x, e2], axis=-1)
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_1')
            x = instance_norm(x, scope='up_ins_1')
            x = relu(x)

            # up-sample deconv_0
            x = tf.concat([x, e1], axis=-1)
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_2')
            x = instance_norm(x, scope='up_ins_2')
            x = relu(x)

        with tf.variable_scope(scope + "_upwithskip", reuse=reuse):
            x = tf.concat([x, e0], axis=-1)
            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x


    def generator_with_skip3(self, x, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            input = x

            #  conv_0
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            e0 = relu(x)

            # down-sample conv_1
            channel = channel * 2
            x = conv(e0, channel, kernel=3, stride=2, pad=1, scope='conv_1')
            x = instance_norm(x, scope='down_ins_1')
            e1 = relu(x)

            # down-sample conv_2
            channel = channel * 2
            x = conv(e1, channel, kernel=3, stride=2, pad=1, scope='conv_2')
            x = instance_norm(x, scope='down_ins_2')
            e2 = relu(x)

            # Bottle-neck
            x = e2
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

        # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_1')
            x = instance_norm(x, scope='up_ins_1')
            x = relu(x)

            # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_2')
            x = instance_norm(x, scope='up_ins_2')
            x = relu(x)

        with tf.variable_scope(scope + "_upwithskip3", reuse=reuse):
            x = tf.concat([x, e0], axis=-1)
            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x


    def generator_with_skip4(self, x, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            input = x

            #  conv_0
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            e0 = relu(x)

            # down-sample conv_1
            channel = channel * 2
            x = conv(e0, channel, kernel=3, stride=2, pad=1, scope='conv_1')
            x = instance_norm(x, scope='down_ins_1')
            e1 = relu(x)

            # down-sample conv_2
            channel = channel * 2
            x = conv(e1, channel, kernel=3, stride=2, pad=1, scope='conv_2')
            x = instance_norm(x, scope='down_ins_2')
            e2 = relu(x)

            # Bottle-neck
            x = e2
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

            # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_1')
            x = instance_norm(x, scope='up_ins_1')
            x = relu(x)

            # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_2')
            x = instance_norm(x, scope='up_ins_2')
            x = relu(x)

        with tf.variable_scope(scope + "_upwithskip", reuse=reuse):
            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x + input)

            return x

    def generator_with_skip5(self, x, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            input = x

            #  conv_0
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            e0 = relu(x)

            # down-sample conv_1
            channel = channel * 2
            x = conv(e0, channel, kernel=3, stride=2, pad=1, scope='conv_1')
            x = instance_norm(x, scope='down_ins_1')
            e1 = relu(x)

            # down-sample conv_2
            channel = channel * 2
            x = conv(e1, channel, kernel=3, stride=2, pad=1, scope='conv_2')
            x = instance_norm(x, scope='down_ins_2')
            e2 = relu(x)

            # Bottle-neck
            x = e2
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))

            # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_1')
            x = instance_norm(x, scope='up_ins_1')
            x = relu(x)

            # up-sample deconv_0
            channel = channel // 2
            x = deconv(x, channel, kernel=3, stride=2, scope='deconv_2')
            x = instance_norm(x, scope='up_ins_2')
            x = relu(x)

        with tf.variable_scope(scope + "_upwithskip5", reuse=reuse):
            x = tf.concat([x, input], axis=-1)
            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x, reuse=False, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=4, stride=2, pad=1, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, scope='conv_'+str(i))
                x = instance_norm(x, scope='ins_'+str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel*2, kernel=4, stride=1, pad=1, scope='conv_'+str(self.n_dis))
            x = instance_norm(x, scope='ins_'+str(self.n_dis))
            x = lrelu(x, 0.2)

            x = conv(x, channels=1, kernel=4, stride=1, pad=1, scope='D_logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################

    def generate_a2b(self, x_A, reuse=False):
        if (self.skip == 1):
            x_ab = self.generator_with_skip(x_A, reuse=reuse, scope='generator_B')
        elif (self.skip == 2):
            x_ab = self.generator_with_skip2(x_A, reuse=reuse, scope='generator_B')
        elif (self.skip == 3):
            x_ab = self.generator_with_skip3(x_A, reuse=reuse, scope='generator_B')
        elif (self.skip == 4):
            x_ab = self.generator_with_skip4(x_A, reuse=reuse, scope='generator_B')
        elif self.skip == 5:
            x_ab = self.generator_with_skip5(x_A, reuse=reuse, scope='generator_B')
        else:
            x_ab = self.generator(x_A, reuse=reuse, scope='generator_B')

        return x_ab

    def generate_b2a(self, x_B, reuse=False):
        if (self.skip == 1):
            x_ba = self.generator_with_skip(x_B, reuse=reuse, scope='generator_A')
        elif (self.skip == 2):
            x_ba = self.generator_with_skip2(x_B, reuse=reuse, scope='generator_A')
        elif (self.skip == 3):
            x_ba = self.generator_with_skip3(x_B, reuse=reuse, scope='generator_A')
        elif (self.skip == 4):
            x_ba = self.generator_with_skip4(x_B, reuse=reuse, scope='generator_A')
        elif (self.skip == 5):
            x_ba = self.generator_with_skip5(x_B, reuse=reuse, scope='generator_A')
        else:
            x_ba = self.generator(x_B, reuse=reuse, scope='generator_A')

        return x_ba

    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag, self.do_random_hue)
        # Image_Data_Class_B = ImageData(self.img_size, self.img_ch, self.augment_flag, self.do_random_hue)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)
        trainP = tf.data.Dataset.from_tensor_slices(self.trainP_dataset)

        # trainA = trainA.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()
        # trainB = trainB.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()
        # trainP = trainP.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()

        trainA = trainA.shuffle(self.dataset_num, reshuffle_each_iteration=True).repeat()
        trainA = trainA.map(Image_Data_Class.image_processing).batch(self.batch_size)
        trainB = trainB.shuffle(self.dataset_num, reshuffle_each_iteration=True).repeat()
        trainB = trainB.map(Image_Data_Class.image_processing).batch(self.batch_size)
        trainP = trainP.shuffle(self.dataset_num, reshuffle_each_iteration=True).repeat()
        trainP = trainP.map(Image_Data_Class.image_processing).batch(self.batch_size)


        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()
        trainP_iterator = trainP.make_one_shot_iterator()


        self.domain_A = trainA_iterator.get_next()
        self.domain_B = trainB_iterator.get_next()
        self.domain_P = trainP_iterator.get_next()


        """ Define Encoder, Generator, Discriminator """
        x_ab = self.generate_a2b(self.domain_A)
        x_ba = self.generate_b2a(self.domain_B)

        x_aba = self.generate_b2a(x_ab, reuse=True)
        x_bab = self.generate_a2b(x_ba, reuse=True)

        if self.identity_w > 0 :
            x_aa = self.generate_b2a(self.domain_A, reuse=True)
            x_bb = self.generate_a2b(self.domain_B, reuse=True)

            identity_loss_a = L1_loss(x_aa, self.domain_A)
            identity_loss_b = L1_loss(x_bb, self.domain_B)

        else :
            identity_loss_a = 0
            identity_loss_b = 0

        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

        """ Define Loss """
        G_ad_loss_a = generator_loss(self.gan_type, fake_A_logit)
        G_ad_loss_b = generator_loss(self.gan_type, fake_B_logit)

        if self.semantic_w > 0:
            semantic_loss_a = semantic_loss_with_attention(self.domain_A, x_ab, self.batch_size)
            semantic_loss_b = semantic_loss_with_attention(self.domain_B, x_ba, self.batch_size)
        else:
            semantic_loss_a = 0
            semantic_loss_b = 0

        if self.pix_loss_w > 0:
            pix_loss_a = pix_loss_with_attention(self.domain_A, x_ab, self.batch_size)
            pix_loss_b = pix_loss_with_attention(self.domain_B, x_ba, self.batch_size)
        else:
            pix_loss_a = 0
            pix_loss_b = 0

        D_ad_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        D_ad_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)

        # we are only doing counter penalty on a2b, (penalty in domainB)
        if self.counter_penalty_w > 0:
            cp_logit = self.discriminator(self.domain_P, reuse=True, scope="discriminator_B")
            cp_loss = lsgan_loss_discriminator_counter_penalty(cp_logit, self.batch_size)
        else:
            cp_loss = 0

        recon_loss_a = L1_loss(x_aba, self.domain_A) # reconstruction
        recon_loss_b = L1_loss(x_bab, self.domain_B) # reconstruction

        Generator_A_loss = self.gan_w * G_ad_loss_a + \
                           self.cycle_w * recon_loss_b + \
                           self.identity_w * identity_loss_a + \
                           self.semantic_w * semantic_loss_a + \
                           self.pix_loss_w * pix_loss_a

        Generator_B_loss = self.gan_w * G_ad_loss_b + \
                           self.cycle_w * recon_loss_a + \
                           self.identity_w * identity_loss_b + \
                           self.semantic_w * semantic_loss_b + \
                           self.pix_loss_w * pix_loss_b

        Discriminator_A_loss = self.gan_w * D_ad_loss_a
        Discriminator_B_loss = self.gan_w * D_ad_loss_b + \
                               self.counter_penalty_w * cp_loss

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.cp_loss_summ = tf.summary.scalar("CP_loss", cp_loss)
        self.semantic_loss_a_summ = tf.summary.scalar("semantic_A_loss", semantic_loss_a)
        self.semantic_loss_b_summ = tf.summary.scalar("semantic_B_loss", semantic_loss_a)
        self.pix_loss_a_summ = tf.summary.scalar("pix_A_loss", pix_loss_a)
        self.pix_loss_b_summ = tf.summary.scalar("pix_B_loss", pix_loss_b)

        self.G_loss_summ = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                                             self.semantic_loss_a_summ, self.semantic_loss_b_summ, self.pix_loss_a_summ, self.pix_loss_b_summ])
        self.D_loss_summ = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss,
                                             self.cp_loss_summ])

        """ Image """
        self.fake_A = x_ba
        self.fake_B = x_ab

        self.real_A = self.domain_A
        self.real_B = self.domain_B

        self.cycle_A = x_aba
        self.cycle_B = x_bab

        """ Test """
        self.test_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_image')

        self.test_fake_A = self.generate_b2a(self.test_image, reuse=True)
        self.test_fake_B = self.generate_a2b(self.test_image, reuse=True)

    def train(self):
        self.total_sample_path = os.path.join(os.path.join(self.sample_dir, "_total_samples.html"))
        self.write_args_to_html()

        # initialize all variables
        tf.global_variables_initializer().run()

        if self.restore_partly:
            # saver to save model
            variables_to_restore = [var for var in tf.global_variables() if 'upwithskip' not in var.name]
            print("[BENZ] partly restore variables, variables_to_restore:")
            print(variables_to_restore)
            self.saver = tf.train.Saver(variables_to_restore)
            # remind don't forget to reconstruct Saver with full variables, or these skipped vars would never be saved.
        else:
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.from_checkpoint)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        if self.restore_partly:
            print("recostruct fully graph saver")
            # reconstruct the saver with full vars
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            if epoch >= self.epoch // 2 :
                lr = self.init_lr * 0.5

            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, \
                d_loss, \
                summary_str = self.sess.run([self.D_optim,
                                             self.Discriminator_loss,
                                             self.D_loss_summ], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_images, batch_B_images,\
                fake_A, fake_B,\
                cycle_A, cycle_B, \
                _, \
                g_loss, \
                summary_str = \
                    self.sess.run([self.real_A, self.real_B,
                                   self.fake_A, self.fake_B,
                                   self.cycle_A, self.cycle_B,
                                   self.G_optim,
                                   self.Generator_loss,
                                   self.G_loss_summ], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(counter+self.print_step, self.print_freq) < self.print_step :
                    image_save_epoch = (counter + self.print_step) // self.print_freq
                    step_mod = np.mod(counter+self.print_step, self.print_freq)
                    html_name = "samples_" + str((counter + self.print_step) // self.print_freq) + '.html'
                    if np.mod(counter+self.print_step, self.print_freq) == 0:
                        with open(self.total_sample_path, 'a') as t_html:
                            t_html.write("<hr style=\"border-bottom: 3px solid red\" />\r\n<h3> Samples_of_" +
                                         str((counter + self.print_step) // self.print_freq) + " </h3>")

                    for j in range(0, self.batch_size):
                        img_id = step_mod * self.batch_size + j

                        save_one_img(batch_A_images[j], './{}/imgs/real_A_{:02d}_{:06d}_{:02d}.jpg'.format(
                            self.sample_dir, epoch, idx + 1, img_id))
                        save_one_img(batch_B_images[j], './{}/imgs/real_B_{:02d}_{:06d}_{:02d}.jpg'.format(
                            self.sample_dir, epoch, idx + 1, img_id))

                        save_one_img(fake_A[j], './{}/imgs/fake_A_{:02d}_{:06d}_{:02d}.jpg'.format(
                            self.sample_dir, epoch, idx + 1, img_id))
                        save_one_img(fake_B[j], './{}/imgs/fake_B_{:02d}_{:06d}_{:02d}.jpg'.format(
                            self.sample_dir, epoch, idx + 1, img_id))

                        save_one_img(cycle_A[j], './{}/imgs/cycle_A_{:02d}_{:06d}_{:02d}.jpg'.format(
                            self.sample_dir, epoch, idx + 1, img_id))
                        save_one_img(cycle_B[j], './{}/imgs/cycle_B_{:02d}_{:06d}_{:02d}.jpg'.format(
                            self.sample_dir, epoch, idx + 1, img_id))

                        self.write_to_html(os.path.join(self.sample_dir, html_name), epoch, idx + 1, img_id)

                if np.mod(idx+1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)
                    self.pix_loss_w = self.pix_loss_w_base + min(counter/200000, 15-self.pix_loss_w_base)
                    self.semantic_w = self.semantic_w_base + min(counter/100000 * 0.0001, 0.0006-self.semantic_w_base)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    def write_args_to_html(self):
        body = ""
        for k, v in self.args_dict.items():
            body = body + "--" + str(k) + " " + str(v) + " \\<br>"
        with open(self.total_sample_path, 'a') as t_html:
            t_html.write("python3 main.py \\<br>")
            t_html.write(body)


    def write_to_html(self, html_path, epoch, idx, img_id):
        names = ['real_A', 'real_B', 'fake_B', 'fake_A', 'cycle_A', 'cycle_B']

        body = ""
        for name in names:
            image_name = '{}_{:02d}_{:06d}_{:02d}.jpg'.format(name, epoch, idx, img_id)
            body = body + str("<img src=\"" + os.path.join('imgs', image_name) + "\">")
        body = body + str("<br>")

        with open(html_path, 'a') as v_html:
            v_html.write(body)
        with open(self.total_sample_path, 'a') as t_html:
            t_html.write(body)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.from_checkpoint)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir, '{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict={self.test_image: sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir, '{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        index.close()
