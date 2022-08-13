import torch
import torch.nn as nn
import argparse
import os
import json
import random
import utils
import numpy as np
import torch.nn.functional as F
import math
import time
import TranSVAE
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


parser = argparse.ArgumentParser()
# ========================= Dataset Configs ==========================
parser.add_argument('--dataset',  default='Sprite',
                    help='datasets')
parser.add_argument('--data_root', default='dataset',
                    help='root directory for data')
parser.add_argument('--num_class',  type=int, default=15,
                    help='the number of class for jester dataset')
parser.add_argument('--input_type',  default='image', choices=['feature', 'image'], 
                    help='the type of input')
parser.add_argument('--src',  default='domain_1', 
                    help='source domain')
parser.add_argument('--tar',  default='domain_2', 
                    help='target domain')

# ========================= Runtime Configs ==========================
parser.add_argument('--seed', default=1, type=int, 
                    help='manual seed')
parser.add_argument('--exp_dir', default='experiments',
                    help='base directory of experiments')
parser.add_argument('--log_indicator', default=0, type=int,
                    help='base directory to save logs')
parser.add_argument('--model_dir', default='',
                    help='base directory to save models')
parser.add_argument('--data_threads', type=int, default=5,
                    help='number of data loading threads')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_hp', default=False, action="store_true",
                    help='whether to use the saved hyper-parameters')
parser.add_argument('--gpu', default='0', type=str, 
                    help='index of GPU to use')
parser.add_argument('--save_model', default=0, type=int,
                    help='whether to save models')
parser.add_argument('--parallel_train', default=False,
                    help='whether to use multi-gpus for training')
parser.add_argument('--print_details', default=False,
                    help='whether print in each mini batch')
parser.add_argument('--eval_freq', default=1, type=int,
                    help='evaluation frequency (default: 5)')
parser.add_argument('--weighted_class_loss', type=str,
                    default='Y', choices=['Y', 'N'])
parser.add_argument('--weighted_class_loss_DA', type=str,
                    default='Y', choices=['Y', 'N'])

# ========================= Model Configs ==========================
parser.add_argument('--num_segments', type=int, default=8,
                    help='the number of frame segment')
parser.add_argument('--backbone', type=str, default="dcgan",
                    choices=['dcgan', 'resnet101', 'I3Dpretrain', 'I3Dfinetune'], help='backbone')
parser.add_argument('--val_segments', type=int, default=-1,
                    help='')
parser.add_argument('--channels', default=3, type=int,
                    help='input channels for image inputs')
parser.add_argument('--add_fc', default=1, type=int, metavar='M',
                    help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2)')
parser.add_argument('--fc_dim', type=int, default=1024,
                    help='dimension of added fc')
parser.add_argument('--frame_aggregation', type=str, default='trn',
                    choices=['rnn', 'trn'], help='aggregation of frame features (none if baseline_type is not video)')
parser.add_argument('--dropout_rate', default=0.5, type=float,
                    help='dropout ratio for frame-level feature (default: 0.5)')
parser.add_argument('--f_dim', type=int, default=256, 
                    help='dim of f')
parser.add_argument('--z_dim', type=int, default=256,
                    help='dimensionality of z_t')
parser.add_argument('--rnn_size', type=int, default=256,
                    help='dimensionality of hidden layer for rnn in VAE')
parser.add_argument('--f_rnn_layers', type=int, default=1,
                    help='number of layers (content lstm)')
parser.add_argument('--triplet_type', type=str, default='mean',
                    choices=['mean', 'post'], help='type of data to calculate triplet loss')
parser.add_argument('--place_adv', default=['Y', 'Y', 'Y'], type=str, nargs="+",
                    metavar='N', help='[frame-based adv, video relation-based adv, video-based adv]')
parser.add_argument('--use_bn', type=str, default='none',
                    choices=['none', 'AdaBN', 'AutoDIAL'], help='normalization-based methods')
parser.add_argument('--prior_sample', type=str, default='random',
                    choices=['random', 'post'], help='how to sample prior')

# ========================= Learning Configs ==========================
parser.add_argument('--optimizer', type=str,
                    default='SGD', choices=['SGD', 'Adam'])
parser.add_argument('--epochs', default=1000, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=30, type=int, 
                    help='-batch size')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay', default=10, type=float,
                    metavar='LRDecay', help='decay factor for learning rate')
parser.add_argument('--lr_adaptive', type=str, default='dann',
                    choices=['none', 'loss', 'dann'])
parser.add_argument('--lr_steps', default=[500, 1000], type=float,
                    nargs="+", metavar='LRSteps', help='epochs to decay learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip_gradient', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', default=True, action="store_true")

# ========================= DA Configs ==========================
parser.add_argument('--use_attn', type=str, default='TransAttn',
                    choices=['none', 'TransAttn', 'general'], help='attention-mechanism')
parser.add_argument('--n_attn', type=int, default=1,
                    help='number of discriminators for transferable attention')
parser.add_argument('--add_loss_DA', type=str, default='none',
                    choices=['none', 'attentive_entropy'], help='add more loss functions for DA')
parser.add_argument('--pretrain_VAE', type=str, default='N',
                    choices=['N', 'Y'], help='whether to pretrain VAE or not')
parser.add_argument('--train_TranSVAE', type=str, default='Y',
                    choices=['N', 'Y'], help='whether to pretrain VAE or not')
parser.add_argument('--use_psuedo', type=str, default='N',
                    choices=['N', 'Y'], help='whether to use target psuedo label')
parser.add_argument('--tar_psuedo_thre', default=0.99, type=float,
                    metavar='W', help='threshold to select pesudo label')
parser.add_argument('--start_psuedo_step', default=100, type=int,
                    metavar='W', help='step to start to use pesudo label')

# ========================= Loss Configs ==========================
# Loss_vae + MI(z_f,z_t)
parser.add_argument('--weight_f', type=float, default=1,
                    help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', type=float, default=1,
                    help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_MI', type=float, default=0,
                    help='weighting on Mutual infomation of f and z')
# loss on z_t: (1) adv_loss (2) cls_loss (3) attendtive entropy
parser.add_argument('--weight_cls', type=float, default=0,
                    help='weighting on video classification loss')
parser.add_argument('--beta', default=[0.75, 0.75, 0.5], type=float, nargs="+", metavar='M',
                    help='weighting for the adversarial loss (use scheduler if < 0; [relation-beta, video-beta, frame-beta])')
parser.add_argument('--weight_entropy', default=0, type=float,
                    help='weighting for the entropy loss')
# loss on z_f: (1) domain_loss (2) triplet_loss
parser.add_argument('--weight_domain', type=float, default=0,
                    help='weighting on domain classification loss')
parser.add_argument('--weight_triplet', type=float,
                    default=0, help='weighting on triplet loss')
parser.add_argument('--weight_adv', type=float, default=0,
                    help='weighting on the adversarial loss')
parser.add_argument('--weight_VAE', type=float, default=1,
                    help='weighting on the VAE loss')

opt = parser.parse_args()
if not opt.parallel_train:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

best_prec1 = 0
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
CE_loss = nn.CrossEntropyLoss().cuda()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Sprite(object):
    def __init__(self, train, data, A_label, D_label, triple):
        self.data = data
        self.A_label = A_label
        self.D_label = D_label
        self.N = self.data.shape[0]
        self.triple = triple

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data_ancher = self.data[index]
        A_label_ancher = self.A_label[index]
        D_label_ancher = self.D_label[index]
        if self.triple:
            perm = np.random.permutation(data_ancher.shape[0])
            data_pos = data_ancher[perm]
            while (1):
                index_neg = np.random.randint(self.N)
                label_neg = self.A_label[index_neg]
                if np.mean(A_label_ancher == label_neg) < 0.9:
                    data_neg = self.data[index_neg]
                    break
            return data_ancher, D_label_ancher, data_pos, data_neg, A_label_ancher
        return data_ancher, D_label_ancher, A_label_ancher


def train(source_loader, target_loader, model, optimizer, train_file, epoch, criterion_src, criterion_domain, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_mse = AverageMeter()
    losses_klf = AverageMeter()
    losses_klz = AverageMeter()
    losses_triplet = AverageMeter()
    losses_dom = AverageMeter()
    losses_adv = AverageMeter()
    losses_entropy = AverageMeter()
    losses_MI = AverageMeter()
    losses_cls = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    iter_src = iter(source_loader)
    iter_tar = iter(target_loader)

    epoch_size = len(source_loader)
    len_target_loader = len(target_loader)
    start_steps = epoch * epoch_size
    total_steps = 1000 * epoch_size


    for i in range(epoch_size):
        src_data = iter_src.next()
        tar_data = iter_tar.next()

        if i % len_target_loader == 0:
            iter_tar = iter(target_loader)

        p = float(i + start_steps) / total_steps
        beta_dann = 2. / (1. + np.exp(-10 * p)) - 1
        beta = [beta_dann if opt.beta[i] < 0 else opt.beta[i]
                for i in range(len(opt.beta))]

        if opt.weight_triplet:
            source_data = src_data[0].cuda()
            source_label = src_data[1].type(torch.LongTensor)
            source_pos = src_data[2].cuda()
            source_neg = src_data[3].cuda()

            target_data = tar_data[0].cuda()
            target_label = tar_data[1].type(torch.LongTensor)
            target_pos = tar_data[2].cuda()
            target_neg = tar_data[3].cuda()

            input_data = torch.cat((source_data, target_data), dim=0)
            input_pos = torch.cat((source_pos, target_pos), dim=0)
            input_neg = torch.cat((source_neg, target_neg), dim=0)

            x = [input_data, input_pos, input_neg]
        else:
            source_data = src_data[0].cuda()
            source_label = src_data[1].type(torch.LongTensor)

            target_data = tar_data[0].cuda()
            target_label = tar_data[1].type(torch.LongTensor)

            x = torch.cat((source_data, target_data), dim=0)

        data_time.update(time.time() - end)

        source_label = source_label.cuda(non_blocking=True)
        target_label = target_label.cuda(non_blocking=True)


        f_mean, f_logvar, f_post, \
            z_post_mean, z_post_logvar, z_post, \
                z_prior_mean, z_prior_logvar, z_prior, \
                    recon_x, pred_domain_all, pred_video_class = model(x, beta)


        # (I) sequential VAE loss
        if isinstance(x, list):
            if opt.triplet_type == 'mean':
                f_src_pos = f_mean[1][:opt.batch_size, :]
                f_tar_pos = f_mean[1][opt.batch_size:, :]
                f_src_neg = f_mean[2][:opt.batch_size, :]
                f_tar_neg = f_mean[2][opt.batch_size:, :]
            elif opt.triplet_type == 'post':
                f_src_pos = f_post[1][:opt.batch_size, :]
                f_tar_pos = f_post[1][opt.batch_size:, :]
                f_src_neg = f_post[2][:opt.batch_size, :]
                f_tar_neg = f_post[2][opt.batch_size:, :]
            x = x[0]
            f_mean = f_mean[0]
            f_post = f_post[0]
        vae_loss_dict = utils.loss_fn_new(
            x, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar, z_post, z_prior_mean, z_prior_logvar, z_prior)
        VAE_loss = vae_loss_dict['mse'] + opt.weight_f * \
            vae_loss_dict['kld_f'] + opt.weight_z*vae_loss_dict['kld_z']
        losses_mse.update(vae_loss_dict['mse'].item(), x.size(0))
        losses_klf.update(vae_loss_dict['kld_f'].item(), x.size(0))
        losses_klz.update(vae_loss_dict['kld_z'].item(), x.size(0))
        loss = opt.weight_VAE * VAE_loss

        # (II) loss on latent factor f
        # 1. calculate triplet loss
        if opt.weight_triplet:
            if opt.triplet_type == 'mean':
                trp_src_loss = triplet_loss(
                    f_mean[:opt.batch_size, :], f_src_pos, f_tar_neg)
                trp_tar_loss = triplet_loss(
                    f_mean[opt.batch_size:, :], f_tar_pos, f_src_neg)
            elif opt.triplet_type == 'post':
                trp_src_loss = triplet_loss(
                    f_post[:opt.batch_size, :], f_src_pos, f_tar_neg)
                trp_tar_loss = triplet_loss(
                    f_post[opt.batch_size:, :], f_tar_pos, f_src_neg)
            trp_loss = trp_src_loss + trp_tar_loss
            losses_triplet.update(trp_loss.item(), x.size(0))
            loss += opt.weight_triplet * trp_loss

        # 2. calculate domain classification loss
        if opt.weight_domain:
            pred_domain = pred_domain_all[3]
            source_domain_label = torch.zeros(source_label.size(0)).long()
            target_domain_label = torch.ones(target_label.size(0)).long()
            domain_label = torch.cat(
                (source_domain_label, target_domain_label), 0)
            domain_label = domain_label.cuda(non_blocking=True)
            loss_dompred = criterion_domain(pred_domain, domain_label)
            losses_dom.update(loss_dompred.item(), pred_domain.size(0))
            loss += opt.weight_domain * loss_dompred

        # (III) loss on latent factor z
        # 1. calculate the classification loss on source
        if opt.weight_cls:
            out = pred_video_class[:opt.batch_size, :]
            label = source_label
            loss_classification = criterion_src(out, label)
            tar_psuedo_len = 0
            if opt.use_psuedo == 'Y' and epoch > opt.start_psuedo_step:
                out2 = pred_video_class[opt.batch_size:, :]
                soft_out2 = F.softmax(out2, dim=1)
                prob, pseudo_label = soft_out2.max(dim=1)
                conf_mask = (prob > opt.tar_psuedo_thre).float()
                if sum(conf_mask):
                    pseudo_cls_loss = CE_loss(
                        out2[conf_mask == 1], pseudo_label[conf_mask == 1])
                else:
                    pseudo_cls_loss = 0
                tar_psuedo_len = sum(conf_mask)
                loss_classification += pseudo_cls_loss
            losses_cls.update(loss_classification.item(),
                              opt.batch_size + tar_psuedo_len)
            loss += opt.weight_cls * loss_classification

        # 2. adversarial discriminative model: adversarial loss
        if opt.weight_adv:
            loss_adversarial = 0
            for l in range(len(opt.place_adv)):
                if opt.place_adv[l] == 'Y':
                    pred_domain_source_single = pred_domain_all[l][:opt.batch_size, :].view(
                        -1, pred_domain_all[l].size()[-1])
                    pred_domain_target_single = pred_domain_all[l][opt.batch_size:, :].view(
                        -1, pred_domain_all[l].size()[-1])
                    pred_domain = torch.cat(
                        (pred_domain_source_single, pred_domain_target_single), 0)
                    source_domain_label = torch.zeros(
                        pred_domain_source_single.size(0)).long()
                    target_domain_label = torch.ones(
                        pred_domain_target_single.size(0)).long()
                    domain_label = torch.cat(
                        (source_domain_label, target_domain_label), 0)
                    domain_label = domain_label.cuda(non_blocking=True)
                    loss_adversarial_single = criterion_domain(
                        pred_domain, domain_label)
                    loss_adversarial += loss_adversarial_single
            losses_adv.update(loss_adversarial.item(), pred_domain.size(0))
            loss += opt.weight_adv * loss_adversarial

        # 3. attentive entropy loss
        if opt.add_loss_DA == 'attentive_entropy' and opt.use_attn != 'none':
            loss_entropy = utils.attentive_entropy(
                pred_video_class, pred_domain_all[2])
            losses_entropy.update(loss_entropy.item(),
                                  pred_video_class.size(0))
            loss += opt.weight_entropy * loss_entropy

        # (IV) MI loss on latent factor z and f
        # calculate the mutual infomation of f and z
        if opt.weight_MI:
            _logq_f_tmp = utils.log_density(f_post.unsqueeze(0).repeat(opt.num_segments, 1, 1).view(opt.num_segments, 2*opt.batch_size, 1, opt.f_dim),
                                            f_mean.unsqueeze(0).repeat(opt.num_segments, 1, 1).view(
                                                opt.num_segments, 1, 2*opt.batch_size, opt.f_dim),
                                            f_logvar.unsqueeze(0).repeat(opt.num_segments, 1, 1).view(opt.num_segments, 1, 2*opt.batch_size, opt.f_dim))
            _logq_z_tmp = utils.log_density(z_post.transpose(0, 1).view(opt.num_segments, 2*opt.batch_size, 1, opt.z_dim),
                                            z_post_mean.transpose(0, 1).view(
                                                opt.num_segments, 1, 2*opt.batch_size, opt.z_dim),
                                            z_post_logvar.transpose(0, 1).view(opt.num_segments, 1, 2*opt.batch_size, opt.z_dim))
            _logq_fz_tmp = torch.cat((_logq_f_tmp, _logq_z_tmp), dim=3)

            logq_f = (utils.logsumexp(_logq_f_tmp.sum(
                3), dim=2, keepdim=False) - math.log(2*opt.batch_size * opt.dataset_size))
            logq_z = (utils.logsumexp(_logq_z_tmp.sum(
                3), dim=2, keepdim=False) - math.log(2*opt.batch_size * opt.dataset_size))
            logq_fz = (utils.logsumexp(_logq_fz_tmp.sum(
                3), dim=2, keepdim=False) - math.log(2*opt.batch_size * opt.dataset_size))

            loss_MI = F.relu(logq_fz - logq_f - logq_z).mean()
            losses_MI.update(loss_MI.item(), 2*opt.batch_size)
            loss += opt.weight_MI * loss_MI

        losses.update(loss.item())
        optimizer.zero_grad()

        if opt.pretrain_VAE == 'Y':
            VAE_loss.backward()
        else:
            loss.backward()

        if opt.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), opt.clip_gradient)
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()

        if opt.lr_adaptive == 'dann':
            utils.adjust_learning_rate_dann(optimizer, p, opt)

        if opt.print_details and opt.pretrain_VAE == 'Y':
            utils.print_log('[%02d][%02d/%02d] | lr: %.6f | batchtime: %.3f | loss: %.4f | mse: %.4f | kld_f: %.4f | kld_z: %.4f' % (epoch, i,
                            epoch_size-1, optimizer.param_groups[0]['lr'], batch_time.avg, losses.avg, losses_mse.avg, losses_klf.avg, losses_klz.avg), train_file)
        elif opt.print_details and opt.pretrain_VAE == 'N':
            utils.print_log('[%02d][%02d/%02d] | lr: %.6f | batchtime: %.3f | loss: %.4f | mse: %.4f | kld_f: %.4f | kld_z: %.4f | mi: %.4f | triple: %.4f | cls_domain: %.4f | adv_loss: %.4f | atten_entropy: %.4f | cls_video: %.4f' % (epoch, i, epoch_size-1, optimizer.param_groups[0]['lr'], batch_time.avg, losses.avg, losses_mse.avg, losses_klf.avg, losses_klz.avg, losses_MI.avg,
                                                                                                                                                                                                                                           losses_triplet.avg, losses_dom.avg, losses_adv.avg, losses_entropy.avg, losses_cls.avg), train_file)
        if i == epoch_size-1 and opt.pretrain_VAE == 'Y':
            utils.print_log('[%02d][%02d/%02d] | lr: %.6f | batchtime: %.3f | loss: %.4f | mse: %.4f | kld_f: %.4f | kld_z: %.4f' % (epoch, i,
                            epoch_size-1, optimizer.param_groups[0]['lr'], batch_time.avg, losses.avg, losses_mse.avg, losses_klf.avg, losses_klz.avg), train_file)
        elif i == epoch_size-1 and opt.pretrain_VAE == 'N':
            utils.print_log('[%02d][%02d/%02d] | lr: %.6f | batchtime: %.3f | loss: %.4f | mse: %.4f | kld_f: %.4f | kld_z: %.4f | mi: %.4f | triple: %.4f | cls_domain: %.4f | adv_loss: %.4f | atten_entropy: %.4f | cls_video: %.4f' % (epoch, i, epoch_size-1, optimizer.param_groups[0]['lr'], batch_time.avg, losses.avg, losses_mse.avg, losses_klf.avg, losses_klz.avg, losses_MI.avg,
                                                                                                                                                                                                                                           losses_triplet.avg, losses_dom.avg, losses_adv.avg, losses_entropy.avg, losses_cls.avg), train_file)

    return losses.avg, losses_cls.avg


def validate(val_loader, model):
    top1 = AverageMeter()

    model.eval()

    iter_val = iter(val_loader)
    val_size = len(iter_val)

    for i in range(val_size):
        val_dataloader = iter_val.next()
        val_data = val_dataloader[0].cuda()
        val_label = val_dataloader[1]

        val_size_ori = val_data.size()
        batch_val_ori = val_size_ori[0]

        val_label = val_label.cuda(non_blocking=True)

        x = torch.cat((val_data, val_data), dim=0)
        with torch.no_grad():
            _, _, _, _, _, _, _, _, _, _, _, pred_video_class = model(x, [0]*len(opt.beta))
            pred = pred_video_class[:batch_val_ori, :]
            prec1, prec5 = utils.accuracy(pred.data, val_label, topk=(1, 5))
            top1.update(prec1.item(), val_label.size(0))

    return top1.avg


def main(opt):
    global best_prec1
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_config = 'TranSVAE-%s-%s(%s)-%s-usePsue_%s(%.4f)-frames_%d' % (opt.dataset, opt.input_type, opt.backbone, opt.frame_aggregation, opt.use_attn, opt.tar_psuedo_thre, opt.num_segments)

    learning_config = 'Optimizer_%s-lr_%.3f-batchsize_%d-fc_dim=%d-z_dim=%d-f_dim=%d-weighted_class_%s-weighted_domain_%s' % (
        opt.optimizer, opt.lr, opt.batch_size, opt.fc_dim, opt.z_dim, opt.f_dim, opt.weighted_class_loss, opt.weighted_class_loss_DA)

    weight_config = 'Weights_VAE(%.4f)-kl_f=%.4f-kl_z=%.4f-MI=%.4f-Weights_zf-triple(%s)=%.4f-Weights_domain=%.4f-Weigth_zt-weight_cls=%.4f-weight_adversarial=%.4f_[%.4f_%.4f_%.4f]-weight_entropy=%.4f-seed=%d' % (
        opt.weight_VAE, opt.weight_f, opt.weight_z, opt.weight_MI, opt.triplet_type, opt.weight_triplet, opt.weight_domain, opt.weight_cls, opt.weight_adv, opt.beta[0], opt.beta[1], opt.beta[2], opt.weight_entropy, opt.seed)

    localtime = time.asctime(time.localtime(time.time()))
    localtime2 = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    path_exp = opt.exp_dir + '/' + model_config + '/' + \
        learning_config + '/' + weight_config + '/'
    
    if not os.path.isdir(path_exp):
        os.makedirs(path_exp)

    pretrain_exp = path_exp + 'pretrained_model/'
    if not os.path.isdir(pretrain_exp):
        os.makedirs(pretrain_exp)

    if opt.log_indicator == 1:
        train_file = path_exp + 'train_log_{}.txt'.format(localtime2)
    else:
        train_file = None
    
    utils.print_log("Run time: {}".format(localtime), train_file)
    utils.print_log("Random Seed: {}".format(opt.seed), train_file)
    utils.print_log('Running parameters:', train_file)
    utils.print_log(json.dumps(vars(opt), indent=4, separators=(',', ':')), train_file)

    model = TranSVAE.TranSVAE_Video(opt)
    if not opt.parallel_train:
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    if opt.optimizer == 'SGD':
        utils.print_log('using SGD', train_file)
        optimizer = torch.optim.SGD(model.parameters(
        ), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optimizer == 'Adam':
        utils.print_log('using Adam', train_file)
        optimizer = torch.optim.Adam(
            model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    else:
        utils.print_log('optimizer not support or specified!!!', train_file)
        exit()

    start_epoch = 1
    print('checking the checkpoint......')
    if opt.resume:
        if os.path.isfile(path_exp + opt.resume):
            checkpoint = torch.load(path_exp + opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            utils.print_log(("=> loaded checkpoint '{}' (epoch {})".format(
                opt.resume, checkpoint['epoch'])), train_file)
            if opt.resume_hp:
                utils.print_log(
                    "=> loaded checkpoint hyper-parameters", train_file)
                optimizer.load_state_dict(checkpoint['optimizer'])
        # if load the pretrained vae
        elif os.path.isfile('pretrained_model/' + opt.resume):
            checkpoint = torch.load('pretrained_model/' + opt.resume)
            model.load_state_dict(checkpoint['state_dict'])
            utils.print_log(
                ("=> loaded pretrained VAE '{}'".format(opt.resume)), train_file)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        model.apply(utils.init_weights)

    utils.print_log(model, train_file)
    utils.print_log('========== start: ' + str(start_epoch),
                    train_file)

    utils.print_log('loading data...', train_file)
    num_source = 900
    num_target = 900
    opt.dataset_size = num_source + num_target

    weight_source_class = torch.ones(opt.num_class).cuda()
    weight_domain_loss = torch.Tensor([1, 1]).cuda()

    criterion_src = torch.nn.CrossEntropyLoss(
        weight=weight_source_class).cuda()
    criterion_domain = torch.nn.CrossEntropyLoss(
        weight=weight_domain_loss).cuda()

    datapath = opt.data_root + '/' + opt.dataset + '/' + 'npy/'
    X_src, X_tar, y_attribute_src, y_attribute_tar, y_action_src, y_action_tar = utils.sprites_loaddata(
        datapath, opt.src, opt.tar)

    src_data = Sprite(train=True, data=X_src, A_label=y_attribute_src,
                      D_label=y_action_src, triple=opt.weight_triplet)
    tar_data = Sprite(train=False, data=X_tar, A_label=y_attribute_tar,
                      D_label=y_action_tar, triple=opt.weight_triplet)

    source_loader = DataLoader(src_data,
                               num_workers=opt.data_threads,
                               batch_size=opt.batch_size,
                               shuffle=True,
                               drop_last=True,
                               pin_memory=True)

    target_loader = DataLoader(tar_data,
                               num_workers=opt.data_threads,
                               batch_size=opt.batch_size,  # 8
                               shuffle=False,
                               drop_last=True,
                               pin_memory=True)
    val_loader = target_loader

    loss_c_current = 999  # random large number
    loss_c_previous = 999  # random large number

    if opt.pretrain_VAE == 'Y':
        is_pretrain = True
        utils.print_log('Pretraining VAE part......', train_file)
        for epoch in range(start_epoch, start_epoch + opt.epochs + 1):
            if opt.lr_adaptive == 'loss':
                utils.adjust_learning_rate_loss(
                    optimizer, opt.lr_decay, loss_c_current, loss_c_previous, '>')
            elif opt.lr_adaptive == 'none' and epoch in opt.lr_steps:
                utils.adjust_learning_rate(optimizer, opt.lr_decay)

            loss, loss_c = train(source_loader, target_loader, model, optimizer,
                                 train_file, epoch, criterion_src, criterion_domain, opt)

            loss_c_previous = loss_c_current
            loss_c_current = loss_c

            if epoch % opt.eval_freq == 0 or epoch == opt.epochs:
                prec1 = validate(val_loader, model)
                is_best = False
                if opt.save_model:
                    utils.save_checkpoint({
                        'epoch': 0,
                        'backbone': opt.backbone,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_prec1,
                        'prec1': 0,
                    }, is_best, is_pretrain, pretrain_exp)

    if opt.train_TranSVAE == 'Y':
        utils.print_log('start training TranSVAE......', train_file)
        is_pretrain = False
        for epoch in range(start_epoch, start_epoch + opt.epochs):
            if opt.lr_adaptive == 'loss':
                utils.adjust_learning_rate_loss(
                    optimizer, opt.lr_decay, loss_c_current, loss_c_previous, '>')
            elif opt.lr_adaptive == 'none' and epoch in opt.lr_steps:
                utils.adjust_learning_rate(optimizer, opt.lr_decay)

            loss, loss_c = train(source_loader, target_loader, model, optimizer,
                                 train_file, epoch, criterion_src, criterion_domain, opt)

            loss_c_previous = loss_c_current
            loss_c_current = loss_c

            if epoch % opt.eval_freq == 0 or epoch == opt.epochs:
                prec1 = validate(val_loader, model)

                is_best = prec1 > best_prec1
                line_update = ' ==> updating the best accuracy' if is_best else ''
                line_best = "Best score {} vs current score {}".format(
                    best_prec1, prec1) + line_update
                utils.print_log(line_best, train_file)
                best_prec1 = max(prec1, best_prec1)

                if opt.save_model:
                    utils.save_checkpoint({
                        'epoch': epoch,
                        'backbone': opt.backbone,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_prec1,
                        'prec1': prec1,
                    }, is_best, is_pretrain, path_exp)


if __name__ == '__main__':
    main(opt)
