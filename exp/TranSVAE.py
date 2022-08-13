import torch
import torch.nn.functional as F
import torch.nn as nn
import TRNmodule
import torchvision
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None    
    

class TranSVAE_Video(nn.Module):
    def __init__(self, opt):
        super(TranSVAE_Video, self).__init__()
        self.f_dim = opt.f_dim
        self.z_dim = opt.z_dim
        self.fc_dim = opt.fc_dim
        self.channels = opt.channels
        self.input_type = opt.input_type
        self.frames = opt.num_segments
        self.use_bn = opt.use_bn
        self.frame_aggregation = opt.frame_aggregation
        self.batch_size = opt.batch_size
        self.use_attn = opt.use_attn
        self.dropout_rate = opt.dropout_rate
        self.num_class = opt.num_class
        self.prior_sample = opt.prior_sample
        
        if self.input_type == 'image':
            from models import dcgan_64
            self.encoder = dcgan_64.encoder(self.fc_dim, self.channels)
            self.decoder = dcgan_64.decoder_woSkip(self.z_dim + self.f_dim, self.channels)
            self.fc_output_dim = self.fc_dim
        
        elif self.input_type == 'feature':
            if opt.backbone == 'resnet101':
                model_backnone = getattr(torchvision.models, opt.backbone)(True)
                self.input_dim = model_backnone.fc.in_features
            elif opt.backbone == 'I3Dpretrain':
                self.input_dim = 2048
            elif opt.backbone == 'I3Dfinetune':
                self.input_dim = 2048
            self.add_fc = opt.add_fc
            self.enc_fc_layer1 = nn.Linear(self.input_dim, self.fc_dim)
            self.dec_fc_layer1 = nn.Linear(self.fc_dim, self.input_dim)
            self.fc_output_dim = self.fc_dim    
            
            if self.use_bn == 'shared':
                self.bn_enc_layer1 = nn.BatchNorm1d(self.fc_output_dim)
                self.bn_dec_layer1 = nn.BatchNorm1d(self.input_dim)
            elif self.use_bn == 'separated':
                self.bn_S_enc_layer1 = nn.BatchNorm1d(self.fc_output_dim) 
                self.bn_T_enc_layer1 = nn.BatchNorm1d(self.fc_output_dim) 
                self.bn_S_dec_layer1 = nn.BatchNorm1d(self.input_dim) 
                self.bn_T_dec_layer1 = nn.BatchNorm1d(self.input_dim)
            
            if self.add_fc > 1:
                self.enc_fc_layer2 = nn.Linear(self.fc_dim, self.fc_dim)
                self.dec_fc_layer2 = nn.Linear(self.fc_dim, self.fc_dim)
                self.fc_output_dim = self.fc_dim
                if self.use_bn == 'shared':
                    self.bn_enc_layer2 = nn.BatchNorm1d(self.fc_output_dim)
                    self.bn_dec_layer2 = nn.BatchNorm1d(self.fc_dim)
                elif self.use_bn == 'separated':
                    self.bn_S_enc_layer2 = nn.BatchNorm1d(self.fc_output_dim) 
                    self.bn_T_enc_layer2 = nn.BatchNorm1d(self.fc_output_dim)
                    self.bn_S_dec_layer2 = nn.BatchNorm1d(self.fc_dim) 
                    self.bn_T_dec_layer2 = nn.BatchNorm1d(self.fc_dim)
            
            if self.add_fc > 2:
                self.enc_fc_layer3 = nn.Linear(self.fc_dim, self.fc_dim)
                self.dec_fc_layer3 = nn.Linear(self.fc_dim, self.fc_dim)
                self.fc_output_dim = self.fc_dim
                if self.use_bn == 'shared':
                    self.bn_enc_layer3 = nn.BatchNorm1d(self.fc_output_dim)
                    self.bn_dec_layer3 = nn.BatchNorm1d(self.fc_dim)
                elif self.use_bn == 'separated':
                    self.bn_S_enc_layer3 = nn.BatchNorm1d(self.fc_output_dim) 
                    self.bn_T_enc_layer3 = nn.BatchNorm1d(self.fc_output_dim)
                    self.bn_S_dec_layer3 = nn.BatchNorm1d(self.fc_dim) 
                    self.bn_T_dec_layer3 = nn.BatchNorm1d(self.fc_dim)
            
            self.z_2_out = nn.Linear(self.z_dim + self.f_dim, self.fc_output_dim)

        self.relu = nn.LeakyReLU(0.1)
        self.dropout_f = nn.Dropout(p=self.dropout_rate)
        self.dropout_v = nn.Dropout(p=self.dropout_rate)
        self.hidden_dim = opt.z_dim
        self.f_rnn_layers = opt.f_rnn_layers

        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.z_lstm = nn.LSTM(self.fc_output_dim, self.hidden_dim, self.f_rnn_layers, bidirectional=True, batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim * 2, self.f_dim)
        self.f_logvar = nn.Linear(self.hidden_dim * 2, self.f_dim)

        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.fc_feature_domain_frame = nn.Linear(self.z_dim, self.z_dim)
        self.fc_classifier_domain_frame = nn.Linear(self.z_dim, 2)

        if self.frame_aggregation == 'rnn': 
            self.bilstm = nn.LSTM(self.z_dim, self.z_dim * 2, self.f_rnn_layers, bidirectional=True, batch_first=True)
            self.feat_aggregated_dim = self.z_dim * 2
        elif self.frame_aggregation == 'trn':
            self.num_bottleneck = 256
            self.TRN = TRNmodule.RelationModuleMultiScale(self.z_dim, self.num_bottleneck, self.frames)
            self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
            self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
            self.feat_aggregated_dim = self.num_bottleneck

        self.fc_feature_domain_video = nn.Linear(self.feat_aggregated_dim, self.feat_aggregated_dim)
        self.fc_classifier_domain_video = nn.Linear(self.feat_aggregated_dim, 2)

        if self.frame_aggregation == 'trn':
            self.relation_domain_classifier_all = nn.ModuleList()
            for i in range(self.frames-1):
                relation_domain_classifier = nn.Sequential(
                    nn.Linear(self.feat_aggregated_dim, self.feat_aggregated_dim),
                    nn.ReLU(),
                    nn.Linear(self.feat_aggregated_dim, 2)
                )
                self.relation_domain_classifier_all += [relation_domain_classifier]
        
        self.pred_classifier_video = nn.Linear(self.feat_aggregated_dim, self.num_class)
        self.fc_feature_domain_latent = nn.Linear(self.f_dim, self.f_dim)
        self.fc_classifier_doamin_latent = nn.Linear(self.f_dim, 2)
        
        if self.use_attn == 'general':
            self.attn_layer = nn.Sequential(
                nn.Linear(self.feat_aggregated_dim, self.feat_aggregated_dim),
                nn.Tanh(),
                nn.Linear(self.feat_aggregated_dim, 1)
                )
    
    def domain_classifier_frame(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta)
        feat_fc_domain_frame = self.fc_feature_domain_frame(feat_fc_domain_frame)
        feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
        pred_fc_domain_frame = self.fc_classifier_domain_frame(feat_fc_domain_frame)
        return pred_fc_domain_frame
    
    def domain_classifier_video(self, feat_video, beta):
        feat_fc_domain_video = GradReverse.apply(feat_video, beta)
        feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
        feat_fc_domain_video = self.relu(feat_fc_domain_video)
        pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)
        return pred_fc_domain_video
    
    def domain_classifier_latent(self, f):
        feat_fc_domain_latent = self.fc_feature_domain_latent(f)
        feat_fc_domain_latent = self.relu(feat_fc_domain_latent)
        pred_fc_domain_latent = self.fc_classifier_doamin_latent(feat_fc_domain_latent)
        return pred_fc_domain_latent
    
    def domain_classifier_relation(self, feat_relation, beta):
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:,i,:].squeeze(1)
            feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta)
            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1,1,2)
            else:
                pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1,1,2)), 1)
        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1,2)
        return pred_fc_domain_relation_video
    
    def get_trans_attn(self, pred_domain):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 - entropy
        return weights

    def get_general_attn(self, feat):
        num_segments = feat.size()[1]
        feat = feat.view(-1, feat.size()[-1])
        weights = self.attn_layer(feat)
        weights = weights.view(-1, num_segments, weights.size()[-1])
        weights = F.softmax(weights, dim=1)
        return weights
    
    def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
        if self.use_attn == 'TransAttn':
            weights_attn = self.get_trans_attn(pred_domain)
        elif self.use_attn == 'general':
            weights_attn = self.get_general_attn(feat_fc)
        weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1])
        feat_fc_attn = (weights_attn+1) * feat_fc
        return feat_fc_attn, weights_attn[:,:,0]
    
    def encode_and_sample_post(self, x):
        if isinstance(x, list):
            conv_x = self.encoder_frame(x[0])
        else:
            conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        f_mean = self.f_mean(lstm_out_f)
        f_logvar = self.f_logvar(lstm_out_f)
        f_post = self.reparameterize(f_mean, f_logvar, random_sampling=False)
        features, _ = self.z_rnn(lstm_out)
        z_mean = self.z_mean(features)
        z_logvar = self.z_logvar(features)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=False)

        if isinstance(x, list):
            f_mean_list = [f_mean]
            f_post_list = [f_post]
            for t in range(1,3,1):
                conv_x = self.encoder_frame(x[t])
                lstm_out, _ = self.z_lstm(conv_x)
                backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
                frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
                lstm_out_f = torch.cat((frontal, backward), dim=1)
                f_mean = self.f_mean(lstm_out_f)
                f_logvar = self.f_logvar(lstm_out_f)
                f_post = self.reparameterize(f_mean, f_logvar, random_sampling=False)
                f_mean_list.append(f_mean)
                f_post_list.append(f_post)
            f_mean = f_mean_list
            f_post = f_post_list
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post
    
    def decoder_frame(self,zf):
        if self.input_type == 'image':
            recon_x = self.decoder(zf)
            return recon_x
        if self.input_type == 'feature':
            zf = self.z_2_out(zf)
            zf = self.relu(zf)
            if self.add_fc > 2:
                zf = self.dec_fc_layer3(zf)
                if self.use_bn == 'shared':
                    zf = self.bn_dec_layer3(zf)
                elif self.use_bn == 'separated':
                    zf_src = self.bn_S_dec_layer3(zf[:self.batchsize,:,:])
                    zf_tar = self.bn_T_dec_layer3(zf[self.batchsize:,:,:])
                    zf = torch.cat([zf_src,zf_tar],axis=0)
                zf = self.relu(zf)
            if self.add_fc > 1:
                zf = self.dec_fc_layer2(zf)
                if self.use_bn == 'shared':
                    zf = self.bn_dec_layer2(zf)
                elif self.use_bn == 'separated':
                    zf_src = self.bn_S_dec_layer2(zf[:self.batchsize,:,:])
                    zf_tar = self.bn_T_dec_layer2(zf[self.batchsize:,:,:])
                    zf = torch.cat([zf_src,zf_tar],axis=0)
                zf = self.relu(zf)
            zf = self.dec_fc_layer1(zf) 
            if self.use_bn == 'shared':
                zf = self.bn_dec_layer2(zf)
            elif self.use_bn == 'separated':
                zf_src = self.bn_S_dec_layer2(zf[:self.batchsize,:,:])
                zf_tar = self.bn_T_dec_layer2(zf[self.batchsize:,:,:])
                zf = torch.cat([zf_src,zf_tar],axis=0)
            recon_x = self.relu(zf)
            return recon_x

    def encoder_frame(self, x):
        if self.input_type == 'image':
            x_shape = x.shape
            x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
            x_embed = self.encoder(x)[0]           
            return x_embed.view(x_shape[0], x_shape[1], -1)
        if self.input_type == 'feature':
            x_embed = self.enc_fc_layer1(x)
            if self.use_bn == 'shared':
                x_embed = self.bn_enc_layer1(x_embed)
            elif self.use_bn == 'separated':
                x_embed_src = self.bn_S_enc_layer1(x_embed[:self.batchsize,:,:])
                x_embed_tar = self.bn_T_enc_layer1(x_embed[self.batchsize:,:,:])
                x_embed = torch.cat([x_embed_src,x_embed_tar],axis=0)
            x_embed = self.relu(x_embed)
            if self.add_fc > 1:
                x_embed = self.enc_fc_layer2(x_embed)
                if self.use_bn == 'shared':
                    x_embed = self.bn_enc_layer2(x_embed)
                elif self.use_bn == 'separated':
                    x_embed_src = self.bn_S_enc_layer2(x_embed[:self.batchsize,:,:])
                    x_embed_tar = self.bn_T_enc_layer2(x_embed[self.batchsize:,:,:])
                    x_embed = torch.cat([x_embed_src,x_embed_tar],axis=0)
                x_embed = self.relu(x_embed)
            if self.add_fc > 2:
                x_embed = self.enc_fc_layer3(x_embed)
                if self.use_bn == 'shared':
                    x_embed = self.bn_enc_layer3(x_embed)
                elif self.use_bn == 'separated':
                    x_embed_src = self.bn_S_enc_layer3(x_embed[:self.batchsize,:,:])
                    x_embed_tar = self.bn_T_enc_layer3(x_embed[self.batchsize:,:,:])
                    x_embed = torch.cat([x_embed_src,x_embed_tar],axis=0)
                x_embed = self.relu(x_embed)
            return x_embed 
    
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def sample_z_prior_train(self, z_post, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None
        batch_size = z_post.shape[0]
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for i in range(self.frames):
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_post[:,i,:]
        return z_means, z_logvars, z_out

    def sample_z(self, batch_size, random_sampling=True):
        z_out = None
        z_means = None
        z_logvars = None
        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))
            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

    def forward(self, x, beta):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        if self.prior_sample == 'random':
            z_mean_prior, z_logvar_prior, z_prior = self.sample_z(z_post.size(0),random_sampling=False)
        elif self.prior_sample == 'post':
            z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(z_post, random_sampling=False)
        
        if isinstance(f_post, list):
            f_expand = f_post[0].unsqueeze(1).expand(-1, self.frames, self.f_dim)
        else:
            f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)

        recon_x = self.decoder_frame(zf)
        
        pred_domain_all = []
        z_post_feat = z_post.view(-1, z_post.size()[-1])
        z_post_feat = self.dropout_f(z_post_feat)
        pred_fc_domain_frame = self.domain_classifier_frame(z_post_feat, beta[2])
        pred_fc_domain_frame = pred_fc_domain_frame.view((z_post.size(0), self.frames) + pred_fc_domain_frame.size()[-1:])
        pred_domain_all.append(pred_fc_domain_frame)

        if self.frame_aggregation == 'rnn': 
            self.bilstm.flatten_parameters()
            z_post_video_feat, _ = self.bilstm(z_post)
            backward = z_post_video_feat[:, 0, self.z_dim:2 * self.z_dim]
            frontal = z_post_video_feat[:, self.frames - 1, 0:self.z_dim]
            z_post_video_feat = torch.cat((frontal, backward), dim=1)
            pred_fc_domain_relation = []
            pred_domain_all.append(pred_fc_domain_relation)

        elif self.frame_aggregation == 'trn':  
            z_post_video_relation = self.TRN(z_post)
            pred_fc_domain_relation = self.domain_classifier_relation(z_post_video_relation, beta[0])
            pred_domain_all.append(pred_fc_domain_relation.view((z_post.size(0), z_post_video_relation.size()[1]) + pred_fc_domain_relation.size()[-1:]))
            if self.use_attn != 'none':
                z_post_video_relation_attn, _ = self.get_attn_feat_relation(z_post_video_relation, pred_fc_domain_relation, self.frames)
            z_post_video_feat = torch.sum(z_post_video_relation_attn, 1)

        z_post_video_feat = self.dropout_v(z_post_video_feat)

        pred_fc_domain_video = self.domain_classifier_video(z_post_video_feat, beta[1])
        pred_fc_domain_video = pred_fc_domain_video.view((z_post.size(0),) + pred_fc_domain_video.size()[-1:])
        pred_domain_all.append(pred_fc_domain_video)
        
        pred_video_class = self.pred_classifier_video(z_post_video_feat)

        if isinstance(f_post, list):
            pred_fc_domain_latent = self.domain_classifier_latent(f_post[0]) 
        else:
            pred_fc_domain_latent = self.domain_classifier_latent(f_post) 
        pred_domain_all.append(pred_fc_domain_latent)
        
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, recon_x, pred_domain_all, pred_video_class
    
    