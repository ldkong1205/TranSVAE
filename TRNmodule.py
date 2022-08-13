import torch
import torch.nn as nn
from math import ceil


class RelationModule(torch.nn.Module):
    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = num_bottleneck
        self.classifier = self.fc_fusion()

    def fc_fusion(self):
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck),
            nn.ReLU(),
        )
        return classifier

    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input


class RelationModuleMultiScale(torch.nn.Module):
    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]
        self.relations_scales = []
        self.subsample_scales = []

        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(
                min(self.subsample_num, len(relations_scale)))

        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList()
        
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
            )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use',
              ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        act_scale_1 = input[:, self.relations_scales[0][0], :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(
            0), self.scales[0] * self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1)
        act_all = act_scale_1.clone()

        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [int(ceil(
                i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

            for idx in idx_relations_evensample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(
                    0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = act_relation.unsqueeze(1)
                act_relation_all += act_relation

            act_all = torch.cat((act_all, act_relation_all), 1)

        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
