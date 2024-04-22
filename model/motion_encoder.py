import torch.nn as nn
import torch


class ClassifierHead(nn.Module):
    def __init__(self, params, num_classes=3, num_joints=17):
        super(ClassifierHead, self).__init__()
        self.params = params
        input_dim = self._get_input_dim(num_joints)
        if self.params['medication']:
            input_dim += 1
        if len(self.params['metadata']) > 0:
            input_dim += len(self.params['metadata'])
        self.dims = [input_dim, *self.params['classifier_hidden_dims'], num_classes]

        self.fc_layers = self._create_fc_layers()
        self.batch_norms = self._create_batch_norms()
        self.dropout = nn.Dropout(p=self.params['classifier_dropout'])
        self.activation = nn.ReLU()

    def _create_fc_layers(self):
        fc_layers = nn.ModuleList()
        mlp_size = len(self.dims)

        for i in range(mlp_size - 1):
            fc_layer = nn.Linear(in_features=self.dims[i],
                                 out_features=self.dims[i+1])
            fc_layers.append(fc_layer)
        
        return fc_layers
    
    def _create_batch_norms(self):
        batch_norms = nn.ModuleList()
        n_batchnorms = len(self.dims) - 2
        if n_batchnorms == 0:
            return batch_norms
        
        for i in range(n_batchnorms):
            batch_norm = nn.BatchNorm1d(self.dims[i+1], momentum=0.1)
            batch_norms.append(batch_norm)
        
        return batch_norms

    def _get_input_dim(self, num_joints):
        backbone = self.params['backbone']
        if backbone == 'poseformer':
            if self.params['preclass_rem_T']:
                return self.params['model_dim']
            else:
                return self.params['model_dim'] * self.params['source_seq_len']
        elif backbone == "motionbert":
            if self.params['merge_joints']:
                return self.params['dim_rep']
            else:
                return self.params['dim_rep'] * num_joints
        elif backbone == 'poseformerv2':
            return self.params['embed_dim_ratio'] * num_joints * 2
        elif backbone == "mixste":
            if self.params['merge_joints']:
                return self.params['embed_dim_ratio']
            else:
                return self.params['embed_dim_ratio'] * num_joints
        elif backbone == "motionagformer":
            if self.params['merge_joints']:
                return self.params['dim_rep']
            else:
                return self.params['dim_rep'] * num_joints

    def forward(self, feat):
        feat = self.dropout(feat)
        if self.params['backbone'] == 'motionbert':
            return self._forward_motionbert(feat)
        elif self.params['backbone'] == 'poseformer':
            return self._forward_poseforemer(feat)
        elif self.params['backbone'] == 'poseformerv2':
            return self._forward_poseformerv2(feat)
        elif self.params['backbone'] == "mixste":
            return self._forward_mixste(feat)
        elif self.params['backbone'] == "motionagformer":
            return self._forward_motionagformer(feat)

    def _forward_fc_layers(self, feat):
        mlp_size = len(self.dims)
        for i in range(mlp_size - 2):
            fc_layer = self.fc_layers[i]
            batch_norm = self.batch_norms[i]

            feat = self.activation(batch_norm(fc_layer(feat)))

        last_fc_layer = self.fc_layers[-1]
        feat = last_fc_layer(feat)
        return feat
    
    def _forward_motionagformer(self, feat):
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        feat  = self._forward_fc_layers(feat)
        return feat
    
    def _forward_mixste(self, feat):
        """
        x: Tensor with shape (batch_size, n_frames, n_joints, dim_representation)
        """
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        feat  = self._forward_fc_layers(feat)
        return feat

    def _forward_poseformerv2(self, feat):
        """
        x: Tensor with shape (batch_size, 1, embed_dim_ratio * num_joints * 2)
        """
        B, _, C = feat.shape
        feat = feat.reshape(B, C)  # (B, 1, C) -> (B, C)
        feat  = self._forward_fc_layers(feat)
        return feat

    def _forward_motionbert(self, feat):
        """
        x: Tensor with shape (batch_size, n_frames, n_joints, dim_representation)
        """
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        feat  = self._forward_fc_layers(feat)
        return feat

    def _forward_poseforemer(self, feat):
        """
        x: Tensor with shape (batch_size, n_frames, dim_representation)
        """
        T, B, C = feat.shape
        if self.params['preclass_rem_T']:
            # Reshape the tensor to (B, 1, C, T)   J=1
            feat = feat.permute(1, 2, 0).unsqueeze(1)
            feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        else:
            feat = feat.permute(1, 0, 2)  # (B, T, C)

        feat = feat.reshape(B, -1)  # (B, J * C) or (B, T * C)
        feat  = self._forward_fc_layers(feat)
        return feat


class MotionEncoder(nn.Module):
    def __init__(self, backbone, params, num_classes=4, num_joints=17, train_mode='end2end'):
        super(MotionEncoder, self).__init__()
        assert train_mode in ['end2end', 'classifier_only'], "train_mode should be either end2end or classifier_only." \
                                                             f" Found {train_mode}"
        self.backbone = backbone
        if train_mode == 'classifier_only':
            self.freeze_backbone()
        self.head = ClassifierHead(params, num_classes=num_classes, num_joints=num_joints)
        self.num_classes = num_classes
        self.medprob = params['medication']
        self.metadata = params['metadata']

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[INFO - MotionEncoder] Backbone parameters are frozen")

    def forward(self, x, metadata, med=None):
        """
        x: Tensor with shape (batch_size, n_frames, n_joints, C=3)
        """
        feat = self.backbone(x)
        if self.medprob and med is not None:
            med = med.to(feat.device)
            med = med.view(*[-1] + [1] * (feat.dim() - 1))
            s = list(feat.shape)
            s[-1] = 1  # Set the last dimension to 1
            med = med.expand(*s)
            feat = torch.cat((feat, med), dim=-1)
        if len(self.metadata) > 0:
            metadata = metadata.view(metadata.shape[0], *([1] * (feat.dim() - 2)), metadata.shape[-1])
            metadata = metadata.expand(*feat.shape[:-1], metadata.shape[-1])
            feat = torch.cat((feat, metadata), dim=-1)
        out = self.head(feat)
        return out


def _test_classifier_head():
    params = {
        "backbone": "motionbert",
        "dim_rep": 512,
        "classifier_hidden_dims": [],
        'classifier_dropout': 0.5
    }
    head = ClassifierHead(params, num_classes=3, num_joints=17)

    B, T, J, C = 4, 243, 17, 512
    feat = torch.randn(B, T, J, C)
    out = head(feat)
    assert out.shape == (4, 3)

if __name__ == "__main__":
    _test_classifier_head()