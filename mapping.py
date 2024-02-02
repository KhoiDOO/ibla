from model import Unet, SegNet, RUnet, AttUnet,\
    Base, get_resnet18, get_resnet18_scratch
from metrics import miou, pixel_accuracy, accuracy, depth_rel_error, depth_abs_error
from loss import VanillaClassifierStableV0, VanillaSegmenterStableV0, CBClassifierV0, CBSegmenterV0, \
    FocalClassifierV0, FocalSegmenterV0, CBFocalClassifierV0, CBFocalSegmenterV0, \
        BSLClassifierV0, BSLSegmenterV0, \
            GumbelFocalClassifierV0, GumbelFocalSegmenterV0, GumbelSegmenterV0, GumbelClassifierV0

# MODEL
enc_dec_mapping = {
    'unet' : Unet,
    'segnet' : SegNet,
    'runet' : RUnet,
    'attunet' : AttUnet
}

clf_model_mapping = {
    'base' : Base,
    'resnet18' : get_resnet18,
    'resnet18_scratch' : get_resnet18_scratch
}

# METRICS
seg_metrics_mapping = {
    'iou' : miou,
    'pixel_accuracy' : pixel_accuracy
}

depth_metrics_mapping = {
    'depth_rel_error' : depth_rel_error,
    'depth_abs_error' : depth_abs_error
}

clf_metrics_mapping = {
    'acc' : accuracy
}

# LOSSES
clf_loss = {
    "vanilla" : VanillaClassifierStableV0,
    "focal" : FocalClassifierV0,
    "cb" : CBClassifierV0,
    "cbfocal" : CBFocalClassifierV0,
    "bsl" : BSLClassifierV0,
    "gumfocal" : GumbelFocalClassifierV0,
    "gum" : GumbelClassifierV0
}

seg_loss = {
    "vanilla" : VanillaSegmenterStableV0,
    "focal" : FocalSegmenterV0,
    "cb" : FocalSegmenterV0,
    "cbfocal" : CBFocalSegmenterV0,
    "bsl" : BSLSegmenterV0,
    "gumfocal" : GumbelFocalSegmenterV0,
    "gum" : GumbelSegmenterV0
}


# MONITOR

mapping = {
    'city' : {
        'seg' : {
            'model' : enc_dec_mapping,
            'metrics' : seg_metrics_mapping,
            'loss' : seg_loss
        },
        # 'depth' : {
        #     'model' : enc_dec_mapping,
        #     'metrics' :depth_metrics_mapping
        # }
    },
    'oxford' : {
        'seg' : {
            'model' : enc_dec_mapping,
            'metrics' : seg_metrics_mapping,
            'loss' : seg_loss
        }
    },
    'cifar10' : {
        'clf' : {
            'model' : clf_model_mapping,
            'metrics' : clf_metrics_mapping,
            'loss' : clf_loss
        }
    },
    'cifar100' : {
        'clf' : {
            'model' : clf_model_mapping,
            'metrics' : clf_metrics_mapping,
            'loss' : clf_loss
        }
    },
    'cifar10lt' : {
        'clf' : {
            'model' : clf_model_mapping,
            'metrics' : clf_metrics_mapping,
            'loss' : clf_loss
        }
    },
    'cifar100lt' : {
        'clf' : {
            'model' : clf_model_mapping,
            'metrics' : clf_metrics_mapping,
            'loss' : clf_loss
        }
    },
    'vocalfolds' : {
        'seg' : {
            'model' : enc_dec_mapping,
            'metrics' : seg_metrics_mapping,
            'loss' : seg_loss
        },
    }
}