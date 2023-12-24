from model import Unet, CLF
from metrics import miou, pixel_accuracy, depth_error, accuracy, depth_rel_error, depth_abs_error
from loss import VanillaClassifierV0

# MODEL
enc_dec_mapping = {
    'unet' : Unet
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
    "vanilla" : VanillaClassifierV0
}


# MONITOR

mapping = {
    'city' : {
        'seg' : {
            'model' : enc_dec_mapping,
            'metrics' : seg_metrics_mapping
        },
        # 'depth' : {
        #     'model' : enc_dec_mapping,
        #     'metrics' :depth_metrics_mapping
        # }
    },
    'oxford' : {
        'seg' : {
            'model' : enc_dec_mapping,
            'metrics' : seg_metrics_mapping
        }
    },
    'cifar10' : {
        'clf' : {
            'model' : CLF,
            'metrics' : clf_metrics_mapping,
            'loss' : clf_loss
        }
    },
    'cifar100' : {
        'clf' : {
            'model' : CLF,
            'metrics' : clf_metrics_mapping,
            'loss' : clf_loss
        }
    }
}