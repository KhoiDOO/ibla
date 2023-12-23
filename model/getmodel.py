from .unet import Unet
import torchvision

model_mapping = {
    "city" : {
        "semantic" : Unet,
        "depth" : Unet,
    },
    "nyu" : {
        "semantic" : Unet,
        "depth" : Unet,
        "surface" : Unet
    },
    "oxford" : {
        "semantic" : Unet,
    },
    "cifar10" : {
        "clf" : torchvision.models.efficientnet_b0(num_class = 10)
    },
    "cifar100" : {
        "clf" : torchvision.models.efficientnet_b0(num_class = 100)
    }
}

def getmodel(args):
    return model_mapping[args.ds][args.task](args=args)