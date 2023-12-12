from .unet import Unet

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
    }
}

def getmodel(args):
    return model_mapping[args.ds][args.task](args=args)