from .vanilla import Vanilla

loss_mapping = {
    "vanilla" : Vanilla
}

def loss_map(args):
    return loss_mapping[args.loss](args=args)