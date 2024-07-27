from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .swin import swinb
from .vit import vitb
from .resnet import r50, r101

def get_model(name, **kwargs):
    # resnet
    if name == "ir18":
        return iresnet18(False, **kwargs)
    elif name == "ir34":
        return iresnet34(False, **kwargs)
    elif name == "ir50":
        return iresnet50(False, **kwargs)
    elif name == "ir100":
        return iresnet100(False, **kwargs)
    elif name == "ir200":
        return iresnet200(False, **kwargs)
    elif name == 'swinb':
        return swinb()
    elif name == 'vitb':
        return vitb()
    elif name == 'r50':
        return r50()
    elif name == 'r101':
        return r101()
    else:
        raise ValueError()

