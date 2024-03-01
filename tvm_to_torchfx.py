from relay_bridge import RelayModule

def tvm_to_torchfx(mod):
    
    relay_module = RelayModule(mod)
    gm = relay_module.prepare_fx()
    gm.requires_grad_(False)
    gm.eval()

    return gm