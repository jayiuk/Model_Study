def get_model(model, **kwargs):
    if model == "netvlad":
        from .netvlad import NetVLAD
        return NetVLAD(num_clusters=64, dim=512, alpha=100)