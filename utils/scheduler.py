import torch.optim.lr_scheduler as sched

# ==

def define_scheduler(config, optimizer):
    if config.scheduler == 'exp':
        scheduler_instance = sched.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif config.scheduler == 'step':
        scheduler_instance = sched.StepLR(optimizer, step_size=config.num_epoch / 2, gamma=0.1)
    elif config.scheduler == 'plateau':
        scheduler_instance = sched.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.scheduler == 'cosine':
        scheduler_instance = sched.CosineAnnealingLR(optimizer, T_max=config.num_epoch, eta_min=0)
    elif config.scheduler == 'None':
        scheduler_instance = None
    else:
        raise NotImplementedError(f'Scheduler [{config.scheduler}] is not implemented')

    return scheduler_instance
