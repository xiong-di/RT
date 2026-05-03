from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math


class RSA(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device

    def forward(self, x, batch_idx, adapt_flag):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler, batch_idx)
            else:
                outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
                loss = (0, 0)
                outputs = (outputs, outputs)

        return outputs, loss

@torch.enable_grad() 
def forward_and_adapt(x, model, optimizer, args, scaler, batch_idx):
    with autocast():
        outputs, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)

    num_classes = outputs.shape[-1]
    e_margin = getattr(args, 'e_margin', math.log(num_classes) * 0.4)
    
    # Renyi Entropy (alpha=3.0)
    probs = torch.softmax(outputs, dim=-1)
    alpha = args.alpha
    entropys = (1.0 / (1.0 - alpha)) * torch.log((probs ** alpha).sum(dim=-1) + 1e-8)
    
    #  Shannon Entropy
    # entropys = -(probs * probs.log()).sum(dim=-1)

    # Filtering and Weighting
    filter_ids = torch.where(entropys < e_margin)[0]
    
    if len(filter_ids) > 0:
        filtered_entropys = entropys[filter_ids]
        
        weights = torch.exp(-(filtered_entropys.detach() - e_margin)) 

        weights = weights / weights.sum()
        
        loss_ent = (weights * filtered_entropys).sum()
        
        current_mean_prob = probs.mean(dim=0)

        loss_bal = - (current_mean_prob * torch.log(current_mean_prob + 1e-8)).sum()

        loss = loss_ent - 1.0* loss_bal 

        # Update
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        
        scaler.step(optimizer)
        scaler.update()
        
        current_loss = loss.item()
    else:
        current_loss = 0.0

    # Verification / Output
    with torch.no_grad():
        with autocast():
            outputs2, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)

    return (outputs, outputs2), (current_loss, current_loss)
    
        



@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model:nn.Module,bias: str = 'none'):

    for n, p in model.named_parameters():
        if 'w_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    if bias == 'none':
        return model
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True   

    return model


