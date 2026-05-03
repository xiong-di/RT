import warnings
import torch
import models
from src.utils import accuracy
from src import source
from src import sumi
from src import read
from src import tsa
from src import tent
from src import abpem
from src import rsa
from tqdm import tqdm


def initiate(args, ttaloader):
    if args.tta_method == "rsa":
        va_model = models.CAVMAEFT(
            label_dim=args.n_class, modality_specific_depth=11, r=args.r, omega=args.omega
        )
    else:
        va_model = models.CAVMAEFT(
            label_dim=args.n_class, modality_specific_depth=11
        )

    if args.pretrain_path == "None":
        warnings.warn("Note no pre-trained models are specified.")
    else:
        # TTA based on a CAV-MAE finetuned model
        mdl_weight = torch.load(args.pretrain_path)
        if not isinstance(va_model, torch.nn.DataParallel):
            va_model = torch.nn.DataParallel(va_model)
        miss, unexpected = va_model.load_state_dict(mdl_weight, strict=False)
        
    if args.tta_method == 'source':
        va_model = sumi.configure_model(va_model)
        params, param_names = source.collect_params(va_model)

    elif args.tta_method == 'sumi':
        va_model = sumi.configure_model(va_model)
        params, param_names = sumi.collect_params(va_model)

    elif args.tta_method == 'read':
        va_model = read.configure_model(va_model)
        trainables = [p for p in va_model.parameters() if p.requires_grad]
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in va_model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
        params, param_names = read.collect_params(va_model)

    elif args.tta_method == 'tsa':
        va_model = tsa.configure_model(va_model,args)
        params = tsa.collect_params(va_model, args)

    elif args.tta_method == 'tent':
        va_model = tent.configure_model(va_model)
        trainables = [p for p in va_model.parameters() if p.requires_grad]
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in va_model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
        params, param_names = tent.collect_params(va_model)

    elif args.tta_method == 'rsa':
        va_model = rsa.configure_model(va_model)

    elif args.tta_method == 'abpem':
        va_model = abpem.configure_model(va_model)
        trainables = [p for p in va_model.parameters() if p.requires_grad]
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in va_model.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
        params, param_names = abpem.collect_params(va_model)

    if args.tta_method == 'sumi':
        optimizer = torch.optim.Adam(
            [{"params": params, "lr": args.lr}],
            weight_decay=0.0,
            betas=(0.9, 0.999),
        )
    elif args.tta_method == 'read' or  args.tta_method == 'tent' or args.tta_method == 'abpem':
        optimizer = torch.optim.Adam([{'params': params, 'lr': args.lr}],
                                    weight_decay=0.0, 
                                    betas=(0.9, 0.999))
    elif args.tta_method == 'tsa':
        optimizer = torch.optim.Adam([{'params': params, 'lr': args.lr}],
                                    weight_decay=0.0, 
                                    betas=(0.9, 0.999))
    elif args.tta_method == 'rsa':
        optimizer = torch.optim.Adam([{'params': [p for p in va_model.parameters() if p.requires_grad], 'lr': args.lr}],
                                    weight_decay=0.0001, 
                                    betas=(0.9, 0.999))
        
    if not isinstance(va_model, torch.nn.DataParallel):
        va_model = torch.nn.DataParallel(va_model)

    va_model.cuda()
    train_model(va_model, optimizer, ttaloader, args)
    
def train_model(model, optimizer, ttaloader, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tta_method == 'source':
        tta_model = source.Source(model, optimizer, device, args)
    elif args.tta_method == 'sumi':
        tta_model = sumi.SuMi(model, optimizer, device, args)
    elif args.tta_method == 'read':
        tta_model = read.READ(model, optimizer, device, args)
    elif args.tta_method == 'tsa':
        tta_model = tsa.TSA(model, optimizer, device, args)
    elif args.tta_method == 'tent':
        tta_model = tent.TENT(model, optimizer, device, args)
    elif args.tta_method == 'rsa':
        tta_model = rsa.RSA(model, optimizer, device, args)
    elif args.tta_method == 'abpem':
        tta_model = abpem.ABPEM(model, optimizer, device, args)
    tta_model.eval()

    with torch.no_grad():
        for epoch in range(1):
            data_bar = tqdm(ttaloader)
            batch_accs = []
            iters = len(data_bar)

            for i, (a_input, v_input, corrupt_type, labels) in enumerate(data_bar):

                a_input = a_input.to(device)
                v_input = v_input.to(device)

                if args.tta_method == 'sumi':
                    outputs,loss = tta_model((a_input, v_input), i, adapt_flag=True)  
   
                elif args.tta_method == 'read':
                    outputs,loss = tta_model((a_input, v_input), i, adapt_flag=True)

                elif args.tta_method == 'tent':
                    outputs,loss = tta_model((a_input, v_input), i, adapt_flag=True)

                elif args.tta_method == 'abpem':
                    outputs,loss = tta_model((a_input, v_input), i, adapt_flag=True)
                    
                elif args.tta_method == 'tsa':
                    outputs,loss = tta_model((a_input, v_input), i, adapt_flag=True)

                elif args.tta_method == 'rsa':
                    outputs,loss = tta_model((a_input, v_input), i, adapt_flag=True)

                else:
                    outputs = tta_model((a_input, v_input), adapt_flag=False)
                        # now it infers and adapts!

                batch_acc = accuracy(outputs[1], labels, topk=(1,))
                batch_acc = round(batch_acc[0].item(), 2)
                batch_accs.append(batch_acc)

                data_bar.set_description(
                    f"Batch#{i}:  ACC#{batch_acc:.2f}"
                )

            epoch_acc = round(sum(batch_accs) / len(batch_accs), 2)

            print(f"Epoch{epoch}: all acc is {epoch_acc}")
