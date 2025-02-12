from knw import knw
import textwrap

class nn_networks(knw):
    def __init__(self):
        super().__init__()
        self.name = 'Fixed_points_of_nonnegative_neural_networks'
        self.description = 'This is fixed_points_of_nonnegative_neural_networks which used fixed point theory to analyze nonnegative neural networks, which we define as neural networks that map nonnegative vectors to nonnegative vectors. Variables: networks: nn_sigmoid, learning rate: 5e-3, epochs: 30, wd: 0, b: 64 '
        self.core_function = 'core'
        self.runnable_function = 'runnable'
        self.test_case = 'case_nn_networks'
        self.mode = 'core'

    def core(self):
        case = """
        args = argparse.ArgumentParser()
        args.net = 'nn_sigmoid'
        args.lr = 5e-3
        args.epochs = 30
        args.wd = 0
        args.b = 64
        train_nn_network(args)
        """
        return case

    def runnable(self):
        code = """
        import numpy as np
        import scipy.io as sio
        import scipy
        import sys
        import time
        import argparse
        import torch
        import math
        from torch import nn
        from torch.nn.utils.parametrizations import spectral_norm
        from pathlib import Path
        from torch import optim
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision import datasets
        from tqdm import tqdm
        
        def initialize_weights(tensor):
            return tensor.uniform_() * math.sqrt(0.25 / (tensor.shape[0] + tensor.shape[1]))
            
        class _RRAutoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_1 = nn.Linear(784, 200)
                self.linear_2 = nn.Linear(200, 784)
                self.encoder = self.linear_1
                self.decoder = self.linear_2
    
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
    
                return x
    
            def clamp(self):
                pass
            
        class _NNAutoencoder(_RRAutoencoder):
            def __init__(self):
                super().__init__()
                self.linear_1.bias.data.zero_()
                self.linear_2.bias.data.zero_()
                self.linear_1.weight = nn.Parameter(
                    initialize_weights(self.linear_1.weight.data)
                )
                self.linear_2.weight = nn.Parameter(
                    initialize_weights(self.linear_2.weight.data)
                )
    
            def clamp(self):
                self.linear_1.weight.data.clamp_(min=0)
                self.linear_2.weight.data.clamp_(min=0)
                self.linear_1.bias.data.clamp_(min=0)
                self.linear_2.bias.data.clamp_(min=0)

        class _PNAutoencoder(_NNAutoencoder):
            def clamp(self):
                self.linear_1.weight.data.clamp_(min=1e-3)
                self.linear_2.weight.data.clamp_(min=1e-3)
                self.linear_1.bias.data.clamp_(min=0)
                self.linear_2.bias.data.clamp_(min=0)

        class _NRAutoencoder(_NNAutoencoder):
            def clamp(self):
                self.linear_1.weight.data.clamp_(min=0)
                self.linear_2.weight.data.clamp_(min=0)

        class SigmoidNNAutoencoder(_NNAutoencoder):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(self.linear_1, nn.Sigmoid())
                self.decoder = nn.Sequential(self.linear_2, nn.Sigmoid())

        class TanhNNAutoencoder(_NNAutoencoder):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(self.linear_1, nn.Tanh())
                self.decoder = nn.Sequential(self.linear_2, nn.Tanh())

        class TanhPNAutoencoder(_PNAutoencoder):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(self.linear_1, nn.Tanh())
                self.decoder = nn.Sequential(self.linear_2, nn.Tanh())

        class ReLUNNAutoencoder(_NNAutoencoder):
            def __init__(self):
                super().__init__()
                self.linear_1 = spectral_norm(self.linear_1)
                self.linear_2 = spectral_norm(self.linear_2)
                self.encoder = nn.Sequential(self.linear_1, nn.ReLU())
                self.decoder = nn.Sequential(self.linear_2, nn.ReLU())
    
            def clamp(self):
                self.linear_1.parametrizations.weight.original.data.clamp_(min=0)
                self.linear_2.parametrizations.weight.original.data.clamp_(min=0)
                self.linear_1.bias.data.clamp_(min=0)
                self.linear_2.bias.data.clamp_(min=0)

        class ReLUPNAutoencoder(_PNAutoencoder):
            def __init__(self):
                super().__init__()
                self.linear_1 = spectral_norm(self.linear_1)
                self.linear_2 = spectral_norm(self.linear_2)
                self.encoder = nn.Sequential(self.linear_1, nn.ReLU())
                self.decoder = nn.Sequential(self.linear_2, nn.ReLU())
    
            def clamp(self):
                self.linear_1.parametrizations.weight.original.data.clamp_(min=1e-3)
                self.linear_2.parametrizations.weight.original.data.clamp_(min=1e-3)
                self.linear_1.bias.data.clamp_(min=0)
                self.linear_2.bias.data.clamp_(min=0)
        
        
        class TanhSwishNNAutoencoder(_NNAutoencoder):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(self.linear_1, nn.Tanh())
                self.decoder = nn.Sequential(self.linear_2, nn.SiLU())

        class ReLUSigmoidNRAutoencoder(_NRAutoencoder):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(self.linear_1, nn.ReLU())
                self.decoder = nn.Sequential(self.linear_2, nn.Sigmoid())

        class ReLUSigmoidRRAutoencoder(_RRAutoencoder):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(self.linear_1, nn.ReLU())
                self.decoder = nn.Sequential(self.linear_2, nn.Sigmoid())
            
        def get_network(name):
            match name:
                case "nn_sigmoid":
                    return SigmoidNNAutoencoder()
                case "nn_tanh":
                    return TanhNNAutoencoder()
                case "pn_tanh":
                    return TanhPNAutoencoder()
                case "nn_relu":
                    return ReLUNNAutoencoder()
                case "pn_relu":
                    return ReLUPNAutoencoder()
                case "nn_tanh_swish":
                    return TanhSwishNNAutoencoder()
                case "nr_relu_sigmoid":
                    return ReLUSigmoidNRAutoencoder()
                case "rr_relu_sigmoid":
                    return ReLUSigmoidRRAutoencoder()
                case _:
                    raise NotImplementedError(
                        f"Autoencoder of name '{name}' currently is not supported"
                    )

        class AverageMeter(object):

            def __init__(self):
                self.reset()
    
            def reset(self):
                self.val = 0
                self.avg = 0
                self.sum = 0
                self.count = 0
    
            def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count

        def epoch(loader, model, device, criterion, opt=None):
            losses = AverageMeter()
    
            if opt is None:
                model.eval()
            else:
                model.train()
            for inputs, _ in tqdm(loader, leave=False):
                inputs = inputs.view(-1, 28 * 28).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                if opt:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    model.clamp()
    
                losses.update(loss.item(), inputs.size(0))
    
            return losses.avg
            
        def train_nn_network(args):
            # p = Path(__file__)
            # weights_path = f"{p.parent}/weights"
            # Path(weights_path).mkdir(parents=True, exist_ok=True)
        
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = get_network(args.net)
            model.to(device)
            mnist_train = datasets.MNIST(
                ".", train=True, download=True, transform=transforms.ToTensor()
            )
            mnist_test = datasets.MNIST(
                ".", train=False, download=True, transform=transforms.ToTensor()
            )
            train_loader = DataLoader(
                mnist_train, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True
            )
            test_loader = DataLoader(
                mnist_test, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True
            )
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            criterion = nn.MSELoss()
        
            best_loss = None
        
            for i in range(1, args.epochs + 1):
                train_loss = epoch(train_loader, model, device, criterion, opt)
                test_loss = epoch(test_loader, model, device, criterion)
                if best_loss is None or best_loss > test_loss:
                    best_loss = test_loss
                    # torch.save(model.state_dict(), f"{weights_path}/{args.net}.pth")
        
                print(f"Epoch: {i} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
        """
        return code

if __name__ == '__main__':
    nnn = nn_networks()
    print(nnn.get_core_function())
    print(nnn.runnable())
