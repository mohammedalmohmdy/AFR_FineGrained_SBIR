
import yaml, torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.afrsbir import AFRSBIR
from losses.triplet import TripletLoss
from data.datasets import PairDataset
from utils.seed import set_seed

def main(cfg):
    set_seed(cfg['experiment']['seed'])
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds = PairDataset(cfg['dataset']['root'], tfm)
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    model = AFRSBIR(cfg['model']['backbone'], cfg['model']['embedding_dim'], cfg['model']['num_freq_hypotheses']).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    loss_fn = TripletLoss(cfg['training']['margin'])
    for e in range(cfg['training']['epochs']):
        model.train()
        for s,p,n in dl:
            s,p,n = s.cuda(), p.cuda(), n.cuda()
            zs,zp_pos,_,_ = model(s,p)
            _,zp_neg,_,_ = model(s,n)
            loss = loss_fn(zs, zp_pos, zp_neg)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {e}: loss={loss.item():.4f}")

if __name__ == "__main__":
    with open("configs/afrsbir_qmul_shoe.yaml") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
