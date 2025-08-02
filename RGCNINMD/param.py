import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--path', type=str, default='./HMDD3')

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument('--kfolds', type=int, default=5)


    parser.add_argument('--GCNlayer', type=int, default=1)
    parser.add_argument('--RGCNlayer', type=int, default=3)
    parser.add_argument('--proj', type=int, default=1000)


   
    parser.add_argument('--mtop_i_percent', type=int, default=0.1)
    parser.add_argument('--dtop_i_percent', type=int, default=0.1)
  
    parser.add_argument('--m_common_i', type=int, default=10)
    parser.add_argument('--d_common_i', type=int, default=9)

    return parser.parse_args()
