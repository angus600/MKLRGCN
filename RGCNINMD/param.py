import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--path', type=str, default='./HMDD3')

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--kfolds', type=int, default=5)

    parser.add_argument('--GCNlayer', type=int, default=2)
    parser.add_argument('--RGCNlayer', type=int, default=2)
    parser.add_argument('--proj', type=int, default=2048)

    parser.add_argument('--mthreshold', type=list, default=[0.75, 0.5, 0.25])
    parser.add_argument('--dthreshold', type=list, default=[0.75, 0.5, 0.25])
    "Heterogeneous 如果m-d中m的度位于百分比i前就作为关键mirna"
    parser.add_argument('--mtop_i_percent', type=int, default=0.2)
    parser.add_argument('--dtop_i_percent', type=int, default=0.2)
    "isomorphic 如果AAT的值大于等于i就把mirna与mirna作为高共现关系"
    parser.add_argument('--m_common_i', type=int, default=10)
    parser.add_argument('--d_common_i', type=int, default=10)

    parser.add_argument('--k', type=int, default=512)
    return parser.parse_args()
