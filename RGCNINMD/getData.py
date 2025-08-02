import numpy as np
import os
from utils import *


def get_mirna_disease_Kernel(args):
    path = args.path
    device = args.device
    kernel_matrix_edge_index = {}

    m_nums = np.loadtxt(os.path.join(path, 'm_gs.csv'), delimiter=',', dtype=float).shape[0]
    d_nums = np.loadtxt(os.path.join(path, 'd_gs.csv'), delimiter=',', dtype=float).shape[0]

    "miRNA functional sim"
    m_func_sim = np.loadtxt(os.path.join(path, 'm_fs.csv'), delimiter=',', dtype=float)
    m_func_sim = t.tensor(m_func_sim, device=device).to(t.float32)

    "miRNA sequence sim"
    m_seq_sim = np.loadtxt(os.path.join(path, 'm_ss.csv'), delimiter=',', dtype=float)
    m_seq_sim = t.tensor(m_seq_sim, device=device).to(t.float32)

    "miRNA GIP sim"
    m_gip_sim = np.loadtxt(os.path.join(path, 'm_gs.csv'), delimiter=',', dtype=float)
    m_gip_sim = t.tensor(m_gip_sim, device=device).to(t.float32)

    "disease functional sim"
    d_func_sim = np.loadtxt(os.path.join(path, 'd_fs.csv'), delimiter=',', dtype=float)
    d_func_sim = t.tensor(d_func_sim, device=device).to(t.float32)

    "disease semantic sim"
    d_sem_sim = np.loadtxt(os.path.join(path, 'd_ss.csv'), delimiter=',', dtype=float)
    d_sem_sim = t.tensor(d_sem_sim, device=device).to(t.float32)

    "disease GIP sim"
    d_gip_sim = np.loadtxt(os.path.join(path, 'd_gs.csv'), delimiter=',', dtype=float)
    d_gip_sim = t.tensor(d_gip_sim, device=device).to(t.float32)

    "miRNA associate disease"
    m_d = np.loadtxt(os.path.join(path, 'm_d.csv'), delimiter=',', dtype=float)
    m_d = t.tensor(m_d, device=device).to(t.float32)

    kernel_matrix_edge_index['m_d'] = {'matrix': m_d}

    "miRNA functional sim + miRNA sequence sim and Hadamard products"
    m_func_add_seq_sim = (m_func_sim + m_seq_sim) / 2
    m_func_hadamard_seq_sim = m_func_sim * m_seq_sim

    "miRNA functional sim + miRNA GIP sim and Hadamard products"
    m_func_add_gip_sim = (m_func_sim + m_gip_sim) / 2
    m_func_hadamard_gip_sim = m_func_sim * m_gip_sim

    "miRNA sequence sim + miRNA GIP sim and Hadamard products "
    m_seq_add_gip_sim = (m_seq_sim + m_gip_sim) / 2
    m_seq_hadamard_gip_sim = m_seq_sim * m_gip_sim

    "disease functional sim + disease semantic sim and Hadamard products"
    d_func_add_sem_sim = (d_func_sim + d_sem_sim) / 2
    d_func_hadamard_sem_sim = d_func_sim * d_sem_sim

    "disease functional sim + disease GIP sim and Hadamard products"
    d_func_add_gip_sim = (d_func_sim + d_gip_sim) / 2
    d_func_hadamard_gip_sim = d_func_sim * d_gip_sim

    "disease semantic sim + disease GIP sim and Hadamard products"
    d_sem_add_gip_sim = (d_sem_sim + d_gip_sim) / 2
    d_sem_hadamard_gip_sim = d_sem_sim * d_gip_sim

    m_stack_kernel = t.stack((m_func_add_seq_sim, m_func_hadamard_seq_sim, m_func_add_gip_sim,
                              m_func_hadamard_gip_sim, m_seq_add_gip_sim, m_seq_hadamard_gip_sim))

    d_stack_kernel = t.stack((d_func_add_sem_sim, d_func_hadamard_sem_sim, d_func_add_gip_sim,
                              d_func_hadamard_gip_sim, d_sem_add_gip_sim, d_sem_hadamard_gip_sim))

    kernel_matrix_edge_index['m_stack_kernel'] = m_stack_kernel
    kernel_matrix_edge_index['d_stack_kernel'] = d_stack_kernel
    kernel_matrix_edge_index['m_nums'] = m_nums
    kernel_matrix_edge_index['d_nums'] = d_nums

    return kernel_matrix_edge_index


def get_positive_negative_samples(kernel_matrix_edge_index):
    m_d = kernel_matrix_edge_index['m_d']['matrix']
    t.manual_seed(42)
    pos_samples_index = t.where(m_d == 1)  # 返回大小为2的元组 类似coo数据形式
    n = pos_samples_index[0].shape[0]
    pos_samples_index = t.stack(pos_samples_index)
    pos_samples_index_shuffled = pos_samples_index[:, t.randperm(n)]

    neg_samples_index = t.where(m_d == 0)
    n = neg_samples_index[0].shape[0]
    neg_samples_index = t.stack(neg_samples_index)
    neg_samples_index_shuffled = neg_samples_index[:, t.randperm(n)]
    return pos_samples_index_shuffled, neg_samples_index_shuffled
