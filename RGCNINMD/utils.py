import torch as t


def get_edge_index(kernel):
    return t.nonzero(kernel != 0).T

def get_2th_bigraph(kernel_matrix_edge_index):
    m_d = kernel_matrix_edge_index['m_d']['matrix']
    m_d_transpose = m_d.T
    AAT = m_d @ m_d_transpose
    ATA = m_d_transpose @ m_d
    the_2th_bigraph = t.vstack([t.hstack([AAT, m_d]), t.hstack([m_d_transpose, ATA])])
    kernel_matrix_edge_index['2th_bigraph'] = the_2th_bigraph


def get_m_d_degree(kernel_matrix_edge_index):
    m_d = kernel_matrix_edge_index['m_d']['matrix']
    m_d_degree = t.sum(m_d, dim=1)
    d_m_degree = t.sum(m_d.T, dim=1)
    kernel_matrix_edge_index['m_d_degree'] = m_d_degree
    kernel_matrix_edge_index['d_m_degree'] = d_m_degree

def get_central_miRNA_and_Disease(kernel_matrix_edge_index, args):
    m_percent = 1 - args.mtop_i_percent
    d_percent = 1 - args.dtop_i_percent
    m_d_degree = kernel_matrix_edge_index['m_d_degree']
    d_m_degree = kernel_matrix_edge_index['d_m_degree']
    m_top_num = t.quantile(m_d_degree, m_percent, interpolation='nearest')
    d_top_num = t.quantile(d_m_degree, d_percent, interpolation='nearest')
    central_m_list = t.where(m_d_degree >= m_top_num, 1., 0.)
    central_d_list = t.where(d_m_degree >= d_top_num, 1., 0.)
    kernel_matrix_edge_index['central_m_list'] = central_m_list
    kernel_matrix_edge_index['central_d_list'] = central_d_list


def get_high_co_occurrence(kernel_matrix_edge_index, args):
    m_d = kernel_matrix_edge_index['m_d']['matrix']
    m_co_occurrence_matrix = m_d @ m_d.T
    d_co_occurrence_matrix = m_d.T @ m_d
    m_co_occurrence = t.where(m_co_occurrence_matrix >= args.m_common_i, m_co_occurrence_matrix, 0.)
    d_co_occurrence = t.where(d_co_occurrence_matrix >= args.d_common_i, d_co_occurrence_matrix, 0.)
    m_not_co_occurrence = t.where(m_co_occurrence_matrix < args.m_common_i, m_co_occurrence_matrix, 0.)
    d_not_co_occurrence = t.where(d_co_occurrence_matrix < args.d_common_i, d_co_occurrence_matrix, 0.)
    kernel_matrix_edge_index['m_co_occurrence'] = m_co_occurrence
    kernel_matrix_edge_index['d_co_occurrence'] = d_co_occurrence

    m_co_occurrence_diagonal = t.diagonal(m_co_occurrence)
    d_co_occurrence_diagonal = t.diagonal(d_co_occurrence)
    m_co_occurrence_diagonal = t.diag(m_co_occurrence_diagonal)
    d_co_occurrence_diagonal = t.diag(d_co_occurrence_diagonal)

    m_co_occurrence = m_co_occurrence - m_co_occurrence_diagonal
    d_co_occurrence = d_co_occurrence - d_co_occurrence_diagonal

    m_co_occurrence_degree = t.sum(m_co_occurrence, dim=1, keepdim=True)
    d_co_occurrence_degree = t.sum(d_co_occurrence, dim=1, keepdim=True)

    m_not_co_occurrence_degree = t.sum(m_not_co_occurrence, dim=1, keepdim=True)
    d_not_co_occurrence_degree = t.sum(d_not_co_occurrence, dim=1, keepdim=True)

    m_co_occurrence_and_not_degree = t.hstack((m_co_occurrence_degree, m_not_co_occurrence_degree))
    d_co_occurrence_and_not_degree = t.hstack((d_co_occurrence_degree, d_not_co_occurrence_degree))

    kernel_matrix_edge_index['m_co_occurrence_and_not_degree'] = m_co_occurrence_and_not_degree
    kernel_matrix_edge_index['d_co_occurrence_and_not_degree'] = d_co_occurrence_and_not_degree
