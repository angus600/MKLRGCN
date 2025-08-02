import torch.nn as nn
from utils import *
from torch_geometric.nn import GCNConv


class kernel_integrator(nn.Module):
    def __init__(self, m_kernel_nums=6, d_kernel_nums=6):
        super(kernel_integrator, self).__init__()
        self.m_weights = nn.Parameter(t.ones(m_kernel_nums))
        self.d_weights = nn.Parameter(t.ones(d_kernel_nums))

    def forward(self, kernel_matrix_edge_index):
        m_stack_kernel = kernel_matrix_edge_index['m_stack_kernel']
        d_stack_kernel = kernel_matrix_edge_index['d_stack_kernel']
        m_weights = t.softmax(self.m_weights.to(m_stack_kernel.device), dim=0)
        d_weights = t.softmax(self.d_weights.to(d_stack_kernel.device), dim=0)
        m_weights = m_weights.reshape(-1, 1, 1)
        d_weights = d_weights.reshape(-1, 1, 1)
        m_stack_kernel = m_stack_kernel * m_weights
        d_stack_kernel = d_stack_kernel * d_weights
        m_sim = t.sum(m_stack_kernel, dim=0)
        d_sim = t.sum(d_stack_kernel, dim=0)
        m_edge_index = get_edge_index(m_sim)
        d_edge_index = get_edge_index(d_sim)
        return m_sim, d_sim, m_edge_index, d_edge_index


class R_GCN(nn.Module):
    def __init__(self, args, kernel_matrix_edge_index):
        super(R_GCN, self).__init__()
        self.m_nums = kernel_matrix_edge_index['m_nums']

        "perspective of similarity needs"
        # self.bigraph = kernel_matrix_edge_index['bigraph']

        self.m_bigger_than_threshold1 = nn.Linear(args.proj, args.proj)
        self.m_between_threshold1_and_threshold2 = nn.Linear(args.proj, args.proj)
        self.m_between_threshold2_and_threshold3 = nn.Linear(args.proj, args.proj)
        self.m_less_than_threshold3 = nn.Linear(args.proj, args.proj)
        self.m_self_loop = nn.Linear(args.proj, args.proj)

        self.d_bigger_than_threshold1 = nn.Linear(args.proj, args.proj)
        self.d_between_threshold1_and_threshold2 = nn.Linear(args.proj, args.proj)
        self.d_between_threshold2_and_threshold3 = nn.Linear(args.proj, args.proj)
        self.d_less_than_threshold3 = nn.Linear(args.proj, args.proj)
        self.d_self_loop = nn.Linear(args.proj, args.proj)

        self.m_to_d_weight = nn.Linear(args.proj, args.proj)
        self.d_to_m_weight = nn.Linear(args.proj, args.proj)

        # self.m_degree = kernel_matrix_edge_index['m_degree']
        # self.d_degree = kernel_matrix_edge_index['d_degree']
        self.m_d_degree = kernel_matrix_edge_index['m_d_degree']
        self.d_m_degree = kernel_matrix_edge_index['d_m_degree']
        "perspective of degree needs"
        self.the_2th_bigraph = kernel_matrix_edge_index['2th_bigraph']

        self.m_co_occurrence = kernel_matrix_edge_index['m_co_occurrence']
        self.m_co_occurrence_and_not_degree = kernel_matrix_edge_index['m_co_occurrence_and_not_degree']
        self.central_m_list = kernel_matrix_edge_index['central_m_list']

        self.d_co_occurrence = kernel_matrix_edge_index['d_co_occurrence']
        self.d_co_occurrence_and_not_degree = kernel_matrix_edge_index['d_co_occurrence_and_not_degree']
        self.central_d_list = kernel_matrix_edge_index['central_d_list']

        self.m_self_loop_1 = nn.Linear(args.proj, args.proj)
        self.m_co_occurrence_weight = nn.Linear(args.proj, args.proj)
        self.m_not_co_occurrence_weight = nn.Linear(args.proj, args.proj)

        self.d_self_loop_1 = nn.Linear(args.proj, args.proj)
        self.d_co_occurrence_weight = nn.Linear(args.proj, args.proj)
        self.d_not_co_occurrence_weight = nn.Linear(args.proj, args.proj)

        self.central_m_to_central_d = nn.Linear(args.proj, args.proj)
        self.central_d_to_central_m = nn.Linear(args.proj, args.proj)
        self.central_m_to_normal_d = nn.Linear(args.proj, args.proj)
        self.central_d_to_normal_m = nn.Linear(args.proj, args.proj)
        self.normal_m_to_central_d = nn.Linear(args.proj, args.proj)
        self.normal_d_to_central_m = nn.Linear(args.proj, args.proj)
        self.normal_m_to_normal_d = nn.Linear(args.proj, args.proj)
        self.normal_d_to_normal_m = nn.Linear(args.proj, args.proj)

    def perspective_of_degree(self, m_d_feat, args):
        m_nums = self.m_nums
        total_nodes = self.the_2th_bigraph.shape[0]
        d_nums = total_nodes - m_nums
        device = args.device

        m_d_new_feat = t.zeros_like(m_d_feat, device=device)

        if m_nums > 0:

            m_feat = m_d_feat[:m_nums]
            m_co_occurrence = self.m_co_occurrence[:m_nums, :m_nums]

            diag_mask = t.eye(m_nums, dtype=t.bool, device=device)

            co_occur_mask = (~diag_mask) & (m_co_occurrence != 0)

            non_co_occur_mask = (~diag_mask) & (m_co_occurrence == 0)

            self_loop_contrib = self.m_self_loop_1(m_feat)
            m_d_new_feat[:m_nums] = self_loop_contrib + m_d_new_feat[:m_nums]

            if co_occur_mask.any():
                deg0 = self.m_co_occurrence_and_not_degree[:m_nums, 0]
                norm_coeff = 1 / t.sqrt(deg0.unsqueeze(1) * deg0.unsqueeze(0) + 1e-10)

                masked_coeff = norm_coeff * co_occur_mask.float()
                transformed = self.m_co_occurrence_weight(m_feat)
                m_d_new_feat[:m_nums] = t.mm(masked_coeff, transformed) + m_d_new_feat[:m_nums]

            if non_co_occur_mask.any():
                deg1 = self.m_co_occurrence_and_not_degree[:m_nums, 1]
                norm_coeff = 1 / t.sqrt(deg1.unsqueeze(1) * deg1.unsqueeze(0) + 1e-10)

                masked_coeff = norm_coeff * non_co_occur_mask.float()
                transformed = self.m_not_co_occurrence_weight(m_feat)
                m_d_new_feat[:m_nums] = t.mm(masked_coeff, transformed) + m_d_new_feat[:m_nums]

        if d_nums > 0:

            d_feat = m_d_feat[m_nums:]

            d_co_occurrence = self.d_co_occurrence
            diag_mask = t.eye(d_nums, dtype=t.bool, device=device)

            co_occur_mask = (~diag_mask) & (d_co_occurrence != 0)

            non_co_occur_mask = (~diag_mask) & (d_co_occurrence == 0)

            self_loop_contrib = self.d_self_loop_1(d_feat)
            m_d_new_feat[m_nums:] = self_loop_contrib + m_d_new_feat[m_nums:]

            if co_occur_mask.any():
                deg0 = self.d_co_occurrence_and_not_degree[:, 0]
                norm_coeff = 1 / t.sqrt(deg0.unsqueeze(1) * deg0.unsqueeze(0) + 1e-10)

                masked_coeff = norm_coeff * co_occur_mask.float()
                transformed = self.d_co_occurrence_weight(d_feat)
                m_d_new_feat[m_nums:] = t.mm(masked_coeff, transformed) + m_d_new_feat[m_nums:]

            if non_co_occur_mask.any():
                deg1 = self.d_co_occurrence_and_not_degree[:, 1]
                norm_coeff = 1 / t.sqrt(deg1.unsqueeze(1) * deg1.unsqueeze(0) + 1e-10)

                masked_coeff = norm_coeff * non_co_occur_mask.float()
                transformed = self.d_not_co_occurrence_weight(d_feat)
                m_d_new_feat[m_nums:] = t.mm(masked_coeff, transformed) + m_d_new_feat[m_nums:]

        if m_nums > 0 and d_nums > 0:
            assoc_md = self.the_2th_bigraph[:m_nums, m_nums:]

            central_m = self.central_m_list[:m_nums]
            central_d = self.central_d_list[:d_nums]

            norm_coeff = 1 / self.m_d_degree[:m_nums].unsqueeze(1)  # (m_nums, 1)

            mask_cc = (central_m != 0).unsqueeze(1) & (central_d != 0).unsqueeze(0)  # 中心 miRNA - 中心疾病
            mask_cn = (central_m != 0).unsqueeze(1) & (central_d == 0).unsqueeze(0)  # 中心 miRNA - 普通疾病
            mask_nc = (central_m == 0).unsqueeze(1) & (central_d != 0).unsqueeze(0)  # 普通 miRNA - 中心疾病
            mask_nn = (central_m == 0).unsqueeze(1) & (central_d == 0).unsqueeze(0)  # 普通 miRNA - 普通疾病

            d_feat = m_d_feat[m_nums:]

            contrib_cc = self.calc_assoc_contrib(assoc_md, mask_cc, norm_coeff,
                                                 self.central_m_to_central_d,
                                                 self.central_d_to_central_m, d_feat)

            contrib_cn = self.calc_assoc_contrib(assoc_md, mask_cn, norm_coeff,
                                                 self.central_m_to_normal_d,
                                                 self.normal_d_to_central_m, d_feat)

            contrib_nc = self.calc_assoc_contrib(assoc_md, mask_nc, norm_coeff,
                                                 self.normal_m_to_central_d,
                                                 self.central_d_to_normal_m, d_feat)

            contrib_nn = self.calc_assoc_contrib(assoc_md, mask_nn, norm_coeff,
                                                 self.normal_m_to_normal_d,
                                                 self.normal_d_to_normal_m, d_feat)

            m_d_new_feat[:m_nums] = contrib_cc + contrib_cn + contrib_nc + contrib_nn + m_d_new_feat[:m_nums]

        if d_nums > 0 and m_nums > 0:
            assoc_dm = self.the_2th_bigraph[m_nums:, :m_nums]

            central_d = self.central_d_list[:d_nums]
            central_m = self.central_m_list[:m_nums]

            norm_coeff = 1 / self.d_m_degree[:d_nums].unsqueeze(1)  # (d_nums, 1)

            mask_cc = (central_d != 0).unsqueeze(1) & (central_m != 0).unsqueeze(0)  # 中心疾病 - 中心 miRNA
            mask_cn = (central_d != 0).unsqueeze(1) & (central_m == 0).unsqueeze(0)  # 中心疾病 - 普通 miRNA
            mask_nc = (central_d == 0).unsqueeze(1) & (central_m != 0).unsqueeze(0)  # 普通疾病 - 中心 miRNA
            mask_nn = (central_d == 0).unsqueeze(1) & (central_m == 0).unsqueeze(0)  # 普通疾病 - 普通 miRNA

            m_feat = m_d_feat[:m_nums]

            contrib_cc = self.calc_assoc_contrib(assoc_dm, mask_cc, norm_coeff,
                                                 self.central_d_to_central_m,
                                                 self.central_m_to_central_d, m_feat)

            contrib_cn = self.calc_assoc_contrib(assoc_dm, mask_cn, norm_coeff,
                                                 self.central_d_to_normal_m,
                                                 self.normal_m_to_central_d, m_feat)

            contrib_nc = self.calc_assoc_contrib(assoc_dm, mask_nc, norm_coeff,
                                                 self.normal_d_to_central_m,
                                                 self.central_m_to_normal_d, m_feat)

            contrib_nn = self.calc_assoc_contrib(assoc_dm, mask_nn, norm_coeff,
                                                 self.normal_d_to_normal_m,
                                                 self.normal_m_to_normal_d, m_feat)

            m_d_new_feat[m_nums:] = contrib_cc + contrib_cn + contrib_nc + contrib_nn + m_d_new_feat[m_nums:]

        return m_d_new_feat

    def calc_assoc_contrib(self, assoc_matrix, mask, norm_coeff, func1, func2, feat):

        if not mask.any():
            return 0

        masked_assoc = assoc_matrix * mask.float()

        transformed = func1(feat) + func2(feat)  # (m, feat_dim)

        contrib = 1 / 2 * norm_coeff * t.mm(masked_assoc, transformed)  # (n, feat_dim)

        return contrib

    def perspective_of_similarity(self, m_d_feat, args, bigraph, m_degree, d_degree):

        m_d_new_feat = t.zeros_like(m_d_feat, device=args.device)
        m_nums = self.m_nums

        d_nums = bigraph.shape[0] - m_nums

        m_feat = m_d_feat[:m_nums]
        m_sim = bigraph[:m_nums, :m_nums]

        diag_mask = t.eye(m_nums, dtype=t.bool, device=args.device)

        masks = [
            (~diag_mask) & (m_sim >= args.mthreshold[0]),
            (~diag_mask) & (m_sim < args.mthreshold[0]) & (m_sim >= args.mthreshold[1]),
            (~diag_mask) & (m_sim < args.mthreshold[1]) & (m_sim >= args.mthreshold[2]),
            (~diag_mask) & (m_sim < args.mthreshold[2])
        ]

        weight_funcs = [
            self.m_bigger_than_threshold1,
            self.m_between_threshold1_and_threshold2,
            self.m_between_threshold2_and_threshold3,
            self.m_less_than_threshold3
        ]

        for i, (mask, weight_func) in enumerate(zip(masks, weight_funcs)):
            deg_i = m_degree[:, i]
            norm_coeff = 1 / t.sqrt(deg_i.unsqueeze(1) * deg_i.unsqueeze(0) + 1e-10)

            masked_coeff = norm_coeff * mask.float()
            transformed = weight_func(m_feat)
            m_d_new_feat[:m_nums] = t.mm(masked_coeff, transformed) + m_d_new_feat[:m_nums]

        m_d_new_feat[:m_nums] = self.m_self_loop(m_feat) + m_d_new_feat[:m_nums]

        d_feat = m_d_feat[m_nums:]
        d_sim = bigraph[m_nums:, m_nums:]

        diag_mask = t.eye(d_nums, dtype=t.bool, device=args.device)

        masks = [
            (~diag_mask) & (d_sim >= args.dthreshold[0]),
            (~diag_mask) & (d_sim < args.dthreshold[0]) & (d_sim >= args.dthreshold[1]),
            (~diag_mask) & (d_sim < args.dthreshold[1]) & (d_sim >= args.dthreshold[2]),
            (~diag_mask) & (d_sim < args.dthreshold[2])
        ]

        weight_funcs = [
            self.d_bigger_than_threshold1,
            self.d_between_threshold1_and_threshold2,
            self.d_between_threshold2_and_threshold3,
            self.d_less_than_threshold3
        ]

        for i, (mask, weight_func) in enumerate(zip(masks, weight_funcs)):
            deg_i = d_degree[:, i]
            norm_coeff = 1 / t.sqrt(deg_i.unsqueeze(1) * deg_i.unsqueeze(0) + 1e-10)

            masked_coeff = norm_coeff * mask.float()
            transformed = weight_func(d_feat)
            m_d_new_feat[m_nums:] = t.mm(masked_coeff, transformed) + m_d_new_feat[m_nums:]

        m_d_new_feat[m_nums:] = self.d_self_loop(d_feat) + m_d_new_feat[m_nums:]

        assoc_md = bigraph[:m_nums, m_nums:]
        norm_coeff = (1 / self.m_d_degree).unsqueeze(1)
        transformed = self.d_to_m_weight(m_d_feat[m_nums:]) + self.m_to_d_weight(m_d_feat[m_nums:])
        m_d_new_feat[:m_nums] = 1 / 2 * norm_coeff * t.mm(assoc_md, transformed) + m_d_new_feat[:m_nums]

        assoc_dm = bigraph[m_nums:, :m_nums]
        norm_coeff = (1 / self.d_m_degree).unsqueeze(1)
        transformed = self.d_to_m_weight(m_d_feat[:m_nums]) + self.m_to_d_weight(m_d_feat[:m_nums])
        m_d_new_feat[m_nums:] = 1 / 2 * norm_coeff * t.mm(assoc_dm, transformed) + m_d_new_feat[m_nums:]

        return m_d_new_feat

    def forward(self, m_d_feat, args, bigraph, m_degree, d_degree):
        m_d_new_feat_1 = self.perspective_of_similarity(m_d_feat, args, bigraph, m_degree, d_degree)
        m_d_new_feat_2 = self.perspective_of_degree(m_d_feat, args)
        m_d_new_feat = m_d_new_feat_1 / 2 + m_d_new_feat_2 / 2
        return m_d_new_feat


class Model(nn.Module):
    def __init__(self, args, kernel_matrix_edge_index):
        super(Model, self).__init__()
        self.kernel_integrator = kernel_integrator()
        self.init(kernel_matrix_edge_index, args)
        self.m_proj = nn.Linear(kernel_matrix_edge_index['m_nums'], args.proj)
        self.d_proj = nn.Linear(kernel_matrix_edge_index['d_nums'], args.proj)

        self.m_gcn = nn.ModuleList()
        self.d_gcn = nn.ModuleList()

        self.relu = nn.ModuleList()

        self.m_GLU = nn.ModuleList()
        self.d_GLU = nn.ModuleList()

        self.r_gcn = nn.ModuleList()
        self.r_relu = nn.ModuleList()

        self.mlp_1 = nn.Linear(2 * args.proj, args.proj)
        self.mlp_2 = nn.Linear(args.proj, args.proj // 2)
        self.mlp_3 = nn.Linear(args.proj // 2, args.proj // 4)
        self.mlp_4 = nn.Linear(args.proj // 4, args.proj // 8)
        self.mlp_5 = nn.Linear(args.proj // 8, 1)

        for _ in range(args.GCNlayer):
            self.m_gcn.append(GCNConv(args.proj, args.proj))
            self.d_gcn.append(GCNConv(args.proj, args.proj))
            self.relu.append(nn.ReLU())

        for _ in range(args.GCNlayer - 1):
            self.m_GLU.append(nn.Linear(args.proj, args.proj))

        for _ in range(args.GCNlayer - 1):
            self.d_GLU.append(nn.Linear(args.proj, args.proj))

        for _ in range(args.RGCNlayer):
            self.r_gcn.append(R_GCN(args, kernel_matrix_edge_index))
            self.r_relu.append(nn.ReLU())

    def init(self, kernel_matrix_edge_index, args):
        get_m_d_degree(kernel_matrix_edge_index)
        get_2th_bigraph(kernel_matrix_edge_index)
        get_central_miRNA_and_Disease(kernel_matrix_edge_index, args)
        get_high_co_occurrence(kernel_matrix_edge_index, args)

    def forward(self, args, kernel_matrix_edge_index, train_samples):
        "gcn need"
        m_sim, d_sim, m_edge_index, d_edge_index = self.kernel_integrator(kernel_matrix_edge_index)
        bigraph = get_bigraph(kernel_matrix_edge_index, m_sim, d_sim)
        m_degree, d_degree = get_degree(args, m_sim, d_sim)

        m_proj_feat = self.m_proj(m_sim)
        d_proj_feat = self.d_proj(d_sim)
        m_gcn_feat_list = []
        d_gcn_feat_list = []
        m_gcn_feat = m_proj_feat
        d_gcn_feat = d_proj_feat

        "r-gcn need"
        m_d_feat = t.vstack([m_proj_feat, d_proj_feat])
        r_gcn_feat_list = []
        "gcn work"
        for cov, relu in zip(self.m_gcn, self.relu):
            m_gcn_feat = cov(m_gcn_feat, m_edge_index, m_sim[m_edge_index[0], m_edge_index[1]])
            m_gcn_feat = relu(m_gcn_feat)
            m_gcn_feat_list.append(m_gcn_feat)

        for cov, relu in zip(self.d_gcn, self.relu):
            d_gcn_feat = cov(d_gcn_feat, d_edge_index, d_sim[d_edge_index[0], d_edge_index[1]])
            d_gcn_feat = relu(d_gcn_feat)
            d_gcn_feat_list.append(d_gcn_feat)

        "GLU residual"
        for index, glu in enumerate(self.m_GLU):
            m_gcn_feat_list[index] = m_gcn_feat_list[index] * t.sigmoid(glu(m_gcn_feat_list[index]))
        final_m_gcn_feat = t.stack(m_gcn_feat_list).sum(dim=0)

        "GLU residual"
        for index, glu in enumerate(self.d_GLU):
            d_gcn_feat_list[index] = d_gcn_feat_list[index] * t.sigmoid(glu(d_gcn_feat_list[index]))
        final_d_gcn_feat = t.stack(d_gcn_feat_list).sum(dim=0)

        "r-gcn work"
        for r_cov, relu in zip(self.r_gcn, self.r_relu):
            m_d_feat = r_cov(m_d_feat, args, bigraph, m_degree, d_degree)
            m_d_feat = relu(m_d_feat)
            r_gcn_feat_list.append(m_d_feat)
        final_r_gcn_feat = t.stack(r_gcn_feat_list).sum(dim=0)

        m_final_r_gcn_feat = final_r_gcn_feat[:kernel_matrix_edge_index['m_nums'], :]
        d_final_r_gcn_feat = final_r_gcn_feat[kernel_matrix_edge_index['m_nums']:, :]
        # 有多种方法融合 看效果
        m_final_feat = final_m_gcn_feat + m_final_r_gcn_feat
        d_final_feat = final_d_gcn_feat + d_final_r_gcn_feat

        m_train_dataset = m_final_feat[train_samples[0]]
        d_train_dataset = d_final_feat[train_samples[1]]
        train_dataset = t.hstack((m_train_dataset, d_train_dataset))
        train_dataset = self.mlp_1(train_dataset)
        train_dataset = self.mlp_2(train_dataset)
        train_dataset = self.mlp_3(train_dataset)
        train_dataset = self.mlp_4(train_dataset)
        result = self.mlp_5(train_dataset)
        return result


""""
    def perspective_of_similarity(self, m_d_feat, args):
        m_d_new_feat = t.zeros(m_d_feat.shape[0], m_d_feat.shape[1], device=args.device)
        for i in range(self.bigraph.shape[0]):
            print(i)
            for j in range(self.bigraph.shape[1]):
                "This if means miRNA or disease"
                if i < self.m_nums:
                    "This if means sim or association"
                    if j < self.m_nums:
                        status = '0'
                        if i == j:
                            status = 'self-loop'
                            relationship_weight_matrix = self.m_self_loop
                        elif self.bigraph[i, j] >= args.mthreshold[0]:
                            nor_coefficient = 1 / t.sqrt(self.m_degree[i, 0] * self.m_degree[j, 0])
                            relationship_weight_matrix = self.m_bigger_than_threshold1
                        elif args.mthreshold[0] > self.bigraph[i, j] >= args.mthreshold[1]:
                            nor_coefficient = 1 / t.sqrt(self.m_degree[i, 1] * self.m_degree[j, 1])
                            relationship_weight_matrix = self.m_between_threshold1_and_threshold2
                        elif args.mthreshold[1] > self.bigraph[i, j] >= args.mthreshold[2]:
                            nor_coefficient = 1 / t.sqrt(self.m_degree[i, 2] * self.m_degree[j, 2])
                            relationship_weight_matrix = self.m_between_threshold2_and_threshold3
                        else:
                            nor_coefficient = 1 / t.sqrt(self.m_degree[i, 3] * self.m_degree[j, 3])
                            relationship_weight_matrix = self.m_less_than_threshold3
                    else:
                        status = '1'
                        nor_coefficient = 1 / self.m_d_degree[i]
                else:
                    if j >= self.m_nums:
                        status = '0'
                        if i == j:
                            status = 'self-loop'
                            relationship_weight_matrix = self.d_self_loop
                        elif self.bigraph[i, j] >= args.dthreshold[0]:
                            nor_coefficient = 1 / t.sqrt(
                                self.d_degree[i - self.m_nums, 0] * self.d_degree[j - self.m_nums, 0])
                            relationship_weight_matrix = self.d_bigger_than_threshold1
                        elif args.dthreshold[0] > self.bigraph[i, j] >= args.dthreshold[1]:
                            nor_coefficient = 1 / t.sqrt(
                                self.d_degree[i - self.m_nums, 1] * self.d_degree[j - self.m_nums, 1])
                            relationship_weight_matrix = self.d_between_threshold1_and_threshold2
                        elif args.dthreshold[1] > self.bigraph[i, j] >= args.dthreshold[2]:
                            nor_coefficient = 1 / t.sqrt(
                                self.d_degree[i - self.m_nums, 2] * self.d_degree[j - self.m_nums, 2])
                            relationship_weight_matrix = self.d_between_threshold2_and_threshold3
                        else:
                            nor_coefficient = 1 / t.sqrt(
                                self.d_degree[i - self.m_nums, 3] * self.d_degree[j - self.m_nums, 3])
                            relationship_weight_matrix = self.d_less_than_threshold3
                    else:
                        status = '1'
                        nor_coefficient = 1 / self.d_m_degree[i - self.m_nums]
                if status == '0':
                    m_d_new_feat[i] = nor_coefficient * relationship_weight_matrix(m_d_feat[j]) + m_d_new_feat[i]
                elif status == '1':
                    m_d_new_feat[i] = 2 * nor_coefficient * self.bigraph[i, j] * (
                            self.d_to_m_weight(m_d_feat[j]) + self.m_to_d_weight(m_d_feat[j])) + m_d_new_feat[i]
                else:
                    m_d_new_feat[i] = relationship_weight_matrix(m_d_feat[i]) + m_d_new_feat[i]
        return m_d_new_feat
"""

"""""
def perspective_of_degree(self, m_d_feat, args):
    m_d_new_feat = t.zeros(m_d_feat.shape[0], m_d_feat.shape[1], device=args.device)
    for i in range(self.the_2th_bigraph.shape[0]):
        for j in range(self.the_2th_bigraph.shape[1]):
            if i < self.m_nums:
                if j < self.m_nums:
                    status = '0'
                    if i == j:
                        status = 'self-loop'
                        relationship_weight_matrix = self.m_self_loop_1
                    elif self.m_co_occurrence[i, j] != 0:
                        nor_coefficient = 1 / t.sqrt(
                            self.m_co_occurrence_and_not_degree[i, 0] * self.m_co_occurrence_and_not_degree[j, 0])
                        relationship_weight_matrix = self.m_co_occurrence_weight
                    else:
                        nor_coefficient = 1 / t.sqrt(
                            self.m_co_occurrence_and_not_degree[i, 1] * self.m_co_occurrence_and_not_degree[j, 1])
                        relationship_weight_matrix = self.m_not_co_occurrence_weight
                else:
                    status = '1'
                    nor_coefficient = 1 / self.m_d_degree[i]
                    if self.central_m_list[i] != 0 and self.central_d_list[j - self.m_nums] != 0:
                        m_to_d = self.central_m_to_central_d
                        d_to_m = self.central_d_to_central_m
                    elif self.central_m_list[i] != 0 and self.central_d_list[j - self.m_nums] == 0:
                        m_to_d = self.central_m_to_normal_d
                        d_to_m = self.normal_d_to_central_m
                    elif self.central_m_list[i] == 0 and self.central_d_list[j - self.m_nums] == 0:
                        m_to_d = self.normal_m_to_normal_d
                        d_to_m = self.normal_d_to_normal_m
                    else:
                        m_to_d = self.normal_m_to_central_d
                        d_to_m = self.central_d_to_normal_m
            else:
                if j >= self.m_nums:
                    status = '0'
                    if i == j:
                        status = 'self-loop'
                        relationship_weight_matrix = self.d_self_loop_1
                    elif self.d_co_occurrence[i, j] != 0:
                        nor_coefficient = 1 / t.sqrt(
                            self.d_co_occurrence_and_not_degree[i - self.m_nums, 0] *
                            self.d_co_occurrence_and_not_degree[
                                j - self.m_nums, 0])
                        relationship_weight_matrix = self.d_co_occurrence_weight
                    else:
                        nor_coefficient = 1 / t.sqrt(
                            self.d_co_occurrence_and_not_degree[i - self.m_nums, 1] *
                            self.d_co_occurrence_and_not_degree[
                                j - self.m_nums, 1])
                        relationship_weight_matrix = self.d_not_co_occurrence_weight
                else:
                    status = '1'
                    nor_coefficient = 1 / self.d_m_degree[i]
                    if self.central_d_list[i - self.m_nums] != 0 and self.central_m_list[j] != 0:
                        m_to_d = self.central_m_to_central_d
                        d_to_m = self.central_d_to_central_m
                    elif self.central_d_list[i - self.m_nums] != 0 and self.central_m_list[j] == 0:
                        m_to_d = self.central_m_to_normal_d
                        d_to_m = self.normal_d_to_central_m
                    elif self.central_d_list[i - self.m_nums] == 0 and self.central_m_list[j] == 0:
                        m_to_d = self.normal_m_to_normal_d
                        d_to_m = self.normal_d_to_normal_m
                    else:
                        m_to_d = self.normal_m_to_central_d
                        d_to_m = self.central_d_to_normal_m
            if status == '0':
                m_d_new_feat[i] = nor_coefficient * relationship_weight_matrix(m_d_feat[j]) + m_d_new_feat[i]
            elif status == '1':
                m_d_new_feat[i] = 2 * nor_coefficient * self.the_2th_bigraph[i, j] * (
                        m_to_d(m_d_feat[j]) + d_to_m(m_d_feat[j])) + \
                                  m_d_new_feat[i]
            else:
                m_d_new_feat[i] = relationship_weight_matrix(m_d_feat[j]) + m_d_new_feat[i]
    return m_d_new_feat
    
        def init(self, kernel_matrix_edge_index, args):
        kernel_integrator()(kernel_matrix_edge_index)
        get_bigraph(kernel_matrix_edge_index)
        get_2th_bigraph(kernel_matrix_edge_index)
        get_degree(kernel_matrix_edge_index, args)
        get_central_miRNA_and_Disease(kernel_matrix_edge_index, args)
        get_high_co_occurrence(kernel_matrix_edge_index, args)
"""
