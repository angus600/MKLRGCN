from model import *
from getData import *
from sklearn.model_selection import KFold
from sklearn import metrics


def train_valid_test(args):
    kernel_matrix_edge_index = get_mirna_disease_Kernel(args)
    pos_samples_index_shuffled, neg_samples_index_shuffled = get_positive_negative_samples(kernel_matrix_edge_index)
    neg_samples_index_shuffled = neg_samples_index_shuffled[:, :pos_samples_index_shuffled.shape[1]]
    pos_samples_index_shuffled = pos_samples_index_shuffled.T
    neg_samples_index_shuffled = neg_samples_index_shuffled.T

    k_fold = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)
    train_idx = []
    test_idx = []
    auc_list = []
    for train_index, test_index in k_fold.split(pos_samples_index_shuffled):
        train_idx.append(train_index)
        test_idx.append(test_index)

    for i in range(args.kfolds):
        train_pos_samples_80_percent = pos_samples_index_shuffled[train_idx[i]]
        train_neg_samples_80_percent = neg_samples_index_shuffled[train_idx[i]]
        nest_k_fold = KFold(n_splits=args.kfolds, shuffle=True, random_state=1)
        nest_train_idx = []
        valid_idx = []
        for nest_train_index, valid_index in nest_k_fold.split(train_pos_samples_80_percent):
            nest_train_idx.append(nest_train_index)
            valid_idx.append(valid_index)

        train_pos_samples_80_percent = train_pos_samples_80_percent.T
        train_neg_samples_80_percent = train_neg_samples_80_percent.T

        test_pos_samples_20_percent = pos_samples_index_shuffled[test_idx[i]]
        test_neg_samples_20_percent = neg_samples_index_shuffled[test_idx[i]]
        test_pos_samples_20_percent = test_pos_samples_20_percent.T
        test_neg_samples_20_percent = test_neg_samples_20_percent.T

        train_samples = t.hstack((train_pos_samples_80_percent, train_neg_samples_80_percent)).to(args.device)
        test_samples = t.hstack((test_pos_samples_20_percent, test_neg_samples_20_percent)).to(args.device)
        train_labels = t.hstack(
            (t.ones(train_pos_samples_80_percent.shape[1]), t.zeros(train_neg_samples_80_percent.shape[1]))).to(
            args.device)
        test_labels = t.hstack(
            (t.ones(test_pos_samples_20_percent.shape[1]), t.zeros(test_neg_samples_20_percent.shape[1]))).to(
            args.device)

        best_para_group = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        max_arr_auc = -np.inf
        GCNlayer = [1, 2, 3]
        RGCNlayer = [1, 2, 3]
        proj = [1000, 1024, 1500, 2000, 2048, 2500, 3000]
        mthreshold = [[0.75, 0.5, 0.25]]
        dthreshold = [[0.75, 0.5, 0.25], [0.7, 0.4, 0.1], [0.8, 0.5, 0.2]]
        mtop_i_percent = [0.1, 0.15, 0.2, 0.25]
        dtop_i_percent = [0.1, 0.15, 0.2, 0.25]
        m_common_i = [10, 11, 12, 13, 14, 15]
        d_common_i = [10, 11, 12, 13, 14, 15]
        print(
            f'###################### out Fold{i + 1} of {args.kfolds}##########################')
        for gcnlayer in GCNlayer:
            args.GCNlayer = gcnlayer
            for rgcnlayer in RGCNlayer:
                args.RGCNlayer = rgcnlayer
                for p in proj:
                    args.proj = p
                    for mth in mthreshold:
                        args.mthreshold = mth
                        for dth in dthreshold:
                            args.dthreshold = dth
                            for mto in mtop_i_percent:
                                args.mtop_i_percent = mto
                                for dto in dtop_i_percent:
                                    args.dtop_i_percent = dto
                                    for m_c in m_common_i:
                                        args.m_common_i = m_c
                                        for d_c in d_common_i:
                                            args.d_common_i = d_c
                                            para_group = [gcnlayer, rgcnlayer, p, mth, dth, mto, dto, m_c, d_c]
                                            nest_auc_list = []

                                            for j in range(args.kfolds):
                                                nest_train_pos_samples_80_percent = train_pos_samples_80_percent.T[
                                                    nest_train_idx[j]]
                                                nest_train_neg_samples_80_percent = train_neg_samples_80_percent.T[
                                                    nest_train_idx[j]]
                                                nest_train_pos_samples_80_percent = nest_train_pos_samples_80_percent.T
                                                nest_train_neg_samples_80_percent = nest_train_neg_samples_80_percent.T

                                                nest_valid_pos_samples_20_percent = train_pos_samples_80_percent.T[
                                                    valid_idx[j]]
                                                nest_valid_neg_samples_20_percent = train_neg_samples_80_percent.T[
                                                    valid_idx[j]]
                                                nest_valid_pos_samples_20_percent = nest_valid_pos_samples_20_percent.T
                                                nest_valid_neg_samples_20_percent = nest_valid_neg_samples_20_percent.T
                                                nest_train_samples = t.hstack((nest_train_pos_samples_80_percent,
                                                                               nest_train_neg_samples_80_percent)).to(
                                                    args.device)
                                                nest_valid_samples = t.hstack((nest_valid_pos_samples_20_percent,
                                                                               nest_valid_neg_samples_20_percent)).to(
                                                    args.device)
                                                nest_train_labels = t.hstack((t.ones(
                                                    nest_train_pos_samples_80_percent.shape[1]), t.zeros(
                                                    nest_train_neg_samples_80_percent.shape[1]))).to(args.device)
                                                nest_valid_labels = t.hstack((t.ones(
                                                    nest_valid_pos_samples_20_percent.shape[1]), t.zeros(
                                                    nest_valid_neg_samples_20_percent.shape[1]))).to(args.device)
                                                model = Model(args, kernel_matrix_edge_index).to(args.device)
                                                optimizer = t.optim.AdamW(params=model.parameters(), weight_decay=1e-4,
                                                                          lr=args.lr)
                                                loss_fn = t.nn.BCEWithLogitsLoss()
                                                print(
                                                    f'******************* nest Fold{j + 1} of {args.kfolds}**************************')
                                                for l in range(args.epoch):
                                                    model.train()
                                                    optimizer.zero_grad()
                                                    result = model(args, kernel_matrix_edge_index, nest_train_samples)
                                                    loss = loss_fn(t.flatten(result), nest_train_labels)
                                                    loss.backward()
                                                    optimizer.step()
                                                    print('epoch {:03d} train_loss {:.8f}  '.format(
                                                        l + 1, loss.item()))
                                                model.eval()
                                                with t.no_grad():
                                                    y_pre = model(args, kernel_matrix_edge_index,
                                                                  nest_valid_samples)
                                                    y_pre = t.flatten(y_pre)
                                                    y_pre = t.sigmoid(y_pre)
                                                    y_pre = y_pre.cpu().numpy()
                                                    y_real = nest_valid_labels.cpu().numpy()
                                                    nest_auc_list.append(metrics.roc_auc_score(y_real, y_pre))
                                                    y_score = np.where(y_pre >= 0.5, 1, 0)
                                                    precision = metrics.precision_score(y_real, y_score)
                                                    print('nest fold{:01d} auc{:.4f} pre{:.4f}'.format(j + 1,
                                                                                                       nest_auc_list[j],
                                                                                                       precision))
                                            arr = np.array(nest_auc_list)
                                            averages = np.round(np.mean(arr, axis=0), 4)
                                            if averages > max_arr_auc:
                                                max_arr_auc = averages
                                                for z in range(len(best_para_group)):
                                                    best_para_group[z] = para_group[z]
        print('*********************************************')
        print(f"""
        最佳参数组合：
        GCNlayer: {best_para_group[0]}
        RGCNlayer: {best_para_group[1]}
        proj: {best_para_group[2]}
        mthreshold: {best_para_group[3]}
        dthreshold: {best_para_group[4]}
        mtop_i_percent: {best_para_group[5]}
        dtop_i_percent: {best_para_group[6]}
        m_common_i: {best_para_group[7]}
        d_common_i: {best_para_group[8]}
        """)
        args.GCNlayer = best_para_group[0]
        args.RGCNlayer = best_para_group[1]
        args.proj = best_para_group[2]
        args.mthreshold = best_para_group[3]
        args.dthreshold = best_para_group[4]
        args.mtop_i_percent = best_para_group[5]
        args.dtop_i_percent = best_para_group[6]
        args.m_common_i = best_para_group[7]
        args.d_common_i = best_para_group[8]
        model = Model(args, kernel_matrix_edge_index).to(args.device)
        optimizer = t.optim.AdamW(params=model.parameters(), weight_decay=1e-4, lr=args.lr)
        loss_fn = t.nn.BCEWithLogitsLoss()
        for _ in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            result = model(args, kernel_matrix_edge_index, train_samples)
            loss = loss_fn(t.flatten(result), train_labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with t.no_grad():
            y_pre = model(args, kernel_matrix_edge_index,
                          test_samples)
            y_pre = t.flatten(y_pre)
            y_pre = t.sigmoid(y_pre)
            y_pre = y_pre.cpu().numpy()
            y_real = test_labels.cpu().numpy()
            auc_list.append(metrics.roc_auc_score(y_real, y_pre))
            y_score = np.where(y_pre >= 0.5, 1, 0)
            precision = metrics.precision_score(y_real, y_score)
            print('out fold{:01d} auc{:.4f} pre{:.4f}'.format(i + 1,
                                                              auc_list[i],
                                                              precision))
    print(f'all auc:{auc_list}')
    arr = np.array(auc_list)
    averages = np.round(np.mean(arr, axis=0), 4)
    print('final auc{:.4f}'.format(averages))
