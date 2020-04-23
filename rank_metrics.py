
def rank_cal(rank_list, target_index):
    rank = 0.
    target_score = rank_list[target_index]
    for score in rank_list:
        if score >= target_score:
            rank += 1.
    return rank

def reciprocal_rank(rank):
    return 1./rank

def accuracy_at_k(rank, k):
    if rank <= k:
        return 1.
    else:
        return 0.

def rank_eval_example(pred, labels):
    mrr = []
    macc1 = []
    macc5 = []
    macc10 = []
    macc50 = []
    cur_pos = []
    for i in range(len(pred)):
        rank = rank_cal(cas_pred[i], cas_labels[i])
        mrr.append(reciprocal_rank(rank))
        macc1.append(accuracy_at_k(rank,1))
        macc5.append(accuracy_at_k(rank,5))
        macc10.append(accuracy_at_k(rank,10))
        macc50.append(accuracy_at_k(rank,50))
    return mrr, macc1, macc5, macc10, macc50

def rank_eval(pred, labels, sl):
    mrr = 0
    macc1 = 0
    macc5 = 0
    macc10 = 0
    macc50 = 0
    macc100 = 0
    cur_pos = 0
    for i in range(len(sl)):
        length = sl[i]
        cas_pred = pred[cur_pos : cur_pos+length]
        cas_labels = labels[cur_pos : cur_pos+length]
        cur_pos += length
        rr = 0
        acc1 = 0
        acc5 = 0
        acc10 = 0
        acc50 = 0
        acc100 = 0
        for j in range(len(cas_pred)):
            rank = rank_cal(cas_pred[j], cas_labels[j])
            rr += reciprocal_rank(rank)
            acc1 += accuracy_at_k(rank,1)
            acc5 += accuracy_at_k(rank,5)
            acc10 += accuracy_at_k(rank,10)
            acc50 += accuracy_at_k(rank,50)
            acc100 += accuracy_at_k(rank,100)
        mrr += rr/float(length)
        macc1 += acc1/float(length)
        macc5 += acc5/float(length)
        macc10 += acc10/float(length)
        macc50 += acc50/float(length)
        macc100 += acc100/float(length)
    return mrr, macc1, macc5, macc10, macc50, macc100