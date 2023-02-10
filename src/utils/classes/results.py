class Results(object):
    def __init__(self, precision = None, jaccard = None,
            recall = None, precision_recall_auc = None, f1 = None, 
            roc_auc = None, avg_med = None, coverage = None, personalisation = None,
            macro_f1 = None, top_1 = None, top_5 = None, 
            top_10 = None, top_20 = None, loss = None) -> None:
        self.precision = precision
        self.jaccard = jaccard
        self.recall = recall
        self.precision_recall_auc = precision_recall_auc
        self.f1 = f1
        self.macro_f1 = macro_f1
        self.roc_auc = roc_auc
        self.avg_med = avg_med
        self.coverage = coverage
        self.personalisation = personalisation
        self.top_1 = top_1
        self.top_5 = top_5
        self.top_10 = top_10
        self.top_20 = top_20
        self.loss = loss

