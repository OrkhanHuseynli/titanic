from numpy import ndarray


class LogisticRegressionUtil:
    @staticmethod
    def get_confusion_matrix(y_pred: ndarray, y_actual: ndarray) -> list:
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_actual[i]:
                if y_pred[i] == 1:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                if y_pred[i] == 1:
                    fp = fp + 1
                else:
                    fn = fn + 1

        return [[tp, fp], [tn, fn]]

    @staticmethod
    def calculate_accuracy(true_positives: int, false_positives: int,
                           true_negatives: int, false_negatives: int) -> float:
        trues = (true_positives + true_negatives)
        return trues / (trues + false_positives + false_negatives)

    @staticmethod
    def calculate_f_measure(true_positives: int, false_positives: int, false_negatives: int) -> float:
        recall = LogisticRegressionUtil.calculate_recall(true_positives, false_negatives)
        precision = LogisticRegressionUtil.calculate_precision(true_positives, false_positives)
        return (2 * recall * precision) / (recall + precision)

    @staticmethod
    def calculate_recall(true_positives: int, false_negatives: int) -> float:
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def calculate_precision(true_positives: int, false_positives: int) -> float:
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def calculate_true_positive_rate(true_positives: int, false_negatives: int) -> float:
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def calculate_false_positive_rate(false_positives: int, true_negatives: int) -> float:
        return false_positives / (false_positives + true_negatives)

    @staticmethod
    def calculate_ROC(y_pred: ndarray, y_actual: ndarray) -> (list, list):
        confusion_matrix = LogisticRegressionUtil.get_confusion_matrix(y_pred, y_actual)
        tp = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        tn = confusion_matrix[1][0]
        fn = confusion_matrix[1][1]
        tpr = LogisticRegressionUtil.calculate_true_positive_rate(tp, fn)
        fpr = LogisticRegressionUtil.calculate_false_positive_rate(fp, tn)
        return [tpr, fpr], confusion_matrix
