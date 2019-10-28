from numpy import ndarray


class LogisticRegressionUtil:
    @staticmethod
    def get_confusion_matrix(y_pred: ndarray, y_actual: ndarray) -> list:
        positives = 0
        negatives = 0
        for item in y_actual:
            if item == 1:
                positives = positives + 1
            elif item == 0:
                negatives = negatives + 1

        result = y_pred - y_actual
        false_positives = 0
        false_negatives = 0
        for item in result:
            if item == 1:
                false_positives = false_positives + 1
            elif item == -1:
                false_negatives = false_negatives + 1
        return [[positives - false_positives, false_positives], [negatives - false_negatives, false_positives]]

    @staticmethod
    def calculate_true_positive_rate(true_positives, false_negatives):
        return true_positives/(true_positives+false_negatives)

    @staticmethod
    def calculate_false_positive_rate(true_positives, false_negatives):
        return 1 - LogisticRegressionUtil.calculate_true_positive_rate(true_positives, false_negatives)