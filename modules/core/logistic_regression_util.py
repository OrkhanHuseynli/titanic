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
    def calculate_accuracy(true_positives: int, false_positives: int,
                           true_negatives: int, false_negatives: int) -> float:
        trues = (true_positives + true_negatives)
        return trues/(trues + false_positives + false_negatives)

    @staticmethod
    def calculate_f_measure(true_positives: int, false_positives: int, false_negatives: int) -> float:
        recall = LogisticRegressionUtil.calculate_recall(true_positives, false_negatives)
        precision = LogisticRegressionUtil.calculate_precision(true_positives, false_positives)
        return (2*recall*precision)/(recall+precision)

    @staticmethod
    def calculate_recall(true_positives: int, false_negatives: int) -> float:
        return true_positives/(true_positives + false_negatives)

    @staticmethod
    def calculate_precision(true_positives: int, false_positives: int) -> float:
        return true_positives/(true_positives + false_positives)

    @staticmethod
    def calculate_true_positive_rate(true_positives: int, false_negatives: int) -> float:
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def calculate_false_positive_rate(false_positives: int, true_negatives: int) -> float:
        return false_positives / (false_positives + true_negatives)
