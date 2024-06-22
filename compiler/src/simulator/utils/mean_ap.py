
class MeanAp:
    def __init__(self, num_class):
        self.num_class = num_class

        self.detected = None
        self.num_truth = None

    def init(self):
        self.detected = [[] for cls in range(self.num_class)]
        self.num_truth = [0 for cls in range(self.num_class)]

    def reset(self):
        self.detected = None
        self.num_truth = None

    def match(self, results, targets):
        truths, num_truth = self.parse_truths(targets)

        if len(results) < self.num_class:
            num_class = self.num_class - 1
        else:
            num_class = self.num_class

        for cls in range(num_class):
            already_matched = []

            tmp = sorted(results[cls], key=lambda x: x[4], reverse=True)
            for result in tmp:
                matched = self.find_match(truths[cls], result)

                if matched in already_matched:
                    matched = -1

                tf = False if matched < 0 else True
                self.detected[cls].append({'tf': tf, 'conf': result[4] })

                if tf:
                    already_matched.append(matched)

            self.num_truth[cls] += num_truth[cls]

    def calc_mean_ap(self):
        ap = []
        num_class = self.num_class

        for cls in range(num_class):
            detected = self.detected[cls]
            num_truth = self.num_truth[cls]

            ap.append(self.calc_ap(detected, num_truth))

        return sum(ap) / num_class,  ap

    def calc_ap(self, detected, num_truth):
        k = lambda x: x['conf']
        detected = sorted(detected, key=k, reverse=True)

        tp = fp = 0.

        recalls = [0.]
        precisions = [1.]

        for _ in detected:
            if _['tf']:
                tp += 1.
            else:
                fp += 1.

            recalls.append(tp / (num_truth + 1e-5))
            precisions.append(tp / (tp + fp))

        recalls.append(0.)
        precisions.append(0.)

        for i in range(len(precisions)-1, 0, -1):
            precisions[i-1] = max(precisions[i-1], precisions[i])

        candidates = []
        for i in range(len(recalls)-1, 0, -1):
            if recalls[i] != recalls[i-1]:
                candidates.append(i)

        ap = 0.
        for i in candidates:
            ap += (recalls[i] - recalls[i-1]) * precisions[i]

        return ap

    def parse_truths(self, targets):
        num_class = self.num_class

        truths = [[] for cls in range(num_class)]
        num_truth = [0 for cls in range(num_class)]

        for target in targets:
            cls = int(target[4])

            truths[cls].append(target[0:4])
            num_truth[cls] += 1

        return truths, num_truth

    @staticmethod
    def find_match(truths, result):
        max_iou = -1.
        max_i = -1

        for i, truth in enumerate(truths):
            iou = MeanAp.calc_iou(truth, result)

            if max_iou < iou:
                max_iou = iou
                max_i = i

        if max_iou < 0.5:
            return -1

        return max_i

    @staticmethod
    def calc_iou(truth, result):
        x0 = max(truth[0], result[0])
        y0 = max(truth[1], result[1])
        x1 = min(truth[2], result[2])
        y1 = min(truth[3], result[3])

        w = max(x1 - x0, 0.)
        h = max(y1 - y0, 0.)

        area_truth = (truth[2] - truth[0]) * (truth[3] - truth[1])
        area_result = (result[2] - result[0]) * (result[3] - result[1])

        intersect = w * h
        union = area_truth + area_result - intersect

        return intersect / union

