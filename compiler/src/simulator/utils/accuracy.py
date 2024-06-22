from enlight.utils import _unpair


class Accuracy:
    def __init__(self, topk=5):
        self.true = 0
        self.num_eval = 0

        self.topk = topk

    def reset(self):
        self.true = 0
        self.num_eval = 0

    def match(self, results, targets):
        shape = results.shape
        
        # Replacing linear layer with conv1x1 cause extension of output dimension
        if len(shape) == 4:
            dim = _unpair(shape[2:])
            assert dim == 1, "Classification output has wrong dimensions {}".format(shape)

            shape = shape[:2]
            results = results.reshape(shape)

        num_class = results.shape[-1]
        assert num_class >= self.topk, 'set args.topk with number of class'

        num_batch = targets.size(0)

        # get top5 indices
        _, results_ = results.topk(self.topk, 1)
        targets_ = targets.unsqueeze(1).expand_as(results_)

        self.true = self.true + (results_ == targets_).sum().item()
        self.num_eval += num_batch

        print('....................................', self.true * 100. / self.num_eval, end='\r')

    def get_result(self):
        return (self.true * 100. / self.num_eval)
