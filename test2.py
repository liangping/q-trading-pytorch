import numpy
import torch

if __name__ == '__main__':

    x = numpy.ones((2, 2))
    x = x * 2
    print(x)

    x1 = torch.ones((2, 2))
    x1 = x1 * 3
    print(torch.topk(x1[0][0], 0).values)

    print(torch.from_numpy(x)+x1)
    print(torch.from_numpy(x)*x1)

    def sigmoid(x):
        # TODO: Implement sigmoid function
        return 1/(1 + numpy.exp(-x))

    # TODO: Calculate the output
    output = sigmoid(x1)

    print('Output:')
    print(output)
