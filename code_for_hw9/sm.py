from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        cur_state = self.__class__.start_state
        ret = []
        for x in input_seq:
            cur_state = self.transition_fn(cur_state, x)
            ret.append(self.output_fn(cur_state))
        return ret



class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0, 0)

    def transition_fn(self, s, x):
        carry = s[0]
        res = x[0] + x[1] + carry
        return (res // 2, res % 2)

    def output_fn(self, s):
        return s[1]


class Reverser(SM):
    start_state = [[], False]

    def transition_fn(self, s, x):
        new_s = s
        if not s[1]:
            new_s[0].append(x)
        if x == 'end':
            new_s[1] = True
        if new_s[1]:
            if new_s[0]:
                new_s[0].pop()
        return new_s

    def output_fn(self, s):
        if s[1]:
            if s[0]:
                return s[0][-1]
        return None


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        pass

    def transition_fn(self, s, i):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass
