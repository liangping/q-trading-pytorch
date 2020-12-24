import torch
import numpy as np


class Environment:
    def __init__(self, open, close, starting_cash=200, randomize_cash=0, starting_shares=0,
                 randomize_shares=0, max_stride=5, series_length=200,
                 starting_point=0, inaction_penalty=0):
        self.open = open
        self.close = close
        self.starting_cash = starting_cash
        self.randomize_cash = randomize_cash
        self.starting_shares = starting_shares
        self.randomize_shares = randomize_shares

        self.starting_cash = max(int(np.random.normal(self.starting_cash, self.randomize_cash)), 0)
        self.series_length = series_length
        self.starting_point = starting_point
        self.cur_timestep = self.starting_point

        self.state = torch.Tensor(torch.zeros(5)).cuda()
        self.state[0] = max(int(np.random.normal(self.starting_shares, self.randomize_shares)), 0)
        self.state[1] = self.starting_cash
        self.state[2] = self.open[self.cur_timestep]
        self.starting_portfolio_value = self.portfolio_value()
        self.state[3] = self.starting_portfolio_value
        self.state[4] = self.five_day_window()

        self.max_stride = max_stride
        self.stride = self.max_stride

        self.done = False
        self.inaction_penalty = inaction_penalty

    def portfolio_value(self):
        return (self.state[0] * self.close[self.cur_timestep]) + self.state[1]

    def next_opening_price(self):
        next_timestep = self.cur_timestep + self.stride
        return self.open[next_timestep]

    def five_day_window(self):
        step = self.cur_timestep
        if step < 5:
            return self.open[0]
        apl5 = self.open[step-5:step].mean()
        return apl5

    def step(self, action):
        action = [action, 1]
        cur_timestep = self.cur_timestep
        ts_left = self.series_length - (cur_timestep - self.starting_point)
        retval = None
        cur_value = self.portfolio_value()
        gain = cur_value - self.starting_portfolio_value

        if cur_timestep >= self.starting_point + self.series_length * self.stride:
            next_opening = self.next_opening_price()
            next_five_day = self.five_day_window()
            new_state = [self.state[0], self.state[1], next_opening, cur_value, next_five_day]
            self.state = new_state
            return new_state, cur_value + gain, True, {'msg': 'done'}

        if action[0] == 2:  # hold
            next_opening = self.next_opening_price()
            next_five_day = self.five_day_window()
            new_state = [self.state[0], self.state[1], next_opening, cur_value, next_five_day]
            self.state = new_state
            retval = new_state, gain - self.inaction_penalty - ts_left, False, {'msg': 'nothing'}

        if action[0] == 0:  # buy
            if action[1] * self.open[cur_timestep] > self.state[1]:
                print("action:", action[1], action[1] * self.open[cur_timestep], self.state[1])
                next_opening = self.next_opening_price()
                next_five_day = self.five_day_window()
                new_state = [self.state[0], self.state[1], next_opening, cur_value, next_five_day]
                self.state = new_state
                retval = new_state, -ts_left + gain / 2, True, {'msg': 'bankrupt'}
            else:
                apl_shares = self.state[0] + action[1]
                cash_spent = action[1] * self.open[cur_timestep] * 1.1
                next_opening = self.next_opening_price()
                next_five_day = self.five_day_window()
                new_state = [apl_shares, self.state[1] - cash_spent, next_opening, cur_value, next_five_day]
                self.state = new_state
                retval = new_state, gain + self.inaction_penalty - ts_left, False, {'msg': 'bought stocks'}

        if action[0] == 1:  # sell
            if action[1] > self.state[0]:
                next_opening = self.next_opening_price()
                next_five_day = self.five_day_window()
                new_state = [self.state[0], self.state[1], next_opening, cur_value, next_five_day]
                self.state = new_state
                retval = new_state, -ts_left + gain / 2, True, {'msg': 'sold more than available'}
            else:
                apl_shares = self.state[0] - action[1]
                cash_gained = action[1] * self.open[cur_timestep] * 0.9
                next_opening = self.next_opening_price()
                next_five_day = self.five_day_window()
                new_state = [apl_shares, self.state[1] + cash_gained, next_opening, cur_value, next_five_day]
                self.state = new_state
                retval = new_state, gain + self.inaction_penalty - ts_left, False, {'msg': 'sold stocks'}

        self.cur_timestep += self.stride
        return retval

    def reset(self):
        self.starting_cash = max(int(np.random.normal(self.starting_cash, self.randomize_cash)), 0)
        self.cur_timestep = self.starting_point

        self.state[0] = max(int(np.random.normal(self.starting_shares, self.randomize_shares)), 0)
        self.state[1] = self.starting_cash
        self.state[2] = self.open[self.cur_timestep]
        self.state[3] = self.starting_portfolio_value
        self.state[4] = self.five_day_window()

        self.done = False
        return self.state