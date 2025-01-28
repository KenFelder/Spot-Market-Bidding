import random
############### Bidder Parameters ################

# First entry is system demand, next four entries are for naive bidders
# last entry is the reinforcement learning agent

### Forecasting ###
re_gen_mean = [0, 400, 400, 400, 400, 400]  # Actual renewable generation (not forecasted)
start_sd_re_gen = [0, 0.4, 0.1, 0.7, 0.05, 0.2]  # Start point for standard deviation of renewable generation forecast

demand_mean = -2500  # Actual demand (not forecasted)
start_sd_demand = 0.2  # Start point for standard deviation of demand forecast

### Thermal capacity ###
true_costs = [0, 5, 12, 16, 18, 11]  # Production costs

x_th_cap = [0, 300, 300, 300, 300, 300]  # Thermal capacity
x_th_start = [random.uniform(0, x_th_cap[i]) for i in range(6)]  # Start point for thermal capacity
ramp_up = [0, 2, 2, 2, 2, 2]  # Ramp up and down MWh / time step
ramp_down = [0, 2, 2, 2, 2, 2]  # Ramp up and down MWh / time step

## Intraday ##
start_aggressiveness_bid = [-0.6, 0.4, 0.7, -0.1, -0.1, -0.1]  # Start point for aggressiveness of bids
start_aggressiveness_ask = [-0.2, -0.1, -0.1, -0.1, -0.1, -0.1]  # Start point for aggressiveness of asks
aggressiveness_step_factor = [2, 1.8, 1.5, 1.9, 1.9, 0.0]  # Step factor for aggressiveness

start_aggressiveness_ask = [random.uniform(-0.8, 0) for i in range(6)]
start_aggressiveness_bid = [random.uniform(-.8, 0) for i in range(6)]
aggressiveness_step_factor = [random.uniform(0.5, 1) for i in range(6)]

start_target_price_param = [-4, -4, -4, -4, -4, -4]  # Start point for target price parameter
target_price_param_step_factor = [0.9, 0.3, 0.4, 0.2, 0.5, 0.0]  # Step factor for target price parameter
start_target_price_param = [random.uniform(-8, 2) for i in range(6)]
target_price_param_step_factor = [random.uniform(0.5, 1) for i in range(6)]  # should be between 0 and 1

bid_step_factor = [1.5 for i in range(6)]  # Rate of convergence
#bid_step_factor = [random.uniform(1, 3) for i in range(6)]

### RL Agent ###
aftermarket_expl = 200  # Steps of aftermarket exploration
max_bid_volume = 2000  # Maximum bid volume
max_ask_volume = 2000  # Maximum ask volume
max_price = 50  # Maximum price
min_price = -20  # Minimum price

############### Game Parameters ################
t_max = 200  # Number of time steps
n = len(re_gen_mean)  # Number of bidders
