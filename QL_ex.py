import numpy as np
import matplotlib.pyplot as plt

size = 50
n_states = size**2
n_actions = 4
num_targs = 7
goal_state = [size-1]
for x in range(num_targs):
    random_target = np.random.randint(0,n_states)
    while random_target in goal_state:
        random_target = np.random.randint(0,n_states)
    goal_state.append(random_target)
# goal_state = [size**2-1, size-1, random_target] #alternative goal nodes
# goal_state = [int(size/2)]
start_state = size**2-int(size/2)
max_steps = (size**2)/2
actions = [1, size, -1, -size]



Q_table = np.zeros((n_states, n_actions))

learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 1
epochs = 100

for epoch in range(epochs):
    # current_state = np.random.randint(0, n_states)
    current_state = start_state
    steps = 0
    targets_alive = np.ones(len(goal_state))
    print(epoch)
    traveled = np.zeros(size**2)
    traveled[start_state] = 0

    # step iterations
    while any(targets_alive):
        possible = actions.copy()
        possible_actions_arr = [0,1,2,3]
        flag = False
        if np.random.rand() < exploration_prob:
            
            if current_state%size==0:
                possible.remove(-1)
            elif current_state%size==size-1:
                possible.remove(1)
            
            if int(current_state/size)==0:
                possible.remove(-size)
            elif int(current_state/size)==size-1:
                possible.remove(size)

            action_mod = np.random.choice(possible)
            next_state = current_state + action_mod
            action_index = actions.index(action_mod)

        else:

            if current_state%size==0:
                possible_actions_arr.remove(2)
            elif current_state%size==size-1:
                possible_actions_arr.remove(0)
            
            if int(current_state/size)==0:
                possible_actions_arr.remove(3)
            elif int(current_state/size)==size-1:
                possible_actions_arr.remove(1)
            action_index = np.argmax(Q_table[current_state, possible_actions_arr])
            action_index = possible_actions_arr[action_index]
            next_state = current_state + actions[action_index]
        

        if any(ele == next_state for ele in goal_state):
            goal_index = goal_state.index(next_state)
            reward = 0.5
            targets_alive[goal_index]=0
        elif next_state==start_state:
            reward = 0.1
        else:
            reward = 0
            traveled[next_state]= 1


        Q_table[current_state, action_index] += learning_rate * \
            (reward + discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action_index])

        current_state = next_state
        steps = steps + 1

q_values_grid = np.max(Q_table, axis=1).reshape((size, size))
q_values_grid_proto = np.max(Q_table, axis=1)
q_values_grid = np.zeros(len(q_values_grid_proto))
for i in range(len(q_values_grid_proto)):
    neighbors = actions.copy()
    if i%size==0:
        neighbors.remove(-1)
    elif i%size==size-1:
        neighbors.remove(1)
    
    if int(i/size)==0:
        neighbors.remove(-size)
    elif int(i/size)==size-1:
        neighbors.remove(size)

    summing = 0
    for ele in neighbors:
        summing = summing + q_values_grid_proto[i+ele]
        if any(ment == i+ele for ment in goal_state) or i+ele == start_state:
            summing = summing+1
    q_values_grid[i] = summing/len(neighbors)

q_values_grid = q_values_grid.reshape((size,size))

# Plot the grid of Q-values
plt.figure(figsize=(10, 10))
plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')

plt.colorbar(label='Q-value')
plt.title('Learned Q-values for each state')
plt.xticks(np.arange(size))  #, ['0', '1', '2', '3'])
plt.yticks(np.arange(size))  #, ['0', '1', '2', '3'])
plt.gca().invert_yaxis()  # To match grid layout
plt.grid(True)

# Annotating the Q-values on the grid
# for i in range(size):
#     for j in range(size):
#         plt.text(j, i, f'{q_values_grid[i, j]:.2f}', ha='center', va='center', color='black')
x_vec = []
y_vec = []
for elemnt in goal_state:
    y_vec.append(int(elemnt/size))
    x_vec.append(elemnt%size)

for i in range(len(x_vec)):
    plt.text(x_vec[i], y_vec[i], 'target', ha='center', va='center', color='black')
plt.text(start_state%size, start_state/size, 'start', ha='center', va='center', color='black')



plt.show()

# Print learned Q-table
print("Learned Q-table:")
print(Q_table)