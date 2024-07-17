import matplotlib.pyplot as plt 
import numpy as np 

class GridWorld:
    def __init__(self, grid_size, start, goal, obstacles):
        self.grid_size = grid_size
        self.start =start 
        self.goal = goal 
        self.obstacles = obstacles 

    def reset(self):
        self.agent_position = self.start 
        return self.agent_position 
    
    def step(self, action):
        new_position = list(self.agent_position)

        if action == 0:
            new_position[0] -= 1
        elif action == 1:
            new_position[1] += 1
        elif action == 2:
            new_position[0] += 1 
        elif action == 3:
            new_position[1] -= 1

        new_position =tuple(new_position)

        if (new_position in self.obstacles or 
            new_position[0] <  0 or new_position[0] >= self.grid_size[0] or
            new_position[1] <  0 or new_position[1] >= self.grid_size[1]):

            new_position = self.agent_position 
        
        self.agent_position = new_position 

        if self.agent_position == self.goal:
            reward =1
            done = True 

        else:
            reward = -0.1
            done = False 

        
        return new_position, reward, done 

    
    def render(self):
        grid  = np.zeros(self.grid_size)
        grid[self.agent_position] = 2
        grid[self.goal] = 3
        for obstacle in self.obstacles:
            grid[obstacle] = 1
        print(grid)

    

state_size = (5,5)
action_size = 4
q_table = np.zeros(state_size + (action_size,))

alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 1000


env = GridWorld((5,5), (0,0),(4,4), [(1,1), (2,2),(3,3)])

def choose_action(state):
    if np.random.uniform(0,1) < epsilon:
        action = np.random.randint(0, action_size)

    else:
        action = np.argmax(q_table[state])
    
    return action 


for episode in range(episodes):
    state = env.reset()
    done = False 

    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)
        q_table[state][action] = q_table[state][action] + alpha * (reward+ gamma * np.max(q_table[next_state]) - q_table[state][action])
        state = next_state 

    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes} completed")

    

print("Training Completed")
state = env.reset()
done = False 
total_reward = 0

while not done:
    env.render()
    action = np.argmax(q_table[state])
    next_state, reward, done = env.step(action)
    total_reward += reward 
    state = next_state 


print(f"Total Reward: {total_reward}")

def plot_path():
    state = env.reset()
    done = False 
    path = [state]


    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state 
    

    x,y = zip(*path)
    plt.plot(y,x, marker = 'o')
    plt.gca().invert_yaxis()
    plt.show()
plot_path()