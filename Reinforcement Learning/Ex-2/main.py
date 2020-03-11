import numpy as np
from time import sleep
from sailing import SailingGridworld
from statistics import mean

# Set up the environment
env = SailingGridworld(rock_penalty=-2)#-10
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

def action_v(state, V):
    A = []
    for transition in env.transitions[state[0], state[1]]:
        action_value = 0
        for next_state, reward, done, prob in transition:
            action_value += prob * (reward + (0.9 * 
                                    V[next_state] if not done else 0))
        A.append(action_value)
    return A

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()
    
    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))
    theta = value_est.copy()
    eps = 0.0001
    width,height=env.w,env.h
    for itera in range(100):
        env.clear_text()
        delta=0
        for i in range(width):
            for j in range(height):
                A = action_v((i, j), value_est)
                value_est[i,j] = np.max(A)
                policy[i,j] = np.argmax(A)
                #From here epsilon code impleted
                delta=max(delta,np.abs(value_est[i,j]-theta[i,j]))
        if (delta < eps):
            print('Stopped',itera)
            break
        else :
            theta = np.copy(value_est)
        #To check values every iteration
        #env.draw_values(value_est)
        #env.draw_actions(policy)
        #env.render()

    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)

    # Save the state values and the policy
    fnames = "values_dis.npy", "policy_dis.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    done = False
    discount_r=[]
    for i in range(1000):
         state = env.reset()
         temp_discount_r=0
         done=False
         j=0
         while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            #action = int(np.random.random()*4)
            action=policy[state]
            # Step the environment
            state, reward, done, _ = env.step(action)
            temp_discount_r+=(0.9**j) * reward
            #Render and sleep
            #env.render()
            #sleep(0.1)\
            j+=1
         discount_r.append(temp_discount_r)
    print("Mean",mean(discount_r))
    print("std deviation",np.std(np.array(discount_r)))