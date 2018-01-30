"""
Self-learning Tic Tac Toe
Made by Lorenzo Mambretti and Hariharan Sezhiyan

Last Update: 1/17/2018 4:53 PM (Lorenzo)
    * initial old weights/biases saved correctly
"""

import random
import numpy as np
import tensorflow as tf

class State:
    board = np.zeros((3,3))
    terminal = False

def is_valid(action, state):
    if state.board[int(np.floor(action / 3))][action % 3] != 0:
        return False
    else:
        return True

def step(state, action):

    # insert
    state_ = State()
    state_.board = np.copy(state.board)
    row_index = int(np.floor(action / 3))
    col_index = action % 3
    state_.board[row_index][col_index] = 1

    # undecided
    terminal = 1

    # to check for 3 in a row horizontal
    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] != 1):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # to check for 3 in a row vertical
    for col in range(3):
        for row in range(3):
            if(state_.board[row][col] != 1):
                terminal = 0
        if(terminal == 1):
            state_.terminal = True
            return +1, state_
        else:
            terminal = 1

    # diagonal top-left to bottom-right
    for diag in range(3):
        if(state_.board[diag][diag] != 1):
            terminal = 0
    if(terminal == 1):
        state_.terminal = True
        return +1, state_
    else:
        terminal = 1

    # diagonal bottom-left to top-right
    for diag in range(3):
        if(state_.board[2 - diag][diag] != 1):
            terminal = 0
    if(terminal == 1):
        state_.terminal = True
        return +1, state_
    else:
        terminal = 1

    # checks if board is filled completely
    for row in range(3):
        for col in range(3):
            if(state_.board[row][col] == 0):
                terminal = 0
                break
    if terminal == 1:
        state_.terminal = True

    return 0, state_

def save(W1, W2, B1, B2):
    np.savez("weights.npz", W1, W2, B1, B2)
    print("file weights.txt has beeen updated successfully")

def load():
    npzfile = np.load("weights.npz")
    W1 = np.reshape(npzfile['arr_0'], (27, 18))
    W2 = np.reshape(npzfile['arr_1'], (18,9))
    b1 = np.reshape(npzfile['arr_2'], (18))
    b2 = np.reshape(npzfile['arr_3'], (9))
    return w1, w2, b1, b2

def extract_policy(state):
    policy = None
    q_values = compute_Q_values(state)
    for action in range(9):
        if is_valid(action,state):
            if policy == None:
                policy = action
                best_q = q_values[action]
            else:
                new_q = q_values[action]
                if new_q > best_q:
                    policy = action
                    best_q = new_q
    return policy
    
def invert_board(state):
    state_ = State()
    state_.board = np.copy(state.board)
    state_.terminal = state.terminal
    for row in range(3):
        for col in range(3):
            if(state.board[row][col] == 1):
                state_.board[row][col] = 2
            elif(state.board[row][col] == 2):
                state_.board[row][col] = 1

    return state_

def play_game():
    while(True):
        start_nb = input("If you would like to move first, enter 1. Otherwise, enter 2. ")
        start = int(start_nb)
        state = State()
        state.board = np.zeros((3,3))

        while not state.terminal:
            if start == 1:
                action = int(input("Please enter your move: "))
                while(is_valid(action, state) == False):
                    action = int(input("Please enter a correct move: "))
                start = 0
                r, state = step(state, action)
            else:
                state = invert_board(state)
                action = extract_policy(state)
                start = 1
                r, state = step(state, action)
                r = -r
                state = invert_board(state)

            print(state.board)
                
        if r == 0:
            print ("Tie")
        elif r == 1:
            print ("You won")
        else:
            print ("You lost")

def convert_state_representation(state):
    new_board = np.zeros((1,27))
    for row in range(3):
        for col in range(3):
            if(state.board[row][col] == 0):
                new_board[0][9 * row + 3 * col] = 1
            elif(state.board[row][col] == 1):
                new_board[0][9 * row + 3 * col + 1] = 1
            else:
                new_board[0][9 * row + 3 * col + 2] = 1

    return(new_board)

def compute_Q_values(state):
    # computes associated Q value based on NN function approximator
    q_board = np.zeros((1,27))
    q_board = np.copy(convert_state_representation(state))

    #NN forward propogation
    q_values = sess.run(y, feed_dict = {x: q_board})
    q_values = np.reshape(q_values, 9)
    return (q_values)

def train(experience_replay, saved_W1, saved_W2, saved_b1, saved_b2):
    # can modify batch size here
    batch_size = 1
    mini_batch = experience_replay[np.random.choice(experience_replay.shape[0], batch_size), :]
    batch = np.zeros((0,27))
    batch_ = np.zeros((0,27))
    a = mini_batch[:,1]
    r = mini_batch[:,2] # is the list of all rewards within the mini_batch

    new_a = np.zeros((0,2))
    new_a_insert = np.zeros((0,2))

    target_a = np.ones((batch_size,1))

    for i in range(batch_size):
        first = mini_batch[i][0]
        second = mini_batch[i][3]
        batch = np.concatenate((batch, convert_state_representation(first)), axis=0)
        batch_ = np.concatenate((batch_, convert_state_representation(second)), axis=0)
        new_a_insert = np.array([[i, a[i]]])
        new_a = np.concatenate((new_a, new_a_insert), axis=0)
        if(second.terminal == True):
            target_a[i] = 0
        
    summary, _, yy= sess.run([merged, train_step, W2], feed_dict={ x: batch,
                                                            x_old : batch_,
                                                            W1_old : saved_W1,
                                                            W2_old : saved_W2,
                                                            b1_old : saved_b1,
                                                            b2_old : saved_b2,
                                                            reward : r, 
                                                            action_tensor : new_a, 
                                                            target_modifier: target_a })
    train_writer.add_summary(summary, i)
    print("W2")
    print(yy)

# Q learner neural network
with tf.name_scope('Q-learner') as scope:
    x = tf.placeholder(tf.float32, [None, 27], name='x')
    with tf.name_scope('hidden_layer') as scope:
        W1 = tf.get_variable("W1", shape=[27, 18],
           initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", shape=[18],
           initializer=tf.contrib.layers.xavier_initializer())
        # W1 = tf.Variable(tf.random_normal([27, 18]), name='W1')
        # b1 = tf.Variable(tf.random_normal([18]), name='b1')
        h1 = tf.tanh(tf.matmul(x, W1) + b1)
    with tf.name_scope('output_layer') as scope:
        W2 = tf.get_variable("W2", shape=[18, 9],
           initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", shape=[9],
           initializer=tf.contrib.layers.xavier_initializer())
        # W2 = tf.Variable(tf.random_normal([18, 9]), name='W2')
        # b2 = tf.Variable(tf.random_normal([9]), name='b2')
        y = tf.tanh(tf.matmul(h1, W2) + b2)
        action_tensor = tf.placeholder(tf.int32, [None, 2])

    new_y = tf.gather_nd(y, action_tensor)
    q_learner = tf.reshape(new_y, [1, 1])

# Q target neural network
with tf.name_scope('Q-target') as scope:
    x_old = tf.placeholder(tf.float32, [None, 27], name='x_old')
    with tf.name_scope('hidden_layer') as scope:
        W1_old = tf.placeholder(tf.float32, [27, 18], name='W1_old')
        b1_old = tf.placeholder(tf.float32, [18], name='b1_old')
        h1_old = tf.tanh(tf.matmul(x_old, W1_old) + b1_old, name='h1')
    with tf.name_scope('output_layer') as scope:
        W2_old =tf.placeholder(tf.float32, [18, 9], name='W2_old')
        b2_old =tf.placeholder(tf.float32, [9], name='b2_old')
        y_old = tf.tanh(tf.matmul(h1_old, W2_old) + b2_old, name='y_old')
        target_modifier = tf.placeholder(tf.float32, [None, 1])
    
    intermediate_target = tf.reduce_max(y_old, axis = 1, keep_dims = True,)
    q_target = tf.multiply(intermediate_target, target_modifier)

with tf.name_scope('loss') as scope:
    reward = tf.placeholder(tf.float32, [None])
    gamma = tf.constant(0.99, name='gamma')
    squared_deltas = tf.square(reward + (gamma * q_target) - q_learner)
    loss = tf.reduce_mean(squared_deltas)
    
#train_step = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
train_step = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, use_locking=False, centered=False, name='RMSProp').minimize(loss)

tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter('tensorflow_logs', sess.graph)
tf.global_variables_initializer().run()

episodes = 100000
n0 = 100.0
start_size = 500
experience_replay = np.zeros((0,4))

print("All set. Start epoch")

for e in range(episodes):
    # print("episode ",e)
    state = State()
    if e >= start_size:
        epsilon = max(n0 / (n0 + (e- start_size)), 0.1)
    else: epsilon = 1
    
    if e % 2 == 1:
        # this is player 2's turn
            state = invert_board(state)
            if random.random() < epsilon:
                # take random action
                action_pool = np.random.choice(9,9, replace = False)
                for a in action_pool:
                    if is_valid(a, state):
                        action = a
                        break
            else:
                # take greedy action
                action = extract_policy(state)

            r, state = step(state, action)
            state = invert_board(state)
            r = -r 
    
    while not state.terminal:
        # this section is player 1's turn
        # select epsilon-greedy action
        if random.random() < epsilon:
            # take random action
            action_pool = np.random.choice(9,9, replace = False)
            for a in action_pool:
                if is_valid(a, state):
                    action = a
                    break
        else:
            # take greedy action
            action = extract_policy(state)

        r, state_ = step(state, action)

        if not state_.terminal:
            # this is player 2's turn
            state_ = invert_board(state_)
            if random.random() < epsilon:
                # take random action
                action_pool = np.random.choice(9,9, replace = False)
                for a in action_pool:
                    if is_valid(a, state_):
                        action2 = a
                        break
            else:
                # take greedy action
                action2 = extract_policy(state_)

            r, state_ = step(state_, action2)
            state_ = invert_board(state_)
            r = -r 

        D0 = State()
        D0.board = np.copy(state.board)
        D0.terminal = state.terminal
        D1 = State()
        D1.board = np.copy(state_.board)
        D1.terminal = state_.terminal
        D = (D0, action, r, D1)
        experience_replay = np.append(experience_replay, [D], axis = 0)
        state.board = np.copy(state_.board)
        state.terminal = state_.terminal

    if e == start_size: print("Start Training")
    if e >= start_size:
        if((e % 50) == 0):
            print("Episode:",e)
            # here save the W1,W2,b1,B2
            saved_W1 = W1.eval()
            saved_W2 = W2.eval()
            saved_b1 = b1.eval()
            saved_b2 = b2.eval()
        train(experience_replay, saved_W1, saved_W2, saved_b1, saved_b2)
print("Training completed")
play_game()
