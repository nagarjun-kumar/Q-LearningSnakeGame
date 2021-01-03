import numpy as np
import utils
import copy
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)


    def reset(self):
        self.points = 0
        self.s = None
        self.a = None


    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        adjoining_wall_x = 0
        adjoining_wall_y = 0
        food_dir_x = 0
        food_dir_y = 0
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        curr_S = (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        m_S = list(curr_S)
        if snake_head_x == 40:
            m_S[0] = 1
        elif snake_head_x == 520:
            m_S[0] = 2
        else:
            m_S[0] = 0
            
        if snake_head_y == 40:
            m_S[1] = 1
        elif snake_head_y == 520:
            m_S[1] = 2
        else:
            m_S[1] = 0
            
        if (food_x == snake_head_x):
            m_S[2] = 0
        elif (food_x < snake_head_x):
            m_S[2] = 1
        else:
            m_S[2] = 2
            
        if (food_y == snake_head_y):
            m_S[3] = 0
        elif (food_y < snake_head_y):
            m_S[3] = 1
        else:
            m_S[3] = 2
            
        for part in snake_body:
            if part[1] - snake_head_y == -40 and part[0] == snake_head_x:
                m_S[4] = 1
            elif part[1] - snake_head_y == 40 and part[0] == snake_head_x:
                m_S[5] = 1
            elif part[0]-snake_head_x == -40 and part[1] == snake_head_y:
                m_S[6] = 1
            elif part[0] - snake_head_x == 40 and part[1] == snake_head_y:
                m_S[7] = 1

        

        if self._train is True:
            if self.s is not None and self.a is not None:
                if dead == True:
                    reward = -1
                elif points>self.points:
                    reward = 1
                else:
                    reward = -0.1
                
                var = -999999999
                for i in self.actions:
                    temp_list = m_S.copy()
                    temp_list.append(i)
                    temp_tuple = tuple(temp_list)

                    if self.Q[temp_tuple]>var:
                        var = self.Q[temp_tuple]
                
                prev_list = list(self.s)
                prev_list.append(self.a)
                prev_tuple = tuple(prev_list)


                lrate = self.C/(self.C+self.N[prev_tuple])
                self.Q[prev_tuple] += lrate*(reward+self.gamma*var-self.Q[prev_tuple])
        

        
        a = None
        var_2 = -99999999
        for move in self.actions:
            temp_2_list = m_S.copy()
            temp_2_list.append(move)
            temp_2_tuple = tuple(temp_2_list)

            if a is None:
                a=move
                if self.N[temp_2_tuple]>=self.Ne:
                    var_2 = self.Q[temp_2_tuple]
                else:
                    var_2 = 1
            
            if self.N[temp_2_tuple] >= self.Ne:
                var = self.Q[temp_2_tuple]
            else:
                var = 1
            
            if var>var_2:
                a=move
                var_2 = var
            
        final_ans = m_S.copy()
        final_ans.append(a)

        final_solution = tuple(final_ans)
        self.N[final_solution]+=1

        if dead:
            self.reset()
            return 0

        self.a = a
        self.points = points
        self.s = tuple(m_S)
        return self.actions[a]

                
