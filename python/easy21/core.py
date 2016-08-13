# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import itertools
import random

import numpy as np
import pyrsistent

# Timestep that the driver sees
DTimestep = pyrsistent.immutable('state, reward, experience, action',
                name='DTimestep')

Observation = pyrsistent.immutable('dealer_sum, player_sum', name='Observation')

# Environment state
State = pyrsistent.immutable('observation, done', name='State')

# Timestep that the learner sees
LTimestep = pyrsistent.immutable('observation, reward, action',
                name='ATimestep')

Experience = pyrsistent.immutable(
                 'Q, E, N0, Ns, Nsa, lmbda, prev_action, prev_observation')

class Action(object): # pylint: disable=too-few-public-methods
    STICK = 0
    HIT   = 1

#### General helper

def iterate(f, x):
    while True:
        yield x
        x = f(x)


def driver(step, think): # pylint: disable=redefined-outer-name
    def next_dtimestep(dtimestep):
        if dtimestep.state.done:
            return dtimestep.set(reward=0, action=None)

        new_experience, new_action = think(dtimestep.experience,
                                           dtimestep.state.observation,
                                           dtimestep.reward)
        new_state, new_reward = step(dtimestep.state, new_action)

        return DTimestep(new_state, new_reward, new_experience, new_action)

    return next_dtimestep


# pylint: disable=redefined-outer-name
def make_train_and_prep(reset, next_dtimestep, wrapup):
    def train_and_prep(first_dtimestep):
        last_dtimestep = itertools.dropwhile(lambda dt: not dt.state.done,
                             iterate(next_dtimestep, first_dtimestep))\
                             .next()
        experience = wrapup(last_dtimestep.experience,
                            last_dtimestep.state.observation,
                            last_dtimestep.reward)
        return DTimestep(reset(), 0, experience, None)

    return train_and_prep



def is_bust(card_sum):
    return not 1 <= card_sum <= 21


def rand_color():
    return -1 if random.randint(1, 3) == 1 else 1


def rand_number():
    return random.randint(1, 10)


def rand_card():
    return rand_color() * rand_number()


def rand_action():
    return random.randint(0, 1) # Refers to Action.


def hit_step(state):
    new_player_sum = state.observation.player_sum + rand_card()

    return (State(state.observation.set(player_sum=new_player_sum),
                  is_bust(new_player_sum)),
            -1 if is_bust(new_player_sum) else 0)

def signum(n):
    return -1 if n < 0 else 1 if n > 0 else 0

def stick_step(state):
    final_dealer_sum \
        = itertools.dropwhile(lambda s: not is_bust(s) or s < 17,
              iterate(lambda s: s + rand_card(),
                      state.observation.dealer_sum)) \
              .next()
    reward = 1 if is_bust(final_dealer_sum) \
               else signum(state.observation.player_sum - final_dealer_sum)

    return (State(state.observation.set(dealer_sum=final_dealer_sum),
                  True),
            reward)


def step(state, action):
    """


    Returns: (State, Reward)
    """
    return [stick_step, hit_step][action](state) # Depends on Action.


def reset():
    return State(Observation(rand_number(), rand_number()), False)


def randlist(a, b, n):
    return [random.randint(a, b) for _ in xrange(n)]

# Number of states: 21 player sums x 10 initial dealer cards
def init(lmbda):
    return Experience(Q=np.zeros((10, 21, 2)),
                      E=np.zeros((10, 21, 2)),
                      N0=100,
                      Ns=np.zeros((10, 21), np.int64),
                      Nsa=np.zeros((10, 21, 2), np.int64),
                      lmbda=lmbda,
                      prev_observation=None,
                      prev_action=None)


def is_rand_explore(experience, observation):
    return random.randint(1, experience.N0
                             + experience.Ns[observation.dealer_sum - 1,
                                             observation.player_sum - 1]) \
           <= experience.N0


def choose_action(experience, observation):
    if is_rand_explore(experience, observation):
        return rand_action()
    else:
        return np.argmax(experience.Q[observation.dealer_sum - 1,
                                      observation.player_sum - 1])


# Follows Sutton and Barto, book2015oct.pdf, p. 162
def think(experience, observation, reward, done=False): # Nasty and mutating.
    if not done:
        action = choose_action(experience, observation)
        experience.Ns[observation.dealer_sum - 1, observation.player_sum - 1] \
            += 1
            # In this way I interpret »number of times that state s has been
            # visited« from the exercise text as "the number of times we've seen
            # it before, excluding this time".
        Qnext  = experience.Q[observation.dealer_sum - 1,
                              observation.player_sum - 1,
                              action]
    else:
        action = None
        Qnext  = 0


    if experience.prev_observation: # Except for first timestep.
        delta  = reward + Qnext \
                        - experience.Q[experience.prev_observation.dealer_sum-1,
                                       experience.prev_observation.player_sum-1,
                                       experience.prev_action]

        experience.E[experience.prev_observation.dealer_sum - 1,
                     experience.prev_observation.player_sum - 1,
                     experience.prev_action] += 1

        experience.Nsa[experience.prev_observation.dealer_sum - 1,
                       experience.prev_observation.player_sum - 1,
                       experience.prev_action] += 1

        alpha = 1.0 / experience.Nsa[experience.prev_observation.dealer_sum - 1,
                                     experience.prev_observation.player_sum - 1,
                                     experience.prev_action]

        experience.Q.__iadd__(experience.E * alpha * delta)

        experience.E.__imul__(experience.lmbda)

    return experience.set(prev_observation=observation, prev_action=action), \
           action
        # The other fields refer to nparrays that have been mutated.


def wrapup(experience, observation, reward):
    experience, _ = think(experience, observation, reward, done=True)
    return experience.set(prev_observation=None, prev_action=None,
                          E=np.zeros((10, 21, 2)))


def V_from_Q(Q):
    return np.amax(Q, axis=2)
