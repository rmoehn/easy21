# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np
import pyrsistent

import easy21.core as easy21

LinfaExperience = pyrsistent.immutable(
                    'theta, E, N0, epsi, alpha, lmbda, p_obs, p_act')


def init(lmbda):
    return LinfaExperience(theta=np.zeros((10)),
                           E=np.zeros((10, 21, 2)), # Same indices as feature map
                           N0=100,
                           epsi=0.05,
                           alpha=0.01,
                           lmbda=lmbda,
                           p_obs=None, # p … previous
                           p_act=None)


#### Make a feature lookup table

def feature_slow(o, a):
    d = o.dealer_sum
    p = o.player_sum
    booleans = [1 <= d <= 4, 4 <= d <= 7, 7 <= d <= 10,
                1 <= p <= 6, 4 <= p <= 9, 7 <= p <= 12, 10 <= p <= 15,
                13 <= p <= 18, 16 <= p <= 21,
                a == easy21.Action.HIT] # Then 1/True indicates HIT.
    return np.array(booleans, np.bool)


# Note: This is not what the exercise specifies, I think, but I don't understand
# how that what the exercise specifies makes sense. As I understand it, the
# features vectors in the the exercise are 36-element vectors in which exactly
# one element is 1 and the others are zero. The 1 indicates that the state is in
# certain dealer card intervals, certain player card intervals and that we've
# chose a certain action. Why not encode these separately? This is what I've
# done here. I'll see if it works or not.
def prepare_feature():
    return np.array([
                        [
                            [feature_slow(easy21.Observation(d, p), a)
                                 for a in [easy21.Action.STICK,
                                           easy21.Action.HIT]
                            ]
                            for p in xrange(1, 22)
                        ]
                        for d in xrange(1, 11)
                    ])


feature = prepare_feature()


def true_with_prob(p):
    return np.random.choice(2, p=[1-p, p])


def choose_action(e, o):
    if true_with_prob(e.epsi):
        return easy21.rand_action()
    else:
        stick_return = feature[o.dealer_sum - 1, o.player_sum - 1,
                               easy21.Action.STICK].dot(e.theta)
        hit_return   = feature[o.dealer_sum - 1, o.player_sum - 1,
                               easy21.Action.HIT].dot(e.theta)
        return easy21.Action.STICK if stick_return > hit_return \
                                   else easy21.Action.HIT
            # Python has not built-in readable argmax. numpy would be overkill.



def think(e, o, r, done=False):
    """

    Args:
        e … experience
        o … observation
        r … reward
    """

    if not done:
        a     = choose_action(e, o) # action
        feat  = feature[o.dealer_sum - 1, o.player_sum - 1, a]
        Qnext = feat.dot(e.theta)
            # expected Q of next action
    else:
        a     = None
        Qnext = 0

    if e.p_obs: # Except for first timestep.
        p_feat = feature[e.p_obs.dealer_sum - 1, e.p_obs.player_sum - 1,
                         e.p_act]
        Qcur  = p_feat.dot(e.theta)
        delta = r + Qnext - Qcur
        e.E[e.p_obs.dealer_sum - 1, e.p_obs.player_sum - 1, e.p_act] += 1

        elig = e.E[e.p_obs.dealer_sum - 1, e.p_obs.player_sum - 1, e.p_act]
        e.theta.__isub__(e.alpha * delta * elig * p_feat)

        e.E.__imul__(e.lmbda)

    return e.set(p_obs=o, p_act=a), a


def wrapup(e, o, r):
    e, _ = think(e, o, r, done=True)
    return e.set(p_obs=None, p_act=None)


def Q_from_theta(theta):
    return feature.dot(theta) # Matrix of products of each feature vector with
                              # weights.
