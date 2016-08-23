from easy21.core import * # pylint: disable=wildcard-import, unused-wildcard-import

import easy21.linfa as linfa

import itertools

from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import matplotlib.pyplot as pyplot

next_dtimestep = driver(step, think)
train_and_prep = make_train_and_prep(reset, next_dtimestep, wrapup)

next_dtimestep_linfa = driver(step, linfa.think)
train_and_prep_linfa = make_train_and_prep(reset, next_dtimestep_linfa,
                                           linfa.wrapup)


# pylint: disable=redefined-outer-name
def train(n_episodes, lmbda):
    dtimestep = DTimestep(reset(), 0, init(lmbda), None)
    for _ in xrange(n_episodes):
        dtimestep = train_and_prep(dtimestep)

    return V_from_Q(dtimestep.experience.Q)


def train_linfa(n_episodes, lmbda, alpha=0.01, feature=None):
    feature = feature if feature is not None \
                      else linfa.prepare_feature(linfa.feature_slow)
    init_experience = linfa.init(lmbda, alpha, feature)

    dtimestep = DTimestep(reset(), 0, init_experience, None)
    for _ in xrange(n_episodes):
        dtimestep = train_and_prep_linfa(dtimestep)

    return V_from_Q(linfa.Q(dtimestep.experience))


def train_Qs(n_episodes, lmbda):
    Qs = np.empty((n_episodes, 10, 21, 2))
    dtimestep = DTimestep(reset(), 0, init(lmbda), None)
    for i in xrange(n_episodes):
        dtimestep = train_and_prep(dtimestep)
        Qs[i] = np.copy(dtimestep.experience.Q)

    return Qs


def train_Qs_linfa(n_episodes, lmbda, alpha=0.01, feature=None):
    feature = feature if feature is not None \
                      else linfa.prepare_feature(linfa.feature_slow)
    init_experience = linfa.init(lmbda, alpha, feature)

    Qs = np.empty((n_episodes, 10, 21, 2))
    dtimestep = DTimestep(reset(), 0, init_experience, None)
    for i in xrange(n_episodes):
        dtimestep = train_and_prep_linfa(dtimestep)
        Qs[i] = linfa.Q(dtimestep.experience)

    return Qs


# Credits: http://stackoverflow.com/a/11776477/5091738
def plot_V(V):
    X, Y    = np.meshgrid(np.arange(1, 22), np.arange(1, 11))
    figure  = pyplot.figure(1)
    axes    = figure.gca(projection='3d')
    axes.plot_wireframe(Y, X, V, rstride=1,
                        cstride=1, antialiased=False, linewidth=0.5)
    pyplot.show()


def plot_msq_errors(train, V, n_episodes):
    Vs = np.array([train(n_episodes, lmbdax10 * 0.1)
                       for lmbdax10 in xrange(0, 11)])
    msq_errors = np.sum((Vs - V) ** 2, axis=(1, 2))
    pyplot.plot(msq_errors)
    pyplot.show()


def plot_avg_msq_errors(train, V, n_episodes, n_runs=100):
    Vss = np.array([
                       [train(n_episodes, lmbdax10 * 0.1)
                            for lmbdax10 in xrange(0, 11)]
                       for _ in xrange(n_runs)
                   ])
    msq_errors = np.sum((Vss - V) ** 2, axis=(0, 2, 3)) \
                     * (1.0 / n_runs / 10 / 21)
    pyplot.plot(msq_errors)
    pyplot.show()


def plot_learning_rate(V, Qs):
    Vs_per_dt = np.amax(Qs, axis=3)
    msq_errors_per_dt = np.sum((Vs_per_dt - V) ** 2, axis=(1,2))
    pyplot.plot(msq_errors_per_dt)
    pyplot.show()
#
##next(itertools.islice(first_timestep_vecs, n, n), None)
##final_Q = first_timestep_vecs.next().experience.Q
#
#
#norms = np.linalg.norm(              # Norms of differences
#            np.diff(                 # Differences in Vs
#                np.amax(Qs, axis=3), # Vs from Qs
#                axis=0
#            ),
#            ord='fro', axis=(1,2))
