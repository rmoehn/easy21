from easy21.core import * # pylint: disable=wildcard-import, unused-wildcard-import

import easy21.linfa as linfa

import itertools

from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import matplotlib.pyplot as pyplot

#next_dtimestep = driver(step, think)
next_dtimestep = driver(step, linfa.think)
#train_and_prep = make_train_and_prep(reset, next_dtimestep, wrapup)
train_and_prep = make_train_and_prep(reset, next_dtimestep, linfa.wrapup)

# pylint: disable=redefined-outer-name
def train(n_episodes, lmbda):
    #dtimestep = DTimestep(reset(), 0, init(lmbda), None)
    dtimestep = DTimestep(reset(), 0, linfa.init(lmbda), None)
    for _ in xrange(n_episodes):
        dtimestep = train_and_prep(dtimestep)

    #return V_from_Q(dtimestep.experience.Q)
    return V_from_Q(linfa.Q_from_theta(dtimestep.experience.theta))

Vs = np.array([train(1000, lmbdax10 * 0.1) for lmbdax10 in xrange(0, 11)])

linfa_V = train(10000, 0.2)

monte_carlo_V = np.loadtxt('../clojure/data.csv')

msq_errors = np.sum((Vs - monte_carlo_V) ** 2, axis=(1, 2))

pyplot.plot(msq_errors)
pyplot.show()

n = 10000
Qs = np.empty((n, 10, 21, 2))
dtimestep = DTimestep(reset(), 0, init(0.2), None)
for i in xrange(n):
    dtimestep = train_and_prep(dtimestep)
    Qs[i] = np.copy(dtimestep.experience.Q)

Vs_per_dt = np.amax(Qs, axis=3)
msq_errors_per_dt = np.sum((Vs_per_dt - monte_carlo_V) ** 2, axis=(1,2))
pyplot.plot(msq_errors_per_dt)
pyplot.show()

#next(itertools.islice(first_timestep_vecs, n, n), None)
#final_Q = first_timestep_vecs.next().experience.Q


norms = np.linalg.norm(              # Norms of differences
            np.diff(                 # Differences in Vs
                np.amax(Qs, axis=3), # Vs from Qs
                axis=0
            ),
            ord='fro', axis=(1,2))



#np.savetxt('data.csv', V_from_Q(final_Q).transpose())

# Credits: http://stackoverflow.com/a/11776477/5091738
final_Q = Qs[-1]
X, Y    = np.meshgrid(np.arange(1, 22), np.arange(1, 11))
V       = linfa_V #V_from_Q(final_Q)
figure  = pyplot.figure(1)
axes    = figure.add_subplot(211, projection='3d')
surface = axes.plot_wireframe(Y, X, V, rstride=1,
                            cstride=1, antialiased=False,
                            linewidth=0.5)
#pyplot.subplot(212)
#pyplot.plot(norms)
pyplot.show(block=False)
