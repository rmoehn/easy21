from easy21.core import * # pylint: disable=wildcard-import, unused-wildcard-import

import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pyplot

next_dtimestep = driver(step, think)
train_and_prep = make_train_and_prep(reset, next_dtimestep, wrapup)

n = 100000
Qs = np.empty((n, 10, 21, 2))
dtimestep = DTimestep(reset(), 0, init(0.8), None)
for i in xrange(n):
    dtimestep = train_and_prep(dtimestep)
    Qs[i] = np.copy(dtimestep.experience.Q)

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
V       = V_from_Q(final_Q)
figure  = pyplot.figure(1)
axes    = figure.add_subplot(211, projection='3d')
surface = axes.plot_wireframe(Y, X, V, rstride=1,
                            cstride=1, antialiased=False,
                            linewidth=0.5)
pyplot.subplot(212)
pyplot.plot(norms)
pyplot.show(block=False)

A = np.array([
    [[[3, 4],  [2, 8],  [1, 7]],
     [[6, 12], [5, 11], [4, 10]]],
    [[[12, 7], [11, 8], [10, 9]],
     [[3, 4],  [2, 5],  [7, 6]]]])

np.amax(A, axis=3)

A[1] - A[0]

np.diff(np.array([
    [[[3, 4],  [2, 8],  [1, 7]],
     [[6, 12], [5, 11], [4, 10]]],
    [[[12, 7], [11, 8], [10, 9]],
     [[3, 4],  [2, 5],  [7, 6]]]]),
    axis=0)

