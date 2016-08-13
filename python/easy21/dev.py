from easy21.core import * # pylint: disable=wildcard-import, unused-wildcard-import

import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pyplot

next_dtimestep = driver(step, think)
train_and_prep = make_train_and_prep(reset, next_dtimestep, wrapup)
first_timestep_vecs = iterate(train_and_prep, DTimestep(reset(), 0, init(0.5), None))
n = 10000
next(itertools.islice(first_timestep_vecs, n, n), None)
final_Q = first_timestep_vecs.next().experience.Q


X.flatten()


Y

V.flatten()

V

np.savetxt('data.csv', V_from_Q(final_Q).transpose())

# Credits: http://stackoverflow.com/a/11776477/5091738
X, Y    = np.meshgrid(np.arange(1, 22), np.arange(1, 11))
V       = V_from_Q(final_Q)
figure  = pyplot.figure()
axes    = figure.gca(projection='3d')
surface = axes.plot_wireframe(Y, X, V, rstride=1,
                            cstride=1, antialiased=False,
                            linewidth=1)
pyplot.show()
