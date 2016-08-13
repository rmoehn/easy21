from easy21.core import * # pylint: disable=wildcard-import, unused-wildcard-import

next_dtimestep = driver(step, think)
train_and_prep = make_train_and_prep(reset, next_dtimestep, wrapup)
first_timestep_vecs = iterate(train_and_prep, DTimestep(reset(), 0, init(0.3), None))
x = first_timestep_vecs.next()

x = first_timestep_vecs.next()
