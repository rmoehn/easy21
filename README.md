# Solution Attempts for Easy21 Assignment

This assignment was part of David Silver's
[course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) in
reinforcement learning at UCL. I didn't attend it in person, but I've been
watching (at the time of writing) the [lecture
videos](https://youtu.be/2pWv7GOvuf0).

My solutions are not complete, not neat and not well-documented, but since I
didn't find much material on Easy21 on the Internet, I thought I'd publish them.
You will find two subdirectories, `clojure` and `python`. I implemented the
Monte Carlo control algorithm in Clojure and the rest in Python. The Clojure
implementation is naive, using `HashMap`s and so on. I just wanted to see how it
goes.

With Python I used `numpy` from the start, not least because I thought a naive
implementation in Python would be unbearably slow. I found out that `numpy` is
crazy powerful. I wonder if I could be as concise with `core.matrix`.

I experimented in the IPython REPL for some time, but I felt that that is not a
good way. – Always closing the graphs and overwriting my experiment code. So I
tried Jupyter notebooks again and now I like them! See `python/notebooks`.

## Copyright and License

See `LICENSE.txt`.
