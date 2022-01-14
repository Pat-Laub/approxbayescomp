# What is ABC?

Let's consider a toy example.
Imagine we are trying to determine the probability $p = \mathbb{P}(\text{Heads})$ of getting a heads when flipping a biased coin.
We observe just three coin flips, and get `Heads, Tails, Heads`.
One way to fit this data is to just start _flipping (virtual) coins_.
Specifically, we randomly guess a value $p' \in (0, 1)$, flip a biased coin with this probability of heads three times, then remember this value of $p'$ if this coin also got `Heads, Tails, Heads`.

<figure markdown>
  <figcaption>Illustration of exact posterior sampling with 3 observed coin tosses from a biased coin.</figcaption>
  {% include '/anims/anim-1-svg.html' %}  
</figure>

!!!tip

    Press the play button (triangular shaped) on these animations to start them.

In this simulation we found 15 values of $p'$ which happened to generate the same `Heads, Tails, Heads`.
These 15 values are actually samples from the _posterior distribution_ for $p$, given that we have a uniform prior belief.

Of course, this method of generating fake data which is _identical_ to the real observed data is not an efficient way to sample from the posterior distribution.
If we increase the sample size of this toy example to eight observed coin flips, the following animation shows that event after 100 attempts we don't find a single $p'$ which generates the exact same eight observations.

<figure markdown>
  <figcaption>Illustration of exact posterior sampling with 8 observed coin tosses from a biased coin.</figcaption>
  {% include '/anims/anim-2-svg.html' %}
</figure>

This is where the _approximate_ part of Approximate Bayesian Computation (ABC) comes in.
Let's just accept a $p'$ value if the fake data it generates is _pretty close_ to the observed data.
In the following, our observed data had five heads, so we accept any fake data that has 4, 5, or 6 heads.

<figure markdown>
  <figcaption>Illustration of ABC sampling with 8 observed coin tosses from a biased coin.</figcaption>
  {% include '/anims/anim-3-svg.html' %}
</figure>

This generated a much larger sample (29 points) from an _approximate posterior_ distribution.
We can compare the true posterior to the various approximate posterior distributions where our "accept when pretty close" rule is more or less stringent.

<figure markdown>
  <figcaption>Comparing the true posterior distribution to various ABC approximate posteriors.</figcaption>
  {% include '/anims/anim-4-svg.html' %}
</figure>
