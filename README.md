# Bandit Ranking Overview

The problem of ranking comments by a crowd-sourced version of "quality" is a common one on the internet.

James Neufeld suggests[1] that Bayesian Bandit algorithms can be applied to this problem.
The basic idea is that you would define a stochastic quality metric whose distribution for
each comment depends on the up and down votes that comment has received.

Normal ranking algorithms try to estimate the single best value for this quality metric.
Neufeld suggests that this value should be sampled from a beta distribution which models
the probability that a user would mark the comment positively given that they have marked
the comment at all. To present comments to a user, the metric would be sampled
independently for each comment and the comments would be sorted according to the resulting
scores. Different presentations would necessarily result in different orders, but as users
mark comments positively or negatively, the order should converge to one where the
comments presented near the top of the list have the highest probability of being marked
positively.

One very nice thing about this approach is that it doesn't waste any cycles on determining
the ranking of low quality comments. Once the quality of these comments has been
determined to be relatively lower than the best columns, no more learning need be done
with those comments. This accelerates learning of the ranking of the best options
dramatically.

[1] http://simplemlhacks.blogspot.ca/2013/04/reddits-best-comment-scoring-algorithm.html

# Running The Code

As you read this, you may want to look at the javadocs[2].

To compile this code you need Java 1.7 and Maven 3 and an internet connection. Once you
have these, do this:

    $ mvn -q package

To run the code on a sample problem, do this:

    $ java -jar target/bandit-ranking-1.0-SNAPSHOT-jar-with-dependencies.jar  [k [p [n]]]

The output will be contained in two files.  The file named "quality.csv" will contain
points that describe a graph of the precision of the ranking for the
first p presented items, the per trial regret and the cumulative regret. Precision is
measured by how many of the top k items are shown in the first p items where there are
n items total. The default values are useful or you can try k=10, p=20, n=500 for a 
quick experiment. In this simulation, we assume that each user will look at the first 
page and will rate all of the items on that page according to the actual quality of the 
item. Item quality will be sampled from a uniform distribution. In order to get decent 
averages, the system will run many (50) simulations in parallel.

The file named "samples.csv" will contain data that defines a graph which shows the number
of samples taken as a function of the rank of a comment.  You should see a sharp drop at the
end of the first page indicating that only the highest quality comments are actually
characterized carefully.

[2] http://tdunning.github.io/bandit-ranking/
