/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mapr.stats.bandit;

import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.mapr.stats.random.AbstractBayesianDistribution;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Implements the common characteristics of the Bayesian Bandit.  All that is
 * missing is the specific distribution that is used to learn and to sample
 * the mean returns.
 * <p/>
 * Typically, an implementation only has to call addModelDistribution in its
 * constructor in order to specialize this class.
 * <p/>
 * This class can also be treated as an interface by non-Bayesian Bandit solvers
 * such as EpsilonGreedy.  Such classes will need to over-ride all of the methods
 * here.
 */
public abstract class BayesianBandit implements Iterable<AbstractBayesianDistribution> {
    // we have one distribution for each bandit
    private final List<AbstractBayesianDistribution> bd = Lists.newArrayList();

    /**
     * Samples probability estimates from each bandit and picks the apparent best
     *
     * @return 0 or 1 according to which bandit seems better
     */
    public int sample() {
        double max = Double.NEGATIVE_INFINITY;
        int r = -1;
        int i = 0;
        for (AbstractBayesianDistribution dist : bd) {
            double p = dist.nextMean();
            if (p > max) {
                r = i;
                max = p;
            }
            i++;
        }
        return r;
    }

    /**
     * Apply feedback to the bandit we chose.
     *
     * @param bandit Which bandit got the impression
     * @param reward Did it pay off?
     */
    public void train(int bandit, double reward) {
        bd.get(bandit).add(reward);
    }

    public boolean addModelDistribution(AbstractBayesianDistribution distribution) {
        return bd.add(distribution);
    }

    /**
     * Samples probability estimates from each bandit and orders the bandits in increasing order.
     * @param sampleSize The number of bandits to sample.
     * @return A list of the indexes of the bandits.
     */
    public List<Integer> rank(int sampleSize) {
        Map<Double, Integer> tmp = Maps.newTreeMap(Ordering.natural().reverse());
        int i = 0;
        for (AbstractBayesianDistribution dist : bd) {
            double p = dist.nextMean();
            tmp.put(p, i);
            i++;
        }

        List<Integer> r = Lists.newArrayList();
        for (Double key : tmp.keySet()) {
            r.add(tmp.get(key));
            if (r.size() >= sampleSize) {
                break;
            }
        }
        return r;
    }

    /**
     * Returns the mean of a particular distribution in the bandit
     */
    public double getMean(int k) {
        return bd.get(k).getMean();
    }

    /**
     * Returns an iterator over a set of elements of type T.
     *
     * @return an Iterator.
     */
    @Override
    public Iterator<AbstractBayesianDistribution> iterator() {
        return Iterators.unmodifiableIterator(bd.iterator());
    }
}
