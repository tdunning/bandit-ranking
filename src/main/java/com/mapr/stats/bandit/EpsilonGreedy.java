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

import com.google.common.collect.Lists;
import com.mapr.stats.random.AbstractBayesianDistribution;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.util.List;
import java.util.Random;

/**
 * Solves a bandit problem using an epsilon greedy algorithm.  In this algorithm, a fixed
 * proportion of trials are allocated to uniform exploration while all others are devoted
 * to the current best bandit alternative.
 */
public class EpsilonGreedy extends BayesianBandit {
    private Random gen;
    private double epsilon;
    private final List<OnlineSummarizer> summaries;

    public EpsilonGreedy(int bandits, double epsilon, Random gen) {
        this.gen = gen;
        this.epsilon = epsilon;
        summaries = Lists.newArrayList();
        for (int i = 0; i < bandits; i++) {
            final OnlineSummarizer s = new OnlineSummarizer();
            summaries.add(s);
            s.add(1);
        }
    }

    /**
     * Samples probability estimates from each bandit and picks the apparent best
     *
     * @return The index of the chosen bandit
     */
    @Override
    public int sample() {
        if (gen.nextDouble() < epsilon) {
            return gen.nextInt(summaries.size());
        } else {
            double max = summaries.get(0).getMean();
            int i = 0;
            int maxIndex = 0;
            for (OnlineSummarizer summary : summaries) {
                if (summary.getMean() > max) {
                    max = summary.getMean();
                    maxIndex = i;
                }
                i++;
            }
            return maxIndex;
        }
    }

    /**
     * Apply feedback to the bandit we chose.
     *
     * @param bandit Which bandit got the impression
     * @param reward Did it pay off?
     */
    @Override
    public void train(int bandit, double reward) {
        summaries.get(bandit).add(reward);
    }

    @Override
    public boolean addModelDistribution(AbstractBayesianDistribution distribution) {
        throw new UnsupportedOperationException("Can't add a distribution to epsilon greedy");
    }
}
