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

package com.mapr.stats.random;

import org.apache.mahout.math.jet.random.AbstractContinousDistribution;

import java.util.Random;

/**
 * Implements a beta-binomial pair of conjugate distributions.
 * <p/>
 * In this model, samples are distributed according to
 * \[
 *   \pi \sim \mathrm {Beta}(\alpha, \beta) \\
 *   x \sim \mathrm {Bernoulli} (\pi)
 * \]
 * The nextDouble() method returns a sample of \(x\) and the nextMean()
 * returns a sample of \(\pi\).
 */
public class BetaBinomialDistribution extends AbstractBayesianDistribution {
    private final Random gen;
    private final BetaDistribution bd;

    public BetaBinomialDistribution(double alpha, double beta, Random gen) {
        this.gen = gen;
        bd = new BetaDistribution(alpha, beta, gen);
    }

    /**
     * Samples from a binomial whose underlying probability is distributed according to a
     * beta distribution.
     *
     * @return A sample.
     */
    @Override
    public double nextDouble() {
        // We don't actually have to sample the probability and then
        // sample the binomial since with only one sample, sampling directly from a binomial
        // with probability $\alpha / (\alpha + \beta)$ is just the same.
        return gen.nextDouble() < bd.getBeta() ? 1 : 0;
    }

    @Override
    public void add(double x) {
        if (x == 0.0) {
            bd.setBeta(bd.getBeta() + 1);
        } else if (x == 1) {
            bd.setAlpha(bd.getAlpha() + 1);
        } else {
            throw new IllegalArgumentException("Samples for beta-binomial distribution must be 0 or 1");
        }
    }

    @Override
    public double nextMean() {
        return bd.nextDouble();
    }

    @Override
    public AbstractContinousDistribution posteriorDistribution() {
        return createBernoulliDistribution(bd.getAlpha() / (bd.getAlpha() + bd.getBeta()));
    }

    @Override
    public double getMean() {
        return bd.mean();
    }

    @Override
    public double getSamples() {
        return bd.getAlpha() + bd.getBeta();
    }

    private AbstractContinousDistribution createBernoulliDistribution(final double p) {
        return new AbstractContinousDistribution() {
            @Override
            public double nextDouble() {
                return gen.nextDouble() < p ? 1 : 0;
            }
        };
    }
}
