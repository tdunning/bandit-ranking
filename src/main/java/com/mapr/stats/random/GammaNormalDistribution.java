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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.jet.random.AbstractContinousDistribution;
import org.apache.mahout.math.jet.random.Gamma;
import org.apache.mahout.math.jet.random.Normal;

import java.util.Random;

/**
 * Samples from a Gamma Normal distribution.  Provision is made for adding
 * observations to update the posterior distribution.
 */
public class GammaNormalDistribution extends AbstractBayesianDistribution {
    private Random gen = RandomUtils.getRandom();
    private double m, n;
    private double ss;
    private final Gamma gd = new Gamma(1, 1, gen);
    private final Normal nd = new Normal(0, 1, gen);

    public GammaNormalDistribution(double m, double n, double sd, Random gen) {
        this.gen = gen;
        this.m = m;
        this.n = n;
        this.ss = sd * sd;
    }

    /**
     * Returns a random number from the distribution.
     *
     * @return A new sample from this distribution.
     */
    @Override
    public double nextDouble() {
        double variance = nextVariance();
        double mean = nd.nextDouble() * Math.sqrt(variance / n) + m;
        return nd.nextDouble() * Math.sqrt(variance) + mean;
    }

    @Override
    public void add(double x) {
        n += 1;
        final double delta = (x - m);
        m += delta / n;
        ss = ss + delta * (x - m);
    }

    @Override
    public double nextMean() {
        double sd = Math.sqrt(nextVariance() / n);
        return nd.nextDouble() * sd + m;
    }

    @Override
    public AbstractContinousDistribution posteriorDistribution() {
        return new Normal(m, Math.sqrt(ss / n), gen);
    }

    public double nextSD() {
        return Math.sqrt(nextVariance());
    }

    private double nextVariance() {
        return 1 / gd.nextDouble(n / 2, ss / 2);
    }
}
