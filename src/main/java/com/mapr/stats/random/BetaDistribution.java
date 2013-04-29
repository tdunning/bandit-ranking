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

import java.util.Random;

/**
 * Sample from a beta distribution.
 *
 * The beta distribution has PDF of
 * \[
 * p(x \mid \alpha, \alpha) = {\frac {\Gamma(\alpha+\beta)} {\Gamma(\alpha) \Gamma(\beta}} x^{\alpha-1} (1-x)^{\beta-1}
 * \]
 * Note that \( \frac {\Gamma(\alpha) \Gamma(\beta)}  {\Gamma(\alpha+\beta)} \)
 * and is known as the beta function \(B(\alpha, \beta)\).
 *
 * Sampling from the beta distribution \(x \sim B(\alpha, \beta)\) can be done by using the following procedure which depends
 * on sampling from the gamma distribution
 * \[
 * u \sim \Gamma(\alpha, 1) \\
 * v \sim \Gamma(\beta, 1) \\
 * x = \frac u {u+v}
 * \]
 */
public class BetaDistribution extends AbstractContinousDistribution {
    private final Gamma gAlpha;
    private final Gamma gBeta;
    private double alpha, beta;

    public BetaDistribution(double alpha, double beta, Random random) {
        this.alpha = alpha;
        this.beta = beta;
        gAlpha = new Gamma(alpha, 1, random);
        gBeta = new Gamma(beta, 1, random);
    }

    public BetaDistribution(double alpha, double beta) {
        this(alpha, beta, RandomUtils.getRandom());
    }

    /**
     * Returns a random number from the distribution.
     *
     * @return A new sample from this distribution.
     */
    @Override
    public double nextDouble() {
        double x = gAlpha.nextDouble(alpha, 1);
        double y = gBeta.nextDouble(beta, 1);
        return x / (x + y);
    }

    public double nextDouble(double alpha, double beta) {
        double x = gAlpha.nextDouble(alpha, 1);
        double y = gBeta.nextDouble(beta, 1);
        return x / (x + y);
    }

    @Override
    public double pdf(double x) {
        return Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1) / org.apache.mahout.math.jet.stat.Gamma.beta(alpha, beta);
    }

    public double logPdf(double x) {
        return x * (alpha - 1) + (1 - x) * (beta - 1) - Math.log(org.apache.mahout.math.jet.stat.Gamma.beta(alpha, beta));
    }

    @Override
    public double cdf(double x) {
        return org.apache.mahout.math.jet.stat.Gamma.incompleteBeta(alpha, beta, x);
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public double getBeta() {
        return beta;
    }

    public double getAlpha() {
        return alpha;
    }

    public double mean() {
        return alpha / (alpha + beta);
    }

    /**
     * Sets the uniform random generator internally used.
     *
     * @param rand the new PRNG
     */
    @Override
    public void setRandomGenerator(Random rand) {
        gAlpha.setRandomGenerator(rand);
        gBeta.setRandomGenerator(rand);
    }
}
