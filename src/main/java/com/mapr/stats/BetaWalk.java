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

package com.mapr.stats;


import com.mapr.stats.random.BetaDistribution;
import org.apache.mahout.common.RandomUtils;

import java.util.Random;

/**
 * Follow a Metropolis random walk that should converge to a beta distribution.  Since the point is
 * to have slowly changing probabilities for simulating non-stationary conversion processes, having
 * substantial sample to sample correlation is good here.
 * <p/>
 * The probabilities returned will be beta distributed if you take enough steps.  Steps are proposed
 * using a normally distributed step in soft-max space which gives a random walk bounded to (0,1) in
 * probability space.  The proposal distribution winds up taking very small steps near the
 * boundaries with larger steps in the middle.  Steps are accepted or rejected according to the
 * Metropolis algorithm.  Computing the probabilities for acceptance or rejection in the probability
 * space while taking the step in log-odds space is OK since the proposal probability is still
 * symmetrical.
 */
public class BetaWalk {
    private final Random rand = RandomUtils.getRandom();
    private final double stepSize;

    private final BetaDistribution bd;

    private double x;
    private double pdf = 0;

    public BetaWalk(double alpha, double beta, double stepSize) {
        this.bd = new BetaDistribution(alpha, beta, rand);
        this.stepSize = stepSize;
        x = bd.nextDouble();
        if (x < 0) {
            System.out.printf("heh?\n");
        }
        pdf = bd.pdf(x);
    }

    public double step() {
        double x1 = x + rand.nextGaussian() * stepSize;
        double pdf1 = bd.pdf(x1);
        if (x1 < 0 || x1 > 1) {
            pdf1 = 0;
        }

        if (pdf1 > 0 && (pdf1 > pdf || rand.nextDouble() < pdf1 / pdf)) {
            if (x1 < 0) {
                System.out.printf("huh?\n");
            }

            x = x1;
            pdf = pdf1;
        }
        return x;
    }
}
