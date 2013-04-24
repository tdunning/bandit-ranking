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

import org.apache.mahout.math.jet.random.Normal;

import java.util.Random;

/**
 * Returns a normal distribution whose mean is uniformly distributed on [0,1) and whose sd is as
 * specified.
 */
public class NormalDistributionSampler extends DistributionGenerator {
    private double sd;
    private Random gen;

    public NormalDistributionSampler(double sd, Random gen) {
        this.sd = sd;
        this.gen = gen;
    }

    @Override
    public DistributionWithMean nextDistribution() {
        double mean = gen.nextDouble();
        return new DistributionWithMean(new Normal(mean, sd, gen), mean);
    }
}
