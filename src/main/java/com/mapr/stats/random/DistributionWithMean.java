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

/**
 * Represents a distribution that knows it's own mean.
 */
public class DistributionWithMean extends AbstractContinousDistribution implements Comparable<DistributionWithMean> {
    private AbstractContinousDistribution delegate;
    private double mean;

    public DistributionWithMean(AbstractContinousDistribution delegate, double mean) {
        this.delegate = delegate;
        this.mean = mean;
    }

    public double getMean() {
        return mean;
    }

    @Override
    public double nextDouble() {
        return delegate.nextDouble();
    }

    @Override
    public int compareTo(DistributionWithMean other) {
        return Double.compare(getMean(), other.getMean());
    }
}
