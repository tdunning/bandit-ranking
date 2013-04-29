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
 * Expresses the common characteristics of a two-level distribution in which
 * the higher level distribution describes a prior distribution of parameters
 * for the lower level distribution.  Generically speaking, we have
 * \[
 *   \theta ~ p_1(\alpha) \\
 *   x | \theta ~ p_2(\theta)
 * \]
 * Here learning involves computing the posterior distribution of \(\theta \mid x\).  Note
 * that \(\theta\) is really a theoretical entity here and isn't really required of an
 * implementation.  The only required operations include:
 * <ul>
 * <li>nextDouble() Sample \(x\) from the posterior of \(p_2\).</li>
 * <li>nextMean() Sample \(E[x]\) from the posterior of \(p_2\).</li>
 * <li>add() Add a new observation x to define a new posterior distribution.</li>
 * <li>posteriorDistribution() Return a copy of the posterior distribution.</li>
 * </ul>
 */
public abstract class AbstractBayesianDistribution extends AbstractContinousDistribution {
    @Override
    public abstract double nextDouble();

    public abstract void add(double x);

    public abstract double nextMean();

    public abstract AbstractContinousDistribution posteriorDistribution();

    public abstract double getMean();

    public abstract double getSamples();
}
