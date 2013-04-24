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

import com.mapr.stats.random.BetaDistribution;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.VectorFunction;


/**
 * Solves the contextual bandit problem using Bayesian sampling.
 */
public class ContextualBayesBandit {
    private final Matrix featureMap;
    private final Matrix state;
    private final int m;
    private final BetaDistribution rand;

    public ContextualBayesBandit(Matrix featureMap) {
        this(featureMap, 1, 1);
    }

    public ContextualBayesBandit(Matrix featureMap, double alpha_0, double beta_0) {
        this.featureMap = featureMap;
        m = featureMap.numCols();
        this.state = new DenseMatrix(m, 2);
        this.state.viewColumn(0).assign(alpha_0);
        this.state.viewColumn(1).assign(beta_0);
        this.rand = new BetaDistribution(1, 1);
    }

    public Vector samplePi() {
        return sampleNoLink().assign(new LogisticFunction());
    }

    public int sample() {
        final Vector pi = sampleNoLink();
        return pi.maxValueIndex();
    }

    private Vector sampleNoLink() {
        final Vector theta = state.aggregateRows(new VectorFunction() {
            final DoubleFunction inverseLink = new InverseLogisticFunction();

            @Override
            public double apply(Vector f) {
                return inverseLink.apply(rand.nextDouble(f.get(0), f.get(1)));
            }
        });
        return featureMap.times(theta);
    }

    public void train(int bandit, boolean success) {
        state.viewColumn(success ? 0 : 1).assign(featureMap.viewRow(bandit), Functions.plusMult(1.0 / m));
    }

    public class LogisticFunction implements DoubleFunction {
        @Override
        public double apply(double x) {
            return 1 / (1 + Math.exp(-x));
        }
    }

    public class InverseLogisticFunction implements DoubleFunction {
        @Override
        public double apply(double p) {
            return Math.log(p / (1 - p));
        }
    }
}
