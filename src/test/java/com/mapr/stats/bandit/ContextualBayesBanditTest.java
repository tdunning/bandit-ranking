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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Test;

import java.util.Random;

public class ContextualBayesBanditTest {
    @Test
    public void testConvergence() {
        final Random rand = RandomUtils.getRandom();
        Matrix recipes = new DenseMatrix(100, 10)
                .assign(new DoubleFunction() {
                    @Override
                    public double apply(double arg1) {
                        return rand.nextDouble() < 0.2 ? 1 : 0;
                    }
                });
        recipes.viewColumn(9).assign(1);

        Vector actualWeights = new DenseVector(new double[]{
                1, 0.25, -0.25, 0, 0,
                0, 0, 0, 0, -1});

        Vector probs = recipes.times(actualWeights);

        ContextualBayesBandit banditry = new ContextualBayesBandit(recipes);

        for (int i = 0; i < 1000; i++) {
            int k = banditry.sample();
            final boolean success = rand.nextDouble() < probs.get(k);
            banditry.train(k, success);
        }
    }
}
