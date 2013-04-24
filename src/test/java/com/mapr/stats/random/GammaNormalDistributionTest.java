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

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.jet.random.AbstractContinousDistribution;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class GammaNormalDistributionTest {
    @Test
    public void testEstimation() {
        final Random gen = new Random(1);
        GammaNormalDistribution gnd = new GammaNormalDistribution(0, 1, 1, gen);

        for (int i = 0; i < 10000; i++) {
            gnd.add(gen.nextGaussian() * 2 + 1);
        }

        assertEquals(1.0, gnd.nextMean(), 0.05);
        assertEquals(2.0, gnd.nextSD(), 0.1);

        double[] x = new double[10000];
        double[] y = new double[10000];
        double[] z = new double[10000];
        AbstractContinousDistribution dist = gnd.posteriorDistribution();
        for (int i = 0; i < 10000; i++) {
            x[i] = gnd.nextDouble();
            y[i] = dist.nextDouble();
            z[i] = gen.nextGaussian() * 2 + 1;
        }

        Arrays.sort(x);
        Arrays.sort(y);
        Arrays.sort(z);

        final Vector xv = new DenseVector(x).viewPart(1000, 8000);
        final Vector yv = new DenseVector(y).viewPart(1000, 8000);
        final Vector zv = new DenseVector(z).viewPart(1000, 8000);
        final double diffX = xv.minus(zv).assign(Functions.ABS).maxValue();
        final double diffY = yv.minus(zv).assign(Functions.ABS).maxValue();
        assertEquals(0, diffX, 0.13);
        assertEquals(0, diffY, 0.13);
    }
}
