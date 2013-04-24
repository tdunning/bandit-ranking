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

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class DistributionTest {
    protected void checkDistribution(AbstractContinousDistribution d, String test, double epsilon) {
        // pull a bunch of samples
        int n = 40001;
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = d.nextDouble();
        }

        // sort
        Arrays.sort(s);

        // and compare all 5 quartiles to see if things look about right
        for (int q = 0; q <= 4; q++) {
            assertEquals(test + ", q = " + q, d.cdf(s[(n - 1) * q / 4]), q / 4.0, epsilon);
        }
    }

    protected void cdfCheck(double[] expected, double[] cdf, AbstractContinousDistribution d) {
        // check against precomputed cdf values
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], d.cdf(cdf[i]), 1e-10);
        }
    }

    protected void pdfCheck(double[] xValues, double[] expected, AbstractContinousDistribution d) {
        int i = 0;
        for (double x : xValues) {
            assertEquals(expected[i], d.pdf(x), 1e-15);
            i++;
        }
    }
}
