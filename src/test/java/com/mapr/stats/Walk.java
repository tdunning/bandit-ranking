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

import com.google.common.collect.Lists;

import java.util.List;

public class Walk {
    public static void main(String[] args) {
        List<BetaWalk> x = Lists.newArrayList();
        for (int i = 0; i < 30; i++) {
            x.add(new BetaWalk(1, 40, 0.001));
        }
        double[] p = new double[x.size()];

        for (long j = 0; j < 20000; j++) {
            for (int i = 0; i < 30; i++) {
                p[i] = x.get(i).step();
            }

            print(j, p);
        }
    }

    private static void print(long step, double[] p) {
        System.out.printf("%d", step);
        for (int i = 0; i < 30; i++) {
            System.out.printf("\t%.8f", p[i]);
        }
        System.out.printf("\n");
    }
}
