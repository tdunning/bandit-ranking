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

import com.mapr.stats.bandit.BayesianBandit;

import java.util.Random;

/**
 * Multi-armed bandit problem where each reward is normally distributed with a gamma prior. An arm
 * is selected by sampling from the mean of the current posterior for each arm and picking the one
 * with highest sampled probability.
 */
public class GammaNormalBayesModel extends BayesianBandit {
    public GammaNormalBayesModel(int bandits, Random gen) {
        for (int i = 0; i < bandits; i++) {
            addModelDistribution(new GammaNormalDistribution(0, 1, 1, gen));
        }
    }
}
