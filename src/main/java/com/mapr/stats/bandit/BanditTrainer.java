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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.mapr.stats.random.BinomialDistributionSampler;
import com.mapr.stats.random.DistributionGenerator;
import com.mapr.stats.random.DistributionWithMean;
import com.mapr.stats.random.NormalDistributionSampler;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Simulate a two-armed bandit playing against a beta-Bayesian model.
 * <p/>
 * The output indicates the quantiles of the distribution for regret relative to the optimal pick.
 * The regret distribution is estimated by picking two random conversion probabilities and then
 * running the beta-Bayesian model for a number of steps.  The regret is computed by taking the
 * expectation for the optimal choice and subtracting from the actual percentage of conversion
 * achieved.  On average, this should be somewhat negative since the model has to spend some effort
 * examining the sub-optimal choice.  The median, 25 and 75%-ile marks all scale downward fairly
 * precisely with the square root of the number of trials which is to be expected from theoretical
 * considerations.
 * <p/>
 * The beta-Bayesian model works by keeping an estimate of the posterior distribution for the
 * conversion probability for each of the bandits.  We take a uniform distribution as the prior so
 * the posterior is a beta distribution.  The model samples probabilities from the two posterior
 * distributions and chooses the model whose sample is larger. As data is collected for the two
 * bandits, the better of the bandits will quickly have a pretty narrow posterior distribution and
 * the lesser bandit will rarely have a sampled probability higher than the better bandit.  This
 * means that we will stop getting data from the less bandit, but only when there is essentially no
 * chance that it is better.
 */
public class BanditTrainer {
    private static final int BUCKET_SIZE = 1;

    public static void main(String[] args) throws FileNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException, InterruptedException {
        int threads = 16;

        if (args.length > 0) {
            threads = Integer.parseInt(args[0]);
        }

        System.out.printf("regret\n");
        ExecutorService ex = Executors.newFixedThreadPool(threads);

        List<Callable<Integer>> tasks = ImmutableList.of(
                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret-normal-0.1.tsv", "local-normal-0.1.tsv", 1000, 2, 10000, new GammaNormalBayesFactory(), new NormalDistributionSampler(0.1, new Random()));
                            System.out.printf("2\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                },
                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret-epsilon-normal-1.tsv", "local-epsilon-normal-1.tsv", 1000, 2, 10000, new EpsilonGreedyFactory(0.05), new NormalDistributionSampler(1, new Random()));
                            System.out.printf("2e\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                },


                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret-normal-1.tsv", "local-normal-1.tsv", 300, 2, 200000, new GammaNormalBayesFactory(), new NormalDistributionSampler(1, new Random()));
                            System.out.printf("normal 1\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                },

                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret-normal-10x0.1.tsv", "local-normal-10x0.1.tsv", 1000, 10, 1000, new GammaNormalBayesFactory(), new NormalDistributionSampler(0.1, new Random()));
                            System.out.printf("10\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                },

                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret-normal-100x0.1.tsv", "local-normal-100x0.1.tsv", 1000, 100, 1000, new GammaNormalBayesFactory(), new NormalDistributionSampler(.1, new Random()));
                            System.out.printf("100\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                },

                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret.tsv", "local.tsv", 1000, 2, 1000, new BetaBayesFactory(), new BinomialDistributionSampler(1, 1, new Random()));
                            System.out.printf("2\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                },

                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret-100.tsv", "local-100.tsv", 1000, 100, 1000, new BetaBayesFactory(), new BinomialDistributionSampler(1, 1, new Random()));
                            System.out.printf("100\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                },

                new Callable<Integer>() {
                    @Override
                    public Integer call() {
                        try {
                            totalRegret("regret-20.tsv", "local-20.tsv", 1000, 20, 1000, new BetaBayesFactory(), new BinomialDistributionSampler(1, 1, new Random()));
                            System.out.printf("20\n");
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }
                        return null;
                    }
                }
        );
        ex.invokeAll(tasks);
        ex.shutdown();
        System.out.printf("All done");

//    System.out.printf("error rates\n");
//    errorRate("errors.tsv");
//    System.out.printf("commit time\n");
//    commitTime("commit.tsv", 3000, 0.1, 0.12, 2000);
//    System.out.printf("done\n");
    }

    /**
     * Records which bandit was chosen for many runs of the same scenario.  This output is kind of big
     * an hard to digest visually.  As such, it is probably better to reduce this using a mean.  In R,
     * this can be done like this:
     * <pre>
     *    plot(tapply(z$k, floor(z$i/10), mean), type='l')
     * </pre>
     *
     * @param outputFile Where to write results
     * @param n          How many steps to follow
     * @param p1         First probability of reward
     * @param p2         Second probability of reward
     * @param cutoff     Only keep results after this many steps
     * @return Average number of correct choices.
     * @throws java.io.FileNotFoundException If the directory holding the output directory doesn't exist.
     */
    public static double commitTime(String outputFile, int n, double p1, double p2, int cutoff) throws FileNotFoundException {
        try (PrintWriter out = new PrintWriter(outputFile)) {
            Random gen = new Random();
            out.printf("i\tk\n");
            int impressions = 0;
            int correct = 0;
            for (int j = 0; j < 1000; j++) {
                // pick probabilities at random
                double[] p = {
                        p1, p2
                };
                Arrays.sort(p);
                BetaBayesModel s = new BetaBayesModel();
                for (int i = 0; i < n; i++) {
                    int k = s.sample();
                    out.printf("%d\t%d\n", i, k);

                    if (i > cutoff) {
                        impressions++;
                        correct += k;
                    }

                    final double u = gen.nextDouble();
                    boolean r = u <= p[k];
                    s.train(k, r ? 1 : 0);
                }
            }
            return (double) correct / impressions;
        }
    }

    /**
     * Computes error rate (the rate at which the sub-optimal choice is made as a function of the two
     * probabilities and the number of trials.  The output report contains p1, p2, number-of-trials,
     * total-correct, total-correct-in-last-half.
     * <p/>
     * The commitTime output is probably more interesting.
     *
     * @param outputFile Where to write the data.
     * @throws java.io.FileNotFoundException If we can't open our output
     */
    @Deprecated
    private static void errorRate(String outputFile) throws FileNotFoundException {
        try (PrintWriter out = new PrintWriter(outputFile)) {
            out.printf("p1\tp2\tn\twins\tlate\n");
            Random gen = new Random();
            for (int n : new int[]{20, 50, 100, 200, 500, 1000, 2000, 5000}) {
                System.out.printf("%d\n", n);
                for (int j = 0; j < 1000 * (n < 500 ? 10 : 1); j++) {
                    // pick probabilities at random
                    double[] p = {
                            gen.nextDouble(), gen.nextDouble()
                    };
                    // order them to make error interpretation easier
                    Arrays.sort(p);
                    BetaBayesModel s = new BetaBayesModel();
                    int wins = 0;
                    int lateWins = 0;
                    for (int i = 0; i < n; i++) {
                        int k = s.sample();
                        final double u = gen.nextDouble();
                        boolean r = u <= p[k];
                        wins += r ? 1 : 0;
                        if (i > n / 2) {
                            lateWins += r ? 1 : 0;
                        }
                        s.train(k, r ? 1 : 0);
                    }
                    out.printf("%.3f\t%.3f\t%d\t%d\t%d\n", p[0], p[1], n, wins, lateWins);
                }
            }
        }
    }

    /**
     * Computes average regret relative to perfect knowledge given uniform random probabilities. The
     * output contains the quartiles for different numbers of trials.  The quartiles are computed by
     * running many experiments for each specified number of trials.
     * <p/>
     * This can be plotted pretty much directly in R
     * <pre>
     * > x=read.delim(file='~/Apache/storm-aggregator/regret.tsv')
     * > bxp(list(com.mapr.stats=t(as.matrix(x[,2:6])), n=rep(1000,times=8),names=x$n))
     * </pre>
     *
     * @param outputFile   Where to put the output
     * @param sizes        The different size experiments to use
     * @param replications Number of times to repeat the experiment
     * @param bandits      How many bandits to simulate
     * @return Returns the average regret per trial
     * @throws java.io.FileNotFoundException If the output file can't be opened due to a missing directory.
     */
    public static double averageRegret(String outputFile, int[] sizes, int replications, int bandits) throws FileNotFoundException {

        try (PrintWriter out = new PrintWriter(outputFile)) {
            double finalMedianRegret = 0;
            Random gen = new Random();
            out.printf("n\tq0\tq1\tq2\tq3\tq4\n");
            // for each horizon time span of interest
            for (int n : sizes) {
                System.out.printf("%d\n", n);
                OnlineSummarizer summary = new OnlineSummarizer();
                // replicate the test many times
                for (int j = 0; j < replications; j++) {
                    // pick probabilities at random

                    double[] p = new double[bandits];
                    for (int k = 0; k < bandits; k++) {
                        p[k] = gen.nextDouble();
                    }

                    // order them to make error interpretation easier
                    Arrays.sort(p);
                    BetaBayesModel s = new BetaBayesModel(bandits, RandomUtils.getRandom());
                    int wins = 0;
                    for (int i = 0; i < n; i++) {
                        int k = s.sample();
                        final double u = gen.nextDouble();
                        boolean r = u <= p[k];
                        wins += r ? 1 : 0;
                        s.train(k, r ? 1 : 0);
                    }
                    summary.add((double) wins / n - p[bandits - 1]);
                }
                out.printf("%d\t", n);
                for (int quartile = 0; quartile <= 4; quartile++) {
                    out.printf("%.3f%s", summary.getQuartile(quartile), quartile < 4 ? "\t" : "\n");
                }
                out.flush();
                finalMedianRegret = summary.getMedian();

                //      System.out.printf("%.3f\n", summary.getMean());
            }
            return finalMedianRegret;
        }
    }

    /**
     * Computes average regret relative to perfect knowledge given uniform random probabilities. The
     * output contains the quartiles for different numbers of trials.  The quartiles are computed by
     * running many experiments for each specified number of trials.
     * <p/>
     * This can be plotted pretty much directly in R
     * <pre>
     * > x=read.delim(file='~/Apache/storm-aggregator/regret.tsv')
     * > bxp(list(com.mapr.stats=t(as.matrix(x[,2:6])), n=rep(1000,times=8),names=x$n))
     * </pre>
     *
     * @param cumulativeOutput Where to write the cumulative regret results
     * @param perTurnOutput    Where to write the per step regret results
     * @param replications     How many times to replicate the experiment
     * @param bandits          How many bandits to emulate
     * @param maxSteps         Maximum number of trials to run per experiment
     * @param modelFactory     How to construct the solver.
     * @param refSampler       How to get reward distributions for bandits
     * @return An estimate of the average final cumulative regret
     * @throws java.io.FileNotFoundException If the output file can't be opened due to
     *                                       a missing directory.
     */
    public static double totalRegret(String cumulativeOutput, String perTurnOutput, int replications, int bandits, int maxSteps, BanditFactory modelFactory, DistributionGenerator refSampler) throws FileNotFoundException {
        List<OnlineSummarizer> cumulativeRegret = Lists.newArrayList();
        List<OnlineSummarizer> localRegret = Lists.newArrayList();
        List<Integer> steps = Lists.newArrayList();
        List<Integer> localSteps = Lists.newArrayList();

        Random gen = new Random();

        // for each horizon time span of interest
        for (int j = 0; j < replications; j++) {
            BayesianBandit s = modelFactory.createBandit(bandits, gen);

            List<DistributionWithMean> refs = Lists.newArrayList();
            for (int k = 0; k < bandits; k++) {
                refs.add(refSampler.nextDistribution());
            }

            Collections.sort(refs);

            double wins = 0;
            int k = 0;
            int delta = 1;
            double totalRegret = 0;
            for (int i = 0; i < maxSteps; i++) {
                if (i > 50 * delta) {
                    delta = bump(delta);
                }
                int choice = s.sample();
                double r = refs.get(choice).nextDouble();

                totalRegret += refs.get(bandits - 1).getMean() - refs.get(choice).getMean();
                if ((i + 1) % delta == 0) {
                    if (cumulativeRegret.size() <= k) {
                        cumulativeRegret.add(new OnlineSummarizer());
                        steps.add(i + 1);
                    }
                    cumulativeRegret.get(k).add(totalRegret);
                    k++;
                }
                if (localRegret.size() <= i / BUCKET_SIZE) {
                    localRegret.add(new OnlineSummarizer());
                    localSteps.add(i);
                }
                double thisTrialRegret = refs.get(bandits - 1).getMean() - refs.get(choice).getMean();
                localRegret.get(i / BUCKET_SIZE).add(thisTrialRegret);
                wins += r;
                s.train(choice, r);
            }
        }

        printRegret(cumulativeOutput, cumulativeRegret, steps);
        printRegret(perTurnOutput, localRegret, localSteps);
        return cumulativeRegret.get(cumulativeRegret.size() - 1).getMedian();
    }

    private static void printRegret(String outputFile, List<OnlineSummarizer> cumulativeRegret, List<Integer> steps) throws FileNotFoundException {
        try (PrintWriter out = new PrintWriter(outputFile)) {
            out.printf("n\tmean\n");
            int k = 0;
            for (OnlineSummarizer summary : cumulativeRegret) {
                out.printf("%d\t%.4f\n", steps.get(k++), summary.getMean());
//        for (int quartile = 0; quartile <= 4; quartile++) {
//          out.printf("%.3f%s", summary.getQuartile(quartile), quartile < 4 ? "\t" : "\n");
//        }
            }
            out.flush();
        }
    }

    private static int bump(int delta) {
        int multiplier = 1;
        while (delta >= 10) {
            multiplier *= 10;
            delta /= 10;
        }
        // steps each of 1,2,5 up to next level
        delta = (int) (4 * delta - delta * delta / 3 - 1.5);
        return delta * multiplier;
    }
}
