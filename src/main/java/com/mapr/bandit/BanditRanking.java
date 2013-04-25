package com.mapr.bandit;

import com.google.common.collect.Lists;
import com.mapr.stats.bandit.BanditFactory;
import com.mapr.stats.bandit.BayesianBandit;
import com.mapr.stats.bandit.BetaBayesFactory;
import com.mapr.stats.bandit.GammaNormalBayesFactory;
import com.mapr.stats.random.AbstractBayesianDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.jet.random.Uniform;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Implements a Bandit ranking.
 */
public class BanditRanking {
    private static Random gen = RandomUtils.getRandom();

    public static void main(String[] args) throws FileNotFoundException {
        int keyItems = 10, pageSize = 20, totalItems = 200;

        if (args.length > 0) {
            keyItems = Integer.parseInt(args[0]);
        }

        if (args.length > 1) {
            pageSize = Integer.parseInt(args[1]);
        }

        if (args.length > 2) {
            totalItems = Integer.parseInt(args[2]);
        }

        BanditFactory bf = new BetaBayesFactory();
        if (args.length > 3) {
            if (args[3].startsWith("beta")) {
                bf = new BetaBayesFactory();
            } else if (args[3].startsWith("gamma")) {
                bf = new GammaNormalBayesFactory();
            } else {
                throw new IllegalArgumentException("Wanted beta or gamma to specify distribution");
            }
        }

        List<BayesianBandit> bandit = Lists.newArrayList();
        for (int m = 0; m < 50; m++) {
            bandit.add(bf.createBandit(totalItems, gen));
        }

        double[] prob = new double[totalItems];
        Uniform u = new Uniform(gen);
        for (int j = 0; j < totalItems; j++) {
            prob[j] = u.nextDouble();
        }
        Arrays.sort(prob);
        for (int j = 0; j < totalItems; j++) {
            prob[j] = 1 - prob[j];
        }

        double cumulativeRegret = 0;
        PrintWriter quality = new PrintWriter("quality.csv");
        quality.printf("Trials,Precision,Regret,CumulativeRegret\n");
        for (int i = 0; i < 1000; i++) {

            double precision = 0;
            double regret = 0;
            for (int m = 0; m < 50; m++) {
                List<Integer> page = bandit.get(m).rank(pageSize);
                for (Integer item : page) {
                    if (item < keyItems) {
                        precision++;
                    }
                }

                for (int j = 0; j < pageSize; j++) {
                    int k = page.get(j);
                    regret += prob[j] - prob[k];
                }

                for (int j = 0; j < pageSize; j++) {
                    int k = page.get(j);
                    int reward = u.nextDouble() < prob[k] ? 1 : 0;
                    bandit.get(m).train(k, reward);
                }
            }
            precision /= keyItems * 50.0;
            regret /= 50;
            cumulativeRegret += regret;

            quality.printf("%d,%.1f,%.3f,%.3f\n", i + 1, precision * 100, regret, cumulativeRegret);
        }
        quality.close();

        // display samples per rank
        PrintWriter samples = new PrintWriter("samples.csv");
        samples.printf("Rank,Samples\n");
        for (int m = 0; m < 10; m++) {
            int i = 0;
            for (AbstractBayesianDistribution distribution : bandit.get(m)) {
                samples.printf("%d,%.1f\n", i++, distribution.getSamples());
            }
        }
        samples.close();
    }
}
